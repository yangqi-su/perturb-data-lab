"""Phase 3: GPU Pipeline with Cross-Dataset Gene Resolution.

``GPUSparsePipeline`` receives flat expression arrays from ``read_batch()``,
maps local→global using ``FeatureRegistry.local_to_global_map``, samples
genes per cell from per-dataset probability pools on GPU, and gathers
expression counts into dense ``(batch_size, seq_len)`` tensors.

``CPUPipeline`` provides an equivalent CPU path for validation, using
numpy searchsorted and ``GlobalGeneSampler``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .feature_registry import FeatureRegistry, GlobalGeneSampler

__all__ = [
    "GPUSparsePipeline",
    "CPUPipeline",
]

# Default sentinel values
DEFAULT_PAD_TOKEN_ID: int = -1
DEFAULT_INVALID_COUNT_VALUE: float = -1.0


# ---------------------------------------------------------------------------
# GPU Pipeline
# ---------------------------------------------------------------------------


class GPUSparsePipeline:
    """GPU sparse pipeline for cross-dataset gene resolution.

    Receives flat expression arrays from ``BatchExecutor.read_batch()``,
    resolves local gene indices to global gene IDs, samples from per-dataset
    probability pools on GPU, and gathers counts via searchsorted+gather.

    Parameters
    ----------
    registry : FeatureRegistry
        Provides local→global mapping and per-dataset gene probabilities.
    seq_len : int
        Number of global gene IDs to sample per cell.
    pad_token_id : int, default -1
        Sentinel value for invalid/out-of-pool sampled gene positions.
    invalid_count_value : float, default -1.0
        Sentinel value for counts at invalid positions.
    """

    def __init__(
        self,
        registry: FeatureRegistry,
        seq_len: int,
        *,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        invalid_count_value: float = DEFAULT_INVALID_COUNT_VALUE,
    ):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self._registry = registry
        self._seq_len = int(seq_len)
        self._pad_token_id = int(pad_token_id)
        self._invalid_count_value = float(invalid_count_value)

        # Pre-fetch numpy arrays for lazy torch conversion
        self._local_to_global_np = registry.local_to_global_map
        self._gene_prob_np = registry.dataset_gene_prob
        self._has_gene_np = registry.dataset_has_gene
        self._max_local_vocab = registry.max_local_vocab
        self._global_vocab = registry.global_vocab_size

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def registry(self) -> FeatureRegistry:
        return self._registry

    # ------------------------------------------------------------------
    # Device-side tensor caches (lazy, one per device)
    # ------------------------------------------------------------------
    _DEVICE_CACHES: dict[str, dict[str, torch.Tensor]] = {}

    def _cached_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        key = str(device)
        if key not in self._DEVICE_CACHES:
            self._DEVICE_CACHES[key] = {
                "local_to_global": torch.as_tensor(
                    self._local_to_global_np, dtype=torch.long, device=device
                ),
                "gene_prob": torch.as_tensor(
                    self._gene_prob_np, dtype=torch.float32, device=device
                ),
                "has_gene": torch.as_tensor(
                    self._has_gene_np, dtype=torch.bool, device=device
                ),
            }
        return self._DEVICE_CACHES[key]

    # ------------------------------------------------------------------
    # process_batch
    # ------------------------------------------------------------------

    def process_batch(
        self,
        batch: dict[str, Any],
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        sampled_gene_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Process a flat batch dict into dense ``(batch_size, seq_len)`` tensors.

        Parameters
        ----------
        batch : dict
            Output of ``BatchExecutor.read_batch()``. Required keys:
            ``expressed_gene_indices``, ``expression_counts``, ``row_offsets``,
            ``dataset_index``, ``batch_size``, ``global_row_index``,
            ``size_factor``.  All numpy arrays.
        device : torch.device or str, optional
            Target device. Defaults to CUDA if available, else CPU.
        generator : torch.Generator, optional
            Torch RNG generator for reproducible ``multinomial`` sampling.
        sampled_gene_ids : torch.Tensor, optional
            Pre-computed ``(batch_size, seq_len)`` tensor of global gene IDs.
            When provided, skips ``multinomial`` sampling and uses these IDs.
            Useful for deterministic equivalence testing.

        Returns
        -------
        dict
            Keys: ``sampled_gene_ids`` (int64), ``sampled_counts`` (float32),
            ``valid_mask`` (bool), ``exact_match_mask`` (bool),
            ``dataset_index``, ``global_row_index``, ``size_factor``,
            ``batch_size``, ``seq_len``.
            All tensor shapes: ``sampled_gene_ids`` / ``sampled_counts`` /
            ``valid_mask`` are ``(batch_size, seq_len)``.
            ``dataset_index`` / ``global_row_index`` / ``size_factor``
            are ``(batch_size,)``.
        """
        # Resolve device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        cached = self._cached_tensors(device)
        local_to_global_t = cached["local_to_global"]
        gene_prob_t = cached["gene_prob"]
        has_gene_t = cached["has_gene"]

        # ---- Move flat arrays to device ----
        flat_gene_indices = torch.as_tensor(
            batch["expressed_gene_indices"], dtype=torch.long, device=device
        )
        flat_counts = torch.as_tensor(
            batch["expression_counts"], dtype=torch.float32, device=device
        )
        row_offsets = torch.as_tensor(
            batch["row_offsets"], dtype=torch.long, device=device
        )
        dataset_indices = torch.as_tensor(
            batch["dataset_index"], dtype=torch.long, device=device
        )
        global_row_index = torch.as_tensor(
            batch["global_row_index"], dtype=torch.long, device=device
        )
        size_factor = torch.as_tensor(
            batch["size_factor"], dtype=torch.float32, device=device
        )

        bsz = int(batch["batch_size"])
        # Validate empty batch short-circuit
        if bsz == 0:
            return {
                "batch_size": 0,
                "seq_len": self._seq_len,
                "sampled_gene_ids": torch.empty(
                    (0, self._seq_len), dtype=torch.long, device=device
                ),
                "sampled_counts": torch.empty(
                    (0, self._seq_len), dtype=torch.float32, device=device
                ),
                "valid_mask": torch.empty(
                    (0, self._seq_len), dtype=torch.bool, device=device
                ),
                "exact_match_mask": torch.empty(
                    (0, self._seq_len), dtype=torch.bool, device=device
                ),
                "dataset_index": dataset_indices,
                "global_row_index": global_row_index,
                "size_factor": size_factor,
            }

        # ---- Compute per-cell NNZ lengths ----
        row_lengths = row_offsets[1:] - row_offsets[:-1]  # (batch_size,)

        # ---- Split + pad local gene indices and counts ----
        split_sizes = tuple(int(x) for x in row_lengths.cpu().tolist())
        split_genes = list(torch.split(flat_gene_indices, split_sizes))
        split_cnts = list(torch.split(flat_counts, split_sizes))

        padded_local_genes = pad_sequence(
            split_genes, batch_first=True, padding_value=0
        )  # (batch_size, max_nnz)
        padded_counts = pad_sequence(
            split_cnts, batch_first=True, padding_value=0.0
        )  # (batch_size, max_nnz)

        max_nnz = int(padded_local_genes.shape[1])

        # ---- Local→Global resolution ----
        # Clamp local indices to [0, max_local_vocab-1] for safe array indexing.
        clamped = padded_local_genes.clamp(0, self._max_local_vocab - 1)

        # Advanced indexing: (n_datasets, max_local_vocab)[ds, local]
        # dataset_indices.unsqueeze(1) broadcasts to (batch_size, max_nnz)
        global_genes = local_to_global_t[
            dataset_indices.unsqueeze(1), clamped
        ]  # (batch_size, max_nnz)

        # ---- Sort global genes (and counts) for searchsorted ----
        global_genes_sorted, sort_idx = global_genes.sort(dim=1)
        counts_sorted = padded_counts.gather(1, sort_idx)

        # ---- Sample global gene IDs from per-dataset probability pools ----
        if sampled_gene_ids is not None:
            sampled_gids = sampled_gene_ids.to(
                dtype=torch.long, device=device, non_blocking=True
            )
            if sampled_gids.shape != (bsz, self._seq_len):
                raise ValueError(
                    f"sampled_gene_ids shape {tuple(sampled_gids.shape)} "
                    f"!= expected ({bsz}, {self._seq_len})"
                )
        else:
            # probs: (batch_size, global_vocab)
            probs = gene_prob_t[dataset_indices]
            sampled_gids = torch.multinomial(
                probs,
                num_samples=self._seq_len,
                replacement=True,
                generator=generator,
            )  # (batch_size, seq_len)

        # ---- searchsorted + gather in global gene space ----
        search_positions = torch.searchsorted(
            global_genes_sorted, sampled_gids, right=False
        )  # (batch_size, seq_len)

        # Clamp to valid column range
        clamped_positions = search_positions.clamp(0, max_nnz - 1)
        gathered_gids = global_genes_sorted.gather(1, clamped_positions)
        gathered_counts = counts_sorted.gather(1, clamped_positions)

        # Exact match: gene ID matches AND position is within real gene range
        exact_match = gathered_gids.eq(sampled_gids) & search_positions.lt(
            row_lengths.unsqueeze(1)
        )

        sampled_counts = torch.where(
            exact_match,
            gathered_counts,
            torch.zeros_like(gathered_counts),
        )

        # ---- Valid mask: sampled gene is valid for this cell's dataset ----
        # Index into has_gene: (batch_size, global_vocab)[:, sampled_gid]
        valid_rows = has_gene_t[dataset_indices]  # (batch_size, global_vocab)
        safe_gids = sampled_gids.clamp(0, self._global_vocab - 1)
        valid_mask = valid_rows.gather(1, safe_gids)  # (batch_size, seq_len)

        # Replace invalid positions with sentinels
        final_gene_ids = torch.where(
            valid_mask,
            sampled_gids,
            torch.full_like(sampled_gids, self._pad_token_id),
        )
        final_counts = torch.where(
            valid_mask,
            sampled_counts,
            torch.full_like(sampled_counts, self._invalid_count_value),
        )

        return {
            "batch_size": bsz,
            "seq_len": self._seq_len,
            "sampled_gene_ids": final_gene_ids,
            "sampled_counts": final_counts,
            "valid_mask": valid_mask,
            "exact_match_mask": exact_match,
            "dataset_index": dataset_indices,
            "global_row_index": global_row_index,
            "size_factor": size_factor,
        }


# ---------------------------------------------------------------------------
# CPU Pipeline (validation / fallback)
# ---------------------------------------------------------------------------


class CPUPipeline:
    """CPU pipeline for cross-dataset gene resolution (validation/comparison).

    Mirrors ``GPUSparsePipeline`` logic but runs on CPU with numpy and
    ``GlobalGeneSampler``.  Useful for equivalence testing and environments
    without a GPU.

    Parameters
    ----------
    registry : FeatureRegistry
        Feature registry with local→global mapping and per-dataset probabilities.
    seq_len : int
        Number of global gene IDs to sample per cell.
    pad_token_id : int, default -1
        Sentinel for invalid gene positions.
    invalid_count_value : float, default -1.0
        Sentinel for invalid count positions.
    seed : int, default 42
        Seed for reproducibility of the internal ``GlobalGeneSampler``.
    """

    def __init__(
        self,
        registry: FeatureRegistry,
        seq_len: int,
        *,
        pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
        invalid_count_value: float = DEFAULT_INVALID_COUNT_VALUE,
        seed: int = 42,
    ):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self._registry = registry
        self._seq_len = int(seq_len)
        self._pad_token_id = int(pad_token_id)
        self._invalid_count_value = float(invalid_count_value)
        self._rng = np.random.default_rng(int(seed))
        self._sampler = GlobalGeneSampler(registry, self._rng)
        self._local_to_global = registry.local_to_global_map

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def registry(self) -> FeatureRegistry:
        return self._registry

    def process_batch(
        self,
        batch: dict[str, Any],
        *,
        seed: int | None = None,
        sampled_gene_ids: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Process a flat batch dict into dense ``(batch_size, seq_len)`` arrays.

        Parameters
        ----------
        batch : dict
            Output of ``BatchExecutor.read_batch()``.
        seed : int or None
            If given, re-seeds the internal sampler for reproducibility.
        sampled_gene_ids : np.ndarray or None
            Pre-computed ``(batch_size, seq_len)`` int32 array of global gene
            IDs.  When provided, skips sampling and uses these IDs directly.
            Useful for equivalence testing against the GPU pipeline.

        Returns
        -------
        dict
            Keys: ``sampled_gene_ids``, ``sampled_counts``, ``valid_mask``,
            ``exact_match_mask``, ``dataset_index``, ``global_row_index``,
            ``size_factor``, ``batch_size``, ``seq_len``.
            All arrays are numpy.
        """
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            self._sampler = GlobalGeneSampler(self._registry, self._rng)

        bsz = int(batch["batch_size"])

        if bsz == 0:
            return {
                "batch_size": 0,
                "seq_len": self._seq_len,
                "sampled_gene_ids": np.empty((0, self._seq_len), dtype=np.int32),
                "sampled_counts": np.empty((0, self._seq_len), dtype=np.float32),
                "valid_mask": np.empty((0, self._seq_len), dtype=bool),
                "exact_match_mask": np.empty((0, self._seq_len), dtype=bool),
                "dataset_index": np.array([], dtype=np.int32),
                "global_row_index": np.array([], dtype=np.int64),
                "size_factor": np.array([], dtype=np.float32),
            }

        expressed_gene_indices = np.asarray(batch["expressed_gene_indices"])
        expression_counts = np.asarray(batch["expression_counts"])
        row_offsets = np.asarray(batch["row_offsets"])
        dataset_indices = np.asarray(batch["dataset_index"])
        global_row_index = np.asarray(batch["global_row_index"])
        size_factor = np.asarray(batch["size_factor"])

        # ---- Sample or accept pre-computed gene IDs ----
        if sampled_gene_ids is not None:
            sids = np.asarray(sampled_gene_ids, dtype=np.int32)
            if sids.shape != (bsz, self._seq_len):
                raise ValueError(
                    f"sampled_gene_ids shape {sids.shape} "
                    f"!= expected ({bsz}, {self._seq_len})"
                )
        else:
            sids = self._sampler.sample(self._seq_len, dataset_indices)
            # GlobalGeneSampler may produce -1 padding when seq_len > n_valid

        # ---- Pre-allocate outputs ----
        sampled_counts = np.full(
            (bsz, self._seq_len), self._invalid_count_value, dtype=np.float32
        )
        valid_mask = np.zeros((bsz, self._seq_len), dtype=bool)
        exact_match_mask = np.zeros((bsz, self._seq_len), dtype=bool)

        has_gene = self._registry.dataset_has_gene

        # ---- Per-cell: resolve and gather ----
        for i in range(bsz):
            ds_idx = int(dataset_indices[i])
            start = int(row_offsets[i])
            stop = int(row_offsets[i + 1])
            cell_local = expressed_gene_indices[start:stop]
            cell_counts = expression_counts[start:stop]

            # Handle zero-gene cells
            if len(cell_local) == 0:
                continue

            # Map local→global
            mapping_row = self._local_to_global[ds_idx]
            cell_global = mapping_row[cell_local]

            # Sort by global gene ID for O(log n) lookup
            sort_idx = np.argsort(cell_global, kind="stable")
            sorted_global = cell_global[sort_idx]
            sorted_counts = cell_counts[sort_idx]

            for j in range(self._seq_len):
                target = int(sids[i, j])
                if target < 0:
                    continue  # pad position (from sampler)

                # Check dataset validity (sampled gene may not be in this
                # cell's dataset, e.g., padding from GlobalGeneSampler)
                if target >= has_gene.shape[1] or not has_gene[ds_idx, target]:
                    continue

                pos = int(np.searchsorted(sorted_global, target, side="left"))
                if pos < len(sorted_global) and sorted_global[pos] == target:
                    sampled_counts[i, j] = float(sorted_counts[pos])
                    valid_mask[i, j] = True
                    exact_match_mask[i, j] = True

        return {
            "batch_size": bsz,
            "seq_len": self._seq_len,
            "sampled_gene_ids": sids,
            "sampled_counts": sampled_counts,
            "valid_mask": valid_mask,
            "exact_match_mask": exact_match_mask,
            "dataset_index": dataset_indices,
            "global_row_index": global_row_index,
            "size_factor": size_factor,
        }
