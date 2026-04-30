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
        self._hvg_mask_np = registry.hvg_mask
        self._max_local_vocab = registry.max_local_vocab
        self._global_vocab = registry.global_vocab_size

        # Per-instance device tensor caches (fork-safe:
        # class-level CUDA tensor dicts would leak across forked
        # DataLoader workers, causing invalid CUDA context errors).
        self._device_caches: dict[str, dict[str, torch.Tensor]] = {}

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def registry(self) -> FeatureRegistry:
        return self._registry

    # ------------------------------------------------------------------
    # Device-side tensor caches (lazy, per-instance, one device key)
    # ------------------------------------------------------------------

    def _cached_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        key = str(device)
        if key not in self._device_caches:
            self._device_caches[key] = {
                "local_to_global": torch.as_tensor(
                    self._local_to_global_np, dtype=torch.long, device=device
                ),
                "gene_prob": torch.as_tensor(
                    self._gene_prob_np, dtype=torch.float32, device=device
                ),
                "has_gene": torch.as_tensor(
                    self._has_gene_np, dtype=torch.bool, device=device
                ),
                "hvg_mask": torch.as_tensor(
                    self._hvg_mask_np, dtype=torch.bool, device=device
                ),
            }
        return self._device_caches[key]

    # ------------------------------------------------------------------
    # Weighted probability construction
    # ------------------------------------------------------------------

    def _build_weighted_probs(
        self,
        device: torch.device,
        bsz: int,
        dataset_indices: torch.Tensor,
        global_genes_sorted: torch.Tensor | None = None,
        *,
        sampling_mode: str = "uniform",
        expressed_weight: float = 3.0,
        hvg_weight: float = 3.0,
        gene_prob_t: torch.Tensor | None = None,
        hvg_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build per-cell probability tensors for weighted ``multinomial`` sampling.

        Returns ``(bsz, global_vocab)`` float32 tensor where each row sums to 1.0.
        """
        if sampling_mode == "uniform":
            # Uniform probabilities over each cell's dataset valid gene pool
            if gene_prob_t is None:
                raise ValueError("gene_prob_t required for uniform mode")
            return gene_prob_t[dataset_indices]  # (bsz, global_vocab)

        if sampling_mode == "expressed":
            # Expressed-biased: expressed genes get weight bonus
            if global_genes_sorted is None:
                raise ValueError("global_genes_sorted required for expressed mode")

            max_nnz = global_genes_sorted.shape[1]

            # Build (bsz, global_vocab) binary mask efficiently using scatter
            valid_positions = global_genes_sorted < self._global_vocab  # (bsz, max_nnz)
            n_valid = int(valid_positions.sum().item())

            if n_valid > 0:
                row_indices = (
                    torch.arange(bsz, device=device)
                    .unsqueeze(1)
                    .expand(-1, max_nnz)
                )  # (bsz, max_nnz)
                valid_rows = row_indices[valid_positions]  # (n_valid,)
                valid_cols = global_genes_sorted[valid_positions]  # (n_valid,)
            else:
                valid_rows = torch.empty(0, dtype=torch.long, device=device)
                valid_cols = torch.empty(0, dtype=torch.long, device=device)

            probs = torch.ones(
                (bsz, self._global_vocab), dtype=torch.float32, device=device
            )
            # Expressed genes get expressed_weight bonus on top of base 1.0
            if n_valid > 0:
                probs[valid_rows, valid_cols] += float(expressed_weight)
            probs /= probs.sum(dim=1, keepdim=True)
            return probs

        if sampling_mode == "hvg":
            # HVG-biased: HVG genes get weight bonus per dataset
            if hvg_t is None:
                raise ValueError("hvg_t required for hvg mode")

            hvgs = hvg_t[dataset_indices]  # (bsz, global_vocab) bool
            probs = torch.ones(
                (bsz, self._global_vocab), dtype=torch.float32, device=device
            )
            probs += hvgs.float() * float(hvg_weight)
            probs /= probs.sum(dim=1, keepdim=True)
            return probs

        raise ValueError(
            f"Unknown sampling_mode: {sampling_mode!r}. "
            f"Valid options: 'uniform', 'expressed', 'hvg'."
        )

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
        sampling_mode: str = "uniform",
        expressed_weight: float = 3.0,
        hvg_weight: float = 3.0,
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
        sampling_mode : str, default "uniform"
            Gene sampling strategy: ``"uniform"`` (all genes equal weight),
            ``"expressed"`` (expressed genes get ``expressed_weight`` bonus),
            or ``"hvg"`` (highly-variable genes get ``hvg_weight`` bonus).
        expressed_weight : float, default 3.0
            Weight bonus for expressed genes in ``"expressed"`` mode.
            Expressed genes receive base 1.0 + ``expressed_weight``.
        hvg_weight : float, default 3.0
            Weight bonus for HVG genes in ``"hvg"`` mode.
            HVG genes receive base 1.0 + ``hvg_weight``.

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
        hvg_t = cached["hvg_mask"]

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

        # Pad with max_local_vocab (>= all valid local indices) so padding
        # positions can be distinguished and mapped to a global sentinel.
        _local_pad = self._max_local_vocab
        padded_local_genes = pad_sequence(
            split_genes, batch_first=True, padding_value=_local_pad
        )  # (batch_size, max_nnz)
        padded_counts = pad_sequence(
            split_cnts, batch_first=True, padding_value=0.0
        )  # (batch_size, max_nnz)

        max_nnz = int(padded_local_genes.shape[1])

        # ---- Short-circuit when all cells have zero expressed genes ----
        if max_nnz == 0:
            # Sample gene IDs from per-dataset probability pools
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
                probs = self._build_weighted_probs(
                    device,
                    bsz,
                    dataset_indices,
                    # No expressed genes, so expressed mode degrades
                    # to per-dataset uniform (global_genes_sorted=None).
                    sampling_mode=sampling_mode,
                    expressed_weight=expressed_weight,
                    hvg_weight=hvg_weight,
                    gene_prob_t=gene_prob_t,
                    hvg_t=hvg_t,
                )
                sampled_gids = torch.multinomial(
                    probs,
                    num_samples=self._seq_len,
                    replacement=False,
                    generator=generator,
                )

            # No expressed genes → all exact_match False, all counts 0
            exact_match = torch.zeros(
                (bsz, self._seq_len), dtype=torch.bool, device=device
            )
            sampled_counts = torch.zeros(
                (bsz, self._seq_len), dtype=torch.float32, device=device
            )

            # Valid mask from dataset_has_gene
            valid_rows = has_gene_t[dataset_indices]
            safe_gids = sampled_gids.clamp(0, self._global_vocab - 1)
            in_vocab = (sampled_gids >= 0) & (sampled_gids < self._global_vocab)
            valid_mask = valid_rows.gather(1, safe_gids) & in_vocab

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

        # ---- Local→Global resolution ----
        is_padding = padded_local_genes >= self._max_local_vocab

        # Clamp local indices to [0, max_local_vocab-1] for safe indexing
        clamped = padded_local_genes.clamp(0, self._max_local_vocab - 1)

        # Advanced indexing: (n_datasets, max_local_vocab)[ds, local]
        # dataset_indices.unsqueeze(1) broadcasts to (batch_size, max_nnz)
        global_genes = local_to_global_t[
            dataset_indices.unsqueeze(1), clamped
        ]  # (batch_size, max_nnz)

        # Set padding positions to a sentinel beyond all valid global gene IDs
        # so they sort after real genes and never match searchsorted queries.
        _global_sentinel = self._global_vocab  # beyond 0..global_vocab-1
        global_genes = torch.where(
            is_padding,
            torch.full_like(global_genes, _global_sentinel),
            global_genes,
        )

        # ---- Sort global genes (and counts) for searchsorted ----
        # Use stable sort so that duplicate global gene IDs (from two local
        # genes mapping to the same global gene) maintain consistent ordering
        # with the CPU path (np.argsort(kind="stable")).
        global_genes_sorted, sort_idx = global_genes.sort(dim=1, stable=True)
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
            # Build weighted probabilities based on sampling mode
            probs = self._build_weighted_probs(
                device,
                bsz,
                dataset_indices,
                global_genes_sorted=global_genes_sorted,
                sampling_mode=sampling_mode,
                expressed_weight=expressed_weight,
                hvg_weight=hvg_weight,
                gene_prob_t=gene_prob_t,
                hvg_t=hvg_t,
            )
            sampled_gids = torch.multinomial(
                probs,
                num_samples=self._seq_len,
                replacement=False,
                generator=generator,
            )  # (batch_size, seq_len)
            # Free probability tensor (may be large for expressed mode)
            del probs

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
        in_vocab = (sampled_gids >= 0) & (sampled_gids < self._global_vocab)
        valid_mask = valid_rows.gather(1, safe_gids) & in_vocab  # (batch_size, seq_len)

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

        # ---- Per-cell: validate and gather ----
        for i in range(bsz):
            ds_idx = int(dataset_indices[i])
            start = int(row_offsets[i])
            stop = int(row_offsets[i + 1])
            cell_local = expressed_gene_indices[start:stop]
            cell_counts = expression_counts[start:stop]

            # Map local→global (may be empty for zero-gene cells)
            if len(cell_local) > 0:
                mapping_row = self._local_to_global[ds_idx]
                cell_global = mapping_row[cell_local]
                # Sort by global gene ID for O(log n) lookup
                sort_idx = np.argsort(cell_global, kind="stable")
                sorted_global = cell_global[sort_idx]
                sorted_counts = cell_counts[sort_idx]
            else:
                sorted_global = np.array([], dtype=np.int32)
                sorted_counts = np.array([], dtype=np.float32)

            for j in range(self._seq_len):
                target = int(sids[i, j])
                if target < 0:
                    continue  # pad position (from sampler)

                # Check dataset validity: gene must be in this cell's dataset
                if target >= has_gene.shape[1] or not has_gene[ds_idx, target]:
                    continue  # stays invalid (count = sentinel, valid_mask = False)

                # Gene is valid for this dataset
                valid_mask[i, j] = True
                # Default count is 0.0 (gene valid but not necessarily expressed)
                sampled_counts[i, j] = 0.0

                # Exact match: check if the gene is expressed in this cell
                if len(sorted_global) > 0:
                    pos = int(np.searchsorted(sorted_global, target, side="left"))
                    if pos < len(sorted_global) and sorted_global[pos] == target:
                        sampled_counts[i, j] = float(sorted_counts[pos])
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
