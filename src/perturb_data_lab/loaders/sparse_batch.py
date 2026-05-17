"""Sparse expression batch processing with cross-dataset gene resolution.

``SparseBatchProcessor`` receives flat raw batch dicts from corpus loaders,
maps local→global using ``FeatureRegistry.local_to_global_map``, samples
genes per cell from per-dataset probability pools, and gathers expression
counts into dense ``(batch_size, seq_len)`` tensors on the requested device.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .feature_registry import FeatureRegistry

__all__ = [
    "SparseBatchProcessor",
]

# Default sentinel values
DEFAULT_PAD_TOKEN_ID: int = -1
DEFAULT_INVALID_COUNT_VALUE: float = -1.0


# ---------------------------------------------------------------------------
# Sparse batch processor
# ---------------------------------------------------------------------------


class SparseBatchProcessor:
    """Process sparse expression arrays into sampled training tensors.

    Receives flat expression arrays from corpus raw batch dicts,
    resolves local gene indices to global gene IDs, samples from per-dataset
    probability pools, and gathers counts via searchsorted+gather.

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
        sampling_gene_mask: np.ndarray | None = None,
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
        self._sampling_gene_mask_np = self._normalize_sampling_gene_mask(sampling_gene_mask)

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
            }
            if self._sampling_gene_mask_np is not None:
                self._device_caches[key]["sampling_gene_mask"] = torch.as_tensor(
                    self._sampling_gene_mask_np, dtype=torch.bool, device=device
                )
        return self._device_caches[key]

    def _cached_hvg_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._cached_tensors(device)
        if "hvg_mask" not in cached:
            cached["hvg_mask"] = torch.as_tensor(
                self._registry.hvg_mask,
                dtype=torch.bool,
                device=device,
            )
            cached["hvg_rank"] = torch.as_tensor(
                self._registry.hvg_rank_matrix,
                dtype=torch.int32,
                device=device,
            )
        return cached["hvg_mask"], cached["hvg_rank"]

    def _normalize_sampling_gene_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        normalized = np.asarray(mask, dtype=bool)
        assert normalized.shape == (
            self._global_vocab,
        ), "sampling_gene_mask must have one entry per global gene"
        return normalized

    def _normalize_probs(self, probs: torch.Tensor) -> torch.Tensor:
        totals = probs.sum(dim=1, keepdim=True)
        return torch.where(totals > 0, probs / totals.clamp_min(1.0), probs)

    def _sample_gene_ids(
        self,
        probs: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        safe_probs = probs.clone()
        empty_rows = safe_probs.sum(dim=1) == 0
        safe_probs[empty_rows, 0] = 1.0
        sampled = torch.multinomial(
            safe_probs,
            self._seq_len,
            replacement=False,
            generator=generator,
        )
        sampled_is_real = probs.gather(1, sampled) > 0
        return torch.where(
            sampled_is_real,
            sampled,
            torch.full_like(sampled, self._pad_token_id),
        )

    def _validate_sampled_gene_ids(
        self,
        sampled_gene_ids: torch.Tensor,
        *,
        bsz: int,
        device: torch.device,
    ) -> torch.Tensor:
        sampled_gids = sampled_gene_ids.to(
            dtype=torch.long,
            device=device,
            non_blocking=True,
        )
        if sampled_gids.shape != (bsz, self._seq_len):
            raise ValueError(
                f"sampled_gene_ids shape {tuple(sampled_gids.shape)} "
                f"!= expected ({bsz}, {self._seq_len})"
            )
        return sampled_gids

    def _valid_gene_mask(
        self,
        sampled_gids: torch.Tensor,
        dataset_indices: torch.Tensor,
        has_gene_t: torch.Tensor,
    ) -> torch.Tensor:
        valid_rows = has_gene_t[dataset_indices]
        safe_gids = sampled_gids.clamp(0, self._global_vocab - 1)
        in_vocab = (sampled_gids >= 0) & (sampled_gids < self._global_vocab)
        return valid_rows.gather(1, safe_gids) & in_vocab

    def _finalize_result(
        self,
        *,
        bsz: int,
        sampled_gids: torch.Tensor,
        sampled_counts: torch.Tensor,
        exact_match: torch.Tensor,
        dataset_indices: torch.Tensor,
        global_row_index: torch.Tensor,
        has_gene_t: torch.Tensor,
        size_factor: torch.Tensor | None,
    ) -> dict[str, Any]:
        valid_mask = self._valid_gene_mask(sampled_gids, dataset_indices, has_gene_t)
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
        result = {
            "batch_size": bsz,
            "seq_len": self._seq_len,
            "sampled_gene_ids": final_gene_ids,
            "sampled_counts": final_counts,
            "valid_mask": valid_mask,
            "exact_match_mask": exact_match,
            "dataset_index": dataset_indices,
            "global_row_index": global_row_index,
        }
        if size_factor is not None:
            result["size_factor"] = size_factor
        return result

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
        hvg_top_k: int | None = None,
        gene_prob_t: torch.Tensor | None = None,
        has_gene_t: torch.Tensor | None = None,
        hvg_t: torch.Tensor | None = None,
        hvg_rank_t: torch.Tensor | None = None,
        sampling_gene_mask_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build per-cell probability tensors for weighted ``multinomial`` sampling.

        Rows with at least one valid gene sum to 1.0; empty rows stay zero and
        are padded after sampling.
        """
        if sampling_mode == "uniform":
            # Uniform probabilities over each cell's dataset valid gene pool
            if sampling_gene_mask_t is None:
                if gene_prob_t is None:
                    raise ValueError("gene_prob_t required for uniform mode")
                return gene_prob_t[dataset_indices]  # (bsz, global_vocab)
            if has_gene_t is None:
                raise ValueError("has_gene_t required for masked uniform mode")
            valid_genes = has_gene_t[dataset_indices] & sampling_gene_mask_t.unsqueeze(0)
            return self._normalize_probs(valid_genes.float())

        if sampling_mode == "expressed":
            # Expressed-biased: expressed genes get weight bonus
            if has_gene_t is None:
                raise ValueError("has_gene_t required for expressed mode")

            valid_genes = has_gene_t[dataset_indices]
            if sampling_gene_mask_t is not None:
                valid_genes = valid_genes & sampling_gene_mask_t.unsqueeze(0)
            probs = valid_genes.float()

            if global_genes_sorted is None:
                return self._normalize_probs(probs)

            max_nnz = global_genes_sorted.shape[1]

            # Upweight expressed genes only if they are valid for that dataset.
            safe_genes = global_genes_sorted.clamp(0, self._global_vocab - 1)
            in_vocab = global_genes_sorted < self._global_vocab
            valid_positions = in_vocab & valid_genes.gather(1, safe_genes)
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

            # Expressed genes get expressed_weight bonus on top of base 1.0
            if n_valid > 0:
                probs[valid_rows, valid_cols] += float(expressed_weight)
            return self._normalize_probs(probs)

        if sampling_mode == "hvg":
            # HVG-biased: HVG genes get weight bonus per dataset
            if has_gene_t is None:
                raise ValueError("has_gene_t required for hvg mode")

            valid_genes = has_gene_t[dataset_indices]  # (bsz, global_vocab) bool
            if sampling_gene_mask_t is not None:
                valid_genes &= sampling_gene_mask_t.unsqueeze(0)
            if hvg_top_k is None:
                if hvg_t is None:
                    raise ValueError("hvg_t required for default hvg mode")
                hvgs = hvg_t[dataset_indices] & valid_genes
            else:
                if hvg_rank_t is None:
                    raise ValueError("hvg_rank_t required for dynamic hvg mode")
                ranks = hvg_rank_t[dataset_indices]
                hvgs = (ranks > 0) & (ranks <= int(hvg_top_k)) & valid_genes

            probs = valid_genes.float()
            probs += hvgs.float() * float(hvg_weight)
            return self._normalize_probs(probs)

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
        hvg_top_k: int | None = None,
    ) -> dict[str, Any]:
        """Process a flat batch dict into dense ``(batch_size, seq_len)`` tensors.

        Parameters
        ----------
        batch : dict
            Raw expression batch with attached ``dataset_index``.
            Required keys:
            ``expressed_gene_indices``, ``expression_counts``, ``row_offsets``,
            ``dataset_index``, ``batch_size``, and ``global_row_index``.
            ``size_factor`` is optional pass-through metadata. All array-like
            inputs may be numpy arrays or torch tensors.
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
        hvg_top_k : int, optional
            Dynamic top-k threshold for ``"hvg"`` mode. When provided, genes
            with ``0 < hvg_rank <= hvg_top_k`` receive the HVG bonus from the
            same canonical ``hvg.parquet`` ranking table without rematerializing
            the corpus. When omitted, the dataset's default HVG selection is used.

        Returns
        -------
        dict
            Keys: ``sampled_gene_ids`` (int64), ``sampled_counts`` (float32),
            ``valid_mask`` (bool), ``exact_match_mask`` (bool),
            ``dataset_index``, ``global_row_index``, ``batch_size``,
            ``seq_len``, and optional ``size_factor``.
            All tensor shapes: ``sampled_gene_ids`` / ``sampled_counts`` /
            ``valid_mask`` are ``(batch_size, seq_len)``.
            ``dataset_index`` / ``global_row_index`` are ``(batch_size,)``.
            ``size_factor`` is ``(batch_size,)`` when present.
        """
        # Resolve device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        if hvg_top_k is not None and int(hvg_top_k) <= 0:
            raise ValueError("hvg_top_k must be positive when provided")

        cached = self._cached_tensors(device)
        local_to_global_t = cached["local_to_global"]
        gene_prob_t = cached["gene_prob"]
        has_gene_t = cached["has_gene"]
        hvg_t = None
        hvg_rank_t = None
        if sampling_mode == "hvg" and sampled_gene_ids is None:
            hvg_t, hvg_rank_t = self._cached_hvg_tensors(device)
        sampling_gene_mask_t = cached.get("sampling_gene_mask")

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
        size_factor = None
        if "size_factor" in batch and batch["size_factor"] is not None:
            size_factor = torch.as_tensor(
                batch["size_factor"], dtype=torch.float32, device=device
            )

        bsz = int(batch["batch_size"])
        # Validate empty batch short-circuit
        if bsz == 0:
            return self._finalize_result(
                bsz=0,
                sampled_gids=torch.empty(
                    (0, self._seq_len), dtype=torch.long, device=device
                ),
                sampled_counts=torch.empty(
                    (0, self._seq_len), dtype=torch.float32, device=device
                ),
                exact_match=torch.empty(
                    (0, self._seq_len), dtype=torch.bool, device=device
                ),
                dataset_indices=dataset_indices,
                global_row_index=global_row_index,
                has_gene_t=has_gene_t,
                size_factor=size_factor,
            )

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
                sampled_gids = self._validate_sampled_gene_ids(
                    sampled_gene_ids,
                    bsz=bsz,
                    device=device,
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
                    hvg_top_k=hvg_top_k,
                    gene_prob_t=gene_prob_t,
                    has_gene_t=has_gene_t,
                    hvg_t=hvg_t,
                    hvg_rank_t=hvg_rank_t,
                    sampling_gene_mask_t=sampling_gene_mask_t,
                )
                sampled_gids = self._sample_gene_ids(probs, generator)

            # No expressed genes → all exact_match False, all counts 0
            exact_match = torch.zeros(
                (bsz, self._seq_len), dtype=torch.bool, device=device
            )
            sampled_counts = torch.zeros(
                (bsz, self._seq_len), dtype=torch.float32, device=device
            )
            return self._finalize_result(
                bsz=bsz,
                sampled_gids=sampled_gids,
                sampled_counts=sampled_counts,
                exact_match=exact_match,
                dataset_indices=dataset_indices,
                global_row_index=global_row_index,
                has_gene_t=has_gene_t,
                size_factor=size_factor,
            )

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
            sampled_gids = self._validate_sampled_gene_ids(
                sampled_gene_ids,
                bsz=bsz,
                device=device,
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
                hvg_top_k=hvg_top_k,
                gene_prob_t=gene_prob_t,
                has_gene_t=has_gene_t,
                hvg_t=hvg_t,
                hvg_rank_t=hvg_rank_t,
                sampling_gene_mask_t=sampling_gene_mask_t,
            )
            sampled_gids = self._sample_gene_ids(probs, generator)
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

        return self._finalize_result(
            bsz=bsz,
            sampled_gids=sampled_gids,
            sampled_counts=sampled_counts,
            exact_match=exact_match,
            dataset_indices=dataset_indices,
            global_row_index=global_row_index,
            has_gene_t=has_gene_t,
            size_factor=size_factor,
        )
