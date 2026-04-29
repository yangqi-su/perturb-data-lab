"""Phase 3 refactored: training-facing dataset and sampling layer.

This module implements:
- CellState container class (composed by BatchExecutor)
- Sparse batch collation pipeline
- Shared sampler implementations using MetadataIndex
- Map-style PerturbDataLoader wrapping BatchExecutor
- Streaming PerturbIterableDataset wrapping BatchExecutor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import polars as pl

__all__ = [
    "CellState",
    "CellIdentity",
    "ExpressionBatch",
    "DatasetContextKey",
    "SparseBatchPayload",
    "ResolvedSparseBatch",
    "GlobalFeatureResolver",
    "SparseBatchCollator",
    "CorpusRandomBatchSampler",
    "DatasetBatchSampler",
    "DatasetContextBatchSampler",
    "CPUDenseRuntimePath",
    "GPUSparseRuntimePath",
    "SamplerState",
    "RandomContextSampler",
    "ExpressedZerosSampler",
    "HVGRandomSampler",
    "PerturbDataLoader",
    "PerturbIterableDataset",
]


@dataclass(frozen=True)
class CellIdentity:
    """Runtime-facing row identity detached from release_id hot-path routing."""

    global_row_index: int
    dataset_index: int
    dataset_id: str
    local_row_index: int


@dataclass(frozen=True)
class DatasetContextKey:
    """Dataset-aware batch grouping key built from RAM metadata."""

    dataset_index: int
    dataset_id: str
    context_value: str


# ---------------------------------------------------------------------------
# Common cell state — what a sampler sees from any backend
# ---------------------------------------------------------------------------


@dataclass
class CellState:
    """The minimal per-cell state a sampler operates on.

    Backend-agnostic; returned by every reader regardless of storage format.
    """

    identity: CellIdentity
    cell_id: str
    expressed_gene_indices: tuple[int, ...]  # dataset-order indices
    expression_counts: tuple[int, ...]
    size_factor: float
    canonical_perturbation: dict[str, str]
    canonical_context: dict[str, str]
    raw_fields: dict[str, Any]

    @property
    def global_row_index(self) -> int:
        return self.identity.global_row_index

    @property
    def dataset_id(self) -> str:
        return self.identity.dataset_id

    @property
    def dataset_index(self) -> int:
        return self.identity.dataset_index

    @property
    def local_row_index(self) -> int:
        return self.identity.local_row_index


# ---------------------------------------------------------------------------
# Legacy reader classes (BackendCellReader, ArrowHFCellReader, etc.)
# removed.  Replaced by backend-agnostic ExpressionReader (expression.py)
# and BatchExecutor (executor.py).
# ---------------------------------------------------------------------------

# Dataset-aware batch helpers and runtime-path payloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpressionBatch:
    """Flat expression arrays for a batch — zero per-cell Python objects.

    Designed as the output of ``BatchExecutor.read_expression_batch()``
    to eliminate ``CellState`` tuple conversions in the hot path.

    Fields
    ------
    batch_size : int
        Number of cells in this batch.
    global_row_index : np.ndarray
        1-D int64 array of shape ``(batch_size,)``.
    row_offsets : np.ndarray
        1-D int64 array of shape ``(batch_size+1,)`` mapping row position
        to a slice in the flat expression arrays.
    expressed_gene_indices : np.ndarray
        1-D int32 flat array of dataset-local gene indices.
    expression_counts : np.ndarray
        1-D int32 flat array of corresponding expression counts.
    """

    batch_size: int
    global_row_index: np.ndarray  # (batch_size,) int64
    row_offsets: np.ndarray  # (batch_size+1,) int64
    expressed_gene_indices: np.ndarray  # flat int32
    expression_counts: np.ndarray  # flat int32

    def row_slice(self, row_position: int) -> slice:
        """Return the slice into the flat expression arrays for *row_position*."""
        start = int(self.row_offsets[row_position])
        stop = int(self.row_offsets[row_position + 1])
        return slice(start, stop)

    def row_gene_indices(self, row_position: int) -> np.ndarray:
        """Return the expressed gene indices array for a single row."""
        return self.expressed_gene_indices[self.row_slice(row_position)]

    def row_counts(self, row_position: int) -> np.ndarray:
        """Return the expression counts array for a single row."""
        return self.expression_counts[self.row_slice(row_position)]


@dataclass(frozen=True)
class SparseBatchPayload:
    """Flat sparse payload plus offsets for runtime-hot-path batch handling."""

    batch_size: int
    global_row_index: np.ndarray
    dataset_index: np.ndarray
    local_row_index: np.ndarray
    row_offsets: np.ndarray
    expressed_gene_indices: np.ndarray
    expression_counts: np.ndarray
    size_factor: np.ndarray
    dataset_id: tuple[str, ...]
    cell_id: tuple[str, ...]
    canonical_perturbation: tuple[dict[str, str], ...]
    canonical_context: tuple[dict[str, str], ...]

    def row_slice(self, row_position: int) -> slice:
        start = int(self.row_offsets[row_position])
        stop = int(self.row_offsets[row_position + 1])
        return slice(start, stop)

    def row_gene_indices(self, row_position: int) -> np.ndarray:
        return self.expressed_gene_indices[self.row_slice(row_position)]

    def row_counts(self, row_position: int) -> np.ndarray:
        return self.expression_counts[self.row_slice(row_position)]


@dataclass(frozen=True)
class ResolvedSparseBatch:
    """Sparse batch resolved into global feature identity without densification."""

    batch_size: int
    global_row_index: np.ndarray
    dataset_index: np.ndarray
    local_row_index: np.ndarray
    row_offsets: np.ndarray
    global_feature_ids: np.ndarray
    expression_counts: np.ndarray
    size_factor: np.ndarray
    dataset_id: tuple[str, ...]
    cell_id: tuple[str, ...]
    canonical_perturbation: tuple[dict[str, str], ...]
    canonical_context: tuple[dict[str, str], ...]
    unresolved_local_features: int = 0

    def row_slice(self, row_position: int) -> slice:
        start = int(self.row_offsets[row_position])
        stop = int(self.row_offsets[row_position + 1])
        return slice(start, stop)

    def row_feature_ids(self, row_position: int) -> np.ndarray:
        return self.global_feature_ids[self.row_slice(row_position)]

    def row_counts(self, row_position: int) -> np.ndarray:
        return self.expression_counts[self.row_slice(row_position)]


class SparseBatchCollator:
    """Collate ``CellState`` rows into flat sparse payloads plus offsets.

    The standard path accepts ``CellState`` objects (backward-compatible).
    For hot-path optimization, use ``from_batch_dict()`` which directly
    consumes the output of ``BatchExecutor.read_batch()`` with zero
    per-cell Python object creation.
    """

    def __call__(self, cells: Sequence[CellState]) -> SparseBatchPayload:
        row_offsets = [0]
        flat_gene_indices: list[np.ndarray] = []
        flat_counts: list[np.ndarray] = []
        for cell in cells:
            gene_indices = np.asarray(cell.expressed_gene_indices, dtype=np.int32)
            counts = np.asarray(cell.expression_counts, dtype=np.int32)
            if gene_indices.shape != counts.shape:
                raise ValueError("cell sparse payload has mismatched gene/count lengths")
            flat_gene_indices.append(gene_indices)
            flat_counts.append(counts)
            row_offsets.append(row_offsets[-1] + int(gene_indices.size))

        return SparseBatchPayload(
            batch_size=len(cells),
            global_row_index=np.asarray([cell.global_row_index for cell in cells], dtype=np.int64),
            dataset_index=np.asarray([cell.dataset_index for cell in cells], dtype=np.int32),
            local_row_index=np.asarray([cell.local_row_index for cell in cells], dtype=np.int64),
            row_offsets=np.asarray(row_offsets, dtype=np.int64),
            expressed_gene_indices=(
                np.concatenate(flat_gene_indices).astype(np.int32, copy=False)
                if flat_gene_indices
                else np.asarray([], dtype=np.int32)
            ),
            expression_counts=(
                np.concatenate(flat_counts).astype(np.int32, copy=False)
                if flat_counts
                else np.asarray([], dtype=np.int32)
            ),
            size_factor=np.asarray([cell.size_factor for cell in cells], dtype=np.float32),
            dataset_id=tuple(cell.dataset_id for cell in cells),
            cell_id=tuple(cell.cell_id for cell in cells),
            canonical_perturbation=tuple(dict(cell.canonical_perturbation) for cell in cells),
            canonical_context=tuple(dict(cell.canonical_context) for cell in cells),
        )

    @classmethod
    def from_batch_dict(cls, batch: dict) -> SparseBatchPayload:
        """Build ``SparseBatchPayload`` from a ``read_batch()`` dict.

        This path avoids all per-cell Python object allocations and
        tuple conversions — expression data stays as numpy arrays
        from Lance reader → collator.

        Parameters
        ----------
        batch : dict
            Output of ``BatchExecutor.read_batch()``.  Must contain keys:
            ``batch_size``, ``global_row_index``, ``row_offsets``,
            ``expressed_gene_indices``, ``expression_counts``,
            ``dataset_index``, ``local_row_index``, ``size_factor``,
            ``dataset_id``, ``cell_id``, ``canonical_perturbation``,
            ``canonical_context``.

        Returns
        -------
        SparseBatchPayload
        """
        return SparseBatchPayload(
            batch_size=batch["batch_size"],
            global_row_index=batch["global_row_index"].copy(),
            dataset_index=batch["dataset_index"].copy(),
            local_row_index=batch["local_row_index"].copy(),
            row_offsets=batch["row_offsets"].copy(),
            expressed_gene_indices=batch["expressed_gene_indices"].copy(),
            expression_counts=batch["expression_counts"].copy(),
            size_factor=batch["size_factor"].copy(),
            dataset_id=tuple(batch["dataset_id"]),
            cell_id=tuple(batch["cell_id"]),
            canonical_perturbation=tuple(
                dict(item) for item in batch["canonical_perturbation"]
            ),
            canonical_context=tuple(
                dict(item) for item in batch["canonical_context"]
            ),
        )


@dataclass(frozen=True)
class GlobalFeatureResolver:
    """Post-canonicalization resolver from dataset-local indices to global ids."""

    dataset_feature_mappings: dict[int, np.ndarray]
    total_features: int
    unknown_feature_id: int = -1

    @classmethod
    def from_dataset_mappings(
        cls,
        dataset_feature_mappings: dict[int, Sequence[int] | np.ndarray],
        *,
        total_features: int | None = None,
        unknown_feature_id: int = -1,
    ) -> "GlobalFeatureResolver":
        normalized: dict[int, np.ndarray] = {}
        inferred_max = -1
        for dataset_index, mapping in dataset_feature_mappings.items():
            array = np.asarray(mapping, dtype=np.int32)
            normalized[int(dataset_index)] = array
            valid = array[array >= 0]
            if valid.size:
                inferred_max = max(inferred_max, int(valid.max()))
        if total_features is None:
            total_features = inferred_max + 1 if inferred_max >= 0 else 0
        return cls(
            dataset_feature_mappings=normalized,
            total_features=int(total_features),
            unknown_feature_id=int(unknown_feature_id),
        )

    def resolve_local_indices(
        self,
        dataset_index: int,
        local_feature_indices: Sequence[int] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if int(dataset_index) not in self.dataset_feature_mappings:
            raise KeyError(f"no global feature mapping registered for dataset_index {dataset_index}")

        mapping = self.dataset_feature_mappings[int(dataset_index)]
        local_indices = np.asarray(local_feature_indices, dtype=np.int64)
        resolved = np.full(local_indices.shape, self.unknown_feature_id, dtype=np.int32)
        within_bounds = (local_indices >= 0) & (local_indices < mapping.shape[0])
        if np.any(within_bounds):
            resolved[within_bounds] = mapping[local_indices[within_bounds]]
        valid = within_bounds & (resolved != self.unknown_feature_id)
        return resolved, valid

    def resolve_batch(
        self,
        payload: SparseBatchPayload,
        *,
        sort_by_global_feature: bool = True,
        drop_unresolved: bool = True,
    ) -> ResolvedSparseBatch:
        row_offsets = [0]
        resolved_gene_ids: list[np.ndarray] = []
        resolved_counts: list[np.ndarray] = []
        unresolved_total = 0

        for row_position in range(payload.batch_size):
            dataset_index = int(payload.dataset_index[row_position])
            local_indices = payload.row_gene_indices(row_position)
            counts = payload.row_counts(row_position)
            resolved_ids, valid_mask = self.resolve_local_indices(dataset_index, local_indices)
            unresolved_total += int(valid_mask.size - valid_mask.sum())

            if drop_unresolved:
                resolved_ids = resolved_ids[valid_mask]
                counts = counts[valid_mask]

            if sort_by_global_feature and resolved_ids.size:
                order = np.argsort(resolved_ids, kind="stable")
                resolved_ids = resolved_ids[order]
                counts = counts[order]

            resolved_gene_ids.append(resolved_ids.astype(np.int32, copy=False))
            resolved_counts.append(counts.astype(np.int32, copy=False))
            row_offsets.append(row_offsets[-1] + int(resolved_ids.size))

        return ResolvedSparseBatch(
            batch_size=payload.batch_size,
            global_row_index=payload.global_row_index.copy(),
            dataset_index=payload.dataset_index.copy(),
            local_row_index=payload.local_row_index.copy(),
            row_offsets=np.asarray(row_offsets, dtype=np.int64),
            global_feature_ids=(
                np.concatenate(resolved_gene_ids).astype(np.int32, copy=False)
                if resolved_gene_ids
                else np.asarray([], dtype=np.int32)
            ),
            expression_counts=(
                np.concatenate(resolved_counts).astype(np.int32, copy=False)
                if resolved_counts
                else np.asarray([], dtype=np.int32)
            ),
            size_factor=payload.size_factor.copy(),
            dataset_id=tuple(payload.dataset_id),
            cell_id=tuple(payload.cell_id),
            canonical_perturbation=tuple(dict(item) for item in payload.canonical_perturbation),
            canonical_context=tuple(dict(item) for item in payload.canonical_context),
            unresolved_local_features=unresolved_total,
        )


class CorpusRandomBatchSampler:
    """Yield corpus-global batches using ``MetadataIndex.sample()``.

    Each batch draws ``batch_size`` random cells from the full corpus.
    Seed varies per epoch and batch to produce different draws.
    """

    def __init__(
        self,
        *,
        metadata_index: "MetadataIndex",
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._meta = metadata_index
        self.total_rows = len(metadata_index)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_rows // self.batch_size
        return (self.total_rows + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        num_batches = len(self)
        for batch_idx in range(num_batches):
            batch_seed = self.seed + self.epoch * 10000 + batch_idx
            sampled = self._meta.sample(self.batch_size, seed=batch_seed)
            yield sampled["global_row_index"].to_list()


class DatasetBatchSampler:
    """Yield batches restricted to a single dataset using ``MetadataIndex``."""

    def __init__(
        self,
        *,
        metadata_index: "MetadataIndex",
        dataset_index: int,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._full_meta = metadata_index
        self._meta = metadata_index.filter(
            pl.col("dataset_index") == int(dataset_index)
        )
        self._row_count = len(self._meta)
        if self._row_count == 0:
            raise ValueError(
                f"dataset_index {dataset_index} has no rows in metadata"
            )
        self.dataset_index = int(dataset_index)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self._row_count // self.batch_size
        return (self._row_count + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        # Get all global indices for this dataset
        all_indices = np.asarray(
            self._meta.df["global_row_index"].to_numpy(),
            dtype=np.int64,
        )
        if self.shuffle:
            rng.shuffle(all_indices)
        for start in range(0, len(all_indices), self.batch_size):
            batch = all_indices[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch.tolist()


class DatasetContextBatchSampler:
    """Yield batches stratified by a context column using ``MetadataIndex``.

    Each context group (unique value of *context_field*) yields one batch
    of *batch_size* randomly sampled cells.
    """

    def __init__(
        self,
        *,
        metadata_index: "MetadataIndex",
        batch_size: int,
        context_field: str = "raw_cell_type",
        dataset_index: int | None = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._meta = metadata_index
        if dataset_index is not None:
            self._meta = self._meta.filter(
                pl.col("dataset_index") == int(dataset_index)
            )
        self.batch_size = int(batch_size)
        self.context_field = context_field
        self.dataset_index = (
            int(dataset_index) if dataset_index is not None else None
        )
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        # Pre-compute unique context values
        if context_field not in self._meta.df.columns:
            raise ValueError(
                f"context_field '{context_field}' not found in metadata. "
                f"Available columns: {self._meta.df.columns}"
            )
        self._context_values = sorted(
            self._meta.df[context_field].unique().to_list()
        )
        if not self._context_values:
            raise ValueError(
                f"no context groups found for field '{context_field}'"
            )

    def __len__(self) -> int:
        return len(self._context_values)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        values = list(self._context_values)
        if self.shuffle:
            rng.shuffle(values)

        for ctx_val in values:
            filtered = self._meta.filter(
                pl.col(self.context_field) == ctx_val
            )
            n_available = len(filtered)
            if n_available == 0:
                continue
            actual_size = min(self.batch_size, n_available)
            if actual_size < self.batch_size and self.drop_last:
                continue
            ctx_seed = self.seed + self.epoch * 10000 + hash(str(ctx_val)) % 10000
            sampled = filtered.sample(actual_size, seed=ctx_seed)
            yield sampled["global_row_index"].to_list()


class CPUDenseRuntimePath:
    """Baseline runtime path that resolves sparse rows, then densifies on CPU."""

    def __init__(self, resolver: GlobalFeatureResolver, *, total_features: int | None = None):
        self.resolver = resolver
        self.total_features = int(
            resolver.total_features if total_features is None else total_features
        )
        if self.total_features <= 0:
            raise ValueError("CPUDenseRuntimePath requires a positive total_features")

    def resolve_batch(self, payload: SparseBatchPayload) -> ResolvedSparseBatch:
        return self.resolver.resolve_batch(payload)

    def densify(self, batch: SparseBatchPayload | ResolvedSparseBatch) -> dict[str, Any]:
        resolved = batch if isinstance(batch, ResolvedSparseBatch) else self.resolve_batch(batch)
        dense_counts = np.zeros((resolved.batch_size, self.total_features), dtype=np.float32)
        for row_position in range(resolved.batch_size):
            feature_ids = resolved.row_feature_ids(row_position)
            if feature_ids.size == 0:
                continue
            counts = resolved.row_counts(row_position).astype(np.float32, copy=False)
            np.add.at(dense_counts[row_position], feature_ids, counts)
        return {
            "batch_size": resolved.batch_size,
            "global_row_index": resolved.global_row_index.copy(),
            "dataset_index": resolved.dataset_index.copy(),
            "local_row_index": resolved.local_row_index.copy(),
            "size_factor": resolved.size_factor.copy(),
            "dense_counts": dense_counts,
            "dataset_id": tuple(resolved.dataset_id),
            "cell_id": tuple(resolved.cell_id),
            "canonical_perturbation": tuple(dict(item) for item in resolved.canonical_perturbation),
            "canonical_context": tuple(dict(item) for item in resolved.canonical_context),
            "unresolved_local_features": resolved.unresolved_local_features,
        }

    def gather_sampled_counts(
        self,
        batch: SparseBatchPayload | ResolvedSparseBatch,
        sampled_feature_ids: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        dense_batch = self.densify(batch)
        dense_counts = dense_batch["dense_counts"]
        sampled = np.asarray(sampled_feature_ids, dtype=np.int64)
        if sampled.ndim == 1:
            sampled = np.broadcast_to(sampled, (dense_counts.shape[0], sampled.shape[0]))
        if sampled.shape[0] != dense_counts.shape[0]:
            raise ValueError("sampled_feature_ids batch dimension does not match dense batch")
        return dense_counts[np.arange(dense_counts.shape[0])[:, None], sampled]


class GPUSparseRuntimePath:
    """Sparse runtime path that keeps flat payloads plus offsets in the hot path."""

    def __init__(self, resolver: GlobalFeatureResolver):
        self.resolver = resolver

    def resolve_batch(self, payload: SparseBatchPayload) -> ResolvedSparseBatch:
        return self.resolver.resolve_batch(payload)

    def gather_sampled_counts(
        self,
        batch: SparseBatchPayload | ResolvedSparseBatch,
        sampled_feature_ids: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        resolved = batch if isinstance(batch, ResolvedSparseBatch) else self.resolve_batch(batch)
        sampled = np.asarray(sampled_feature_ids, dtype=np.int64)
        if sampled.ndim == 1:
            sampled = np.broadcast_to(sampled, (resolved.batch_size, sampled.shape[0]))
        if sampled.shape[0] != resolved.batch_size:
            raise ValueError("sampled_feature_ids batch dimension does not match sparse batch")

        gathered = np.zeros(sampled.shape, dtype=np.float32)
        for row_position in range(resolved.batch_size):
            row_feature_ids = resolved.row_feature_ids(row_position)
            if row_feature_ids.size == 0:
                continue
            row_counts = resolved.row_counts(row_position).astype(np.float32, copy=False)
            row_targets = sampled[row_position]
            positions = np.searchsorted(row_feature_ids, row_targets, side="left")
            in_bounds = positions < row_feature_ids.size
            if not np.any(in_bounds):
                continue
            clamped = np.clip(positions, 0, row_feature_ids.size - 1)
            exact = in_bounds & (row_feature_ids[clamped] == row_targets)
            if np.any(exact):
                gathered[row_position, exact] = row_counts[clamped[exact]]
        return gathered


# ---------------------------------------------------------------------------
# Sampler state — shared across all sampler modes
# ---------------------------------------------------------------------------


@dataclass
class SamplerState:
    """Per-sampler mutable state for tracking sampling decisions."""

    mode: str  # random_context | expressed_zeros | hvg_random
    total_cells: int
    n_genes: int
    expressed_threshold: int = 1  # minimum count to be considered "expressed"
    hvg_set: tuple[int, ...] = field(default_factory=tuple)  # HVG token IDs

    def __post_init__(self) -> None:
        if self.mode not in {"random_context", "expressed_zeros", "hvg_random"}:
            raise ValueError(f"unknown sampler mode: {self.mode}")


# ---------------------------------------------------------------------------
# Random Context Sampler
# ---------------------------------------------------------------------------


class RandomContextSampler:
    """Sampler: selects a random gene context of fixed size per cell.

    Produces a fixed-size gene subset uniformly at random from the full vocab,
    regardless of whether those genes are expressed. Used for baseline context
    training where the model learns to predict random missing genes.

    ``sample_indices`` accepts either a ``CellState`` (backward-compatible)
    or a raw ``np.ndarray`` of expressed gene indices for the hot path.
    """

    def __init__(self, state: SamplerState, rng: np.random.Generator):
        self.state = state
        self.rng = rng

    def sample_indices(
        self,
        cell_or_genes: CellState | np.ndarray,
        context_size: int,
    ) -> np.ndarray:
        """Return a random context of *context_size* gene indices.

        Parameters
        ----------
        cell_or_genes : CellState or np.ndarray
            ``CellState`` for backward compatibility, or a numpy array
            of expressed gene indices for the hot path.  This sampler
            does not use expression data — it samples uniformly from
            the full vocab regardless.
        context_size : int
            Number of random genes to sample.
        """
        if context_size > self.state.n_genes:
            context_size = self.state.n_genes
        return self.rng.choice(
            self.state.n_genes, size=context_size, replace=False
        ).astype(np.int32)

    def sample_batch(
        self, cells: list[CellState], context_size: int
    ) -> list[tuple[CellState, np.ndarray]]:
        """Sample a batch of (cell, random_context) pairs."""
        return [(cell, self.sample_indices(cell, context_size)) for cell in cells]


# ---------------------------------------------------------------------------
# Expressed + Zeros Sampler
# ---------------------------------------------------------------------------


class ExpressedZerosSampler:
    """Sampler: selects expressed genes + an equal number of zero genes.

    Produces a mixed context with all expressed genes plus an equal count of
    randomly sampled unexpressed genes. Used for training on expressed+
    zero context so the model learns both signal and silence.

    ``sample_indices`` accepts either a ``CellState`` (backward-compatible)
    or a raw ``np.ndarray`` of expressed gene indices for the hot path.
    """

    def __init__(self, state: SamplerState, rng: np.random.Generator):
        self.state = state
        self.rng = rng

    def sample_indices(
        self,
        cell_or_genes: CellState | np.ndarray,
        max_context: int | None = None,
    ) -> np.ndarray:
        """Return expressed + equal zeros, capped at *max_context*.

        Parameters
        ----------
        cell_or_genes : CellState or np.ndarray
            ``CellState`` for backward compatibility, or a numpy array
            of expressed gene indices for the hot path.
        max_context : int, optional
            Maximum context size.  If ``None``, uses the full vocabulary.
        """
        expressed = set(
            cell_or_genes.expressed_gene_indices
            if isinstance(cell_or_genes, CellState)
            else cell_or_genes.tolist()
        )
        n_expressed = len(expressed)
        max_zeros = (
            (max_context - n_expressed) // 2
            if max_context
            else self.state.n_genes - n_expressed
        )
        n_zeros = min(max_zeros, self.state.n_genes - n_expressed)
        zero_candidates = list(set(range(self.state.n_genes)) - expressed)
        zero_indices = self.rng.choice(zero_candidates, size=n_zeros, replace=False)
        context = np.array(list(expressed) + list(zero_indices), dtype=np.int32)
        return context

    def sample_batch(
        self,
        cells: list[CellState],
        max_context: int | None = None,
    ) -> list[tuple[CellState, np.ndarray]]:
        """Sample a batch of (cell, expressed+zeros context) pairs."""
        return [(cell, self.sample_indices(cell, max_context)) for cell in cells]


# ---------------------------------------------------------------------------
# HVGs + Random Sampler
# ---------------------------------------------------------------------------


class HVGRandomSampler:
    """Sampler: selects HVG genes + an equal number of random non-HVG genes.

    Produces a mixed context with highly variable genes plus randomly sampled
    non-HVG genes. Used for focusing training on variable genes while keeping
    a baseline representation of other genes.

    ``sample_indices`` accepts either a ``CellState`` (backward-compatible)
    or a raw ``np.ndarray`` of expressed gene indices for the hot path.
    """

    def __init__(self, state: SamplerState, rng: np.random.Generator):
        self.state = state
        self.rng = rng

    def sample_indices(
        self,
        cell_or_genes: CellState | np.ndarray,
        max_context: int | None = None,
    ) -> np.ndarray:
        """Return HVG + equal random non-HVG, capped at *max_context*.

        Parameters
        ----------
        cell_or_genes : CellState or np.ndarray
            ``CellState`` for backward compatibility, or a numpy array
            of expressed gene indices for the hot path.
        max_context : int, optional
            Maximum context size.  If ``None``, uses the full vocabulary.
        """
        hvg_set = set(self.state.hvg_set)
        expressed = set(
            cell_or_genes.expressed_gene_indices
            if isinstance(cell_or_genes, CellState)
            else cell_or_genes.tolist()
        )
        hvg_expressed = hvg_set & expressed
        n_hvg = len(hvg_expressed)
        max_nonhvg = (
            (max_context - n_hvg) // 2 if max_context else self.state.n_genes - n_hvg
        )
        nonhvg_candidates = list(set(range(self.state.n_genes)) - hvg_set)
        n_nonhvg = min(max_nonhvg, len(nonhvg_candidates))
        nonhvg_indices = self.rng.choice(
            nonhvg_candidates, size=n_nonhvg, replace=False
        )
        context = np.array(list(hvg_expressed) + list(nonhvg_indices), dtype=np.int32)
        return context

    def sample_batch(
        self,
        cells: list[CellState],
        max_context: int | None = None,
    ) -> list[tuple[CellState, np.ndarray]]:
        """Sample a batch of (cell, HVG+random context) pairs."""
        return [(cell, self.sample_indices(cell, max_context)) for cell in cells]


# ---------------------------------------------------------------------------
# Streaming IterableDataset
# ---------------------------------------------------------------------------


class PerturbIterableDataset:
    """PyTorch-friendly IterableDataset wrapping a ``BatchExecutor``.

    Yields dicts with sparse expression data, context indices, and metadata
    for each cell.  Collators are expected to handle padding and masking.
    """

    def __init__(
        self,
        batch_executor: "BatchExecutor",
        n_genes: int,
        *,
        sampler_mode: str = "random_context",
        shuffle: bool = True,
        seed: int = 42,
        context_size: int | None = None,
        max_context: int | None = None,
        hvg_set: tuple[int, ...] = (),
    ):
        self._exec = batch_executor
        self.shuffle = shuffle
        self.seed = seed
        self.context_size = context_size
        self.max_context = max_context
        self.rng = np.random.default_rng(seed)

        state = SamplerState(
            mode=sampler_mode,
            total_cells=len(batch_executor),
            n_genes=n_genes,
            hvg_set=hvg_set,
        )

        if sampler_mode == "random_context":
            self.sampler = RandomContextSampler(state, self.rng)
        elif sampler_mode == "expressed_zeros":
            self.sampler = ExpressedZerosSampler(state, self.rng)
        elif sampler_mode == "hvg_random":
            self.sampler = HVGRandomSampler(state, self.rng)
        else:
            raise ValueError(f"unknown sampler mode: {sampler_mode}")

        self._indices: list[int] | None = None

    def _ensure_indices(self) -> None:
        if self._indices is None:
            self._indices = list(range(len(self._exec)))
            if self.shuffle:
                self.rng.shuffle(self._indices)

    def __iter__(self):
        self._ensure_indices()
        for cell_idx in self._indices:
            cells = self._exec.read_cells([cell_idx])
            if not cells:
                continue
            cell = cells[0]
            if self.context_size is not None:
                context = self.sampler.sample_indices(cell, self.context_size)
            elif self.max_context is not None:
                context = self.sampler.sample_indices(cell, self.max_context)
            else:
                context = np.array(cell.expressed_gene_indices, dtype=np.int32)

            yield {
                "cell_id": cell.cell_id,
                "dataset_id": cell.dataset_id,
                "dataset_index": cell.dataset_index,
                "global_row_index": cell.global_row_index,
                "local_row_index": cell.local_row_index,
                "expressed_gene_indices": np.array(
                    cell.expressed_gene_indices, dtype=np.int32
                ),
                "expression_counts": np.array(cell.expression_counts, dtype=np.int32),
                "context_indices": context,
                "size_factor": cell.size_factor,
                "canonical_perturbation": cell.canonical_perturbation,
                "canonical_context": cell.canonical_context,
            }

    def __len__(self) -> int:
        return len(self._exec)


# ---------------------------------------------------------------------------
# Optional map-style dataset
# ---------------------------------------------------------------------------


class PerturbDataLoader:
    """Map-style dataset wrapper around a ``BatchExecutor``.

    Supports indexed random access via ``__getitem__`` and batched reads
    via ``__getitems__`` (PyTorch 2.0 plural indexing).
    """

    def __init__(
        self,
        batch_executor: "BatchExecutor",
        n_genes: int,
        *,
        sampler_mode: str = "random_context",
        shuffle: bool = False,
        seed: int = 42,
        context_size: int | None = None,
        max_context: int | None = None,
        hvg_set: tuple[int, ...] = (),
    ):
        self._exec = batch_executor
        self.shuffle = shuffle
        self.seed = seed
        self.context_size = context_size
        self.max_context = max_context
        self.rng = np.random.default_rng(seed)

        state = SamplerState(
            mode=sampler_mode,
            total_cells=len(batch_executor),
            n_genes=n_genes,
            hvg_set=hvg_set,
        )

        if sampler_mode == "random_context":
            self.sampler = RandomContextSampler(state, self.rng)
        elif sampler_mode == "expressed_zeros":
            self.sampler = ExpressedZerosSampler(state, self.rng)
        elif sampler_mode == "hvg_random":
            self.sampler = HVGRandomSampler(state, self.rng)
        else:
            raise ValueError(f"unknown sampler mode: {sampler_mode}")

    def __len__(self) -> int:
        return len(self._exec)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cells = self._exec.read_cells([idx])
        if not cells:
            raise IndexError(f"index {idx} out of range")
        cell = cells[0]
        expressed = np.array(cell.expressed_gene_indices, dtype=np.int32)
        counts = np.array(cell.expression_counts, dtype=np.int32)
        if self.context_size is not None:
            context = self.sampler.sample_indices(cell, self.context_size)
        elif self.max_context is not None:
            context = self.sampler.sample_indices(cell, self.max_context)
        else:
            context = expressed

        return {
            "cell_id": cell.cell_id,
            "dataset_id": cell.dataset_id,
            "dataset_index": cell.dataset_index,
            "global_row_index": cell.global_row_index,
            "local_row_index": cell.local_row_index,
            "expressed_gene_indices": expressed,
            "expression_counts": counts,
            "context_indices": context,
            "size_factor": cell.size_factor,
            "canonical_perturbation": cell.canonical_perturbation,
            "canonical_context": cell.canonical_context,
        }

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """PyTorch 2.0 plural indexing — batch-read cells."""
        cells = self._exec.read_cells(list(indices))
        results: list[dict[str, Any]] = []
        for cell in cells:
            expressed = np.asarray(cell.expressed_gene_indices, dtype=np.int32)
            counts = np.asarray(cell.expression_counts, dtype=np.int32)
            if self.context_size is not None:
                context = self.sampler.sample_indices(cell, self.context_size)
            elif self.max_context is not None:
                context = self.sampler.sample_indices(cell, self.max_context)
            else:
                context = expressed
            results.append({
                "cell_id": cell.cell_id,
                "dataset_id": cell.dataset_id,
                "dataset_index": cell.dataset_index,
                "global_row_index": cell.global_row_index,
                "local_row_index": cell.local_row_index,
                "expressed_gene_indices": expressed,
                "expression_counts": counts,
                "context_indices": context,
                "size_factor": cell.size_factor,
                "canonical_perturbation": cell.canonical_perturbation,
                "canonical_context": cell.canonical_context,
            })
        return results
