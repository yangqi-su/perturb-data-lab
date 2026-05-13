"""Modern expression-batch datasets, samplers, and collates.

The intended public composition pattern is::

    dataset = corpus.dataset()  # ExpressionBatchDataset
    sampler = corpus.set_sampler(batch_size=256, seed=0)
    gpu_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_expression_batch,
    )

For CPU worker-side sparse processing, use ``collate_expression_batch_cpu``
with a ``GPUSparsePipeline(feature_registry, seq_len=...)`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch

__all__ = [
    "ExpressionBatch",
    "DatasetRoutingTable",
    "CorpusRandomBatchSampler",
    "DatasetBatchSampler",
    "DatasetContextBatchSampler",
    "ExpressionBatchDataset",
    "collate_expression_batch",
    "collate_expression_batch_cpu",
]


def _coerce_optional_float32_array(values: np.ndarray | None) -> np.ndarray | None:
    """Convert a full-column numeric array to float32 or omit if all-missing."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or np.isnan(arr).all():
        return None
    return arr


def _normalize_candidate_row_indices(
    metadata_index: "MetadataIndex",
    row_indices: Sequence[int] | np.ndarray | None,
) -> np.ndarray | None:
    """Validate an optional corpus-global candidate row universe."""
    if row_indices is None:
        return None

    raw = np.asarray(row_indices)
    if raw.ndim != 1:
        raise ValueError(
            "row_indices must be a 1-D sequence of corpus-global row indices"
        )
    if raw.size == 0:
        raise ValueError("row_indices must contain at least one corpus-global row index")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("row_indices must contain only integer corpus-global row indices")

    normalized = raw.astype(np.int64, copy=False)
    if np.unique(normalized).size != normalized.size:
        raise ValueError("row_indices must be unique corpus-global row indices")
    if np.any(normalized < 0) or np.any(normalized >= len(metadata_index)):
        raise IndexError("row_indices contains out-of-range corpus-global row indices")
    return normalized.copy()


@dataclass(frozen=True)
class ExpressionBatch:
    """Flat expression arrays for a batch — zero per-cell Python objects.

    Designed as the output of ``read_expression_flat(...)`` so datasets and
    corpus helpers can assemble loader-ready raw batch dicts without per-cell
    Python objects.

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
class DatasetRoutingTable:
    """Compact dataset-index routing state for expression-only batches.

    The table stores only the worker-side information needed to map a batch of
    corpus-global row indices back to dataset indices, plus optional size-factor
    pass-through data.
    """

    dataset_starts: np.ndarray
    dataset_stops: np.ndarray
    dataset_indices: np.ndarray
    size_factor: np.ndarray | None = None
    dataset_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        starts = np.asarray(self.dataset_starts, dtype=np.int64)
        stops = np.asarray(self.dataset_stops, dtype=np.int64)
        indices = np.asarray(self.dataset_indices, dtype=np.int32)
        object.__setattr__(self, "dataset_starts", starts)
        object.__setattr__(self, "dataset_stops", stops)
        object.__setattr__(self, "dataset_indices", indices)

        n_entries = len(starts)
        if len(stops) != n_entries or len(indices) != n_entries:
            raise ValueError(
                "dataset_starts, dataset_stops, and dataset_indices must have matching lengths"
            )
        if np.any(stops <= starts):
            raise ValueError("dataset routing entries must have positive width")
        if n_entries and starts[0] != 0:
            raise ValueError("dataset routing coverage must start at global row 0")
        if n_entries > 1 and not np.array_equal(starts[1:], stops[:-1]):
            raise ValueError(
                "dataset routing entries must provide contiguous, non-overlapping coverage"
            )

        normalized_ids = tuple(str(dataset_id) for dataset_id in self.dataset_ids)
        if normalized_ids and len(normalized_ids) != n_entries:
            raise ValueError("dataset_ids must match the number of routing entries")
        object.__setattr__(self, "dataset_ids", normalized_ids)

        normalized_size_factor = _coerce_optional_float32_array(self.size_factor)
        if normalized_size_factor is not None and len(normalized_size_factor) != self.total_rows:
            raise ValueError(
                "size_factor must align with the full corpus-global row range"
            )
        object.__setattr__(self, "size_factor", normalized_size_factor)

    @property
    def total_rows(self) -> int:
        """Total number of routable corpus-global rows."""
        if self.dataset_stops.size == 0:
            return 0
        return int(self.dataset_stops[-1])

    def resolve_dataset_indices(self, indices: np.ndarray) -> np.ndarray:
        """Map corpus-global row indices to dataset indices."""
        if self.dataset_stops.size == 0:
            if len(indices) == 0:
                return np.array([], dtype=np.int32)
            raise IndexError("DatasetRoutingTable has no routing entries")

        entry_pos = np.searchsorted(self.dataset_stops, indices, side="right")
        if np.any(entry_pos < 0) or np.any(entry_pos >= len(self.dataset_stops)):
            raise IndexError("Expression batch contains out-of-range indices")

        if np.any(indices < self.dataset_starts[entry_pos]):
            raise IndexError("Expression batch contains indices outside routing coverage")

        return self.dataset_indices[entry_pos]

    def take_size_factor(self, indices: np.ndarray) -> np.ndarray | None:
        """Gather optional size factors for corpus-global row indices."""
        if self.size_factor is None:
            return None
        return self.size_factor[indices]


# ---------------------------------------------------------------------------
# Batch samplers
# ---------------------------------------------------------------------------


class CorpusRandomBatchSampler:
    """Yield corpus-global batches using direct integer sampling (numpy).

    Each batch draws ``batch_size`` random cells from the full corpus
    via ``numpy.random`` without any Polars DataFrame ``.sample()``
    overhead.  Seed varies per epoch and batch to produce different draws.

    This replaces the earlier Polars-based path and is the only sampler
    targeted by the Phase 6 fast-path optimization.
    """

    def __init__(
        self,
        *,
        metadata_index: "MetadataIndex",
        batch_size: int,
        row_indices: Sequence[int] | np.ndarray | None = None,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._meta = metadata_index
        candidate_row_indices = _normalize_candidate_row_indices(
            metadata_index,
            row_indices,
        )
        if candidate_row_indices is None:
            candidate_row_indices = np.arange(len(metadata_index), dtype=np.int64)
        self._candidate_row_indices = candidate_row_indices
        self.total_rows = len(candidate_row_indices)
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
        tail_size = self.total_rows % self.batch_size
        for batch_idx in range(num_batches):
            batch_seed = self.seed + self.epoch * 10000 + batch_idx
            rng = np.random.default_rng(batch_seed)
            current_batch_size = self.batch_size
            if (
                not self.drop_last
                and tail_size
                and batch_idx == num_batches - 1
            ):
                current_batch_size = tail_size
            positions = rng.choice(
                self.total_rows, size=current_batch_size, replace=False
            )
            indices = self._candidate_row_indices[positions]
            yield sorted(indices.tolist())


class DatasetBatchSampler:
    """Yield batches restricted to a single dataset using ``MetadataIndex``."""

    def __init__(
        self,
        *,
        metadata_index: "MetadataIndex",
        dataset_index: int,
        batch_size: int,
        row_indices: Sequence[int] | np.ndarray | None = None,
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
        self._row_indices = np.asarray(
            self._meta.df["global_row_index"].to_numpy(),
            dtype=np.int64,
        ).copy()
        candidate_row_indices = _normalize_candidate_row_indices(
            metadata_index,
            row_indices,
        )
        if candidate_row_indices is not None:
            self._row_indices = self._row_indices[
                np.isin(
                    self._row_indices,
                    candidate_row_indices,
                    assume_unique=True,
                )
            ]
        self._row_count = len(self._row_indices)
        if self._row_count == 0:
            raise ValueError(
                f"dataset_index {dataset_index} has no rows under the requested row_indices restriction"
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
        all_indices = self._row_indices.copy()
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
        row_indices: Sequence[int] | np.ndarray | None = None,
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
        candidate_row_indices = _normalize_candidate_row_indices(
            metadata_index,
            row_indices,
        )
        if candidate_row_indices is not None:
            self._meta = self._meta.filter(
                pl.col("global_row_index").is_in(candidate_row_indices.tolist())
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


class ExpressionBatchDataset:
    """Backend-neutral expression-only dataset with lightweight routing state.

    Works for any backend whose expression reader exposes
    ``read_expression_flat()`` with order-preserving output. ``Corpus.dataset()``
    returns this type for custom ``torch.utils.data.DataLoader`` composition.
    """

    def __init__(
        self,
        expression_reader: Any,
        *,
        routing_table: DatasetRoutingTable,
        topology: str = "aggregate",
        backend: str = "lance",
    ):
        self._reader = expression_reader
        self._routing_table = routing_table
        self._topology = topology
        self._backend = backend

    @property
    def routing_table(self) -> DatasetRoutingTable:
        """Return the compact routing table used for dataset-index lookup."""
        return self._routing_table

    @property
    def topology(self) -> str:
        return self._topology

    @property
    def backend(self) -> str:
        return self._backend

    def __len__(self) -> int:
        return self._routing_table.total_rows

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        index_array = np.asarray(indices, dtype=np.int64)
        expr = self._reader.read_expression_flat(index_array.tolist())
        dataset_index = self._routing_table.resolve_dataset_indices(index_array)
        batch: dict[str, Any] = {
            "batch_size": expr.batch_size,
            "global_row_index": expr.global_row_index,
            "dataset_index": dataset_index,
            "row_offsets": expr.row_offsets,
            "expressed_gene_indices": expr.expressed_gene_indices,
            "expression_counts": expr.expression_counts,
        }
        size_factor = self._routing_table.take_size_factor(index_array)
        if size_factor is not None:
            batch["size_factor"] = size_factor
        return [batch]


def _unwrap_single_prebatched_item(
    items: list[dict[str, Any]],
    *,
    collate_name: str,
) -> dict[str, Any]:
    """Return the single pre-batched item emitted by loader datasets."""
    if not items:
        raise ValueError(f"{collate_name} received empty list")
    if len(items) != 1:
        raise ValueError(
            f"{collate_name} expected a single pre-batched item, "
            f"got {len(items)}"
        )
    return items[0]


def collate_expression_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Unwrap one raw batch item and convert array fields to torch tensors.

    This is the public collate for custom ``DataLoader`` users who want the raw
    expression batch tensors before sparse processing runs in the main process.
    """
    batch = _unwrap_single_prebatched_item(
        items,
        collate_name="collate_expression_batch",
    )

    result: dict[str, Any] = {
        "batch_size": batch["batch_size"],
        "global_row_index": torch.as_tensor(
            batch["global_row_index"], dtype=torch.long
        ),
        "dataset_index": torch.as_tensor(
            batch["dataset_index"], dtype=torch.long
        ),
        "row_offsets": torch.as_tensor(batch["row_offsets"], dtype=torch.long),
        "expressed_gene_indices": torch.as_tensor(
            batch["expressed_gene_indices"], dtype=torch.long
        ),
        "expression_counts": torch.as_tensor(
            batch["expression_counts"], dtype=torch.float32
        ),
    }

    if "local_row_index" in batch:
        result["local_row_index"] = torch.as_tensor(
            batch["local_row_index"], dtype=torch.long
        )
    if "size_factor" in batch:
        result["size_factor"] = torch.as_tensor(
            batch["size_factor"], dtype=torch.float32
        )
    if "meta_columns" in batch:
        result["meta_columns"] = batch["meta_columns"]

    return result


# ---------------------------------------------------------------------------
# CPU-parallel collate function for worker-side compute
# ---------------------------------------------------------------------------

# Per-worker flag to ensure torch.set_num_threads(1) is called exactly once.
# Set to True on first invocation in each worker process.
_cpu_collate_threads_initialized: bool = False


def collate_expression_batch_cpu(
    items: list[dict[str, Any]],
    pipeline: "GPUSparsePipeline",
    sampling_mode: str = "uniform",
    *,
    expressed_weight: float = 3.0,
    hvg_weight: float = 3.0,
    hvg_top_k: int | None = None,
    generator: torch.Generator | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run sparse processing on CPU during DataLoader collation.

    Designed for use with ``torch.utils.data.DataLoader`` and
    pre-batched expression datasets. The pipeline runs inside each DataLoader
    worker with ``device="cpu"``, so all heavy compute (local→global
    resolution, sort, weighted sampling, searchsorted+gather) is
    parallelized across workers via CPU vectorized ops.

    Parameters
    ----------
    items : list[dict]
        Single-element list from a pre-batched expression dataset.
        The batch dict contains numpy arrays from the I/O layer.
    pipeline : GPUSparsePipeline
        Pipeline instance providing ``process_batch()``.  Must be
        picklable (no pre-created CUDA tensors).
    sampling_mode : str, default "uniform"
        Gene sampling strategy: ``"uniform"``, ``"expressed"``, or ``"hvg"``.
    expressed_weight : float, default 3.0
        Weight bonus for expressed genes in ``"expressed"`` mode.
    hvg_weight : float, default 3.0
        Weight bonus for HVG genes in ``"hvg"`` mode.
    hvg_top_k : int, optional
        Dynamic top-k threshold for ``"hvg"`` mode.
    generator : torch.Generator, optional
        Torch RNG generator for reproducible sampling.
    **kwargs
        Forwarded to ``pipeline.process_batch()``.

    Returns
    -------
    dict
        Fully processed batch dict with all tensors on CPU device.
        Keys: ``sampled_gene_ids``, ``sampled_counts``, ``valid_mask``,
        ``exact_match_mask``, ``dataset_index``, ``global_row_index``,
        ``batch_size``, ``seq_len``, and optional ``size_factor``.

    Use this collate when you want worker-side sparse processing instead of the
    default ``Corpus.loader(processing="gpu")`` main-process pipeline.
    """
    global _cpu_collate_threads_initialized

    # Prevent OpenMP thread explosion: each worker uses 1 thread.
    # Parallelism comes from multiple workers, not intra-op threading.
    if not _cpu_collate_threads_initialized:
        torch.set_num_threads(1)
        _cpu_collate_threads_initialized = True

    batch = _unwrap_single_prebatched_item(
        items,
        collate_name="collate_expression_batch_cpu",
    )

    result = pipeline.process_batch(
        batch,
        device="cpu",
        sampling_mode=sampling_mode,
        expressed_weight=expressed_weight,
        hvg_weight=hvg_weight,
        hvg_top_k=hvg_top_k,
        generator=generator,
        **kwargs,
    )
    if "local_row_index" in batch:
        result["local_row_index"] = torch.as_tensor(
            batch["local_row_index"], dtype=torch.long
        )
    if "meta_columns" in batch:
        result["meta_columns"] = batch["meta_columns"]
    return result
