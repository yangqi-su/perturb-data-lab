"""Phase 3 refactored: flat-batch dataset and collation layer.

This module implements:
- ExpressionBatch container class (composed by BatchExecutor)
- Shared batch samplers using MetadataIndex
- Map-style PerturbBatchDataset wrapping BatchExecutor
- Collation functions for DataLoader integration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch

__all__ = [
    "ExpressionBatch",
    "CorpusRandomBatchSampler",
    "DatasetBatchSampler",
    "DatasetContextBatchSampler",
    "PerturbBatchDataset",
    "collate_batch_dict",
    "cpu_parallel_collate_fn",
]


# ---------------------------------------------------------------------------
# Flat expression batch — zero per-cell Python objects
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


# ---------------------------------------------------------------------------
# Batch samplers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PerturbBatchDataset — production-facing Dataset for DataLoader + GPU pipeline
# ---------------------------------------------------------------------------


class PerturbBatchDataset:
    """Map-style Dataset bridging ``BatchExecutor.read_batch()`` with the GPU pipeline.

    Designed for use with ``torch.utils.data.DataLoader`` and a batch
    sampler that yields lists of corpus-global cell indices.  The
    ``__getitems__`` method calls ``executor.read_batch(indices)`` and
    returns a flat batch dict with **zero** per-cell Python objects.

    The Dataset does **not** call ``GPUSparsePipeline.process_batch()``.
    Sampling parameters (``seq_len``, ``sampling_mode``, etc.) are
    stored for informational purposes and should be forwarded to the
    pipeline in the training loop.

    Parameters
    ----------
    executor : BatchExecutor
        Corpus-level batch reader providing ``read_batch()``.
    seq_len : int, optional
        Number of genes to sample per cell (forwarded to the pipeline;
        not used by the Dataset itself).
    sampling_mode : str, default "uniform"
        Sampling strategy (``"uniform"``, ``"expressed"``, ``"hvg"``).
    expressed_weight : float, default 3.0
        Weight bonus for expressed genes in ``"expressed"`` mode.
    hvg_weight : float, default 3.0
        Weight bonus for HVG genes in ``"hvg"`` mode.
    """

    def __init__(
        self,
        executor: "BatchExecutor",
        seq_len: int | None = None,
        *,
        sampling_mode: str = "uniform",
        expressed_weight: float = 3.0,
        hvg_weight: float = 3.0,
    ):
        self._exec = executor
        self._seq_len = seq_len
        self._sampling_mode = sampling_mode
        self._expressed_weight = float(expressed_weight)
        self._hvg_weight = float(hvg_weight)

    # -- Properties ---------------------------------------------------------

    @property
    def executor(self) -> "BatchExecutor":
        return self._exec

    @property
    def seq_len(self) -> int | None:
        return self._seq_len

    @property
    def sampling_mode(self) -> str:
        return self._sampling_mode

    @property
    def expressed_weight(self) -> float:
        return self._expressed_weight

    @property
    def hvg_weight(self) -> float:
        return self._hvg_weight

    # -- Map-style interface -------------------------------------------------

    def __len__(self) -> int:
        return len(self._exec)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Read a batch of cells and return the flat batch dict.

        Parameters
        ----------
        indices : sequence of int
            Corpus-global cell indices for this batch.

        Returns
        -------
        list[dict]
            Single-element list containing the flat batch dict from
            ``BatchExecutor.read_batch()``.  All values are numpy arrays
            (no GPU tensors, no ``CellState`` objects).
        """
        batch = self._exec.read_batch(list(indices))
        return [batch]


def collate_batch_dict(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate ``PerturbBatchDataset.__getitems__`` output into a batch dict.

    Converts numpy arrays to CPU tensors so that ``DataLoader(pin_memory=True)``
    can perform asynchronous host→device transfers.  Non-array fields
    (``dataset_id``, ``cell_id``, ``canonical_perturbation``,
    ``canonical_context``) are passed through unchanged.

    Parameters
    ----------
    items : list[dict]
        Single-element list from ``PerturbBatchDataset.__getitems__``.

    Returns
    -------
    dict
        Flat batch dict with numpy arrays replaced by CPU tensors.
        Compatible with ``GPUSparsePipeline.process_batch()`` and
        ``CPUPipeline.process_batch()``.
    """
    if not items:
        raise ValueError("collate_batch_dict received empty list")
    batch = items[0]

    return {
        "batch_size": batch["batch_size"],
        "global_row_index": torch.as_tensor(
            batch["global_row_index"], dtype=torch.long
        ),
        "row_offsets": torch.as_tensor(
            batch["row_offsets"], dtype=torch.long
        ),
        "expressed_gene_indices": torch.as_tensor(
            batch["expressed_gene_indices"], dtype=torch.long
        ),
        "expression_counts": torch.as_tensor(
            batch["expression_counts"], dtype=torch.float32
        ),
        "dataset_index": torch.as_tensor(
            batch["dataset_index"], dtype=torch.long
        ),
        "size_factor": torch.as_tensor(
            batch["size_factor"], dtype=torch.float32
        ),
        "local_row_index": torch.as_tensor(
            batch.get("local_row_index", batch["global_row_index"]),
            dtype=torch.long,
        ),
        # Non-tensor fields: pass through unchanged
        "dataset_id": batch.get("dataset_id", ()),
        "cell_id": batch.get("cell_id", ()),
        "canonical_perturbation": batch.get("canonical_perturbation", ()),
        "canonical_context": batch.get("canonical_context", ()),
    }


# ---------------------------------------------------------------------------
# CPU-parallel collate function for worker-side compute
# ---------------------------------------------------------------------------

# Per-worker flag to ensure torch.set_num_threads(1) is called exactly once.
# Set to True on first invocation in each worker process.
_cpu_collate_threads_initialized: bool = False


def cpu_parallel_collate_fn(
    items: list[dict[str, Any]],
    pipeline: "GPUSparsePipeline",
    sampling_mode: str = "uniform",
    *,
    expressed_weight: float = 3.0,
    hvg_weight: float = 3.0,
    generator: torch.Generator | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Worker-safe collate function that runs the full pipeline on CPU.

    Designed for use with ``torch.utils.data.DataLoader`` and
    ``PerturbBatchDataset``.  The pipeline runs inside each DataLoader
    worker with ``device="cpu"``, so all heavy compute (local→global
    resolution, sort, weighted sampling, searchsorted+gather) is
    parallelized across workers via CPU vectorized ops.

    Parameters
    ----------
    items : list[dict]
        Single-element list from ``PerturbBatchDataset.__getitems__``.
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
        ``size_factor``, ``batch_size``, ``seq_len``.
    """
    global _cpu_collate_threads_initialized

    # Prevent OpenMP thread explosion: each worker uses 1 thread.
    # Parallelism comes from multiple workers, not intra-op threading.
    if not _cpu_collate_threads_initialized:
        torch.set_num_threads(1)
        _cpu_collate_threads_initialized = True

    if not items:
        raise ValueError("cpu_parallel_collate_fn received empty list")
    batch = items[0]

    return pipeline.process_batch(
        batch,
        device="cpu",
        sampling_mode=sampling_mode,
        expressed_weight=expressed_weight,
        hvg_weight=hvg_weight,
        generator=generator,
        **kwargs,
    )
