"""PyTorch expression datasets, samplers, and loader construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from .gpu_pipeline import GPUSparsePipeline

__all__ = [
    "ExpressionBatch",
    "CorpusRandomBatchSampler",
    "DatasetBatchSampler",
    "DatasetContextBatchSampler",
    "ExpressionBatchDataset",
    "build_loader",
    "collate_expression_batch",
    "read_expression_raw_batch",
]


_RAW_BATCH_RESERVED_KEYS: frozenset[str] = frozenset(
    {
        "batch_size",
        "global_row_index",
        "dataset_index",
        "local_row_index",
        "size_factor",
        "row_offsets",
        "expressed_gene_indices",
        "expression_counts",
        "meta_columns",
    }
)

_LOADER_METADATA_RESERVED_OVERRIDES: frozenset[str] = frozenset(
    {"local_row_index", "size_factor"}
)


@dataclass(frozen=True)
class ExpressionBatch:
    """Flat sparse expression arrays for one batch."""

    batch_size: int
    global_row_index: np.ndarray
    row_offsets: np.ndarray
    expressed_gene_indices: np.ndarray
    expression_counts: np.ndarray

    def row_slice(self, row_position: int) -> slice:
        start = int(self.row_offsets[row_position])
        stop = int(self.row_offsets[row_position + 1])
        return slice(start, stop)

    def row_gene_indices(self, row_position: int) -> np.ndarray:
        return self.expressed_gene_indices[self.row_slice(row_position)]

    def row_counts(self, row_position: int) -> np.ndarray:
        return self.expression_counts[self.row_slice(row_position)]


def _normalize_batch_indices(indices: torch.Tensor | np.ndarray | Sequence[int]) -> np.ndarray:
    if isinstance(indices, torch.Tensor):
        return indices.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(indices, dtype=np.int64)


def _normalize_candidate_row_indices(
    metadata_index: "MetadataIndex",
    row_indices: Sequence[int] | np.ndarray | None,
) -> np.ndarray | None:
    if row_indices is None:
        return None

    raw = np.asarray(row_indices)
    if raw.ndim != 1:
        raise ValueError("row_indices must be a 1-D sequence of corpus-global row indices")
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


def _normalize_metadata_columns(
    metadata_index: "MetadataIndex",
    metadata_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    if metadata_columns is None:
        return ()
    if isinstance(metadata_columns, (str, bytes)):
        raise TypeError("metadata_columns must be a sequence of column names")

    normalized: list[str] = []
    seen: set[str] = set()
    for name in metadata_columns:
        if not isinstance(name, str):
            raise TypeError("metadata_columns must contain strings")
        if not name:
            raise ValueError("metadata_columns cannot contain empty names")
        if name in _RAW_BATCH_RESERVED_KEYS and name not in _LOADER_METADATA_RESERVED_OVERRIDES:
            raise ValueError(f"metadata_columns cannot request reserved raw batch field {name!r}")
        if name not in metadata_index.df.columns:
            raise ValueError(
                f"metadata column {name!r} not found. Available columns: {metadata_index.df.columns}"
            )
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return tuple(normalized)


def _optional_float32(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or np.isnan(arr).all():
        return None
    return arr


def _gather_batch_metadata(
    corpus: Any,
    global_row_index: torch.Tensor | np.ndarray | Sequence[int],
    metadata_columns: tuple[str, ...],
) -> dict[str, Any]:
    indices = _normalize_batch_indices(global_row_index)
    columns = ["dataset_index"]
    if "size_factor" in corpus.metadata_index.df.columns:
        columns.append("size_factor")
    columns.extend(column for column in metadata_columns if column not in columns)
    return corpus.metadata_index.take(indices, columns)


def _attach_required_batch_metadata(
    corpus: Any,
    raw_batch: dict[str, Any],
    metadata_columns: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gathered = _gather_batch_metadata(
        corpus,
        raw_batch["global_row_index"],
        metadata_columns,
    )
    resolved = dict(raw_batch)
    resolved["dataset_index"] = np.asarray(gathered["dataset_index"], dtype=np.int32)
    size_factor = _optional_float32(gathered.get("size_factor"))
    if size_factor is not None:
        resolved["size_factor"] = size_factor
    return resolved, gathered


def _attach_requested_metadata(
    batch: dict[str, Any],
    gathered: dict[str, Any],
    metadata_columns: tuple[str, ...],
) -> dict[str, Any]:
    if not metadata_columns:
        return batch
    batch["meta_columns"] = {column: gathered[column] for column in metadata_columns}
    return batch


def read_expression_raw_batch(
    expression_reader: Any,
    indices: torch.Tensor | np.ndarray | Sequence[int],
) -> dict[str, Any]:
    """Read expression-only raw batch fields for corpus-global row indices."""
    index_array = _normalize_batch_indices(indices)
    expr = expression_reader.read_expression_flat(index_array.tolist())
    return {
        "batch_size": expr.batch_size,
        "global_row_index": expr.global_row_index,
        "row_offsets": expr.row_offsets,
        "expressed_gene_indices": expr.expressed_gene_indices,
        "expression_counts": expr.expression_counts,
    }


class CorpusRandomBatchSampler:
    """Yield random corpus-global row batches from an optional row universe."""

    def __init__(
        self,
        *,
        metadata_index: "MetadataIndex",
        batch_size: int,
        row_indices: Sequence[int] | np.ndarray | None = None,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        candidate_row_indices = _normalize_candidate_row_indices(metadata_index, row_indices)
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
            rng = np.random.default_rng(self.seed + self.epoch * 10000 + batch_idx)
            current_batch_size = self.batch_size
            if not self.drop_last and tail_size and batch_idx == num_batches - 1:
                current_batch_size = tail_size
            positions = rng.choice(self.total_rows, size=current_batch_size, replace=False)
            indices = self._candidate_row_indices[positions]
            yield sorted(indices.tolist())


class DatasetBatchSampler:
    """Yield batches restricted to one ``dataset_index``."""

    def __init__(
        self,
        *,
        metadata_index: "MetadataIndex",
        dataset_index: int,
        batch_size: int,
        row_indices: Sequence[int] | np.ndarray | None = None,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        frame = metadata_index.df.filter(pl.col("dataset_index") == int(dataset_index))
        row_indices_arr = np.asarray(frame["global_row_index"].to_numpy(), dtype=np.int64).copy()
        candidate_row_indices = _normalize_candidate_row_indices(metadata_index, row_indices)
        if candidate_row_indices is not None:
            row_indices_arr = row_indices_arr[
                np.isin(row_indices_arr, candidate_row_indices, assume_unique=True)
            ]
        if len(row_indices_arr) == 0:
            raise ValueError(
                f"dataset_index {dataset_index} has no rows under the requested row_indices restriction"
            )
        self._row_indices = row_indices_arr
        self.dataset_index = int(dataset_index)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self._row_indices) // self.batch_size
        return (len(self._row_indices) + self.batch_size - 1) // self.batch_size

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
    """Yield one sampled batch per context group."""

    def __init__(
        self,
        *,
        metadata_index: "MetadataIndex",
        batch_size: int,
        context_field: str = "raw_cell_type",
        dataset_index: int | None = None,
        row_indices: Sequence[int] | np.ndarray | None = None,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if context_field not in metadata_index.df.columns:
            raise ValueError(
                f"context_field {context_field!r} not found. Available columns: {metadata_index.df.columns}"
            )

        frame = metadata_index.df
        if dataset_index is not None:
            frame = frame.filter(pl.col("dataset_index") == int(dataset_index))
        candidate_row_indices = _normalize_candidate_row_indices(metadata_index, row_indices)
        if candidate_row_indices is not None:
            frame = frame.filter(pl.col("global_row_index").is_in(candidate_row_indices.tolist()))

        grouped = frame.group_by(context_field, maintain_order=True).agg(
            pl.col("global_row_index").alias("row_indices")
        )
        if grouped.height == 0:
            raise ValueError(f"no context groups found for field {context_field!r}")

        self._groups: list[tuple[Any, np.ndarray]] = [
            (row[context_field], np.asarray(row["row_indices"], dtype=np.int64))
            for row in grouped.iter_rows(named=True)
        ]
        self.batch_size = int(batch_size)
        self.context_field = context_field
        self.dataset_index = int(dataset_index) if dataset_index is not None else None
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(rows) >= self.batch_size for _, rows in self._groups)
        return len(self._groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        group_order = np.arange(len(self._groups), dtype=np.int64)
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(group_order)

        for group_pos in group_order.tolist():
            _, rows = self._groups[group_pos]
            actual_size = min(self.batch_size, len(rows))
            if actual_size < self.batch_size and self.drop_last:
                continue
            group_rng = np.random.default_rng(self.seed + self.epoch * 10000 + group_pos)
            sampled = group_rng.choice(rows, size=actual_size, replace=False)
            yield sorted(sampled.astype(np.int64, copy=False).tolist())


class ExpressionBatchDataset:
    """Expression-only PyTorch dataset wrapper around an expression reader."""

    def __init__(self, expression_reader: Any, *, total_rows: int):
        self._reader = expression_reader
        self._total_rows = int(total_rows)

    def __len__(self) -> int:
        return self._total_rows

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        return [read_expression_raw_batch(self._reader, indices)]


def _unwrap_single_prebatched_item(items: list[dict[str, Any]], *, collate_name: str) -> dict[str, Any]:
    if not items:
        raise ValueError(f"{collate_name} received empty list")
    if len(items) != 1:
        raise ValueError(f"{collate_name} expected a single pre-batched item, got {len(items)}")
    return items[0]


def collate_expression_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Unwrap one expression-only batch and convert arrays to torch tensors."""
    batch = _unwrap_single_prebatched_item(items, collate_name="collate_expression_batch")
    return {
        "batch_size": batch["batch_size"],
        "global_row_index": torch.as_tensor(batch["global_row_index"], dtype=torch.long),
        "row_offsets": torch.as_tensor(batch["row_offsets"], dtype=torch.long),
        "expressed_gene_indices": torch.as_tensor(batch["expressed_gene_indices"], dtype=torch.long),
        "expression_counts": torch.as_tensor(batch["expression_counts"], dtype=torch.float32),
    }


def _normalize_sampler_kind(sampler: str) -> str:
    if not isinstance(sampler, str):
        raise TypeError("sampler must be a string")
    kind = sampler.strip().lower().replace("-", "_")
    if kind not in {"corpus_random", "dataset", "dataset_context"}:
        raise ValueError("sampler must be one of 'corpus_random', 'dataset', or 'dataset_context'")
    return kind


def _build_sampler(
    corpus: Any,
    *,
    sampler: str,
    batch_size: int,
    drop_last: bool,
    seed: int,
    shuffle: bool,
    dataset_index: int | None,
    context_field: str,
    row_indices: Sequence[int] | np.ndarray | None,
) -> Any:
    kind = _normalize_sampler_kind(sampler)
    if kind == "corpus_random":
        return CorpusRandomBatchSampler(
            metadata_index=corpus.metadata_index,
            batch_size=batch_size,
            row_indices=row_indices,
            drop_last=drop_last,
            seed=seed,
        )
    if dataset_index is None:
        raise ValueError(f"dataset_index is required for sampler={kind!r}")
    if kind == "dataset":
        return DatasetBatchSampler(
            metadata_index=corpus.metadata_index,
            dataset_index=dataset_index,
            batch_size=batch_size,
            row_indices=row_indices,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
        )
    return DatasetContextBatchSampler(
        metadata_index=corpus.metadata_index,
        batch_size=batch_size,
        context_field=context_field,
        dataset_index=dataset_index,
        row_indices=row_indices,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
    )


def _build_dataloader_kwargs(
    *,
    num_workers: int,
    multiprocessing_context: str | None,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    backend: str,
) -> dict[str, Any]:
    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers must be >= 0")
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers) if workers > 0 else False,
    }
    if workers == 0:
        if multiprocessing_context is not None:
            raise ValueError("multiprocessing_context requires num_workers > 0")
        if persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")
        if prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0")
        return kwargs

    kwargs["multiprocessing_context"] = (
        multiprocessing_context if multiprocessing_context is not None else ("spawn" if backend == "lance" else None)
    )
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def build_loader(
    corpus: Any,
    *,
    seq_len: int,
    num_workers: int = 0,
    multiprocessing_context: str | None = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    metadata_columns: Sequence[str] | None = None,
    sampler: str = "corpus_random",
    batch_size: int = 128,
    drop_last: bool = False,
    seed: int = 0,
    shuffle: bool = True,
    dataset_index: int | None = None,
    context_field: str = "raw_cell_type",
    sampling_mode: str = "uniform",
    expressed_weight: float = 3.0,
    hvg_weight: float = 3.0,
    hvg_top_k: int | None = None,
    row_indices: Sequence[int] | np.ndarray | None = None,
    device: torch.device | str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield processed sparse training batches from a loaded corpus.

    Expression reads stay expression-only in DataLoader workers. The main
    process attaches ``dataset_index`` and optional metadata from
    ``corpus.metadata_index`` by ``global_row_index`` before sparse processing.
    """
    resolved_seq_len = int(seq_len)
    if resolved_seq_len <= 0:
        raise ValueError("seq_len must be positive")
    resolved_batch_size = int(batch_size)
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")

    resolved_metadata_columns = _normalize_metadata_columns(corpus.metadata_index, metadata_columns)
    dataset_obj = ExpressionBatchDataset(corpus.expression_reader, total_rows=len(corpus.metadata_index))
    batch_sampler = _build_sampler(
        corpus,
        sampler=sampler,
        batch_size=resolved_batch_size,
        drop_last=drop_last,
        seed=seed,
        shuffle=shuffle,
        dataset_index=dataset_index,
        context_field=context_field,
        row_indices=row_indices,
    )
    data_loader = DataLoader(
        dataset_obj,
        batch_sampler=batch_sampler,
        collate_fn=collate_expression_batch,
        **_build_dataloader_kwargs(
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            backend=corpus.backend,
        ),
    )
    pipeline = GPUSparsePipeline(corpus.feature_registry, seq_len=resolved_seq_len)

    def _iterator() -> Iterator[dict[str, Any]]:
        for expression_batch in data_loader:
            raw_batch, gathered = _attach_required_batch_metadata(
                corpus,
                expression_batch,
                resolved_metadata_columns,
            )
            processed = pipeline.process_batch(
                raw_batch,
                device=device,
                sampling_mode=sampling_mode,
                expressed_weight=expressed_weight,
                hvg_weight=hvg_weight,
                hvg_top_k=hvg_top_k,
            )
            yield _attach_requested_metadata(processed, gathered, resolved_metadata_columns)

    return _iterator()
