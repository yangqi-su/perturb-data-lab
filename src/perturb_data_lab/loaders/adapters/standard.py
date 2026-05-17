"""Standard PyTorch adapter for corpus expression batches."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from ..expression import ExpressionBatch
from ..gene_token_mapper import GeneTokenMapper
from ..index import _normalize_candidate_row_indices
from ..sparse_batch import SparseBatchProcessor

__all__ = [
    "ExpressionBatch",
    "CorpusRandomBatchSampler",
    "ContextBatchSampler",
    "ExpressionBatchDataset",
    "build_loader",
    "collate_expression_batch",
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


def _attach_pipeline_metadata(
    corpus: Any,
    raw_batch: dict[str, Any],
    metadata_columns: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    global_row_index = raw_batch["global_row_index"]
    if isinstance(global_row_index, torch.Tensor):
        indices = global_row_index.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        indices = np.asarray(global_row_index, dtype=np.int64)

    columns = ["dataset_index"]
    if "size_factor" in corpus.metadata_index.df.columns:
        columns.append("size_factor")
    columns.extend(column for column in metadata_columns if column not in columns)

    gathered = corpus.metadata_index.take(indices, columns)
    resolved = dict(raw_batch)
    resolved["dataset_index"] = np.asarray(gathered["dataset_index"], dtype=np.int32)
    if "size_factor" in gathered:
        size_factor = np.asarray(gathered["size_factor"], dtype=np.float32)
        if size_factor.size == 0 or np.isnan(size_factor).all():
            size_factor = None
    else:
        size_factor = None
    if size_factor is not None:
        resolved["size_factor"] = size_factor
    return resolved, gathered


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


def _normalize_context_columns(context_columns: Sequence[str] | None) -> tuple[str, ...]:
    if context_columns is None:
        raise ValueError("context_columns is required when sampler='context'")
    if isinstance(context_columns, (str, bytes)):
        raise TypeError("context_columns must be a sequence of metadata column names")
    columns = tuple(context_columns)
    if not columns:
        raise ValueError("context_columns must contain at least one metadata column")
    for column in columns:
        if not isinstance(column, str) or not column:
            raise ValueError("context_columns must contain non-empty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("context_columns must be unique")
    return columns


class ContextBatchSampler:
    """Exhaust pure-context batches grouped by metadata columns."""

    def __init__(
        self,
        *,
        metadata_index: "MetadataIndex",
        context_columns: Sequence[str] | None,
        batch_size: int,
        row_indices: Sequence[int] | np.ndarray | None = None,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        columns = _normalize_context_columns(context_columns)
        missing = [column for column in columns if column not in metadata_index.df.columns]
        if missing:
            raise ValueError(f"context_columns not found in metadata_index: {missing}")

        frame = metadata_index.df
        candidate_row_indices = _normalize_candidate_row_indices(metadata_index, row_indices)
        if candidate_row_indices is not None:
            frame = frame.filter(pl.col("global_row_index").is_in(candidate_row_indices.tolist()))
        grouped = frame.group_by(list(columns), maintain_order=True).agg(
            pl.col("global_row_index").alias("row_indices")
        )
        if grouped.height == 0:
            raise ValueError("no context groups found for the requested rows")

        self._groups: list[np.ndarray] = [
            np.asarray(row["row_indices"], dtype=np.int64)
            for row in grouped.iter_rows(named=True)
        ]
        self.context_columns = columns
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(rows) // self.batch_size for rows in self._groups)
        return sum((len(rows) + self.batch_size - 1) // self.batch_size for rows in self._groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = np.arange(len(self._groups), dtype=np.int64)
        if self.shuffle:
            rng.shuffle(group_order)

        for group_pos in group_order.tolist():
            rows = self._groups[group_pos].copy()
            if self.shuffle:
                rng.shuffle(rows)
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch.astype(np.int64, copy=False).tolist()


class ExpressionBatchDataset:
    """Expression-only PyTorch dataset wrapper around an expression reader."""

    def __init__(self, expression_reader: Any, *, total_rows: int):
        self._reader = expression_reader
        self._total_rows = int(total_rows)

    def __len__(self) -> int:
        return self._total_rows

    def __getitems__(self, indices: Sequence[int]) -> list[ExpressionBatch]:
        return [self._reader.read_expression_flat(list(indices))]


def collate_expression_batch(items: list[ExpressionBatch]) -> dict[str, Any]:
    """Unwrap one expression-only batch and convert arrays to torch tensors."""
    if not items:
        raise ValueError("collate_expression_batch received empty list")
    if len(items) != 1:
        raise ValueError(
            f"collate_expression_batch expected a single pre-batched item, got {len(items)}"
        )
    batch = items[0]
    return {
        "batch_size": batch.batch_size,
        "global_row_index": torch.as_tensor(batch.global_row_index, dtype=torch.long),
        "row_offsets": torch.as_tensor(batch.row_offsets, dtype=torch.long),
        "expressed_gene_indices": torch.as_tensor(batch.expressed_gene_indices, dtype=torch.long),
        "expression_counts": torch.as_tensor(batch.expression_counts, dtype=torch.float32),
    }


def _build_sampler(
    corpus: Any,
    *,
    sampler: str,
    batch_size: int,
    drop_last: bool,
    seed: int,
    shuffle: bool,
    context_columns: Sequence[str] | None,
    row_indices: Sequence[int] | np.ndarray | None,
) -> Any:
    if not isinstance(sampler, str):
        raise TypeError("sampler must be a string")
    kind = sampler.strip().lower().replace("-", "_")
    if kind not in {"corpus_random", "context"}:
        raise ValueError("sampler must be one of 'corpus_random' or 'context'")
    if kind == "corpus_random":
        return CorpusRandomBatchSampler(
            metadata_index=corpus.metadata_index,
            batch_size=batch_size,
            row_indices=row_indices,
            drop_last=drop_last,
            seed=seed,
        )
    return ContextBatchSampler(
        metadata_index=corpus.metadata_index,
        batch_size=batch_size,
        context_columns=context_columns,
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
    context_columns: Sequence[str] | None = None,
    sampling_mode: str = "uniform",
    expressed_weight: float = 3.0,
    hvg_weight: float = 3.0,
    hvg_top_k: int | None = None,
    row_indices: Sequence[int] | np.ndarray | None = None,
    gene_token_mapper: GeneTokenMapper | None = None,
    missing_token_policy: str = "exclude",
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
    assert missing_token_policy in {
        "exclude",
        "pad",
    }, "missing_token_policy must be one of 'exclude' or 'pad'"
    if gene_token_mapper is not None:
        gene_token_mapper.check_feature_registry(corpus.feature_registry)

    resolved_metadata_columns = _normalize_metadata_columns(corpus.metadata_index, metadata_columns)
    dataset_obj = ExpressionBatchDataset(corpus.expression_reader, total_rows=len(corpus.metadata_index))
    batch_sampler = _build_sampler(
        corpus,
        sampler=sampler,
        batch_size=resolved_batch_size,
        drop_last=drop_last,
        seed=seed,
        shuffle=shuffle,
        context_columns=context_columns,
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
    pipeline = SparseBatchProcessor(
        corpus.feature_registry,
        seq_len=resolved_seq_len,
        sampling_gene_mask=(
            gene_token_mapper.tokenizable_by_global_id
            if gene_token_mapper is not None and missing_token_policy == "exclude"
            and not bool(gene_token_mapper.tokenizable_by_global_id.all())
            else None
        ),
    )

    def _iterator() -> Iterator[dict[str, Any]]:
        for expression_batch in data_loader:
            raw_batch, gathered = _attach_pipeline_metadata(
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
            if gene_token_mapper is not None:
                gene_ids, gene_token_mask = gene_token_mapper.encode_global_ids(
                    processed["sampled_gene_ids"],
                    processed["valid_mask"],
                )
                processed["gene_ids"] = gene_ids
                processed["gene_token_mask"] = gene_token_mask
            if resolved_metadata_columns:
                processed["meta_columns"] = {
                    column: gathered[column]
                    for column in resolved_metadata_columns
                }
            yield processed

    return _iterator()
