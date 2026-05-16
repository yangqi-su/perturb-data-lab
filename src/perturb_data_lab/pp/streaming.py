"""Shared low-memory per-dataset streaming helpers for ``perturb_data_lab.pp``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy import sparse

from ..loaders.corpus_loader import Corpus
from ..loaders.loaders import ExpressionBatch


@dataclass(frozen=True)
class PpFeatureContext:
    """Dataset-local feature mapping context for streamed pp batches."""

    dataset_id: str
    dataset_index: int
    global_start: int
    global_end: int
    local_to_global: np.ndarray
    local_feature_ids: tuple[str, ...]
    global_feature_ids: tuple[str, ...]

    @property
    def n_features(self) -> int:
        """Number of dataset-local features addressable in this stream."""
        return int(self.local_to_global.shape[0])


@dataclass(frozen=True)
class PpBatch:
    """One sparse per-dataset batch emitted by ``iter_dataset_batches``."""

    dataset_id: str
    dataset_index: int
    global_row_index: np.ndarray
    local_row_index: np.ndarray
    size_factor: np.ndarray | None
    expression: sparse.csr_matrix
    feature_context: PpFeatureContext

    @property
    def batch_size(self) -> int:
        """Number of rows in the streamed batch."""
        return int(self.global_row_index.shape[0])


@dataclass(frozen=True)
class _SparseFeatureSummary:
    """Per-feature sparse batch sums shared across pp streaming reducers."""

    sum: np.ndarray
    sum_sq: np.ndarray
    n_nonzero: np.ndarray


def iter_dataset_batches(
    corpus: Corpus,
    *,
    dataset_id: str | None = None,
    batch_size: int = 1024,
    scope: str = "dataset",
) -> Iterator[PpBatch]:
    """Yield dataset-pure sparse batches without materializing the full matrix."""
    if scope != "dataset":
        raise NotImplementedError(
            "Phase 6 pp streaming currently supports only scope='dataset'"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    contexts = _resolve_feature_contexts(corpus, dataset_id=dataset_id)
    for context in contexts:
        for start in range(context.global_start, context.global_end, batch_size):
            stop = min(start + batch_size, context.global_end)
            batch_indices = np.arange(start, stop, dtype=np.int64)
            expr = corpus.expression_reader.read_expression_flat(batch_indices.tolist())
            metadata = corpus.take_metadata(
                batch_indices,
                columns=("local_row_index", "size_factor"),
            )
            local_row_index = np.asarray(metadata["local_row_index"], dtype=np.int64)
            raw_size_factor = metadata.get("size_factor")
            size_factor = (
                None
                if raw_size_factor is None
                else np.asarray(raw_size_factor, dtype=np.float32)
            )
            yield PpBatch(
                dataset_id=context.dataset_id,
                dataset_index=context.dataset_index,
                global_row_index=np.asarray(expr.global_row_index, dtype=np.int64),
                local_row_index=local_row_index,
                size_factor=size_factor,
                expression=_expression_batch_to_csr(expr, n_features=context.n_features),
                feature_context=context,
            )


def log1p_size_factor_batch(
    batch: PpBatch,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float32,
) -> sparse.csr_matrix:
    """Return ``log1p(count / size_factor)`` for the nonzero entries in a batch."""
    if batch.size_factor is None:
        raise ValueError("log1p_size_factor_batch requires batch.size_factor")

    size_factor = np.asarray(batch.size_factor, dtype=np.float64)
    if size_factor.shape != (batch.batch_size,):
        raise ValueError("size_factor must align with the streamed batch rows")
    if np.any(~np.isfinite(size_factor)) or np.any(size_factor <= 0):
        raise ValueError("size_factor values must be finite and positive")

    row_nonzero = np.diff(batch.expression.indptr)
    repeated_size_factor = np.repeat(size_factor, row_nonzero)
    normalized_data = np.log1p(
        np.asarray(batch.expression.data, dtype=np.float64) / repeated_size_factor
    ).astype(dtype, copy=False)
    return sparse.csr_matrix(
        (
            normalized_data,
            batch.expression.indices.copy(),
            batch.expression.indptr.copy(),
        ),
        shape=batch.expression.shape,
    )


def _summarize_sparse_features(
    values: sparse.csr_matrix,
    *,
    n_features: int,
) -> _SparseFeatureSummary:
    """Return per-feature sums, squared sums, and nonzero counts for a CSR batch."""
    data = np.asarray(values.data, dtype=np.float64)
    if not data.size:
        return _SparseFeatureSummary(
            sum=np.zeros(n_features, dtype=np.float64),
            sum_sq=np.zeros(n_features, dtype=np.float64),
            n_nonzero=np.zeros(n_features, dtype=np.int64),
        )

    indices = np.asarray(values.indices, dtype=np.int64)
    return _SparseFeatureSummary(
        sum=np.bincount(indices, weights=data, minlength=n_features),
        sum_sq=np.bincount(
            indices,
            weights=np.square(data),
            minlength=n_features,
        ),
        n_nonzero=np.bincount(indices[data != 0.0], minlength=n_features).astype(
            np.int64,
            copy=False,
        ),
    )


def _resolve_feature_contexts(
    corpus: Corpus,
    *,
    dataset_id: str | None,
) -> list[PpFeatureContext]:
    entries = sorted(corpus.dataset_entries, key=lambda entry: entry.global_start)
    dataset_ids = tuple(entry.dataset_id for entry in entries)
    starts = np.asarray([entry.global_start for entry in entries], dtype=np.int64)
    stops = np.asarray([entry.global_end for entry in entries], dtype=np.int64)
    selected_dataset_ids = dataset_ids if dataset_id is None else (dataset_id,)

    contexts: list[PpFeatureContext] = []
    global_feature_ids = corpus.feature_registry.global_feature_ids
    dense_map = np.asarray(corpus.feature_registry.local_to_global_map, dtype=np.int32)
    positions_by_dataset_id = {ds_id: pos for pos, ds_id in enumerate(dataset_ids)}
    for selected_id in selected_dataset_ids:
        if selected_id not in positions_by_dataset_id:
            raise KeyError(f"Unknown dataset_id for pp streaming: {selected_id!r}")
        position = positions_by_dataset_id[selected_id]
        dataset_index = int(corpus.dataset_index_by_id[selected_id])
        local_to_global_full = dense_map[dataset_index]
        valid = local_to_global_full >= 0
        local_to_global = local_to_global_full[valid].copy()
        local_feature_ids = tuple(global_feature_ids[int(global_id)] for global_id in local_to_global)
        contexts.append(
            PpFeatureContext(
                dataset_id=str(selected_id),
                dataset_index=dataset_index,
                global_start=int(starts[position]),
                global_end=int(stops[position]),
                local_to_global=local_to_global,
                local_feature_ids=local_feature_ids,
                global_feature_ids=global_feature_ids,
            )
        )
    return contexts


def _expression_batch_to_csr(
    batch: ExpressionBatch,
    *,
    n_features: int,
) -> sparse.csr_matrix:
    """Convert a flat ``ExpressionBatch`` into a dataset-local CSR matrix."""
    indices = np.asarray(batch.expressed_gene_indices, dtype=np.int32)
    if indices.size and (np.any(indices < 0) or np.any(indices >= n_features)):
        raise IndexError("Expression batch contains out-of-range dataset-local feature indices")
    return sparse.csr_matrix(
        (
            np.asarray(batch.expression_counts, dtype=np.int32),
            indices,
            np.asarray(batch.row_offsets, dtype=np.int64),
        ),
        shape=(batch.batch_size, n_features),
    )
