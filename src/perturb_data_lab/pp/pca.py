"""Streamed per-dataset PCA-like dimensionality reduction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from ..loaders.corpus_loader import Corpus
from ..loaders.index import _normalize_global_row_indices
from .artifacts import prepare_pp_output, write_pp_provenance
from .streaming import (
    PpBatch,
    PpFeatureContext,
    _expression_batch_to_csr,
    _resolve_feature_contexts,
    iter_dataset_batches,
    log1p_size_factor_batch,
)

DEFAULT_PCA_ARTIFACT_NAME = "pca"
_LOGNORM_FORMULA = "log1p(count / size_factor)"
_INCREMENTAL_PCA_METHOD_SEMANTICS = "centered_incremental_pca_on_lognorm_expression"
_TRUNCATED_SVD_UNSUPPORTED_MESSAGE = (
    "method='truncated_svd' is unsupported in slim main. Use method='incremental_pca' "
    "or switch to experimental/all-backends-pre-slim-20260514 for the legacy truncated SVD route."
)


@dataclass(frozen=True)
class PpPcaResult:
    """Collected outputs for one streamed PCA run."""

    embeddings: pl.DataFrame
    components: pl.DataFrame
    component_stats: pl.DataFrame
    selected_features: pl.DataFrame


@dataclass(frozen=True)
class _SelectedFeatureSet:
    local_feature_positions: np.ndarray
    global_feature_ids: np.ndarray
    gene_ids: tuple[str, ...]
    selection_source: str
    hvg_rank: np.ndarray | None

    @property
    def n_features(self) -> int:
        return int(self.local_feature_positions.shape[0])


@dataclass(frozen=True)
class _DatasetPcaPlan:
    context: PpFeatureContext
    selected_features: _SelectedFeatureSet
    fit_row_indices: np.ndarray
    transform_row_indices: np.ndarray
    fit_chunks: tuple[np.ndarray, ...]
    estimated_dense_batch_bytes: int | None

    @property
    def fit_n_obs(self) -> int:
        return int(self.fit_row_indices.shape[0])

    @property
    def transform_n_obs(self) -> int:
        return int(self.transform_row_indices.shape[0])


def run_pca(
    corpus: Corpus,
    *,
    dataset_id: str | None = None,
    batch_size: int = 1024,
    n_components: int = 50,
    method: str = "incremental_pca",
    fit_row_indices: Sequence[int] | np.ndarray | None = None,
    transform_row_indices: Sequence[int] | np.ndarray | None = None,
    hvg_frame: pl.DataFrame | None = None,
    max_dense_batch_bytes: int | None = None,
    output_dir: str | Path | None = None,
    artifact_name: str = DEFAULT_PCA_ARTIFACT_NAME,
    overwrite: bool = False,
    random_seed: int = 0,
) -> PpPcaResult:
    """Run streamed per-dataset dimensionality reduction on lognorm expression.

    Slim main keeps IncrementalPCA as the streamed PCA route. The public name
    stays ``run_pca`` because the outputs remain PCA-like, while provenance and
    component statistics record the exact centered IncrementalPCA semantics.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if max_dense_batch_bytes is not None and int(max_dense_batch_bytes) <= 0:
        raise ValueError("max_dense_batch_bytes must be positive when provided")

    if method == "truncated_svd":
        raise NotImplementedError(_TRUNCATED_SVD_UNSUPPORTED_MESSAGE)
    if method != "incremental_pca":
        raise NotImplementedError(
            "run_pca supports only method='incremental_pca' in slim main"
        )
    if dataset_id is None and (
        fit_row_indices is not None or transform_row_indices is not None
    ):
        raise ValueError(
            "dataset_id is required when fit_row_indices or transform_row_indices are provided"
        )

    output_root = None if output_dir is None else Path(output_dir)
    embedding_frames: list[pl.DataFrame] = []
    component_frames: list[pl.DataFrame] = []
    component_stat_frames: list[pl.DataFrame] = []
    selected_feature_frames: list[pl.DataFrame] = []

    plans = [
        _prepare_dataset_pca_plan(
            context,
            hvg_frame=hvg_frame,
            batch_size=batch_size,
            n_components=n_components,
            fit_row_indices=fit_row_indices,
            transform_row_indices=transform_row_indices,
            max_dense_batch_bytes=max_dense_batch_bytes,
        )
        for context in _resolve_feature_contexts(corpus, dataset_id=dataset_id)
    ]

    incremental_pca_cls = _load_incremental_pca_class()

    for plan in plans:
        selected_frame = _build_selected_features_frame(plan.context, plan.selected_features)
        embeddings_frame, components_frame, component_stats_frame = _run_incremental_pca_plan(
            corpus,
            plan=plan,
            incremental_pca_cls=incremental_pca_cls,
            batch_size=batch_size,
            n_components=n_components,
        )

        embedding_frames.append(embeddings_frame)
        component_frames.append(components_frame)
        component_stat_frames.append(component_stats_frame)
        selected_feature_frames.append(selected_frame)

        if output_root is not None:
            _write_dataset_outputs(
                output_root,
                corpus=corpus,
                plan=plan,
                embeddings=embeddings_frame,
                components=components_frame,
                component_stats=component_stats_frame,
                selected_features=selected_frame,
                artifact_name=artifact_name,
                overwrite=overwrite,
                batch_size=batch_size,
                method=method,
                max_dense_batch_bytes=max_dense_batch_bytes,
                random_seed=random_seed,
            )

    return PpPcaResult(
        embeddings=_concat_or_empty(
            embedding_frames,
            _empty_embeddings_frame(n_components),
            sort_by=["dataset_index", "global_row_index"],
        ),
        components=_concat_or_empty(
            component_frames,
            _empty_components_frame(),
            sort_by=["dataset_index", "component_index", "selected_feature_index"],
        ),
        component_stats=_concat_or_empty(
            component_stat_frames,
            _empty_component_stats_frame(),
            sort_by=["dataset_index", "component_index"],
        ),
        selected_features=_concat_or_empty(
            selected_feature_frames,
            _empty_selected_features_frame(),
            sort_by=["dataset_index", "selected_feature_index"],
        ),
    )


def _prepare_dataset_pca_plan(
    context: PpFeatureContext,
    *,
    hvg_frame: pl.DataFrame | None,
    batch_size: int,
    n_components: int,
    fit_row_indices: Sequence[int] | np.ndarray | None,
    transform_row_indices: Sequence[int] | np.ndarray | None,
    max_dense_batch_bytes: int | None,
) -> _DatasetPcaPlan:
    selected_features = _resolve_selected_features(context, hvg_frame=hvg_frame)

    fit_rows = _normalize_context_row_indices(context, fit_row_indices)
    transform_rows = _normalize_context_row_indices(context, transform_row_indices)
    if fit_rows.size < 2:
        raise ValueError(
            f"dataset {context.dataset_id!r} needs at least two fit rows for incremental PCA"
        )
    min_dim = min(int(fit_rows.size), selected_features.n_features)
    if n_components > min_dim:
        raise ValueError(
            f"n_components={n_components} must be <= min(n_fit_rows, n_selected_features)={min_dim}"
        )

    fit_chunks = _build_incremental_fit_chunks(
        fit_rows,
        batch_size=batch_size,
        min_chunk_rows=n_components,
    )
    transform_batch_rows = min(batch_size, int(transform_rows.size))
    max_chunk_rows = max(
        max(chunk.shape[0] for chunk in fit_chunks),
        transform_batch_rows,
    )
    estimated_dense_batch_bytes = _estimate_dense_batch_bytes(
        n_rows=max_chunk_rows,
        n_features=selected_features.n_features,
    )
    if (
        max_dense_batch_bytes is not None
        and estimated_dense_batch_bytes > int(max_dense_batch_bytes)
    ):
        raise MemoryError(
            "Estimated dense IncrementalPCA batch for dataset "
            f"{context.dataset_id!r} requires {estimated_dense_batch_bytes} bytes "
            f"({max_chunk_rows} rows x {selected_features.n_features} features x 8 bytes), "
            f"which exceeds max_dense_batch_bytes={int(max_dense_batch_bytes)}."
        )
    return _DatasetPcaPlan(
        context=context,
        selected_features=selected_features,
        fit_row_indices=fit_rows,
        transform_row_indices=transform_rows,
        fit_chunks=fit_chunks,
        estimated_dense_batch_bytes=estimated_dense_batch_bytes,
    )


def _concat_or_empty(
    frames: list[pl.DataFrame],
    empty_frame: pl.DataFrame,
    *,
    sort_by: list[str],
) -> pl.DataFrame:
    if not frames:
        return empty_frame
    return pl.concat(frames, rechunk=True).sort(sort_by)


def _resolve_selected_features(
    context: PpFeatureContext,
    *,
    hvg_frame: pl.DataFrame | None,
) -> _SelectedFeatureSet:
    if hvg_frame is None:
        return _SelectedFeatureSet(
            local_feature_positions=np.arange(context.n_features, dtype=np.int32),
            global_feature_ids=np.asarray(context.local_to_global, dtype=np.int64),
            gene_ids=tuple(str(gene_id) for gene_id in context.local_feature_ids),
            selection_source="all_features",
            hvg_rank=None,
        )

    required_columns = {"dataset_id", "global_feature_id"}
    missing_columns = required_columns.difference(hvg_frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"hvg_frame is missing required columns: {missing}")

    selection_frame = hvg_frame.filter(pl.col("dataset_id") == context.dataset_id)
    if "is_hvg" in selection_frame.columns:
        selection_frame = selection_frame.filter(pl.col("is_hvg"))
    if "hvg_rank" in selection_frame.columns:
        selection_frame = selection_frame.sort(["hvg_rank", "global_feature_id"])
    else:
        selection_frame = selection_frame.sort("global_feature_id")
    selection_frame = selection_frame.unique(
        subset=["global_feature_id"],
        keep="first",
        maintain_order=True,
    )

    available_positions = {
        int(global_id): position
        for position, global_id in enumerate(np.asarray(context.local_to_global, dtype=np.int64))
    }

    local_positions: list[int] = []
    global_feature_ids: list[int] = []
    gene_ids: list[str] = []
    hvg_ranks: list[int] = []
    has_hvg_rank = "hvg_rank" in selection_frame.columns
    for row in selection_frame.iter_rows(named=True):
        global_feature_id = int(row["global_feature_id"])
        position = available_positions.get(global_feature_id)
        if position is None:
            continue
        local_positions.append(position)
        global_feature_ids.append(global_feature_id)
        gene_ids.append(str(context.local_feature_ids[position]))
        if has_hvg_rank:
            hvg_ranks.append(int(row["hvg_rank"]))

    if not local_positions:
        raise ValueError(
            f"No selected features remained for dataset {context.dataset_id!r} after HVG filtering"
        )

    return _SelectedFeatureSet(
        local_feature_positions=np.asarray(local_positions, dtype=np.int32),
        global_feature_ids=np.asarray(global_feature_ids, dtype=np.int64),
        gene_ids=tuple(gene_ids),
        selection_source="hvg_is_hvg",
        hvg_rank=np.asarray(hvg_ranks, dtype=np.int32) if has_hvg_rank else None,
    )


def _normalize_context_row_indices(
    context: PpFeatureContext,
    row_indices: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    normalized = _normalize_global_row_indices(
        row_indices,
        start=context.global_start,
        end=context.global_end,
        field_name="row_indices",
        none_policy="range",
    )
    assert normalized is not None
    return normalized


def _build_incremental_fit_chunks(
    row_indices: np.ndarray,
    *,
    batch_size: int,
    min_chunk_rows: int,
) -> tuple[np.ndarray, ...]:
    chunk_size = max(batch_size, min_chunk_rows)
    chunks = [
        row_indices[start : start + chunk_size]
        for start in range(0, row_indices.size, chunk_size)
    ]
    if len(chunks) > 1 and chunks[-1].shape[0] < min_chunk_rows:
        chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
        chunks.pop()
    return tuple(chunk.copy() for chunk in chunks)


def _estimate_dense_batch_bytes(*, n_rows: int, n_features: int) -> int:
    return int(n_rows) * int(n_features) * np.dtype(np.float64).itemsize


def _iter_context_batches(
    corpus: Corpus,
    *,
    context: PpFeatureContext,
    batch_size: int,
    row_indices: np.ndarray | None,
) -> Iterator[PpBatch]:
    if row_indices is None:
        yield from iter_dataset_batches(corpus, dataset_id=context.dataset_id, batch_size=batch_size)
        return
    for start in range(0, row_indices.size, batch_size):
        chunk = row_indices[start : start + batch_size]
        yield _read_pp_batch_for_rows(corpus, context=context, row_indices=chunk)


def _read_pp_batch_for_rows(
    corpus: Corpus,
    *,
    context: PpFeatureContext,
    row_indices: np.ndarray,
) -> PpBatch:
    expression_batch = corpus.expression_reader.read_expression_flat(row_indices.tolist())
    metadata = corpus.take_metadata(row_indices, columns=("local_row_index", "size_factor"))
    raw_size_factor = metadata.get("size_factor")
    size_factor = (
        None if raw_size_factor is None else np.asarray(raw_size_factor, dtype=np.float32)
    )
    return PpBatch(
        dataset_id=context.dataset_id,
        dataset_index=context.dataset_index,
        global_row_index=np.asarray(expression_batch.global_row_index, dtype=np.int64),
        local_row_index=np.asarray(metadata["local_row_index"], dtype=np.int64),
        size_factor=size_factor,
        expression=_expression_batch_to_csr(expression_batch, n_features=context.n_features),
        feature_context=context,
    )


def _select_lognorm_features(
    batch: PpBatch,
    local_feature_positions: np.ndarray,
) -> csr_matrix:
    lognorm = log1p_size_factor_batch(batch, dtype=np.float64)
    return lognorm[:, local_feature_positions]


def _canonicalize_component_signs(
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    canonical = np.asarray(components, dtype=np.float64).copy()
    signs = np.ones(canonical.shape[0], dtype=np.float64)
    for component_index in range(canonical.shape[0]):
        component = canonical[component_index]
        if component.size == 0:
            continue
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            canonical[component_index] *= -1.0
            signs[component_index] = -1.0
    return canonical, signs


def _run_incremental_pca_plan(
    corpus: Corpus,
    *,
    plan: _DatasetPcaPlan,
    incremental_pca_cls: type[Any],
    batch_size: int,
    n_components: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    model = incremental_pca_cls(
        n_components=n_components,
        batch_size=max(batch_size, n_components),
    )
    for fit_chunk in plan.fit_chunks:
        dense_batch = _materialize_selected_lognorm_dense_batch(
            corpus,
            context=plan.context,
            selected_features=plan.selected_features,
            row_indices=fit_chunk,
        )
        model.partial_fit(dense_batch)

    components, component_signs = _canonicalize_component_signs(model.components_)
    embeddings_frame = _transform_incremental_embeddings(
        corpus,
        plan=plan,
        model=model,
        component_signs=component_signs,
        batch_size=batch_size,
    )
    components_frame = _build_components_frame(
        plan.context,
        plan.selected_features,
        components=components,
    )
    component_stats_frame = _build_component_stats_frame(
        plan.context,
        singular_values=np.asarray(model.singular_values_, dtype=np.float64),
        explained_variance=np.asarray(model.explained_variance_, dtype=np.float64),
        explained_variance_ratio=np.asarray(model.explained_variance_ratio_, dtype=np.float64),
        n_selected_features=plan.selected_features.n_features,
        n_obs=plan.fit_n_obs,
        method="incremental_pca",
        is_centered=True,
        method_semantics=_INCREMENTAL_PCA_METHOD_SEMANTICS,
    )
    return embeddings_frame, components_frame, component_stats_frame


def _materialize_selected_lognorm_dense_batch(
    corpus: Corpus,
    *,
    context: PpFeatureContext,
    selected_features: _SelectedFeatureSet,
    row_indices: np.ndarray,
) -> np.ndarray:
    batch = _read_pp_batch_for_rows(corpus, context=context, row_indices=row_indices)
    selected = _select_lognorm_features(batch, selected_features.local_feature_positions)
    return np.asarray(selected.toarray(), dtype=np.float64)


def _transform_incremental_embeddings(
    corpus: Corpus,
    *,
    plan: _DatasetPcaPlan,
    model: Any,
    component_signs: np.ndarray,
    batch_size: int,
) -> pl.DataFrame:
    global_rows: list[np.ndarray] = []
    local_rows: list[np.ndarray] = []
    embedding_chunks: list[np.ndarray] = []

    for batch in _iter_context_batches(
        corpus,
        context=plan.context,
        batch_size=batch_size,
        row_indices=plan.transform_row_indices,
    ):
        selected = _select_lognorm_features(batch, plan.selected_features.local_feature_positions)
        transformed = np.asarray(model.transform(selected.toarray()), dtype=np.float64)
        transformed *= component_signs[None, :]
        global_rows.append(np.asarray(batch.global_row_index, dtype=np.int64))
        local_rows.append(np.asarray(batch.local_row_index, dtype=np.int64))
        embedding_chunks.append(transformed)

    if not embedding_chunks:
        return _empty_embeddings_frame(model.n_components_)

    embeddings = np.vstack(embedding_chunks)
    payload: dict[str, object] = {
        "dataset_id": [plan.context.dataset_id] * embeddings.shape[0],
        "dataset_index": np.full(embeddings.shape[0], plan.context.dataset_index, dtype=np.int32),
        "global_row_index": np.concatenate(global_rows),
        "local_row_index": np.concatenate(local_rows),
    }
    for component_index in range(embeddings.shape[1]):
        payload[f"component_{component_index + 1}"] = embeddings[:, component_index]
    return pl.DataFrame(payload).sort("global_row_index")


def _build_selected_features_frame(
    context: PpFeatureContext,
    selected_features: _SelectedFeatureSet,
) -> pl.DataFrame:
    payload: dict[str, object] = {
        "dataset_id": [context.dataset_id] * selected_features.n_features,
        "dataset_index": np.full(selected_features.n_features, context.dataset_index, dtype=np.int32),
        "selected_feature_index": np.arange(selected_features.n_features, dtype=np.int32),
        "local_feature_index": np.asarray(selected_features.local_feature_positions, dtype=np.int32),
        "gene_id": list(selected_features.gene_ids),
        "global_feature_id": np.asarray(selected_features.global_feature_ids, dtype=np.int64),
        "selection_source": [selected_features.selection_source] * selected_features.n_features,
    }
    if selected_features.hvg_rank is not None:
        payload["hvg_rank"] = np.asarray(selected_features.hvg_rank, dtype=np.int32)
    return pl.DataFrame(payload).sort("selected_feature_index")


def _build_components_frame(
    context: PpFeatureContext,
    selected_features: _SelectedFeatureSet,
    *,
    components: np.ndarray,
) -> pl.DataFrame:
    n_components = int(components.shape[0])
    n_features = selected_features.n_features
    payload: dict[str, object] = {
        "dataset_id": [context.dataset_id] * (n_components * n_features),
        "dataset_index": np.full(n_components * n_features, context.dataset_index, dtype=np.int32),
        "component_index": np.repeat(np.arange(1, n_components + 1, dtype=np.int32), n_features),
        "selected_feature_index": np.tile(np.arange(n_features, dtype=np.int32), n_components),
        "gene_id": list(selected_features.gene_ids) * n_components,
        "global_feature_id": np.tile(
            np.asarray(selected_features.global_feature_ids, dtype=np.int64),
            n_components,
        ),
        "loading": np.asarray(components, dtype=np.float64).reshape(-1),
    }
    if selected_features.hvg_rank is not None:
        payload["hvg_rank"] = np.tile(
            np.asarray(selected_features.hvg_rank, dtype=np.int32),
            n_components,
        )
    return pl.DataFrame(payload).sort(["component_index", "selected_feature_index"])


def _build_component_stats_frame(
    context: PpFeatureContext,
    *,
    singular_values: np.ndarray,
    explained_variance: np.ndarray,
    explained_variance_ratio: np.ndarray,
    n_selected_features: int,
    n_obs: int,
    method: str,
    is_centered: bool,
    method_semantics: str,
) -> pl.DataFrame:
    resolved_singular_values = np.asarray(singular_values, dtype=np.float64)
    resolved_explained_variance = np.asarray(explained_variance, dtype=np.float64)
    resolved_explained_variance_ratio = np.asarray(explained_variance_ratio, dtype=np.float64)
    payload = {
        "dataset_id": [context.dataset_id] * len(resolved_singular_values),
        "dataset_index": np.full(len(resolved_singular_values), context.dataset_index, dtype=np.int32),
        "component_index": np.arange(1, len(resolved_singular_values) + 1, dtype=np.int32),
        "singular_value": resolved_singular_values,
        "explained_variance": resolved_explained_variance,
        "explained_variance_ratio": resolved_explained_variance_ratio,
        "cumulative_explained_variance_ratio": np.cumsum(resolved_explained_variance_ratio),
        "n_obs": np.full(len(resolved_singular_values), n_obs, dtype=np.int64),
        "n_features_selected": np.full(len(resolved_singular_values), n_selected_features, dtype=np.int32),
        "method": [method] * len(resolved_singular_values),
        "is_centered": [is_centered] * len(resolved_singular_values),
        "method_semantics": [method_semantics] * len(resolved_singular_values),
    }
    return pl.DataFrame(payload).sort("component_index")


def _write_dataset_outputs(
    output_root: Path,
    *,
    corpus: Corpus,
    plan: _DatasetPcaPlan,
    embeddings: pl.DataFrame,
    components: pl.DataFrame,
    component_stats: pl.DataFrame,
    selected_features: pl.DataFrame,
    artifact_name: str,
    overwrite: bool,
    batch_size: int,
    method: str,
    max_dense_batch_bytes: int | None,
    random_seed: int,
) -> None:
    embeddings_spec = prepare_pp_output(
        output_root,
        dataset_id=plan.context.dataset_id,
        artifact_name=f"{artifact_name}-embeddings",
        suffix="parquet",
    )
    components_spec = prepare_pp_output(
        output_root,
        dataset_id=plan.context.dataset_id,
        artifact_name=f"{artifact_name}-components",
        suffix="parquet",
    )
    stats_spec = prepare_pp_output(
        output_root,
        dataset_id=plan.context.dataset_id,
        artifact_name=f"{artifact_name}-component-stats",
        suffix="parquet",
    )
    selected_spec = prepare_pp_output(
        output_root,
        dataset_id=plan.context.dataset_id,
        artifact_name=f"{artifact_name}-selected-features",
        suffix="parquet",
    )

    for spec in (embeddings_spec, components_spec, stats_spec, selected_spec):
        if spec.artifact_path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {spec.artifact_path}")

    embeddings.write_parquet(embeddings_spec.artifact_path)
    components.write_parquet(components_spec.artifact_path)
    component_stats.write_parquet(stats_spec.artifact_path)
    selected_features.write_parquet(selected_spec.artifact_path)
    write_pp_provenance(
        stats_spec,
        corpus=corpus,
        operation="run_pca",
        parameters={
            "batch_size": batch_size,
            "dataset_scope": "dataset",
            "method": method,
            "method_semantics": _method_semantics(method),
            "normalization": _LOGNORM_FORMULA,
            "n_components": int(component_stats.height),
            "fit_row_count": int(plan.fit_n_obs),
            "transform_row_count": int(plan.transform_n_obs),
            "max_dense_batch_bytes": (
                None if max_dense_batch_bytes is None else int(max_dense_batch_bytes)
            ),
            "random_seed": int(random_seed),
        },
        extra={
            "artifact_paths": {
                "embeddings": str(embeddings_spec.artifact_path),
                "components": str(components_spec.artifact_path),
                "component_stats": str(stats_spec.artifact_path),
                "selected_features": str(selected_spec.artifact_path),
            },
            "n_obs": int(embeddings.height),
            "fit_n_obs": int(plan.fit_n_obs),
            "transform_n_obs": int(plan.transform_n_obs),
            "n_features_selected": int(selected_features.height),
            "global_row_start": int(plan.context.global_start),
            "global_row_end": int(plan.context.global_end),
            "estimated_dense_batch_bytes": plan.estimated_dense_batch_bytes,
        },
    )


def _method_semantics(method: str) -> str:
    if method == "incremental_pca":
        return _INCREMENTAL_PCA_METHOD_SEMANTICS
    if method == "truncated_svd":
        raise NotImplementedError(_TRUNCATED_SVD_UNSUPPORTED_MESSAGE)
    raise NotImplementedError(f"Unknown PCA method semantics for {method!r}")


def _load_incremental_pca_class() -> type[Any]:
    try:
        from sklearn.decomposition import IncrementalPCA
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch test
        raise ImportError(
            "method='incremental_pca' requires optional dependency scikit-learn. "
            "Install perturb-data-lab[pca] or add scikit-learn manually."
        ) from exc
    return IncrementalPCA


def _empty_embeddings_frame(n_components: int) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "dataset_id": pl.String,
        "dataset_index": pl.Int32,
        "global_row_index": pl.Int64,
        "local_row_index": pl.Int64,
    }
    for component_index in range(n_components):
        schema[f"component_{component_index + 1}"] = pl.Float64
    return pl.DataFrame(schema=schema)


def _empty_components_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "dataset_id": pl.String,
            "dataset_index": pl.Int32,
            "component_index": pl.Int32,
            "selected_feature_index": pl.Int32,
            "gene_id": pl.String,
            "global_feature_id": pl.Int64,
            "loading": pl.Float64,
            "hvg_rank": pl.Int32,
        }
    )


def _empty_component_stats_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "dataset_id": pl.String,
            "dataset_index": pl.Int32,
            "component_index": pl.Int32,
            "singular_value": pl.Float64,
            "explained_variance": pl.Float64,
            "explained_variance_ratio": pl.Float64,
            "cumulative_explained_variance_ratio": pl.Float64,
            "n_obs": pl.Int64,
            "n_features_selected": pl.Int32,
            "method": pl.String,
            "is_centered": pl.Boolean,
            "method_semantics": pl.String,
        }
    )


def _empty_selected_features_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "dataset_id": pl.String,
            "dataset_index": pl.Int32,
            "selected_feature_index": pl.Int32,
            "local_feature_index": pl.Int32,
            "gene_id": pl.String,
            "global_feature_id": pl.Int64,
            "selection_source": pl.String,
            "hvg_rank": pl.Int32,
        }
    )
