"""Streamed per-dataset PCA-like dimensionality reduction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, svds

from ..loaders.corpus_loader import Corpus
from .artifacts import prepare_pp_output, write_pp_provenance
from .streaming import (
    PpBatch,
    PpFeatureContext,
    _resolve_feature_contexts,
    iter_dataset_batches,
    log1p_size_factor_batch,
)

DEFAULT_PCA_ARTIFACT_NAME = "pca"
_LOGNORM_FORMULA = "log1p(count / size_factor)"
_METHOD_SEMANTICS = "uncentered_truncated_svd_on_lognorm_expression"


@dataclass(frozen=True)
class PpPcaResult:
    """Collected outputs for one streamed PCA/SVD run."""

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


class _SelectedLognormLinearOperator(LinearOperator):
    """Linear operator exposing streamed selected lognorm rows for sparse SVD."""

    def __init__(
        self,
        corpus: Corpus,
        *,
        context: PpFeatureContext,
        selected_features: _SelectedFeatureSet,
        batch_size: int,
    ) -> None:
        self._corpus = corpus
        self._context = context
        self._selected_features = selected_features
        self._batch_size = int(batch_size)
        super().__init__(
            dtype=np.dtype(np.float64),
            shape=(
                int(context.global_end - context.global_start),
                selected_features.n_features,
            ),
        )

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        return self._matmat(np.asarray(vector, dtype=np.float64)[:, None]).reshape(-1)

    def _matmat(self, matrix: np.ndarray) -> np.ndarray:
        weights = np.asarray(matrix, dtype=np.float64)
        if weights.ndim != 2 or weights.shape[0] != self.shape[1]:
            raise ValueError("right operand must have shape (n_selected_features, n_vectors)")

        projected = np.empty((self.shape[0], weights.shape[1]), dtype=np.float64)
        row_offset = 0
        for _, selected in _iter_selected_lognorm_batches(
            self._corpus,
            context=self._context,
            selected_features=self._selected_features,
            batch_size=self._batch_size,
        ):
            batch_projection = selected @ weights
            next_offset = row_offset + selected.shape[0]
            projected[row_offset:next_offset] = np.asarray(batch_projection, dtype=np.float64)
            row_offset = next_offset
        return projected

    def _rmatvec(self, vector: np.ndarray) -> np.ndarray:
        return self._rmatmat(np.asarray(vector, dtype=np.float64)[:, None]).reshape(-1)

    def _rmatmat(self, matrix: np.ndarray) -> np.ndarray:
        left = np.asarray(matrix, dtype=np.float64)
        if left.ndim != 2 or left.shape[0] != self.shape[0]:
            raise ValueError("left operand must have shape (n_obs, n_vectors)")

        projected = np.zeros((self.shape[1], left.shape[1]), dtype=np.float64)
        row_offset = 0
        for _, selected in _iter_selected_lognorm_batches(
            self._corpus,
            context=self._context,
            selected_features=self._selected_features,
            batch_size=self._batch_size,
        ):
            next_offset = row_offset + selected.shape[0]
            projected += np.asarray(
                selected.transpose() @ left[row_offset:next_offset],
                dtype=np.float64,
            )
            row_offset = next_offset
        return projected


def run_pca(
    corpus: Corpus,
    *,
    dataset_id: str | None = None,
    batch_size: int = 1024,
    n_components: int = 50,
    method: str = "truncated_svd",
    hvg_frame: pl.DataFrame | None = None,
    output_dir: str | Path | None = None,
    artifact_name: str = DEFAULT_PCA_ARTIFACT_NAME,
    overwrite: bool = False,
    random_seed: int = 0,
) -> PpPcaResult:
    """Run streamed per-dataset dimensionality reduction on lognorm expression.

    Phase 9 currently exposes an uncentered sparse truncated SVD route. The
    public name stays ``run_pca`` because the plan calls for a PCA-like API, but
    provenance and component statistics record the exact method semantics.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if method != "truncated_svd":
        raise NotImplementedError(
            "Phase 9 currently supports only method='truncated_svd'"
        )

    output_root = None if output_dir is None else Path(output_dir)
    embedding_frames: list[pl.DataFrame] = []
    component_frames: list[pl.DataFrame] = []
    component_stat_frames: list[pl.DataFrame] = []
    selected_feature_frames: list[pl.DataFrame] = []

    for context in _resolve_feature_contexts(corpus, dataset_id=dataset_id):
        selected_features = _resolve_selected_features(context, hvg_frame=hvg_frame)
        min_dim = min(int(context.global_end - context.global_start), selected_features.n_features)
        if min_dim <= 1:
            raise ValueError(
                f"dataset {context.dataset_id!r} needs at least two rows/features for truncated SVD"
            )
        if n_components >= min_dim:
            raise ValueError(
                f"n_components={n_components} must be smaller than min(n_obs, n_selected_features)={min_dim}"
            )

        operator = _SelectedLognormLinearOperator(
            corpus,
            context=context,
            selected_features=selected_features,
            batch_size=batch_size,
        )
        _, singular_values, right_singular_vectors = svds(
            operator,
            k=n_components,
            solver="arpack",
            rng=np.random.default_rng(random_seed),
        )
        singular_values, right_singular_vectors = _canonicalize_svd(
            singular_values,
            right_singular_vectors,
        )

        embeddings_frame, frobenius_norm_sq = _project_dataset_embeddings(
            corpus,
            context=context,
            selected_features=selected_features,
            components=right_singular_vectors,
            batch_size=batch_size,
        )
        selected_frame = _build_selected_features_frame(context, selected_features)
        components_frame = _build_components_frame(
            context,
            selected_features,
            components=right_singular_vectors,
        )
        component_stats_frame = _build_component_stats_frame(
            context,
            singular_values=singular_values,
            n_selected_features=selected_features.n_features,
            frobenius_norm_sq=frobenius_norm_sq,
            method=method,
        )

        embedding_frames.append(embeddings_frame)
        component_frames.append(components_frame)
        component_stat_frames.append(component_stats_frame)
        selected_feature_frames.append(selected_frame)

        if output_root is not None:
            _write_dataset_outputs(
                output_root,
                corpus=corpus,
                context=context,
                embeddings=embeddings_frame,
                components=components_frame,
                component_stats=component_stats_frame,
                selected_features=selected_frame,
                artifact_name=artifact_name,
                overwrite=overwrite,
                batch_size=batch_size,
                method=method,
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


def _iter_selected_lognorm_batches(
    corpus: Corpus,
    *,
    context: PpFeatureContext,
    selected_features: _SelectedFeatureSet,
    batch_size: int,
):
    row_offset = 0
    for batch in iter_dataset_batches(corpus, dataset_id=context.dataset_id, batch_size=batch_size):
        selected = _select_lognorm_features(batch, selected_features.local_feature_positions)
        yield row_offset, selected
        row_offset += batch.batch_size


def _select_lognorm_features(
    batch: PpBatch,
    local_feature_positions: np.ndarray,
) -> csr_matrix:
    lognorm = log1p_size_factor_batch(batch, dtype=np.float64)
    return lognorm[:, local_feature_positions]


def _canonicalize_svd(
    singular_values: np.ndarray,
    right_singular_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(singular_values, dtype=np.float64))[::-1]
    sorted_values = np.asarray(singular_values, dtype=np.float64)[order]
    sorted_vectors = np.asarray(right_singular_vectors, dtype=np.float64)[order].copy()
    for component_index in range(sorted_vectors.shape[0]):
        component = sorted_vectors[component_index]
        if component.size == 0:
            continue
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            sorted_vectors[component_index] *= -1.0
    return sorted_values, sorted_vectors


def _project_dataset_embeddings(
    corpus: Corpus,
    *,
    context: PpFeatureContext,
    selected_features: _SelectedFeatureSet,
    components: np.ndarray,
    batch_size: int,
) -> tuple[pl.DataFrame, float]:
    global_rows: list[np.ndarray] = []
    local_rows: list[np.ndarray] = []
    embedding_chunks: list[np.ndarray] = []
    frobenius_norm_sq = 0.0

    for batch in iter_dataset_batches(corpus, dataset_id=context.dataset_id, batch_size=batch_size):
        selected = _select_lognorm_features(batch, selected_features.local_feature_positions)
        global_rows.append(np.asarray(batch.global_row_index, dtype=np.int64))
        local_rows.append(np.asarray(batch.local_row_index, dtype=np.int64))
        embedding_chunks.append(np.asarray(selected @ components.transpose(), dtype=np.float64))
        frobenius_norm_sq += float(np.square(np.asarray(selected.data, dtype=np.float64)).sum())

    if not embedding_chunks:
        return _empty_embeddings_frame(components.shape[0]), 0.0

    embeddings = np.vstack(embedding_chunks)
    payload: dict[str, object] = {
        "dataset_id": [context.dataset_id] * embeddings.shape[0],
        "dataset_index": np.full(embeddings.shape[0], context.dataset_index, dtype=np.int32),
        "global_row_index": np.concatenate(global_rows),
        "local_row_index": np.concatenate(local_rows),
    }
    for component_index in range(embeddings.shape[1]):
        payload[f"component_{component_index + 1}"] = embeddings[:, component_index]
    return pl.DataFrame(payload).sort("global_row_index"), frobenius_norm_sq


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
    n_selected_features: int,
    frobenius_norm_sq: float,
    method: str,
) -> pl.DataFrame:
    singular_value_sq = np.square(np.asarray(singular_values, dtype=np.float64))
    explained_variance = singular_value_sq / float(max(context.global_end - context.global_start, 1))
    if frobenius_norm_sq > 0.0:
        explained_variance_ratio = singular_value_sq / frobenius_norm_sq
    else:
        explained_variance_ratio = np.zeros_like(singular_value_sq)
    payload = {
        "dataset_id": [context.dataset_id] * len(singular_values),
        "dataset_index": np.full(len(singular_values), context.dataset_index, dtype=np.int32),
        "component_index": np.arange(1, len(singular_values) + 1, dtype=np.int32),
        "singular_value": np.asarray(singular_values, dtype=np.float64),
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance_ratio": np.cumsum(explained_variance_ratio),
        "n_obs": np.full(len(singular_values), context.global_end - context.global_start, dtype=np.int64),
        "n_features_selected": np.full(len(singular_values), n_selected_features, dtype=np.int32),
        "method": [method] * len(singular_values),
        "is_centered": [False] * len(singular_values),
        "method_semantics": [_METHOD_SEMANTICS] * len(singular_values),
    }
    return pl.DataFrame(payload).sort("component_index")


def _write_dataset_outputs(
    output_root: Path,
    *,
    corpus: Corpus,
    context: PpFeatureContext,
    embeddings: pl.DataFrame,
    components: pl.DataFrame,
    component_stats: pl.DataFrame,
    selected_features: pl.DataFrame,
    artifact_name: str,
    overwrite: bool,
    batch_size: int,
    method: str,
    random_seed: int,
) -> None:
    embeddings_spec = prepare_pp_output(
        output_root,
        dataset_id=context.dataset_id,
        artifact_name=f"{artifact_name}-embeddings",
        suffix="parquet",
    )
    components_spec = prepare_pp_output(
        output_root,
        dataset_id=context.dataset_id,
        artifact_name=f"{artifact_name}-components",
        suffix="parquet",
    )
    stats_spec = prepare_pp_output(
        output_root,
        dataset_id=context.dataset_id,
        artifact_name=f"{artifact_name}-component-stats",
        suffix="parquet",
    )
    selected_spec = prepare_pp_output(
        output_root,
        dataset_id=context.dataset_id,
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
            "method_semantics": _METHOD_SEMANTICS,
            "normalization": _LOGNORM_FORMULA,
            "n_components": int(component_stats.height),
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
            "n_features_selected": int(selected_features.height),
            "global_row_start": int(context.global_start),
            "global_row_end": int(context.global_end),
        },
    )


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
