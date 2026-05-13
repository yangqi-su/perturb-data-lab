"""Streamed per-dataset HVG recalculation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .streaming import PpFeatureContext, iter_dataset_batches

if TYPE_CHECKING:
    from ..loaders.corpus_loader import Corpus


@dataclass
class _StreamingHvgAccumulator:
    """Vectorized per-feature accumulators for streamed HVG recomputation."""

    n_features: int
    n_obs: int = 0

    def __post_init__(self) -> None:
        self.sum_log1p = np.zeros(self.n_features, dtype=np.float64)
        self.sum_log1p_sq = np.zeros(self.n_features, dtype=np.float64)
        self.n_cells_detected = np.zeros(self.n_features, dtype=np.int64)

    def update(self, expression) -> None:
        data = np.asarray(expression.data, dtype=np.float64)
        indices = np.asarray(expression.indices, dtype=np.int64)
        if data.size:
            log1p_data = np.log1p(data)
            self.sum_log1p += np.bincount(
                indices,
                weights=log1p_data,
                minlength=self.n_features,
            )
            self.sum_log1p_sq += np.bincount(
                indices,
                weights=np.square(log1p_data),
                minlength=self.n_features,
            )
            detected = indices[data > 0.0]
            if detected.size:
                self.n_cells_detected += np.bincount(
                    detected,
                    minlength=self.n_features,
                ).astype(np.int64, copy=False)
        self.n_obs += int(expression.shape[0])


def compute_hvg_ranking_arrays(
    sum_log1p: np.ndarray,
    sum_log1p_sq: np.ndarray,
    n_cells_total: int,
    *,
    n_hvg: int = 2000,
) -> dict[str, np.ndarray]:
    """Compute ranked HVG metrics from streamed per-feature accumulators."""
    import pandas as pd

    if n_cells_total <= 0:
        raise ValueError("n_cells_total must be positive")
    if n_hvg < 0:
        raise ValueError("n_hvg must be non-negative")
    if len(sum_log1p) != len(sum_log1p_sq):
        raise ValueError("streaming HVG accumulators must have matching lengths")

    n_vars = len(sum_log1p)
    origin_index = np.arange(n_vars, dtype=np.int32)
    mean_log1p_expr = sum_log1p / float(n_cells_total)
    variance_log1p_expr = (sum_log1p_sq - np.square(sum_log1p) / float(n_cells_total)) / float(
        max(n_cells_total - 1, 1)
    )

    mean_for_dispersion = mean_log1p_expr.copy()
    mean_for_dispersion[mean_for_dispersion == 0] = 1e-12
    dispersion = variance_log1p_expr / mean_for_dispersion
    dispersion_for_log = dispersion.copy()
    dispersion_for_log[dispersion_for_log == 0] = np.nan
    dispersion_log = np.log(dispersion_for_log)

    mean_for_binning = np.log1p(mean_log1p_expr)
    df = pd.DataFrame(
        {
            "origin_index": origin_index,
            "means": mean_for_binning,
            "dispersions": dispersion_log,
        },
        index=origin_index,
    )
    df["mean_bin"] = pd.cut(df["means"], bins=20)
    disp_grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    disp_stats = disp_grouped.agg(avg="mean", dev="std")
    one_gene_per_bin = disp_stats["dev"].isnull()
    disp_stats.loc[one_gene_per_bin, "dev"] = disp_stats.loc[one_gene_per_bin, "avg"]
    disp_stats.loc[one_gene_per_bin, "avg"] = 0

    df["dispersions_norm"] = (
        df["dispersions"] - disp_stats.loc[df["mean_bin"], "avg"].values
    ) / disp_stats.loc[df["mean_bin"], "dev"].values

    dispersion_norm = df["dispersions_norm"].to_numpy(dtype=np.float64, copy=False)
    ranked_dispersion = np.nan_to_num(dispersion_norm, nan=-np.inf)
    rank_order = np.lexsort((origin_index, -ranked_dispersion))
    hvg_rank = np.empty(n_vars, dtype=np.int32)
    hvg_rank[rank_order] = np.arange(1, n_vars + 1, dtype=np.int32)
    selected_at_default_n_hvg = hvg_rank <= min(n_hvg, n_vars)

    return {
        "origin_index": origin_index,
        "mean_log1p_expr": mean_log1p_expr.astype(np.float64, copy=False),
        "variance_log1p_expr": variance_log1p_expr.astype(np.float64, copy=False),
        "dispersion": dispersion.astype(np.float64, copy=False),
        "dispersion_log": dispersion_log.astype(np.float64, copy=False),
        "dispersion_norm": dispersion_norm.astype(np.float64, copy=False),
        "hvg_rank": hvg_rank,
        "selected_at_default_n_hvg": selected_at_default_n_hvg.astype(bool, copy=False),
    }


def build_ranked_hvg_frame(
    *,
    dataset_id: str,
    dataset_index: int,
    gene_ids: tuple[str, ...] | list[str],
    sum_log1p: np.ndarray,
    sum_log1p_sq: np.ndarray,
    n_cells_total: int,
    n_cells_detected: np.ndarray,
    n_hvg: int,
    global_feature_ids: np.ndarray | None = None,
) -> pl.DataFrame:
    """Build one dataset-local ranked HVG table with legacy compatibility fields."""
    gene_id_list = [str(gene_id) for gene_id in gene_ids]
    if len(gene_id_list) != len(sum_log1p):
        raise ValueError("gene_ids must align with streamed HVG accumulators")

    arrays = compute_hvg_ranking_arrays(
        sum_log1p,
        sum_log1p_sq,
        n_cells_total,
        n_hvg=n_hvg,
    )
    payload: dict[str, object] = {
        "dataset_id": [str(dataset_id)] * len(gene_id_list),
        "dataset_index": np.full(len(gene_id_list), int(dataset_index), dtype=np.int32),
        "origin_index": arrays["origin_index"],
        "gene_id": gene_id_list,
        "feature_id": gene_id_list,
        "mean": arrays["mean_log1p_expr"],
        "variance": arrays["variance_log1p_expr"],
        "dispersion": arrays["dispersion"],
        "dispersion_log": arrays["dispersion_log"],
        "dispersion_norm": arrays["dispersion_norm"],
        "hvg_rank": arrays["hvg_rank"],
        "is_hvg": arrays["selected_at_default_n_hvg"],
        "selected_at_default_n_hvg": arrays["selected_at_default_n_hvg"],
        "n_cells_detected": np.asarray(n_cells_detected, dtype=np.int64),
        "mean_log1p_expr": arrays["mean_log1p_expr"],
        "variance_log1p_expr": arrays["variance_log1p_expr"],
    }
    if global_feature_ids is not None:
        global_ids = np.asarray(global_feature_ids, dtype=np.int64)
        if global_ids.shape != (len(gene_id_list),):
            raise ValueError("global_feature_ids must align with gene_ids")
        payload["global_feature_id"] = global_ids
    return pl.DataFrame(payload).sort("origin_index")


def calculate_hvgs(
    corpus: Corpus,
    *,
    dataset_id: str | None = None,
    batch_size: int = 1024,
    n_hvg: int = 2000,
) -> pl.DataFrame:
    """Stream per-dataset ranked HVG metrics from an existing corpus."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_hvg <= 0:
        raise ValueError("n_hvg must be positive")

    frames: list[pl.DataFrame] = []
    current_context: PpFeatureContext | None = None
    current_accumulator: _StreamingHvgAccumulator | None = None

    for batch in iter_dataset_batches(corpus, dataset_id=dataset_id, batch_size=batch_size):
        if current_context is None:
            current_context = batch.feature_context
            current_accumulator = _StreamingHvgAccumulator(current_context.n_features)
        elif batch.dataset_id != current_context.dataset_id:
            assert current_accumulator is not None
            frames.append(
                build_ranked_hvg_frame(
                    dataset_id=current_context.dataset_id,
                    dataset_index=current_context.dataset_index,
                    gene_ids=current_context.local_feature_ids,
                    global_feature_ids=current_context.local_to_global,
                    sum_log1p=current_accumulator.sum_log1p,
                    sum_log1p_sq=current_accumulator.sum_log1p_sq,
                    n_cells_total=current_accumulator.n_obs,
                    n_cells_detected=current_accumulator.n_cells_detected,
                    n_hvg=n_hvg,
                )
            )
            current_context = batch.feature_context
            current_accumulator = _StreamingHvgAccumulator(current_context.n_features)

        assert current_accumulator is not None
        current_accumulator.update(batch.expression)

    if current_context is not None and current_accumulator is not None:
        frames.append(
            build_ranked_hvg_frame(
                dataset_id=current_context.dataset_id,
                dataset_index=current_context.dataset_index,
                gene_ids=current_context.local_feature_ids,
                global_feature_ids=current_context.local_to_global,
                sum_log1p=current_accumulator.sum_log1p,
                sum_log1p_sq=current_accumulator.sum_log1p_sq,
                n_cells_total=current_accumulator.n_obs,
                n_cells_detected=current_accumulator.n_cells_detected,
                n_hvg=n_hvg,
            )
        )

    if not frames:
        return _empty_hvg_frame()
    return pl.concat(frames, rechunk=True).sort(["dataset_index", "origin_index"])


def _empty_hvg_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "dataset_id": pl.String,
            "dataset_index": pl.Int32,
            "origin_index": pl.Int32,
            "gene_id": pl.String,
            "feature_id": pl.String,
            "global_feature_id": pl.Int64,
            "mean": pl.Float64,
            "variance": pl.Float64,
            "dispersion": pl.Float64,
            "dispersion_log": pl.Float64,
            "dispersion_norm": pl.Float64,
            "hvg_rank": pl.Int32,
            "is_hvg": pl.Boolean,
            "selected_at_default_n_hvg": pl.Boolean,
            "n_cells_detected": pl.Int64,
            "mean_log1p_expr": pl.Float64,
            "variance_log1p_expr": pl.Float64,
        }
    )
