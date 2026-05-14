"""Streamed per-dataset log-normalized summary statistics."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import numpy as np
import polars as pl

from ..loaders.corpus_loader import Corpus
from .artifacts import prepare_pp_output, write_pp_provenance
from .streaming import (
    PpFeatureContext,
    _summarize_sparse_features,
    iter_dataset_batches,
    log1p_size_factor_batch,
)

DEFAULT_LOGNORM_STATS_ARTIFACT_NAME = "lognorm-stats"
_LOGNORM_FORMULA = "log1p(count / size_factor)"
_SIZE_FACTOR_SOURCE = "canonical_obs.size_factor"


@dataclass
class _StreamingVarianceAccumulator:
    """Vectorized per-feature streaming mean/variance state."""

    n_features: int
    n_obs: int = 0

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.n_features, dtype=np.float64)
        self.m2 = np.zeros(self.n_features, dtype=np.float64)
        self.n_nonzero = np.zeros(self.n_features, dtype=np.int64)

    def update(self, values, *, n_obs: int) -> None:
        if n_obs <= 0:
            raise ValueError("streamed lognorm batches must contain at least one row")

        summary = _summarize_sparse_features(values, n_features=self.n_features)
        batch_sum = summary.sum
        batch_sum_sq = summary.sum_sq
        batch_n_nonzero = summary.n_nonzero

        batch_mean = batch_sum / float(n_obs)
        batch_m2 = np.maximum(batch_sum_sq - float(n_obs) * np.square(batch_mean), 0.0)

        if self.n_obs == 0:
            self.mean[:] = batch_mean
            self.m2[:] = batch_m2
            self.n_obs = int(n_obs)
            self.n_nonzero += batch_n_nonzero
            return

        total_n = self.n_obs + int(n_obs)
        delta = batch_mean - self.mean
        self.mean += delta * (float(n_obs) / float(total_n))
        self.m2 += batch_m2 + np.square(delta) * (
            float(self.n_obs) * float(n_obs) / float(total_n)
        )
        self.n_obs = int(total_n)
        self.n_nonzero += batch_n_nonzero


def calculate_lognorm_stats(
    corpus: Corpus,
    *,
    dataset_id: str | None = None,
    batch_size: int = 1024,
    output_dir: str | Path | None = None,
    artifact_name: str = DEFAULT_LOGNORM_STATS_ARTIFACT_NAME,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Stream per-dataset ``log1p(count / size_factor)`` mean/variance stats.

    Returns one deterministic long-form table across the requested dataset scope.
    When ``output_dir`` is provided, this also writes one parquet table and one
    provenance sidecar per dataset under ``output_dir/<dataset_id>/``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_root = None if output_dir is None else Path(output_dir)
    frames: list[pl.DataFrame] = []
    current_context: PpFeatureContext | None = None
    current_accumulator: _StreamingVarianceAccumulator | None = None

    for batch in iter_dataset_batches(corpus, dataset_id=dataset_id, batch_size=batch_size):
        if batch.size_factor is None:
            raise ValueError(
                "calculate_lognorm_stats requires canonical size_factor metadata"
            )

        if current_context is None:
            current_context = batch.feature_context
            current_accumulator = _StreamingVarianceAccumulator(current_context.n_features)
        elif batch.dataset_id != current_context.dataset_id:
            assert current_accumulator is not None
            frames.append(
                _finalize_dataset_stats(
                    corpus,
                    context=current_context,
                    accumulator=current_accumulator,
                    output_root=output_root,
                    artifact_name=artifact_name,
                    overwrite=overwrite,
                    batch_size=batch_size,
                )
            )
            current_context = batch.feature_context
            current_accumulator = _StreamingVarianceAccumulator(current_context.n_features)

        assert current_accumulator is not None
        current_accumulator.update(
            log1p_size_factor_batch(batch, dtype=np.float64),
            n_obs=batch.batch_size,
        )

    if current_context is not None and current_accumulator is not None:
        frames.append(
            _finalize_dataset_stats(
                corpus,
                context=current_context,
                accumulator=current_accumulator,
                output_root=output_root,
                artifact_name=artifact_name,
                overwrite=overwrite,
                batch_size=batch_size,
            )
        )

    if not frames:
        return _empty_stats_frame()
    return pl.concat(frames, rechunk=True).sort(["dataset_index", "global_feature_id"])


def _finalize_dataset_stats(
    corpus: Corpus,
    *,
    context: PpFeatureContext,
    accumulator: _StreamingVarianceAccumulator,
    output_root: Path | None,
    artifact_name: str,
    overwrite: bool,
    batch_size: int,
) -> pl.DataFrame:
    variance = np.maximum(accumulator.m2 / float(accumulator.n_obs), 0.0)
    frame = pl.DataFrame(
        {
            "dataset_id": [context.dataset_id] * context.n_features,
            "dataset_index": np.full(context.n_features, context.dataset_index, dtype=np.int32),
            "gene_id": list(context.local_feature_ids),
            "global_feature_id": np.asarray(context.local_to_global, dtype=np.int64),
            "mean_lognorm": accumulator.mean,
            "var_lognorm": variance,
            "std_lognorm": np.sqrt(variance),
            "n_obs": np.full(context.n_features, accumulator.n_obs, dtype=np.int64),
            "n_nonzero": accumulator.n_nonzero,
        }
    ).sort("global_feature_id")

    if output_root is not None:
        spec = prepare_pp_output(
            output_root,
            dataset_id=context.dataset_id,
            artifact_name=artifact_name,
            suffix="parquet",
        )
        if spec.artifact_path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {spec.artifact_path}")
        frame.write_parquet(spec.artifact_path)
        extra: dict[str, object] = {
            "n_obs": accumulator.n_obs,
            "n_features": context.n_features,
            "global_row_start": context.global_start,
            "global_row_end": context.global_end,
        }
        software_version = _resolve_software_version()
        if software_version is not None:
            extra["software_version"] = software_version
        write_pp_provenance(
            spec,
            corpus=corpus,
            operation="calculate_lognorm_stats",
            parameters={
                "batch_size": batch_size,
                "dataset_scope": "dataset",
                "normalization": _LOGNORM_FORMULA,
                "size_factor_source": _SIZE_FACTOR_SOURCE,
                "variance_ddof": 0,
            },
            extra=extra,
        )

    return frame


def _resolve_software_version() -> str | None:
    try:
        return package_version("perturb-data-lab")
    except PackageNotFoundError:
        return None


def _empty_stats_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "dataset_id": pl.String,
            "dataset_index": pl.Int32,
            "gene_id": pl.String,
            "global_feature_id": pl.Int64,
            "mean_lognorm": pl.Float64,
            "var_lognorm": pl.Float64,
            "std_lognorm": pl.Float64,
            "n_obs": pl.Int64,
            "n_nonzero": pl.Int64,
        }
    )
