"""Streamed per-dataset Welch t-test differential expression helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import t as student_t

from ..loaders.corpus_loader import Corpus
from .artifacts import prepare_pp_output, write_pp_provenance
from .streaming import PpFeatureContext, iter_dataset_batches, log1p_size_factor_batch

DEFAULT_TTEST_ARTIFACT_NAME = "ttest-degs"
_LOGNORM_FORMULA = "log1p(count / size_factor)"
_LOGFOLDCHANGE_SEMANTICS = "log2_difference_of_mean_lognorm_expression"
_P_VALUE_METHOD = "welch_t_test_two_sided"
_P_ADJUST_METHOD = "benjamini_hochberg"


@dataclass
class _GroupAccumulator:
    """Per-group streamed sums needed for Welch t-test DE."""

    n_features: int
    n_obs: int = 0

    def __post_init__(self) -> None:
        self.sum = np.zeros(self.n_features, dtype=np.float64)
        self.sum_sq = np.zeros(self.n_features, dtype=np.float64)
        self.n_nonzero = np.zeros(self.n_features, dtype=np.int64)

    def update(self, values) -> None:
        n_obs = int(values.shape[0])
        if n_obs <= 0:
            raise ValueError("streamed DE groups must contain at least one row")

        data = np.asarray(values.data, dtype=np.float64)
        indices = np.asarray(values.indices, dtype=np.int64)
        if data.size:
            self.sum += np.bincount(indices, weights=data, minlength=self.n_features)
            self.sum_sq += np.bincount(
                indices,
                weights=np.square(data),
                minlength=self.n_features,
            )
            nonzero_indices = indices[data != 0.0]
            if nonzero_indices.size:
                self.n_nonzero += np.bincount(
                    nonzero_indices,
                    minlength=self.n_features,
                ).astype(np.int64, copy=False)
        self.n_obs += n_obs


def rank_genes_ttest(
    corpus: Corpus,
    *,
    control_label: str,
    dataset_id: str | None = None,
    batch_size: int = 1024,
    perturbation_column: str = "perturb_label",
    top_k: int = 50,
    output_dir: str | Path | None = None,
    artifact_name: str = DEFAULT_TTEST_ARTIFACT_NAME,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Rank streamed per-dataset perturbation DE by Welch t-test.

    The test is run on streamed ``log1p(count / size_factor)`` values. Output is
    long-form and limited to the best ``top_k`` genes per perturbation-vs-control
    contrast in each dataset.
    """
    if not control_label:
        raise ValueError("control_label must be a non-empty string")
    if not perturbation_column:
        raise ValueError("perturbation_column must be a non-empty string")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    output_root = None if output_dir is None else Path(output_dir)
    control_label = str(control_label)

    frames: list[pl.DataFrame] = []
    current_context: PpFeatureContext | None = None
    current_accumulators: dict[str, _GroupAccumulator] = {}

    for batch in iter_dataset_batches(corpus, dataset_id=dataset_id, batch_size=batch_size):
        if batch.size_factor is None:
            raise ValueError("rank_genes_ttest requires canonical size_factor metadata")

        if current_context is None:
            current_context = batch.feature_context
            current_accumulators = {}
        elif batch.dataset_id != current_context.dataset_id:
            frames.append(
                _finalize_dataset_de(
                    corpus,
                    context=current_context,
                    accumulators=current_accumulators,
                    perturbation_column=perturbation_column,
                    control_label=control_label,
                    top_k=top_k,
                    output_root=output_root,
                    artifact_name=artifact_name,
                    overwrite=overwrite,
                    batch_size=batch_size,
                )
            )
            current_context = batch.feature_context
            current_accumulators = {}

        labels = _fetch_group_labels(
            corpus,
            batch.global_row_index,
            column=perturbation_column,
        )
        lognorm = log1p_size_factor_batch(batch, dtype=np.float64)
        _update_group_accumulators(current_accumulators, lognorm, labels)

    if current_context is not None:
        frames.append(
            _finalize_dataset_de(
                corpus,
                context=current_context,
                accumulators=current_accumulators,
                perturbation_column=perturbation_column,
                control_label=control_label,
                top_k=top_k,
                output_root=output_root,
                artifact_name=artifact_name,
                overwrite=overwrite,
                batch_size=batch_size,
            )
        )

    if not frames:
        return _empty_deg_frame()
    return pl.concat(frames, rechunk=True).sort(["dataset_index", "perturbation", "rank"])


def _fetch_group_labels(
    corpus: Corpus,
    global_row_index: np.ndarray,
    *,
    column: str,
) -> np.ndarray:
    metadata = corpus.take_metadata(global_row_index, columns=(column,))
    raw_values = metadata[column]
    values = np.asarray(raw_values, dtype=object)

    labels: list[str] = []
    bad_rows: list[int] = []
    for position, value in enumerate(values.tolist()):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            bad_rows.append(int(global_row_index[position]))
            continue
        labels.append(str(value))

    if bad_rows:
        preview = ", ".join(str(row) for row in bad_rows[:5])
        raise ValueError(
            f"rank_genes_ttest requires non-null {column!r} labels; found nulls at global rows {preview}"
        )
    return np.asarray(labels, dtype=object)


def _update_group_accumulators(
    accumulators: dict[str, _GroupAccumulator],
    values,
    labels: np.ndarray,
) -> None:
    unique_labels = np.unique(labels)
    for label in unique_labels.tolist():
        mask = labels == label
        group_values = values[mask]
        accumulator = accumulators.get(label)
        if accumulator is None:
            accumulator = _GroupAccumulator(values.shape[1])
            accumulators[label] = accumulator
        accumulator.update(group_values)


def _finalize_dataset_de(
    corpus: Corpus,
    *,
    context: PpFeatureContext,
    accumulators: dict[str, _GroupAccumulator],
    perturbation_column: str,
    control_label: str,
    top_k: int,
    output_root: Path | None,
    artifact_name: str,
    overwrite: bool,
    batch_size: int,
) -> pl.DataFrame:
    control = accumulators.get(control_label)
    if control is None:
        raise ValueError(
            f"dataset {context.dataset_id!r} does not contain control label {control_label!r}"
        )

    contrast_frames = [
        _build_ranked_contrast_frame(
            context,
            control_label=control_label,
            control=control,
            perturbation=label,
            perturbed=accumulator,
            top_k=top_k,
        )
        for label, accumulator in sorted(accumulators.items())
        if label != control_label
    ]
    frame = (
        _empty_deg_frame()
        if not contrast_frames
        else pl.concat(contrast_frames, rechunk=True).sort(["perturbation", "rank"])
    )

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
        write_pp_provenance(
            spec,
            corpus=corpus,
            operation="rank_genes_ttest",
            parameters={
                "batch_size": batch_size,
                "dataset_scope": "dataset",
                "perturbation_column": perturbation_column,
                "control_label": control_label,
                "top_k": top_k,
                "normalization": _LOGNORM_FORMULA,
                "test": _P_VALUE_METHOD,
                "p_adjustment": _P_ADJUST_METHOD,
                "mean_semantics": "mean_lognorm_expression",
                "logfoldchange_semantics": _LOGFOLDCHANGE_SEMANTICS,
                "group_filtering": "all non-control labels in perturbation_column",
            },
            extra={
                "contrast_labels": sorted(label for label in accumulators if label != control_label),
                "n_contrasts": sum(1 for label in accumulators if label != control_label),
                "n_obs_total": int(sum(accumulator.n_obs for accumulator in accumulators.values())),
                "n_obs_reference": int(control.n_obs),
                "n_features": int(context.n_features),
                "global_row_start": int(context.global_start),
                "global_row_end": int(context.global_end),
                "gene_token_id_semantics": "shared global feature id without special-token offset",
            },
        )

    return frame


def _build_ranked_contrast_frame(
    context: PpFeatureContext,
    *,
    control_label: str,
    control: _GroupAccumulator,
    perturbation: str,
    perturbed: _GroupAccumulator,
    top_k: int,
) -> pl.DataFrame:
    mean_reference = control.sum / float(control.n_obs)
    mean_perturbed = perturbed.sum / float(perturbed.n_obs)
    var_reference = _sample_variance(control)
    var_perturbed = _sample_variance(perturbed)
    t_stat, p_value = _welch_ttest(
        mean_reference=mean_reference,
        var_reference=var_reference,
        n_reference=control.n_obs,
        mean_perturbed=mean_perturbed,
        var_perturbed=var_perturbed,
        n_perturbed=perturbed.n_obs,
    )
    p_adj = _benjamini_hochberg(p_value)
    pct_reference = control.n_nonzero.astype(np.float64) / float(control.n_obs)
    pct_perturbed = perturbed.n_nonzero.astype(np.float64) / float(perturbed.n_obs)
    global_feature_id = np.asarray(context.local_to_global, dtype=np.int64)

    rank_order = np.lexsort(
        (
            global_feature_id,
            -np.abs(t_stat),
            np.nan_to_num(p_value, nan=np.inf, posinf=np.inf, neginf=np.inf),
            np.nan_to_num(p_adj, nan=np.inf, posinf=np.inf, neginf=np.inf),
        )
    )
    top_indices = rank_order[: min(top_k, context.n_features)]

    return pl.DataFrame(
        {
            "dataset_id": [context.dataset_id] * len(top_indices),
            "dataset_index": np.full(len(top_indices), context.dataset_index, dtype=np.int32),
            "perturbation": [perturbation] * len(top_indices),
            "reference_label": [control_label] * len(top_indices),
            "gene_id": [context.local_feature_ids[int(index)] for index in top_indices],
            "gene_token_id": global_feature_id[top_indices],
            "global_feature_id": global_feature_id[top_indices],
            "rank": np.arange(1, len(top_indices) + 1, dtype=np.int32),
            "t_stat": t_stat[top_indices],
            "p_value": p_value[top_indices],
            "p_adj": p_adj[top_indices],
            "logfoldchange": (mean_perturbed[top_indices] - mean_reference[top_indices]) / np.log(2.0),
            "mean_reference": mean_reference[top_indices],
            "mean_perturbed": mean_perturbed[top_indices],
            "pct_reference": pct_reference[top_indices],
            "pct_perturbed": pct_perturbed[top_indices],
            "n_reference": np.full(len(top_indices), control.n_obs, dtype=np.int64),
            "n_perturbed": np.full(len(top_indices), perturbed.n_obs, dtype=np.int64),
        }
    ).sort("rank")


def _sample_variance(accumulator: _GroupAccumulator) -> np.ndarray:
    if accumulator.n_obs <= 1:
        return np.zeros(accumulator.n_features, dtype=np.float64)
    centered = accumulator.sum_sq - np.square(accumulator.sum) / float(accumulator.n_obs)
    return np.maximum(centered / float(accumulator.n_obs - 1), 0.0)


def _welch_ttest(
    *,
    mean_reference: np.ndarray,
    var_reference: np.ndarray,
    n_reference: int,
    mean_perturbed: np.ndarray,
    var_perturbed: np.ndarray,
    n_perturbed: int,
) -> tuple[np.ndarray, np.ndarray]:
    ref_term = var_reference / float(n_reference)
    pert_term = var_perturbed / float(n_perturbed)
    stderr_sq = ref_term + pert_term
    diff = mean_perturbed - mean_reference

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = diff / np.sqrt(stderr_sq)

    zero_stderr = stderr_sq == 0.0
    t_stat = np.where(zero_stderr & (diff > 0.0), np.inf, t_stat)
    t_stat = np.where(zero_stderr & (diff < 0.0), -np.inf, t_stat)
    t_stat = np.where(zero_stderr & (diff == 0.0), 0.0, t_stat)

    ref_df_term = np.zeros_like(stderr_sq)
    pert_df_term = np.zeros_like(stderr_sq)
    if n_reference > 1:
        ref_df_term = np.square(ref_term) / float(n_reference - 1)
    if n_perturbed > 1:
        pert_df_term = np.square(pert_term) / float(n_perturbed - 1)

    df_denominator = ref_df_term + pert_df_term
    with np.errstate(divide="ignore", invalid="ignore"):
        degrees_of_freedom = np.square(stderr_sq) / df_denominator
    degrees_of_freedom = np.where(zero_stderr, np.inf, degrees_of_freedom)

    p_value = np.full(diff.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(t_stat) & np.isfinite(degrees_of_freedom) & (degrees_of_freedom > 0.0)
    if np.any(finite):
        p_value[finite] = student_t.sf(np.abs(t_stat[finite]), degrees_of_freedom[finite]) * 2.0
    p_value[zero_stderr & (diff == 0.0)] = 1.0
    p_value[zero_stderr & (diff != 0.0)] = 0.0
    return t_stat.astype(np.float64, copy=False), np.clip(p_value, 0.0, 1.0)


def _benjamini_hochberg(p_value: np.ndarray) -> np.ndarray:
    adjusted = np.full(p_value.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(p_value)
    if not np.any(finite_mask):
        return adjusted

    finite = np.clip(np.asarray(p_value[finite_mask], dtype=np.float64), 0.0, 1.0)
    order = np.argsort(finite, kind="mergesort")
    sorted_p = finite[order]
    n_tests = sorted_p.shape[0]
    sorted_adj = np.minimum.accumulate(
        (sorted_p * float(n_tests) / np.arange(1, n_tests + 1, dtype=np.float64))[::-1]
    )[::-1]
    sorted_adj = np.clip(sorted_adj, 0.0, 1.0)

    finite_adjusted = np.empty_like(sorted_adj)
    finite_adjusted[order] = sorted_adj
    adjusted[finite_mask] = finite_adjusted
    return adjusted


def _empty_deg_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "dataset_id": pl.String,
            "dataset_index": pl.Int32,
            "perturbation": pl.String,
            "reference_label": pl.String,
            "gene_id": pl.String,
            "gene_token_id": pl.Int64,
            "global_feature_id": pl.Int64,
            "rank": pl.Int32,
            "t_stat": pl.Float64,
            "p_value": pl.Float64,
            "p_adj": pl.Float64,
            "logfoldchange": pl.Float64,
            "mean_reference": pl.Float64,
            "mean_perturbed": pl.Float64,
            "pct_reference": pl.Float64,
            "pct_perturbed": pl.Float64,
            "n_reference": pl.Int64,
            "n_perturbed": pl.Int64,
        }
    )
