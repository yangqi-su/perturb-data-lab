from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from ..contracts import CONTRACT_VERSION
from .models import (
    CountSourceCandidate,
    CountSourceDecision,
    DatasetIdentity,
    DatasetSummaryDocument,
    FieldProfile,
    InspectionBatchConfig,
    InspectionBatchManifest,
    InspectionBatchRecord,
    InspectionTarget,
    StructureSummary,
)


DEFAULT_FIELD_SAMPLE_SIZE = 5
DEFAULT_MATRIX_SAMPLE_ROWS = 32


@dataclass(frozen=True)
class InspectionArtifacts:
    dataset_id: str
    review_bundle: Path  # absolute path to this dataset's review bundle (dataset-summary.yaml)
    selected_count_source: str
    materialization_readiness: str


def _normalize_label(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _sample_examples(
    series: object, sample_size: int = DEFAULT_FIELD_SAMPLE_SIZE
) -> tuple[str, ...]:
    non_null = series.dropna()
    if non_null.empty:
        return ()
    values = []
    seen = set()
    for value in non_null.astype(str).head(sample_size * 3).tolist():
        if value not in seen:
            values.append(value)
            seen.add(value)
        if len(values) >= sample_size:
            break
    return tuple(values)


def _profile_fields(frame: object) -> tuple[FieldProfile, ...]:
    profiles = []
    for column_name in frame.columns.tolist():
        series = frame[column_name]
        sampled = series.dropna().astype(str).head(64)
        profiles.append(
            FieldProfile(
                name=str(column_name),
                dtype=str(series.dtype),
                null_count=int(series.isna().sum()),
                sampled_unique_values=int(sampled.nunique()),
                examples=_sample_examples(series),
            )
        )
    return tuple(profiles)


def _sample_row_indices(row_count: int, sample_rows: int) -> np.ndarray:
    if row_count <= 0:
        return np.array([], dtype=int)
    if row_count <= sample_rows:
        return np.arange(row_count, dtype=int)
    return np.unique(np.linspace(0, row_count - 1, num=sample_rows, dtype=int))


def _infer_transform_family(candidate_name: str) -> str:
    label = candidate_name.lower()
    if any(token in label for token in ("log", "norm", "scale")):
        return "transformed"
    if "bin" in label:
        return "binned"
    if "count" in label:
        return "count-like"
    if candidate_name == ".raw.X":
        return "raw"
    return "identity"


def _source_priority(candidate_name: str) -> int:
    label = candidate_name.lower()
    if "count" in label and not any(token in label for token in ("norm", "log", "bin")):
        return 500
    if candidate_name == ".raw.X":
        return 450
    if candidate_name == ".X":
        return 400
    if any(token in label for token in ("norm", "log", "scale")):
        return 100
    if "bin" in label:
        return 80
    return 250


def _audit_matrix_candidate(
    candidate_name: str,
    matrix: object,
    row_count: int,
    column_count: int,
    sample_rows: int = DEFAULT_MATRIX_SAMPLE_ROWS,
) -> CountSourceCandidate:
    indices = _sample_row_indices(row_count, sample_rows)
    sampled = matrix[indices] if len(indices) else matrix[:0]
    if issparse(sampled):
        sampled_nonzero = np.asarray(sampled.data)
        sampled_density = (
            0.0
            if sampled.shape[0] * sampled.shape[1] == 0
            else sampled.nnz / float(sampled.shape[0] * sampled.shape[1])
        )
        storage = "sparse"
        dtype = str(sampled.dtype)
    else:
        sampled_array = np.asarray(sampled)
        sampled_nonzero = sampled_array[sampled_array != 0]
        sampled_density = (
            0.0
            if sampled_array.size == 0
            else np.count_nonzero(sampled_array) / float(sampled_array.size)
        )
        storage = "dense"
        dtype = str(sampled_array.dtype)

    inferred_transform = _infer_transform_family(candidate_name)
    notes = []
    if sampled_nonzero.size == 0:
        fraction_noninteger = 0.0
        max_deviation = 0.0
        nonnegative = True
        status = "needs-review"
        recovery_policy = "disallowed"
        notes.append("sample contained no nonzero values")
    else:
        deviations = np.abs(sampled_nonzero - np.rint(sampled_nonzero))
        max_deviation = float(np.max(deviations))
        fraction_noninteger = float(np.mean(deviations > 1e-8))
        nonnegative = bool(np.all(sampled_nonzero >= 0))
        if nonnegative and fraction_noninteger == 0.0:
            status = "pass"
            recovery_policy = "not-needed"
            notes.append("sample is exactly integer-valued")
        elif nonnegative and max_deviation <= 1e-6 and inferred_transform == "identity":
            status = "needs-review"
            recovery_policy = "allowed-with-explicit-assumption"
            notes.append(
                "sample is near-integer but would require explicit recovery approval"
            )
        else:
            status = "fail"
            recovery_policy = "disallowed"
            notes.append(
                "sample includes non-integer or unsupported transformed values"
            )

    if inferred_transform in {"transformed", "binned"}:
        notes.append(f"candidate name suggests {inferred_transform} data")

    return CountSourceCandidate(
        candidate=candidate_name,
        rank=0,
        status=status,
        storage=storage,
        dtype=dtype,
        shape=(int(row_count), int(column_count)),
        sampled_rows=int(len(indices)),
        sampled_nonzero_values=int(sampled_nonzero.size),
        sampled_density=float(sampled_density),
        fraction_noninteger_nonzero=float(fraction_noninteger),
        max_abs_integer_deviation=float(max_deviation),
        nonnegative=nonnegative,
        inferred_transform=inferred_transform,
        recovery_policy=recovery_policy,
        notes=tuple(notes),
    )


def _rank_candidates(
    candidates: Iterable[CountSourceCandidate],
) -> tuple[CountSourceCandidate, ...]:
    scored = []
    for candidate in candidates:
        score = _source_priority(candidate.candidate)
        score += {"pass": 50, "needs-review": 10, "fail": -200}[candidate.status]
        scored.append((score, candidate))
    ranked = []
    for rank, (_, candidate) in enumerate(
        sorted(scored, key=lambda item: item[0], reverse=True), start=1
    ):
        ranked.append(
            CountSourceCandidate(
                candidate=candidate.candidate,
                rank=rank,
                status=candidate.status,
                storage=candidate.storage,
                dtype=candidate.dtype,
                shape=candidate.shape,
                sampled_rows=candidate.sampled_rows,
                sampled_nonzero_values=candidate.sampled_nonzero_values,
                sampled_density=candidate.sampled_density,
                fraction_noninteger_nonzero=candidate.fraction_noninteger_nonzero,
                max_abs_integer_deviation=candidate.max_abs_integer_deviation,
                nonnegative=candidate.nonnegative,
                inferred_transform=candidate.inferred_transform,
                recovery_policy=candidate.recovery_policy,
                notes=candidate.notes,
            )
        )
    return tuple(ranked)


def _attempt_reverse_normalization(
    candidate: CountSourceCandidate,
    adata: ad.AnnData,
) -> CountSourceCandidate | None:
    """Attempt to recover integer counts via the approved reverse-normalization path.

    The only approved path is ``expm1(expr) / size_factor`` on log-normalized data,
    where ``size_factor`` is the minimum nonzero value of ``expm1(expr)`` computed
    per-row across all candidate expression sources.

    Recovery is attempted when:
    - ``candidate`` is not a direct integer source (status != "pass")
    - the adata object is available so raw matrix access is possible

    When no direct integer source exists anywhere in the adata, the candidate
    matrix itself is the recovery source. Per-row scale factors are derived from
    ``expm1(nonzero values)`` in each row — specifically the minimum nonzero
    recovered value per row. This represents the "unit" count that the forward
    log-normalization used as a reference.

    Returns an updated ``CountSourceCandidate`` with ``recovery_policy`` set to
    ``"expm1_over_size_factor"`` on successful recovery, or ``None`` if recovery
    cannot be performed. Does not modify the original candidate object.
    """
    import numpy as np

    # Only recover when candidate is not already integer
    if candidate.status == "pass":
        return None

    # Determine which matrix to recover from
    if candidate.candidate == ".raw.X":
        matrix_ref = adata.raw.X
    elif candidate.candidate.startswith(".layers["):
        layer_name = candidate.candidate[len(".layers[") : -1]
        matrix_ref = adata.layers[layer_name]
    elif candidate.candidate == ".X":
        matrix_ref = adata.X
    else:
        return None

    n_obs = adata.n_obs
    sample_rows = min(DEFAULT_MATRIX_SAMPLE_ROWS, n_obs)
    indices = _sample_row_indices(n_obs, sample_rows)

    # Apply expm1 / size_factor on sampled rows.
    # Scale factor per row = minimum nonzero value of expm1(row).
    # This represents the unit the forward log-normalization preserved.
    recovered_values: list[np.ndarray] = []
    for idx in indices:
        row = matrix_ref[idx]
        if hasattr(row, "toarray"):
            row = np.asarray(row.toarray().ravel())
        else:
            row = np.asarray(row).ravel()

        # expm1 on nonzero entries
        nonzero_mask = row != 0
        if not nonzero_mask.any():
            continue

        expm1_row = np.expm1(row[nonzero_mask])

        # Per-row scale factor = minimum nonzero recovered value
        sf = float(np.min(expm1_row))
        if sf <= 0:
            continue

        recovered = expm1_row / sf
        recovered_values.append(recovered)

    if not recovered_values:
        return None

    all_recovered = np.concatenate(recovered_values)
    if all_recovered.size == 0:
        return None

    deviations = np.abs(all_recovered - np.rint(all_recovered))
    max_deviation = float(np.max(deviations))
    fraction_noninteger = float(np.mean(deviations > 1e-8))

    # Integer threshold: max_abs_integer_deviation < 0.01
    if max_deviation > 0.01:
        return None

    # Recovery succeeded — build updated candidate
    status = "pass"
    if fraction_noninteger > 0.0:
        status = "needs-review"

    notes = (
        f"recovered via expm1/size_factor (scale=min_nonzero_expm1); "
        f"max_deviation={max_deviation:.6f}; "
        f"fraction_noninteger={fraction_noninteger:.6f}"
    )
    return CountSourceCandidate(
        candidate=candidate.candidate,
        rank=candidate.rank,
        status=status,
        storage=candidate.storage,
        dtype=candidate.dtype,
        shape=candidate.shape,
        sampled_rows=candidate.sampled_rows,
        sampled_nonzero_values=candidate.sampled_nonzero_values,
        sampled_density=candidate.sampled_density,
        fraction_noninteger_nonzero=fraction_noninteger,
        max_abs_integer_deviation=max_deviation,
        nonnegative=candidate.nonnegative,
        inferred_transform="recovered-count",
        recovery_policy="expm1_over_size_factor",
        notes=(notes,),
    )


def _choose_count_source(
    candidates: tuple[CountSourceCandidate, ...],
    adata: ad.AnnData | None = None,
) -> CountSourceDecision:
    """Choose the winning count source under the Phase 2 contract.

    Precedence (in order):
    1. Largest feature dimension (columns) among all passing sources.
    2. Among equally-sized passing sources, direct integer beats recovered.
    3. If still tied, preserve existing source enumeration order.

    Every candidate receives both a direct check (from _audit_matrix_candidate) and,
    if needed, a reverse-normalized check via _attempt_reverse_normalization.
    Bin-named sources are included in both checks.
    """
    if not candidates:
        return CountSourceDecision(
            selected_candidate="none",
            status="fail",
            confidence="low",
            recovery_policy="disallowed",
            rationale="No matrix candidates were available for audit.",
            uses_recovery=False,
            pass_mode=None,
        )

    passing: list[tuple[CountSourceCandidate, str]] = []  # (candidate, pass_mode)

    for candidate in candidates:
        if candidate.status == "pass":
            # Direct integer pass
            passing.append((candidate, "direct"))
        elif adata is not None:
            # Attempt reverse-normalization recovery
            recovered = _attempt_reverse_normalization(candidate, adata)
            if recovered is not None:
                passing.append((recovered, "recovered"))

    if not passing:
        return CountSourceDecision(
            selected_candidate="none",
            status="fail",
            confidence="low",
            recovery_policy="disallowed",
            rationale="No passing count source found among candidates.",
            uses_recovery=False,
            pass_mode=None,
        )

    # Sort: (1) feature count desc, (2) pass_mode direct before recovered, (3) source order
    source_order = {c.candidate: idx for idx, c in enumerate(candidates)}
    passing.sort(
        key=lambda item: (
            -item[0].shape[1],  # largest feature count first
            0 if item[1] == "direct" else 1,  # direct before recovered
            source_order.get(item[0].candidate, 10**9),  # existing source order
        )
    )

    selected_candidate, pass_mode = passing[0]
    uses_recovery = pass_mode == "recovered"

    confidence = (
        "high"
        if selected_candidate.status == "pass"
        else "medium"
        if selected_candidate.status == "needs-review"
        else "low"
    )
    pass_mode_str = "direct integer" if pass_mode == "direct" else "recovered via expm1/size_factor"
    rationale = (
        f"Selected {selected_candidate.candidate} because it is the largest passing source "
        f"({selected_candidate.shape[1]} features) with {pass_mode_str}."
    )

    return CountSourceDecision(
        selected_candidate=selected_candidate.candidate,
        status=selected_candidate.status,
        confidence=confidence,
        recovery_policy=selected_candidate.recovery_policy,
        rationale=rationale,
        uses_recovery=uses_recovery,
        pass_mode=pass_mode,
    )


def _derive_readiness_from_count(count_status: str) -> str:
    """Derive materialization readiness from count-source evidence only.

    Per Phase 2 contract: no schema field completeness check.
    """
    if count_status == "fail":
        return "fail"
    if count_status == "needs-review":
        return "needs-review"
    return "pass"


def inspect_target(target: InspectionTarget, output_root: Path) -> InspectionArtifacts:
    dataset_dir = output_root / target.dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"[inspect] start {target.dataset_id}")

    adata = ad.read_h5ad(target.source_path, backed="r")
    try:
        obs_fields = _profile_fields(adata.obs)
        var_fields = _profile_fields(adata.var)

        candidates = []
        try:
            candidates.append(_audit_matrix_candidate(".X", adata.X, adata.n_obs, adata.n_vars))
        except KeyError:
            pass  # no .X in this file
        if adata.raw is not None:
            candidates.append(
                _audit_matrix_candidate(
                    ".raw.X",
                    adata.raw.X,
                    adata.n_obs,
                    int(adata.raw.shape[1]),
                )
            )
        for layer_name in adata.layers.keys():
            candidates.append(
                _audit_matrix_candidate(
                    f".layers[{layer_name}]",
                    adata.layers[layer_name],
                    adata.n_obs,
                    adata.n_vars,
                )
            )

        ranked_candidates = _rank_candidates(candidates)
        count_decision = _choose_count_source(ranked_candidates, adata)
        readiness = _derive_readiness_from_count(count_decision.status)

        summary = DatasetSummaryDocument(
            kind="dataset-summary",
            contract_version=CONTRACT_VERSION,
            dataset=DatasetIdentity(
                dataset_id=target.dataset_id,
                source_release=target.source_release,
                source_path=target.source_path,
                obs_rows=int(adata.n_obs),
                var_rows=int(adata.n_vars),
                obs_index_name=str(adata.obs.index.name or "index"),
                var_index_name=str(adata.var.index.name or "index"),
            ),
            structure=StructureSummary(
                has_raw=adata.raw is not None,
                raw_var_rows=0 if adata.raw is None else int(adata.raw.shape[1]),
                layers=tuple(str(layer_name) for layer_name in adata.layers.keys()),
            ),
            obs_fields=obs_fields,
            var_fields=var_fields,
            count_source_candidates=ranked_candidates,
            count_source_decision=count_decision,
            materialization_readiness=readiness,
            inspector_notes=(
                "metadata-first scan used backed h5ad access",
                "matrix audit used sampled rows only and did not materialize the full matrix",
            ),
        )

        summary_path = dataset_dir / "dataset-summary.yaml"
        summary.validate()
        summary.write_yaml(summary_path)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    print(
        f"[inspect] done {target.dataset_id} count={count_decision.selected_candidate} readiness={readiness}"
    )
    return InspectionArtifacts(
        dataset_id=target.dataset_id,
        review_bundle=summary_path,
        selected_count_source=count_decision.selected_candidate,
        materialization_readiness=readiness,
    )


def run_batch(
    config: InspectionBatchConfig, workers: int = 1
) -> InspectionBatchManifest:
    """Thin batch wrapper: run Stage 1 inspection on each configured dataset.

    This function is a pure aggregator — it calls ``inspect_target()`` for each
    dataset and records one manifest entry per dataset. The manifest is a batch
    index; the authoritative artifact for any dataset is its review bundle
    (dataset-summary.yaml) on disk, not the manifest itself.

    Each manifest record points to exactly one review bundle path. No schema.yaml
    or other Stage 1 dual-artifact output is produced or recorded.
    """
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, min(workers, len(config.datasets) or 1))
    if worker_count == 1:
        artifacts = [inspect_target(target, output_root) for target in config.datasets]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(inspect_target, target, output_root)
                for target in config.datasets
            ]
            artifacts = [future.result() for future in futures]

    manifest = InspectionBatchManifest(
        kind="inspection-batch-manifest",
        contract_version=CONTRACT_VERSION,
        output_root=str(output_root),
        records=tuple(
            InspectionBatchRecord(
                dataset_id=artifact.dataset_id,
                review_bundle=str(artifact.review_bundle),
                selected_count_source=artifact.selected_count_source,
                materialization_readiness=artifact.materialization_readiness,
            )
            for artifact in sorted(artifacts, key=lambda item: item.dataset_id)
        ),
    )
    manifest.write_yaml(output_root / "inspection-manifest.yaml")
    return manifest
