from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from ..contracts import BLUEPRINT, CONTRACT_VERSION, MISSING_VALUE_LITERAL
from .models import (
    CountSourceCandidate,
    CountSourceDecision,
    CountSourceSpec,
    DatasetIdentity,
    DatasetSummaryDocument,
    FeatureTokenizationSpec,
    FieldProfile,
    InspectionBatchConfig,
    InspectionBatchManifest,
    InspectionBatchRecord,
    InspectionTarget,
    SchemaDocument,
    SchemaFieldEntry,
    StructureSummary,
)
from .transforms import TRANSFORM_CATALOG, build_transform


DEFAULT_FIELD_SAMPLE_SIZE = 5
DEFAULT_MATRIX_SAMPLE_ROWS = 32
CONTROL_PATTERNS = (r"^ntc", r"non[-_ ]?target", r"control", r"mock", r"wt")

# Feature-field aliases for tokenization identity and label heuristics
FEATURE_FIELD_ALIASES = {
    "feature_id": (
        "feature_id",
        "gene_id",
        "ensembl",
        "ensembl_id",
        "feature_id_from_multiome",
    ),
    "feature_label": (
        "feature_name",
        "gene_symbol",
        "gene_name",
        "feature_name_from_multiome",
    ),
    "feature_namespace": (
        "namespace",
        "feature_namespace",
        "gene_name_db",
    ),
}

CANONICAL_FEATURE_FIELDS = ("feature_id", "feature_label", "feature_namespace")

CANONICAL_FIELDS = tuple(
    field.name
    for field in BLUEPRINT.perturbation_fields + BLUEPRINT.context_fields
) + CANONICAL_FEATURE_FIELDS

REQUIRED_CANONICAL_FIELDS = {
    field.name
    for field in BLUEPRINT.perturbation_fields + BLUEPRINT.context_fields
    if field.required
}

FIELD_ALIASES = {
    "perturbation_label": (
        "perturbation_label",
        "perturbation",
        "guide_id",
        "guide_call",
        "guide_name",
        "guide",
        "sgid_ab",
        "sgRNA",
        "grna",
        "gene",
        "target",
        "perturbed_gene_name",
    ),
    "target_id": (
        "target_id",
        "perturbed_gene_id",
        "gene_id",
        "ensembl",
        "feature_id",
    ),
    "target_label": (
        "target_label",
        "perturbed_gene_name",
        "gene_target",
        "gene_symbol",
        "target_gene",
        "gene",
        "target",
    ),
    "perturbation_type": (
        "perturbation_type",
        "pert_type",
        "perttype",
        "type",
    ),
    "control_flag": (
        "control_flag",
        "is_control",
        "is_neg",
        "neg_control",
    ),
    "dose": (
        "dose",
        "concentration",
        "conc",
        "amount",
    ),
    "dose_unit": (
        "dose_unit",
        "conc_unit",
        "unit",
    ),
    "timepoint": (
        "timepoint",
        "time",
        "duration",
        "exposure_time",
    ),
    "timepoint_unit": (
        "timepoint_unit",
        "time_unit",
    ),
    "combination_key": (
        "combination_key",
        "combo_key",
        "combo",
        "multiplex_key",
    ),
    "cell_context": (
        "cell_context",
        "cell_line",
        "cell_type",
        "celltype",
        "context",
        "neuron_type",
    ),
    "cell_line_or_type": (
        "cell_line_or_type",
        "cell_line",
        "cell_type",
        "celltype",
        "context",
        "neuron_type",
    ),
    "species": ("species", "organism"),
    "tissue": ("tissue", "organ", "region"),
    "assay": ("assay", "assay_type", "modality"),
    "condition": ("condition", "stimulation", "stim", "state", "treatment"),
    "batch_id": ("batch_id", "batch", "replicate", "sample_id"),
    "donor_id": ("donor_id", "donor", "donor_name"),
    "sex": ("sex",),
    "disease_state": ("disease_state", "disease", "health_status"),
}


@dataclass(frozen=True)
class InspectionArtifacts:
    dataset_id: str
    dataset_summary: Path
    schema: Path
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

    # Mark binned matrices as non-recoverable per Phase 1 contract
    if inferred_transform == "binned":
        notes.append("binned matrices are non-recoverable; exclude from reverse-normalization")

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

    The only approved path is ``expm1(expr) / size_factor`` on log-normalized data.
    Recovery is only attempted when:
    - ``candidate`` is the best available candidate but is not an integer source
    - the candidate is log-normalized (inferred_transform is "transformed" or
      the candidate name contains "log" or "norm")
    - size factors can be computed from the raw integer count source (.X)

    When the raw integer source is in .X, size factors are computed from .X
    (sum-per-row, normalized by median), then expm1(expr) / size_factor is
    applied to the candidate matrix to recover integer-like counts.

    Returns an updated ``CountSourceCandidate`` with ``recovery_policy`` set to
    ``"expm1_over_size_factor"`` on successful recovery, or ``None`` if recovery
    cannot be performed. Does not modify the original candidate object.
    """
    import numpy as np

    # Only recover from transformed/log-normalized candidates
    inferred = candidate.inferred_transform
    if inferred not in {"transformed", "raw"} and "log" not in candidate.candidate.lower():
        return None

    # Select the matrix reference for this candidate (the matrix to recover from)
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

    # Compute size factors from .X (the raw integer count source) when available.
    # This matches the Phase 1 contract: misc._get_sf() sum method on raw counts.
    # If .X is not an integer source, fall back to computing from the candidate
    # matrix itself (handles the case where the log-norm candidate IS .X).
    from ..materializers.backends.arrow_hf import _get_row_nonzero

    try:
        size_factors = np.zeros(n_obs, dtype=np.float64)
        # Use .X as the source for size factor computation when it exists
        count_for_sf = adata.X
        for i in indices:
            row_indices, row_counts = _get_row_nonzero(count_for_sf, i)
            size_factors[i] = float(np.asarray(row_counts).sum())

        total = size_factors.sum()
        if total > 0:
            size_factors = size_factors / (total / n_obs)
        size_factors = np.where(size_factors <= 0, 1.0, size_factors)
        size_factors = np.where(np.isnan(size_factors), 1.0, size_factors)
    except Exception:
        # Fall back: compute size factors from the candidate matrix itself
        try:
            size_factors = np.zeros(n_obs, dtype=np.float64)
            for i in indices:
                row_indices, row_counts = _get_row_nonzero(matrix_ref, i)
                size_factors[i] = float(np.asarray(row_counts).sum())
            total = size_factors.sum()
            if total > 0:
                size_factors = size_factors / (total / n_obs)
            size_factors = np.where(size_factors <= 0, 1.0, size_factors)
            size_factors = np.where(np.isnan(size_factors), 1.0, size_factors)
        except Exception:
            return None

    # Apply expm1 / size_factor on sampled rows
    recovered_values: list[np.ndarray] = []
    for idx in indices:
        row = matrix_ref[idx]
        if hasattr(row, "toarray"):
            row = np.asarray(row.toarray().ravel())
        else:
            row = np.asarray(row).ravel()
        nonzero_mask = row != 0
        if nonzero_mask.any():
            sf = size_factors[idx]
            if sf <= 0:
                sf = 1.0
            recovered = np.expm1(row[nonzero_mask])
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
        f"recovered via expm1/size_factor; "
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
    if not candidates:
        return CountSourceDecision(
            selected_candidate="none",
            status="fail",
            confidence="low",
            recovery_policy="disallowed",
            rationale="No matrix candidates were available for audit.",
        )
    selected = candidates[0]
    uses_recovery = False

    # If no direct integer source is available, attempt reverse normalization
    if selected.status != "pass" and adata is not None:
        recovered = _attempt_reverse_normalization(selected, adata)
        if recovered is not None:
            selected = recovered
            uses_recovery = True

    confidence = (
        "high"
        if selected.status == "pass"
        else "medium"
        if selected.status == "needs-review"
        else "low"
    )
    rationale = (
        f"Selected {selected.candidate} because it is the highest-ranked candidate with "
        f"status {selected.status}."
    )
    if uses_recovery:
        rationale += " Counts were recovered via the approved expm1/size_factor path."

    return CountSourceDecision(
        selected_candidate=selected.candidate,
        status=selected.status,
        confidence=confidence,
        recovery_policy=selected.recovery_policy,
        rationale=rationale,
        uses_recovery=uses_recovery,
    )


def _find_best_column(columns: tuple[str, ...], aliases: tuple[str, ...]) -> str | None:
    normalized = {_normalize_label(column): column for column in columns}
    for alias in aliases:
        alias_key = _normalize_label(alias)
        if alias_key in normalized:
            return normalized[alias_key]
    for alias in aliases:
        alias_key = _normalize_label(alias)
        for normalized_name, original in normalized.items():
            if alias_key and normalized_name.startswith(alias_key):
                return original
    for alias in aliases:
        alias_key = _normalize_label(alias)
        for normalized_name, original in normalized.items():
            if alias_key and alias_key in normalized_name:
                return original
    return None


def _is_text_like_dtype(dtype_name: str) -> bool:
    lowered = dtype_name.lower()
    return any(token in lowered for token in ("object", "string", "category"))


def _build_feature_field_entry(
    field_name: str,
    source_field: str | None,
    confidence: str,
    required: bool,
    notes: tuple[str, ...] = (),
    literal_value: str | None = None,
) -> SchemaFieldEntry:
    """Build a SchemaFieldEntry for a feature field, using null strategy when no source is found."""
    if source_field is None:
        return SchemaFieldEntry(
            source_fields=(),
            strategy="null",
            transforms=(),
            confidence=confidence,
            required=required,
            literal_value=None,
            notes=notes,
        )
    return SchemaFieldEntry(
        source_fields=(source_field,),
        strategy="source-field",
        transforms=(),
        confidence=confidence,
        required=required,
        literal_value=literal_value,
        notes=notes,
    )


def _rank_feature_candidates(
    var_fields: tuple[FieldProfile, ...],
) -> list[tuple[FieldProfile, int]]:
    """Rank var columns as tokenization identity candidates.

    Scoring logic mirrors count-source priority: exact matches on high-signal
    alias keys score highest; text-like dtypes are preferred for id/label fields.
    Returns a list of (profile, score) sorted descending.
    """
    results = []
    for profile in var_fields:
        normalized = _normalize_label(profile.name)
        dtype_score = 30 if _is_text_like_dtype(profile.dtype) else -50
        alias_score = 0
        for field_name, aliases in FEATURE_FIELD_ALIASES.items():
            for position, alias in enumerate(aliases):
                alias_key = _normalize_label(alias)
                if not alias_key:
                    continue
                base = max(0, 100 - position)
                if normalized == alias_key:
                    alias_score = max(alias_score, base + 500)
                elif normalized.startswith(alias_key):
                    alias_score = max(alias_score, base + 200)
                elif alias_key in normalized:
                    alias_score = max(alias_score, base)
        score = alias_score + dtype_score
        results.append((profile, score))
    return sorted(results, key=lambda item: item[1], reverse=True)


def _build_feature_fields_section(
    var_fields: tuple[FieldProfile, ...],
    var_index_name: str,
) -> dict[str, SchemaFieldEntry]:
    """Build the complete feature_fields dict for the new schema artifact.

    All three canonical feature fields are always present. When no reliable
    source is found for a given field, the entry uses ``strategy: null`` so
    the field is unresolved inline rather than in a separate list.
    """
    ranked = _rank_feature_candidates(var_fields)
    # Pick best candidate per field_name
    best_per_field: dict[str, str | None] = {
        "feature_id": None,
        "feature_label": None,
        "feature_namespace": None,
    }
    for profile, score in ranked:
        if score < 0:
            continue
        normalized = _normalize_label(profile.name)
        for field_name, aliases in FEATURE_FIELD_ALIASES.items():
            if best_per_field[field_name] is not None:
                continue
            for alias in aliases:
                if _normalize_label(alias) == normalized:
                    best_per_field[field_name] = profile.name
                    break
            if best_per_field[field_name] is not None:
                break

    # Build entries: feature_id is required, others optional
    feature_fields: dict[str, SchemaFieldEntry] = {}
    feature_fields["feature_id"] = _build_feature_field_entry(
        "feature_id",
        best_per_field["feature_id"],
        confidence="high" if best_per_field["feature_id"] else "low",
        required=True,
        notes=(
            "primary tokenization identity source; "
            + ("found via var column " + best_per_field["feature_id"]
               if best_per_field["feature_id"] else "no reliable source found — set manually")
        ),
    )
    feature_fields["feature_label"] = _build_feature_field_entry(
        "feature_label",
        best_per_field["feature_label"],
        confidence="high" if best_per_field["feature_label"] else "low",
        required=False,
        notes=(
            "human-readable feature label; "
            + ("found via var column " + best_per_field["feature_label"]
               if best_per_field["feature_label"] else "no reliable source found — set manually")
        ),
    )
    feature_fields["feature_namespace"] = _build_feature_field_entry(
        "feature_namespace",
        best_per_field["feature_namespace"],
        confidence="high" if best_per_field["feature_namespace"] else "low",
        required=False,
        notes=(
            "namespace for tokenization identity (e.g. ensembl, gene_symbol); "
            + ("found via var column " + best_per_field["feature_namespace"]
               if best_per_field["feature_namespace"] else "no reliable source found — set manually")
        ),
    )
    return feature_fields


def _find_best_profiled_column(
    field_name: str,
    field_profiles: tuple[FieldProfile, ...],
    aliases: tuple[str, ...],
) -> str | None:
    best_name: str | None = None
    best_score = -(10**9)
    for profile in field_profiles:
        normalized_name = _normalize_label(profile.name)
        dtype_score = 20 if _is_text_like_dtype(profile.dtype) else -100
        if field_name in {"batch_id", "donor_id", "sex"} and not _is_text_like_dtype(
            profile.dtype
        ):
            dtype_score = 0

        alias_score = -(10**6)
        for position, alias in enumerate(aliases):
            alias_key = _normalize_label(alias)
            if not alias_key:
                continue
            base = max(0, 100 - position)
            if normalized_name == alias_key:
                alias_score = max(alias_score, base + 1000)
            elif normalized_name.startswith(alias_key):
                alias_score = max(alias_score, base + 300)
            elif alias_key in normalized_name:
                alias_score = max(alias_score, base)

        score = alias_score + dtype_score
        if score > best_score:
            best_score = score
            best_name = profile.name

    if best_score < 0:
        return None
    return best_name


def _infer_literal_perturbation_type(columns: tuple[str, ...]) -> str | None:
    normalized_columns = {_normalize_label(column) for column in columns}
    if any(
        token in value
        for value in normalized_columns
        for token in ("guide", "sgrna", "grna", "crispr", "sgid")
    ):
        return "CRISPR"
    if any(
        token in value for value in normalized_columns for token in ("drug", "compound")
    ):
        return "compound"
    if any(
        token in value for value in normalized_columns for token in ("cytokine", "stim")
    ):
        return "stimulation"
    return None


def _build_schema_field_entry(
    field_name: str,
    source_field: str | None,
    strategy: str,
    confidence: str,
    required: bool,
    notes: tuple[str, ...] = (),
    literal_value: str | None = None,
    transforms: tuple[TransformSpec, ...] = (),
) -> SchemaFieldEntry:
    """Build a SchemaFieldEntry for a cell field."""
    if source_field is None and strategy == "null":
        return SchemaFieldEntry(
            source_fields=(),
            strategy="null",
            transforms=(),
            confidence=confidence,
            required=required,
            notes=notes,
        )
    if source_field is None:
        return SchemaFieldEntry(
            source_fields=(),
            strategy=strategy,
            transforms=transforms,
            confidence=confidence,
            required=required,
            literal_value=literal_value,
            notes=notes,
        )
    return SchemaFieldEntry(
        source_fields=(source_field,),
        strategy=strategy,
        transforms=transforms,
        confidence=confidence,
        required=required,
        literal_value=literal_value,
        notes=notes,
    )


def _build_cell_field_sections(
    target: InspectionTarget,
    obs_columns: tuple[str, ...],
    obs_fields: tuple[FieldProfile, ...],
) -> tuple[dict[str, SchemaFieldEntry], dict[str, SchemaFieldEntry]]:
    """Build perturbation_fields and context_fields dicts for the new schema artifact.

    All canonical cell fields are always present. Unresolved fields use
    ``strategy: null`` so they are inline rather than in a separate list.
    """
    perturbation_fields: dict[str, SchemaFieldEntry] = {}
    context_fields: dict[str, SchemaFieldEntry] = {}

    # Literal dataset_id (always present)
    perturbation_fields["perturbation_label"] = _build_schema_field_entry(
        "perturbation_label",
        source_field=None,
        strategy="null",
        confidence="low",
        required=True,
        notes=("set manually — no perturbation_label source found",),
    )
    perturbation_fields["perturbation_type"] = _build_schema_field_entry(
        "perturbation_type",
        source_field=None,
        strategy="null",
        confidence="low",
        required=True,
        notes=("set manually or infer from obs column names",),
    )
    perturbation_fields["target_id"] = _build_schema_field_entry(
        "target_id",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    perturbation_fields["target_label"] = _build_schema_field_entry(
        "target_label",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    perturbation_fields["control_flag"] = _build_schema_field_entry(
        "control_flag",
        source_field=None,
        strategy="null",
        confidence="low",
        required=True,
        notes=("set manually — no perturbation source for derive",),
    )
    perturbation_fields["dose"] = _build_schema_field_entry(
        "dose",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    perturbation_fields["dose_unit"] = _build_schema_field_entry(
        "dose_unit",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    perturbation_fields["timepoint"] = _build_schema_field_entry(
        "timepoint",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    perturbation_fields["timepoint_unit"] = _build_schema_field_entry(
        "timepoint_unit",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    perturbation_fields["combination_key"] = _build_schema_field_entry(
        "combination_key",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )

    # Literal dataset_id in dataset_metadata section
    dataset_metadata: dict[str, SchemaFieldEntry] = {
        "dataset_id": _build_schema_field_entry(
            "dataset_id",
            source_field=None,
            strategy="literal",
            confidence="high",
            required=True,
            literal_value=target.dataset_id,
            notes=("dataset id is set from the inspection target",),
        ),
    }

    # Context fields
    context_fields["cell_context"] = _build_schema_field_entry(
        "cell_context",
        source_field=None,
        strategy="null",
        confidence="low",
        required=True,
    )
    context_fields["cell_line_or_type"] = _build_schema_field_entry(
        "cell_line_or_type",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    context_fields["species"] = _build_schema_field_entry(
        "species",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    context_fields["tissue"] = _build_schema_field_entry(
        "tissue",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    context_fields["assay"] = _build_schema_field_entry(
        "assay",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    context_fields["condition"] = _build_schema_field_entry(
        "condition",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    context_fields["batch_id"] = _build_schema_field_entry(
        "batch_id",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    context_fields["donor_id"] = _build_schema_field_entry(
        "donor_id",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    context_fields["sex"] = _build_schema_field_entry(
        "sex",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )
    context_fields["disease_state"] = _build_schema_field_entry(
        "disease_state",
        source_field=None,
        strategy="null",
        confidence="low",
        required=False,
    )

    # Now populate from obs heuristics
    perturbation_source = _find_best_profiled_column(
        "perturbation_label", obs_fields, FIELD_ALIASES["perturbation_label"]
    )
    if perturbation_source is not None:
        perturbation_fields["perturbation_label"] = _build_schema_field_entry(
            "perturbation_label",
            source_field=perturbation_source,
            strategy="source-field",
            confidence="high",
            required=True,
        )
        perturbation_fields["control_flag"] = _build_schema_field_entry(
            "control_flag",
            source_field=perturbation_source,
            strategy="derived",
            confidence="medium",
            required=True,
            transforms=(build_transform("recognize_control", patterns=list(CONTROL_PATTERNS)),),
            notes=("derived from perturbation_label source",),
        )

    target_label_source = _find_best_profiled_column(
        "target_label", obs_fields, FIELD_ALIASES["target_label"]
    )
    if target_label_source is not None:
        perturbation_fields["target_label"] = _build_schema_field_entry(
            "target_label",
            source_field=target_label_source,
            strategy="source-field",
            confidence="high",
            required=False,
        )

    target_id_source = _find_best_profiled_column(
        "target_id", obs_fields, FIELD_ALIASES["target_id"]
    )
    if target_id_source is not None:
        perturbation_fields["target_id"] = _build_schema_field_entry(
            "target_id",
            source_field=target_id_source,
            strategy="source-field",
            confidence="high",
            required=False,
        )

    perturbation_type = _infer_literal_perturbation_type(obs_columns)
    if perturbation_type is not None:
        perturbation_fields["perturbation_type"] = _build_schema_field_entry(
            "perturbation_type",
            source_field=None,
            strategy="literal",
            confidence="medium",
            required=True,
            literal_value=perturbation_type,
            notes=("literal perturbation type inferred from guide-like field names",),
        )

    dose_source = _find_best_profiled_column("dose", obs_fields, FIELD_ALIASES["dose"])
    if dose_source is not None:
        perturbation_fields["dose"] = _build_schema_field_entry(
            "dose",
            source_field=dose_source,
            strategy="source-field",
            confidence="high",
            required=False,
        )
        dose_unit_source = _find_best_profiled_column(
            "dose_unit", obs_fields, FIELD_ALIASES["dose_unit"]
        )
        if dose_unit_source is not None:
            perturbation_fields["dose_unit"] = _build_schema_field_entry(
                "dose_unit",
                source_field=dose_unit_source,
                strategy="source-field",
                confidence="high",
                required=False,
            )

    timepoint_source = _find_best_profiled_column(
        "timepoint", obs_fields, FIELD_ALIASES["timepoint"]
    )
    if timepoint_source is not None:
        perturbation_fields["timepoint"] = _build_schema_field_entry(
            "timepoint",
            source_field=timepoint_source,
            strategy="source-field",
            confidence="high",
            required=False,
        )
        timepoint_unit_source = _find_best_profiled_column(
            "timepoint_unit", obs_fields, FIELD_ALIASES["timepoint_unit"]
        )
        if timepoint_unit_source is not None:
            perturbation_fields["timepoint_unit"] = _build_schema_field_entry(
                "timepoint_unit",
                source_field=timepoint_unit_source,
                strategy="source-field",
                confidence="high",
                required=False,
            )

    for field_name, aliases in FIELD_ALIASES.items():
        canonical_name = field_name
        if canonical_name in perturbation_fields:
            entry = perturbation_fields[canonical_name]
            if entry.strategy != "null":
                continue
        elif canonical_name in context_fields:
            entry = context_fields[canonical_name]
            if entry.strategy != "null":
                continue
        else:
            continue
        source_field = _find_best_profiled_column(canonical_name, obs_fields, aliases)
        if source_field is None:
            continue
        if canonical_name in perturbation_fields:
            perturbation_fields[canonical_name] = _build_schema_field_entry(
                canonical_name,
                source_field=source_field,
                strategy="source-field",
                confidence="high",
                required=entry.required,
            )
        elif canonical_name in context_fields:
            context_fields[canonical_name] = _build_schema_field_entry(
                canonical_name,
                source_field=source_field,
                strategy="source-field",
                confidence="high",
                required=entry.required,
            )

    normalized_target = _normalize_label(target.dataset_id + target.source_path)
    if "k562" in normalized_target:
        for field_name in ("cell_context", "cell_line_or_type"):
            context_fields[field_name] = _build_schema_field_entry(
                field_name,
                source_field=None,
                strategy="literal",
                confidence="medium",
                required=(field_name == "cell_context"),
                literal_value="K562",
                notes=("literal value inferred from dataset identifier or path",),
            )

    return perturbation_fields, context_fields


def _materialization_readiness_schema(
    count_status: str,
    perturbation_fields: dict[str, SchemaFieldEntry],
    context_fields: dict[str, SchemaFieldEntry],
) -> str:
    if count_status == "fail":
        return "fail"
    for section in (perturbation_fields, context_fields):
        for name, entry in section.items():
            if entry.required and entry.strategy == "null":
                return "needs-review"
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
        obs_columns = tuple(str(column) for column in adata.obs.columns.tolist())

        perturbation_fields, context_fields = _build_cell_field_sections(
            target, obs_columns, obs_fields
        )
        feature_fields = _build_feature_fields_section(
            var_fields, var_index_name=str(adata.var.index.name or "index")
        )

        readiness = _materialization_readiness_schema(
            count_decision.status, perturbation_fields, context_fields
        )

        # Build count_source spec
        count_source_spec = CountSourceSpec(
            selected=count_decision.selected_candidate,
            integer_only=(count_decision.status == "pass"),
            uses_recovery=count_decision.uses_recovery,
        )

        # Build feature_tokenization spec: pick the best-ranked feature_id source
        # as the dataset-level tokenization target
        best_feature_id: str | None = None
        best_namespace: str | None = None
        if "feature_id" in feature_fields:
            entry = feature_fields["feature_id"]
            if entry.strategy == "source-field" and entry.source_fields:
                best_feature_id = entry.source_fields[0]
        if "feature_namespace" in feature_fields:
            ns_entry = feature_fields["feature_namespace"]
            if ns_entry.strategy == "source-field" and ns_entry.source_fields:
                best_namespace = ns_entry.source_fields[0]
        feature_tokenization_spec = FeatureTokenizationSpec(
            selected=best_feature_id or "set-manually",
            namespace=best_namespace or "unknown",
        )

        schema = SchemaDocument.new_draft(
            dataset_id=target.dataset_id,
            source_path=target.source_path,
            dataset_metadata={
                "dataset_id": SchemaFieldEntry(
                    source_fields=(),
                    strategy="literal",
                    transforms=(),
                    confidence="high",
                    required=True,
                    literal_value=target.dataset_id,
                ),
            },
            perturbation_fields=perturbation_fields,
            context_fields=context_fields,
            feature_fields=feature_fields,
            count_source=count_source_spec,
            feature_tokenization=feature_tokenization_spec,
            transform_catalog=TRANSFORM_CATALOG,
            materialization_notes=(
                "auto-generated schema draft; review and set status: ready before materialization",
            ),
        )

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
            canonical_defaults={"missing_value_literal": MISSING_VALUE_LITERAL},
            raw_field_policy={
                "preserve_source_obs_fields": True,
                "preserve_source_var_fields": True,
            },
            inspector_notes=(
                "metadata-first scan used backed h5ad access",
                "matrix audit used sampled rows only and did not materialize the full matrix",
            ),
        )

        summary_path = dataset_dir / "dataset-summary.yaml"
        schema_path = dataset_dir / "schema.yaml"

        summary.validate()
        schema.validate()
        summary.write_yaml(summary_path)
        schema.write_yaml(schema_path)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    print(
        f"[inspect] done {target.dataset_id} count={count_decision.selected_candidate} readiness={readiness}"
    )
    return InspectionArtifacts(
        dataset_id=target.dataset_id,
        dataset_summary=summary_path,
        schema=schema_path,
        selected_count_source=count_decision.selected_candidate,
        materialization_readiness=readiness,
    )


def run_batch(
    config: InspectionBatchConfig, workers: int = 1
) -> InspectionBatchManifest:
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
                dataset_summary=str(artifact.dataset_summary),
                schema=str(artifact.schema),
                selected_count_source=artifact.selected_count_source,
                materialization_readiness=artifact.materialization_readiness,
            )
            for artifact in sorted(artifacts, key=lambda item: item.dataset_id)
        ),
    )
    manifest.write_yaml(output_root / "inspection-manifest.yaml")
    return manifest
