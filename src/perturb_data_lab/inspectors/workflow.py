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
    DatasetIdentity,
    DatasetSummaryDocument,
    FieldMapping,
    FieldProfile,
    InspectionBatchConfig,
    InspectionBatchManifest,
    InspectionBatchRecord,
    InspectionTarget,
    SchemaPatchDocument,
    SchemaProposalDocument,
    StructureSummary,
)
from .transforms import TRANSFORM_CATALOG, build_transform


DEFAULT_FIELD_SAMPLE_SIZE = 5
DEFAULT_MATRIX_SAMPLE_ROWS = 32
CONTROL_PATTERNS = (r"^ntc", r"non[-_ ]?target", r"control", r"mock", r"wt")

CANONICAL_FIELDS = tuple(
    field.name for field in BLUEPRINT.perturbation_fields + BLUEPRINT.context_fields
)
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
    schema_proposal: Path
    schema_patch: Path
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


def _choose_count_source(
    candidates: tuple[CountSourceCandidate, ...],
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
    return CountSourceDecision(
        selected_candidate=selected.candidate,
        status=selected.status,
        confidence=confidence,
        recovery_policy=selected.recovery_policy,
        rationale=rationale,
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


def _build_field_mappings(
    target: InspectionTarget,
    obs_columns: tuple[str, ...],
    obs_fields: tuple[FieldProfile, ...],
) -> dict[str, FieldMapping]:
    mappings: dict[str, FieldMapping] = {
        "dataset_id": FieldMapping(
            source_fields=(),
            strategy="literal",
            transforms=(),
            confidence="high",
            literal_value=target.dataset_id,
            notes=("dataset id is set from the inspection target",),
        ),
        "dataset_release": FieldMapping(
            source_fields=(),
            strategy="literal",
            transforms=(),
            confidence="high",
            literal_value=target.source_release,
            notes=("dataset release is set from the inspection target",),
        ),
    }

    perturbation_source = _find_best_profiled_column(
        "perturbation_label", obs_fields, FIELD_ALIASES["perturbation_label"]
    )
    if perturbation_source is not None:
        mappings["perturbation_label"] = FieldMapping(
            source_fields=(perturbation_source,),
            strategy="source-field",
            transforms=(),
            confidence="high",
        )
        mappings["target_label"] = FieldMapping(
            source_fields=(perturbation_source,),
            strategy="source-field",
            transforms=(),
            confidence="medium",
            notes=("fallback target label reuses the perturbation label source",),
        )
        mappings["control_flag"] = FieldMapping(
            source_fields=(perturbation_source,),
            strategy="derived",
            transforms=(
                build_transform("recognize_control", patterns=list(CONTROL_PATTERNS)),
            ),
            confidence="medium",
        )

    target_label_source = _find_best_profiled_column(
        "target_label", obs_fields, FIELD_ALIASES["target_label"]
    )
    if target_label_source is not None:
        mappings["target_label"] = FieldMapping(
            source_fields=(target_label_source,),
            strategy="source-field",
            transforms=(),
            confidence="high",
        )

    perturbation_type = _infer_literal_perturbation_type(obs_columns)
    if perturbation_type is not None:
        mappings["perturbation_type"] = FieldMapping(
            source_fields=(),
            strategy="literal",
            transforms=(),
            confidence="medium",
            literal_value=perturbation_type,
            notes=("literal perturbation type inferred from guide-like field names",),
        )

    for field_name, aliases in FIELD_ALIASES.items():
        if field_name in mappings:
            continue
        source_field = _find_best_profiled_column(field_name, obs_fields, aliases)
        if source_field is None:
            continue
        mappings[field_name] = FieldMapping(
            source_fields=(source_field,),
            strategy="source-field",
            transforms=(),
            confidence="high",
        )

    normalized_target = _normalize_label(target.dataset_id + target.source_path)
    if "k562" in normalized_target:
        for field_name in ("cell_context", "cell_line_or_type"):
            mappings.setdefault(
                field_name,
                FieldMapping(
                    source_fields=(),
                    strategy="literal",
                    transforms=(),
                    confidence="medium",
                    literal_value="K562",
                    notes=("literal value inferred from dataset identifier or path",),
                ),
            )

    return mappings


def _materialization_readiness(
    count_source: CountSourceDecision,
    field_mappings: dict[str, FieldMapping],
) -> str:
    if count_source.status == "fail":
        return "fail"
    unresolved_required = REQUIRED_CANONICAL_FIELDS.difference(field_mappings)
    if unresolved_required:
        return "needs-review"
    if count_source.status == "needs-review":
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

        candidates = [_audit_matrix_candidate(".X", adata.X, adata.n_obs, adata.n_vars)]
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
        count_decision = _choose_count_source(ranked_candidates)
        obs_columns = tuple(str(column) for column in adata.obs.columns.tolist())
        field_mappings = _build_field_mappings(target, obs_columns, obs_fields)
        readiness = _materialization_readiness(count_decision, field_mappings)
        unresolved_fields = tuple(
            field_name
            for field_name in CANONICAL_FIELDS
            if field_name not in field_mappings
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
        proposal = SchemaProposalDocument(
            kind="schema-proposal",
            contract_version=CONTRACT_VERSION,
            dataset_id=target.dataset_id,
            source_path=target.source_path,
            summary_artifact="dataset-summary.yaml",
            canonical_defaults={"missing_value_literal": MISSING_VALUE_LITERAL},
            count_source_decision=count_decision,
            transform_catalog=TRANSFORM_CATALOG,
            field_mappings=field_mappings,
            unresolved_fields=unresolved_fields,
            materialization_readiness=readiness,
        )
        patch = SchemaPatchDocument(
            kind="schema-patch",
            contract_version=CONTRACT_VERSION,
            dataset_id=target.dataset_id,
            review_status="pending",
            summary_artifact="dataset-summary.yaml",
            proposal_artifact="schema-proposal.yaml",
            unresolved_fields=unresolved_fields,
            patches=(),
            notes=(
                "fill this file only when review needs to override the generated proposal",
            ),
        )

        summary_path = dataset_dir / "dataset-summary.yaml"
        proposal_path = dataset_dir / "schema-proposal.yaml"
        patch_path = dataset_dir / "schema-patch.yaml"

        summary.validate()
        proposal.validate()
        patch.validate()
        summary.write_yaml(summary_path)
        proposal.write_yaml(proposal_path)
        patch.write_yaml(patch_path)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    print(
        f"[inspect] done {target.dataset_id} count={count_decision.selected_candidate} readiness={readiness}"
    )
    return InspectionArtifacts(
        dataset_id=target.dataset_id,
        dataset_summary=summary_path,
        schema_proposal=proposal_path,
        schema_patch=patch_path,
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
                schema_proposal=str(artifact.schema_proposal),
                schema_patch=str(artifact.schema_patch),
                selected_count_source=artifact.selected_count_source,
                materialization_readiness=artifact.materialization_readiness,
            )
            for artifact in sorted(artifacts, key=lambda item: item.dataset_id)
        ),
    )
    manifest.write_yaml(output_root / "inspection-manifest.yaml")
    return manifest
