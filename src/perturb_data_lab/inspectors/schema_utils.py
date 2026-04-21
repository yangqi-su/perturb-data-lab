"""Phase 5 schema preview and explanation utilities.

Provides structured human-readable explanations of schema field resolution status,
preview of canonical field values on sample rows, and namespace compatibility
checks for multi-dataset corpus formation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import MISSING_VALUE_LITERAL
from .models import SchemaDocument, SchemaFieldEntry


# ---------------------------------------------------------------------------
# Explain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldExplanation:
    """One-line explanation of a schema field's current resolution state."""

    field: str
    section: str  # perturbation_fields | context_fields | feature_fields | dataset_metadata
    strategy: str  # source-field | literal | derived | null
    required: bool
    confidence: str
    source_fields: tuple[str, ...]
    literal_value: str | None
    transform_names: tuple[str, ...]
    status: str  # resolved | unresolved | null_strategy
    issue: str | None  # human-readable issue when not fully resolved
    resolution_hint: str | None  # what to do to fix

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "section": self.section,
            "strategy": self.strategy,
            "required": self.required,
            "confidence": self.confidence,
            "source_fields": self.source_fields,
            "literal_value": self.literal_value,
            "transform_names": self.transform_names,
            "status": self.status,
            "issue": self.issue,
            "resolution_hint": self.resolution_hint,
        }


@dataclass(frozen=True)
class SchemaExplanation:
    """Complete schema field-by-field explanation with summary."""

    dataset_id: str
    status: str
    field_count: int
    resolved_count: int
    unresolved_count: int
    null_required_count: int
    field_explanations: tuple[FieldExplanation, ...]
    namespace: str
    namespace_status: str  # valid | ambiguous | unset
    namespace_issue: str | None
    readiness_blockers: tuple[str, ...]  # short list of critical blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "field_count": self.field_count,
            "resolved_count": self.resolved_count,
            "unresolved_count": self.unresolved_count,
            "null_required_count": self.null_required_count,
            "field_explanations": [fe.as_dict() for fe in self.field_explanations],
            "namespace": self.namespace,
            "namespace_status": self.namespace_status,
            "namespace_issue": self.namespace_issue,
            "readiness_blockers": self.readiness_blockers,
        }


# ---------------------------------------------------------------------------
# Core explain function
# ---------------------------------------------------------------------------


def explain_schema(schema: SchemaDocument) -> SchemaExplanation:
    """Produce a structured explanation of every field in a schema document.

    Use this to audit a draft schema before marking it ``status: ready``,
    or to understand why a schema is blocked from materialization.

    Parameters
    ----------
    schema : SchemaDocument
        The schema to explain.

    Returns
    -------
    SchemaExplanation
        Per-field explanations plus summary counters and critical blockers.
    """
    explanations: list[FieldExplanation] = []
    unresolved_count = 0
    null_required_count = 0

    for section_name, section in [
        ("perturbation_fields", schema.perturbation_fields),
        ("context_fields", schema.context_fields),
        ("feature_fields", schema.feature_fields),
        ("dataset_metadata", schema.dataset_metadata),
    ]:
        for field_name, entry in section.items():
            exp = _explain_field_entry(field_name, section_name, entry)
            explanations.append(exp)
            if exp.status in ("unresolved", "null_strategy"):
                unresolved_count += 1
            if exp.status == "null_strategy" and exp.required:
                null_required_count += 1

    # Namespace assessment
    ns = schema.feature_tokenization.namespace
    ns_status: str
    ns_issue: str | None
    if ns in ("unknown", "set-manually", ""):
        ns_status = "unset"
        ns_issue = (
            f"feature_tokenization.namespace is '{ns}' — "
            "this must be set to a real namespace (e.g., ensembl, gene_symbol) "
            "before the dataset can join a corpus."
        )
    else:
        ns_status = "valid"
        ns_issue = None

    # Readiness blockers
    blockers: list[str] = []
    if schema.status != "ready":
        blockers.append(f"schema status is '{schema.status}', not 'ready'")
    if null_required_count > 0:
        blockers.append(
            f"{null_required_count} required field(s) have strategy 'null' — set source or literal values"
        )
    if ns_status == "unset":
        blockers.append(f"feature namespace is unset ('{ns}') — corpus append will reject this dataset")

    return SchemaExplanation(
        dataset_id=schema.dataset_id,
        status=schema.status,
        field_count=len(explanations),
        resolved_count=len(explanations) - unresolved_count,
        unresolved_count=unresolved_count,
        null_required_count=null_required_count,
        field_explanations=tuple(explanations),
        namespace=ns,
        namespace_status=ns_status,
        namespace_issue=ns_issue,
        readiness_blockers=tuple(blockers),
    )


def _explain_field_entry(
    field_name: str,
    section: str,
    entry: SchemaFieldEntry,
) -> FieldExplanation:
    strategy = entry.strategy
    required = entry.required
    transform_names = tuple(t.name for t in entry.transforms)

    if strategy == "null":
        status = "null_strategy"
        issue = (
            "no source configured — this field will always resolve to NA"
            if required
            else "no source configured (optional field)"
        )
        resolution_hint = (
            "set source_fields to an existing obs column name, or set a literal_value"
            if required
            else "no action needed for optional fields"
        )
    elif strategy == "source-field":
        status = "resolved"
        issue = None
        resolution_hint = None
    elif strategy == "derived":
        status = "resolved"
        issue = None if transform_names else (
            "derived strategy with no transforms — result will be NA"
        )
        resolution_hint = (
            None
            if transform_names
            else "add at least one transform (e.g., recognize_control, coalesce_values)"
        )
    elif strategy == "literal":
        status = "resolved"
        issue = None
        resolution_hint = None
    else:
        status = "unresolved"
        issue = f"unknown strategy '{strategy}'"
        resolution_hint = "use one of: source-field, literal, derived, null"

    return FieldExplanation(
        field=field_name,
        section=section,
        strategy=strategy,
        required=required,
        confidence=entry.confidence,
        source_fields=entry.source_fields,
        literal_value=entry.literal_value,
        transform_names=transform_names,
        status=status,
        issue=issue,
        resolution_hint=resolution_hint,
    )


# ---------------------------------------------------------------------------
# Preview utilities
# ---------------------------------------------------------------------------


def preview_field_resolution(
    entry: SchemaFieldEntry,
    sample_values: dict[str, str],
) -> str:
    """Preview what a single field entry would resolve to for given sample values.

    Parameters
    ----------
    entry : SchemaFieldEntry
        The field entry to evaluate.
    sample_values : dict[str, str]
        A dict mapping source field names to raw string values.

    Returns
    -------
    str
        The resolved canonical value, or NA if unresolved.
    """
    # Re-use resolve_field_entry from schema_execution
    from ..materializers.schema_execution import resolve_field_entry

    result = resolve_field_entry(entry, sample_values)
    return result.value


def preview_cell_row(
    schema: SchemaDocument,
    obs_row: dict[str, Any],
) -> dict[str, str]:
    """Preview resolved canonical values for a sample cell row.

    Parameters
    ----------
    schema : SchemaDocument
        The schema to use for resolution.
    obs_row : dict[str, Any]
        A dictionary of obs column names to raw values for one cell.

    Returns
    -------
    dict[str, str]
        Keys are canonical field names; values are resolved strings (or NA).
        Format: ``{"perturbation": {...}, "context": {...}}``.
    """
    from ..materializers.schema_execution import resolve_cell_row

    pert, ctx = resolve_cell_row(schema, obs_row)
    return {"perturbation": pert, "context": ctx}


def preview_feature_row(
    schema: SchemaDocument,
    var_row: dict[str, Any],
) -> dict[str, str]:
    """Preview resolved canonical values for a sample feature row.

    Parameters
    ----------
    schema : SchemaDocument
        The schema to use for resolution.
    var_row : dict[str, Any]
        A dictionary of var column names to raw values for one feature.

    Returns
    -------
    dict[str, str]
        Canonical feature field names → resolved string values.
    """
    from ..materializers.schema_execution import resolve_feature_row

    return resolve_feature_row(schema, var_row)


# ---------------------------------------------------------------------------
# Namespace compatibility utility
# ---------------------------------------------------------------------------


def check_namespace_compatibility(
    dataset_schema: SchemaDocument,
    corpus_namespace: str,
) -> tuple[bool, str]:
    """Check whether a dataset's feature namespace is compatible with a corpus namespace.

    Per the Phase 1 append compatibility contract, namespace mismatch is a hard
    rejection condition before any I/O or tokenization.

    Parameters
    ----------
    dataset_schema : SchemaDocument
        The dataset's schema (must be status: ready).
    corpus_namespace : str
        The target corpus namespace (e.g., "ensembl", "gene_symbol").

    Returns
    -------
    tuple[bool, str]
        (True, "") when compatible; (False, reason) when incompatible.
    """
    if dataset_schema.status != "ready":
        return False, (
            f"dataset schema status is '{dataset_schema.status}', not 'ready' — "
            "cannot assess namespace compatibility"
        )

    ds_ns = dataset_schema.feature_tokenization.namespace
    if ds_ns in ("unknown", "set-manually", ""):
        return False, (
            f"dataset namespace is '{ds_ns}' — "
            f"dataset cannot join a corpus with namespace '{corpus_namespace}' "
            "until its feature_tokenization.namespace is set to a real value"
        )

    if ds_ns != corpus_namespace:
        return False, (
            f"dataset namespace '{ds_ns}' != corpus namespace '{corpus_namespace}' — "
            "namespace mismatch is a hard rejection condition"
        )

    return True, ""
