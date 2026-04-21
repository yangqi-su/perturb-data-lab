"""Schema readiness validation.

Provides explicit validation gates so only schemas with status: ready and all
required fields resolved (strategy != "null") can proceed to materialization.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadinessViolation:
    field: str
    section: str
    reason: str


@dataclass(frozen=True)
class SchemaReadinessResult:
    valid: bool
    violations: tuple[ReadinessViolation, ...]

    def raise_if_not_ready(self) -> None:
        """Raise ValueError if the schema is not ready."""
        if not self.valid:
            lines = ["Schema readiness check failed:"]
            for v in self.violations:
                lines.append(f"  - [{v.section}.{v.field}] {v.reason}")
            raise ValueError("\n".join(lines))


def validate_schema_readiness(
    schema_path: str,
) -> SchemaReadinessResult:
    """Validate that a schema is ready for materialization.

    Checks:
    1. status == "ready"
    2. All required cell fields (perturbation + context) have strategy != "null"
    3. All required feature fields have strategy != "null"

    Parameters
    ----------
    schema_path : str
        Path to the schema YAML file.

    Returns
    -------
    SchemaReadinessResult
        valid=True if all gates pass; otherwise valid=False with a list of violations.

    Raises
    ------
    FileNotFoundError
        If the schema file does not exist.
    ValueError
        If the schema cannot be parsed or is structurally invalid.
    """
    from pathlib import Path

    from ..inspectors.models import SchemaDocument

    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"schema file not found: {schema_path}")

    schema = SchemaDocument.from_yaml_file(path)
    return _validate_schema_readiness_object(schema)


def _validate_schema_readiness_object(
    schema: "SchemaDocument",
) -> SchemaReadinessResult:
    """Validate a SchemaDocument object directly (for use without filesystem)."""
    violations: list[ReadinessViolation] = []

    # Gate 1: status must be "ready"
    if schema.status != "ready":
        violations.append(
            ReadinessViolation(
                field="status",
                section="schema",
                reason=f"status is '{schema.status}', expected 'ready'",
            )
        )
        # If status is not ready, skip field-level checks — schema is not usable
        return SchemaReadinessResult(
            valid=False,
            violations=tuple(violations),
        )

    # Gate 2: required perturbation fields
    for name, entry in schema.perturbation_fields.items():
        if entry.required and entry.strategy == "null":
            violations.append(
                ReadinessViolation(
                    field=name,
                    section="perturbation_fields",
                    reason="required field has strategy 'null'",
                )
            )

    # Gate 3: required context fields
    for name, entry in schema.context_fields.items():
        if entry.required and entry.strategy == "null":
            violations.append(
                ReadinessViolation(
                    field=name,
                    section="context_fields",
                    reason="required field has strategy 'null'",
                )
            )

    # Gate 4: required feature fields
    for name, entry in schema.feature_fields.items():
        if entry.required and entry.strategy == "null":
            violations.append(
                ReadinessViolation(
                    field=name,
                    section="feature_fields",
                    reason="required field has strategy 'null'",
                )
            )

    # Gate 5: feature_tokenization namespace must be a real value, not "unknown" or "set-manually"
    # A dataset cannot join any corpus unless its namespace is explicitly set
    ns = schema.feature_tokenization.namespace
    if ns in ("unknown", "set-manually", ""):
        violations.append(
            ReadinessViolation(
                field="feature_tokenization.namespace",
                section="feature_fields",
                reason=(
                    f"namespace is '{ns}' — "
                    "this dataset cannot join a corpus with this namespace. "
                    "Set feature_tokenization.namespace to the actual namespace "
                    "(e.g., ensembl, gene_symbol) before materialization."
                ),
            )
        )

    return SchemaReadinessResult(
        valid=len(violations) == 0,
        violations=tuple(violations),
    )
