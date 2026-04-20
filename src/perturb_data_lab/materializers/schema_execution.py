"""Phase 5 schema execution engine for cell rows and feature tables.

This module implements row-wise schema resolution:
- literal/source-field/derived/null strategy execution
- multi-source `_` default join rule
- null-to-NA normalization
- transform application order:
    1. collect source values from the row
    2. handle missing/null values
    3. apply default join if multiple sources and no explicit transform overrides it
    4. apply transforms
    5. normalize final null-like output to NA
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..contracts import MISSING_VALUE_LITERAL
from ..inspectors.models import SchemaDocument, SchemaFieldEntry, TransformSpec


# ---------------------------------------------------------------------------
# Execution result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaExecutionResult:
    """Result of resolving one schema field entry for one row.

    Attributes
    ----------
    value : str
        The resolved canonical value, or NA if unresolved.
    was_resolved : bool
        True when at least one source contributed a non-null value.
    source_values : tuple[str, ...]
        The raw source values that were collected before transform application.
    """

    value: str
    was_resolved: bool
    source_values: tuple[str, ...]


# ---------------------------------------------------------------------------
# Transform dispatch
# ---------------------------------------------------------------------------


def _apply_transform(
    value: str | bool,
    transform: TransformSpec,
) -> str | bool:
    """Apply a single transform spec to a value.

    Returns the transformed value. Boolean values are passed through directly
    for ``recognize_control``; all other transforms expect string input.
    """
    name = transform.name
    args = transform.args

    if name == "strip_prefix":
        prefix = str(args.get("prefix", ""))
        val = str(value)
        return val[len(prefix):] if val.startswith(prefix) else val

    if name == "strip_suffix":
        suffix = str(args.get("suffix", ""))
        val = str(value)
        return val[: -len(suffix)] if suffix and val.endswith(suffix) else val

    if name == "regex_sub":
        pattern = str(args.get("pattern", ""))
        replacement = str(args.get("replacement", ""))
        return re.sub(pattern, replacement, str(value))

    if name == "normalize_case":
        mode = str(args.get("mode", "lower"))
        val = str(value)
        if mode == "lower":
            return val.lower()
        if mode == "upper":
            return val.upper()
        if mode == "title":
            return val.title()
        return val

    if name == "recognize_control":
        # recognize_control returns bool; pass through so caller can convert
        patterns = tuple(args.get("patterns", []))
        return re.search(
            "|".join(patterns), str(value), flags=re.IGNORECASE
        ) is not None

    if name == "join_with_plus":
        # join_with_plus expects tuple[str, ...]; this path should not normally
        # be hit when applying a transform to a single value
        return value

    # Unknown transform: pass through
    return value


def _apply_transform_chain(
    value: str | bool,
    transforms: tuple[TransformSpec, ...],
) -> str:
    """Apply an ordered chain of transforms, then normalize to string/NA."""
    current: str | bool = value
    for transform in transforms:
        current = _apply_transform(current, transform)

    # Normalize to string
    if isinstance(current, bool):
        # recognize_control result → "true"/"false"
        return "true" if current else "false"

    if current is None:
        return MISSING_VALUE_LITERAL

    return str(current)


# ---------------------------------------------------------------------------
# Multi-source join
# ---------------------------------------------------------------------------


def _join_sources(
    values: tuple[str, ...],
    transforms: tuple[TransformSpec, ...],
) -> str:
    """Join multiple source values.

    If any transform in the chain is ``join_with_plus``, use `+` as the separator.
    Otherwise use `_` as the default separator (plan decision: multi-source `_` join).

    Returns the joined string, or NA if all values are empty/null.
    """
    # Check for explicit join_with_plus override
    for transform in transforms:
        if transform.name == "join_with_plus":
            # Explicit join_with_plus should have been handled upstream as a
            # special derivation case, but handle it defensively here
            joined = "+".join(v for v in values if v)
            return joined if joined else MISSING_VALUE_LITERAL

    # Default: join with underscore
    joined = "_".join(v for v in values if v)
    return joined if joined else MISSING_VALUE_LITERAL


# ---------------------------------------------------------------------------
# Null-like detection
# ---------------------------------------------------------------------------


_NA_LITERALS = {"", "na", "n/a", "none", "null", "nan", ".", "-"}
_NA_LITERALS_RAW: frozenset[str] = frozenset()


def _is_null_like(value: str | None) -> bool:
    """Return True for values that should be treated as null/NA."""
    if value is None:
        return True
    lowered = str(value).strip().lower()
    if not lowered:
        return True
    return lowered in _NA_LITERALS


# ---------------------------------------------------------------------------
# Core field resolution
# ---------------------------------------------------------------------------


def resolve_field_entry(
    entry: SchemaFieldEntry,
    source_row: dict[str, Any],
) -> SchemaExecutionResult:
    """Resolve one schema field entry against one data row.

    Execution order:
    1. collect source values from the row (for source-field/derived strategies)
    2. handle missing/null values
    3. apply default join if multiple sources and no explicit transform overrides it
    4. apply transforms
    5. normalize final null-like output to NA

    Parameters
    ----------
    entry : SchemaFieldEntry
        The field entry from the schema document.
    source_row : dict[str, Any]
        A dictionary mapping column names to row values (e.g. adata.obs.iloc[i]
        as a dict, or a single var row as a dict).

    Returns
    -------
    SchemaExecutionResult
        The resolved value, resolution status, and raw source values.
    """
    strategy = entry.strategy

    # --- Null strategy: always return NA ---
    if strategy == "null":
        return SchemaExecutionResult(
            value=MISSING_VALUE_LITERAL,
            was_resolved=False,
            source_values=(),
        )

    # --- Literal strategy: use literal_value directly ---
    if strategy == "literal":
        raw = entry.literal_value
        if raw is None or _is_null_like(raw):
            return SchemaExecutionResult(
                value=MISSING_VALUE_LITERAL,
                was_resolved=False,
                source_values=(),
            )
        result = _apply_transform_chain(raw, entry.transforms)
        return SchemaExecutionResult(
            value=result,
            was_resolved=True,
            source_values=(str(raw),),
        )

    # --- Source-field and derived strategies ---
    if strategy not in {"source-field", "derived"}:
        # Unknown strategy: treat as null
        return SchemaExecutionResult(
            value=MISSING_VALUE_LITERAL,
            was_resolved=False,
            source_values=(),
        )

    # Collect source values from the row
    source_values: list[str] = []
    for field_name in entry.source_fields:
        raw_val = source_row.get(field_name, None)
        if raw_val is None:
            continue
        str_val = str(raw_val).strip()
        if _is_null_like(str_val):
            continue
        source_values.append(str_val)

    # Handle derived strategy specially: recognize_control
    if strategy == "derived":
        # Check if this is the recognize_control transform pattern
        has_recognize_control = any(
            t.name == "recognize_control" for t in entry.transforms
        )
        if has_recognize_control and source_values:
            # Apply recognize_control to the first available source value
            control_patterns = ()
            for t in entry.transforms:
                if t.name == "recognize_control":
                    control_patterns = tuple(t.args.get("patterns", []))
            if control_patterns:
                first_val = source_values[0]
                matched = any(
                    re.search(p, first_val, flags=re.IGNORECASE)
                    for p in control_patterns
                )
                # Convert bool to string ("true"/"false")
                str_result = "true" if matched else "false"
                # Apply remaining transforms (excluding recognize_control)
                remaining_transforms = tuple(
                    t for t in entry.transforms if t.name != "recognize_control"
                )
                if remaining_transforms:
                    str_result = _apply_transform_chain(str_result, remaining_transforms)
                return SchemaExecutionResult(
                    value=str_result,
                    was_resolved=True,
                    source_values=tuple(source_values),
                )

    # No source values collected → NA
    if not source_values:
        return SchemaExecutionResult(
            value=MISSING_VALUE_LITERAL,
            was_resolved=False,
            source_values=(),
        )

    # Apply multi-source join
    joined = _join_sources(tuple(source_values), entry.transforms)

    # Apply transforms to the joined result
    # (join_with_plus should have been handled as a special derivation case;
    #  here we just apply the remaining transforms)
    result = _apply_transform_chain(joined, entry.transforms)

    return SchemaExecutionResult(
        value=result,
        was_resolved=True,
        source_values=tuple(source_values),
    )


# ---------------------------------------------------------------------------
# Row-level resolvers
# ---------------------------------------------------------------------------


def resolve_cell_row(
    schema: SchemaDocument,
    obs_row: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve all canonical cell fields for one cell row.

    Returns two dicts: (canonical_perturbation, canonical_context), each mapping
    field names to resolved string values (or NA).

    Parameters
    ----------
    schema : SchemaDocument
        The reviewed schema document.
    obs_row : dict[str, Any]
        A dictionary of obs column names to values for this cell.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        (perturbation dict, context dict) with all canonical fields resolved.
    """
    perturbation: dict[str, str] = {}
    for field_name, entry in schema.perturbation_fields.items():
        result = resolve_field_entry(entry, obs_row)
        perturbation[field_name] = result.value

    context: dict[str, str] = {}
    for field_name, entry in schema.context_fields.items():
        result = resolve_field_entry(entry, obs_row)
        context[field_name] = result.value

    return perturbation, context


def resolve_feature_row(
    schema: SchemaDocument,
    var_row: dict[str, Any],
) -> dict[str, str]:
    """Resolve all canonical feature fields for one feature row.

    Returns a dict mapping field names to resolved string values (or NA).

    Parameters
    ----------
    schema : SchemaDocument
        The reviewed schema document.
    var_row : dict[str, Any]
        A dictionary of var column names to values for this feature.

    Returns
    -------
    dict[str, str]
        Feature field values resolved from the var row.
    """
    feature: dict[str, str] = {}
    for field_name, entry in schema.feature_fields.items():
        result = resolve_field_entry(entry, var_row)
        feature[field_name] = result.value
    return feature


# ---------------------------------------------------------------------------
# Batch resolvers (for materialization)
# ---------------------------------------------------------------------------


def resolve_all_cell_rows(
    schema: SchemaDocument,
    obs_dataframe: Any,
) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, str], ...]]:
    """Resolve canonical cell fields for all cells.

    Parameters
    ----------
    schema : SchemaDocument
        The reviewed schema document.
    obs_dataframe : pandas.DataFrame or dict-of-dicts
        The obs data to resolve against.

    Returns
    -------
    tuple[tuple[dict[str, str], ...], tuple[dict[str, str], ...]]
        Tuple of (perturbation tuples, context tuples), each tuple having one
        dict per cell in original order.
    """
    import pandas as pd

    perturbations: list[dict[str, str]] = []
    contexts: list[dict[str, str]] = []

    if isinstance(obs_dataframe, pd.DataFrame):
        # Iterate row by row
        for _, row in obs_dataframe.iterrows():
            pert, ctx = resolve_cell_row(schema, row.to_dict())
            perturbations.append(pert)
            contexts.append(ctx)
    else:
        # Dict-of-dicts or similar
        for key, row in obs_dataframe.items():
            pert, ctx = resolve_cell_row(schema, dict(row))
            perturbations.append(pert)
            contexts.append(ctx)

    return tuple(perturbations), tuple(contexts)


def resolve_all_feature_rows(
    schema: SchemaDocument,
    var_dataframe: Any,
) -> tuple[dict[str, str], ...]:
    """Resolve canonical feature fields for all features.

    Parameters
    ----------
    schema : SchemaDocument
        The reviewed schema document.
    var_dataframe : pandas.DataFrame or dict-of-dicts
        The var data to resolve against.

    Returns
    -------
    tuple[dict[str, str], ...]
        Tuple of feature dicts, one per feature in original order.
    """
    import pandas as pd

    features: list[dict[str, str]] = []

    if isinstance(var_dataframe, pd.DataFrame):
        for _, row in var_dataframe.iterrows():
            feat = resolve_feature_row(schema, row.to_dict())
            features.append(feat)
    else:
        for key, row in var_dataframe.items():
            feat = resolve_feature_row(schema, dict(row))
            features.append(feat)

    return tuple(features)
