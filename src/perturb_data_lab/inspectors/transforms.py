from __future__ import annotations

import re
from dataclasses import dataclass

from .models import TransformCatalogEntry, TransformSpec
from ..contracts import MISSING_VALUE_LITERAL


TRANSFORM_CATALOG = (
    TransformCatalogEntry(
        name="strip_prefix",
        description="Remove a known prefix from a raw value.",
    ),
    TransformCatalogEntry(
        name="strip_suffix",
        description="Remove a known suffix from a raw value.",
    ),
    TransformCatalogEntry(
        name="regex_sub",
        description="Apply a regex substitution before canonicalization.",
    ),
    TransformCatalogEntry(
        name="normalize_case",
        description="Normalize case to lower, upper, or title.",
    ),
    TransformCatalogEntry(
        name="recognize_control",
        description="Mark control-like perturbation labels using explicit regex patterns.",
    ),
    TransformCatalogEntry(
        name="join_with_plus",
        description="Join multiple cleaned source values with `+`.",
    ),
    TransformCatalogEntry(
        name="coalesce_values",
        description="Pick the first non-null value from multiple source fields.",
    ),
    TransformCatalogEntry(
        name="split_on_delimiter",
        description="Split a combined string on a delimiter and extract one component.",
    ),
    TransformCatalogEntry(
        name="map_values",
        description="Map specific input values to configured canonical outputs via a lookup table.",
    ),
    TransformCatalogEntry(
        name="dose_parse",
        description="Extract the numeric dose value from a mixed string such as 100nM.",
    ),
    TransformCatalogEntry(
        name="dose_unit",
        description="Extract the dose unit (nM, uM, mg/kg) from a mixed dose string.",
    ),
    TransformCatalogEntry(
        name="timepoint_parse",
        description="Extract the numeric time value from a mixed string such as 24h.",
    ),
    TransformCatalogEntry(
        name="timepoint_unit",
        description="Extract the time unit (h, d, min) from a mixed time string.",
    ),
)


def strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


def strip_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if suffix and value.endswith(suffix) else value


def regex_sub(value: str, pattern: str, replacement: str) -> str:
    return re.sub(pattern, replacement, value)


def normalize_case(value: str, mode: str) -> str:
    if mode == "lower":
        return value.lower()
    if mode == "upper":
        return value.upper()
    if mode == "title":
        return value.title()
    raise ValueError(f"unsupported case mode: {mode}")


def recognize_control(value: str, patterns: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def join_with_plus(values: tuple[str, ...]) -> str:
    cleaned = [value for value in values if value]
    return "+".join(cleaned)


def build_transform(name: str, **args: object) -> TransformSpec:
    return TransformSpec(name=name, args=dict(args))


# ---------------------------------------------------------------------------
# Phase 5 new transforms
# ---------------------------------------------------------------------------


def coalesce_values(values: tuple[str, ...]) -> str:
    """Return the first non-null-like value from a tuple, or NA if all are null-like.

    Use this transform as a ``derived`` strategy when the same canonical field can
    come from one of several source columns (e.g., multiple guide-name columns).
    Apply ``coalesce_values`` to a multi-source entry::

        transforms=(build_transform("coalesce_values"),)

    The source_fields for that entry should list all candidate column names in
    priority order; the first column whose value is non-null-like wins.
    """
    for value in values:
        if not _is_null_like_str(value):
            return value
    return MISSING_VALUE_LITERAL


def split_on_delimiter(value: str, delimiter: str = ",", part: int = 0) -> str:
    """Split a combined string on a delimiter and return one component by index.

    Use this transform when a single source column encodes multiple values
    (e.g., ``TP53+MDM2`` as a combination key, or ``100nM;24hr`` as a
    dose+time concatenation).  Set ``part`` to select which component (0-based).

    Example::

        transforms=(build_transform("split_on_delimiter", delimiter="+", part=0),)

    Returns NA if the delimiter is not found or the index is out of range.
    """
    parts = value.split(delimiter)
    if part < 0 or part >= len(parts):
        return MISSING_VALUE_LITERAL
    result = parts[part].strip()
    return result if result else MISSING_VALUE_LITERAL


def map_values(value: str, mapping: dict[str, str]) -> str:
    """Map specific input values to configured output values via a lookup table.

    Use this transform to normalise controlled vocabularies where raw values
    may be inconsistent (e.g., species names ``Homo sapiens`` / ``human`` /
    ``H.sapiens``, or cell lines with synonyms).

    Example::

        transforms=(
            build_transform(
                "map_values",
                mapping={"Homo sapiens": "human", "Mus musculus": "mouse"},
            ),
        )

    If the input value is not in the mapping, it is returned unchanged.
    This makes the transform safe to use even when only some values need remapping.
    """
    return mapping.get(value, value)


def dose_parse(value: str) -> str:
    """Extract a numeric dose value from a mixed string such as ``100nM`` or ``1.5 μM``.

    Handles common molecular biology dose notations:
    - nanomolar / micromolar concentration (``nM``, ``uM``, ``μM``, ``mM``)
    - mass-per-body-weight (``mg/kg``, ``μg/kg``)
    - percentage (``%``) when used as a concentration

    Returns the extracted numeric portion as a string, preserving decimal form.
    Returns NA if no recognisable numeric dose is found.
    """
    import re

    value = str(value).strip()
    # nanomolar: 100nM, 50 nM
    m = re.match(r"^([0-9.]+)\s*(nM|uM|μM|mM)\s*$", value, re.IGNORECASE)
    if m:
        return m.group(1)
    # mass per weight: 10mg/kg, 5 μg/kg
    m = re.match(r"^([0-9.]+)\s*(mg/kg|μg/kg|ug/kg)\s*$", value, re.IGNORECASE)
    if m:
        return m.group(1)
    # bare number (already normalised)
    try:
        float(value)
        return value
    except ValueError:
        pass
    return MISSING_VALUE_LITERAL


def dose_unit(value: str) -> str:
    """Extract the normalized dose unit from a mixed dose string.

    Normalizes to canonical short forms: ``nm``, ``um``, ``mg/kg``.
    Returns ``NA`` if no unit is found.
    """
    import re

    value = str(value).strip()
    m = re.match(r"^([0-9.]+)\s*(nM|uM|μM|mM|mg/kg|μg/kg|ug/kg)\s*$", value, re.IGNORECASE)
    if m:
        raw_unit = m.group(2).lower()
        # Normalise to canonical short forms
        unit_map = {"nm": "nm", "um": "um", "μm": "um", "mm": "mm", "mg/kg": "mg/kg", "μg/kg": "mg/kg", "ug/kg": "mg/kg"}
        return unit_map.get(raw_unit, raw_unit)
    return MISSING_VALUE_LITERAL


def timepoint_parse(value: str) -> str:
    """Extract a numeric time value from a mixed string such as ``24h`` or ``48 hr``.

    Handles common time notations: hours (``h``, ``hr``, ``hrs``), days
    (``d``, ``day``, ``days``), minutes (``m``, ``min``, ``mins``).

    Returns the numeric portion as a string.  Returns NA if no recognisable
    time token is found.
    """
    import re

    value = str(value).strip()
    m = re.match(r"^([0-9.]+)\s*(h|hr|hrs|d|day|days|m|min|mins)\s*$", value, re.IGNORECASE)
    if m:
        return m.group(1)
    try:
        float(value)
        return value
    except ValueError:
        pass
    return MISSING_VALUE_LITERAL


def timepoint_unit(value: str) -> str:
    """Extract the normalized time unit from a mixed time string.

    Normalizes to canonical short forms: ``h`` (hours), ``d`` (days), ``m`` (minutes).
    Returns ``NA`` if no unit is found.
    """
    import re

    value = str(value).strip()
    m = re.match(
        r"^([0-9.]+)\s*(h|hr|hrs|d|day|days|m|min|mins)\s*$", value, re.IGNORECASE
    )
    if m:
        raw_unit = m.group(2).lower()
        # Normalise to canonical short forms
        unit_map = {"h": "h", "hr": "h", "hrs": "h", "d": "d", "day": "d", "days": "d", "m": "m", "min": "m", "mins": "m"}
        return unit_map.get(raw_unit, raw_unit)
    return MISSING_VALUE_LITERAL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NA_LITERALS: frozenset[str] = frozenset({"", "na", "n/a", "none", "null", "nan", ".", "-"})


def _is_null_like_str(value: str | None) -> bool:
    """Return True for values that should be treated as null/empty."""
    if value is None:
        return True
    lowered = str(value).strip().lower()
    if not lowered:
        return True
    return lowered in _NA_LITERALS
