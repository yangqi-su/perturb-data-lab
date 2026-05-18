from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..contracts import MISSING_VALUE_LITERAL


@dataclass(frozen=True)
class TransformCatalogEntry:
    name: str
    description: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransformCatalogEntry":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class TransformSpec:
    name: str
    args: dict[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransformSpec":
        return cls(name=str(data["name"]), args=dict(data.get("args", {})))


TRANSFORM_CATALOG = (
    TransformCatalogEntry(
        name="map_control_labels",
        description="Map configured control-like labels to a canonical output such as `ctrl`.",
    ),
    TransformCatalogEntry(
        name="strip_whitespace",
        description="Trim leading and trailing whitespace from a raw value.",
    ),
    TransformCatalogEntry(
        name="replace_empty_with_null",
        description="Convert empty or whitespace-only values into the missing-value literal.",
    ),
    TransformCatalogEntry(
        name="strip_prefix",
        description="Remove a known prefix from a raw value.",
    ),
    TransformCatalogEntry(
        name="strip_suffix",
        description="Remove a known suffix from a raw value.",
    ),
    TransformCatalogEntry(
        name="strip_guide_suffix",
        description="Remove common guide suffixes such as `_sg1` or `-1` from perturbation labels.",
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
        name="normalize_dose_unit",
        description="Normalize an extracted or standalone dose unit to a canonical short form.",
    ),
    TransformCatalogEntry(
        name="timepoint_parse",
        description="Extract the numeric time value from a mixed string such as 24h.",
    ),
    TransformCatalogEntry(
        name="timepoint_unit",
        description="Extract the time unit (h, d, min) from a mixed time string.",
    ),
    TransformCatalogEntry(
        name="normalize_time_unit",
        description="Normalize an extracted or standalone time unit to a canonical short form.",
    ),
    TransformCatalogEntry(
        name="strip_ensembl_version",
        description="Remove trailing version suffixes like `.18` from Ensembl identifiers.",
    ),
    TransformCatalogEntry(
        name="normalize_boolean",
        description="Normalize common boolean spellings such as yes/no or 1/0 to true/false.",
    ),
)

_DECIMAL_RE = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_DOSE_RE = re.compile(
    rf"^\s*({_DECIMAL_RE})\s*([A-Za-zµμ%]+(?:/[A-Za-zµμ%]+)?)\s*$",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    rf"^\s*({_DECIMAL_RE})\s*([A-Za-z]+)\s*$",
    re.IGNORECASE,
)
_GUIDE_SUFFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(?:[_\-\s](?:sg|guide|grna)\d+)$"),
    re.compile(r"(?i)(?:[_\-]g\d+)$"),
    re.compile(r"(?i)(?:[_\-]\d+)$"),
)
_BOOLEAN_TRUE: frozenset[str] = frozenset({"1", "t", "true", "y", "yes"})
_BOOLEAN_FALSE: frozenset[str] = frozenset({"0", "f", "false", "n", "no"})
_DOSE_UNIT_ALIASES: dict[str, str] = {
    "%": "%",
    "nm": "nm",
    "nanomolar": "nm",
    "nanomol": "nm",
    "um": "um",
    "micromolar": "um",
    "mm": "mm",
    "millimolar": "mm",
    "mg/kg": "mg/kg",
    "ug/kg": "ug/kg",
}
_TIME_UNIT_ALIASES: dict[str, str] = {
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "d": "d",
    "day": "d",
    "days": "d",
    "m": "m",
    "min": "m",
    "mins": "m",
    "minute": "m",
    "minutes": "m",
}


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _normalize_micro(value: str) -> str:
    return value.replace("μ", "u").replace("µ", "u")


def _normalize_for_match(value: Any, *, case_sensitive: bool, strip_whitespace: bool) -> str:
    text = _as_text(value)
    if strip_whitespace:
        text = text.strip()
    text = _normalize_micro(text)
    return text if case_sensitive else text.lower()


def _coerce_candidates(candidates: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(candidates, str):
        return (candidates,)
    return tuple(str(candidate) for candidate in candidates)


def _extract_number_and_unit(value: str, pattern: re.Pattern[str]) -> tuple[str | None, str | None]:
    match = pattern.match(_as_text(value).strip())
    if not match:
        return None, None
    return match.group(1), match.group(2)


def map_control_labels(
    value: str,
    candidates: Sequence[str] | str,
    output: str = "ctrl",
    case_sensitive: bool = False,
    strip_whitespace: bool = True,
) -> str:
    normalized_candidates = {
        _normalize_for_match(
            candidate,
            case_sensitive=case_sensitive,
            strip_whitespace=strip_whitespace,
        )
        for candidate in _coerce_candidates(candidates)
    }
    probe = _normalize_for_match(
        value,
        case_sensitive=case_sensitive,
        strip_whitespace=strip_whitespace,
    )
    if probe in normalized_candidates:
        return output
    return _as_text(value)


def strip_whitespace(value: str) -> str:
    return _as_text(value).strip()


def replace_empty_with_null(value: str, strip_whitespace: bool = True) -> str:
    text = _as_text(value)
    probe = text.strip() if strip_whitespace else text
    return MISSING_VALUE_LITERAL if probe == "" else text


def strip_prefix(value: str, prefix: str) -> str:
    text = _as_text(value)
    return text[len(prefix) :] if text.startswith(prefix) else text


def strip_suffix(value: str, suffix: str) -> str:
    text = _as_text(value)
    return text[: -len(suffix)] if suffix and text.endswith(suffix) else text


def strip_guide_suffix(value: str, pattern: str | None = None) -> str:
    text = _as_text(value)
    patterns = (re.compile(pattern),) if pattern else _GUIDE_SUFFIX_PATTERNS
    for compiled in patterns:
        stripped, count = compiled.subn("", text, count=1)
        if count:
            return stripped.rstrip("_- ")
    return text


def regex_sub(value: str, pattern: str, replacement: str) -> str:
    return re.sub(pattern, replacement, _as_text(value))


def normalize_case(value: str, mode: str) -> str:
    value = _as_text(value)
    if mode == "lower":
        return value.lower()
    if mode == "upper":
        return value.upper()
    if mode == "title":
        return value.title()
    raise ValueError(f"unsupported case mode: {mode}")


def recognize_control(value: str, patterns: tuple[str, ...]) -> bool:
    lowered = _as_text(value).lower()
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
    parts = _as_text(value).split(delimiter)
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
    text = _as_text(value)
    return mapping.get(text, text)


def dose_parse(value: str) -> str:
    """Extract a numeric dose value from a mixed string such as ``100nM`` or ``1.5 μM``.

    Handles common molecular biology dose notations:
    - nanomolar / micromolar concentration (``nM``, ``uM``, ``μM``, ``mM``)
    - mass-per-body-weight (``mg/kg``, ``μg/kg``)
    - percentage (``%``) when used as a concentration

    Returns the extracted numeric portion as a string, preserving decimal form.
    Returns NA if no recognisable numeric dose is found.
    """
    value = _as_text(value).strip()
    number, unit = _extract_number_and_unit(value, _DOSE_RE)
    if number is not None and unit is not None:
        return number
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
    value = _as_text(value).strip()
    _, unit = _extract_number_and_unit(value, _DOSE_RE)
    if unit is not None:
        return normalize_dose_unit(unit)
    return MISSING_VALUE_LITERAL


def normalize_dose_unit(value: str) -> str:
    text = _as_text(value).strip()
    if not text:
        return MISSING_VALUE_LITERAL
    _, extracted = _extract_number_and_unit(text, _DOSE_RE)
    unit = extracted if extracted is not None else text
    normalized = _DOSE_UNIT_ALIASES.get(_normalize_micro(unit).lower())
    return normalized if normalized is not None else text


def timepoint_parse(value: str) -> str:
    """Extract a numeric time value from a mixed string such as ``24h`` or ``48 hr``.

    Handles common time notations: hours (``h``, ``hr``, ``hrs``), days
    (``d``, ``day``, ``days``), minutes (``m``, ``min``, ``mins``).

    Returns the numeric portion as a string.  Returns NA if no recognisable
    time token is found.
    """
    value = _as_text(value).strip()
    number, unit = _extract_number_and_unit(value, _TIME_RE)
    if number is not None and unit is not None:
        return number
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
    value = _as_text(value).strip()
    _, unit = _extract_number_and_unit(value, _TIME_RE)
    if unit is not None:
        return normalize_time_unit(unit)
    return MISSING_VALUE_LITERAL


def normalize_time_unit(value: str) -> str:
    text = _as_text(value).strip()
    if not text:
        return MISSING_VALUE_LITERAL
    _, extracted = _extract_number_and_unit(text, _TIME_RE)
    unit = extracted if extracted is not None else text
    normalized = _TIME_UNIT_ALIASES.get(unit.lower())
    return normalized if normalized is not None else text


def strip_ensembl_version(value: str) -> str:
    text = _as_text(value)
    if not text.upper().startswith("ENS"):
        return text
    return re.sub(r"\.\d+$", "", text)


def normalize_boolean(
    value: str,
    true_output: str = "true",
    false_output: str = "false",
    true_values: Sequence[str] | None = None,
    false_values: Sequence[str] | None = None,
    case_sensitive: bool = False,
    strip_whitespace: bool = True,
) -> str:
    probe = _normalize_for_match(
        value,
        case_sensitive=case_sensitive,
        strip_whitespace=strip_whitespace,
    )
    true_set = {
        _normalize_for_match(
            item,
            case_sensitive=case_sensitive,
            strip_whitespace=strip_whitespace,
        )
        for item in (true_values or tuple(_BOOLEAN_TRUE))
    }
    false_set = {
        _normalize_for_match(
            item,
            case_sensitive=case_sensitive,
            strip_whitespace=strip_whitespace,
        )
        for item in (false_values or tuple(_BOOLEAN_FALSE))
    }
    if probe in true_set:
        return true_output
    if probe in false_set:
        return false_output
    return _as_text(value)


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


TRANSFORM_FUNCTIONS: dict[str, Callable[..., str]] = {
    "map_control_labels": map_control_labels,
    "strip_whitespace": strip_whitespace,
    "replace_empty_with_null": replace_empty_with_null,
    "strip_prefix": strip_prefix,
    "strip_suffix": strip_suffix,
    "strip_guide_suffix": strip_guide_suffix,
    "regex_sub": regex_sub,
    "normalize_case": normalize_case,
    "map_values": map_values,
    "split_on_delimiter": split_on_delimiter,
    "dose_parse": dose_parse,
    "dose_unit": dose_unit,
    "timepoint_parse": timepoint_parse,
    "timepoint_unit": timepoint_unit,
    "normalize_time_unit": normalize_time_unit,
    "normalize_dose_unit": normalize_dose_unit,
    "strip_ensembl_version": strip_ensembl_version,
    "normalize_boolean": normalize_boolean,
}


def get_transform(name: str) -> Callable[..., str] | None:
    return TRANSFORM_FUNCTIONS.get(name)
