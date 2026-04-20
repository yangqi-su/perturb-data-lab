from __future__ import annotations

import re
from dataclasses import dataclass

from .models import TransformCatalogEntry, TransformSpec


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
