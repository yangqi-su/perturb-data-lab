"""Canonicalization contract and schema dataclass definitions.

Defines the immutable canonical obs/var contracts, the corpus-level
``CanonicalVocab`` structure, and the per-dataset ``CanonicalizationSchema``
YAML format used by the canonicalization runner (Phase 2).

This module is a **definition layer** only.  No loader or materializer code
is modified by adding types here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL

# ---------------------------------------------------------------------------
# Canonical obs field names (must-have)
#
# Sorted alphabetically for readability.  Extensible columns may appear
# alongside these in produced parquet files; missing values are filled with
# ``MISSING_VALUE_LITERAL`` ("NA").
# ---------------------------------------------------------------------------

CANONICAL_OBS_MUST_HAVE: tuple[str, ...] = (
    "assay",
    "batch_id",
    "cell_context",
    "cell_id",
    "cell_line_or_type",
    "condition",
    "dataset_id",
    "dataset_index",
    "disease_state",
    "donor_id",
    "dose",
    "dose_unit",
    "global_row_index",
    "local_row_index",
    "perturb_label",
    "perturb_type",
    "sex",
    "size_factor",
    "species",
    "timepoint",
    "timepoint_unit",
    "tissue",
)

# ---------------------------------------------------------------------------
# Canonical var field names (must-have)
# ---------------------------------------------------------------------------

CANONICAL_VAR_MUST_HAVE: tuple[str, ...] = (
    "origin_index",
    "gene_id",
    "canonical_gene_id",
    "global_id",
)


# ---------------------------------------------------------------------------
# CanonicalObsSchema — runtime contract checker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalObsSchema:
    """Immutable contract for canonical obs (cell metadata).

    Describes which columns **must** be present and what dtype they carry.
    This is a definition object, not a validator that processes data.
    """

    required_columns: tuple[str, ...] = field(default=CANONICAL_OBS_MUST_HAVE)
    optional_columns: tuple[str, ...] = field(default=())
    fallback_value: str = field(default=MISSING_VALUE_LITERAL)

    @property
    def all_columns(self) -> tuple[str, ...]:
        """All expected columns in insertion order (required first)."""
        return self.required_columns + self.optional_columns

    def missing_required(self, available: set[str]) -> tuple[str, ...]:
        """Return required column names absent from *available*."""
        return tuple(c for c in self.required_columns if c not in available)


# ---------------------------------------------------------------------------
# CanonicalVarSchema — runtime contract checker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalVarSchema:
    """Immutable contract for canonical var (gene/feature metadata)."""

    required_columns: tuple[str, ...] = field(default=CANONICAL_VAR_MUST_HAVE)
    optional_columns: tuple[str, ...] = field(default=())
    fallback_value: str = field(default=MISSING_VALUE_LITERAL)

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.required_columns + self.optional_columns

    def missing_required(self, available: set[str]) -> tuple[str, ...]:
        return tuple(c for c in self.required_columns if c not in available)


# ---------------------------------------------------------------------------
# CanonicalVocab — corpus-global vocabulary
# ---------------------------------------------------------------------------

@dataclass
class CanonicalVocab:
    """Corpus-global deduplicated vocabulary across all canonicalized datasets.

    Produced by the canonicalization runner after all per-dataset schemas
    have been applied.  Contains sorted, unique values for each canonical
    category that appears across the corpus.

    This is written as ``canonical-vocab.yaml`` at the corpus root.
    """

    obs_categories: dict[str, list[str]] = field(default_factory=dict)
    """Category name → sorted unique string values across all datasets."""

    var_categories: dict[str, list[str]] = field(default_factory=dict)
    """Category name → sorted unique string values across all datasets."""

    gene_id_mappings: dict[str, str] = field(default_factory=dict)
    """``gene_id → canonical_gene_id`` map across the full corpus."""

    global_vocab_size: int = 0
    """Total number of unique ``canonical_gene_id`` values."""


# ---------------------------------------------------------------------------
# CanonicalizationSchema — per-dataset transform configuration
# ---------------------------------------------------------------------------

# Mapping strategy enum values recognised by the runner.
_OBS_STRATEGIES: frozenset[str] = frozenset({
    "source-field",
    "literal",
    "passthrough",
    "row-index",
    "null",
})
_VAR_STRATEGIES: frozenset[str] = frozenset({
    "source-field",
    "literal",
    "passthrough",
    "gene-mapping",
    "auto",
    "null",
})


@dataclass(frozen=True)
class ObsColumnMapping:
    """Specifies how one canonical obs column is produced from raw data."""

    canonical_name: str
    """Name of the output canonical column (e.g. ``perturb_label``)."""

    strategy: str
    """One of ``source-field``, ``literal``, ``passthrough``, ``row-index``, ``null``."""

    source_column: str | None = None
    """Raw obs column to read from when strategy is ``source-field``."""

    literal_value: str | None = None
    """Static value when strategy is ``literal``."""

    transforms: tuple[TransformRule, ...] = ()
    """Ordered list of value transforms applied after source extraction."""

    fallback: str = MISSING_VALUE_LITERAL
    """Value used when the source column is missing or produces null."""

    def __post_init__(self):
        if self.strategy not in _OBS_STRATEGIES:
            raise ValueError(
                f"Unknown obs mapping strategy {self.strategy!r}; "
                f"allowed: {sorted(_OBS_STRATEGIES)}"
            )


@dataclass(frozen=True)
class VarColumnMapping:
    """Specifies how one canonical var column is produced from raw data."""

    canonical_name: str
    strategy: str
    source_column: str | None = None
    literal_value: str | None = None
    transforms: tuple[TransformRule, ...] = ()
    fallback: str = MISSING_VALUE_LITERAL

    # Gene-mapping specific config (used when strategy == "gene-mapping")
    enabled: bool = False
    engine: str = "gget"
    source_namespace: str = "gene_symbol"
    target_namespace: str = "gene_symbol"
    mapping_file: str | None = None

    def __post_init__(self):
        if self.strategy not in _VAR_STRATEGIES:
            raise ValueError(
                f"Unknown var mapping strategy {self.strategy!r}; "
                f"allowed: {sorted(_VAR_STRATEGIES)}"
            )


@dataclass(frozen=True)
class ExtensibleColumn:
    """Declares an optional column to pass through from raw to canonical."""

    raw_source_column: str
    canonical_name: str | None = None
    """Output name.  Defaults to ``raw_source_column``."""

    def __post_init__(self):
        if self.canonical_name is None:
            object.__setattr__(self, "canonical_name", self.raw_source_column)


@dataclass(frozen=True)
class TransformRule:
    """A single named transform with its keyword arguments.

    References transform functions defined in ``inspectors/transforms.py``.
    """

    name: str
    """Function name in the transform catalog (e.g. ``strip_suffix``)."""

    args: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to the transform function."""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransformRule":
        return cls(
            name=str(data["name"]),
            args=dict(data.get("args", {})),
        )


@dataclass(frozen=True)
class GeneMappingConfig:
    """Per-dataset gene identifier mapping configuration.

    Top-level section of ``CanonicalizationSchema`` that controls how
    ``gene_id`` values are converted to ``canonical_gene_id`` values.
    """

    enabled: bool = False
    """When ``False``, ``canonical_gene_id = gene_id`` (identity)."""

    engine: str = "gget"
    """Mapping engine: ``gget``, ``identity``, or ``mapping_file``."""

    source_namespace: str = "gene_symbol"
    """Identifier namespace of raw ``gene_id`` values."""

    target_namespace: str = "gene_symbol"
    """Desired canonical identifier namespace."""

    mapping_file: str | None = None
    """Path to a tabular mapping file when ``engine == "mapping_file"``."""

    def is_identity(self) -> bool:
        return not self.enabled or self.engine == "identity"


# ---------------------------------------------------------------------------
# Top-level: CanonicalizationSchema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalizationSchema:
    """Per-dataset canonicalization configuration.

    Describes how raw obs/var sidecars are transformed into canonical
    metadata parquet files.  Designed to be serialised as YAML.

    Serialisation helpers (``to_yaml``, ``from_yaml_file``) are provided
    on the class itself to keep the module dependency footprint small.
    """

    kind: str = "canonicalization-schema"
    contract_version: str = CONTRACT_VERSION
    dataset_id: str = ""
    status: str = "draft"
    description: str = ""

    # Obs mappings
    obs_column_mappings: tuple[ObsColumnMapping, ...] = ()
    obs_extensible: tuple[ExtensibleColumn, ...] = ()

    # Var mappings
    var_column_mappings: tuple[VarColumnMapping, ...] = ()
    var_extensible: tuple[ExtensibleColumn, ...] = ()

    # Gene mapping
    gene_mapping: GeneMappingConfig = field(default_factory=GeneMappingConfig)

    # Notes
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        """Raise ``ValueError`` if required invariants are violated."""
        if self.kind != "canonicalization-schema":
            raise ValueError(f"kind must be 'canonicalization-schema', got {self.kind!r}")
        if not self.dataset_id:
            raise ValueError("dataset_id is required")
        if self.status not in {"draft", "ready"}:
            raise ValueError(f"invalid status {self.status!r}; must be 'draft' or 'ready'")

        # Check for duplicate canonical names
        obs_names = [m.canonical_name for m in self.obs_column_mappings]
        obs_names += [e.canonical_name for e in self.obs_extensible]
        dups = _find_duplicates(obs_names)
        if dups:
            raise ValueError(f"Duplicate obs canonical names: {dups}")

        var_names = [m.canonical_name for m in self.var_column_mappings]
        var_names += [e.canonical_name for e in self.var_extensible]
        dups = _find_duplicates(var_names)
        if dups:
            raise ValueError(f"Duplicate var canonical names: {dups}")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary suitable for YAML dumping."""
        return {
            "kind": self.kind,
            "contract_version": self.contract_version,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "description": self.description,
            "gene_mapping": {
                "enabled": self.gene_mapping.enabled,
                "engine": self.gene_mapping.engine,
                "source_namespace": self.gene_mapping.source_namespace,
                "target_namespace": self.gene_mapping.target_namespace,
                "mapping_file": self.gene_mapping.mapping_file,
            },
            "obs_column_mappings": [
                _obs_mapping_to_dict(m) for m in self.obs_column_mappings
            ],
            "obs_extensible": [
                {"raw_source_column": e.raw_source_column, "canonical_name": e.canonical_name}
                for e in self.obs_extensible
            ],
            "var_column_mappings": [
                _var_mapping_to_dict(m) for m in self.var_column_mappings
            ],
            "var_extensible": [
                {"raw_source_column": e.raw_source_column, "canonical_name": e.canonical_name}
                for e in self.var_extensible
            ],
            "notes": list(self.notes),
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def write_yaml(self, output_path: str | Path) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalizationSchema":
        gm = data.get("gene_mapping", {})
        gene_mapping = GeneMappingConfig(
            enabled=bool(gm.get("enabled", False)),
            engine=str(gm.get("engine", "gget")),
            source_namespace=str(gm.get("source_namespace", "gene_symbol")),
            target_namespace=str(gm.get("target_namespace", "gene_symbol")),
            mapping_file=gm.get("mapping_file"),
        )

        obs_mappings = tuple(
            ObsColumnMapping(
                canonical_name=str(m["canonical_name"]),
                strategy=_nullsafe_str(m.get("strategy"), "null"),
                source_column=m.get("source_column"),
                literal_value=m.get("literal_value"),
                transforms=tuple(
                    TransformRule.from_dict(t) for t in m.get("transforms", [])
                ),
                fallback=str(m.get("fallback", MISSING_VALUE_LITERAL)),
            )
            for m in data.get("obs_column_mappings", [])
        )

        obs_ext = tuple(
            ExtensibleColumn(
                raw_source_column=str(e["raw_source_column"]),
                canonical_name=e.get("canonical_name"),
            )
            for e in data.get("obs_extensible", [])
        )

        var_mappings = tuple(
            VarColumnMapping(
                canonical_name=str(m["canonical_name"]),
                strategy=_nullsafe_str(m.get("strategy"), "null"),
                source_column=m.get("source_column"),
                literal_value=m.get("literal_value"),
                transforms=tuple(
                    TransformRule.from_dict(t) for t in m.get("transforms", [])
                ),
                fallback=str(m.get("fallback", MISSING_VALUE_LITERAL)),
                enabled=bool(m.get("enabled", gene_mapping.enabled)),
                engine=str(m.get("engine", gene_mapping.engine)),
                source_namespace=str(m.get("source_namespace", gene_mapping.source_namespace)),
                target_namespace=str(m.get("target_namespace", gene_mapping.target_namespace)),
                mapping_file=m.get("mapping_file"),
            )
            for m in data.get("var_column_mappings", [])
        )

        var_ext = tuple(
            ExtensibleColumn(
                raw_source_column=str(e["raw_source_column"]),
                canonical_name=e.get("canonical_name"),
            )
            for e in data.get("var_extensible", [])
        )

        schema = cls(
            kind=str(data.get("kind", "canonicalization-schema")),
            contract_version=str(data.get("contract_version", CONTRACT_VERSION)),
            dataset_id=str(data["dataset_id"]),
            status=str(data.get("status", "draft")),
            description=str(data.get("description", "")),
            obs_column_mappings=obs_mappings,
            obs_extensible=obs_ext,
            var_column_mappings=var_mappings,
            var_extensible=var_ext,
            gene_mapping=gene_mapping,
            notes=tuple(str(n) for n in data.get("notes", [])),
        )
        schema.validate()
        return schema

    @classmethod
    def from_yaml_file(cls, file_path: str | Path) -> "CanonicalizationSchema":
        payload = yaml.safe_load(Path(file_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected a YAML mapping in {file_path}")
        return cls.from_dict(payload)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nullsafe_str(value: Any, default: str) -> str:
    """Convert *value* to string, treating ``None`` as *default*.

    YAML ``null`` is parsed as Python ``None``; certain fields (e.g.
    ``strategy: null``) should become the literal string ``"null"``.
    """
    if value is None:
        return default
    return str(value)


def _find_duplicates(names: Sequence[str]) -> frozenset[str]:
    seen: set[str] = set()
    dups: set[str] = set()
    for n in names:
        if n in seen:
            dups.add(n)
        seen.add(n)
    return frozenset(dups)


def _obs_mapping_to_dict(m: ObsColumnMapping) -> dict[str, Any]:
    d: dict[str, Any] = {
        "canonical_name": m.canonical_name,
        "strategy": m.strategy,
    }
    if m.source_column is not None:
        d["source_column"] = m.source_column
    if m.literal_value is not None:
        d["literal_value"] = m.literal_value
    if m.transforms:
        d["transforms"] = [{"name": t.name, "args": t.args} for t in m.transforms]
    if m.fallback != MISSING_VALUE_LITERAL:
        d["fallback"] = m.fallback
    return d


def _var_mapping_to_dict(m: VarColumnMapping) -> dict[str, Any]:
    d: dict[str, Any] = {
        "canonical_name": m.canonical_name,
        "strategy": m.strategy,
    }
    if m.source_column is not None:
        d["source_column"] = m.source_column
    if m.literal_value is not None:
        d["literal_value"] = m.literal_value
    if m.transforms:
        d["transforms"] = [{"name": t.name, "args": t.args} for t in m.transforms]
    if m.fallback != MISSING_VALUE_LITERAL:
        d["fallback"] = m.fallback
    if m.strategy == "gene-mapping":
        d["enabled"] = m.enabled
        d["engine"] = m.engine
        d["source_namespace"] = m.source_namespace
        d["target_namespace"] = m.target_namespace
        if m.mapping_file is not None:
            d["mapping_file"] = m.mapping_file
    return d
