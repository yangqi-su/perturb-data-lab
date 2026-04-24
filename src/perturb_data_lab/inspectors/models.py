from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL


T = TypeVar("T")


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    return value


def _coerce_tuple(data: Any, item_type: type[T]) -> tuple[T, ...]:
    return tuple(item_type.from_dict(item) for item in data or [])


@dataclass(frozen=True)
class YamlDocument:
    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def write_yaml(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def _load_yaml_dict(cls, file_path: Path) -> dict[str, Any]:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected a YAML mapping in {file_path}")
        return payload


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    source_release: str
    source_path: str
    obs_rows: int
    var_rows: int
    obs_index_name: str
    var_index_name: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetIdentity":
        return cls(**data)


@dataclass(frozen=True)
class StructureSummary:
    has_raw: bool
    raw_var_rows: int
    layers: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructureSummary":
        return cls(
            has_raw=bool(data["has_raw"]),
            raw_var_rows=int(data["raw_var_rows"]),
            layers=tuple(data.get("layers", [])),
        )


@dataclass(frozen=True)
class FieldProfile:
    name: str
    dtype: str
    null_count: int
    sampled_unique_values: int
    examples: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FieldProfile":
        return cls(
            name=str(data["name"]),
            dtype=str(data["dtype"]),
            null_count=int(data["null_count"]),
            sampled_unique_values=int(data["sampled_unique_values"]),
            examples=tuple(str(item) for item in data.get("examples", [])),
        )


@dataclass(frozen=True)
class CountSourceCandidate:
    candidate: str
    rank: int
    status: str
    storage: str
    dtype: str
    shape: tuple[int, int]
    sampled_rows: int
    sampled_nonzero_values: int
    sampled_density: float
    fraction_noninteger_nonzero: float
    max_abs_integer_deviation: float
    nonnegative: bool
    inferred_transform: str
    recovery_policy: str
    notes: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CountSourceCandidate":
        shape = data.get("shape", [0, 0])
        return cls(
            candidate=str(data["candidate"]),
            rank=int(data["rank"]),
            status=str(data["status"]),
            storage=str(data["storage"]),
            dtype=str(data["dtype"]),
            shape=(int(shape[0]), int(shape[1])),
            sampled_rows=int(data["sampled_rows"]),
            sampled_nonzero_values=int(data["sampled_nonzero_values"]),
            sampled_density=float(data["sampled_density"]),
            fraction_noninteger_nonzero=float(data["fraction_noninteger_nonzero"]),
            max_abs_integer_deviation=float(data["max_abs_integer_deviation"]),
            nonnegative=bool(data["nonnegative"]),
            inferred_transform=str(data["inferred_transform"]),
            recovery_policy=str(data["recovery_policy"]),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )


@dataclass(frozen=True)
class CountSourceDecision:
    selected_candidate: str
    status: str
    confidence: str
    recovery_policy: str
    rationale: str
    uses_recovery: bool = False
    pass_mode: str | None = None  # "direct" | "recovered" | None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CountSourceDecision":
        return cls(
            **{k: v for k, v in data.items() if k in cls.__annotations__},
        )


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


@dataclass(frozen=True)
class FieldMapping:
    source_fields: tuple[str, ...]
    strategy: str
    transforms: tuple[TransformSpec, ...]
    confidence: str
    literal_value: str | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FieldMapping":
        return cls(
            source_fields=tuple(str(item) for item in data.get("source_fields", [])),
            strategy=str(data["strategy"]),
            transforms=_coerce_tuple(data.get("transforms"), TransformSpec),
            confidence=str(data["confidence"]),
            literal_value=(
                None
                if data.get("literal_value") is None
                else str(data["literal_value"])
            ),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )


@dataclass(frozen=True)
class SchemaPatchEntry:
    field: str
    action: str
    source_fields: tuple[str, ...] = ()
    value: str | None = None
    transforms: tuple[TransformSpec, ...] = ()
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SchemaPatchEntry":
        return cls(
            field=str(data["field"]),
            action=str(data["action"]),
            source_fields=tuple(str(item) for item in data.get("source_fields", [])),
            value=None if data.get("value") is None else str(data["value"]),
            transforms=_coerce_tuple(data.get("transforms"), TransformSpec),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )


@dataclass(frozen=True)
class DatasetSummaryDocument(YamlDocument):
    kind: str
    contract_version: str
    dataset: DatasetIdentity
    structure: StructureSummary
    obs_fields: tuple[FieldProfile, ...]
    var_fields: tuple[FieldProfile, ...]
    count_source_candidates: tuple[CountSourceCandidate, ...]
    count_source_decision: CountSourceDecision
    materialization_readiness: str
    inspector_notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.kind != "dataset-summary":
            raise ValueError("dataset summary kind mismatch")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("dataset summary contract version mismatch")
        if not self.dataset.dataset_id:
            raise ValueError("dataset_id is required")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetSummaryDocument":
        document = cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            dataset=DatasetIdentity.from_dict(data["dataset"]),
            structure=StructureSummary.from_dict(data["structure"]),
            obs_fields=_coerce_tuple(data.get("obs_fields"), FieldProfile),
            var_fields=_coerce_tuple(data.get("var_fields"), FieldProfile),
            count_source_candidates=_coerce_tuple(
                data.get("count_source_candidates"), CountSourceCandidate
            ),
            count_source_decision=CountSourceDecision.from_dict(
                data["count_source_decision"]
            ),
            materialization_readiness=str(data["materialization_readiness"]),
            inspector_notes=tuple(
                str(item) for item in data.get("inspector_notes", [])
            ),
        )
        document.validate()
        return document

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "DatasetSummaryDocument":
        return cls.from_dict(cls._load_yaml_dict(file_path))


@dataclass(frozen=True)
class SchemaFieldEntry:
    source_fields: tuple[str, ...]
    strategy: str  # source-field | literal | derived | null
    transforms: tuple[TransformSpec, ...]
    confidence: str  # high | medium | low
    required: bool
    literal_value: str | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SchemaFieldEntry":
        return cls(
            source_fields=tuple(str(item) for item in data.get("source_fields", [])),
            strategy=str(data["strategy"]),
            transforms=_coerce_tuple(data.get("transforms"), TransformSpec),
            confidence=str(data["confidence"]),
            required=bool(data.get("required", False)),
            literal_value=data.get("literal_value"),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class CountSourceSpec:
    selected: str
    integer_only: bool
    uses_recovery: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CountSourceSpec":
        return cls(
            selected=str(data["selected"]),
            integer_only=bool(data.get("integer_only", True)),
            uses_recovery=bool(data.get("uses_recovery", False)),
        )


@dataclass(frozen=True)
class FeatureTokenizationSpec:
    selected: str  # which var column is the tokenization target
    namespace: str  # e.g. "ensembl", "gene_symbol"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureTokenizationSpec":
        return cls(
            selected=str(data["selected"]),
            namespace=str(data.get("namespace", "unknown")),
        )

    def is_compatible_for_append(self, corpus_namespace: str) -> bool:
        """Return True when this spec's namespace matches the corpus namespace.

        Per the Phase 1 append compatibility contract, namespace mismatch is a
        hard rejection condition before any I/O or tokenization.
        """
        if not self.namespace or self.namespace in ("unknown", "set-manually"):
            return False
        return self.namespace == corpus_namespace


@dataclass(frozen=True)
class SchemaDocument(YamlDocument):
    """Unified schema artifact: one editable document replacing schema-proposal + schema-patch.

    Status lifecycle:
    - ``status: draft`` — auto-generated; requires human review before materialization
    - ``status: ready`` — reviewed; all required fields have non-null strategies; materialization permitted

    Unresolved fields are represented inline with ``strategy: null`` (no separate unresolved list).
    Multi-source values are joined with ``_`` before transforms are applied.
    """

    kind: str
    contract_version: str
    dataset_id: str
    source_path: str
    status: str  # draft | ready
    dataset_metadata: dict[str, SchemaFieldEntry]
    perturbation_fields: dict[str, SchemaFieldEntry]
    context_fields: dict[str, SchemaFieldEntry]
    feature_fields: dict[str, SchemaFieldEntry]
    count_source: CountSourceSpec
    feature_tokenization: FeatureTokenizationSpec
    transform_catalog: tuple[TransformCatalogEntry, ...]
    materialization_notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.kind != "schema":
            raise ValueError("schema kind mismatch")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("schema contract version mismatch")
        if not self.dataset_id:
            raise ValueError("dataset_id is required")
        if self.status not in {"draft", "ready"}:
            raise ValueError("invalid schema status")

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        # Serialize nested dicts of SchemaFieldEntry
        for section in (
            "dataset_metadata",
            "perturbation_fields",
            "context_fields",
            "feature_fields",
        ):
            if section in payload and isinstance(payload[section], dict):
                payload[section] = {
                    key: (
                        value.to_dict()
                        if isinstance(value, YamlDocument)
                        else _serialize(value)
                    )
                    for key, value in payload[section].items()
                }
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SchemaDocument":
        def _parse_field_entry(entry_data: Any) -> SchemaFieldEntry:
            if isinstance(entry_data, SchemaFieldEntry):
                return entry_data
            return SchemaFieldEntry.from_dict(entry_data)

        document = cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            dataset_id=str(data["dataset_id"]),
            source_path=str(data["source_path"]),
            status=str(data["status"]),
            dataset_metadata={
                str(k): _parse_field_entry(v)
                for k, v in dict(data.get("dataset_metadata", {})).items()
            },
            perturbation_fields={
                str(k): _parse_field_entry(v)
                for k, v in dict(data.get("perturbation_fields", {})).items()
            },
            context_fields={
                str(k): _parse_field_entry(v)
                for k, v in dict(data.get("context_fields", {})).items()
            },
            feature_fields={
                str(k): _parse_field_entry(v)
                for k, v in dict(data.get("feature_fields", {})).items()
            },
            count_source=CountSourceSpec.from_dict(data["count_source"]),
            feature_tokenization=FeatureTokenizationSpec.from_dict(
                data.get("feature_tokenization", {})
            ),
            transform_catalog=_coerce_tuple(
                data.get("transform_catalog"), TransformCatalogEntry
            ),
            materialization_notes=tuple(
                str(item) for item in data.get("materialization_notes", [])
            ),
        )
        document.validate()
        return document

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "SchemaDocument":
        return cls.from_dict(cls._load_yaml_dict(file_path))

    @classmethod
    def new_draft(
        cls,
        dataset_id: str,
        source_path: str,
        dataset_metadata: dict[str, SchemaFieldEntry],
        perturbation_fields: dict[str, SchemaFieldEntry],
        context_fields: dict[str, SchemaFieldEntry],
        feature_fields: dict[str, SchemaFieldEntry],
        count_source: CountSourceSpec,
        feature_tokenization: FeatureTokenizationSpec,
        transform_catalog: tuple[TransformCatalogEntry, ...] = (),
        materialization_notes: tuple[str, ...] = (),
    ) -> "SchemaDocument":
        """Factory: create a new ``status: draft`` schema document."""
        return cls(
            kind="schema",
            contract_version=CONTRACT_VERSION,
            dataset_id=dataset_id,
            source_path=source_path,
            status="draft",
            dataset_metadata=dataset_metadata,
            perturbation_fields=perturbation_fields,
            context_fields=context_fields,
            feature_fields=feature_fields,
            count_source=count_source,
            feature_tokenization=feature_tokenization,
            transform_catalog=transform_catalog,
            materialization_notes=materialization_notes,
        )

    def is_ready(self) -> bool:
        """Return True when all required fields have a non-null strategy."""
        if self.status != "ready":
            return False
        for section in (
            self.perturbation_fields,
            self.context_fields,
            self.feature_fields,
        ):
            for name, entry in section.items():
                if entry.required and entry.strategy == "null":
                    return False
        return True

    def required_cell_fields(self) -> list[str]:
        """List of required cell field names (both perturbation and context)."""
        result = []
        for section in (self.perturbation_fields, self.context_fields):
            for name, entry in section.items():
                if entry.required:
                    result.append(name)
        return result

    def required_feature_fields(self) -> list[str]:
        """List of required feature field names."""
        return [name for name, entry in self.feature_fields.items() if entry.required]


@dataclass(frozen=True)
class SchemaProposalDocument(YamlDocument):
    kind: str
    contract_version: str
    dataset_id: str
    source_path: str
    summary_artifact: str
    canonical_defaults: dict[str, str]
    count_source_decision: CountSourceDecision
    transform_catalog: tuple[TransformCatalogEntry, ...]
    field_mappings: dict[str, FieldMapping]
    unresolved_fields: tuple[str, ...]
    materialization_readiness: str

    def validate(self) -> None:
        if self.kind != "schema-proposal":
            raise ValueError("schema proposal kind mismatch")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("schema proposal contract version mismatch")
        if not self.dataset_id:
            raise ValueError("dataset_id is required")

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["field_mappings"] = {
            key: value.to_dict()
            if isinstance(value, YamlDocument)
            else _serialize(value)
            for key, value in self.field_mappings.items()
        }
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SchemaProposalDocument":
        document = cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            dataset_id=str(data["dataset_id"]),
            source_path=str(data["source_path"]),
            summary_artifact=str(data["summary_artifact"]),
            canonical_defaults=dict(data.get("canonical_defaults", {})),
            count_source_decision=CountSourceDecision.from_dict(
                data["count_source_decision"]
            ),
            transform_catalog=_coerce_tuple(
                data.get("transform_catalog"), TransformCatalogEntry
            ),
            field_mappings={
                str(key): FieldMapping.from_dict(value)
                for key, value in dict(data.get("field_mappings", {})).items()
            },
            unresolved_fields=tuple(
                str(item) for item in data.get("unresolved_fields", [])
            ),
            materialization_readiness=str(data["materialization_readiness"]),
        )
        document.validate()
        return document

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "SchemaProposalDocument":
        return cls.from_dict(cls._load_yaml_dict(file_path))


@dataclass(frozen=True)
class SchemaPatchDocument(YamlDocument):
    kind: str
    contract_version: str
    dataset_id: str
    review_status: str
    summary_artifact: str
    proposal_artifact: str
    unresolved_fields: tuple[str, ...]
    patches: tuple[SchemaPatchEntry, ...]
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.kind != "schema-patch":
            raise ValueError("schema patch kind mismatch")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("schema patch contract version mismatch")
        if self.review_status not in {"pending", "accepted", "rejected"}:
            raise ValueError("invalid schema patch review status")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SchemaPatchDocument":
        document = cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            dataset_id=str(data["dataset_id"]),
            review_status=str(data["review_status"]),
            summary_artifact=str(data["summary_artifact"]),
            proposal_artifact=str(data["proposal_artifact"]),
            unresolved_fields=tuple(
                str(item) for item in data.get("unresolved_fields", [])
            ),
            patches=_coerce_tuple(data.get("patches"), SchemaPatchEntry),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )
        document.validate()
        return document

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "SchemaPatchDocument":
        return cls.from_dict(cls._load_yaml_dict(file_path))


@dataclass(frozen=True)
class InspectionTarget:
    dataset_id: str
    source_path: str
    source_release: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InspectionTarget":
        return cls(
            dataset_id=str(data["dataset_id"]),
            source_path=str(data["source_path"]),
            source_release=str(data["source_release"]),
        )


@dataclass(frozen=True)
class InspectionBatchConfig(YamlDocument):
    output_root: str
    datasets: tuple[InspectionTarget, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InspectionBatchConfig":
        return cls(
            output_root=str(data["output_root"]),
            datasets=_coerce_tuple(data.get("datasets"), InspectionTarget),
        )

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "InspectionBatchConfig":
        return cls.from_dict(cls._load_yaml_dict(file_path))


@dataclass(frozen=True)
class InspectionBatchRecord:
    dataset_id: str
    review_bundle: str  # absolute path to the authoritative review bundle for this dataset
    selected_count_source: str
    materialization_readiness: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InspectionBatchRecord":
        return cls(**data)


@dataclass(frozen=True)
class InspectionBatchManifest(YamlDocument):
    kind: str
    contract_version: str
    output_root: str
    records: tuple[InspectionBatchRecord, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InspectionBatchManifest":
        return cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            output_root=str(data["output_root"]),
            records=_coerce_tuple(data.get("records"), InspectionBatchRecord),
        )
