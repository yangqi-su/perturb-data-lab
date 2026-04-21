"""Phase 3 materializer typed models."""

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
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
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
class CountSourceSpec:
    selected: str
    integer_only: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CountSourceSpec":
        return cls(
            selected=str(data["selected"]),
            integer_only=bool(data.get("integer_only", True)),
        )


@dataclass(frozen=True)
class OutputRoots:
    metadata_root: str
    matrix_root: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputRoots":
        return cls(
            metadata_root=str(data["metadata_root"]),
            matrix_root=str(data["matrix_root"]),
        )


@dataclass(frozen=True)
class ProvenanceSpec:
    source_path: str
    schema: str  # single unified schema artifact path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProvenanceSpec":
        return cls(
            source_path=str(data["source_path"]),
            schema=str(data["schema"]),
        )


@dataclass(frozen=True)
class FeatureManifestEntry:
    token_id: int
    feature_id: str
    feature_label: str
    namespace: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureManifestEntry":
        return cls(
            token_id=int(data["token_id"]),
            feature_id=str(data["feature_id"]),
            feature_label=str(data["feature_label"]),
            namespace=str(data.get("namespace", "unknown")),
        )


@dataclass(frozen=True)
class SizeFactorEntry:
    cell_id: str
    size_factor: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SizeFactorEntry":
        return cls(cell_id=str(data["cell_id"]), size_factor=float(data["size_factor"]))


@dataclass(frozen=True)
class QAMetric:
    name: str
    value: float
    threshold: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QAMetric":
        return cls(
            name=str(data["name"]),
            value=float(data["value"]),
            threshold=float(data["threshold"])
            if data.get("threshold") is not None
            else None,
        )

    def passed(self) -> bool:
        if self.threshold is None:
            return True
        return self.value <= self.threshold


@dataclass(frozen=True)
class MaterializationManifest(YamlDocument):
    kind: str
    contract_version: str
    dataset_id: str
    release_id: str
    route: str  # create_new | append_monolithic | append_routed
    count_source: CountSourceSpec
    outputs: OutputRoots
    provenance: ProvenanceSpec
    tokenizer_path: str | None = None
    feature_meta_paths: dict[str, str] | None = None
    size_factor_manifest_path: str | None = None
    qa_manifest_path: str | None = None
    hvg_sidecar_path: str | None = None
    integer_verified: bool = False
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.kind != "materialization-manifest":
            raise ValueError("materialization manifest kind mismatch")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("materialization manifest contract version mismatch")
        if self.route not in {"create_new", "append_routed"}:
            raise ValueError(f"invalid route: {self.route}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaterializationManifest":
        document = cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            dataset_id=str(data["dataset_id"]),
            release_id=str(data["release_id"]),
            route=str(data["route"]),
            count_source=CountSourceSpec.from_dict(data["count_source"]),
            outputs=OutputRoots.from_dict(data["outputs"]),
            provenance=ProvenanceSpec.from_dict(data["provenance"]),
            tokenizer_path=data.get("tokenizer_path"),
            feature_meta_paths={
                k: str(v) for k, v in (data.get("feature_meta_paths") or {}).items()
            },
            size_factor_manifest_path=data.get("size_factor_manifest_path"),
            qa_manifest_path=data.get("qa_manifest_path"),
            hvg_sidecar_path=data.get("hvg_sidecar_path"),
            integer_verified=bool(data.get("integer_verified", False)),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )
        document.validate()
        return document

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "MaterializationManifest":
        return cls.from_dict(cls._load_yaml_dict(file_path))


@dataclass(frozen=True)
class FeatureRegistryEntry:
    token_id: int
    feature_id: str
    feature_label: str
    namespace: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureRegistryEntry":
        return cls(
            token_id=int(data["token_id"]),
            feature_id=str(data["feature_id"]),
            feature_label=str(data["feature_label"]),
            namespace=str(data.get("namespace", "unknown")),
        )


@dataclass(frozen=True)
class FeatureRegistryManifest(YamlDocument):
    kind: str
    contract_version: str
    registry_id: str
    append_only: bool
    namespace: str
    feature_id_field: str
    feature_label_field: str
    default_missing_value: str
    entries: tuple[FeatureRegistryEntry, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureRegistryManifest":
        return cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            registry_id=str(data["registry_id"]),
            append_only=bool(data.get("append_only", True)),
            namespace=str(data.get("namespace", "unknown")),
            feature_id_field=str(data.get("feature_id_field", "gene_id")),
            feature_label_field=str(data.get("feature_label_field", "gene_symbol")),
            default_missing_value=str(
                data.get("default_missing_value", MISSING_VALUE_LITERAL)
            ),
            entries=tuple(
                FeatureRegistryEntry.from_dict(e) for e in data.get("entries", [])
            ),
        )

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "FeatureRegistryManifest":
        return cls.from_dict(cls._load_yaml_dict(file_path))

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["entries"] = [
            e.to_dict() if isinstance(e, YamlDocument) else e
            for e in payload["entries"]
        ]
        return payload


@dataclass(frozen=True)
class SizeFactorManifest(YamlDocument):
    kind: str
    contract_version: str
    release_id: str
    method: str  # e.g. sum, median
    entries: tuple[SizeFactorEntry, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SizeFactorManifest":
        return cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            release_id=str(data["release_id"]),
            method=str(data.get("method", "sum")),
            entries=tuple(
                SizeFactorEntry.from_dict(e) for e in data.get("entries", [])
            ),
        )

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "SizeFactorManifest":
        return cls.from_dict(cls._load_yaml_dict(file_path))


@dataclass(frozen=True)
class QAManifest(YamlDocument):
    kind: str
    contract_version: str
    release_id: str
    metrics: tuple[QAMetric, ...]
    all_passed: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QAManifest":
        metrics = tuple(QAMetric.from_dict(m) for m in data.get("metrics", []))
        all_passed = all(m.passed() for m in metrics)
        return cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            release_id=str(data["release_id"]),
            metrics=metrics,
            all_passed=all_passed,
        )

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "QAManifest":
        return cls.from_dict(cls._load_yaml_dict(file_path))


@dataclass(frozen=True)
class CellMetadataRecord:
    cell_id: str
    dataset_id: str
    dataset_release: str
    raw_fields: dict[str, Any]
    canonical_perturbation: dict[str, str]
    canonical_context: dict[str, str]
    size_factor: float

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CellMetadataRecord":
        return cls(
            cell_id=str(data["cell_id"]),
            dataset_id=str(data["dataset_id"]),
            dataset_release=str(data["dataset_release"]),
            raw_fields=dict(data.get("raw_fields", {})),
            canonical_perturbation=dict(data.get("canonical_perturbation", {})),
            canonical_context=dict(data.get("canonical_context", {})),
            size_factor=float(data.get("size_factor", 1.0)),
        )


@dataclass(frozen=True)
class DatasetJoinRecord:
    dataset_id: str
    release_id: str
    join_mode: str
    manifest_path: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetJoinRecord":
        return cls(
            dataset_id=str(data["dataset_id"]),
            release_id=str(data["release_id"]),
            join_mode=str(data["join_mode"]),
            manifest_path=str(data["manifest_path"]),
        )


@dataclass(frozen=True)
class CorpusIndexDocument(YamlDocument):
    kind: str
    contract_version: str
    corpus_id: str
    global_metadata: dict[str, Any]
    datasets: tuple[DatasetJoinRecord, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusIndexDocument":
        return cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            corpus_id=str(data["corpus_id"]),
            global_metadata=dict(data.get("global_metadata", {})),
            datasets=tuple(
                DatasetJoinRecord.from_dict(d) for d in data.get("datasets", [])
            ),
        )

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "CorpusIndexDocument":
        return cls.from_dict(cls._load_yaml_dict(file_path))


@dataclass(frozen=True)
class GlobalMetadataDocument(YamlDocument):
    kind: str
    contract_version: str
    schema_version: str
    feature_registry_id: str  # deprecated: replaced by tokenizer_path in contract 0.2.0
    missing_value_literal: str
    raw_field_policy: str
    tokenizer_path: str | None = None  # relative path from corpus root to tokenizer.json
    emission_spec_path: str | None = None  # relative path from corpus root to corpus-emission-spec.yaml
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GlobalMetadataDocument":
        return cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            schema_version=str(data.get("schema_version", "0.1.0")),
            feature_registry_id=str(data.get("feature_registry_id", "")),
            missing_value_literal=str(
                data.get("missing_value_literal", MISSING_VALUE_LITERAL)
            ),
            raw_field_policy=str(data.get("raw_field_policy", "preserve-unchanged")),
            tokenizer_path=data.get("tokenizer_path"),
            emission_spec_path=data.get("emission_spec_path"),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "GlobalMetadataDocument":
        return cls.from_dict(cls._load_yaml_dict(file_path))
