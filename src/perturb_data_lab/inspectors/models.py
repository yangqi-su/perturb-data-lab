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
