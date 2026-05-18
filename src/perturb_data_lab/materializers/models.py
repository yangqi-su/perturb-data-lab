"""Typed documents used by materialization and corpus loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..contracts import CONTRACT_VERSION


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
    uses_recovery: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CountSourceSpec":
        return cls(
            selected=data["selected"],
            integer_only=data["integer_only"],
            uses_recovery=data.get("uses_recovery", False),
        )


@dataclass(frozen=True)
class OutputRoots:
    metadata_root: str
    matrix_root: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputRoots":
        return cls(
            metadata_root=data["metadata_root"],
            matrix_root=data["matrix_root"],
        )


@dataclass(frozen=True)
class ProvenanceSpec:
    source_path: str
    inspection_summary_path: str  # dataset-summary.yaml copied into the materialized dataset

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProvenanceSpec":
        return cls(
            source_path=data["source_path"],
            inspection_summary_path=data["inspection_summary_path"],
        )


@dataclass(frozen=True)
class MaterializationManifest(YamlDocument):
    kind: str
    contract_version: str
    dataset_id: str
    route: str  # create_new | append_routed
    backend: str  # lance | zarr
    topology: str  # federated | aggregate
    count_source: CountSourceSpec
    outputs: OutputRoots
    provenance: ProvenanceSpec
    raw_cell_meta_path: str | None = None  # raw-obs.parquet
    raw_feature_meta_path: str | None = None  # raw-var.parquet
    provenance_spec_path: str | None = None  # feature-order/provenance artifact
    size_factor_parquet_path: str | None = None  # Parquet: per-cell size factors (separate from cells parquet)
    hvg_ranking_path: str | None = None  # canonical per-dataset hvg.parquet artifact
    default_n_hvg: int | None = None
    integer_verified: bool = False
    cell_count: int = 0  # number of cells materialized (set by materialize() for corpus index use)
    feature_count: int = 0  # number of features in dataset-local feature space
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.kind != "materialization-manifest":
            raise ValueError("materialization manifest kind mismatch")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("materialization manifest contract version mismatch")
        if self.route not in {"create_new", "append_routed"}:
            raise ValueError(f"invalid route: {self.route}")
        if self.backend not in {"zarr", "lance"}:
            raise ValueError(f"invalid backend: {self.backend}")
        if self.topology not in {"federated", "aggregate"}:
            raise ValueError(f"invalid topology: {self.topology}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaterializationManifest":
        document = cls(
            kind=data["kind"],
            contract_version=data["contract_version"],
            dataset_id=data["dataset_id"],
            route=data["route"],
            backend=data["backend"],
            topology=data["topology"],
            count_source=CountSourceSpec.from_dict(data["count_source"]),
            outputs=OutputRoots.from_dict(data["outputs"]),
            provenance=ProvenanceSpec.from_dict(data["provenance"]),
            raw_cell_meta_path=data.get("raw_cell_meta_path"),
            raw_feature_meta_path=data.get("raw_feature_meta_path"),
            provenance_spec_path=data.get("provenance_spec_path"),
            size_factor_parquet_path=data.get("size_factor_parquet_path"),
            hvg_ranking_path=data.get("hvg_ranking_path"),
            default_n_hvg=data.get("default_n_hvg"),
            integer_verified=data.get("integer_verified", False),
            cell_count=data.get("cell_count", 0),
            feature_count=data.get("feature_count", 0),
            notes=tuple(data.get("notes", ())),
        )
        document.validate()
        return document

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "MaterializationManifest":
        return cls.from_dict(cls._load_yaml_dict(file_path))


@dataclass(frozen=True)
class DatasetJoinRecord:
    """One dataset's membership record in ``corpus-index.yaml``."""

    dataset_id: str
    join_mode: str
    manifest_path: str
    dataset_index: int = 0
    cell_count: int = 0  # number of cells in this dataset
    global_start: int = 0  # inclusive start of global cell range for this dataset
    global_end: int = 0  # exclusive end of global cell range for this dataset

    def validate(self) -> None:
        if self.dataset_index < 0:
            raise ValueError(f"dataset_index must be non-negative: {self.dataset_index}")
        if self.join_mode not in {"create_new", "append_routed"}:
            raise ValueError(f"invalid join_mode: {self.join_mode}")
        if self.cell_count < 0:
            raise ValueError(f"cell_count must be non-negative: {self.cell_count}")
        if self.global_start < 0:
            raise ValueError(f"global_start must be non-negative: {self.global_start}")
        if self.global_end < self.global_start:
            raise ValueError(
                f"global_end ({self.global_end}) must be >= global_start ({self.global_start})"
            )
        if self.global_end - self.global_start != self.cell_count:
            raise ValueError(
                f"global range [{self.global_start}, {self.global_end}) length "
                f"({self.global_end - self.global_start}) does not match cell_count ({self.cell_count})"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetJoinRecord":
        record = cls(
            dataset_id=data["dataset_id"],
            dataset_index=data["dataset_index"],
            join_mode=data["join_mode"],
            manifest_path=data["manifest_path"],
            cell_count=data["cell_count"],
            global_start=data["global_start"],
            global_end=data["global_end"],
        )
        record.validate()
        return record


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
            kind=data["kind"],
            contract_version=data["contract_version"],
            corpus_id=data["corpus_id"],
            global_metadata=dict(data["global_metadata"]),
            datasets=tuple(
                DatasetJoinRecord.from_dict(d) for d in data["datasets"]
            ),
        )

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "CorpusIndexDocument":
        return cls.from_dict(cls._load_yaml_dict(file_path))
