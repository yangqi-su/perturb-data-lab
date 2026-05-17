"""Phase 3 materializer typed models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL


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
            selected=str(data["selected"]),
            integer_only=bool(data.get("integer_only", True)),
            uses_recovery=bool(data.get("uses_recovery", False)),
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
    review_bundle: str  # Stage 1 dataset-summary.yaml path (the gating artifact)
    accepted_schema_copy: str | None = None  # optional audit copy, NOT read back

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProvenanceSpec":
        return cls(
            source_path=str(data["source_path"]),
            review_bundle=str(data.get("review_bundle", "")),
            accepted_schema_copy=data.get("accepted_schema_copy"),
        )


@dataclass(frozen=True)
class CorpusRegistrationInfo:
    """Registration metadata produced when Stage2Materializer registers with a corpus.

    This is written into the MaterializationManifest when register=True is set
    on Stage2Materializer. It records whether the dataset was registered as a
    new corpus creation or an append, and the paths to the corpus artifacts.
    """

    corpus_id: str
    is_create: bool  # True = new corpus, False = append to existing
    corpus_index_path: str  # path to corpus-index.yaml
    dataset_index: int      # assigned dataset index in the corpus
    global_start: int       # inclusive start of global cell range
    global_end: int         # exclusive end of global cell range

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusRegistrationInfo":
        return cls(
            corpus_id=str(data["corpus_id"]),
            is_create=bool(data["is_create"]),
            corpus_index_path=str(data["corpus_index_path"]),
            dataset_index=int(data["dataset_index"]),
            global_start=int(data["global_start"]),
            global_end=int(data["global_end"]),
        )


@dataclass(frozen=True)
class MaterializationManifest(YamlDocument):
    kind: str
    contract_version: str
    dataset_id: str
    route: str  # create_new | append_routed
    backend: str  # lance | zarr
    topology: str  # federated | aggregate (Stage 2 contract: backend and topology are separate)
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
    # Phase 4: corpus registration info (set when Stage2Materializer.register=True)
    corpus_registration: CorpusRegistrationInfo | None = None
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
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            dataset_id=str(data["dataset_id"]),
            route=str(data["route"]),
            backend=str(data["backend"]),
            topology=str(data["topology"]),
            count_source=CountSourceSpec.from_dict(data["count_source"]),
            outputs=OutputRoots.from_dict(data["outputs"]),
            provenance=ProvenanceSpec.from_dict(data["provenance"]),
            raw_cell_meta_path=data.get("raw_cell_meta_path"),
            raw_feature_meta_path=data.get("raw_feature_meta_path"),
            provenance_spec_path=data.get("provenance_spec_path"),
            size_factor_parquet_path=data.get("size_factor_parquet_path"),
            hvg_ranking_path=data.get("hvg_ranking_path"),
            default_n_hvg=(
                int(data["default_n_hvg"])
                if data.get("default_n_hvg") is not None
                else None
            ),
            integer_verified=bool(data.get("integer_verified", False)),
            cell_count=int(data.get("cell_count", 0)),
            feature_count=int(data.get("feature_count", 0)),
            corpus_registration=(
                CorpusRegistrationInfo.from_dict(data["corpus_registration"])
                if data.get("corpus_registration") is not None
                else None
            ),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )
        document.validate()
        return document

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "MaterializationManifest":
        return cls.from_dict(cls._load_yaml_dict(file_path))


@dataclass(frozen=True)
class DatasetJoinRecord:
    """A single dataset's entry in the corpus index.

    This record is the authoritative source for dataset membership, append
    order, and global cell routing within the Arrow/HF-only plan scope.

    Fields required for correct routing:
    - ``dataset_id`` — stable dataset identifier
    - ``join_mode`` — ``create_new`` or ``append_routed``
    - ``manifest_path`` — relative path from corpus root to the dataset's manifest
    - ``cell_count`` — number of cells in this dataset (used for global range computation)
    - ``global_start`` — inclusive start of this dataset's cell range in the corpus
    - ``global_end`` — exclusive end of this dataset's cell range in the corpus

    The ``global_start``/``global_end`` fields form a deterministic, contiguous,
    non-overlapping partition of the corpus and are the sole authority for
    global-index-to-dataset routing. They are set by ``update_corpus_index`` when
    appending a new dataset to the corpus, based on the total cell count before
    the new dataset is added.

    Deferred (out of plan scope):
    - Tokenizer and token-id semantics are handled separately.
    - Unified in-memory canonical metadata is handled separately.
    - ``canonicalize-meta`` is explicitly deferred.
    """

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
        return cls(
            dataset_id=str(data["dataset_id"]),
            dataset_index=int(data.get("dataset_index", 0)),
            join_mode=str(data["join_mode"]),
            manifest_path=str(data["manifest_path"]),
            cell_count=int(data.get("cell_count", 0)),
            global_start=int(data.get("global_start", 0)),
            global_end=int(data.get("global_end", 0)),
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
        dataset_payloads = []
        for idx, item in enumerate(data.get("datasets", [])):
            payload = dict(item)
            payload.setdefault("dataset_index", idx)
            dataset_payloads.append(payload)
        return cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            corpus_id=str(data["corpus_id"]),
            global_metadata=dict(data.get("global_metadata", {})),
            datasets=tuple(
                DatasetJoinRecord.from_dict(d) for d in dataset_payloads
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
    missing_value_literal: str
    raw_field_policy: str
    backend: str | None = None  # lance | zarr
    topology: str | None = None  # federated | aggregate (Stage 2 contract: separate from backend)
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.backend is not None and self.backend not in {"zarr", "lance"}:
            raise ValueError(f"invalid backend in global-metadata: {self.backend}")
        if self.topology is not None and self.topology not in {"federated", "aggregate"}:
            raise ValueError(f"invalid topology in global-metadata: {self.topology}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GlobalMetadataDocument":
        document = cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            schema_version=str(data.get("schema_version", "0.1.0")),
            missing_value_literal=str(
                data.get("missing_value_literal", MISSING_VALUE_LITERAL)
            ),
            raw_field_policy=str(data.get("raw_field_policy", "preserve-unchanged")),
            backend=data.get("backend"),
            topology=data.get("topology"),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )
        document.validate()
        return document

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "GlobalMetadataDocument":
        return cls.from_dict(cls._load_yaml_dict(file_path))
