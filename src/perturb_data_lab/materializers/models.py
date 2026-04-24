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
class RawCellMetadataRecord:
    """A single cell's raw metadata record — fields directly from h5ad obs, no canonical mapping applied.

    This record preserves the full obs row as the user saw it in the source h5ad,
    before any canonical field resolution. It is the authoritative source for
    canonical cell metadata rebuild in ``canonicalize-meta``.
    """

    cell_id: str
    dataset_id: str
    dataset_release: str
    raw_fields: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RawCellMetadataRecord":
        return cls(
            cell_id=str(data["cell_id"]),
            dataset_id=str(data["dataset_id"]),
            dataset_release=str(data["dataset_release"]),
            raw_fields=dict(data.get("raw_fields", {})),
        )


@dataclass(frozen=True)
class RawFeatureMetadataRecord:
    """A single feature's raw metadata record — fields directly from h5ad var, no canonical mapping applied.

    This record preserves the full var row as the user saw it in the source h5ad,
    before any canonical field resolution. It is the authoritative source for
    canonical feature metadata rebuild in ``canonicalize-meta``.
    """

    origin_index: int
    feature_id: str
    raw_fields: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RawFeatureMetadataRecord":
        return cls(
            origin_index=int(data["origin_index"]),
            feature_id=str(data["feature_id"]),
            raw_fields=dict(data.get("raw_fields", {})),
        )

    def write_yaml(self, output_path: Path) -> None:
        """Serialize to YAML and write to output_path."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_yaml(), encoding="utf-8")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


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
    route: str  # create_new | append_routed
    backend: str  # arrow-hf | webdataset | zarr | lance (legacy names normalized on read)
    topology: str  # federated | aggregate (Stage 2 contract: backend and topology are separate)
    count_source: CountSourceSpec
    outputs: OutputRoots
    provenance: ProvenanceSpec
    # New Phase 3 dataset-local artifact paths (tokenizer removed)
    raw_cell_meta_path: str | None = None  # Parquet: raw cell metadata (no canonical mapping); SQLite deprecated
    raw_feature_meta_path: str | None = None  # Parquet: raw feature metadata (no canonical mapping)
    accepted_schema_path: str | None = None  # optional audit copy of accepted schema; NOT read back
    metadata_summary_path: str | None = None  # per-dataset metadata summary (field coverage, null fractions)
    provenance_spec_path: str | None = None  # feature-order/provenance artifact
    # Existing paths
    feature_meta_paths: dict[str, str] | None = None  # canonical feature metadata (origin + token parquet paths)
    size_factor_manifest_path: str | None = None
    qa_manifest_path: str | None = None
    hvg_sidecar_path: str | None = None
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
        if self.backend not in {
            "arrow-hf",
            "webdataset",
            "zarr",
            "lance",
        }:
            raise ValueError(f"invalid backend: {self.backend}")
        if self.topology not in {"federated", "aggregate"}:
            raise ValueError(f"invalid topology: {self.topology}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaterializationManifest":
        backend_raw = data.get("backend", "arrow-hf")
        # Normalize legacy backend names to the contract's clean names
        _LEGACY_BACKEND_MAP = {
            "zarr-ts": "zarr",
            "lancedb-aggregated": "lance",
            "zarr-aggregated": "zarr",
        }
        backend_normalized = _LEGACY_BACKEND_MAP.get(backend_raw, backend_raw)
        # Normalize legacy route names to contract names
        route_raw = data.get("route", "create_new")
        if route_raw == "create_new":
            route_normalized = "create_new"
        elif route_raw in ("append_routed", "append"):
            route_normalized = "append_routed"
        else:
            route_normalized = route_raw
        # Normalize topology: legacy routes had topology baked in
        # "lancedb-aggregated" implies topology="aggregate"; others are federated
        topo_raw = data.get("topology")
        if topo_raw is None:
            topo_normalized = "aggregate" if backend_normalized == "lance" else "federated"
        else:
            topo_normalized = str(topo_raw)
        document = cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            dataset_id=str(data["dataset_id"]),
            release_id=str(data["release_id"]),
            route=route_normalized,
            backend=backend_normalized,
            topology=topo_normalized,
            count_source=CountSourceSpec.from_dict(data["count_source"]),
            outputs=OutputRoots.from_dict(data["outputs"]),
            provenance=ProvenanceSpec.from_dict(data["provenance"]),
            raw_cell_meta_path=data.get("raw_cell_meta_path"),
            raw_feature_meta_path=data.get("raw_feature_meta_path"),
            accepted_schema_path=data.get("accepted_schema_path"),
            metadata_summary_path=data.get("metadata_summary_path"),
            provenance_spec_path=data.get("provenance_spec_path"),
            feature_meta_paths={
                k: str(v) for k, v in (data.get("feature_meta_paths") or {}).items()
            },
            size_factor_manifest_path=data.get("size_factor_manifest_path"),
            qa_manifest_path=data.get("qa_manifest_path"),
            hvg_sidecar_path=data.get("hvg_sidecar_path"),
            integer_verified=bool(data.get("integer_verified", False)),
            cell_count=int(data.get("cell_count", 0)),
            feature_count=int(data.get("feature_count", 0)),
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
class DatasetMetadataSummary(YamlDocument):
    """Per-dataset metadata summary persisted during materialization for later schema review.

    This summary provides field-level coverage statistics (null fractions,
    dtype, unique value counts) extracted from the raw obs/var before any canonical
    mapping. It is written as a dataset-local artifact alongside the accepted schema
    and can be used to assist schema review when onboarding new datasets.
    """

    kind: str
    contract_version: str
    dataset_id: str
    release_id: str
    source_path: str
    obs_field_count: int
    var_field_count: int
    obs_null_fractions: dict[str, float]
    var_null_fractions: dict[str, float]
    obs_dtypes: dict[str, str]
    var_dtypes: dict[str, str]
    obs_rows: int
    var_rows: int
    obs_index_name: str
    var_index_name: str
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.kind != "dataset-metadata-summary":
            raise ValueError("dataset metadata summary kind mismatch")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("dataset metadata summary contract version mismatch")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetMetadataSummary":
        document = cls(
            kind=str(data["kind"]),
            contract_version=str(data["contract_version"]),
            dataset_id=str(data["dataset_id"]),
            release_id=str(data["release_id"]),
            source_path=str(data["source_path"]),
            obs_field_count=int(data.get("obs_field_count", 0)),
            var_field_count=int(data.get("var_field_count", 0)),
            obs_null_fractions=dict(data.get("obs_null_fractions", {})),
            var_null_fractions=dict(data.get("var_null_fractions", {})),
            obs_dtypes=dict(data.get("obs_dtypes", {})),
            var_dtypes=dict(data.get("var_dtypes", {})),
            obs_rows=int(data.get("obs_rows", 0)),
            var_rows=int(data.get("var_rows", 0)),
            obs_index_name=str(data.get("obs_index_name", "")),
            var_index_name=str(data.get("var_index_name", "")),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )
        document.validate()
        return document


@dataclass(frozen=True)
class FeatureProvenanceSpec(YamlDocument):
    """Feature-order and provenance artifact written during materialization.

    Records the per-dataset feature ordering (origin_index in original dataset
    var order) and the provenance of each feature (source h5ad, source var index,
    accepted schema used). Used by ``canonicalize-meta`` to rebuild the corpus
    feature set without requiring a tokenizer.
    """

    release_id: str
    feature_count: int
    source_path: str
    schema_path: str
    count_source: CountSourceSpec
    origin_index_to_feature_id: dict[int, str]
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureProvenanceSpec":
        return cls(
            release_id=str(data["release_id"]),
            feature_count=int(data.get("feature_count", 0)),
            source_path=str(data.get("source_path", "")),
            schema_path=str(data.get("schema_path", "")),
            count_source=CountSourceSpec.from_dict(data.get("count_source", {})),
            origin_index_to_feature_id=dict(data.get("origin_index_to_feature_id", {})),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


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
    """A single dataset's entry in the corpus index.

    This record is the authoritative source for dataset membership, append
    order, and global cell routing within the Arrow/HF-only plan scope.

    Fields required for correct routing:
    - ``dataset_id`` — stable dataset identifier
    - ``release_id`` — immutable processed release identifier
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
    release_id: str
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
            release_id=str(data["release_id"]),
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
    feature_registry_id: str  # deprecated: replaced by tokenizer_path in contract 0.2.0
    missing_value_literal: str
    raw_field_policy: str
    backend: str | None = None  # arrow-hf | webdataset | zarr | lance (legacy names normalized on read)
    topology: str | None = None  # federated | aggregate (Stage 2 contract: separate from backend)
    tokenizer_path: str | None = None  # relative path from corpus root to tokenizer.json
    emission_spec_path: str | None = None  # relative path from corpus root to corpus-emission-spec.yaml
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.backend is not None and self.backend not in {
            "arrow-hf",
            "webdataset",
            "zarr",
            "lance",
        }:
            raise ValueError(f"invalid backend in global-metadata: {self.backend}")
        if self.topology is not None and self.topology not in {"federated", "aggregate"}:
            raise ValueError(f"invalid topology in global-metadata: {self.topology}")

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
            backend=data.get("backend"),  # may be None for backward compat with older files
            tokenizer_path=data.get("tokenizer_path"),
            emission_spec_path=data.get("emission_spec_path"),
            notes=tuple(str(item) for item in data.get("notes", [])),
        )

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "GlobalMetadataDocument":
        return cls.from_dict(cls._load_yaml_dict(file_path))
