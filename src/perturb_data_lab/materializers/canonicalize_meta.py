"""Phase 4 canonicalize-meta: corpus-level canonical metadata rebuild from raw artifacts.

This module provides the ``CanonicalizeMetaRoute`` class that:

1. Reads the corpus index to enumerate all registered datasets.
2. For each dataset, loads its raw cell metadata (SQLite), raw feature metadata
   (Parquet), and accepted schema (YAML).
3. Applies the schema's field resolution to raw obs/var rows, producing
   per-dataset canonical cell and feature metadata.
4. Assigns global cell index ranges (contiguous, deterministic) across the corpus.
5. Concatenates per-dataset canonical metadata into corpus-level outputs.
6. Rebuilds the corpus feature set from all datasets' feature provenance specs,
   writing a pickle-backed feature set and corpus-level canonical feature metadata.
7. Overwrites prior corpus canonical metadata outputs in place.

Raw metadata is the sole input — prior canonical metadata is NOT reused.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL
from ..inspectors.models import SchemaDocument
from .models import (
    CellMetadataRecord,
    CorpusIndexDocument,
    DatasetJoinRecord,
    DatasetMetadataSummary,
    FeatureProvenanceSpec,
    GlobalMetadataDocument,
    MaterializationManifest,
    RawCellMetadataRecord,
    RawFeatureMetadataRecord,
)
from .schema_execution import resolve_all_cell_rows, resolve_all_feature_rows

__all__ = ["CanonicalizeMetaRoute", "CorpusCellIndexRange", "run_canonicalize_meta"]


# ---------------------------------------------------------------------------
# Corpus cell index range bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusCellIndexRange:
    """Global cell index range for a dataset within the corpus.

    Range convention: inclusive start, exclusive end — same as Python slicing.
    This makes global-to-local routing trivial: a global index falls within
    [global_start, global_end) if and only if it is in the owning dataset's range.
    """

    dataset_id: str
    release_id: str
    global_start: int  # inclusive
    global_end: int  # exclusive

    @property
    def count(self) -> int:
        return self.global_end - self.global_start

    def contains_global(self, global_idx: int) -> bool:
        return self.global_start <= global_idx < self.global_end

    def to_local(self, global_idx: int) -> int:
        """Convert a global index to a local dataset index."""
        return global_idx - self.global_start


# ---------------------------------------------------------------------------
# CanonicalizeMetaRoute
# ---------------------------------------------------------------------------


class CanonicalizeMetaRoute:
    """Rebuild corpus canonical metadata from raw metadata plus accepted schemas.

    This route is corpus-level: it reads all datasets' raw artifacts, generates
    per-dataset canonical metadata, concatenates them into corpus outputs, and
    overwrites any prior canonical metadata outputs in place.

    Parameters
    ----------
    corpus_index_path : Path
        Path to the corpus index YAML. The corpus root is derived as the parent.
    """

    def __init__(self, corpus_index_path: Path):
        self._corpus_index_path = corpus_index_path
        self._corpus_root = corpus_index_path.parent

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(self) -> "CanonicalizeMetaResult":
        """Execute the full canonicalize-meta rebuild.

        Returns
        -------
        CanonicalizeMetaResult
            Summary of all outputs written and cell ranges assigned.
        """
        # --- Load corpus index ---
        corpus_index_path = self._corpus_index_path
        if not corpus_index_path.exists():
            raise FileNotFoundError(f"corpus index not found: {corpus_index_path}")

        corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)
        if not corpus.datasets:
            raise ValueError("corpus index has no datasets; nothing to canonicalize")

        # -----------------------------------------------------------------
        # Phase 4 step 1: Enumerate datasets and collect feature counts
        # -----------------------------------------------------------------
        dataset_records: list[_DatasetRawArtifacts] = []
        for ds_record in corpus.datasets:
            # Skip placeholder records
            if ds_record.dataset_id.startswith("__"):
                continue
            artifacts = self._load_dataset_artifacts(ds_record)
            dataset_records.append(artifacts)

        # -----------------------------------------------------------------
        # Phase 4 step 2: Assign global cell index ranges (contiguous, stable)
        # -----------------------------------------------------------------
        cell_ranges = self._assign_global_cell_ranges(dataset_records)

        # -----------------------------------------------------------------
        # Phase 4 step 3: Load global metadata (for backend and other corpus-level fields)
        # -----------------------------------------------------------------
        gmeta = self._load_global_metadata()

        # -----------------------------------------------------------------
        # Phase 4 step 4: Canonicalize cell metadata per dataset
        # -----------------------------------------------------------------
        cell_range: CorpusCellIndexRange
        canonical_cells: list[CellMetadataRecord] = []
        for artifacts, cell_range in zip(dataset_records, cell_ranges):
            cells = self._canonicalize_dataset_cells(artifacts, cell_range)
            canonical_cells.extend(cells)

        # -----------------------------------------------------------------
        # Phase 4 step 5: Canonicalize feature metadata per dataset
        # -----------------------------------------------------------------
        all_feature_records: list[dict[str, Any]] = []
        provenance_specs: list[FeatureProvenanceSpec] = []
        for artifacts in dataset_records:
            feat_records, prov_spec = self._canonicalize_dataset_features(artifacts)
            all_feature_records.extend(feat_records)
            provenance_specs.append(prov_spec)

        # -----------------------------------------------------------------
        # Phase 4 step 6: Rebuild corpus feature set from provenance specs
        # -----------------------------------------------------------------
        feature_set_path, token_to_feature = self._rebuild_corpus_feature_set(
            provenance_specs
        )

        # -----------------------------------------------------------------
        # Phase 4 step 7: Write corpus-level outputs
        # -----------------------------------------------------------------
        corpus_cell_meta_path = self._write_corpus_cell_metadata(canonical_cells)
        corpus_feature_meta_path = self._write_corpus_feature_metadata(
            all_feature_records, token_to_feature
        )

        # -----------------------------------------------------------------
        # Phase 4 step 8: Update global metadata with feature set path
        # -----------------------------------------------------------------
        self._update_global_metadata(feature_set_path=feature_set_path)

        # -----------------------------------------------------------------
        # Phase 4 step 9: Update corpus index with cell ranges and feature set path
        # -----------------------------------------------------------------
        self._update_corpus_index_cell_ranges(cell_ranges, feature_set_path)

        return CanonicalizeMetaResult(
            corpus_id=corpus.corpus_id,
            datasets=len(dataset_records),
            total_cells=sum(r.count for r in cell_ranges),
            cell_ranges=tuple(cell_ranges),
            corpus_cell_meta_path=corpus_cell_meta_path,
            corpus_feature_meta_path=corpus_feature_meta_path,
            feature_set_path=feature_set_path,
        )

    # -----------------------------------------------------------------------
    # Per-dataset artifact loading
    # -----------------------------------------------------------------------

    def _load_dataset_artifacts(
        self, ds_record: DatasetJoinRecord
    ) -> "_DatasetRawArtifacts":
        """Load all raw artifacts needed for canonicalization for one dataset."""
        manifest_path = self._corpus_root / ds_record.manifest_path
        manifest = MaterializationManifest.from_yaml_file(manifest_path)
        meta_root = manifest_path.parent

        # Load accepted schema
        if not manifest.accepted_schema_path:
            raise ValueError(
                f"dataset {ds_record.dataset_id} has no accepted_schema_path — "
                "canonicalize-meta requires the accepted schema artifact"
            )
        schema_path = meta_root / manifest.accepted_schema_path
        schema = SchemaDocument.from_yaml_file(schema_path)

        # Load raw cell metadata (SQLite)
        if not manifest.raw_cell_meta_path:
            raise ValueError(
                f"dataset {ds_record.dataset_id} has no raw_cell_meta_path"
            )
        raw_cell_sqlite = meta_root / manifest.raw_cell_meta_path

        # Load raw feature metadata (Parquet)
        if not manifest.raw_feature_meta_path:
            raise ValueError(
                f"dataset {ds_record.dataset_id} has no raw_feature_meta_path"
            )
        raw_feature_parquet = meta_root / manifest.raw_feature_meta_path

        # Load metadata summary (optional, for field coverage reference)
        metadata_summary: DatasetMetadataSummary | None = None
        if manifest.metadata_summary_path:
            summary_path = meta_root / manifest.metadata_summary_path
            if summary_path.exists():
                metadata_summary = DatasetMetadataSummary.from_yaml_file(summary_path)

        return _DatasetRawArtifacts(
            dataset_id=ds_record.dataset_id,
            release_id=ds_record.release_id,
            manifest=manifest,
            schema=schema,
            raw_cell_sqlite=raw_cell_sqlite,
            raw_feature_parquet=raw_feature_parquet,
            metadata_summary=metadata_summary,
        )

    def _assign_global_cell_ranges(
        self, dataset_records: list["_DatasetRawArtifacts"]
    ) -> list[CorpusCellIndexRange]:
        """Assign contiguous global cell index ranges across all datasets.

        Append order from corpus index is authoritative. Ranges are deterministic
        and stable: the same datasets in the same order always produce the same ranges.
        """
        ranges: list[CorpusCellIndexRange] = []
        global_start = 0
        for artifacts in dataset_records:
            n_cells = self._count_cells_in_raw_sqlite(artifacts.raw_cell_sqlite)
            global_end = global_start + n_cells
            ranges.append(
                CorpusCellIndexRange(
                    dataset_id=artifacts.dataset_id,
                    release_id=artifacts.release_id,
                    global_start=global_start,
                    global_end=global_end,
                )
            )
            global_start = global_end
        return ranges

    def _count_cells_in_raw_sqlite(self, sqlite_path: Path) -> int:
        """Return the number of rows in the raw cell metadata SQLite."""
        conn = sqlite3.connect(str(sqlite_path))
        try:
            cur = conn.execute("SELECT COUNT(*) FROM raw_cell_meta")
            count = int(cur.fetchone()[0])
        finally:
            conn.close()
        return count

    def _load_global_metadata(self) -> GlobalMetadataDocument | None:
        """Load global-metadata.yaml from corpus root, if present."""
        gmeta_path = self._corpus_root / "global-metadata.yaml"
        if gmeta_path.exists():
            return GlobalMetadataDocument.from_yaml_file(gmeta_path)
        return None

    # -----------------------------------------------------------------------
    # Canonicalization
    # -----------------------------------------------------------------------

    def _canonicalize_dataset_cells(
        self,
        artifacts: "_DatasetRawArtifacts",
        cell_range: CorpusCellIndexRange,
    ) -> list[CellMetadataRecord]:
        """Canonicalize cell metadata for one dataset using its accepted schema.

        Parameters
        ----------
        artifacts : _DatasetRawArtifacts
            Raw cell/feature metadata, schema, and manifest for this dataset.
        cell_range : CorpusCellIndexRange
            The global cell index range assigned to this dataset.

        Returns
        -------
        list[CellMetadataRecord]
            Canonical cell metadata records in original cell order.
        """
        schema = artifacts.schema
        raw_cell_sqlite = artifacts.raw_cell_sqlite

        # Load raw cell metadata rows from SQLite
        conn = sqlite3.connect(str(raw_cell_sqlite))
        try:
            cur = conn.execute(
                "SELECT cell_id, raw_obs FROM raw_cell_meta ORDER BY rowid"
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        # Parse raw obs JSON for each row and batch-resolve via schema
        raw_obs_list: list[dict[str, Any]] = []
        for cell_id, raw_obs_json in rows:
            raw_obs = json.loads(raw_obs_json) if raw_obs_json else {}
            raw_obs_list.append(raw_obs)

        # Batch resolve via schema execution
        import pandas as pd

        obs_df = pd.DataFrame.from_dict(raw_obs_list)
        if obs_df.empty:
            return []

        # Determine size factor for each cell
        # For now, use 1.0 as placeholder; real size factors were computed during
        # materialization and stored in the per-dataset backend output.
        # canonicalize-meta reads raw metadata only and does not re-read expression data,
        # so we use a neutral size factor here. The per-dataset expression readers
        # will provide real size factors at load time.
        size_factors = [1.0] * len(obs_df)

        # Resolve canonical perturbation and context fields
        perturbations, contexts = resolve_all_cell_rows(schema, obs_df)

        # Build canonical cell records
        records: list[CellMetadataRecord] = []
        for i, (cell_id, raw_obs_json) in enumerate(rows):
            raw_obs = json.loads(raw_obs_json) if raw_obs_json else {}
            records.append(
                CellMetadataRecord(
                    cell_id=str(cell_id),
                    dataset_id=artifacts.dataset_id,
                    dataset_release=artifacts.release_id,
                    raw_fields=dict(raw_obs),
                    canonical_perturbation=dict(perturbations[i]),
                    canonical_context=dict(contexts[i]),
                    size_factor=size_factors[i],
                )
            )

        return records

    def _canonicalize_dataset_features(
        self, artifacts: "_DatasetRawArtifacts"
    ) -> tuple[list[dict[str, Any]], FeatureProvenanceSpec]:
        """Canonicalize feature metadata for one dataset using its accepted schema.

        Returns
        -------
        tuple[list[dict[str, Any]], FeatureProvenanceSpec]
            Canonical feature records as dicts (for parquet write),
            and the feature provenance spec for this dataset.
        """
        schema = artifacts.schema
        raw_feature_parquet = artifacts.raw_feature_parquet

        # Load raw feature metadata from Parquet
        table = pq.read_table(str(raw_feature_parquet))
        n_vars = table.num_rows

        # Parse raw_var JSON for each feature row
        raw_var_col = table.column("raw_var")
        origin_index_col = table.column("origin_index")
        feature_id_col = table.column("feature_id")

        raw_var_rows: list[dict[str, Any]] = []
        for i in range(n_vars):
            raw_var_json = raw_var_col[i].as_py()
            raw_var = json.loads(raw_var_json) if raw_var_json else {}
            raw_var_rows.append(raw_var)

        # Build DataFrame for batch resolution
        import pandas as pd

        feat_df = pd.DataFrame.from_dict(raw_var_rows)
        if feat_df.empty:
            raise ValueError(
                f"no feature metadata found in {raw_feature_parquet} for dataset {artifacts.dataset_id}"
            )

        # Resolve canonical feature fields via schema execution
        canonical_feature_dicts = resolve_all_feature_rows(schema, feat_df)

        # Build feature records with origin_index for ordering
        feature_records: list[dict[str, Any]] = []
        origin_index_to_feature_id: dict[int, str] = {}
        for i in range(n_vars):
            origin_idx = origin_index_col[i].as_py()
            feature_id = feature_id_col[i].as_py()
            origin_index_to_feature_id[int(origin_idx)] = str(feature_id)
            rec = {
                "origin_index": int(origin_idx),
                "feature_id": str(feature_id),
                **canonical_feature_dicts[i],
            }
            feature_records.append(rec)

        # Build provenance spec (needed for corpus feature set rebuild)
        prov_spec = FeatureProvenanceSpec(
            release_id=artifacts.release_id,
            feature_count=n_vars,
            source_path=str(artifacts.manifest.provenance.source_path),
            schema_path=str(artifacts.schema.path if hasattr(artifacts.schema, "path") else ""),
            count_source=artifacts.manifest.count_source,
            origin_index_to_feature_id=origin_index_to_feature_id,
        )

        return feature_records, prov_spec

    # -----------------------------------------------------------------------
    # Corpus feature set rebuild
    # -----------------------------------------------------------------------

    def _rebuild_corpus_feature_set(
        self, provenance_specs: list[FeatureProvenanceSpec]
    ) -> tuple[Path, dict[int, str]]:
        """Rebuild the corpus feature set from per-dataset provenance specs.

        The corpus feature set records which features exist across all datasets
        and their mapping to global token IDs. Features are assigned token IDs
        in deterministic order: first by dataset append order, then by
        origin_index within each dataset.

        Parameters
        ----------
        provenance_specs : list[FeatureProvenanceSpec]
            One provenance spec per dataset, in corpus append order.

        Returns
        -------
        tuple[Path, dict[int, str]]
            Path to the written pickle file, and the token-to-feature_id mapping.
        """
        # Collect all unique (dataset, origin_index) → feature_id entries
        # in deterministic order
        entries: list[tuple[int, int, str]] = []  # (token_id, origin_index, feature_id)
        token_id = 0
        for prov_spec in provenance_specs:
            for origin_idx in sorted(prov_spec.origin_index_to_feature_id.keys()):
                feature_id = prov_spec.origin_index_to_feature_id[origin_idx]
                entries.append((token_id, origin_idx, feature_id))
                token_id += 1

        total_features = token_id

        # Build token → feature_id mapping
        token_to_feature_id: dict[int, str] = {
            token: feature_id for token, _, feature_id in entries
        }

        # Write pickle-backed feature set
        feature_set_path = self._corpus_root / "corpus-feature-set.pkl"
        feature_set = _CorpusFeatureSet(
            entries={
                token: feature_id for token, _, feature_id in entries
            },
            total_features=total_features,
        )
        with open(feature_set_path, "wb") as f:
            pickle.dump(feature_set, f)

        return feature_set_path, token_to_feature_id

    # -----------------------------------------------------------------------
    # Corpus-level output writers
    # -----------------------------------------------------------------------

    def _write_corpus_cell_metadata(
        self, canonical_cells: list[CellMetadataRecord]
    ) -> Path:
        """Write concatenated corpus cell metadata as Parquet.

        Each record carries global identity fields plus canonical perturbation/context
        and raw fields. The file is written in global cell index order.
        """
        import pyarrow as pa

        cell_ids = [c.cell_id for c in canonical_cells]
        dataset_ids = [c.dataset_id for c in canonical_cells]
        dataset_releases = [c.dataset_release for c in canonical_cells]
        size_factors = [c.size_factor for c in canonical_cells]
        raw_fields = [json.dumps(c.raw_fields) for c in canonical_cells]
        canonical_perturbation = [json.dumps(c.canonical_perturbation) for c in canonical_cells]
        canonical_context = [json.dumps(c.canonical_context) for c in canonical_cells]

        table = pa.table(
            {
                "cell_id": pa.array(cell_ids, type=pa.string()),
                "dataset_id": pa.array(dataset_ids, type=pa.string()),
                "dataset_release": pa.array(dataset_releases, type=pa.string()),
                "size_factor": pa.array(size_factors, type=pa.float64()),
                "raw_fields": pa.array(raw_fields, type=pa.string()),
                "canonical_perturbation": pa.array(canonical_perturbation, type=pa.string()),
                "canonical_context": pa.array(canonical_context, type=pa.string()),
            }
        )

        output_path = self._corpus_root / "corpus-cell-metadata.parquet"
        pq.write_table(table, output_path)
        return output_path

    def _write_corpus_feature_metadata(
        self,
        all_feature_records: list[dict[str, Any]],
        token_to_feature_id: dict[int, str],
    ) -> Path:
        """Write concatenated corpus feature metadata as Parquet.

        Each record carries origin_index, global token_id, feature_id, and
        canonical feature fields.
        """
        import pyarrow as pa

        origin_indices = [r["origin_index"] for r in all_feature_records]
        feature_ids = [r["feature_id"] for r in all_feature_records]
        # Assign token IDs in the order features appear in all_feature_records
        # (which follows corpus append order and per-dataset origin_index order)
        token_ids = list(range(len(all_feature_records)))

        # Separate canonical fields from structural ones
        canonical_field_names = [
            k for k in all_feature_records[0].keys()
            if k not in ("origin_index", "feature_id")
        ] if all_feature_records else []

        columns: dict[str, pa.Array] = {
            "origin_index": pa.array(origin_indices, type=pa.int32()),
            "token_id": pa.array(token_ids, type=pa.int32()),
            "feature_id": pa.array(feature_ids, type=pa.string()),
        }

        # Add canonical struct column
        if canonical_field_names:
            canonical_struct = pa.struct([
                (fname, pa.string()) for fname in canonical_field_names
            ])
            canonical_values = [
                [str(rec.get(fname, MISSING_VALUE_LITERAL)) for rec in all_feature_records]
                for fname in canonical_field_names
            ]
            canonical_array = pa.array(
                [dict(zip(canonical_field_names, vals)) for vals in zip(*canonical_values)],
                type=canonical_struct,
            )
            columns["canonical"] = canonical_array

        table = pa.table(columns)
        output_path = self._corpus_root / "corpus-feature-metadata.parquet"
        pq.write_table(table, output_path)
        return output_path

    def _update_global_metadata(self, feature_set_path: Path) -> None:
        """Update global-metadata.yaml with the feature set path."""
        gmeta_path = self._corpus_root / "global-metadata.yaml"
        if gmeta_path.exists():
            gmeta = GlobalMetadataDocument.from_yaml_file(gmeta_path)
            updated = GlobalMetadataDocument(
                kind=gmeta.kind,
                contract_version=gmeta.contract_version,
                schema_version=gmeta.schema_version,
                feature_registry_id=gmeta.feature_registry_id,
                missing_value_literal=gmeta.missing_value_literal,
                raw_field_policy=gmeta.raw_field_policy,
                backend=gmeta.backend,
                tokenizer_path=None,  # No tokenizer in new architecture
                emission_spec_path=gmeta.emission_spec_path,
                notes=gmeta.notes,
            )
            updated.write_yaml(gmeta_path)
        else:
            # Create minimal global metadata
            gmeta_dict = {
                "kind": "global-metadata",
                "contract_version": CONTRACT_VERSION,
                "schema_version": CONTRACT_VERSION,
                "missing_value_literal": MISSING_VALUE_LITERAL,
                "raw_field_policy": "preserve-unchanged",
            }
            GlobalMetadataDocument.from_dict(gmeta_dict).write_yaml(gmeta_path)

    def _update_corpus_index_cell_ranges(
        self,
        cell_ranges: list[CorpusCellIndexRange],
        feature_set_path: Path,
    ) -> None:
        """Update corpus-index.yaml with global cell ranges and feature set path.

        This overwrites the corpus index in place with updated entries that
        include per-dataset cell counts and global ranges.
        """
        corpus_index_path = self._corpus_index_path
        corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)

        # Build updated global metadata dict
        global_meta = dict(corpus.global_metadata) if corpus.global_metadata else {}
        global_meta["feature_set_path"] = str(feature_set_path.relative_to(self._corpus_root))
        global_meta["total_corpus_cells"] = sum(r.count for r in cell_ranges)
        global_meta["cell_index_ranges"] = [
            {
                "dataset_id": r.dataset_id,
                "release_id": r.release_id,
                "global_start": r.global_start,
                "global_end": r.global_end,
                "count": r.count,
            }
            for r in cell_ranges
        ]

        # Build updated dataset records with cell counts
        updated_datasets: list[DatasetJoinRecord] = []
        for ds_record in corpus.datasets:
            if ds_record.dataset_id.startswith("__"):
                updated_datasets.append(ds_record)
                continue
            # Find the range for this dataset
            range_ = None
            for r in cell_ranges:
                if r.dataset_id == ds_record.dataset_id:
                    range_ = r
                    break
            if range_ is None:
                updated_datasets.append(ds_record)
                continue

            # Extend the join record with cell count and range via a new class
            # (DatasetJoinRecord itself is not versioned for this, so we store
            # the extra fields in global_metadata per dataset)
            updated_datasets.append(
                DatasetJoinRecord(
                    dataset_id=ds_record.dataset_id,
                    release_id=ds_record.release_id,
                    join_mode=ds_record.join_mode,
                    manifest_path=ds_record.manifest_path,
                )
            )

        updated_corpus = CorpusIndexDocument(
            kind=corpus.kind,
            contract_version=corpus.contract_version,
            corpus_id=corpus.corpus_id,
            global_metadata=global_meta,
            datasets=tuple(updated_datasets),
        )
        updated_corpus.write_yaml(corpus_index_path)


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DatasetRawArtifacts:
    """Container for all raw artifacts needed for one dataset's canonicalization."""

    dataset_id: str
    release_id: str
    manifest: MaterializationManifest
    schema: SchemaDocument
    raw_cell_sqlite: Path
    raw_feature_parquet: Path
    metadata_summary: DatasetMetadataSummary | None = None


@dataclass(frozen=True)
class _CorpusFeatureSet:
    """Pickle-backed corpus feature set: maps token IDs to feature identifiers."""

    entries: dict[int, str] = field(default_factory=dict)
    total_features: int = 0


@dataclass(frozen=True)
class CanonicalizeMetaResult:
    """Result summary from a canonicalize-meta run."""

    corpus_id: str
    datasets: int
    total_cells: int
    cell_ranges: tuple[CorpusCellIndexRange, ...]
    corpus_cell_meta_path: Path
    corpus_feature_meta_path: Path
    feature_set_path: Path


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def run_canonicalize_meta(corpus_index_path: str | Path) -> CanonicalizeMetaResult:
    """Convenience entry point for canonicalize-meta.

    Parameters
    ----------
    corpus_index_path : str | Path
        Path to the corpus index YAML file.

    Returns
    -------
    CanonicalizeMetaResult
        Summary of all outputs written.
    """
    route = CanonicalizeMetaRoute(Path(corpus_index_path))
    return route.run()
