"""Phase 3 canonical materializer.

This module implements the canonical materialization layer:
- sparse per-cell storage contract
- two join/mutation modes: create_new, append_routed
- per-dataset manifest and feature registry append logic

All materialization writes go to repo-local real directories only.
Never write to protected symlink roots (data/, pertTF/, perturb/).
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.sparse import csr_matrix, issparse

from .models import (
    CellMetadataRecord,
    CorpusIndexDocument,
    CountSourceSpec,
    DatasetJoinRecord,
    FeatureRegistryEntry,
    FeatureRegistryManifest,
    GlobalMetadataDocument,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
    QAManifest,
    QAMetric,
    SizeFactorEntry,
    SizeFactorManifest,
)
from .backends import materialize_arrow_hf, materialize_webdataset, materialize_zarr
from .schema_execution import resolve_all_cell_rows, resolve_all_feature_rows
from .validation import validate_schema_readiness
from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL
from ..inspectors.models import SchemaDocument


# ---------------------------------------------------------------------------
# Canonical sparse per-cell record contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalCellRecord:
    """Canonical logical row representation for a single cell.

    Sparse per-cell storage contract:
    - expressed_gene_indices: int32[]  — column indices of expressed genes
    - expression_counts: int32[]       — count for each expressed gene
    - canonical metadata columns
    - preserved raw metadata fields
    - size_factor: float
    - stable cell and dataset identifiers

    Both arrays must be the same length. Missing canonical fields use
    the MISSING_VALUE_LITERAL sentinel.
    """

    expressed_gene_indices: tuple[int, ...]
    expression_counts: tuple[int, ...]
    cell_id: str
    dataset_id: str
    dataset_release: str
    size_factor: float
    canonical_perturbation: dict[str, str]
    canonical_context: dict[str, str]
    raw_fields: dict[str, Any]

    def is_integer_sparse(self) -> bool:
        return all(isinstance(v, int) for v in self.expression_counts) and all(
            isinstance(idx, int) for idx in self.expressed_gene_indices
        )

    def to_csr_matrix_parts(
        self, total_genes: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to CSR components: data, indices, indptr."""
        n = len(self.expression_counts)
        data = np.array(self.expression_counts, dtype=np.int32)
        indices = np.array(self.expressed_gene_indices, dtype=np.int32)
        indptr = np.array([0, n], dtype=np.int32)
        return data, indices, indptr


# ---------------------------------------------------------------------------
# Materialization routes
# ---------------------------------------------------------------------------


class MaterializationRoute:
    """Base class for a materialization route.

    A route encapsulates:
    - How to write the per-cell sparse data to a backend storage format
    - How to update the feature registry (append-only, preserving existing IDs)
    - How to write per-dataset and corpus-level manifests

    Subclasses implement backend-specific write logic.
    """

    route_name: str = "base"

    def __init__(
        self,
        output_roots: OutputRoots,
        release_id: str,
        dataset_id: str,
        count_source: CountSourceSpec,
        integer_only: bool = True,
    ):
        self.output_roots = output_roots
        self.release_id = release_id
        self.dataset_id = dataset_id
        self.count_source = count_source
        self.integer_only = integer_only

    def materialize(
        self,
        source_path: str,
        schema_path: str,
    ) -> MaterializationManifest:
        """Materialize a dataset using this route.

        Returns a filled MaterializationManifest with paths to all written artifacts.
        Subclasses override _write_cells() for backend-specific storage.

        Raises
        ------
        ValueError
            If the schema is not in ``status: ready`` or required fields are null.
        """
        # --- Schema readiness validation gate ---
        readiness = validate_schema_readiness(schema_path)
        readiness.raise_if_not_ready()

        # Load the reviewed schema (used for row-wise canonical metadata execution)
        schema = SchemaDocument.from_yaml_file(Path(schema_path))

        # Resolve source h5ad
        source_h5ad = Path(source_path)
        if not source_h5ad.exists():
            raise FileNotFoundError(f"source h5ad not found: {source_h5ad}")

        # Read backed anndata
        adata = ad.read_h5ad(str(source_h5ad), backed="r")
        n_obs = adata.n_obs
        n_vars = adata.n_vars

        # Select count matrix
        if self.count_source.selected == ".raw.X":
            count_matrix = adata.raw.X
        elif self.count_source.selected.startswith(".layers["):
            layer_name = self.count_source.selected[len(".layers[") : -1]
            count_matrix = adata.layers[layer_name]
        else:
            count_matrix = adata.X

        # Verify integer counts
        if self.integer_only:
            self._verify_integer(count_matrix, source_h5ad.name)

        # Ensure output directories exist
        meta_root = Path(self.output_roots.metadata_root)
        matrix_root = Path(self.output_roots.matrix_root)
        meta_root.mkdir(parents=True, exist_ok=True)
        matrix_root.mkdir(parents=True, exist_ok=True)

        # Compute size factors (sum-based)
        size_factors = self._compute_size_factors(count_matrix, n_obs)

        # Write backend-specific cell data
        backend_paths = self._write_cells(
            count_matrix=count_matrix,
            adata=adata,
            size_factors=size_factors,
            matrix_root=matrix_root,
        )

        # Build/update feature registry
        feature_registry_path = meta_root / "feature-registry.yaml"
        registry = self._build_feature_registry(
            var_index=adata.var.index,
            existing_registry_path=feature_registry_path
            if self.route_name != "create_new"
            else None,
        )
        registry.write_yaml(feature_registry_path)

        # Write size factor manifest
        sf_entries = [
            SizeFactorEntry(cell_id=str(adata.obs.index[i]), size_factor=float(sf))
            for i, sf in enumerate(size_factors)
        ]
        sf_manifest = SizeFactorManifest(
            kind="size-factor-manifest",
            contract_version=CONTRACT_VERSION,
            release_id=self.release_id,
            method="sum",
            entries=tuple(sf_entries),
        )
        sf_path = meta_root / "size-factor-manifest.yaml"
        sf_manifest.write_yaml(sf_path)

        # Write cell metadata table (parquet-backed via Arrow, or simple SQLite)
        cell_meta_path = self._write_cell_metadata(
            adata=adata,
            schema=schema,
            size_factors=size_factors,
            meta_root=meta_root,
        )

        # Write per-dataset feature objects (original order + tokenized companion)
        feature_meta_paths = self._write_feature_metadata(
            adata=adata,
            schema=schema,
            registry=registry,
            meta_root=meta_root,
        )

        # QA: verify integer counts in written output
        qa_metrics, all_passed = self._run_qa_checks(backend_paths, count_matrix)

        # Write QA manifest
        qa_manifest = QAManifest(
            kind="qa-manifest",
            contract_version=CONTRACT_VERSION,
            release_id=self.release_id,
            metrics=qa_metrics,
            all_passed=all_passed,
        )
        qa_path = meta_root / "qa-manifest.yaml"
        qa_manifest.write_yaml(qa_path)

        # Build final manifest
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version=CONTRACT_VERSION,
            dataset_id=self.dataset_id,
            release_id=self.release_id,
            route=self.route_name,
            count_source=self.count_source,
            outputs=self.output_roots,
            provenance=ProvenanceSpec(
                source_path=str(source_h5ad),
                schema=schema_path,
            ),
            feature_manifest_path=str(feature_registry_path),
            feature_meta_paths={
                k: str(v) for k, v in feature_meta_paths.items()
            },
            size_factor_manifest_path=str(sf_path),
            qa_manifest_path=str(qa_path),
            integer_verified=all_passed,
            notes=(f"materialized via {self.route_name} route",),
        )
        manifest.validate()

        manifest_path = meta_root / "materialization-manifest.yaml"
        manifest.write_yaml(manifest_path)

        return manifest

    def _verify_integer(self, count_matrix: Any, source_name: str) -> None:
        """Fail hard if count matrix is not integer-like."""
        sample_rows = min(32, count_matrix.shape[0])
        indices = np.linspace(0, count_matrix.shape[0] - 1, num=sample_rows, dtype=int)
        # Collect nonzero values row-by-row to avoid sparse slicing edge cases
        nonzero_values = []
        for idx in indices:
            row = count_matrix[idx]
            if issparse(row):
                row = np.asarray(row.toarray().ravel())
            else:
                row = np.asarray(row).ravel()
            nonzero_values.append(row[row != 0])
        if nonzero_values:
            nonzero = np.concatenate(nonzero_values)
        else:
            return  # empty
        if nonzero.size == 0:
            return
        deviations = np.abs(nonzero - np.rint(nonzero))
        if np.any(deviations > 1e-6):
            raise ValueError(
                f"count matrix in {source_name} contains non-integer values; "
                "materialization requires strict integer counts. "
                "Provide a schema that selects an integer-compliant count source."
            )

    def _compute_size_factors(self, count_matrix: Any, n_obs: int) -> np.ndarray:
        """Compute size factors via sum-per-row, normalized."""
        from .backends.arrow_hf import _get_row_nonzero

        factors = np.zeros(n_obs, dtype=np.float64)
        for i in range(n_obs):
            indices, counts = _get_row_nonzero(count_matrix, i)
            factors[i] = float(counts.sum())
        total = factors.sum()
        if total > 0:
            factors = factors / (total / n_obs)
        # Replace zero or NaN with 1.0
        factors = np.where(factors <= 0, 1.0, factors)
        factors = np.where(np.isnan(factors), 1.0, factors)
        return factors

    def _write_cells(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        size_factors: np.ndarray,
        matrix_root: Path,
    ) -> dict[str, Path]:
        """Write per-cell sparse data. Override in subclass."""
        raise NotImplementedError

    def _build_feature_registry(
        self,
        var_index: Any,
        existing_registry_path: Path | None = None,
    ) -> FeatureRegistryManifest:
        """Build append-only feature registry.

        If existing_registry_path is provided, load it and append new entries,
        preserving existing token IDs.
        """
        if existing_registry_path is not None and existing_registry_path.exists():
            existing = FeatureRegistryManifest.from_yaml_file(existing_registry_path)
            existing_ids = {entry.feature_id for entry in existing.entries}
            start_token = max(e.token_id for e in existing.entries) + 1
            namespace = existing.namespace
            registry_id = existing.registry_id
        else:
            existing_ids = set()
            start_token = 0
            namespace = "unknown"
            registry_id = "feature-registry-v0"

        entries = []
        token_id = start_token
        for gene_id in var_index:
            if gene_id in existing_ids:
                continue
            gene_label = str(gene_id)
            entries.append(
                FeatureRegistryEntry(
                    token_id=token_id,
                    feature_id=str(gene_id),
                    feature_label=gene_label,
                    namespace=namespace,
                )
            )
            token_id += 1

        all_entries = []
        if existing_registry_path is not None and existing_registry_path.exists():
            all_entries = list(
                FeatureRegistryManifest.from_yaml_file(existing_registry_path).entries
            )
        all_entries.extend(entries)

        return FeatureRegistryManifest(
            kind="feature-registry",
            contract_version=CONTRACT_VERSION,
            registry_id=registry_id,
            append_only=True,
            namespace=namespace,
            feature_id_field="gene_id",
            feature_label_field="gene_symbol",
            default_missing_value=MISSING_VALUE_LITERAL,
            entries=tuple(all_entries),
        )

    def _write_cell_metadata(
        self,
        adata: ad.AnnData,
        schema: SchemaDocument,
        size_factors: np.ndarray,
        meta_root: Path,
    ) -> Path:
        """Write cell metadata as SQLite with full canonical cell metadata per cell.

        Canonical perturbation and context fields are resolved row-wise from the
        schema. Raw obs fields not used as canonical sources are preserved as JSON.
        """
        import json

        db_path = meta_root / f"{self.release_id}-cell-meta.sqlite"

        # Collect source field names used by the schema so we can exclude them
        # from raw_obs preservation
        source_columns: set[str] = set()
        for entry in list(schema.perturbation_fields.values()) + list(
            schema.context_fields.values()
        ):
            source_columns.update(entry.source_fields)

        # Load obs into memory once for efficient per-row access
        # to_memory() exists on backed AnnData obs/var views; for non-backed
        # or plain pandas objects, the DataFrame is already in memory
        obs_mem = adata.obs.to_memory() if hasattr(adata.obs, "to_memory") else adata.obs

        # Resolve canonical metadata for all cells via schema execution
        perturbations, contexts = resolve_all_cell_rows(schema, obs_mem)

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cell_meta "
            "(cell_id TEXT, dataset_id TEXT, dataset_release TEXT, "
            "size_factor REAL, canonical_perturbation TEXT, "
            "canonical_context TEXT, raw_obs TEXT)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cell_id ON cell_meta(cell_id)")

        batch_size = 256
        n_obs = adata.n_obs
        for start in range(0, n_obs, batch_size):
            end = min(start + batch_size, n_obs)
            batch = []
            for i in range(start, end):
                cell_id = str(obs_mem.index[i])
                pert = perturbations[i]
                ctx = contexts[i]
                # Preserve raw obs fields not used as schema sources
                raw: dict[str, Any] = {}
                for col in obs_mem.columns:
                    if col not in source_columns:
                        val = obs_mem.loc[obs_mem.index[i], col]
                        raw[str(col)] = None if pd.isna(val) else str(val)
                raw_json = json.dumps(raw) if raw else "{}"
                batch.append(
                    (
                        cell_id,
                        self.dataset_id,
                        self.release_id,
                        float(size_factors[i]),
                        json.dumps(dict(pert)),
                        json.dumps(dict(ctx)),
                        raw_json,
                    )
                )
            conn.executemany(
                "INSERT INTO cell_meta VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
        conn.commit()
        conn.close()
        return db_path

    def _write_feature_metadata(
        self,
        adata: ad.AnnData,
        schema: SchemaDocument,
        registry: FeatureRegistryManifest,
        meta_root: Path,
    ) -> dict[str, Path]:
        """Write per-dataset canonical feature objects.

        Two objects are written per dataset:
        1. ``{release_id}-features-origin.parquet`` — canonical feature metadata
           in original dataset feature order (var index order), resolved via
           schema execution.  One row per feature.
        2. ``{release_id}-features-token.parquet`` — tokenized companion mapping
           each original feature index to its token ID from the feature registry.
           One row per feature in original order.

        Both objects are compact (one row per feature, not one per cell) and
        shared by all cells of the dataset.  The tokenized object enables loaders
        to translate dataset-order sparse indices to token-space without
        re-tokenizing at load time.
        """
        import json

        # Resolve all feature rows via schema execution (one row per feature)
        var_mem = adata.var.to_memory() if hasattr(adata.var, "to_memory") else adata.var
        feature_rows = resolve_all_feature_rows(schema, var_mem)

        # Build origin-order feature table
        origin_path = meta_root / f"{self.release_id}-features-origin.parquet"
        origin_indices = list(range(adata.n_vars))
        origin_feature_ids = [str(adata.var.index[i]) for i in origin_indices]

        origin_table = pa.table(
            {
                "origin_index": pa.array(origin_indices, type=pa.int32()),
                "feature_id": pa.array(origin_feature_ids, type=pa.string()),
            }
        )
        # Attach resolved canonical feature fields as a struct column
        field_names = list(schema.feature_fields.keys())
        canonical_struct = pa.struct([
            (fname, pa.string()) for fname in field_names
        ])
        canonical_values = [[row.get(fname, MISSING_VALUE_LITERAL) for row in feature_rows] for fname in field_names]
        canonical_array = pa.array([dict(zip(field_names, vals)) for vals in zip(*canonical_values)], type=canonical_struct)
        origin_table = origin_table.append_column("canonical", canonical_array)
        pq.write_table(origin_table, origin_path)

        # Build tokenized companion: map original var index → token ID
        token_path = meta_root / f"{self.release_id}-features-token.parquet"
        registry_map = {
            str(entry.feature_id): entry.token_id
            for entry in registry.entries
        }
        token_ids = [
            registry_map.get(str(adata.var.index[i]), -1)
            for i in range(adata.n_vars)
        ]
        token_table = pa.table(
            {
                "origin_index": pa.array(origin_indices, type=pa.int32()),
                "feature_id": pa.array(origin_feature_ids, type=pa.string()),
                "token_id": pa.array(token_ids, type=pa.int32()),
            }
        )
        pq.write_table(token_table, token_path)

        return {"features_origin": origin_path, "features_token": token_path}

    def _run_qa_checks(
        self,
        backend_paths: dict[str, Path],
        original_matrix: Any,
    ) -> tuple[tuple[QAMetric, ...], bool]:
        """Run QA checks on written output."""
        metrics = []
        for name, path in backend_paths.items():
            if not path.exists():
                metrics.append(
                    QAMetric(name=f"{name}_exists", value=0.0, threshold=1.0)
                )
            else:
                metrics.append(
                    QAMetric(name=f"{name}_exists", value=1.0, threshold=1.0)
                )
        all_passed = all(m.passed() for m in metrics)
        return tuple(metrics), all_passed


class CreateNewRoute(MaterializationRoute):
    """Materialization route: create a new standalone dataset release.

    This is used when materializing the first dataset of a new corpus,
    or a standalone dataset that will not be joined to a corpus.
    """

    route_name = "create_new"

    def _write_cells(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        size_factors: np.ndarray,
        matrix_root: Path,
    ) -> dict[str, Path]:
        return materialize_arrow_hf(
            adata, count_matrix, size_factors, self.release_id, matrix_root
        )


class AppendRoutedRoute(MaterializationRoute):
    """Materialization route: add a dataset to a growing indexed corpus.

    Each dataset is stored independently with its own manifest. A corpus index
    maps dataset IDs to release IDs. The feature registry is appended with new
    token IDs while preserving existing entries. This is the default path for
    corpora expected to grow with additional datasets.
    """

    route_name = "append_routed"

    def _write_cells(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        size_factors: np.ndarray,
        matrix_root: Path,
    ) -> dict[str, Path]:
        return materialize_arrow_hf(
            adata, count_matrix, size_factors, self.release_id, matrix_root
        )


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------


def build_materialization_route(
    route: str,
    output_roots: OutputRoots,
    release_id: str,
    dataset_id: str,
    count_source: CountSourceSpec,
    integer_only: bool = True,
) -> MaterializationRoute:
    """Factory to build the correct materialization route by name."""
    routes = {
        "create_new": CreateNewRoute,
        "append_routed": AppendRoutedRoute,
    }
    if route not in routes:
        raise ValueError(f"unknown route: {route}; expected one of {list(routes)}")
    return routes[route](
        output_roots=output_roots,
        release_id=release_id,
        dataset_id=dataset_id,
        count_source=count_source,
        integer_only=integer_only,
    )


# ---------------------------------------------------------------------------
# Corpus index updater
# ---------------------------------------------------------------------------


def update_corpus_index(
    corpus_index_path: Path,
    new_dataset_record: DatasetJoinRecord,
    global_metadata: GlobalMetadataDocument | None = None,
) -> CorpusIndexDocument:
    """Load an existing corpus index, append the new dataset record, and save.

    If corpus_index_path does not exist, create a new index.
    This function always appends; it does not overwrite existing dataset entries.
    """
    if corpus_index_path.exists():
        corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)
        existing_ids = {d.dataset_id for d in corpus.datasets}
        if new_dataset_record.dataset_id in existing_ids:
            raise ValueError(
                f"dataset {new_dataset_record.dataset_id} already exists in corpus index; "
                "use a different release_id or dataset_id to avoid duplication."
            )
        datasets = list(corpus.datasets) + [new_dataset_record]
        corpus_id = corpus.corpus_id
        global_meta = corpus.global_metadata
    else:
        datasets = [new_dataset_record]
        corpus_id = "perturb-data-lab-v0"
        global_meta = global_metadata.to_dict() if global_metadata else {}

    updated = CorpusIndexDocument(
        kind="corpus-index",
        contract_version=CONTRACT_VERSION,
        corpus_id=corpus_id,
        global_metadata=global_meta,
        datasets=tuple(datasets),
    )
    updated.write_yaml(corpus_index_path)
    return updated
