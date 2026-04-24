"""Stage 2 materializer — schema-independent, Stage-1-gated, count-first.

Phase 3 canonical materializer (expression-first, tokenizer-free) is preserved
as the legacy schema-first path. Stage 2 adds:

- ``Stage2Materializer``: schema-independent materialization entry that accepts
  a Stage 1 ``dataset-summary.yaml`` as the only gating artifact (no schema.yaml)
- Count-first path driven by the Stage 1 approved count source decision
- Parquet raw metadata sidecars (SQLite deprecated for new artifacts)
- Backend/topology separation in all interfaces

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

from .backends import materialize_arrow_hf, materialize_webdataset, materialize_zarr
from .backends.lancedb_aggregated import mark_lance_append_committed
from .models import (
    CellMetadataRecord,
    CorpusIndexDocument,
    CountSourceSpec,
    DatasetJoinRecord,
    DatasetMetadataSummary,
    FeatureProvenanceSpec,
    GlobalMetadataDocument,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
    QAManifest,
    QAMetric,
    RawCellMetadataRecord,
    RawFeatureMetadataRecord,
    SizeFactorManifest,
    SizeFactorEntry,
)
from .schema_execution import resolve_all_cell_rows, resolve_all_feature_rows
from .validation import validate_schema_readiness
from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL
from ..inspectors.models import DatasetSummaryDocument, SchemaDocument


# ---------------------------------------------------------------------------
# Stage 2 Materializer — schema-independent, Stage-1-gated, count-first
# ---------------------------------------------------------------------------


class Stage2Materializer:
    """Schema-independent materialization entry point for Stage 2.

    This class replaces the schema-first ``MaterializationRoute.materialize()``
    entry point with a Stage-1-gated, count-first path.

    Inputs
    ------
    source_path : str
        Absolute path to the source h5ad file.
    review_bundle_path : str
        Path to the Stage 1 ``dataset-summary.yaml`` that gates this materialization.
        This is the only required gating artifact; no ``schema.yaml`` is accepted.
    output_roots : OutputRoots
        ``metadata_root`` and ``matrix_root`` for this dataset's outputs.
    release_id : str
        Immutable release identifier for this dataset version.
    dataset_id : str
        Stable dataset identifier.
    backend : str
        Storage backend: ``arrow-hf`` (default), ``webdataset``, ``zarr``.
    topology : str
        Corpus topology: ``federated`` (default) or ``aggregate``.
    rerun_stage1 : bool, default False
        If True, reruns Stage 1 inspection before materialization as a preflight
        step. The resulting ``dataset-summary.yaml`` replaces ``review_bundle_path``
        as the gating artifact.
    n_hvg : int, default 2000
        Number of top-dispersion genes to select as HVGs.

    Contract
    --------
    This materializer does NOT accept ``schema_path``. Canonical metadata mapping
    is deferred to a later stage. The count path is driven exclusively by the
    Stage 1 ``count_source_decision`` in the review bundle. Integer verification
    occurs after any approved recovery step. Raw ``obs`` and ``var`` are preserved
    in Parquet sidecars. SQLite is not produced for new artifacts.

    Backend-Topology Separation
    ---------------------------
    ``backend`` names the storage format only. ``topology`` names the corpus
    organization only. The combination determines the supported subset per
    STAGE2_CONTRACT.md Section 10. The minimum supported combination is
    ``arrow-hf`` × ``federated``.
    """

    def __init__(
        self,
        source_path: str,
        review_bundle_path: str,
        output_roots: OutputRoots,
        release_id: str,
        dataset_id: str,
        backend: str = "arrow-hf",
        topology: str = "federated",
        rerun_stage1: bool = False,
        n_hvg: int = 2000,
    ):
        self.source_path = source_path
        self.review_bundle_path = review_bundle_path
        self.output_roots = output_roots
        self.release_id = release_id
        self.dataset_id = dataset_id
        self.backend = backend
        self.topology = topology
        self.rerun_stage1 = rerun_stage1
        self.n_hvg = n_hvg

    def materialize(self) -> MaterializationManifest:
        """Run Stage 2 materialization gated by Stage 1 review bundle."""
        # --- Stage 1 gate: resolve review bundle path ---
        if self.rerun_stage1:
            summary_path = self._rerun_stage1_preflight()
        else:
            summary_path = Path(self.review_bundle_path)
            if not summary_path.exists():
                raise FileNotFoundError(
                    f"review bundle not found: {summary_path}; "
                    "pass rerun_stage1=True to run Stage 1 as preflight"
                )

        # --- Load and validate Stage 1 summary ---
        summary = DatasetSummaryDocument.from_yaml_file(summary_path)

        if summary.materialization_readiness != "pass":
            raise ValueError(
                f"materialization_readiness is '{summary.materialization_readiness}' "
                f"(expected 'pass') for dataset {self.dataset_id}; "
                f"gate review bundle: {summary_path}"
            )

        # --- Extract approved count-source decision ---
        decision = summary.count_source_decision
        # Map Stage 1 field names to CountSourceSpec fields
        count_source = CountSourceSpec(
            selected=decision.selected_candidate,
            integer_only=(decision.status == "pass"),
            uses_recovery=decision.uses_recovery,
        )

        # --- Load source h5ad ---
        source_h5ad = Path(self.source_path)
        if not source_h5ad.exists():
            raise FileNotFoundError(f"source h5ad not found: {source_h5ad}")
        adata = ad.read_h5ad(str(source_h5ad), backed="r")
        try:
            n_obs = adata.n_obs

            # Determine var space for the selected count source
            if count_source.selected == ".raw.X":
                n_vars = int(adata.raw.shape[1])
                var_ref = adata.raw.var
                var_index = adata.raw.var.index
            else:
                n_vars = adata.n_vars
                var_ref = adata.var
                var_index = adata.var.index

            # --- Select count matrix ---
            if count_source.selected == ".raw.X":
                count_matrix = adata.raw.X
            elif count_source.selected.startswith(".layers["):
                layer_name = count_source.selected[len(".layers[") : -1]
                count_matrix = adata.layers[layer_name]
            else:
                count_matrix = adata.X

            # --- Apply approved recovery ---
            if count_source.uses_recovery:
                count_matrix = self._apply_reverse_normalization(
                    count_matrix, adata, count_source.selected, n_obs
                )

            # --- Integer verification (post-recovery) ---
            self._verify_integer(count_matrix, source_h5ad.name)

            # --- Ensure output directories exist ---
            meta_root = Path(self.output_roots.metadata_root)
            matrix_root = Path(self.output_roots.matrix_root)
            meta_root.mkdir(parents=True, exist_ok=True)
            matrix_root.mkdir(parents=True, exist_ok=True)

            # --- Write raw cell metadata (Parquet, not SQLite) ---
            raw_cell_meta_parquet_path = self._write_raw_cell_metadata_parquet(
                adata=adata,
                meta_root=meta_root,
            )

            # --- Write raw feature metadata (Parquet) ---
            var_mem = var_ref.to_memory() if hasattr(var_ref, "to_memory") else var_ref
            raw_feature_meta_parquet_path = self._write_raw_feature_metadata_parquet(
                var_mem=var_mem,
                meta_root=meta_root,
            )

            # --- Write per-dataset metadata summary ---
            metadata_summary_path = self._write_metadata_summary(
                adata=adata,
                var_mem=var_mem,
                meta_root=meta_root,
            )

            # --- Compute and write size factors ---
            size_factors = self._compute_size_factors(count_matrix, n_obs)
            size_factor_manifest_path = self._write_size_factor_manifest(
                adata=adata,
                size_factors=size_factors,
                meta_root=meta_root,
            )

            # --- Write feature provenance Parquet ---
            provenance_spec_path = self._write_feature_provenance_parquet(
                var_index=var_index,
                n_vars=n_vars,
                meta_root=meta_root,
                count_source_selected=count_source.selected,
                source_path=str(source_h5ad),
            )

            # --- Write backend-specific sparse cell data ---
            backend_paths = self._write_cells(
                count_matrix=count_matrix,
                adata=adata,
                size_factors=size_factors,
                matrix_root=matrix_root,
            )

            # --- Compute and write HVG arrays ---
            hvg_sidecar_path = self._compute_and_write_hvg_arrays(
                count_matrix=count_matrix,
                n_vars=n_vars,
                meta_root=meta_root,
                n_hvg=self.n_hvg,
            )

            # --- Run QA checks ---
            qa_metrics, all_passed = self._run_qa_checks(backend_paths, count_matrix)
            qa_manifest = QAManifest(
                kind="qa-manifest",
                contract_version=CONTRACT_VERSION,
                release_id=self.release_id,
                metrics=qa_metrics,
                all_passed=all_passed,
            )
            qa_path = meta_root / "qa-manifest.yaml"
            qa_manifest.write_yaml(qa_path)

            # --- Build final manifest ---
            manifest = MaterializationManifest(
                kind="materialization-manifest",
                contract_version=CONTRACT_VERSION,
                dataset_id=self.dataset_id,
                release_id=self.release_id,
                route="create_new",
                backend=self.backend,
                topology=self.topology,
                count_source=count_source,
                outputs=self.output_roots,
                provenance=ProvenanceSpec(
                    source_path=str(source_h5ad),
                    review_bundle=str(summary_path),
                ),
                raw_cell_meta_path=str(raw_cell_meta_parquet_path),
                raw_feature_meta_path=str(raw_feature_meta_parquet_path),
                metadata_summary_path=str(metadata_summary_path),
                provenance_spec_path=str(provenance_spec_path),
                size_factor_manifest_path=str(size_factor_manifest_path),
                qa_manifest_path=str(qa_path),
                hvg_sidecar_path=str(hvg_sidecar_path),
                integer_verified=all_passed,
                cell_count=n_obs,
                feature_count=n_vars,
                notes=(
                    f"materialized via Stage2Materializer (schema-independent, count-first)",
                    f"topology={self.topology}",
                ),
            )
            manifest.validate()

            manifest_path = meta_root / "materialization-manifest.yaml"
            manifest.write_yaml(manifest_path)

            return manifest

        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    def _rerun_stage1_preflight(self) -> Path:
        """Re-run Stage 1 inspection as preflight and return the summary path."""
        from ..inspectors.workflow import inspect_target, InspectionTarget
        from ..inspectors.models import InspectionBatchConfig

        # Stage 1 writes to the same parent directory as the review bundle's dataset folder
        review_bundle = Path(self.review_bundle_path)
        output_root = review_bundle.parent.parent  # go up to dataset dir, then to run dir
        dataset_id = review_bundle.parent.name  # dataset dir name

        # Construct a temporary InspectionTarget for the rerun
        source_h5ad = Path(self.source_path)
        source_release = self.release_id

        target = InspectionTarget(
            dataset_id=dataset_id,
            source_path=str(source_h5ad),
            source_release=source_release,
        )

        # Run inspection; it writes dataset-summary.yaml into output_root/dataset_id/
        artifacts = inspect_target(target, Path(output_root))

        # Return the path to the newly written summary
        return artifacts.review_bundle

    def _select_count_matrix(self, adata: ad.AnnData, selected: str) -> Any:
        """Select the count matrix from the AnnData based on the approved candidate."""
        if selected == ".raw.X":
            return adata.raw.X
        elif selected.startswith(".layers["):
            layer_name = selected[len(".layers[") : -1]
            if layer_name not in adata.layers:
                raise KeyError(
                    f"layer '{layer_name}' not found in adata.layers; "
                    f"available layers: {list(adata.layers.keys())}"
                )
            return adata.layers[layer_name]
        else:
            return adata.X

    def _apply_reverse_normalization(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        selected_candidate: str,
        n_obs: int,
    ) -> np.ndarray:
        """Apply the approved expm1/size_factor reverse-normalization path.

        Recovery is only called when the Stage 1 decision set ``uses_recovery=True``.
        The approved path is: recovered = expm1(source_row) / min_nonzero_expm1(source_row)
        per row.
        """
        recovered = np.zeros((n_obs, count_matrix.shape[1]), dtype=np.float64)

        for i in range(n_obs):
            row = count_matrix[i]
            if hasattr(row, "toarray"):
                row = np.asarray(row.toarray().ravel())
            else:
                row = np.asarray(row).ravel()
            nonzero_mask = row != 0
            if nonzero_mask.any():
                expm1_vals = np.expm1(row[nonzero_mask])
                min_nonzero = float(np.min(expm1_vals))
                if min_nonzero <= 0:
                    min_nonzero = 1.0
                recovered[i, nonzero_mask] = expm1_vals / min_nonzero

        nonzero_mask = recovered != 0
        if nonzero_mask.any():
            deviations = np.abs(recovered[nonzero_mask] - np.rint(recovered[nonzero_mask]))
            max_deviation = float(np.max(deviations))
            if max_deviation > 0.01:
                raise ValueError(
                    f"reverse-normalization of {selected_candidate} produced "
                    f"non-integer values (max_deviation={max_deviation:.6f}); "
                    "recovery validation failed"
                )

        return np.rint(recovered).astype(np.int32)

    def _verify_integer(self, count_matrix: Any, source_name: str) -> None:
        """Fail hard if count matrix is not integer-like (max deviation > 1e-6)."""
        sample_rows = min(32, count_matrix.shape[0])
        indices = np.linspace(0, count_matrix.shape[0] - 1, num=sample_rows, dtype=int)
        nonzero_values = []
        for idx in indices:
            row = count_matrix[idx]
            if hasattr(row, "toarray"):
                row = np.asarray(row.toarray().ravel())
            else:
                row = np.asarray(row).ravel()
            nonzero_values.append(row[row != 0])
        if not nonzero_values:
            return
        nonzero = np.concatenate(nonzero_values)
        if nonzero.size == 0:
            return
        deviations = np.abs(nonzero - np.rint(nonzero))
        if np.any(deviations > 1e-6):
            raise ValueError(
                f"count matrix in {source_name} contains non-integer values "
                f"(max deviation {float(np.max(deviations)):.6f}); "
                "materialization requires strict integer counts from the Stage 1 approved source."
            )

    def _compute_size_factors(self, count_matrix: Any, n_obs: int) -> np.ndarray:
        """Compute size factors as row_sum / median(row_sum)."""
        from .backends.arrow_hf import _get_row_nonzero

        factors = np.zeros(n_obs, dtype=np.float64)
        for i in range(n_obs):
            indices, counts = _get_row_nonzero(count_matrix, i)
            factors[i] = float(counts.sum())

        row_median = float(np.median(factors))
        if row_median > 0:
            factors = factors / row_median
        factors = np.where(factors <= 0, 1.0, factors)
        factors = np.where(np.isnan(factors), 1.0, factors)
        return factors

    def _write_size_factor_manifest(
        self,
        adata: ad.AnnData,
        size_factors: np.ndarray,
        meta_root: Path,
    ) -> Path:
        """Write per-cell size factors as a SizeFactorManifest YAML."""
        obs_mem = adata.obs.to_memory() if hasattr(adata.obs, "to_memory") else adata.obs
        n_obs = adata.n_obs
        entries = []
        for i in range(n_obs):
            cell_id = str(obs_mem.index[i])
            entries.append(SizeFactorEntry(cell_id=cell_id, size_factor=float(size_factors[i])))
        manifest = SizeFactorManifest(
            kind="size-factor-manifest",
            contract_version=CONTRACT_VERSION,
            release_id=self.release_id,
            method="sum",
            entries=tuple(entries),
        )
        sf_path = meta_root / f"{self.release_id}-size-factors.yaml"
        manifest.write_yaml(sf_path)
        return sf_path

    def _write_raw_cell_metadata_parquet(
        self,
        adata: ad.AnnData,
        meta_root: Path,
    ) -> Path:
        """Write raw cell metadata as Parquet (no SQLite, no canonical mapping).

        Schema:
        - cell_id: string
        - dataset_id: string
        - dataset_release: string
        - raw_obs: string (JSON-serialized dict of all obs columns)
        """
        import json

        parquet_path = meta_root / f"{self.release_id}-raw-obs.parquet"
        obs_mem = adata.obs.to_memory() if hasattr(adata.obs, "to_memory") else adata.obs
        n_obs = adata.n_obs

        cell_ids = []
        raw_obs_list = []
        for i in range(n_obs):
            cell_ids.append(str(obs_mem.index[i]))
            raw_dict: dict[str, Any] = {}
            for col in obs_mem.columns:
                val = obs_mem.loc[obs_mem.index[i], col]
                raw_dict[str(col)] = None if pd.isna(val) else str(val)
            raw_obs_list.append(json.dumps(raw_dict))

        table = pa.table({
            "cell_id": pa.array(cell_ids, type=pa.string()),
            "dataset_id": pa.array([self.dataset_id] * n_obs, type=pa.string()),
            "dataset_release": pa.array([self.release_id] * n_obs, type=pa.string()),
            "raw_obs": pa.array(raw_obs_list, type=pa.string()),
        })
        pq.write_table(table, parquet_path)
        return parquet_path

    def _write_raw_feature_metadata_parquet(
        self,
        var_mem: Any,
        meta_root: Path,
    ) -> Path:
        """Write raw feature metadata as Parquet (no canonical mapping).

        Schema:
        - origin_index: int32
        - feature_id: string
        - raw_var: string (JSON-serialized dict of all var columns)
        """
        import json

        parquet_path = meta_root / f"{self.release_id}-raw-var.parquet"
        n_vars = var_mem.shape[0]

        origin_indices = list(range(n_vars))
        feature_ids = [str(var_mem.index[i]) for i in range(n_vars)]
        raw_var_list = []
        for i in range(n_vars):
            raw_dict: dict[str, Any] = {}
            for col in var_mem.columns:
                val = var_mem.loc[var_mem.index[i], col]
                raw_dict[str(col)] = None if pd.isna(val) else str(val)
            raw_var_list.append(json.dumps(raw_dict))

        table = pa.table({
            "origin_index": pa.array(origin_indices, type=pa.int32()),
            "feature_id": pa.array(feature_ids, type=pa.string()),
            "raw_var": pa.array(raw_var_list, type=pa.string()),
        })
        pq.write_table(table, parquet_path)
        return parquet_path

    def _write_metadata_summary(
        self,
        adata: ad.AnnData,
        var_mem: Any,
        meta_root: Path,
    ) -> Path:
        """Write per-dataset metadata summary (field coverage, null fractions)."""
        obs_mem = adata.obs.to_memory() if hasattr(adata.obs, "to_memory") else adata.obs
        n_obs = obs_mem.shape[0]
        n_vars = var_mem.shape[0]

        obs_null_fractions: dict[str, float] = {}
        for col in obs_mem.columns:
            null_count = int(obs_mem[col].isna().sum())
            obs_null_fractions[str(col)] = float(null_count) / float(n_obs)

        var_null_fractions: dict[str, float] = {}
        for col in var_mem.columns:
            null_count = int(var_mem[col].isna().sum())
            var_null_fractions[str(col)] = float(null_count) / float(n_vars)

        obs_dtypes: dict[str, str] = {str(col): str(obs_mem[col].dtype) for col in obs_mem.columns}
        var_dtypes: dict[str, str] = {str(col): str(var_mem[col].dtype) for col in var_mem.columns}

        summary = DatasetMetadataSummary(
            kind="dataset-metadata-summary",
            contract_version=CONTRACT_VERSION,
            dataset_id=self.dataset_id,
            release_id=self.release_id,
            source_path=str(Path(self.output_roots.metadata_root).parent / f"{self.release_id}.h5ad"),
            obs_field_count=int(obs_mem.shape[1]),
            var_field_count=int(var_mem.shape[1]),
            obs_null_fractions=obs_null_fractions,
            var_null_fractions=var_null_fractions,
            obs_dtypes=obs_dtypes,
            var_dtypes=var_dtypes,
            obs_rows=n_obs,
            var_rows=n_vars,
            obs_index_name=str(obs_mem.index.name or "obs_index"),
            var_index_name=str(var_mem.index.name or "var_index"),
        )
        summary_path = meta_root / f"{self.release_id}-metadata-summary.yaml"
        summary.write_yaml(summary_path)
        return summary_path

    def _write_feature_provenance_parquet(
        self,
        var_index: pd.Index,
        n_vars: int,
        meta_root: Path,
        count_source_selected: str,
        source_path: str,
    ) -> Path:
        """Write feature provenance Parquet (origin_index, feature_id, count_source, source_path)."""
        parquet_path = meta_root / f"{self.release_id}-feature-provenance.parquet"

        origin_indices = list(range(n_vars))
        feature_ids = [str(var_index[i]) for i in range(n_vars)]
        count_source_vals = [count_source_selected] * n_vars
        source_paths = [source_path] * n_vars

        table = pa.table({
            "origin_index": pa.array(origin_indices, type=pa.int32()),
            "feature_id": pa.array(feature_ids, type=pa.string()),
            "count_source": pa.array(count_source_vals, type=pa.string()),
            "source_path": pa.array(source_paths, type=pa.string()),
        })
        pq.write_table(table, parquet_path)
        return parquet_path

    def _write_cells(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        size_factors: np.ndarray,
        matrix_root: Path,
    ) -> dict[str, Path]:
        """Write sparse cell data via the configured backend."""
        from .backends import build_backend_fn

        backend_fn = build_backend_fn(self.backend)
        kwargs: dict[str, Any] = {
            "dataset_id": self.dataset_id,
        }
        return backend_fn(
            adata, count_matrix, size_factors, self.release_id, matrix_root,
            **kwargs,
        )

    def _run_qa_checks(
        self,
        backend_paths: dict[str, Path],
        original_matrix: Any,
    ) -> tuple[tuple[QAMetric, ...], bool]:
        """Run QA checks on written output artifacts."""
        metrics = []
        for name, path in backend_paths.items():
            if not path.exists():
                metrics.append(QAMetric(name=f"{name}_exists", value=0.0, threshold=1.0))
            else:
                metrics.append(QAMetric(name=f"{name}_exists", value=1.0, threshold=1.0))
        all_passed = all(m.passed() for m in metrics)
        return tuple(metrics), all_passed

    def _compute_and_write_hvg_arrays(
        self,
        count_matrix: Any,
        n_vars: int,
        meta_root: Path,
        n_hvg: int = 2000,
    ) -> Path:
        """Compute HVG/non-HVG index arrays in dataset-local feature space and write as NumPy."""
        from .backends.arrow_hf import _get_row_nonzero

        sidecar_dir = meta_root / "hvg_sidecar"
        sidecar_dir.mkdir(parents=True, exist_ok=True)

        n_obs = count_matrix.shape[0]
        sample_cells = min(512, n_obs)
        cell_indices = np.linspace(0, n_obs - 1, num=sample_cells, dtype=int)

        log_expr = np.zeros((sample_cells, n_vars), dtype=np.float64)
        for row_i, ci in enumerate(cell_indices):
            indices, counts = _get_row_nonzero(count_matrix, ci)
            for idx, cnt in zip(indices, counts):
                log_expr[row_i, idx] = np.log1p(float(cnt))

        gene_means = np.zeros(n_vars, dtype=np.float64)
        gene_vars = np.zeros(n_vars, dtype=np.float64)
        for j in range(n_vars):
            col = log_expr[:, j]
            nonzero = col[col > 0]
            if len(nonzero) > 0:
                gene_means[j] = np.mean(nonzero)
                gene_vars[j] = np.var(nonzero)

        eps = 1e-10
        dispersion = np.where(gene_means > eps, gene_vars / gene_means, 0.0)

        sorted_indices = np.argsort(dispersion)
        hvg_indices: np.ndarray = sorted_indices[-n_hvg:].astype(np.int32)
        hvg_indices = np.sort(hvg_indices)

        hvg_set = set(hvg_indices)
        nonhvg_indices = np.array(
            [j for j in range(n_vars) if j not in hvg_set], dtype=np.int32
        )

        np.save(str(sidecar_dir / "hvg.npy"), hvg_indices, allow_pickle=False)
        np.save(str(sidecar_dir / "nonhvg.npy"), nonhvg_indices, allow_pickle=False)

        return sidecar_dir


# ---------------------------------------------------------------------------
# Canonical sparse per-cell record contract (existing)
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
    - How to build/update the corpus tokenizer (append-only, preserving existing IDs)
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
        backend: str = "arrow-hf",
        corpus_index_path: Path | None = None,
    ):
        self.output_roots = output_roots
        self.release_id = release_id
        self.dataset_id = dataset_id
        self.count_source = count_source
        self.integer_only = integer_only
        self.backend = backend
        self._corpus_index_path = corpus_index_path

    @property
    def _corpus_root(self) -> Path:
        """Corpus-level root directory (where corpus-wide artifacts live).

        For ``create_new`` this is the metadata_root itself.
        For ``append_routed`` this is derived from the corpus index path.
        """
        if self._corpus_index_path is not None:
            return self._corpus_index_path.parent
        return Path(self.output_roots.metadata_root)

    @property
    def _matrix_root(self) -> Path:
        return Path(self.output_roots.matrix_root)

    def materialize(
        self,
        source_path: str,
        schema_path: str,
    ) -> MaterializationManifest:
        """Materialize a dataset using this route.

        Expression-first: the integer count matrix is the primary output.
        All other artifacts (raw metadata, accepted schema, summaries, provenance)
        are dataset-local and required for later canonical metadata rebuild.

        Tokenizer is NOT produced or consumed during materialization. The corpus
        feature set is maintained separately by ``canonicalize-meta``.

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

        # Determine which var reference to use based on selected count source.
        # When .raw.X wins, it may have a different feature dimension than .X,
        # so feature-axis-dependent steps (HVG, feature metadata) must follow
        # the winning source's feature space.
        raw_is_selected = self.count_source.selected == ".raw.X"
        if raw_is_selected:
            n_vars = int(adata.raw.shape[1])
            var_ref = adata.raw.var
            var_index = adata.raw.var.index
        else:
            n_vars = adata.n_vars
            var_ref = adata.var
            var_index = adata.var.index

        # Select count matrix
        if self.count_source.selected == ".raw.X":
            count_matrix = adata.raw.X
        elif self.count_source.selected.startswith(".layers["):
            layer_name = self.count_source.selected[len(".layers[") : -1]
            count_matrix = adata.layers[layer_name]
        else:
            count_matrix = adata.X

        # Apply reverse-normalization if the inspector selected a recovered source.
        # The inspector validated the recovery during inspection; at materialization
        # time we apply the same approved expm1/size_factor path to produce the
        # integer count matrix used for all downstream steps (size factors, QA, writes).
        if self.count_source.uses_recovery:
            count_matrix = self._apply_reverse_normalization(
                count_matrix, adata, self.count_source.selected, n_obs
            )

        # Verify integer counts
        if self.integer_only:
            self._verify_integer(count_matrix, source_h5ad.name)

        # Ensure output directories exist
        meta_root = Path(self.output_roots.metadata_root)
        matrix_root = Path(self.output_roots.matrix_root)
        meta_root.mkdir(parents=True, exist_ok=True)
        matrix_root.mkdir(parents=True, exist_ok=True)

        # --- Phase 3: Write accepted schema copy ---
        # Persist the accepted schema alongside dataset artifacts for later
        # canonicalize-meta reference. The schema_path may be a user-managed
        # path; we write a dataset-local copy.
        import shutil

        accepted_schema_path = meta_root / f"{self.release_id}-accepted-schema.yaml"
        shutil.copy2(schema_path, accepted_schema_path)

        # --- Phase 3: Write raw cell metadata (no canonical mapping) ---
        raw_cell_meta_sqlite_path = self._write_raw_cell_metadata(
            adata=adata,
            meta_root=meta_root,
        )

        # --- Phase 3: Write raw feature metadata (no canonical mapping) ---
        # Load var into memory once for efficient per-row access
        var_mem = var_ref.to_memory() if hasattr(var_ref, "to_memory") else var_ref
        raw_feature_meta_parquet_path = self._write_raw_feature_metadata(
            adata=adata,
            var_mem=var_mem,
            meta_root=meta_root,
        )

        # --- Phase 3: Write per-dataset metadata summary ---
        metadata_summary_path = self._write_metadata_summary(
            adata=adata,
            var_mem=var_mem,
            meta_root=meta_root,
        )

        # --- Phase 3: Compute size factors (sum-based, dataset-specific) ---
        size_factors = self._compute_size_factors(count_matrix, n_obs)

        # --- Phase 3: Write size-factor artifact for later reuse ---
        size_factor_manifest_path = self._write_size_factor_manifest(
            adata=adata,
            size_factors=size_factors,
            meta_root=meta_root,
        )

        # --- Phase 3: Write feature provenance spec ---
        # Build origin_index → feature_id mapping for canonicalize-meta
        origin_index_to_feature_id: dict[int, str] = {
            i: str(var_index[i]) for i in range(n_vars)
        }
        provenance_spec_path = self._write_feature_provenance(
            source_h5ad=source_h5ad,
            schema_path=accepted_schema_path,
            n_vars=n_vars,
            origin_index_to_feature_id=origin_index_to_feature_id,
            meta_root=meta_root,
        )

        # --- Resolve canonical cell metadata from schema (for backend write pass-through) ---
        # Phase 3: resolved canonical perturbation/context tuples are passed to the
        # backend writer so Arrow/HF can write the canonical cell SQLite atomically.
        # This avoids a second-pass scan of obs after the fact.
        obs_mem = adata.obs.to_memory() if hasattr(adata.obs, "to_memory") else adata.obs
        perturbations, contexts = resolve_all_cell_rows(schema, obs_mem)
        canonical_perturbation: tuple[dict[str, str], ...] = tuple(perturbations)
        canonical_context: tuple[dict[str, str], ...] = tuple(contexts)
        raw_cell_tuples: tuple[dict[str, Any], ...] = tuple(
            {col: (None if pd.isna(obs_mem.loc[obs_mem.index[i], col]) else str(obs_mem.loc[obs_mem.index[i], col]))
             for col in obs_mem.columns}
            for i in range(n_obs)
        )

        # --- Write backend-specific cell data ---
        # Phase 3: canonical_perturbation, canonical_context, and raw_fields are
        # passed so the Arrow/HF backend can write the canonical cell SQLite.
        # WebDataset/Zarr backends accept these but write metadata into their own formats.
        backend_paths = self._write_cells(
            count_matrix=count_matrix,
            adata=adata,
            size_factors=size_factors,
            matrix_root=matrix_root,
            canonical_perturbation=canonical_perturbation,
            canonical_context=canonical_context,
            raw_fields=raw_cell_tuples,
        )

        # --- Phase 3: Write canonical feature metadata (origin parquet) ---
        # This is the canonical feature table in original dataset var order,
        # resolved via schema execution. Token-space mapping is handled by
        # canonicalize-meta (not during materialization).
        feature_meta_paths = self._write_canonical_feature_metadata(
            adata=adata,
            schema=schema,
            var_mem=var_mem,
            meta_root=meta_root,
            n_vars=n_vars,
        )

        # --- Compute and write HVG/non-HVG arrays in original dataset feature indices ---
        hvg_sidecar_path = self._compute_and_write_hvg_arrays(
            count_matrix=count_matrix,
            n_vars=n_vars,
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

        # Build final manifest (tokenizer_path removed in Phase 3)
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version=CONTRACT_VERSION,
            dataset_id=self.dataset_id,
            release_id=self.release_id,
            route=self.route_name,
            backend=self.backend,
            count_source=self.count_source,
            outputs=self.output_roots,
            provenance=ProvenanceSpec(
                source_path=str(source_h5ad),
                schema=schema_path,
            ),
            # Phase 3 dataset-local artifact paths
            raw_cell_meta_path=str(raw_cell_meta_sqlite_path),
            raw_feature_meta_path=str(raw_feature_meta_parquet_path),
            accepted_schema_path=str(accepted_schema_path),
            metadata_summary_path=str(metadata_summary_path),
            provenance_spec_path=str(provenance_spec_path),
            # Remaining existing paths
            feature_meta_paths={
                k: str(v) for k, v in feature_meta_paths.items()
            },
            size_factor_manifest_path=str(size_factor_manifest_path),
            qa_manifest_path=str(qa_path),
            hvg_sidecar_path=str(hvg_sidecar_path),
            integer_verified=all_passed,
            cell_count=n_obs,
            notes=(f"materialized via {self.route_name} route (tokenizer-free)",),
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

    def _apply_reverse_normalization(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        selected_candidate: str,
        n_obs: int,
    ) -> np.ndarray:
        """Apply the approved expm1/size_factor reverse-normalization path.

        This is only called when ``count_source.uses_recovery`` is True, meaning
        the inspector validated that recovery is possible and the selected
        candidate is a log-normalized layer that can be reversed.

        Recovery is transient: per-row scale factors are recomputed during
        materialization from the log-normalized source itself and are NOT
        persisted. The final integer count matrix is what is stored.

        Per-row scale_factor = smallest nonzero value in expm1(source_row).
        Recovery formula: recovered = expm1(source_row) / scale_factor.

        Parameters
        ----------
        count_matrix : Any
            The log-normalized matrix selected by the inspector
            (e.g. ``adata.layers["log_norm"]``).
        adata : ad.AnnData
            The source AnnData object (backed, read-only is fine).
        selected_candidate : str
            The candidate name from the inspector's decision
            (e.g. ``".layers[log_norm]"``).
        n_obs : int
            Number of observations (cells) in the dataset.

        Returns
        -------
        np.ndarray
            Recovered integer count matrix as a dense ``np.int32`` array.

        Raises
        ------
        ValueError
            If the recovered values are not sufficiently integer-like.
        """
        # Allocate full recovered matrix (conservative for memory but correct)
        recovered = np.zeros((n_obs, count_matrix.shape[1]), dtype=np.float64)

        for i in range(n_obs):
            row = count_matrix[i]
            if hasattr(row, "toarray"):
                row = np.asarray(row.toarray().ravel())
            else:
                row = np.asarray(row).ravel()
            nonzero_mask = row != 0
            if nonzero_mask.any():
                # expm1 of the log-normalized values
                expm1_vals = np.expm1(row[nonzero_mask])
                # Per-row scale factor = smallest nonzero expm1 value
                min_nonzero = float(np.min(expm1_vals))
                if min_nonzero <= 0:
                    min_nonzero = 1.0  # fallback for degenerate rows
                # Recover: divide by per-row scale factor
                recovered[i, nonzero_mask] = expm1_vals / min_nonzero

        # Verify recovered values are integer-like before returning
        nonzero_mask = recovered != 0
        if nonzero_mask.any():
            deviations = np.abs(recovered[nonzero_mask] - np.rint(recovered[nonzero_mask]))
            max_deviation = float(np.max(deviations))
            if max_deviation > 0.01:
                raise ValueError(
                    f"reverse-normalization of {selected_candidate} produced "
                    f"non-integer values (max_deviation={max_deviation:.6f}); "
                    "recovery validation failed — check that the selected source "
                    "is a log-normalized layer compatible with expm1/size_factor"
                )

        # Convert to int32 for sparse write efficiency
        return np.rint(recovered).astype(np.int32)

    def _compute_size_factors(self, count_matrix: Any, n_obs: int) -> np.ndarray:
        """Compute size factors as row_sum / median(row_sum).

        Dataset size factors are derived from final processed count sums
        (post-recovery when applicable) using median normalization.
        """
        from .backends.arrow_hf import _get_row_nonzero

        factors = np.zeros(n_obs, dtype=np.float64)
        for i in range(n_obs):
            indices, counts = _get_row_nonzero(count_matrix, i)
            factors[i] = float(counts.sum())

        row_median = float(np.median(factors))
        if row_median > 0:
            factors = factors / row_median
        # Replace zero or NaN with 1.0
        factors = np.where(factors <= 0, 1.0, factors)
        factors = np.where(np.isnan(factors), 1.0, factors)
        return factors

    def _write_size_factor_manifest(
        self,
        adata: ad.AnnData,
        size_factors: np.ndarray,
        meta_root: Path,
    ) -> Path:
        """Write size-factor artifact for later reuse.

        Persists the per-cell size factors computed during materialization so they
        can be reused by later metadata rebuild steps without recomputing from counts.
        Written as a SizeFactorManifest YAML containing one entry per cell.
        """
        obs_mem = adata.obs.to_memory() if hasattr(adata.obs, "to_memory") else adata.obs
        n_obs = adata.n_obs
        entries = []
        for i in range(n_obs):
            cell_id = str(obs_mem.index[i])
            entries.append(SizeFactorEntry(cell_id=cell_id, size_factor=float(size_factors[i])))
        manifest = SizeFactorManifest(
            kind="size-factor-manifest",
            contract_version=CONTRACT_VERSION,
            release_id=self.release_id,
            method="sum",
            entries=tuple(entries),
        )
        sf_path = meta_root / f"{self.release_id}-size-factors.yaml"
        manifest.write_yaml(sf_path)
        return sf_path

    def _write_cells(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        size_factors: np.ndarray,
        matrix_root: Path,
    ) -> dict[str, Path]:
        """Write per-cell sparse data. Override in subclass."""
        raise NotImplementedError

    # ---------------------------------------------------------------------------
    # Phase 3: New raw metadata writers (tokenizer-free)
    # ---------------------------------------------------------------------------

    def _write_raw_cell_metadata(
        self,
        adata: ad.AnnData,
        meta_root: Path,
    ) -> Path:
        """Write raw cell metadata as SQLite (no canonical mapping applied).

        Raw obs fields are preserved as-is from the h5ad. Each row records the
        cell_id, dataset_id, dataset_release, and a JSON blob of all obs columns.
        This is the authoritative source for canonical cell metadata rebuild
        in ``canonicalize-meta``.
        """
        import json

        db_path = meta_root / f"{self.release_id}-raw-cell-meta.sqlite"

        # Load obs into memory once for efficient per-row access
        obs_mem = adata.obs.to_memory() if hasattr(adata.obs, "to_memory") else adata.obs

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS raw_cell_meta "
            "(cell_id TEXT, dataset_id TEXT, dataset_release TEXT, raw_obs TEXT)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cell_id ON raw_cell_meta(cell_id)")

        batch_size = 256
        n_obs = adata.n_obs
        for start in range(0, n_obs, batch_size):
            end = min(start + batch_size, n_obs)
            batch = []
            for i in range(start, end):
                cell_id = str(obs_mem.index[i])
                # Serialize all obs columns as JSON
                raw: dict[str, Any] = {}
                for col in obs_mem.columns:
                    val = obs_mem.loc[obs_mem.index[i], col]
                    raw[str(col)] = None if pd.isna(val) else str(val)
                raw_json = json.dumps(raw) if raw else "{}"
                batch.append(
                    (
                        cell_id,
                        self.dataset_id,
                        self.release_id,
                        raw_json,
                    )
                )
            conn.executemany(
                "INSERT INTO raw_cell_meta VALUES (?, ?, ?, ?)",
                batch,
            )
        conn.commit()
        conn.close()
        return db_path

    def _write_raw_feature_metadata(
        self,
        adata: ad.AnnData,
        var_mem: Any,
        meta_root: Path,
    ) -> Path:
        """Write raw feature metadata as Parquet (no canonical mapping applied).

        Raw var fields are preserved as-is from the h5ad. Each row records the
        origin_index (original dataset var order), feature_id (var index value),
        and a JSON blob of all var columns. This is the authoritative source for
        canonical feature metadata rebuild in ``canonicalize-meta``.
        """
        import json

        origin_path = meta_root / f"{self.release_id}-raw-feature-meta.parquet"

        n_vars = var_mem.shape[0]
        origin_indices = list(range(n_vars))
        feature_ids = [str(var_mem.index[i]) for i in range(n_vars)]

        # Serialize all var columns as a JSON string per feature
        raw_var_list: list[str] = []
        for i in range(n_vars):
            raw: dict[str, Any] = {}
            for col in var_mem.columns:
                val = var_mem.loc[var_mem.index[i], col]
                raw[str(col)] = None if pd.isna(val) else str(val)
            raw_var_list.append(json.dumps(raw))

        table = pa.table(
            {
                "origin_index": pa.array(origin_indices, type=pa.int32()),
                "feature_id": pa.array(feature_ids, type=pa.string()),
                "raw_var": pa.array(raw_var_list, type=pa.string()),
            }
        )
        pq.write_table(table, origin_path)
        return origin_path

    def _write_metadata_summary(
        self,
        adata: ad.AnnData,
        var_mem: Any,
        meta_root: Path,
    ) -> Path:
        """Write per-dataset metadata summary (field coverage, null fractions).

        This summary provides dataset-level statistics extracted from raw obs/var
        before any canonical mapping. It can assist schema review when onboarding
        new datasets. Written as YAML for human readability.
        """
        obs_mem = adata.obs.to_memory() if hasattr(adata.obs, "to_memory") else adata.obs
        n_obs = obs_mem.shape[0]
        n_vars = var_mem.shape[0]

        # Compute null fractions per obs column
        obs_null_fractions: dict[str, float] = {}
        for col in obs_mem.columns:
            null_count = int(obs_mem[col].isna().sum())
            obs_null_fractions[str(col)] = float(null_count) / float(n_obs)

        # Compute null fractions per var column
        var_null_fractions: dict[str, float] = {}
        for col in var_mem.columns:
            null_count = int(var_mem[col].isna().sum())
            var_null_fractions[str(col)] = float(null_count) / float(n_vars)

        # Collect dtypes
        obs_dtypes: dict[str, str] = {str(col): str(obs_mem[col].dtype) for col in obs_mem.columns}
        var_dtypes: dict[str, str] = {str(col): str(var_mem[col].dtype) for col in var_mem.columns}

        summary = DatasetMetadataSummary(
            kind="dataset-metadata-summary",
            contract_version=CONTRACT_VERSION,
            dataset_id=self.dataset_id,
            release_id=self.release_id,
            source_path=str(Path(self.output_roots.metadata_root).parent / f"{self.release_id}.h5ad"),
            obs_field_count=int(obs_mem.shape[1]),
            var_field_count=int(var_mem.shape[1]),
            obs_null_fractions=obs_null_fractions,
            var_null_fractions=var_null_fractions,
            obs_dtypes=obs_dtypes,
            var_dtypes=var_dtypes,
            obs_rows=n_obs,
            var_rows=n_vars,
            obs_index_name=str(obs_mem.index.name or "obs_index"),
            var_index_name=str(var_mem.index.name or "var_index"),
        )
        summary_path = meta_root / f"{self.release_id}-metadata-summary.yaml"
        summary.write_yaml(summary_path)
        return summary_path

    def _write_feature_provenance(
        self,
        source_h5ad: Path,
        schema_path: Path,
        n_vars: int,
        origin_index_to_feature_id: dict[int, str],
        meta_root: Path,
    ) -> Path:
        """Write feature-order/provenance artifact.

        Records the per-dataset feature ordering and provenance needed by
        ``canonicalize-meta`` to rebuild the corpus feature set without a tokenizer.
        """
        provenance = FeatureProvenanceSpec(
            release_id=self.release_id,
            feature_count=n_vars,
            source_path=str(source_h5ad),
            schema_path=str(schema_path),
            count_source=self.count_source,
            origin_index_to_feature_id=origin_index_to_feature_id,
        )
        provenance_path = meta_root / f"{self.release_id}-feature-provenance.yaml"
        provenance.write_yaml(provenance_path)
        return provenance_path

    def _write_canonical_feature_metadata(
        self,
        adata: ad.AnnData,
        schema: SchemaDocument,
        var_mem: Any,
        meta_root: Path,
        n_vars: int,
    ) -> dict[str, Path]:
        """Write canonical feature metadata in original dataset var order.

        Writes one parquet file: ``{release_id}-features-origin.parquet``
        containing origin_index and feature_id, with canonical fields resolved
        via schema execution. Token-space mapping is NOT done here — it is
        handled by ``canonicalize-meta`` which maintains the corpus feature set.

        Returns dict with key "features_origin".
        """
        origin_path = meta_root / f"{self.release_id}-features-origin.parquet"
        origin_indices = list(range(n_vars))

        # Resolve canonical feature fields via schema execution
        feature_rows = resolve_all_feature_rows(schema, var_mem)

        # Build feature_id list from var index
        if self.count_source.selected == ".raw.X" and hasattr(adata.raw, "var"):
            var_index = adata.raw.var.index
        else:
            var_index = adata.var.index
        origin_feature_ids = [str(var_index[i]) for i in origin_indices]

        # Build base table
        origin_table = pa.table(
            {
                "origin_index": pa.array(origin_indices, type=pa.int32()),
                "feature_id": pa.array(origin_feature_ids, type=pa.string()),
            }
        )

        # Attach resolved canonical feature fields as a struct column
        field_names = list(schema.feature_fields.keys())
        if field_names:
            canonical_struct = pa.struct([
                (fname, pa.string()) for fname in field_names
            ])
            canonical_values = [
                [row.get(fname, MISSING_VALUE_LITERAL) for row in feature_rows]
                for fname in field_names
            ]
            canonical_array = pa.array(
                [dict(zip(field_names, vals)) for vals in zip(*canonical_values)],
                type=canonical_struct,
            )
            origin_table = origin_table.append_column("canonical", canonical_array)

        pq.write_table(origin_table, origin_path)
        return {"features_origin": origin_path}

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

    def _compute_and_write_hvg_arrays(
        self,
        count_matrix: Any,
        n_vars: int,
        meta_root: Path,
        n_hvg: int = 2000,
    ) -> Path:
        """Compute HVG/non-HVG index arrays from the count matrix and write them.

        HVG selection uses top-N dispersion on log-normalized values:
        dispersion = variance / mean for each feature across cells.
        The selected indices are written in **original dataset feature index**
        space (not token space) as ``hvg.npy`` and ``nonhvg.npy``.

        Parameters
        ----------
        count_matrix : Any
            The count matrix; must be log-normalized before calling this method.
            This method applies ``log1p`` internally to compute dispersion.
        n_vars : int
            Total number of features (n_vars from adata).
        meta_root : Path
            Directory to write the sidecar files.
        n_hvg : int
            Number of top-dispersion genes to select as HVGs. Default 2000.

        Returns
        -------
        Path
            Path to the directory containing ``hvg.npy`` and ``nonhvg.npy``.
        """
        sidecar_dir = meta_root / "hvg_sidecar"
        sidecar_dir.mkdir(parents=True, exist_ok=True)

        # Compute mean and variance of log1p(count_matrix) per feature (column)
        # We sample cells to keep memory bounded for large datasets.
        from .backends.arrow_hf import _get_row_nonzero

        n_obs = count_matrix.shape[0]
        sample_cells = min(512, n_obs)
        cell_indices = np.linspace(0, n_obs - 1, num=sample_cells, dtype=int)

        log_expr = np.zeros((sample_cells, n_vars), dtype=np.float64)
        for row_i, ci in enumerate(cell_indices):
            indices, counts = _get_row_nonzero(count_matrix, ci)
            # Place log1p(count) at the gene indices
            for idx, cnt in zip(indices, counts):
                log_expr[row_i, idx] = np.log1p(float(cnt))

        # Compute per-feature mean and variance of log-normalized expression
        # Use a clipped mean to avoid zero-mean artifacts
        gene_means = np.zeros(n_vars, dtype=np.float64)
        gene_vars = np.zeros(n_vars, dtype=np.float64)
        for j in range(n_vars):
            col = log_expr[:, j]
            nonzero = col[col > 0]
            if len(nonzero) > 0:
                gene_means[j] = np.mean(nonzero)
                gene_vars[j] = np.var(nonzero)
            else:
                gene_means[j] = 0.0
                gene_vars[j] = 0.0

        # Dispersion = variance / mean (with a small epsilon to avoid division by zero)
        eps = 1e-10
        dispersion = np.where(gene_means > eps, gene_vars / gene_means, 0.0)

        # Select top-N HVG indices by dispersion
        #.argsort() gives ascending; take last N for descending
        sorted_indices = np.argsort(dispersion)
        hvg_indices: np.ndarray = sorted_indices[-n_hvg:].astype(np.int32)
        # Sort for consistent ordering
        hvg_indices = np.sort(hvg_indices)

        # Non-HVG = complement in original index space
        hvg_set = set(hvg_indices)
        nonhvg_indices = np.array(
            [j for j in range(n_vars) if j not in hvg_set], dtype=np.int32
        )

        # Write artifacts
        np.save(str(sidecar_dir / "hvg.npy"), hvg_indices, allow_pickle=False)
        np.save(str(sidecar_dir / "nonhvg.npy"), nonhvg_indices, allow_pickle=False)

        return sidecar_dir


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
        canonical_perturbation: tuple[dict[str, str], ...] | None = None,
        canonical_context: tuple[dict[str, str], ...] | None = None,
        raw_fields: tuple[dict[str, Any], ...] | None = None,
    ) -> dict[str, Path]:
        from .backends import build_backend_fn

        backend_fn = build_backend_fn(self.backend)
        kwargs: dict[str, Any] = {
            "canonical_perturbation": canonical_perturbation,
            "canonical_context": canonical_context,
            "raw_fields": raw_fields,
            "dataset_id": self.dataset_id,
        }
        if self.backend == "lancedb-aggregated":
            kwargs["corpus_index_path"] = self._corpus_index_path
        return backend_fn(
            adata, count_matrix, size_factors, self.release_id, matrix_root,
            **kwargs,
        )


class AppendRoutedRoute(MaterializationRoute):
    """Materialization route: add a dataset to a growing indexed corpus.

    Each dataset is stored independently with its own manifest. A corpus index
    maps dataset IDs to release IDs. The corpus tokenizer is appended with new
    token IDs while preserving existing entries. This is the default path for
    corpora expected to grow with additional datasets.

    ``corpus_index_path`` must be provided and must point to an existing
    ``corpus-index.yaml`` so the route can locate the corpus tokenizer for
    append compatibility checking.
    """

    route_name = "append_routed"

    @property
    def _corpus_root(self) -> Path:
        """Corpus root is derived from the corpus index path for append routes."""
        if self._corpus_index_path is None:
            raise ValueError(
                "append_routed requires corpus_index_path to be set so the "
                "existing corpus tokenizer can be located"
            )
        return self._corpus_index_path.parent

    def _write_cells(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        size_factors: np.ndarray,
        matrix_root: Path,
        canonical_perturbation: tuple[dict[str, str], ...] | None = None,
        canonical_context: tuple[dict[str, str], ...] | None = None,
        raw_fields: tuple[dict[str, Any], ...] | None = None,
    ) -> dict[str, Path]:
        from .backends import build_backend_fn

        backend_fn = build_backend_fn(self.backend)
        kwargs: dict[str, Any] = {
            "canonical_perturbation": canonical_perturbation,
            "canonical_context": canonical_context,
            "raw_fields": raw_fields,
            "dataset_id": self.dataset_id,
        }
        if self.backend == "lancedb-aggregated":
            kwargs["corpus_index_path"] = self._corpus_index_path
        return backend_fn(
            adata, count_matrix, size_factors, self.release_id, matrix_root,
            **kwargs,
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
    backend: str = "arrow-hf",
    corpus_index_path: Path | None = None,
) -> MaterializationRoute:
    """Factory to build the correct materialization route by name.

    Parameters
    ----------
    backend : str
        Storage backend for this materialization: ``arrow-hf``, ``webdataset``,
        or ``zarr-ts``. This is recorded in the MaterializationManifest and
        must match the corpus's declared backend. Defaults to ``arrow-hf``.
    corpus_index_path : Path | None
        Path to the corpus index YAML.  Used by ``append_routed`` to locate
        the corpus root for artifact writes.  Optional for ``create_new``.
        Tokenizer is NOT required — corpus feature set is maintained separately.
    """
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
        backend=backend,
        corpus_index_path=corpus_index_path,
    )


# ---------------------------------------------------------------------------
# Corpus index updater
# ---------------------------------------------------------------------------


def update_corpus_index(
    corpus_index_path: Path,
    new_dataset_record: DatasetJoinRecord,
    global_metadata: GlobalMetadataDocument | None = None,
    emission_spec_path: str | None = None,
    backend: str | None = None,
) -> CorpusIndexDocument:
    """Load an existing corpus index, append the new dataset record, and save.

    If corpus_index_path does not exist, create a new index and also write
    a new ``global-metadata.yaml`` next to the index.

    This function always appends; it does not overwrite existing dataset entries.

    The new dataset record's ``global_start``/``global_end`` are computed
    automatically from the existing corpus's total cell count, forming a
    deterministic, contiguous, non-overlapping partition of corpus cells.
    The ``cell_count`` field of ``new_dataset_record`` must be set to the
    number of cells in the new dataset for range computation to be correct.

    Tokenizer is NOT managed here — the corpus feature set is maintained separately
    by ``canonicalize-meta``.

    Parameters
    ----------
    corpus_index_path : Path
        Path to the corpus index YAML file.
    new_dataset_record : DatasetJoinRecord
        The dataset join record for the new dataset. Must have ``cell_count`` set.
        ``global_start``/``global_end`` in the input are ignored; they are
        recomputed from the existing corpus.
    global_metadata : GlobalMetadataDocument | None
        Global metadata for new corpus creation.  Required when creating
        a new corpus; ignored for existing corpora.
    emission_spec_path : str | None
        Relative path to ``corpus-emission-spec.yaml`` from the corpus root.
    backend : str | None
        Backend declaration for the corpus (e.g., "arrow-hf", "webdataset",
        "zarr-ts"). Required when creating a new corpus; written to
        ``global-metadata.yaml``.
    """
    corpus_root = corpus_index_path.parent

    # Convert manifest_path to relative to corpus_root for portability.
    # If the manifest is outside corpus_root, store the absolute path as fallback.
    manifest_path_input = Path(new_dataset_record.manifest_path)
    try:
        manifest_path_relative = manifest_path_input.relative_to(corpus_root)
    except ValueError:
        # Manifest is outside corpus_root — store absolute path
        manifest_path_relative = manifest_path_input

    if corpus_index_path.exists():
        corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)
        existing_ids = {d.dataset_id for d in corpus.datasets}
        if new_dataset_record.dataset_id in existing_ids:
            raise ValueError(
                f"dataset {new_dataset_record.dataset_id} already exists in corpus index; "
                "use a different release_id or dataset_id to avoid duplication."
            )
        # Compute total cells in existing corpus
        total_existing_cells = sum(d.cell_count for d in corpus.datasets)
        # Compute global range for the new dataset
        new_global_start = total_existing_cells
        new_global_end = total_existing_cells + new_dataset_record.cell_count
        # Build updated record with computed global range
        new_record = DatasetJoinRecord(
            dataset_id=new_dataset_record.dataset_id,
            dataset_index=len(corpus.datasets),
            release_id=new_dataset_record.release_id,
            join_mode=new_dataset_record.join_mode,
            manifest_path=str(manifest_path_relative),
            cell_count=new_dataset_record.cell_count,
            global_start=new_global_start,
            global_end=new_global_end,
        )
        datasets = list(corpus.datasets) + [new_record]
        corpus_id = corpus.corpus_id
        global_meta = corpus.global_metadata
    else:
        # For new corpus, global range is [0, cell_count)
        new_record = DatasetJoinRecord(
            dataset_id=new_dataset_record.dataset_id,
            dataset_index=0,
            release_id=new_dataset_record.release_id,
            join_mode=new_dataset_record.join_mode,
            manifest_path=str(manifest_path_relative),
            cell_count=new_dataset_record.cell_count,
            global_start=0,
            global_end=new_dataset_record.cell_count,
        )
        datasets = [new_record]
        corpus_id = "perturb-data-lab-v0"
        # Build global metadata dict for new corpus
        gmeta_dict: dict[str, Any] = {
            "kind": "global-metadata",
            "contract_version": CONTRACT_VERSION,
            "schema_version": CONTRACT_VERSION,
            "missing_value_literal": MISSING_VALUE_LITERAL,
            "raw_field_policy": "preserve-unchanged",
        }
        if backend is not None:
            gmeta_dict["backend"] = backend
        if global_metadata is not None:
            gmeta_dict = global_metadata.to_dict()
        if emission_spec_path is not None:
            gmeta_dict["emission_spec_path"] = emission_spec_path
        global_meta = gmeta_dict
        # Write global-metadata.yaml only when we have meaningful content
        if gmeta_dict:
            global_meta_path = corpus_index_path.parent / "global-metadata.yaml"
            GlobalMetadataDocument.from_dict(gmeta_dict).write_yaml(global_meta_path)

    updated = CorpusIndexDocument(
        kind="corpus-index",
        contract_version=CONTRACT_VERSION,
        corpus_id=corpus_id,
        global_metadata=global_meta,
        datasets=tuple(datasets),
    )
    updated.write_yaml(corpus_index_path)
    if backend == "lancedb-aggregated":
        mark_lance_append_committed(corpus_index_path, new_record)
    return updated
