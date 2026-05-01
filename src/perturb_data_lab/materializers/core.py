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

import json
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

from .backends import build_backend_fn


def _safe_serialize(val: Any) -> Any:
    """Serialize a value to a JSON-safe representation.

    Handles numpy scalars, pandas NA/NaN, and other common types
    that json.dumps cannot serialize directly.
    """
    if val is None or (isinstance(val, float) and (val != val)):  # NaN check
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, pd.CategoricalDtype):
        return str(val)
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (str, int, float, bool)):
        return val
    return str(val)

from .models import (
    CellMetadataRecord,
    CorpusIndexDocument,
    CorpusLedgerEntry,
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
from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL
from ..inspectors.models import DatasetSummaryDocument


# ---------------------------------------------------------------------------
# Stage 2 Materializer — schema-independent, Stage-1-gated, count-first
# ---------------------------------------------------------------------------


class Stage2Materializer:
    """Schema-independent materialization entry point for Stage 2.

    This class provides a Stage-1-gated, count-first materialization path.

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
        Storage backend: ``arrow-parquet`` (default), ``arrow-ipc``, ``webdataset``, ``zarr``, ``lance``.
    topology : str
        Corpus topology: ``federated`` (default) or ``aggregate``.
    rerun_stage1 : bool, default False
        If True, reruns Stage 1 inspection before materialization as a preflight
        step. The resulting ``dataset-summary.yaml`` replaces ``review_bundle_path``
        as the gating artifact.
    n_hvg : int, default 2000
        Number of top-dispersion genes to select as HVGs.
    chunk_rows : int, default 100000
        Number of rows per chunk when streaming the count matrix through the
        write path. Larger values reduce the number of loop iterations but
        increase per-chunk memory. Must be positive.
    corpus_index_path : str | None, default None
        Path to ``corpus-index.yaml`` for corpus registration. Required when
        ``register=True``. When provided, the corpus ledger
        (``corpus-ledger.parquet``) is updated after materialization.
    corpus_id : str | None, default None
        Corpus identifier. Required when registering a new corpus; inferred from
        existing ledger for append operations.
    register : bool, default False
        If True, automatically register this dataset with the corpus ledger
        after successful materialization. Requires ``corpus_index_path`` to be set.
        When True, the returned manifest's ``corpus_registration`` field is
        populated and the manifest YAML is rewritten with registration metadata.

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
    ``arrow-parquet`` × ``federated``.
    """

    def __init__(
        self,
        source_path: str,
        review_bundle_path: str,
        output_roots: OutputRoots,
        release_id: str,
        dataset_id: str,
        backend: str = "arrow-parquet",
        topology: str = "federated",
        rerun_stage1: bool = False,
        n_hvg: int = 2000,
        chunk_rows: int = 100_000,
        corpus_index_path: str | None = None,
        corpus_id: str | None = None,
        register: bool = False,
        dataset_index: int = 0,
        global_row_start: int = 0,
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
        self.chunk_rows = chunk_rows
        self.corpus_index_path = corpus_index_path
        self.corpus_id = corpus_id
        self.register = register
        self.dataset_index = dataset_index
        self.global_row_start = global_row_start
        self.writer_state: dict | None = None
        self._is_last_dataset: bool = False
        if register and corpus_index_path is None:
            raise ValueError(
                "register=True requires corpus_index_path to be set; "
                "pass corpus_index_path='/path/to/corpus-index.yaml'"
            )

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

            # --- Write feature provenance Parquet ---
            provenance_spec_path = self._write_feature_provenance_parquet(
                var_index=var_index,
                n_vars=n_vars,
                meta_root=meta_root,
                count_source_selected=count_source.selected,
                source_path=str(source_h5ad),
            )

            # --- Select count matrix from the approved count source ---
            count_matrix = self._select_count_matrix(adata, count_source.selected)

            # --- Write backend-specific sparse cell data ---
            # HVG statistics are accumulated during the chunk loop via np.add.at.
            # Global size factors are computed after the loop from all row_sums.
            # Both HVG arrays and size factors are written to sidecars after the loop.
            backend_result: tuple[dict[str, Path], np.ndarray, Path] | dict[str, Path]
            backend_result = self._write_cells(
                count_matrix=count_matrix,
                adata=adata,
                matrix_root=matrix_root,
                needs_recovery=count_source.uses_recovery,
            )
            if isinstance(backend_result, tuple):
                backend_paths, size_factors, hvg_sidecar_path = backend_result
                size_factor_parquet_path = self._write_size_factor_parquet(
                    size_factors=size_factors,
                    cell_ids=adata.obs.index,
                    meta_root=meta_root,
                )
            else:
                backend_paths = backend_result
                size_factor_parquet_path = None
                hvg_sidecar_path = None

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
                size_factor_parquet_path=(
                    str(size_factor_parquet_path) if size_factor_parquet_path else None
                ),
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

            # --- Phase 4: Corpus registration ---
            # If register=True, register this dataset with the corpus ledger.
            # This updates corpus-index.yaml and corpus-ledger.parquet.
            if self.register:
                from .models import CorpusRegistrationInfo
                from .registration import register_materialization

                join_record, resolved_corpus_id, is_create = register_materialization(
                    manifest=manifest,
                    corpus_index_path=Path(self.corpus_index_path),
                    corpus_id=self.corpus_id,
                    backend=self.backend,
                    topology=self.topology,
                )
                # Attach registration info to the manifest and rewrite
                reg_info = CorpusRegistrationInfo(
                    corpus_id=resolved_corpus_id,
                    is_create=is_create,
                    corpus_index_path=str(Path(self.corpus_index_path).resolve()),
                    ledger_path=str(
                        (Path(self.corpus_index_path).parent / "corpus-ledger.parquet").resolve()
                    ),
                    dataset_index=join_record.dataset_index,
                    global_start=join_record.global_start,
                    global_end=join_record.global_end,
                )
                manifest = MaterializationManifest(
                    kind=manifest.kind,
                    contract_version=manifest.contract_version,
                    dataset_id=manifest.dataset_id,
                    release_id=manifest.release_id,
                    route=manifest.route,
                    backend=manifest.backend,
                    topology=manifest.topology,
                    count_source=manifest.count_source,
                    outputs=manifest.outputs,
                    provenance=manifest.provenance,
                    raw_cell_meta_path=manifest.raw_cell_meta_path,
                    raw_feature_meta_path=manifest.raw_feature_meta_path,
                    accepted_schema_path=manifest.accepted_schema_path,
                    metadata_summary_path=manifest.metadata_summary_path,
                    provenance_spec_path=manifest.provenance_spec_path,
                    feature_meta_paths=manifest.feature_meta_paths,
                    size_factor_manifest_path=manifest.size_factor_manifest_path,
                    size_factor_parquet_path=manifest.size_factor_parquet_path,
                    qa_manifest_path=manifest.qa_manifest_path,
                    hvg_sidecar_path=manifest.hvg_sidecar_path,
                    integer_verified=manifest.integer_verified,
                    cell_count=manifest.cell_count,
                    feature_count=manifest.feature_count,
                    corpus_registration=reg_info,
                    notes=manifest.notes,
                )
                # Re-write manifest with registration info
                manifest_path.write_text(manifest.to_yaml())

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

    def _write_raw_cell_metadata_parquet(
        self,
        adata: ad.AnnData,
        meta_root: Path,
    ) -> Path:
        """Write raw cell metadata (obs) as a Parquet sidecar.

        Each row contains:
        - cell_id: the obs index value (string)
        - dataset_id: stable dataset identifier
        - dataset_release: immutable release identifier
        - raw_fields: JSON string of all obs fields for this cell

        This is the Stage 2 Parquet replacement for the legacy SQLite
        raw cell metadata store.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_path = meta_root / f"{self.release_id}-raw-obs.parquet"
        obs = adata.obs
        n = len(obs)

        cell_ids = [str(idx) for idx in obs.index]
        dataset_ids = [self.dataset_id] * n
        dataset_releases = [self.release_id] * n

        # Efficiently build raw_fields as JSON strings per row.
        # Use Series.to_list() for each column and avoid iterrows().
        col_names = list(obs.columns)
        col_lists = {col: obs[col].apply(_safe_serialize).to_list() for col in col_names}

        raw_fields = [
            json.dumps({col: col_lists[col][i] for col in col_names})
            for i in range(n)
        ]

        table = pa.table({
            "cell_id": pa.array(cell_ids, type=pa.string()),
            "dataset_id": pa.array(dataset_ids, type=pa.string()),
            "dataset_release": pa.array(dataset_releases, type=pa.string()),
            "raw_fields": pa.array(raw_fields, type=pa.string()),
        })
        pq.write_table(table, parquet_path)
        return parquet_path

    def _write_raw_feature_metadata_parquet(
        self,
        var_mem: pd.DataFrame,
        meta_root: Path,
    ) -> Path:
        """Write raw feature metadata (var) as a Parquet sidecar.

        Each row contains:
        - origin_index: the feature index in the original var ordering
        - feature_id: the var index value (gene identifier)
        - raw_var: JSON string of all var fields for this feature
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_path = meta_root / f"{self.release_id}-raw-var.parquet"
        n = len(var_mem)

        origin_indices = list(range(n))
        feature_ids = [str(idx) for idx in var_mem.index]

        # Efficient vectorized approach: column-wise apply, then combine
        col_names = list(var_mem.columns)
        col_lists = {col: var_mem[col].apply(_safe_serialize).to_list() for col in col_names}

        raw_var = [
            json.dumps({col: col_lists[col][i] for col in col_names})
            for i in range(n)
        ]

        table = pa.table({
            "origin_index": pa.array(origin_indices, type=pa.int32()),
            "feature_id": pa.array(feature_ids, type=pa.string()),
            "raw_var": pa.array(raw_var, type=pa.string()),
        })
        pq.write_table(table, parquet_path)
        return parquet_path

    def _write_metadata_summary(
        self,
        adata: ad.AnnData,
        var_mem: pd.DataFrame,
        meta_root: Path,
    ) -> Path:
        """Write a per-dataset metadata summary YAML file.

        Summarizes field-level coverage statistics (null fractions, dtypes)
        extracted from the raw obs/var before any canonical mapping.
        """
        obs = adata.obs

        obs_null_fractions: dict[str, float] = {}
        obs_dtypes: dict[str, str] = {}
        for col in obs.columns:
            obs_null_fractions[col] = float(obs[col].isna().mean())
            obs_dtypes[col] = str(obs[col].dtype)

        var_null_fractions: dict[str, float] = {}
        var_dtypes: dict[str, str] = {}
        for col in var_mem.columns:
            var_null_fractions[col] = float(var_mem[col].isna().mean())
            var_dtypes[col] = str(var_mem[col].dtype)

        summary = DatasetMetadataSummary(
            kind="dataset-metadata-summary",
            contract_version=CONTRACT_VERSION,
            dataset_id=self.dataset_id,
            release_id=self.release_id,
            source_path=self.source_path,
            obs_field_count=len(obs.columns),
            var_field_count=len(var_mem.columns),
            obs_null_fractions=obs_null_fractions,
            var_null_fractions=var_null_fractions,
            obs_dtypes=obs_dtypes,
            var_dtypes=var_dtypes,
            obs_rows=len(obs),
            var_rows=len(var_mem),
            obs_index_name=str(obs.index.name or "index"),
            var_index_name=str(var_mem.index.name or "index"),
            notes=(
                f"materialized via Stage2Materializer (schema-independent)",
            ),
        )

        summary_path = meta_root / f"{self.release_id}-metadata-summary.yaml"
        summary.write_yaml(summary_path)
        return summary_path

    def _write_feature_provenance_parquet(
        self,
        var_index: pd.Index,
        n_vars: int,
        meta_root: Path,
        count_source_selected: str = ".X",
        source_path: str = "",
    ) -> Path:
        """Write a feature provenance Parquet recording feature ID ordering.

        Tracks the per-dataset feature ordering (origin_index in original dataset
        var order) and provenance info (source h5ad, count source selected).
        Used by canonicalize-meta to rebuild the corpus feature set.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_path = meta_root / f"{self.release_id}-feature-provenance.parquet"

        origin_indices = list(range(n_vars))
        feature_ids = [str(var_index[i]) for i in range(n_vars)]

        table = pa.table({
            "origin_index": pa.array(origin_indices, type=pa.int32()),
            "feature_id": pa.array(feature_ids, type=pa.string()),
            "count_source": pa.array(
                [count_source_selected] * n_vars, type=pa.string()
            ),
            "source_path": pa.array(
                [source_path] * n_vars, type=pa.string()
            ),
        })
        pq.write_table(table, parquet_path)
        return parquet_path

    def _write_size_factor_parquet(
        self,
        size_factors: np.ndarray,
        cell_ids: pd.Index,
        meta_root: Path,
    ) -> Path:
        """Write per-cell size factors as a separate Parquet (not in cells parquet).

        The array must already be normalized (median-normalized, clamped).
        """
        parquet_path = meta_root / f"{self.release_id}-size-factor.parquet"
        table = pa.table({
            "cell_id": pa.array([str(c) for c in cell_ids], type=pa.string()),
            "size_factor": pa.array(size_factors.tolist(), type=pa.float64()),
        })
        pq.write_table(table, parquet_path)
        return parquet_path

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

    def _write_cells(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        matrix_root: Path,
        *,
        needs_recovery: bool = False,
    ) -> tuple[dict[str, Path], np.ndarray, Path] | dict[str, Path]:
        """Write sparse cell data using the configured backend and topology.

        This method dispatches to either ``_write_cells_federated`` (single dataset)
        or ``_write_cells_aggregate`` (multi-dataset corpus) based on ``self.topology``.

        HVG accumulation and global size factor computation are done during the
        chunk loop; both are finalized and written to sidecars after the loop.

        For aggregate topology, writer state is passed through ``self.writer_state``
        and ``self._is_last_dataset``, set by the corpus-level orchestrator before
        calling ``materialize()``.
        """
        if self.topology == "aggregate":
            return self._write_cells_aggregate(
                count_matrix, adata, matrix_root,
                needs_recovery=needs_recovery,
            )
        return self._write_cells_federated(
            count_matrix, adata, matrix_root, needs_recovery=needs_recovery
        )

    def _write_cells_federated(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        matrix_root: Path,
        *,
        needs_recovery: bool = False,
    ) -> tuple[dict[str, Path], np.ndarray, Path]:
        """Write federated (single-dataset) sparse cell data.

        Loops over the count matrix in chunks, calls ``_translate_chunk()`` per chunk,
        accumulates row sums and HVG statistics during the loop, then computes
        global size factors and finalizes HVG selection after the loop.

        Returns ``(paths_dict, size_factors_array, hvg_sidecar_dir)``.
        """
        from .chunk_translation import DatasetSpec, _finalize_hvg, _translate_chunk

        backend_fn = build_backend_fn(self.backend, "federated")

        n_obs = count_matrix.shape[0]
        n_vars = count_matrix.shape[1]

        # Build DatasetSpec for this dataset.
        # For federated topology, global_row_start=0 (single-dataset corpus).
        dataset_spec = DatasetSpec(
            dataset_id=self.dataset_id,
            dataset_index=0,
            file_path=Path(self.source_path),
            rows=n_obs,
            pairs=0,  # computed from obs if needed
            local_vocabulary_size=n_vars,
            nnz_total=int(count_matrix.nnz) if hasattr(count_matrix, "nnz") else 0,
            global_row_start=0,
            global_row_stop=n_obs,
        )

        # --- Streaming accumulators ---
        all_paths: dict[str, Path] = {}
        all_row_sums: list[np.ndarray] = []
        sum_log1p = np.zeros(n_vars, dtype=np.float64)
        sum_log1p_sq = np.zeros(n_vars, dtype=np.float64)
        n_cells_total = 0

        # --- Stateful writer for multi-chunk streaming ---
        # A single writer_state is used across all backends. backend_fn (the
        # thin wrapper from backends/__init__.py) handles backend-specific
        # state initialization, streaming writes, and finalization.
        writer_state: dict | None = None

        for chunk_start in range(0, n_obs, self.chunk_rows):
            chunk_end = min(chunk_start + self.chunk_rows, n_obs)
            is_last = (chunk_end == n_obs)

            # Slice the CSR matrix chunk.
            if hasattr(count_matrix, "local_slice"):
                matrix_chunk = count_matrix.local_slice(chunk_start, chunk_end)
            else:
                matrix_chunk = count_matrix[chunk_start:chunk_end].tocsr()

            # Translate the chunk via the shared translation layer.
            bundle = _translate_chunk(
                dataset=dataset_spec,
                matrix_chunk=matrix_chunk,
                chunk_start=chunk_start,
                needs_recovery=needs_recovery,
            )

            # Accumulate row sums for global size factor computation.
            all_row_sums.append(bundle.row_sums)

            # Accumulate HVG statistics via np.add.at on dataset-local indices.
            log1p_counts = np.log1p(bundle.counts.astype(np.float64))
            np.add.at(sum_log1p, bundle.indices, log1p_counts)
            np.add.at(sum_log1p_sq, bundle.indices, log1p_counts ** 2)
            n_cells_total += bundle.row_count

            # Call the thin backend serializer with state management.
            # backend_fn returns (paths_dict, state_or_none).
            # On first chunk: writer_state is None, writer opens/initializes.
            # On subsequent chunks: writer_state passed back, writer reuses/appends.
            # On last chunk: _is_last_chunk=True, writer closes/commits, returns None.
            cell_ids_chunk: tuple[str, ...] | None = None
            if self.backend == "webdataset":
                cell_ids_chunk = tuple(adata.obs.index[chunk_start:chunk_end])

            paths, writer_state = backend_fn(
                bundle=bundle,
                release_id=self.release_id,
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=is_last,
                cell_ids=cell_ids_chunk,
                dataset_id=self.dataset_id,
            )

            all_paths.update(paths)

        # --- After loop: compute GLOBAL size factors from all row sums ---
        global_row_sums = np.concatenate(all_row_sums)
        global_median = float(np.median(global_row_sums))
        if global_median > 0:
            size_factors = global_row_sums / global_median
        else:
            size_factors = np.ones_like(global_row_sums)
        size_factors = np.where(size_factors <= 0, 1.0, size_factors)
        size_factors = np.where(np.isnan(size_factors), 1.0, size_factors)
        size_factors = size_factors.astype(np.float32)

        # --- After loop: write size factor Parquet sidecar ---
        size_factor_parquet_path = self._write_size_factor_parquet(
            size_factors=size_factors,
            cell_ids=adata.obs.index,
            meta_root=Path(self.output_roots.metadata_root),
        )

        # --- After loop: compute Seurat-style HVG from streaming accumulators ---
        hvg_indices, nonhvg_indices = _finalize_hvg(
            sum_log1p, sum_log1p_sq, n_cells_total, n_vars, n_hvg=self.n_hvg
        )

        # --- After loop: write HVG sidecar ---
        meta_root = Path(self.output_roots.metadata_root)
        hvg_sidecar_dir = meta_root / "hvg_sidecar"
        hvg_sidecar_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(hvg_sidecar_dir / "hvg.npy"), hvg_indices, allow_pickle=False)
        np.save(str(hvg_sidecar_dir / "nonhvg.npy"), nonhvg_indices, allow_pickle=False)

        return (all_paths, size_factors, hvg_sidecar_dir)

    def _write_cells_aggregate(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        matrix_root: Path,
        *,
        needs_recovery: bool = False,
    ) -> tuple[dict[str, Path], np.ndarray, Path]:
        """Write aggregate (multi-dataset corpus) sparse cell data.

        Loops over the count matrix in chunks, calling the aggregate backend
        writer per chunk with shared ``self.writer_state``. The writer state is
        carried across datasets via ``self.writer_state``, which the corpus-level
        orchestrator sets before each ``materialize()`` call and reads afterward.

        Row sums and HVG statistics are accumulated during the chunk loop.
        Global size factors and HVG selection are computed and written to
        per-dataset sidecars after the loop (same as federated).

        Returns ``(paths_dict, size_factors_array, hvg_sidecar_dir)`` — same
        format as ``_write_cells_federated``.

        Note: Only ``lance``, ``zarr``, and ``webdataset`` backends support
        aggregate topology. ``arrow-parquet`` and ``arrow-ipc`` do not support
        aggregate because they lack true incremental append capability.
        """
        from .chunk_translation import DatasetSpec, _finalize_hvg, _translate_chunk

        backend_fn = build_backend_fn(self.backend, "aggregate")

        n_obs = count_matrix.shape[0]
        n_vars = count_matrix.shape[1]

        # Build DatasetSpec with the configured global_row_start (pre-computed
        # by the corpus-level orchestrator to ensure contiguous global_row_index).
        dataset_spec = DatasetSpec(
            dataset_id=self.dataset_id,
            dataset_index=self.dataset_index,
            file_path=Path(self.source_path),
            rows=n_obs,
            pairs=0,
            local_vocabulary_size=n_vars,
            nnz_total=int(count_matrix.nnz) if hasattr(count_matrix, "nnz") else 0,
            global_row_start=self.global_row_start,
            global_row_stop=self.global_row_start + n_obs,
        )

        # --- Streaming accumulators ---
        all_paths: dict[str, Path] = {}
        all_row_sums: list[np.ndarray] = []
        sum_log1p = np.zeros(n_vars, dtype=np.float64)
        sum_log1p_sq = np.zeros(n_vars, dtype=np.float64)
        n_cells_total = 0

        # Carry writer state from previous dataset (None on first call).
        writer_state: dict | None = self.writer_state

        for chunk_start in range(0, n_obs, self.chunk_rows):
            chunk_end = min(chunk_start + self.chunk_rows, n_obs)
            # is_last_chunk is True only when this is the final chunk of the
            # final dataset — triggers aggregate backend finalization.
            is_last_chunk = (chunk_end == n_obs) and self._is_last_dataset

            # Slice the CSR matrix chunk.
            if hasattr(count_matrix, "local_slice"):
                matrix_chunk = count_matrix.local_slice(chunk_start, chunk_end)
            else:
                matrix_chunk = count_matrix[chunk_start:chunk_end].tocsr()

            # Translate the chunk via the shared translation layer.
            bundle = _translate_chunk(
                dataset=dataset_spec,
                matrix_chunk=matrix_chunk,
                chunk_start=chunk_start,
                needs_recovery=needs_recovery,
            )

            # Accumulate row sums for global size factor computation.
            all_row_sums.append(bundle.row_sums)

            # Accumulate HVG statistics via np.add.at on dataset-local indices.
            log1p_counts = np.log1p(bundle.counts.astype(np.float64))
            np.add.at(sum_log1p, bundle.indices, log1p_counts)
            np.add.at(sum_log1p_sq, bundle.indices, log1p_counts ** 2)
            n_cells_total += bundle.row_count

            # Call the aggregate backend writer with streaming state.
            cell_ids_chunk: tuple[str, ...] | None = None
            if self.backend == "webdataset":
                cell_ids_chunk = tuple(adata.obs.index[chunk_start:chunk_end])

            paths, writer_state = backend_fn(
                bundle=bundle,
                release_id=self.release_id,
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=is_last_chunk,
                cell_ids=cell_ids_chunk,
                dataset_id=self.dataset_id,
            )

            all_paths.update(paths)

        # Store final writer state (None after last dataset) for next iteration.
        self.writer_state = writer_state

        # --- After loop: compute GLOBAL size factors from all row sums ---
        global_row_sums = np.concatenate(all_row_sums)
        global_median = float(np.median(global_row_sums))
        if global_median > 0:
            size_factors = global_row_sums / global_median
        else:
            size_factors = np.ones_like(global_row_sums)
        size_factors = np.where(size_factors <= 0, 1.0, size_factors)
        size_factors = np.where(np.isnan(size_factors), 1.0, size_factors)
        size_factors = size_factors.astype(np.float32)

        # --- After loop: write size factor Parquet sidecar ---
        size_factor_parquet_path = self._write_size_factor_parquet(
            size_factors=size_factors,
            cell_ids=adata.obs.index,
            meta_root=Path(self.output_roots.metadata_root),
        )

        # --- After loop: compute Seurat-style HVG from streaming accumulators ---
        hvg_indices, nonhvg_indices = _finalize_hvg(
            sum_log1p, sum_log1p_sq, n_cells_total, n_vars, n_hvg=self.n_hvg
        )

        # --- After loop: write HVG sidecar ---
        meta_root = Path(self.output_roots.metadata_root)
        hvg_sidecar_dir = meta_root / "hvg_sidecar"
        hvg_sidecar_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(hvg_sidecar_dir / "hvg.npy"), hvg_indices, allow_pickle=False)
        np.save(str(hvg_sidecar_dir / "nonhvg.npy"), nonhvg_indices, allow_pickle=False)

        return (all_paths, size_factors, hvg_sidecar_dir)


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
# Corpus index updater
# ---------------------------------------------------------------------------


def update_corpus_index(
    corpus_index_path: Path,
    new_dataset_record: DatasetJoinRecord,
    global_metadata: GlobalMetadataDocument | None = None,
    emission_spec_path: str | None = None,
    backend: str | None = None,
    topology: str | None = None,
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
        Backend declaration for the corpus (e.g., "arrow-parquet", "arrow-ipc",
        "webdataset", "zarr", "lance"). Required when creating a new corpus; written to
        ``global-metadata.yaml``.
    topology : str | None
        Corpus topology for the Parquet ledger entry (e.g., "federated",
        "aggregate"). Required for new corpus creation; inferred from ``backend``
        for known backends (lance → aggregate, others → federated) when not
        explicitly provided.
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

    # Write Parquet corpus ledger (Stage 2 contract: Parquet primary)
    # Rebuild the full ledger from the updated corpus index to ensure consistency.
    # This is safe because we always load the existing YAML index on append.
    ledger_path = corpus_index_path.parent / "corpus-ledger.parquet"
    # Resolve topology: use explicit parameter, infer from backend, or default to federated
    effective_topology = topology or ("aggregate" if backend == "lance" else "federated")
    _write_corpus_ledger_parquet(ledger_path, updated, backend, effective_topology)

    return updated


def _backfill_feature_count(manifest_path: Path) -> int:
    """Read a materialization manifest and return the feature_count.

    Parameters
    ----------
    manifest_path : Path
        Path to a materialization-manifest.yaml.

    Returns
    -------
    int
        The feature_count from the manifest, or 0 if unreadable.
    """
    try:
        m = MaterializationManifest.from_yaml_file(manifest_path)
        return m.feature_count
    except Exception:
        return 0


def _write_corpus_ledger_parquet(
    ledger_path: Path,
    corpus: CorpusIndexDocument,
    backend: str | None,
    topology: str,
) -> None:
    """Write or overwrite the Parquet corpus ledger from the corpus index.

    The Parquet ledger is the primary machine-readable artifact for Stage 2
    corpus tracking. The YAML corpus-index.yaml remains for human review.

    feature_count is back-filled from each dataset's materialization-manifest.yaml
    when the manifest path is resolvable from the corpus root.
    """
    import datetime

    corpus_root = ledger_path.parent

    entries = []
    for d in corpus.datasets:
        # Back-fill feature_count from the per-dataset manifest
        feature_count = 0
        manifest_path = corpus_root / d.manifest_path
        if manifest_path.exists():
            feature_count = _backfill_feature_count(manifest_path)

        entry = CorpusLedgerEntry(
            corpus_id=corpus.corpus_id,
            dataset_id=d.dataset_id,
            release_id=d.release_id,
            dataset_index=d.dataset_index,
            join_mode=d.join_mode,
            manifest_path=d.manifest_path,
            backend=backend or "arrow-parquet",
            topology=topology,
            cell_count=d.cell_count,
            feature_count=feature_count,
            global_start=d.global_start,
            global_end=d.global_end,
            created_at=datetime.datetime.utcnow().isoformat(),
        )
        entry.validate()
        entries.append(entry.to_dict())

    if not entries:
        return

    table = pa.table({
        "corpus_id": pa.array([e["corpus_id"] for e in entries], type=pa.string()),
        "dataset_id": pa.array([e["dataset_id"] for e in entries], type=pa.string()),
        "release_id": pa.array([e["release_id"] for e in entries], type=pa.string()),
        "dataset_index": pa.array([e["dataset_index"] for e in entries], type=pa.int32()),
        "join_mode": pa.array([e["join_mode"] for e in entries], type=pa.string()),
        "manifest_path": pa.array([e["manifest_path"] for e in entries], type=pa.string()),
        "backend": pa.array([e["backend"] for e in entries], type=pa.string()),
        "topology": pa.array([e["topology"] for e in entries], type=pa.string()),
        "cell_count": pa.array([e["cell_count"] for e in entries], type=pa.int64()),
        "feature_count": pa.array([e["feature_count"] for e in entries], type=pa.int64()),
        "global_start": pa.array([e["global_start"] for e in entries], type=pa.int64()),
        "global_end": pa.array([e["global_end"] for e in entries], type=pa.int64()),
        "created_at": pa.array([e["created_at"] for e in entries], type=pa.string()),
    })
    pq.write_table(table, str(ledger_path))
