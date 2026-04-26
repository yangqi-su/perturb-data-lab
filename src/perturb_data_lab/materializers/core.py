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

from .backends import build_backend_fn
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
from .schema_execution import resolve_all_cell_rows, resolve_all_feature_rows
from .validation import validate_schema_readiness
from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL
from ..inspectors.models import DatasetSummaryDocument, SchemaDocument


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
        corpus_index_path: str | None = None,
        corpus_id: str | None = None,
        register: bool = False,
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
        self.corpus_index_path = corpus_index_path
        self.corpus_id = corpus_id
        self.register = register
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

            # --- Write feature provenance Parquet ---
            provenance_spec_path = self._write_feature_provenance_parquet(
                var_index=var_index,
                n_vars=n_vars,
                meta_root=meta_root,
                count_source_selected=count_source.selected,
                source_path=str(source_h5ad),
            )

            # --- Write backend-specific sparse cell data ---
            # Size factors are computed inline during the write pass (one scan, not two),
            # then written as a separate Parquet. The backend returns
            # ``(paths_dict, computed_normalized_factors)``.
            backend_result: tuple[dict[str, Path], np.ndarray] | dict[str, Path]
            backend_result = self._write_cells(
                count_matrix=count_matrix,
                adata=adata,
                matrix_root=matrix_root,
            )
            if isinstance(backend_result, tuple):
                backend_paths, size_factors = backend_result
                size_factor_parquet_path = self._write_size_factor_parquet(
                    size_factors=size_factors,
                    cell_ids=adata.obs.index,
                    meta_root=meta_root,
                )
            else:
                backend_paths = backend_result
                size_factor_parquet_path = None  # backend does not provide size factors

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

    def _compute_and_write_hvg_arrays(
        self,
        count_matrix: Any,
        n_vars: int,
        meta_root: Path,
        n_hvg: int = 2000,
    ) -> Path:
        """Compute HVG/non-HVG index arrays in dataset-local feature space and write as NumPy.

        Uses batched row access for sparse matrices to avoid O(sample_cells)
        individual row slices on large backed datasets.
        """
        from scipy.sparse import issparse

        sidecar_dir = meta_root / "hvg_sidecar"
        sidecar_dir.mkdir(parents=True, exist_ok=True)

        n_obs = count_matrix.shape[0]
        sample_cells = min(512, n_obs)

        log_expr = np.zeros((sample_cells, n_vars), dtype=np.float64)

        # Check if backed anndata _CSRDataset (class name check avoids import)
        is_backed_csr = count_matrix.__class__.__name__ == "_CSRDataset"
        if is_backed_csr or issparse(count_matrix):
            # Sample a contiguous block and extract in one batch CSR conversion.
            # For large datasets this avoids O(sample_cells) individual row slices.
            start_idx = max(0, (n_obs - sample_cells) // 2)
            end_idx = start_idx + sample_cells
            batch = count_matrix[start_idx:end_idx].tocsr()
            batch_indptr = batch.indptr
            batch_data = batch.data.astype(np.float64)
            batch_indices = batch.indices
            for local_i in range(sample_cells):
                j_start = batch_indptr[local_i]
                j_end = batch_indptr[local_i + 1]
                for j_pos in range(j_start, j_end):
                    log_expr[local_i, batch_indices[j_pos]] = np.log1p(batch_data[j_pos])
        else:
            # Dense path
            cell_indices = np.linspace(0, n_obs - 1, num=sample_cells, dtype=int)
            for row_i, ci in enumerate(cell_indices):
                row = np.asarray(count_matrix[ci]).ravel()
                nonzero_mask = row != 0
                for idx in np.where(nonzero_mask)[0]:
                    log_expr[row_i, idx] = np.log1p(float(row[idx]))

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

    def _write_cells(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        matrix_root: Path,
    ) -> tuple[dict[str, Path], np.ndarray] | dict[str, Path]:
        """Write sparse cell data using the configured backend and topology.

        This method dispatches to either ``_write_cells_federated`` (single dataset)
        or ``_write_cells_aggregate`` (multi-dataset corpus) based on ``self.topology``.

        Size factors are computed inline by ``_translate_chunk()`` during the same
        CSR traversal as sparse translation — one pass, not two.
        """
        if self.topology == "aggregate":
            raise NotImplementedError(
                "aggregate topology requires a corpus-level orchestrator that "
                "collects ChunkBundles from multiple Stage2Materializer calls "
                "and passes them to the aggregate writer. "
                "Use federated topology for single-dataset materialization."
            )
        return self._write_cells_federated(count_matrix, adata, matrix_root)

    def _write_cells_federated(
        self,
        count_matrix: Any,
        adata: ad.AnnData,
        matrix_root: Path,
    ) -> tuple[dict[str, Path], np.ndarray]:
        """Write federated (single-dataset) sparse cell data.

        Loops over the count matrix in chunks, calls ``_translate_chunk()`` per chunk,
        and passes each ``ChunkBundle`` to the thin backend serializer.

        Returns ``(paths_dict, size_factors_array)``.
        """
        from .chunk_translation import DatasetSpec, _translate_chunk

        backend_fn = build_backend_fn(self.backend, "federated")

        n_obs = count_matrix.shape[0]

        # Build DatasetSpec for this dataset.
        # For federated topology, global_row_start=0 (single-dataset corpus).
        dataset_spec = DatasetSpec(
            dataset_id=self.dataset_id,
            dataset_index=0,
            file_path=Path(self.source_path),
            rows=n_obs,
            pairs=0,  # computed from obs if needed
            local_vocabulary_size=count_matrix.shape[1],
            nnz_total=int(count_matrix.nnz) if hasattr(count_matrix, "nnz") else 0,
            global_row_start=0,
            global_row_stop=n_obs,
        )

        CHUNK_ROWS = 100_000  # rows per chunk

        all_paths: dict[str, Path] = {}
        all_size_factors_list: list[np.ndarray] = []

        for chunk_start in range(0, n_obs, CHUNK_ROWS):
            chunk_end = min(chunk_start + CHUNK_ROWS, n_obs)

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
            )

            # Call the thin backend serializer.
            paths, size_factors = backend_fn(
                bundle=bundle,
                release_id=self.release_id,
                matrix_root=matrix_root,
            )
            all_paths.update(paths)
            all_size_factors_list.append(size_factors)

        # Concatenate all size factor arrays.
        all_size_factors = np.concatenate(all_size_factors_list) if all_size_factors_list else np.array([])

        return (all_paths, all_size_factors)

    def _write_cells_aggregate(
        self,
        bundles: list[Any],
        matrix_root: Path,
    ) -> tuple[dict[str, Path], list[np.ndarray]]:
        """Write aggregate (multi-dataset corpus) sparse cell data.

        This is called by a corpus-level orchestrator after collecting
        ``ChunkBundle`` objects from multiple ``Stage2Materializer`` calls.
        """
        from .chunk_translation import ChunkBundle

        backend_fn = build_backend_fn(self.backend, "aggregate")

        paths, size_factors_list = backend_fn(
            bundles=bundles,
            matrix_root=matrix_root,
        )
        return (paths, size_factors_list)


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
