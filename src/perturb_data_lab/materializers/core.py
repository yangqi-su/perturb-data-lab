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
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, issparse

from .backends import build_backend_fn
from .obs_filter import ObsFilterError, filter_obs_rows


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
    CorpusIndexDocument,
    CountSourceSpec,
    DatasetJoinRecord,
    GlobalMetadataDocument,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
)
from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL
from ..inspectors.models import DatasetSummaryDocument


def _slice_matrix_chunk_as_csr(
    count_matrix: Any,
    chunk_start: int,
    chunk_end: int,
) -> csr_matrix:
    """Slice a row chunk and normalize it to SciPy CSR.

    Aggregate/federated write paths operate on sparse chunk translation. Some
    datasets expose direct count matrices as dense numpy arrays after backed
    slicing, so convert those chunks to CSR before translation instead of
    assuming ``.tocsr()`` exists on the slice object.
    """
    if hasattr(count_matrix, "local_slice"):
        matrix_chunk = count_matrix.local_slice(chunk_start, chunk_end)
    else:
        matrix_chunk = count_matrix[chunk_start:chunk_end]

    if issparse(matrix_chunk):
        return matrix_chunk.tocsr(copy=False)
    return csr_matrix(matrix_chunk)


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
    dataset_id : str
        Stable dataset identifier.
    backend : str
        Storage backend: ``lance`` (default) or ``zarr``.
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
        ``register=True``.
    corpus_id : str | None, default None
        Corpus identifier. Required when registering a new corpus.
    register : bool, default False
        If True, automatically register this dataset with ``corpus-index.yaml``
        after successful materialization. Requires ``corpus_index_path`` to be set.
        When True, the returned manifest's ``corpus_registration`` field is
        populated and the manifest YAML is rewritten with registration metadata.
    mode : str, default "create"
        Materialization mode: ``"create"`` (first dataset in a new corpus) or
        ``"append"`` (subsequent dataset appended to an existing corpus).
        Controls the manifest ``route`` field and corpus-index ``join_mode``.
        ``"create"`` produces ``route="create_new"``; ``"append"`` produces
        ``route="append_routed"``.
    dataset_index : int, default 0
        Position of this dataset in the corpus (0-based). Used for global row
        range computation in aggregate topology.
    global_row_start : int, default 0
        Starting row index in the global corpus for this dataset's cells.
        Pre-computed by the corpus-level orchestrator to ensure contiguous
        global row indices in aggregate topology.
    writer_state : dict | None, default None
        Opaque writer state carried across datasets in aggregate topology.
        The corpus-level orchestrator reads this from the previous dataset's
        materializer and passes it to the next. None on the first dataset.
        Only meaningful for aggregate topology backends (lance, zarr).
    _is_last_dataset : bool, default False
        When True, signals to the aggregate backend that this is the final
        dataset in the series, triggering backend finalization.
        Only meaningful for aggregate topology.

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
    ``lance`` × ``federated``.
    """

    def __init__(
        self,
        source_path: str,
        review_bundle_path: str,
        output_roots: OutputRoots,
        dataset_id: str,
        backend: str = "lance",
        topology: str = "federated",
        rerun_stage1: bool = False,
        n_hvg: int = 2000,
        chunk_rows: int = 100_000,
        corpus_index_path: str | None = None,
        corpus_id: str | None = None,
        register: bool = False,
        mode: str = "create",
        dataset_index: int = 0,
        global_row_start: int = 0,
        writer_state: dict | None = None,
        _is_last_dataset: bool = False,
    ):
        if mode not in ("create", "append"):
            raise ValueError(
                f"mode must be 'create' or 'append', got {mode!r}"
            )
        assert backend in {"lance", "zarr"}, f"unknown backend: {backend}"
        assert topology in {"federated", "aggregate"}, f"unknown topology: {topology}"
        self.source_path = source_path
        self.review_bundle_path = review_bundle_path
        self.output_roots = output_roots
        self.dataset_id = dataset_id
        self.backend = backend
        self.topology = topology
        self.rerun_stage1 = rerun_stage1
        self.n_hvg = n_hvg
        self.chunk_rows = chunk_rows
        self.corpus_index_path = corpus_index_path
        self.corpus_id = corpus_id
        self.register = register
        self.mode = mode
        self.dataset_index = dataset_index
        self.global_row_start = global_row_start
        self.writer_state = writer_state
        self._is_last_dataset = _is_last_dataset
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
            source_obs = adata.obs.to_memory() if hasattr(adata.obs, "to_memory") else adata.obs
            retained_source_rows = filter_obs_rows(source_obs, summary.obs_filter)
            if len(retained_source_rows) == 0:
                raise ObsFilterError(
                    f"obs_filter retained zero rows for dataset {self.dataset_id}"
                )
            filtered_obs = source_obs.iloc[retained_source_rows].copy()
            n_obs = len(filtered_obs)
            filter_applied = summary.obs_filter is not None and bool(summary.obs_filter.strip())
            source_obs_rows = int(adata.n_obs)

            # Determine var space for the selected count source
            if count_source.selected == ".raw.X":
                n_vars = int(adata.raw.shape[1])
                var_ref = adata.raw.var
                var_index = adata.raw.var.index
            else:
                n_vars = adata.n_vars
                var_ref = adata.var
                var_index = adata.var.index

            count_matrix = self._select_count_matrix(adata, count_source.selected)
            if filter_applied:
                count_matrix = _RowSubsetMatrixView(count_matrix, retained_source_rows)

            # --- Ensure output directories exist ---
            meta_root = Path(self.output_roots.metadata_root)
            matrix_root = Path(self.output_roots.matrix_root)
            meta_root.mkdir(parents=True, exist_ok=True)
            matrix_root.mkdir(parents=True, exist_ok=True)

            # Keep dataset meta self-contained by copying the authoritative
            # Stage 1 inspection summary into meta_root/dataset-summary.yaml.
            summary_copy_path = meta_root / "dataset-summary.yaml"
            if summary_path.resolve() != summary_copy_path.resolve():
                shutil.copy2(summary_path, summary_copy_path)

            # --- Write raw cell metadata (Parquet, not SQLite) ---
            raw_cell_meta_parquet_path = self._write_raw_cell_metadata_parquet(
                obs=filtered_obs,
                source_row_indices=(retained_source_rows if filter_applied else None),
                meta_root=meta_root,
            )

            # --- Write raw feature metadata (Parquet) ---
            var_mem = var_ref.to_memory() if hasattr(var_ref, "to_memory") else var_ref
            raw_feature_meta_parquet_path = self._write_raw_feature_metadata_parquet(
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
            # HVG statistics are accumulated during the chunk loop via np.add.at.
            # Global size factors are computed after the loop from all row_sums.
            # Size factors and the canonical HVG ranking parquet are written after the loop.
            backend_paths, size_factor_parquet_path, hvg_ranking_path = self._write_cells(
                count_matrix=count_matrix,
                matrix_root=matrix_root,
                cell_ids=filtered_obs.index,
                feature_ids=tuple(str(v) for v in var_index),
                needs_recovery=count_source.uses_recovery,
            )
            self._assert_backend_outputs_exist(backend_paths)

            manifest_outputs = OutputRoots(
                metadata_root=self._manifest_artifact_path(meta_root),
                matrix_root=self._manifest_artifact_path(matrix_root),
            )

            # --- Build final manifest ---
            manifest = MaterializationManifest(
                kind="materialization-manifest",
                contract_version=CONTRACT_VERSION,
                dataset_id=self.dataset_id,
                route="create_new" if self.mode == "create" else "append_routed",
                backend=self.backend,
                topology=self.topology,
                count_source=count_source,
                outputs=manifest_outputs,
                provenance=ProvenanceSpec(
                    source_path=str(source_h5ad),
                    review_bundle=self._manifest_artifact_path(summary_copy_path),
                ),
                raw_cell_meta_path=self._manifest_artifact_path(raw_cell_meta_parquet_path),
                raw_feature_meta_path=self._manifest_artifact_path(raw_feature_meta_parquet_path),
                provenance_spec_path=self._manifest_artifact_path(provenance_spec_path),
                size_factor_parquet_path=self._manifest_artifact_path(size_factor_parquet_path),
                hvg_ranking_path=self._manifest_artifact_path(hvg_ranking_path),
                default_n_hvg=self.n_hvg,
                integer_verified=True,
                cell_count=n_obs,
                feature_count=n_vars,
                notes=(
                    f"materialized via Stage2Materializer (schema-independent, count-first)",
                    f"topology={self.topology}",
                    *(
                        (
                            f"obs_filter retained {n_obs}/{source_obs_rows} rows",
                        )
                        if filter_applied
                        else ()
                    ),
                ),
            )
            manifest.validate()

            manifest_path = meta_root / "materialization-manifest.yaml"
            manifest.write_yaml(manifest_path)

            # --- Phase 4: Corpus registration ---
            # If register=True, register this dataset with corpus-index.yaml.
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
                    corpus_index_path=self._manifest_artifact_path(Path(self.corpus_index_path)),
                    dataset_index=join_record.dataset_index,
                    global_start=join_record.global_start,
                    global_end=join_record.global_end,
                )
                manifest = replace(manifest, corpus_registration=reg_info)
                # Re-write manifest with registration info
                manifest_path.write_text(manifest.to_yaml(), encoding="utf-8")

            return manifest

        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    def _manifest_artifact_path(self, path: Path | str | None) -> str | None:
        if path is None:
            return None
        artifact_path = Path(path)
        if self.corpus_index_path is None:
            return str(artifact_path)

        corpus_root = Path(self.corpus_index_path).parent.resolve()
        try:
            return str(artifact_path.resolve().relative_to(corpus_root))
        except ValueError as exc:
            raise ValueError(
                f"materialized artifact path is outside corpus root: {artifact_path}"
            ) from exc

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
        source_release = self.dataset_id

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
        obs: pd.DataFrame,
        source_row_indices: np.ndarray | None,
        meta_root: Path,
    ) -> Path:
        """Write raw cell metadata (obs) as a Parquet sidecar.

        Each row contains:
        - cell_id: the obs index value (string)
        - dataset_id: stable dataset identifier
        - optional source_row_index/source_obs_index provenance for filtered runs
        - raw_fields: JSON string of all obs fields for this cell

        This is the Stage 2 Parquet replacement for the legacy SQLite
        raw cell metadata store.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_path = meta_root / "raw-obs.parquet"
        n = len(obs)

        cell_ids = [str(idx) for idx in obs.index]
        dataset_ids = [self.dataset_id] * n

        # Efficiently build raw_fields as JSON strings per row.
        # Use Series.to_list() for each column and avoid iterrows().
        col_names = [
            col for col in obs.columns if col not in {"source_row_index", "source_obs_index"}
        ]
        col_lists = {col: obs[col].apply(_safe_serialize).to_list() for col in col_names}

        raw_fields = [
            json.dumps({col: col_lists[col][i] for col in col_names})
            for i in range(n)
        ]

        columns: dict[str, pa.Array] = {
            "cell_id": pa.array(cell_ids, type=pa.string()),
            "dataset_id": pa.array(dataset_ids, type=pa.string()),
            "raw_fields": pa.array(raw_fields, type=pa.string()),
        }
        if source_row_indices is not None:
            columns["source_row_index"] = pa.array(
                source_row_indices.tolist(), type=pa.int64()
            )
            columns["source_obs_index"] = pa.array(cell_ids, type=pa.string())
        table = pa.table(columns)
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

        parquet_path = meta_root / "raw-var.parquet"
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

        parquet_path = meta_root / "feature-provenance.parquet"

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
        parquet_path = meta_root / "size-factor.parquet"
        table = pa.table({
            "cell_id": pa.array([str(c) for c in cell_ids], type=pa.string()),
            "size_factor": pa.array(size_factors.tolist(), type=pa.float64()),
        })
        pq.write_table(table, parquet_path)
        return parquet_path

    def _assert_backend_outputs_exist(self, backend_paths: dict[str, Path]) -> None:
        for name, path in backend_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"backend output {name!r} was not written: {path}")

    def _write_cells(
        self,
        count_matrix: Any,
        matrix_root: Path,
        *,
        cell_ids: pd.Index,
        feature_ids: Sequence[str],
        needs_recovery: bool = False,
    ) -> tuple[dict[str, Path], Path, Path]:
        """Write sparse cell data using the configured backend and topology.

        This method dispatches to either ``_write_cells_federated`` (single dataset)
        or ``_write_cells_aggregate`` (multi-dataset corpus) based on ``self.topology``.

        HVG accumulation and global size factor computation are done during the
        chunk loop; the canonical ``hvg.parquet`` artifact is finalized after the loop.

        For aggregate topology, writer state is passed through ``self.writer_state``
        and ``self._is_last_dataset``, set by the corpus-level orchestrator before
        calling ``materialize()``.
        """
        if self.topology == "aggregate":
            return self._write_cells_aggregate(
                count_matrix,
                matrix_root,
                cell_ids=cell_ids,
                feature_ids=feature_ids,
                needs_recovery=needs_recovery,
            )
        return self._write_cells_federated(
            count_matrix,
            matrix_root,
            cell_ids=cell_ids,
            feature_ids=feature_ids,
            needs_recovery=needs_recovery,
        )

    def _write_hvg_ranking_parquet(
        self,
        *,
        sum_log1p: np.ndarray,
        sum_log1p_sq: np.ndarray,
        n_cells_total: int,
        feature_ids: Sequence[str],
        meta_root: Path,
    ) -> Path:
        """Write the canonical per-dataset HVG ranking artifact."""
        from .chunk_translation import _build_hvg_ranking_table

        table = _build_hvg_ranking_table(
            sum_log1p=sum_log1p,
            sum_log1p_sq=sum_log1p_sq,
            n_cells_total=n_cells_total,
            feature_ids=feature_ids,
            n_hvg=self.n_hvg,
        )
        output_path = meta_root / "hvg.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path)
        return output_path

    def _write_cells_federated(
        self,
        count_matrix: Any,
        matrix_root: Path,
        *,
        cell_ids: pd.Index,
        feature_ids: Sequence[str],
        needs_recovery: bool = False,
    ) -> tuple[dict[str, Path], Path, Path]:
        """Write federated (single-dataset) sparse cell data.

        Loops over the count matrix in chunks, calls ``_translate_chunk()`` per chunk,
        accumulates row sums and HVG statistics during the loop, then computes
        global size factors and writes ``hvg.parquet`` after the loop.

        Returns ``(paths_dict, size_factor_parquet_path, hvg_ranking_path)``.
        """
        from .chunk_translation import DatasetSpec, _translate_chunk

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

            # Slice the matrix chunk and normalize dense/sparse inputs to CSR.
            matrix_chunk = _slice_matrix_chunk_as_csr(
                count_matrix,
                chunk_start,
                chunk_end,
            )

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
            paths, writer_state = backend_fn(
                bundle=bundle,
                dataset_id=self.dataset_id,
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=is_last,
                local_vocabulary_size=n_vars,
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
            cell_ids=cell_ids,
            meta_root=Path(self.output_roots.metadata_root),
        )

        meta_root = Path(self.output_roots.metadata_root)
        hvg_ranking_path = self._write_hvg_ranking_parquet(
            sum_log1p=sum_log1p,
            sum_log1p_sq=sum_log1p_sq,
            n_cells_total=n_cells_total,
            feature_ids=feature_ids,
            meta_root=meta_root,
        )

        return (all_paths, size_factor_parquet_path, hvg_ranking_path)

    def _write_cells_aggregate(
        self,
        count_matrix: Any,
        matrix_root: Path,
        *,
        cell_ids: pd.Index,
        feature_ids: Sequence[str],
        needs_recovery: bool = False,
    ) -> tuple[dict[str, Path], Path, Path]:
        """Write aggregate (multi-dataset corpus) sparse cell data.

        Loops over the count matrix in chunks, calling the aggregate backend
        writer per chunk with shared ``self.writer_state``. The writer state is
        carried across datasets via ``self.writer_state``, which the corpus-level
        orchestrator sets before each ``materialize()`` call and reads afterward.

        Row sums and HVG statistics are accumulated during the chunk loop.
        Global size factors and the canonical HVG ranking parquet are written
        after the loop (same as federated).

        Returns ``(paths_dict, size_factor_parquet_path, hvg_ranking_path)`` — same
        format as ``_write_cells_federated``.

        Note: Only ``lance`` and ``zarr`` backends support aggregate topology in
        slim main.
        """
        from .chunk_translation import DatasetSpec, _translate_chunk

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

            # Slice the matrix chunk and normalize dense/sparse inputs to CSR.
            matrix_chunk = _slice_matrix_chunk_as_csr(
                count_matrix,
                chunk_start,
                chunk_end,
            )

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
            paths, writer_state = backend_fn(
                bundle=bundle,
                dataset_id=self.dataset_id,
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=is_last_chunk,
                local_vocabulary_size=n_vars,
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
            cell_ids=cell_ids,
            meta_root=Path(self.output_roots.metadata_root),
        )

        meta_root = Path(self.output_roots.metadata_root)
        hvg_ranking_path = self._write_hvg_ranking_parquet(
            sum_log1p=sum_log1p,
            sum_log1p_sq=sum_log1p_sq,
            n_cells_total=n_cells_total,
            feature_ids=feature_ids,
            meta_root=meta_root,
        )

        return (all_paths, size_factor_parquet_path, hvg_ranking_path)


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


class _RowSubsetMatrixView:
    """Minimal row-subset view for backed count matrices."""

    def __init__(self, matrix: Any, row_indices: np.ndarray):
        self._matrix = matrix
        self._row_indices = np.asarray(row_indices, dtype=np.int64)
        self.shape = (len(self._row_indices), int(matrix.shape[1]))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            rows = self._row_indices[key]
        else:
            rows = self._row_indices[key]
        return self._matrix[rows]

    def local_slice(self, start: int, end: int) -> Any:
        return self._matrix[self._row_indices[start:end]]


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
        Backend declaration for the corpus (e.g., "lance", "zarr"). Required when creating a new corpus; written to
        ``global-metadata.yaml``.
    topology : str | None
        Corpus topology ("federated" or "aggregate"). Required for new corpus
        creation unless supplied through ``global_metadata``.
    """
    corpus_root = corpus_index_path.parent.resolve()

    # Convert manifest_path to relative to corpus_root for portability.
    manifest_path_input = Path(new_dataset_record.manifest_path)
    if manifest_path_input.is_absolute():
        manifest_path_relative = manifest_path_input.resolve().relative_to(corpus_root)
    else:
        manifest_path_relative = manifest_path_input

    if corpus_index_path.exists():
        corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)
        existing_ids = {d.dataset_id for d in corpus.datasets}
        if new_dataset_record.dataset_id in existing_ids:
            raise ValueError(
                f"dataset {new_dataset_record.dataset_id} already exists in corpus index; "
                "use a different dataset_id to avoid duplication."
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
            join_mode=new_dataset_record.join_mode,
            manifest_path=str(manifest_path_relative),
            cell_count=new_dataset_record.cell_count,
            global_start=0,
            global_end=new_dataset_record.cell_count,
        )
        datasets = [new_record]
        corpus_id = "perturb-data-lab-v0"
        # Build global metadata dict for new corpus
        if global_metadata is None:
            if backend is None:
                raise ValueError("backend is required when creating a corpus index")
            if topology is None:
                raise ValueError("topology is required when creating a corpus index")
            assert backend in {"lance", "zarr"}, f"unknown backend: {backend}"
            assert topology in {"federated", "aggregate"}, f"unknown topology: {topology}"
        gmeta_dict: dict[str, Any] = {
            "kind": "global-metadata",
            "contract_version": CONTRACT_VERSION,
            "schema_version": CONTRACT_VERSION,
            "missing_value_literal": MISSING_VALUE_LITERAL,
            "raw_field_policy": "preserve-unchanged",
            "backend": backend,
            "topology": topology,
        }
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

    return updated
