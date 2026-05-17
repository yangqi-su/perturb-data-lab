"""Dataset materialization orchestration.

The materializer reads an inspected ``dataset-summary.yaml``, streams the selected
count matrix into the configured backend, and writes the per-dataset metadata
sidecars needed by corpus loaders.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from .backends import build_backend_fn
from .metadata_writers import (
    write_feature_provenance_parquet,
    write_hvg_ranking_parquet,
    write_raw_cell_metadata_parquet,
    write_raw_feature_metadata_parquet,
    write_size_factor_parquet,
)
from .obs_filter import ObsFilterError, filter_obs_rows

from .models import (
    CountSourceSpec,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
)
from ..contracts import CONTRACT_VERSION
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


class DatasetMaterializer:
    """Materialize one inspected dataset into a corpus backend.

    Inputs
    ------
    source_path : str
        Absolute path to the source h5ad file.
    inspection_summary_path : str
        Path to the ``dataset-summary.yaml`` produced by inspection.
    output_roots : OutputRoots
        ``metadata_root`` and ``matrix_root`` for this dataset's outputs.
    dataset_id : str
        Stable dataset identifier.
    backend : str
        Storage backend: ``lance`` (default) or ``zarr``.
    topology : str
        Corpus topology: ``federated`` (default) or ``aggregate``.
    rerun_inspection : bool, default False
        If True, reruns inspection before materialization and uses the resulting
        ``dataset-summary.yaml``.
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

    Backend/topology separation
    ---------------------------
    ``backend`` names the storage format only. ``topology`` names the corpus
    organization only.
    """

    def __init__(
        self,
        source_path: str,
        inspection_summary_path: str,
        output_roots: OutputRoots,
        dataset_id: str,
        backend: str = "lance",
        topology: str = "federated",
        rerun_inspection: bool = False,
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
        self.inspection_summary_path = inspection_summary_path
        self.output_roots = output_roots
        self.dataset_id = dataset_id
        self.backend = backend
        self.topology = topology
        self.rerun_inspection = rerun_inspection
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
        """Run materialization gated by the inspection summary."""
        if self.rerun_inspection:
            summary_path = self._rerun_inspection_preflight()
        else:
            summary_path = Path(self.inspection_summary_path)
            if not summary_path.exists():
                raise FileNotFoundError(
                    f"inspection summary not found: {summary_path}; "
                    "pass rerun_inspection=True to run inspection as preflight"
                )

        summary = DatasetSummaryDocument.from_yaml_file(summary_path)

        if summary.materialization_readiness != "pass":
            raise ValueError(
                f"materialization_readiness is '{summary.materialization_readiness}' "
                f"(expected 'pass') for dataset {self.dataset_id}; "
                f"inspection summary: {summary_path}"
            )

        decision = summary.count_source_decision
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
            # inspection summary into meta_root/dataset-summary.yaml.
            summary_copy_path = meta_root / "dataset-summary.yaml"
            if summary_path.resolve() != summary_copy_path.resolve():
                shutil.copy2(summary_path, summary_copy_path)

            # --- Write raw cell metadata (Parquet, not SQLite) ---
            raw_cell_meta_parquet_path = write_raw_cell_metadata_parquet(
                obs=filtered_obs,
                source_row_indices=(retained_source_rows if filter_applied else None),
                meta_root=meta_root,
                dataset_id=self.dataset_id,
            )

            # --- Write raw feature metadata (Parquet) ---
            var_mem = var_ref.to_memory() if hasattr(var_ref, "to_memory") else var_ref
            raw_feature_meta_parquet_path = write_raw_feature_metadata_parquet(
                var_mem=var_mem,
                meta_root=meta_root,
            )

            # --- Write feature provenance Parquet ---
            provenance_spec_path = write_feature_provenance_parquet(
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
                    "materialized via DatasetMaterializer",
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

    def _rerun_inspection_preflight(self) -> Path:
        """Re-run inspection as preflight and return the summary path."""
        from ..inspectors.workflow import inspect_target, InspectionTarget

        summary_path = Path(self.inspection_summary_path)
        output_root = summary_path.parent.parent
        dataset_id = summary_path.parent.name

        source_h5ad = Path(self.source_path)
        source_release = self.dataset_id

        target = InspectionTarget(
            dataset_id=dataset_id,
            source_path=str(source_h5ad),
            source_release=source_release,
        )

        artifacts = inspect_target(target, Path(output_root))
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

        HVG accumulation and global size factor computation are done during the
        chunk loop; the canonical ``hvg.parquet`` artifact is finalized after the loop.
        """
        from .chunk_translation import _translate_chunk

        backend_fn = build_backend_fn(self.backend, self.topology)
        n_obs = count_matrix.shape[0]
        n_vars = count_matrix.shape[1]
        global_row_start = self.global_row_start if self.topology == "aggregate" else 0
        writer_state: dict | None = (
            self.writer_state if self.topology == "aggregate" else None
        )

        all_paths: dict[str, Path] = {}
        all_row_sums: list[np.ndarray] = []
        sum_log1p = np.zeros(n_vars, dtype=np.float64)
        sum_log1p_sq = np.zeros(n_vars, dtype=np.float64)
        n_cells_total = 0

        for chunk_start in range(0, n_obs, self.chunk_rows):
            chunk_end = min(chunk_start + self.chunk_rows, n_obs)
            is_last_dataset = (
                self._is_last_dataset if self.topology == "aggregate" else True
            )
            is_last_chunk = (chunk_end == n_obs) and is_last_dataset
            matrix_chunk = _slice_matrix_chunk_as_csr(count_matrix, chunk_start, chunk_end)
            bundle = _translate_chunk(
                dataset_id=self.dataset_id,
                global_row_start=global_row_start,
                matrix_chunk=matrix_chunk,
                chunk_start=chunk_start,
                needs_recovery=needs_recovery,
            )

            all_row_sums.append(bundle.row_sums)
            log1p_counts = np.log1p(bundle.counts.astype(np.float64))
            np.add.at(sum_log1p, bundle.indices, log1p_counts)
            np.add.at(sum_log1p_sq, bundle.indices, log1p_counts ** 2)
            n_cells_total += bundle.row_count

            paths, writer_state = backend_fn(
                bundle=bundle,
                dataset_id=self.dataset_id,
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=is_last_chunk,
            )
            all_paths.update(paths)

        if self.topology == "aggregate":
            self.writer_state = writer_state

        global_row_sums = np.concatenate(all_row_sums)
        global_median = float(np.median(global_row_sums))
        if global_median > 0:
            size_factors = global_row_sums / global_median
        else:
            size_factors = np.ones_like(global_row_sums)
        size_factors = np.where(size_factors <= 0, 1.0, size_factors)
        size_factors = np.where(np.isnan(size_factors), 1.0, size_factors)
        size_factors = size_factors.astype(np.float32)

        size_factor_parquet_path = write_size_factor_parquet(
            size_factors=size_factors,
            cell_ids=cell_ids,
            meta_root=Path(self.output_roots.metadata_root),
        )
        hvg_ranking_path = write_hvg_ranking_parquet(
            sum_log1p=sum_log1p,
            sum_log1p_sq=sum_log1p_sq,
            n_cells_total=n_cells_total,
            feature_ids=feature_ids,
            n_hvg=self.n_hvg,
            meta_root=Path(self.output_roots.metadata_root),
        )

        return (all_paths, size_factor_parquet_path, hvg_ranking_path)


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
