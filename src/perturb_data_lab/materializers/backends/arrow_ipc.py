"""Backend adapter: Arrow IPC federated and aggregate writers.

Phase 3: refactored to consume the shared flat-buffer Arrow list-array
pattern from the translation layer instead of per-backend sparse re-encoding.

Phase 4: adds aggregate topology writer for corpus-scoped single-file output
with deterministic global row indices across datasets.

Produces:
- {release_id}-cells.arrow: heavy-row Arrow IPC file (global_row_index,
  expressed_gene_indices LIST<INT32>, expression_counts LIST<INT32>)
- Caller writes the separate size-factor Parquet sidecar.

Topology: federated (per-dataset files) and aggregate (corpus-scoped single file).
Backend name in registry: ``arrow-ipc``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import issparse

from ..chunk_translation import DatasetSpec, _build_list_array
import pyarrow.ipc as pa_ipc


def _is_csr_dataset(x: object) -> bool:
    """Check if x is an anndata _CSRDataset (backed sparse)."""
    return x.__class__.__name__ == "_CSRDataset"


def write_arrow_ipc_federated(
    adata: ad.AnnData,
    count_matrix: Any,
    size_factors: np.ndarray | None,
    release_id: str,
    matrix_root: Path,
    dataset_id: str = "",
    chunk_rows: int = 100_000,
) -> tuple[dict[str, Path], np.ndarray]:
    """Write federated sparse per-cell data as Arrow IPC format.

    This is the ``arrow-ipc × federated`` backend writer.
    Uses the same flat-buffer Arrow list-array pattern as arrow-parquet.

    Size factors are NOT stored in the IPC file. They are computed inline
    during the write traversal and returned to the caller, who writes them as a
    separate ``{release_id}-size-factor.parquet``.

    Parameters
    ----------
    adata : ad.AnnData
        Source AnnData (backed or in-memory).
    count_matrix : Any
        Sparse count matrix (CSR or dense).
    size_factors : np.ndarray | None
        Pre-computed size factors. If None, row sums are accumulated during
        the write pass and median-normalized at the end.
    release_id : str
        Release identifier used for output file naming.
    matrix_root : Path
        Output directory for matrix artifacts.
    dataset_id : str, optional
        Dataset identifier (used for metadata context only).
    chunk_rows : int, default 100_000
        Rows per write batch.

    Returns
    -------
    tuple[dict[str, Path], np.ndarray]
        ``(paths_dict, size_factors_array)`` where paths_dict contains
        ``{"cells": cells_arrow_path}``.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / f"{release_id}-cells.arrow"
    n_obs = adata.n_obs

    if size_factors is None:
        raw_sums = np.zeros(n_obs, dtype=np.float64)
    else:
        raw_sums = size_factors.copy()

    writer: pa.ipc.RecordBatchFileWriter | None = None

    for start in range(0, n_obs, chunk_rows):
        end = min(start + chunk_rows, n_obs)

        if _is_csr_dataset(count_matrix) or issparse(count_matrix):
            batch_csr = count_matrix[start:end].tocsr()
            batch_indptr = np.asarray(batch_csr.indptr, dtype=np.int64)
            batch_data = np.asarray(batch_csr.data, dtype=np.int32)
            batch_indices = np.asarray(batch_csr.indices, dtype=np.int32)
            batch_n_rows = end - start

            batch_row_sums = np.asarray(batch_csr.sum(axis=1)).ravel()
            raw_sums[start:start + batch_n_rows] = batch_row_sums

            indices_list_array = _build_list_array(batch_indptr, batch_indices)
            counts_list_array = _build_list_array(batch_indptr, batch_data)
        else:
            batch_dense = np.asarray(count_matrix[start:end])
            batch_n_rows = end - start

            batch_row_sums = np.asarray(batch_dense.sum(axis=1)).ravel()
            raw_sums[start:start + batch_n_rows] = batch_row_sums

            indices_offsets = np.zeros(batch_n_rows + 1, dtype=np.int32)
            counts_offsets = np.zeros(batch_n_rows + 1, dtype=np.int32)
            total_nnz = int((batch_dense != 0).sum())
            all_indices = np.empty(total_nnz, dtype=np.int32)
            all_counts = np.empty(total_nnz, dtype=np.int32)
            offset = 0
            for local_i in range(batch_n_rows):
                row = batch_dense[local_i]
                nz_mask = row != 0
                nnz = int(nz_mask.sum())
                if nnz > 0:
                    all_indices[offset:offset + nnz] = np.where(nz_mask)[0].astype(np.int32)
                    all_counts[offset:offset + nnz] = row[nz_mask].astype(np.int32)
                indices_offsets[local_i + 1] = offset + nnz
                counts_offsets[local_i + 1] = offset + nnz
                offset += nnz

            indices_list_array = _build_list_array(indices_offsets, all_indices)
            counts_list_array = _build_list_array(counts_offsets, all_counts)

        table = pa.table(
            {
                "expressed_gene_indices": indices_list_array,
                "expression_counts": counts_list_array,
            }
        )

        if writer is None:
            writer = pa.ipc.new_file(str(cell_path), table.schema)

        batch = table.to_batches()[0]
        writer.write_batch(batch)

    if writer is not None:
        writer.close()

    row_median = float(np.median(raw_sums))
    if row_median > 0:
        size_factors_out = raw_sums / row_median
    else:
        size_factors_out = raw_sums.copy()
    size_factors_out = np.where(size_factors_out <= 0, 1.0, size_factors_out)
    size_factors_out = np.where(np.isnan(size_factors_out), 1.0, size_factors_out)

    return ({"cells": cell_path}, size_factors_out)


def read_arrow_ipc_cell(
    arrow_path: Path,
    cell_index: int,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from an Arrow IPC file.

    Returns ``(expressed_gene_indices, expression_counts, size_factor)``.
    """
    with pa.memory_map(str(arrow_path), "r") as source:
        reader = pa_ipc.RecordBatchFileReader(source)
        running = 0
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            next_running = running + batch.num_rows
            if running <= cell_index < next_running:
                local_idx = cell_index - running
                indices = batch.column("expressed_gene_indices")[local_idx].as_py()
                counts = batch.column("expression_counts")[local_idx].as_py()
                break
            running = next_running
        else:
            raise IndexError(cell_index)

    if size_factor_path is not None and size_factor_path.exists():
        sf_table = pq.read_table(str(size_factor_path))
        sf = float(sf_table["size_factor"][cell_index].as_py())
    else:
        sf = 1.0

    return (tuple(indices), tuple(counts), sf)


# ---------------------------------------------------------------------------
# Aggregate writer (Phase 4)
# ---------------------------------------------------------------------------


def write_arrow_ipc_aggregate(
    datasets: list[DatasetSpec],
    count_matrices: list[Any],
    size_factors_list: list[np.ndarray],
    matrix_root: Path,
    chunk_rows: int = 100_000,
) -> tuple[dict[str, Path], list[np.ndarray]]:
    """Write aggregate sparse per-cell data as a single Arrow IPC file.

    This is the ``arrow-ipc × aggregate`` backend writer. It produces a
    single corpus-scoped IPC file with deterministic global_row_index values
    spanning all datasets in order.

    Parameters
    ----------
    datasets : list[DatasetSpec]
        Dataset specifications in the order they should appear in the corpus.
    count_matrices : list[Any]
        Sparse count matrices (CSR or dense), one per dataset.
    size_factors_list : list[np.ndarray]
        Pre-computed size factors for each dataset, or None for inline computation.
    matrix_root : Path
        Output directory for matrix artifacts.
    chunk_rows : int, default 100_000
        Rows per write batch.

    Returns
    -------
    tuple[dict[str, Path], list[np.ndarray]]
        ``(paths_dict, size_factors_out_list)`` where paths_dict contains
        ``{"cells": cells_arrow_path}``.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / "aggregated-cells.arrow"

    writer: pa.ipc.RecordBatchFileWriter | None = None
    size_factors_out_list: list[np.ndarray] = []

    for dataset_spec, count_matrix, size_factors in zip(
        datasets, count_matrices, size_factors_list
    ):
        n_obs = dataset_spec.rows
        global_row_start = dataset_spec.global_row_start

        if size_factors is None:
            raw_sums = np.zeros(n_obs, dtype=np.float64)
        else:
            raw_sums = size_factors.copy()

        for start in range(0, n_obs, chunk_rows):
            end = min(start + chunk_rows, n_obs)
            chunk_rows_count = end - start

            if _is_csr_dataset(count_matrix) or issparse(count_matrix):
                batch_csr = count_matrix[start:end].tocsr()
                batch_indptr = np.asarray(batch_csr.indptr, dtype=np.int64)
                batch_data = np.asarray(batch_csr.data, dtype=np.int32)
                batch_indices = np.asarray(batch_csr.indices, dtype=np.int32)

                batch_row_sums = np.asarray(batch_csr.sum(axis=1)).ravel()
                raw_sums[start:start + chunk_rows_count] = batch_row_sums

                indices_list_array = _build_list_array(batch_indptr, batch_indices)
                counts_list_array = _build_list_array(batch_indptr, batch_data)
            else:
                batch_dense = np.asarray(count_matrix[start:end])

                batch_row_sums = np.asarray(batch_dense.sum(axis=1)).ravel()
                raw_sums[start:start + chunk_rows_count] = batch_row_sums

                indices_offsets = np.zeros(chunk_rows_count + 1, dtype=np.int32)
                counts_offsets = np.zeros(chunk_rows_count + 1, dtype=np.int32)
                total_nnz = int((batch_dense != 0).sum())
                all_indices = np.empty(total_nnz, dtype=np.int32)
                all_counts = np.empty(total_nnz, dtype=np.int32)
                offset = 0
                for local_i in range(chunk_rows_count):
                    row = batch_dense[local_i]
                    nz_mask = row != 0
                    nnz = int(nz_mask.sum())
                    if nnz > 0:
                        all_indices[offset:offset + nnz] = np.where(nz_mask)[0].astype(np.int32)
                        all_counts[offset:offset + nnz] = row[nz_mask].astype(np.int32)
                    indices_offsets[local_i + 1] = offset + nnz
                    counts_offsets[local_i + 1] = offset + nnz
                    offset += nnz

                indices_list_array = _build_list_array(indices_offsets, all_indices)
                counts_list_array = _build_list_array(counts_offsets, all_counts)

            # Build table with global_row_index from DatasetSpec.
            global_row_indices = np.arange(
                global_row_start + start,
                global_row_start + end,
                dtype=np.int64,
            )
            table = pa.table(
                {
                    "global_row_index": pa.array(global_row_indices, type=pa.int64()),
                    "expressed_gene_indices": indices_list_array,
                    "expression_counts": counts_list_array,
                }
            )

            if writer is None:
                writer = pa.ipc.new_file(str(cell_path), table.schema)

            batch = table.to_batches()[0]
            writer.write_batch(batch)

        # Normalize size factors for this dataset.
        row_median = float(np.median(raw_sums))
        if row_median > 0:
            size_factors_norm = raw_sums / row_median
        else:
            size_factors_norm = raw_sums.copy()
        size_factors_norm = np.where(size_factors_norm <= 0, 1.0, size_factors_norm)
        size_factors_norm = np.where(np.isnan(size_factors_norm), 1.0, size_factors_norm)
        size_factors_out_list.append(size_factors_norm)

    if writer is not None:
        writer.close()

    return ({"cells": cell_path}, size_factors_out_list)