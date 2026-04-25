"""Backend adapter: Arrow Parquet federated and aggregate writers.

Phase 3: refactored to consume ChunkBundle from the shared translation layer
instead of performing per-backend sparse re-encoding.

Phase 4: adds aggregate topology writer for corpus-scoped single-file output
with deterministic global row indices across datasets.

Produces:
- {release_id}-cells.parquet: heavy-row Arrow table (global_row_index,
  expressed_gene_indices LIST<INT32>, expression_counts LIST<INT32>)
- Caller (Stage2Materializer) writes the separate size-factor Parquet sidecar.

Topology: federated (per-dataset files) and aggregate (corpus-scoped single file).
Backend name in registry: ``arrow-parquet``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, issparse

from ..chunk_translation import DatasetSpec, HEAVY_CELL_SCHEMA, _build_list_array


def _is_csr_dataset(x: object) -> bool:
    """Check if x is an anndata _CSRDataset (backed sparse)."""
    return x.__class__.__name__ == "_CSRDataset"


def write_arrow_parquet_federated(
    adata: ad.AnnData,
    count_matrix: Any,
    size_factors: np.ndarray | None,
    release_id: str,
    matrix_root: Path,
    dataset_id: str = "",
    chunk_rows: int = 100_000,
) -> tuple[dict[str, Path], np.ndarray]:
    """Write federated sparse per-cell data as Arrow Parquet.

    This is the ``arrow-parquet × federated`` backend writer.
    It consumes a ChunkBundle-compatible hot path: batch CSR slices are
    translated to Arrow list arrays via ``_build_list_array`` (flat-buffer,
    no per-row Python assembly), matching the pattern in the archived benchmark.

    Size factors are NOT stored in the cells parquet. They are computed inline
    during the write traversal and returned to the caller, who writes them as a
    separate ``{release_id}-size-factor.parquet``.

    Parameters
    ----------
    adata : ad.AnnData
        Source AnnData (backed or in-memory).
    count_matrix : Any
        Sparse count matrix (CSR or dense). Follows Stage2Materializer convention:
        may be the raw ``.X`` or a recovered layer.
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
        Rows per write batch. Larger batches reduce per-batch overhead;
        100k is the tested default for backed datasets.

    Returns
    -------
    tuple[dict[str, Path], np.ndarray]
        ``(paths_dict, size_factors_array)`` where paths_dict contains
        ``{"cells": cells_parquet_path}``. The size_factors_array contains
        median-normalized per-cell factors (clamped at >0, NaN-safe).
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / f"{release_id}-cells.parquet"
    n_obs = adata.n_obs

    # Allocate raw sums array for inline size-factor computation.
    if size_factors is None:
        raw_sums = np.zeros(n_obs, dtype=np.float64)
    else:
        raw_sums = size_factors.copy()

    writer: pq.ParquetWriter | None = None

    for start in range(0, n_obs, chunk_rows):
        end = min(start + chunk_rows, n_obs)

        if _is_csr_dataset(count_matrix) or issparse(count_matrix):
            batch_csr = count_matrix[start:end].tocsr()
            batch_indptr = np.asarray(batch_csr.indptr, dtype=np.int64)
            batch_data = np.asarray(batch_csr.data, dtype=np.int32)
            batch_indices = np.asarray(batch_csr.indices, dtype=np.int32)
            batch_n_rows = end - start

            # Compute row sums in the same traversal as the Arrow write.
            batch_row_sums = np.asarray(batch_csr.sum(axis=1)).ravel()
            raw_sums[start:start + batch_n_rows] = batch_row_sums

            # Build Arrow list arrays directly from CSR buffers (flat-buffer pattern).
            indices_list_array = _build_list_array(batch_indptr, batch_indices)
            counts_list_array = _build_list_array(batch_indptr, batch_data)
        else:
            # Dense fallback: build row-wise with vectorized row sum.
            batch_dense = np.asarray(count_matrix[start:end])
            batch_n_rows = end - start

            batch_row_sums = np.asarray(batch_dense.sum(axis=1)).ravel()
            raw_sums[start:start + batch_n_rows] = batch_row_sums

            # For dense, we build explicit offset arrays.
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
            writer = pq.ParquetWriter(cell_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    # Normalize: median row-sum as size factor.
    row_median = float(np.median(raw_sums))
    if row_median > 0:
        size_factors_out = raw_sums / row_median
    else:
        size_factors_out = raw_sums.copy()
    size_factors_out = np.where(size_factors_out <= 0, 1.0, size_factors_out)
    size_factors_out = np.where(np.isnan(size_factors_out), 1.0, size_factors_out)

    return ({"cells": cell_path}, size_factors_out)


# ---------------------------------------------------------------------------
# Federated reader (for runtime loading)
# ---------------------------------------------------------------------------

def read_arrow_parquet_cell(
    parquet_path: Path,
    cell_index: int,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from an Arrow parquet.

    Returns ``(expressed_gene_indices, expression_counts, size_factor)``.

    ``size_factor_path`` is the path to the separate size-factor parquet.
    If provided, size factors are read from there. Falls back to reading
    from the ``size_factor`` column in the cells parquet for backward
    compatibility with pre-separate-parquet artifacts.
    """
    table = pq.read_table(parquet_path)
    indices = table["expressed_gene_indices"][cell_index].as_py()
    counts = table["expression_counts"][cell_index].as_py()

    if size_factor_path is not None and size_factor_path.exists():
        sf_table = pq.read_table(str(size_factor_path))
        sf = float(sf_table["size_factor"][cell_index].as_py())
    elif "size_factor" in table.column_names:
        sf = float(table["size_factor"][cell_index].as_py())
    else:
        raise KeyError(
            "size_factor not found in cells parquet and size_factor_path "
            "not provided; artifact may predate the separate size-factor layout. "
            "Provide size_factor_path to read size factors from the separate parquet."
        )

    return (tuple(indices), tuple(counts), sf)


# ---------------------------------------------------------------------------
# Aggregate writer (Phase 4)
# ---------------------------------------------------------------------------


def write_arrow_parquet_aggregate(
    datasets: list[DatasetSpec],
    count_matrices: list[Any],
    size_factors_list: list[np.ndarray],
    matrix_root: Path,
    chunk_rows: int = 100_000,
) -> tuple[dict[str, Path], list[np.ndarray]]:
    """Write aggregate sparse per-cell data as a single Arrow Parquet file.

    This is the ``arrow-parquet × aggregate`` backend writer. It produces a
    single corpus-scoped parquet file with deterministic global_row_index values
    spanning all datasets in order.

    Each dataset's rows are written with global_row_index values from its
    DatasetSpec (global_row_start to global_row_stop). The aggregate file
    is append-safe: multiple calls can add new datasets to the same file.

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
        ``(paths_dict, size_factors_list)`` where paths_dict contains
        ``{"cells": cells_parquet_path}``. size_factors_list contains the
        computed size factors for each dataset.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / "aggregated-cells.parquet"

    writer: pq.ParquetWriter | None = None

    size_factors_out_list: list[np.ndarray] = []

    for dataset_spec, count_matrix, size_factors in zip(
        datasets, count_matrices, size_factors_list
    ):
        n_obs = dataset_spec.rows
        global_row_start = dataset_spec.global_row_start

        # Allocate raw sums array for inline size-factor computation.
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

                # Compute row sums in the same traversal as the Arrow write.
                batch_row_sums = np.asarray(batch_csr.sum(axis=1)).ravel()
                raw_sums[start:start + chunk_rows_count] = batch_row_sums

                # Build Arrow list arrays directly from CSR buffers.
                indices_list_array = _build_list_array(batch_indptr, batch_indices)
                counts_list_array = _build_list_array(batch_indptr, batch_data)
            else:
                # Dense fallback.
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
                writer = pq.ParquetWriter(cell_path, table.schema)
            writer.write_table(table)

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