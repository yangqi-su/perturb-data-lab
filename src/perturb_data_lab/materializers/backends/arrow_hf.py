"""Backend adapter: Arrow + Hugging Face datasets with sparse per-cell storage.

Refactored (Phase 1) to use flat-buffer Arrow list arrays built directly from
CSR indptr/indices/data buffers, avoiding per-row Python nested-list assembly
on the sparse hot path. Size factors are fused into the same write traversal.
SQLite cell metadata output has been removed (Phase 2); the sole metadata
artifact is the raw-obs Parquet sidecar written by Stage2Materializer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import issparse

from ...contracts import CONTRACT_VERSION


def _is_csr_dataset(x: object) -> bool:
    """Check if x is an anndata _CSRDataset (backed sparse)."""
    return x.__class__.__name__ == "_CSRDataset"


def _build_list_array(offsets: np.ndarray, values: np.ndarray) -> pa.Array:
    """Build a pa.ListArray from flat offsets and values arrays.

    This is the core pattern adapted from the archived benchmark implementation:
    instead of building per-row Python lists, we pass the CSR indptr (offsets)
    and flat value buffer directly to Arrow. This eliminates the hottest Python
    overhead in the write path.

    Parameters
    ----------
    offsets : np.ndarray
        Row-offset array (length n_rows + 1). Must be int32-castable for
        Arrow offset dtype. For CSR, this is ``indptr``.
    values : np.ndarray
        Flat value buffer. For CSR indices/data, this is the flat
        ``indices`` or ``data`` array.

    Returns
    -------
    pa.ListArray
        Arrow list array suitable for direct Parquet write.
    """
    return pa.ListArray.from_arrays(
        pa.array(offsets.astype(np.int32, copy=False), type=pa.int32()),
        pa.array(values, type=pa.int32()),
    )


def write_arrow_hf_sparse(
    adata: ad.AnnData,
    count_matrix: Any,
    size_factors: np.ndarray | None,
    release_id: str,
    matrix_root: Path,
    dataset_id: str = "",
) -> tuple[dict[str, Path], np.ndarray]:
    """Write sparse per-cell data in Arrow + Parquet format.

    The Arrow/HF backend stores each cell as a sparse struct:
    - expressed_gene_indices: LIST<INT32>
    - expression_counts: LIST<INT32>

    Size factors are NOT stored in the cells parquet. They are computed inline
    during the write traversal and returned to the caller, who writes them as a
    separate ``{release_id}-size-factor.parquet``.

    If ``size_factors`` is provided, those values are used directly. If None,
    raw row sums are accumulated during the write pass and normalized once at
    the end using the global median.

    The write uses the flat-buffer Arrow pattern: CSR chunk ``indptr``,
    flat ``indices``, and flat ``counts`` are used to build Arrow list arrays
    directly via ``pa.ListArray.from_arrays()``, avoiding per-row Python
    nested-list assembly. Row sums are computed from the same CSR data in the
    same traversal — no separate scan of the count matrix.

    Returns ``(paths_dict, size_factors_array)`` where size_factors_array
    contains the computed (median-normalized) per-cell size factors.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / f"{release_id}-cells.parquet"
    n_obs = adata.n_obs
    batch_size = 100000

    # Allocate raw sums array if not provided. Will be normalized at the end.
    if size_factors is None:
        raw_sums = np.zeros(n_obs, dtype=np.float64)
    else:
        # Pre-provided: treat as already-normalized raw sums for backward compat
        raw_sums = size_factors.copy()

    # Write Arrow tables incrementally — one batch at a time, no cross-batch buffer.
    writer = None
    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)

        if _is_csr_dataset(count_matrix) or issparse(count_matrix):
            batch_sliced = count_matrix[start:end].tocsr()
            batch_indptr = batch_sliced.indptr
            batch_data = np.asarray(batch_sliced.data, dtype=np.int32)
            batch_indices = batch_sliced.indices.astype(np.int32)
            batch_n_rows = end - start

            # Compute row sums vectorized from the CSR batch in the same traversal.
            batch_row_sums = np.asarray(batch_sliced.sum(axis=1)).ravel()
            raw_sums[start:start + batch_n_rows] = batch_row_sums

            # Build Arrow list arrays directly from CSR buffers.
            # The indptr is already the correct offset array for pa.ListArray:
            # indptr[i] is the start of row i's data, indptr[i+1] is the end.
            # For empty rows (indptr[i] == indptr[i+1]), the list is empty.
            indices_list_array = _build_list_array(batch_indptr, batch_indices)
            counts_list_array = _build_list_array(batch_indptr, batch_data)
        else:
            # Dense fallback: build row-wise, but vectorize where possible.
            batch_dense = np.asarray(count_matrix[start:end])
            batch_n_rows = end - start
            row_sums = np.asarray(batch_dense.sum(axis=1)).ravel()
            raw_sums[start:start + batch_n_rows] = row_sums

            # For dense, we still build lists per row since there's no CSR indptr.
            indices_offsets = np.zeros(batch_n_rows + 1, dtype=np.int32)
            counts_offsets = np.zeros(batch_n_rows + 1, dtype=np.int32)

            # Count total nonzeros to pre-allocate
            nonzero_masks = [batch_dense[i] != 0 for i in range(batch_n_rows)]
            total_nnz = sum(m.sum() for m in nonzero_masks)

            all_indices = np.empty(total_nnz, dtype=np.int32)
            all_counts = np.empty(total_nnz, dtype=np.int32)
            offset = 0
            for local_i in range(batch_n_rows):
                row = batch_dense[local_i]
                nz_mask = nonzero_masks[local_i]
                nnz = int(nz_mask.sum())
                if nnz > 0:
                    all_indices[offset:offset + nnz] = np.where(nz_mask)[0].astype(np.int32)
                    all_counts[offset:offset + nnz] = row[nz_mask].astype(np.int32)
                indices_offsets[local_i + 1] = offset + nnz
                counts_offsets[local_i + 1] = offset + nnz
                offset += nnz

            indices_list_array = _build_list_array(indices_offsets, all_indices)
            counts_list_array = _build_list_array(counts_offsets, all_counts)

        # Write this batch's Arrow table immediately — no cross-batch buffering.
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

    # Normalize once using the global median after all batches are processed.
    row_median = float(np.median(raw_sums))
    if row_median > 0:
        size_factors = raw_sums / row_median
    else:
        size_factors = raw_sums.copy()
    size_factors = np.where(size_factors <= 0, 1.0, size_factors)
    size_factors = np.where(np.isnan(size_factors), 1.0, size_factors)

    return (
        {"cells": cell_path},
        size_factors,
    )


def read_arrow_hf_sparse_cell(
    parquet_path: Path,
    cell_index: int,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from an Arrow parquet.

    Returns (expressed_gene_indices, expression_counts, size_factor).

    ``size_factor_path`` is the path to the separate size-factor parquet. If
    provided, size factors are read from there. If not provided, the function
    falls back to reading from the ``size_factor`` column in the cells parquet
    (for backward compatibility with pre-separate-parquet artifacts).
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
            f"size_factor not found in cells parquet and size_factor_path "
            f"not provided; artifact may predate the separate size-factor layout. "
            f"Provide size_factor_path to read size factors from the separate parquet."
        )

    return (tuple(indices), tuple(counts), sf)


# ---------------------------------------------------------------------------
# Legacy compatibility: helpers still used by other backends and legacy routes
# ---------------------------------------------------------------------------


def _get_row_nonzero(count_matrix: Any, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Get (indices, counts) for row i, handling _CSRDataset safely.

    .. deprecated::
        This per-row accessor is retained for backward compatibility with the
        LanceDB aggregated backend and legacy route HVG computation. The primary
        Arrow/HF write path now uses the flat-buffer pattern instead.
    """
    if _is_csr_dataset(count_matrix):
        sliced = count_matrix[[i]]
        coo = sliced.tocoo()
        indices = coo.col.astype(np.int32)
        counts = np.asarray(coo.data).astype(np.int32)
        return indices, counts
    elif issparse(count_matrix):
        sliced = count_matrix[[i]]
        coo = sliced.tocoo()
        indices = coo.col.astype(np.int32)
        counts = np.asarray(coo.data).astype(np.int32)
        return indices, counts
    else:
        row = np.asarray(count_matrix[i]).ravel()
        nonzero_mask = row != 0
        indices = np.where(nonzero_mask)[0].astype(np.int32)
        counts = row[nonzero_mask].astype(np.int32)
        return indices, counts


def write_arrow_hf_metadata_artifacts(
    adata: ad.AnnData,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
    dataset_id: str = "",
) -> dict[str, Path]:
    """Write Arrow/HF-compatible metadata sidecars (legacy, SQLite-including path).

    .. deprecated::
        This function writes a SQLite cell metadata file, which is deprecated
        for new Stage 2 artifacts. The Stage 2 materializer writes raw-obs
        Parquet and separate size-factor Parquet instead. This function is
        retained only for backward compatibility with the LanceDB aggregated
        backend and the legacy schema-first materialization route.
    """
    import json
    import sqlite3

    matrix_root.mkdir(parents=True, exist_ok=True)
    meta_path = matrix_root / f"{release_id}-meta.parquet"
    cell_meta_sqlite_path = matrix_root / f"{release_id}-cell-meta.sqlite"

    n_obs = adata.n_obs
    cell_ids = [str(adata.obs.index[i]) for i in range(n_obs)]
    meta_table = pa.table(
        {
            "cell_id": pa.array(cell_ids, type=pa.string()),
            "raw_obs": pa.array([""] * n_obs, type=pa.string()),
        }
    )
    pq.write_table(meta_table, meta_path)

    pert_tuple = canonical_perturbation or tuple([{}] * n_obs)
    ctx_tuple = canonical_context or tuple([{}] * n_obs)
    raw_tuple = raw_fields or tuple([{}] * n_obs)

    conn = sqlite3.connect(str(cell_meta_sqlite_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cell_meta "
        "(cell_id TEXT, dataset_id TEXT, dataset_release TEXT, "
        "size_factor REAL, canonical_perturbation TEXT, canonical_context TEXT, raw_obs TEXT)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cell_id ON cell_meta(cell_id)")

    batch_size_sqlite = 256
    for start in range(0, n_obs, batch_size_sqlite):
        end = min(start + batch_size_sqlite, n_obs)
        sql_batch = []
        for i in range(start, end):
            sql_batch.append((
                str(adata.obs.index[i]),
                dataset_id,
                release_id,
                float(size_factors[i]),
                json.dumps(dict(pert_tuple[i])),
                json.dumps(dict(ctx_tuple[i])),
                json.dumps(dict(raw_tuple[i])),
            ))
        conn.executemany(
            "INSERT INTO cell_meta VALUES (?, ?, ?, ?, ?, ?, ?)",
            sql_batch,
        )
    conn.commit()
    conn.close()

    return {
        "metadata": meta_path,
        "cell_meta_sqlite": cell_meta_sqlite_path,
    }