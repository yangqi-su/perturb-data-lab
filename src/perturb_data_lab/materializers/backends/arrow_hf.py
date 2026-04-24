"""Backend adapter: Arrow + Hugging Face datasets with sparse per-cell storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import issparse

from ...contracts import CONTRACT_VERSION
from ..models import OutputRoots


def _is_csr_dataset(x: object) -> bool:
    """Check if x is an anndata _CSRDataset (backed sparse)."""
    return x.__class__.__name__ == "_CSRDataset"


def _get_row_nonzero(count_matrix: Any, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Get (indices, counts) for row i, handling _CSRDataset safely."""
    if _is_csr_dataset(count_matrix):
        # _CSRDataset: slice to get CSR, then tocoo for safe nonzero extraction
        sliced = count_matrix[[i]]
        coo = sliced.tocoo()
        indices = coo.col.astype(np.int32)
        counts = np.asarray(coo.data).astype(np.int32)
        return indices, counts
    elif issparse(count_matrix):
        # Normal scipy sparse
        sliced = count_matrix[[i]]
        coo = sliced.tocoo()
        indices = coo.col.astype(np.int32)
        counts = np.asarray(coo.data).astype(np.int32)
        return indices, counts
    else:
        # Dense
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
    """Write Arrow/HF-compatible metadata sidecars without a heavy parquet payload."""
    matrix_root.mkdir(parents=True, exist_ok=True)
    meta_path = matrix_root / f"{release_id}-meta.parquet"
    cell_meta_sqlite_path = matrix_root / f"{release_id}-cell-meta.sqlite"

    n_obs = adata.n_obs
    cell_ids = [str(adata.obs.index[i]) for i in range(n_obs)]
    sf_vals = [float(size_factors[i]) for i in range(n_obs)]
    meta_table = pa.table(
        {
            "cell_id": pa.array(cell_ids, type=pa.string()),
            "size_factor": pa.array(sf_vals, type=pa.float64()),
            "raw_obs": pa.array([""] * n_obs, type=pa.string()),
        }
    )
    pq.write_table(meta_table, meta_path)

    import sqlite3

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


def write_arrow_hf_sparse(
    adata: ad.AnnData,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
    dataset_id: str = "",
) -> dict[str, Path]:
    """Write sparse per-cell data in Arrow + Parquet format.

    The Arrow/HF backend stores each cell as a sparse struct:
    - expressed_gene_indices: LIST<INT>
    - expression_counts: LIST<INT>
    - size_factor: DOUBLE

    Canonical cell metadata (canonical_perturbation, canonical_context, raw_fields)
    is stored in a SQLite file at ``{release_id}-cell-meta.sqlite`` alongside the
    parquet files. The ArrowHFCellReader reads this SQLite to provide full
    CellState at load time.

    This avoids densifying the sparse matrix and keeps per-cell access cheap.

    Returns a dict with keys: "cells", "metadata", "cell_meta_sqlite".
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / f"{release_id}-cells.parquet"
    n_obs = adata.n_obs
    batch_size = 256

    cell_chunks = []
    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)
        indices_list = []
        counts_list = []
        sf_list = []

        for i in range(start, end):
            indices, counts = _get_row_nonzero(count_matrix, i)
            indices_list.append(indices.tolist())
            counts_list.append(counts.tolist())
            sf_list.append(float(size_factors[i]))

        table = pa.table(
            {
                "expressed_gene_indices": pa.array(
                    indices_list, type=pa.list_(pa.int32())
                ),
                "expression_counts": pa.array(counts_list, type=pa.list_(pa.int32())),
                "size_factor": pa.array(sf_list, type=pa.float64()),
            }
        )
        cell_chunks.append(table)

    full_cells = pa.concat_tables(cell_chunks)
    pq.write_table(full_cells, cell_path)

    metadata_paths = write_arrow_hf_metadata_artifacts(
        adata=adata,
        size_factors=size_factors,
        release_id=release_id,
        matrix_root=matrix_root,
        canonical_perturbation=canonical_perturbation,
        canonical_context=canonical_context,
        raw_fields=raw_fields,
        dataset_id=dataset_id,
    )

    return {
        "cells": cell_path,
        **metadata_paths,
    }


def read_arrow_hf_sparse_cell(
    parquet_path: Path,
    cell_index: int,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from an Arrow parquet.

    Returns (expressed_gene_indices, expression_counts, size_factor).
    """
    table = pq.read_table(parquet_path)
    indices = table["expressed_gene_indices"][cell_index].as_py()
    counts = table["expression_counts"][cell_index].as_py()
    sf = table["size_factor"][cell_index].as_py()
    return (tuple(indices), tuple(counts), sf)
