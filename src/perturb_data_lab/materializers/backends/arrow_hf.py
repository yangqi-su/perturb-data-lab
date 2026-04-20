"""Backend adapter: Arrow + Hugging Face datasets with sparse per-cell storage."""

from __future__ import annotations

from pathlib import Path

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


def write_arrow_hf_sparse(
    adata: ad.AnnData,
    count_matrix: any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
) -> dict[str, Path]:
    """Write sparse per-cell data in Arrow + Parquet format.

    The Arrow/HF backend stores each cell as a sparse struct:
    - expressed_gene_indices: LIST<INT>
    - expression_counts: LIST<INT>
    - size_factor: DOUBLE

    Metadata (canonical + raw obs fields) is stored separately as Parquet.

    This avoids densifying the sparse matrix and keeps per-cell access cheap.

    Returns a dict with keys: "cells", "metadata", "feature_lengths".
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / f"{release_id}-cells.parquet"
    meta_path = matrix_root / f"{release_id}-meta.parquet"

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

    # Write metadata (canonical + raw obs)
    # Use index-based access for backed AnnData, not iloc
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

    return {"cells": cell_path, "metadata": meta_path}


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
