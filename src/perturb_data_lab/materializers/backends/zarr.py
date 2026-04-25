"""Backend adapter: Zarr federated and aggregate writers.

Phase 3: refactored to consume the shared flat-buffer Arrow list-array
pattern from the translation layer for heavy-row construction, while
preserving Zarr's chunked 2D array layout (indices, counts per cell-chunk)
as the native Zarr storage format.

Phase 4: adds aggregate topology writer for corpus-scoped single-file output.

Zarr layout (federated):
- {release_id}-indices.zarr: (n_cells, chunk_cells) padded sparse indices (int32)
- {release_id}-counts.zarr:   (n_cells, chunk_cells) padded sparse counts (int32)
- {release_id}-meta.json:    cell metadata + size_factor path reference

The indices/counts arrays use a global 1D flat-buffer approach: all rows
are concatenated and stored as 1D zarr arrays, with a separate row_offsets
array to delineate boundaries. This avoids the padded 2D cell-chunk layout
of the Phase 1 Zarr writer and aligns with the archived benchmark's pattern.

Zarr layout (aggregate):
- aggregated-indices.zarr: 1D flat buffer of all gene indices across datasets
- aggregated-counts.zarr: 1D flat buffer of all counts across datasets
- aggregated-row-offsets.zarr: row offsets including dataset boundaries
- aggregated-meta.json: corpus-level metadata with dataset offsets

Topology: federated (per-dataset files) and aggregate (corpus-scoped single file).
Backend name in registry: ``zarr``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from ..chunk_translation import DatasetSpec, _build_list_array


def _is_csr_dataset(x: object) -> bool:
    """Check if x is an anndata _CSRDataset (backed sparse)."""
    return x.__class__.__name__ == "_CSRDataset"


def write_zarr_federated(
    adata: ad.AnnData,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    chunk_cells: int = 1024,
    dataset_id: str = "",
) -> dict[str, Path]:
    """Write federated sparse per-cell data in Zarr format.

    Uses a flat-buffer 1D layout (matching archived benchmark):
    - all gene indices concatenated into one flat array
    - all counts concatenated into one flat array
    - row_offsets delineates cell boundaries

    Returns a dict with keys: ``{"indices": ..., "counts": ..., "meta": ...}``.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)

    try:
        import zarr
    except ImportError:
        raise ImportError(
            "zarr is required for Zarr backend; "
            "install with: pip install zarr"
        )

    n_obs = adata.n_obs

    indices_path = matrix_root / f"{release_id}-indices.zarr"
    counts_path = matrix_root / f"{release_id}-counts.zarr"

    # Collect all row data into flat 1D arrays.
    # We pre-compute total_nnz by scanning once (batched).
    total_nnz = 0
    batch_size = 50_000
    row_offsets = [0]
    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)
        batch_csr = count_matrix[start:end].tocsr()
        for i in range(end - start):
            total_nnz += batch_csr.indptr[i + 1] - batch_csr.indptr[i]
            row_offsets.append(row_offsets[-1] + batch_csr.indptr[i + 1] - batch_csr.indptr[i])

    row_offsets_arr = np.array(row_offsets, dtype=np.int64)

    # Pre-allocate flat arrays and fill them.
    all_indices = np.empty(total_nnz, dtype=np.int32)
    all_counts = np.zeros(total_nnz, dtype=np.int32)

    offset = 0
    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)
        batch_csr = count_matrix[start:end].tocsr()
        for i in range(end - start):
            nnz = batch_csr.indptr[i + 1] - batch_csr.indptr[i]
            if nnz > 0:
                all_indices[offset:offset + nnz] = batch_csr.indices[
                    batch_csr.indptr[i] : batch_csr.indptr[i + 1]
                ].astype(np.int32)
                all_counts[offset:offset + nnz] = batch_csr.data[
                    batch_csr.indptr[i] : batch_csr.indptr[i + 1]
                ].astype(np.int32)
            offset += nnz

    # Write as zarr 1D arrays.
    indices_zarr = zarr.open(str(indices_path), mode="w")
    counts_zarr = zarr.open(str(counts_path), mode="w")

    indices_zarr.create_dataset("indices", data=all_indices, dtype="i4")
    counts_zarr.create_dataset("counts", data=all_counts, dtype="i4")
    indices_zarr.create_dataset("row_offsets", data=row_offsets_arr, dtype="i8")

    # Write metadata JSON.
    import json

    meta_path = matrix_root / f"{release_id}-meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "release_id": release_id,
                "n_obs": n_obs,
                "size_factor_path": str(matrix_root / f"{release_id}-size-factor.zarr"),
                "indices_path": str(indices_path),
                "counts_path": str(counts_path),
            },
            f,
            indent=2,
        )

    return {
        "indices": indices_path,
        "counts": counts_path,
        "meta": meta_path,
    }


def read_zarr_cell(
    indices_path: Path,
    counts_path: Path,
    row_index: int,
    row_offsets_path: Path | None = None,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from Zarr storage.

    Returns ``(expressed_gene_indices, expression_counts, size_factor)``.
    """
    import zarr

    indices_group = zarr.open(str(indices_path), mode="r")
    counts_group = zarr.open(str(counts_path), mode="r")

    if row_offsets_path is not None:
        row_offsets = zarr.open(str(row_offsets_path), mode="r")["row_offsets"][:]
    else:
        # Fallback: use the row_offsets stored inside the indices zarr group
        row_offsets = indices_group["row_offsets"][:]

    start = int(row_offsets[row_index])
    stop = int(row_offsets[row_index + 1])

    gene_indices = tuple(indices_group["indices"][start:stop].astype(np.int32).tolist())
    expr_counts = tuple(counts_group["counts"][start:stop].astype(np.int32).tolist())

    sf = 1.0
    if size_factor_path is not None:
        import pyarrow.parquet as pq

        sf_table = pq.read_table(str(size_factor_path))
        sf = float(sf_table["size_factor"][row_index].as_py())

    return (gene_indices, expr_counts, sf)


# ---------------------------------------------------------------------------
# Aggregate writer (Phase 4)
# ---------------------------------------------------------------------------


def write_zarr_aggregate(
    datasets: list[DatasetSpec],
    count_matrices: list[Any],
    size_factors_list: list[np.ndarray],
    matrix_root: Path,
) -> tuple[dict[str, Path], list[np.ndarray]]:
    """Write aggregate sparse per-cell data in Zarr format.

    This is the ``zarr × aggregate`` backend writer. It produces a single
    corpus-scoped Zarr store with deterministic global row indices spanning
    all datasets in order.

    Zarr aggregate layout:
    - aggregated-indices.zarr: 1D flat buffer of all gene indices
    - aggregated-counts.zarr: 1D flat buffer of all counts
    - aggregated-row-offsets.zarr: row offsets (cumulative) across all datasets
    - aggregated-meta.json: corpus-level metadata with per-dataset offsets

    Parameters
    ----------
    datasets : list[DatasetSpec]
        Dataset specifications in order.
    count_matrices : list[Any]
        Sparse count matrices (CSR or dense), one per dataset.
    size_factors_list : list[np.ndarray]
        Pre-computed size factors for each dataset.
    matrix_root : Path
        Output directory.

    Returns
    -------
    tuple[dict[str, Path], list[np.ndarray]]
        ``(paths_dict, size_factors_out_list)``.
    """
    import json
    import zarr

    matrix_root.mkdir(parents=True, exist_ok=True)

    # Pre-scan all datasets to compute total sizes.
    total_rows = sum(ds.rows for ds in datasets)
    total_nnz = 0
    dataset_row_offsets = []  # cumulative row counts at each dataset boundary

    for ds in datasets:
        dataset_row_offsets.append(total_rows)
        for i in range(ds.rows):
            pass  # we need to count nnz per row

    # Actually count nnz per dataset
    nnz_per_dataset = []
    current_row_offset = 0
    for ds, cm in zip(datasets, count_matrices):
        ds_nnz = 0
        for start in range(0, ds.rows, 10000):
            end = min(start + 10000, ds.rows)
            batch_csr = cm[start:end].tocsr()
            for i in range(end - start):
                ds_nnz += batch_csr.indptr[i + 1] - batch_csr.indptr[i]
        nnz_per_dataset.append(ds_nnz)
        current_row_offset += ds.rows

    total_nnz = sum(nnz_per_dataset)

    # Build flat arrays and row_offsets across all datasets.
    all_indices = np.empty(total_nnz, dtype=np.int32)
    all_counts = np.zeros(total_nnz, dtype=np.int32)
    row_offsets = [0]
    global_row_offset = 0

    for ds, cm, nnz in zip(datasets, count_matrices, nnz_per_dataset):
        for start in range(0, ds.rows, 10000):
            end = min(start + 10000, ds.rows)
            batch_csr = cm[start:end].tocsr()
            for i in range(end - start):
                nnz_row = batch_csr.indptr[i + 1] - batch_csr.indptr[i]
                if nnz_row > 0:
                    all_indices[global_row_offset:global_row_offset + nnz_row] = (
                        batch_csr.indices[batch_csr.indptr[i]:batch_csr.indptr[i + 1]]
                    ).astype(np.int32)
                    all_counts[global_row_offset:global_row_offset + nnz_row] = (
                        batch_csr.data[batch_csr.indptr[i]:batch_csr.indptr[i + 1]]
                    ).astype(np.int32)
                global_row_offset += nnz_row
                row_offsets.append(global_row_offset)

    row_offsets_arr = np.array(row_offsets, dtype=np.int64)

    # Write Zarr arrays.
    indices_path = matrix_root / "aggregated-indices.zarr"
    counts_path = matrix_root / "aggregated-counts.zarr"
    row_offsets_path = matrix_root / "aggregated-row-offsets.zarr"

    indices_zarr = zarr.open(str(indices_path), mode="w")
    counts_zarr = zarr.open(str(counts_path), mode="w")
    row_offsets_zarr = zarr.open(str(row_offsets_path), mode="w")

    indices_zarr.create_dataset("indices", data=all_indices, dtype="i4")
    counts_zarr.create_dataset("counts", data=all_counts, dtype="i4")
    row_offsets_zarr.create_dataset("row_offsets", data=row_offsets_arr, dtype="i8")

    # Write corpus-level metadata JSON.
    meta_path = matrix_root / "aggregated-meta.json"
    dataset_offsets = []
    cum_rows = 0
    for ds in datasets:
        dataset_offsets.append({
            "dataset_id": ds.dataset_id,
            "dataset_index": int(ds.dataset_index),
            "global_row_start": int(ds.global_row_start),
            "global_row_stop": int(ds.global_row_stop),
            "rows": int(ds.rows),
        })
        cum_rows += ds.rows

    with open(meta_path, "w") as f:
        json.dump(
            {
                "total_rows": int(total_rows),
                "total_nnz": int(total_nnz),
                "indices_path": str(indices_path),
                "counts_path": str(counts_path),
                "row_offsets_path": str(row_offsets_path),
                "datasets": dataset_offsets,
            },
            f,
            indent=2,
        )

    # Compute normalized size factors per dataset.
    size_factors_out_list: list[np.ndarray] = []
    for ds, size_factors in zip(datasets, size_factors_list):
        if size_factors is None:
            raw_sums = np.ones(ds.rows, dtype=np.float64)
        else:
            raw_sums = size_factors.copy()

        row_median = float(np.median(raw_sums))
        if row_median > 0:
            sf_norm = raw_sums / row_median
        else:
            sf_norm = raw_sums.copy()
        sf_norm = np.where(sf_norm <= 0, 1.0, sf_norm)
        sf_norm = np.where(np.isnan(sf_norm), 1.0, sf_norm)
        size_factors_out_list.append(sf_norm)

    return (
        {
            "indices": indices_path,
            "counts": counts_path,
            "row_offsets": row_offsets_path,
            "meta": meta_path,
        },
        size_factors_out_list,
    )