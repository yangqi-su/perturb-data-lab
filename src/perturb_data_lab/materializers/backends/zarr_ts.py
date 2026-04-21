"""Backend adapter: Zarr/TensorStore with cell-chunked sparse components."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from ..models import OutputRoots


def write_zarr_sparse_cell_chunks(
    adata: ad.AnnData,
    count_matrix: any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    chunk_cells: int = 1024,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Path]:
    """Write sparse per-cell data in Zarr format with cell-chunked storage.

    Zarr layout:
    - <release_id>-indices.zarr: (n_cells, chunk_cells) sparse indices as int32
    - <release_id>-counts.zarr:   (n_cells, chunk_cells) sparse counts as int32
    - <release_id>-sf.zarr:       (n_cells,) size factors as float64
    - <release_id>-meta.json:    cell metadata + canonical fields

    Canonical metadata (canonical_perturbation, canonical_context, raw_fields)
    is written to the meta.json so the Zarr reader can return full CellState
    parity with Arrow/HF backend.

    This format supports semi-random access at cell-chunk granularity while
    keeping sparse representation compact.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)

    n_obs = adata.n_obs
    n_vars = adata.n_vars

    try:
        import zarr
    except ImportError:
        raise ImportError(
            "zarr is required for Zarr/TensorStore backend; "
            "install with: pip install zarr"
        )

    indices_zarr_path = matrix_root / f"{release_id}-indices.zarr"
    counts_zarr_path = matrix_root / f"{release_id}-counts.zarr"
    sf_zarr_path = matrix_root / f"{release_id}-sf.zarr"

    indices_store = zarr.open(str(indices_zarr_path), mode="w")
    counts_store = zarr.open(str(counts_zarr_path), mode="w")
    sf_store = zarr.open(str(sf_zarr_path), mode="w")

    n_chunks = (n_obs + chunk_cells - 1) // chunk_cells
    max_nonzero_per_cell = n_vars  # worst-case

    # Normalize metadata tuples
    pert_tuple = canonical_perturbation or tuple([{}] * n_obs)
    ctx_tuple = canonical_context or tuple([{}] * n_obs)
    raw_tuple = raw_fields or tuple([{}] * n_obs)

    # Write size factors as a simple array
    sf_store.create_dataset("sf", data=size_factors, shape=(n_obs,), dtype="f8")

    # Write sparse chunks as padded arrays
    indices_arr = np.full(
        (n_chunks, chunk_cells, max_nonzero_per_cell), -1, dtype=np.int32
    )
    counts_arr = np.zeros((n_chunks, chunk_cells, max_nonzero_per_cell), dtype=np.int32)
    nonzero_counts = np.zeros((n_chunks, chunk_cells), dtype=np.int32)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_cells
        end = min(start + chunk_cells, n_obs)
        local_cells = end - start

        for local_i, global_i in enumerate(range(start, end)):
            if issparse(count_matrix):
                row = count_matrix[global_i]
                if hasattr(row, "toarray"):
                    row = row.toarray().ravel()
                else:
                    row = np.asarray(row).ravel()
            else:
                row = np.asarray(count_matrix[global_i]).ravel()

            nonzero_mask = row != 0
            indices = np.where(nonzero_mask)[0]
            nz_counts = row[nonzero_mask].astype(np.int32)

            count = len(indices)
            nonzero_counts[chunk_idx, local_i] = count
            indices_arr[chunk_idx, local_i, :count] = indices
            counts_arr[chunk_idx, local_i, :count] = nz_counts

        indices_store.create_dataset(
            f"chunk_{chunk_idx}",
            data=indices_arr[chunk_idx, :local_cells],
            shape=(local_cells, max_nonzero_per_cell),
            dtype="i4",
        )
        counts_store.create_dataset(
            f"chunk_{chunk_idx}",
            data=counts_arr[chunk_idx, :local_cells],
            shape=(local_cells, max_nonzero_per_cell),
            dtype="i4",
        )

    # Write metadata as JSON including canonical fields
    meta_path = matrix_root / f"{release_id}-meta.json"
    import json

    # Build per-cell metadata list for JSON serialization
    cell_meta_list = []
    for i in range(n_obs):
        cell_meta_list.append({
            "cell_id": str(adata.obs.index[i]),
            "canonical_perturbation": dict(pert_tuple[i]),
            "canonical_context": dict(ctx_tuple[i]),
            "raw_fields": dict(raw_tuple[i]),
        })

    with open(meta_path, "w") as f:
        json.dump(
            {
                "release_id": release_id,
                "n_obs": n_obs,
                "n_vars": n_vars,
                "chunk_cells": chunk_cells,
                "n_chunks": n_chunks,
                "size_factor_path": str(sf_zarr_path),
                "indices_path": str(indices_zarr_path),
                "counts_path": str(counts_zarr_path),
                "cells": cell_meta_list,
            },
            f,
            indent=2,
        )

    return {
        "indices": indices_zarr_path,
        "counts": counts_zarr_path,
        "size_factors": sf_zarr_path,
        "meta": meta_path,
    }
