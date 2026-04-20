"""Materialization route implementations using backend adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .arrow_hf import write_arrow_hf_sparse
from .webdataset import write_webdataset_shards
from .zarr_ts import write_zarr_sparse_cell_chunks
from ..models import OutputRoots


def materialize_arrow_hf(
    adata: Any,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
) -> dict[str, Path]:
    """Write using Arrow/HF backend (primary, hardened first)."""
    return write_arrow_hf_sparse(
        adata, count_matrix, size_factors, release_id, matrix_root
    )


def materialize_webdataset(
    adata: Any,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    shard_size: int = 10000,
) -> dict[str, Path]:
    """Write using WebDataset shard format."""
    return write_webdataset_shards(
        adata,
        count_matrix,
        size_factors,
        release_id,
        matrix_root,
        shard_size=shard_size,
    )


def materialize_zarr(
    adata: Any,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    chunk_cells: int = 1024,
) -> dict[str, Path]:
    """Write using Zarr/TensorStore cell-chunked format."""
    return write_zarr_sparse_cell_chunks(
        adata,
        count_matrix,
        size_factors,
        release_id,
        matrix_root,
        chunk_cells=chunk_cells,
    )


# Backend registry for explicit selection
AVAILABLE_BACKENDS = {
    "arrow-hf": materialize_arrow_hf,
    "webdataset": materialize_webdataset,
    "zarr-ts": materialize_zarr,
}


def build_backend_fn(backend: str):
    """Select backend materialization function by name."""
    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"unknown backend: {backend}; available: {list(AVAILABLE_BACKENDS)}"
        )
    return AVAILABLE_BACKENDS[backend]
