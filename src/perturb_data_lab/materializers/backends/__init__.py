"""Materialization route implementations using backend adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .arrow_hf import write_arrow_hf_sparse
from .webdataset import write_webdataset_shards
from .zarr_ts import write_zarr_sparse_cell_chunks
from ..models import OutputRoots


def materialize_lancedb_aggregated(
    adata: Any,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
    dataset_id: str = "",
) -> dict[str, Path]:
    """Phase 5 bounded alias while true Lance materialization is pending.

    The Phase 2 contract locks `lancedb-aggregated` as a distinct backend name,
    but the heavy-row semantics needed for Phase 5 smoke are currently the same
    mixed-dataset sparse-row contract already exercised through Arrow/HF.
    Reuse the Arrow/HF writer as the temporary aggregated smoke adapter so the
    backend can be registered and validated without broadening scope beyond Phase 5.
    """
    return write_arrow_hf_sparse(
        adata,
        count_matrix,
        size_factors,
        release_id,
        matrix_root,
        canonical_perturbation=canonical_perturbation,
        canonical_context=canonical_context,
        raw_fields=raw_fields,
        dataset_id=dataset_id,
    )


def materialize_zarr_aggregated(
    adata: Any,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    chunk_cells: int = 1024,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
    dataset_id: str = "",
) -> dict[str, Path]:
    """Phase 5 bounded alias while true aggregated Zarr append layout is pending."""
    return write_zarr_sparse_cell_chunks(
        adata,
        count_matrix,
        size_factors,
        release_id,
        matrix_root,
        chunk_cells=chunk_cells,
        canonical_perturbation=canonical_perturbation,
        canonical_context=canonical_context,
        raw_fields=raw_fields,
        dataset_id=dataset_id,
    )


def materialize_arrow_hf(
    adata: Any,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
    dataset_id: str = "",
) -> dict[str, Path]:
    """Write using Arrow/HF backend (primary, hardened first)."""
    return write_arrow_hf_sparse(
        adata, count_matrix, size_factors, release_id, matrix_root,
        canonical_perturbation=canonical_perturbation,
        canonical_context=canonical_context,
        raw_fields=raw_fields,
        dataset_id=dataset_id,
    )


def materialize_webdataset(
    adata: Any,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    shard_size: int = 10000,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Path]:
    """Write using WebDataset shard format with canonical metadata parity."""
    return write_webdataset_shards(
        adata,
        count_matrix,
        size_factors,
        release_id,
        matrix_root,
        shard_size=shard_size,
        canonical_perturbation=canonical_perturbation,
        canonical_context=canonical_context,
        raw_fields=raw_fields,
    )


def materialize_zarr(
    adata: Any,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    chunk_cells: int = 1024,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Path]:
    """Write using Zarr/TensorStore cell-chunked format with canonical metadata parity."""
    return write_zarr_sparse_cell_chunks(
        adata,
        count_matrix,
        size_factors,
        release_id,
        matrix_root,
        chunk_cells=chunk_cells,
        canonical_perturbation=canonical_perturbation,
        canonical_context=canonical_context,
        raw_fields=raw_fields,
    )


# Backend registry for explicit selection
AVAILABLE_BACKENDS = {
    "arrow-hf": materialize_arrow_hf,
    "webdataset": materialize_webdataset,
    "zarr-ts": materialize_zarr,
    "lancedb-aggregated": materialize_lancedb_aggregated,
    "zarr-aggregated": materialize_zarr_aggregated,
}


def build_backend_fn(backend: str):
    """Select backend materialization function by name."""
    if backend not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"unknown backend: {backend}; available: {list(AVAILABLE_BACKENDS)}"
        )
    return AVAILABLE_BACKENDS[backend]
