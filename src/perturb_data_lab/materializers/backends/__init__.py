"""Materialization route implementations using backend adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .arrow_hf import write_arrow_hf_sparse
from .lancedb_aggregated import write_lancedb_aggregated
from .webdataset import write_webdataset_shards
from .zarr_ts import write_zarr_sparse_cell_chunks


def materialize_lancedb_aggregated(
    adata: Any,
    count_matrix: Any,
    release_id: str,
    matrix_root: Path,
    size_factors: np.ndarray | None = None,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
    dataset_id: str = "",
    corpus_index_path: Path | None = None,
) -> tuple[dict[str, Path], np.ndarray | None]:
    """Write into the true corpus-scoped Lance aggregated store.

    Returns ``(paths_dict, size_factors_or_None)``. Size factors are not
    computed inline for Lance; a placeholder ``None`` is returned.
    """
    if size_factors is None:
        n_obs = adata.n_obs
        size_factors = np.ones(n_obs, dtype=np.float64)
    result = write_lancedb_aggregated(
        adata,
        count_matrix,
        size_factors,
        release_id,
        matrix_root,
        canonical_perturbation=canonical_perturbation,
        canonical_context=canonical_context,
        raw_fields=raw_fields,
        dataset_id=dataset_id,
        corpus_index_path=corpus_index_path,
    )
    return (result, None)


def materialize_zarr_aggregated(
    adata: Any,
    count_matrix: Any,
    release_id: str,
    matrix_root: Path,
    size_factors: np.ndarray | None = None,
    chunk_cells: int = 1024,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
    dataset_id: str = "",
) -> tuple[dict[str, Path], np.ndarray | None]:
    """Phase 5 bounded alias while true aggregated Zarr append layout is pending.

    Returns ``(paths_dict, size_factors_or_None)``. Size factors are not
    computed inline for Zarr; a placeholder ``None`` is returned.
    """
    if size_factors is None:
        n_obs = adata.n_obs
        size_factors = np.ones(n_obs, dtype=np.float64)
    result = write_zarr_sparse_cell_chunks(
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
    return (result, None)


def materialize_arrow_hf(
    adata: Any,
    count_matrix: Any,
    release_id: str,
    matrix_root: Path,
    size_factors: np.ndarray | None = None,
    dataset_id: str = "",
    # Legacy parameters kept for backward compat with legacy route signatures
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
) -> tuple[dict[str, Path], np.ndarray]:
    """Write using Arrow/HF backend (primary, hardened first).

    Returns ``(paths_dict, size_factors_array)`` so the caller can write the
    separate size-factor parquet. Size factors are computed inline during the
    write traversal when ``size_factors`` is None, avoiding a redundant source scan.

    SQLite cell metadata is no longer written by this backend. The caller
    (Stage2Materializer) writes raw-obs Parquet and separate size-factor
    Parquet as the metadata sidecars.
    """
    return write_arrow_hf_sparse(
        adata, count_matrix, size_factors, release_id, matrix_root,
        dataset_id=dataset_id,
    )


def materialize_webdataset(
    adata: Any,
    count_matrix: Any,
    release_id: str,
    matrix_root: Path,
    size_factors: np.ndarray | None = None,
    shard_size: int = 10000,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
) -> tuple[dict[str, Path], np.ndarray | None]:
    """Write using WebDataset shard format with canonical metadata parity.

    Returns ``(paths_dict, size_factors_or_None)``. Size factors are not
    computed inline for WebDataset; a placeholder ``None`` is returned.
    """
    if size_factors is None:
        # Placeholder: webdataset does not compute size factors inline
        n_obs = adata.n_obs
        size_factors = np.ones(n_obs, dtype=np.float64)
    result = write_webdataset_shards(
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
    return (result, None)


def materialize_zarr(
    adata: Any,
    count_matrix: Any,
    release_id: str,
    matrix_root: Path,
    size_factors: np.ndarray | None = None,
    chunk_cells: int = 1024,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
) -> tuple[dict[str, Path], np.ndarray | None]:
    """Write using Zarr/TensorStore cell-chunked format with canonical metadata parity.

    Returns ``(paths_dict, size_factors_or_None)``. Size factors are not
    computed inline for Zarr; a placeholder ``None`` is returned.
    """
    if size_factors is None:
        n_obs = adata.n_obs
        size_factors = np.ones(n_obs, dtype=np.float64)
    result = write_zarr_sparse_cell_chunks(
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
    return (result, None)


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
