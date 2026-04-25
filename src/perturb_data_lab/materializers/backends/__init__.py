"""Materialization route implementations using backend adapters.

Phase 3 adds the new ``AVAILABLE_WRITERS[backend][topology]`` dispatch table
with explicit backend/topology separation. The legacy ``AVAILABLE_BACKENDS``
registry is preserved for backward compatibility with code that uses the old
fused names like ``arrow-hf``.

New backend names (Phase 3):
- ``arrow-parquet``: Arrow IPC over Parquet storage (federated)
- ``arrow-ipc``: Arrow IPC file storage (federated)
- ``webdataset``: WebDataset shard format (federated)
- ``zarr``: Zarr 1D flat-buffer storage (federated)
- ``lance``: Lance dataset storage (federated)

New topology names (Phase 3):
- ``federated``: per-dataset output files
- ``aggregate``: corpus-scoped single output files

Migration map:
- ``arrow-hf`` → ``arrow-parquet × federated``
- ``webdataset`` → ``webdataset × federated``
- ``zarr-ts`` → ``zarr × federated``
- ``lancedb-aggregated`` → ``lance × aggregate``
- ``zarr-aggregated`` → ``zarr × aggregate``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ---- New Phase 3 federated writers ----
from .arrow_parquet import write_arrow_parquet_federated
from .arrow_ipc import write_arrow_ipc_federated
from .webdataset import write_webdataset_federated
from .zarr import write_zarr_federated
from .lance import write_lance_federated

# ---- Phase 4 aggregate writers ----
from .arrow_parquet import write_arrow_parquet_aggregate
from .arrow_ipc import write_arrow_ipc_aggregate
from .webdataset import write_webdataset_aggregate
from .zarr import write_zarr_aggregate
from .lance import write_lance_aggregate

# ---- Legacy writers (kept for backward compat) ----
from .arrow_hf import write_arrow_hf_sparse
from .lancedb_aggregated import write_lancedb_aggregated
from .zarr_ts import write_zarr_sparse_cell_chunks


# ---------------------------------------------------------------------------
# Federated writer functions (topology=federated)
# ---------------------------------------------------------------------------

def materialize_arrow_parquet(
    adata: Any,
    count_matrix: Any,
    release_id: str,
    matrix_root: Path,
    size_factors: np.ndarray | None = None,
    dataset_id: str = "",
    chunk_rows: int = 100_000,
) -> tuple[dict[str, Path], np.ndarray]:
    """Write using arrow-parquet × federated backend.

    Returns ``(paths_dict, size_factors_array)``.
    """
    return write_arrow_parquet_federated(
        adata, count_matrix, size_factors, release_id, matrix_root,
        dataset_id=dataset_id, chunk_rows=chunk_rows,
    )


def materialize_arrow_ipc(
    adata: Any,
    count_matrix: Any,
    release_id: str,
    matrix_root: Path,
    size_factors: np.ndarray | None = None,
    dataset_id: str = "",
    chunk_rows: int = 100_000,
) -> tuple[dict[str, Path], np.ndarray]:
    """Write using arrow-ipc × federated backend.

    Returns ``(paths_dict, size_factors_array)``.
    """
    return write_arrow_ipc_federated(
        adata, count_matrix, size_factors, release_id, matrix_root,
        dataset_id=dataset_id, chunk_rows=chunk_rows,
    )


def materialize_webdataset(
    adata: Any,
    count_matrix: Any,
    release_id: str,
    matrix_root: Path,
    size_factors: np.ndarray | None = None,
    shard_size: int = 10_000,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
) -> tuple[dict[str, Path], np.ndarray | None]:
    """Write using webdataset × federated backend.

    Returns ``(paths_dict, size_factors_or_None)``. WebDataset writes size
    factors into per-cell records; a placeholder ``None`` is returned for
    the caller's separate size-factor parquet (since WebDataset embeds them).
    """
    if size_factors is None:
        n_obs = adata.n_obs
        size_factors = np.ones(n_obs, dtype=np.float64)
    result = write_webdataset_federated(
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
    """Write using zarr × federated backend.

    Returns ``(paths_dict, size_factors_or_None)``. Zarr does not compute
    size factors inline; a placeholder ``None`` is returned.
    """
    if size_factors is None:
        n_obs = adata.n_obs
        size_factors = np.ones(n_obs, dtype=np.float64)
    result = write_zarr_federated(
        adata,
        count_matrix,
        size_factors,
        release_id,
        matrix_root,
        chunk_cells=chunk_cells,
    )
    return (result, None)


def materialize_lance(
    adata: Any,
    count_matrix: Any,
    release_id: str,
    matrix_root: Path,
    size_factors: np.ndarray | None = None,
    dataset_id: str = "",
) -> tuple[dict[str, Path], np.ndarray | None]:
    """Write using lance × federated backend.

    Returns ``(paths_dict, size_factors_or_None)``. Lance does not compute
    size factors inline; a placeholder ``None`` is returned.
    """
    if size_factors is None:
        n_obs = adata.n_obs
        size_factors = np.ones(n_obs, dtype=np.float64)
    result = write_lance_federated(
        adata,
        count_matrix,
        size_factors,
        release_id,
        matrix_root,
        dataset_id=dataset_id,
    )
    return (result, None)


# ---------------------------------------------------------------------------
# Legacy aliases (backward compat for existing callers)
# ---------------------------------------------------------------------------

def materialize_arrow_hf(
    adata: Any,
    count_matrix: Any,
    release_id: str,
    matrix_root: Path,
    size_factors: np.ndarray | None = None,
    dataset_id: str = "",
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
) -> tuple[dict[str, Path], np.ndarray]:
    """Legacy alias: ``arrow-hf`` → delegates to ``arrow-parquet`` writer.

    Returns ``(paths_dict, size_factors_array)`` so the caller can write the
    separate size-factor parquet.
    """
    return write_arrow_parquet_federated(
        adata, count_matrix, size_factors, release_id, matrix_root,
        dataset_id=dataset_id,
    )


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
    """Write using zarr-aggregated (Phase 5 bounded alias)."""
    if size_factors is None:
        n_obs = adata.n_obs
        size_factors = np.ones(n_obs, dtype=np.float64)
    result = write_zarr_sparse_cell_chunks(
        adata, count_matrix, size_factors, release_id, matrix_root,
        chunk_cells=chunk_cells,
        canonical_perturbation=canonical_perturbation,
        canonical_context=canonical_context,
        raw_fields=raw_fields,
        dataset_id=dataset_id,
    )
    return (result, None)


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
    """Write into the true corpus-scoped Lance aggregated store."""
    if size_factors is None:
        n_obs = adata.n_obs
        size_factors = np.ones(n_obs, dtype=np.float64)
    result = write_lancedb_aggregated(
        adata, count_matrix, size_factors, release_id, matrix_root,
        canonical_perturbation=canonical_perturbation,
        canonical_context=canonical_context,
        raw_fields=raw_fields,
        dataset_id=dataset_id,
        corpus_index_path=corpus_index_path,
    )
    return (result, None)


# ---------------------------------------------------------------------------
# New dispatch table: AVAILABLE_WRITERS[backend][topology]
# ---------------------------------------------------------------------------

AVAILABLE_WRITERS: dict[str, dict[str, Any]] = {
    "arrow-parquet": {
        "federated": materialize_arrow_parquet,
        "aggregate": write_arrow_parquet_aggregate,
    },
    "arrow-ipc": {
        "federated": materialize_arrow_ipc,
        "aggregate": write_arrow_ipc_aggregate,
    },
    "webdataset": {
        "federated": materialize_webdataset,
        "aggregate": write_webdataset_aggregate,
    },
    "zarr": {
        "federated": materialize_zarr,
        "aggregate": write_zarr_aggregate,
    },
    "lance": {
        "federated": materialize_lance,
        "aggregate": write_lance_aggregate,
    },
}


# ---------------------------------------------------------------------------
# Legacy registry (fused names, backward compat)
# ---------------------------------------------------------------------------

AVAILABLE_BACKENDS: dict[str, Any] = {
    # Legacy fused names → same writer functions as above
    "arrow-hf": materialize_arrow_hf,
    "webdataset": materialize_webdataset,
    "zarr-ts": materialize_zarr,          # alias to new federated zarr
    "lancedb-aggregated": materialize_lancedb_aggregated,
    "zarr-aggregated": materialize_zarr_aggregated,
}


def build_backend_fn(backend: str, topology: str = "federated"):
    """Select backend materialization function by name.

    Supports both legacy fused names (``arrow-hf``, ``webdataset``, ``zarr-ts``)
    and new Phase 3 backend names (``arrow-parquet``, ``arrow-ipc``,
    ``webdataset``, ``zarr``, ``lance``).

    Parameters
    ----------
    backend : str
        Backend name.
    topology : str, default "federated"
        Topology: "federated" (per-dataset files) or "aggregate" (corpus-scoped).
    """
    # Check new-style dispatch first when topology is explicitly non-federated
    if backend in AVAILABLE_WRITERS:
        topo_map = AVAILABLE_WRITERS[backend]
        if topology in topo_map:
            return topo_map[topology]
        if "federated" in topo_map:
            return topo_map["federated"]
    # Fall back to legacy BACKEND registry for backward compat
    if backend in AVAILABLE_BACKENDS:
        return AVAILABLE_BACKENDS[backend]
    raise ValueError(
        f"unknown backend: {backend}; "
        f"available legacy: {list(AVAILABLE_BACKENDS)}, "
        f"available new: {list(AVAILABLE_WRITERS)}"
    )