"""Materialization route implementations using backend adapters.

Phase 3 introduces the new ``AVAILABLE_WRITERS[backend][topology]`` dispatch table
with explicit backend/topology separation. The legacy ``AVAILABLE_BACKENDS``
registry and all fused-name aliases have been removed.

New backend names (Phase 3):
- ``arrow-parquet``: Arrow IPC over Parquet storage
- ``arrow-ipc``: Arrow IPC file storage
- ``webdataset``: WebDataset shard format
- ``zarr``: Zarr 1D flat-buffer storage
- ``lance``: Lance dataset storage

New topology names (Phase 3):
- ``federated``: per-dataset output files
- ``aggregate``: corpus-scoped single output files

Removed (Phase 1 — backend-topology validation):
- ``arrow-parquet × aggregate``: not supported (no true append in Parquet)
- ``arrow-ipc × aggregate``: not supported (no true append in IPC files)

Migration map (legacy → canonical):
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

# ---- Phase 3 federated writers ----
from .arrow_parquet import write_arrow_parquet_federated
from .arrow_ipc import write_arrow_ipc_federated
from .webdataset import write_webdataset_federated
from .zarr import write_zarr_federated
from .lance import write_lance_federated

# ---- Phase 3 aggregate writers ----
from .webdataset import write_webdataset_aggregate
from .zarr import write_zarr_aggregate
from .lance import write_lance_aggregate


# ---------------------------------------------------------------------------
# Thin federated writer wrappers — accept ChunkBundle + path/config kwargs
# ---------------------------------------------------------------------------

def materialize_arrow_parquet(
    bundle: Any,
    release_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, Path], dict | None]:
    """Write using arrow-parquet × federated backend.

    Thin serializer: accepts a single ``ChunkBundle`` and streams it to Parquet.
    Returns ``({"cells": path}, writer_state_or_none)``.
    """
    return write_arrow_parquet_federated(
        bundle=bundle,
        release_id=release_id,
        matrix_root=matrix_root,
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
    )


def materialize_arrow_ipc(
    bundle: Any,
    release_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, Path], dict | None]:
    """Write using arrow-ipc × federated backend.

    Thin serializer: accepts a single ``ChunkBundle`` and streams it to IPC.
    Returns ``({"cells": path}, writer_state_or_none)``.
    """
    return write_arrow_ipc_federated(
        bundle=bundle,
        release_id=release_id,
        matrix_root=matrix_root,
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
    )


def materialize_webdataset(
    bundle: Any,
    release_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
    cell_ids: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Path], dict | None]:
    """Write using webdataset × federated backend.

    Thin serializer: streams each ``ChunkBundle`` directly to a tar shard.
    Returns ``({"shard_path": ..., "meta": ...}, writer_state_or_none)``.
    """
    return write_webdataset_federated(
        bundle=bundle,
        release_id=release_id,
        matrix_root=matrix_root,
        cell_ids=cell_ids,
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
    )


def materialize_zarr(
    bundle: Any,
    release_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, Path], dict | None]:
    """Write using zarr × federated backend.

    Thin serializer: streams each ``ChunkBundle`` directly to open zarr groups.
    Returns ``({"indices": ..., "counts": ..., "row_offsets": ..., "meta": ...}, writer_state_or_none)``.
    """
    return write_zarr_federated(
        bundle=bundle,
        release_id=release_id,
        matrix_root=matrix_root,
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
    )


def materialize_lance(
    bundle: Any,
    release_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
    dataset_id: str = "",
    **kwargs: Any,
) -> tuple[dict[str, Path], dict | None]:
    """Write using lance × federated backend.

    Thin serializer: streams bundles to Lance with append mode after first chunk.
    Returns ``({"cells": path}, writer_state_or_none)``.
    """
    return write_lance_federated(
        bundle=bundle,
        release_id=release_id,
        matrix_root=matrix_root,
        dataset_id=dataset_id,
        _lance_writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
    )


# ---------------------------------------------------------------------------
# Phase 3 dispatch table: AVAILABLE_WRITERS[backend][topology]
# ---------------------------------------------------------------------------

AVAILABLE_WRITERS: dict[str, dict[str, Any]] = {
    "arrow-parquet": {
        "federated": materialize_arrow_parquet,
    },
    "arrow-ipc": {
        "federated": materialize_arrow_ipc,
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


def build_backend_fn(backend: str, topology: str = "federated"):
    """Select backend materialization function by canonical name and topology.

    Parameters
    ----------
    backend : str
        Backend name: ``arrow-parquet``, ``arrow-ipc``, ``webdataset``,
        ``zarr``, or ``lance``.
    topology : str, default "federated"
        Topology: ``federated`` (per-dataset files) or ``aggregate``
        (corpus-scoped).

    Returns
    -------
    Callable
        The writer function for the specified backend and topology.

    Raises
    ------
    KeyError
        If the backend/topology combination is not supported.
    """
    if backend not in AVAILABLE_WRITERS:
        raise KeyError(
            f"unknown backend: {backend}; "
            f"available backends: {list(AVAILABLE_WRITERS)}"
        )
    topo_map = AVAILABLE_WRITERS[backend]
    if topology not in topo_map:
        raise KeyError(
            f"unknown topology '{topology}' for backend '{backend}'; "
            f"available topologies: {list(topo_map)}"
        )
    return topo_map[topology]
