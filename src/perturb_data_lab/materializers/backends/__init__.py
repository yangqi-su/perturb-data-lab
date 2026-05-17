"""Materialization route implementations for the slim Lance/Zarr mainline.

Only Lance and Zarr remain supported in slim main, and both continue to expose
``federated`` and ``aggregate`` topology writers through the
``AVAILABLE_WRITERS[backend][topology]`` dispatch table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .zarr import write_zarr_federated
from .lance import write_lance_federated

# ---- aggregate writers ----
from .zarr import write_zarr_aggregate
from .lance import write_lance_aggregate


def materialize_zarr(
    bundle: Any,
    dataset_id: str,
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
        dataset_id=dataset_id,
        matrix_root=matrix_root,
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
    )


def materialize_lance(
    bundle: Any,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, Path], dict | None]:
    """Write using lance × federated backend.

    Thin serializer: streams bundles to Lance with append mode after first chunk.
    Returns ``({"cells": path}, writer_state_or_none)``.
    """
    return write_lance_federated(
        bundle=bundle,
        dataset_id=dataset_id,
        matrix_root=matrix_root,
        _lance_writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
    )


# ---------------------------------------------------------------------------
# Phase 3 dispatch table: AVAILABLE_WRITERS[backend][topology]
# ---------------------------------------------------------------------------

AVAILABLE_WRITERS: dict[str, dict[str, Any]] = {
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
        Backend name: ``lance`` or ``zarr``.
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
    assert backend in AVAILABLE_WRITERS, f"unknown backend: {backend}"
    topo_map = AVAILABLE_WRITERS[backend]
    assert topology in topo_map, f"unknown topology: {topology}"
    return topo_map[topology]
