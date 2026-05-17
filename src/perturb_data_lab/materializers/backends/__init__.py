"""Materialization writer dispatch for the current Lance/Zarr backends."""

from __future__ import annotations

from typing import Any

from .lance import write_lance_aggregate, write_lance_federated
from .zarr import write_zarr_aggregate, write_zarr_federated


AVAILABLE_WRITERS: dict[str, dict[str, Any]] = {
    "zarr": {
        "federated": write_zarr_federated,
        "aggregate": write_zarr_aggregate,
    },
    "lance": {
        "federated": write_lance_federated,
        "aggregate": write_lance_aggregate,
    },
}


def build_backend_fn(backend: str, topology: str = "federated"):
    """Select the writer function for ``backend`` and ``topology``."""
    assert backend in AVAILABLE_WRITERS, f"unknown backend: {backend}"
    topo_map = AVAILABLE_WRITERS[backend]
    assert topology in topo_map, f"unknown topology: {topology}"
    return topo_map[topology]
