"""Backend adapter: Lance federated and aggregate writers.

Phase 5 (this file): thin serializer refactor — all writers accept ``ChunkBundle``
directly. No per-writer CSR logic, no legacy fallback, no ``_is_csr_dataset()``.
Gene indices in ``ChunkBundle.indices`` are always dataset-local.

Produces per-dataset .lance files using ``lance.write_dataset()`` directly.

Topology: federated (per-dataset files) and aggregate (corpus-scoped single store).
Backend name in registry: ``lance``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..chunk_translation import ChunkBundle


def _import_lance():
    try:
        import lance
    except ImportError as exc:
        raise ImportError(
            "lance backend requires the Lance library; "
            "install lance in the selected runtime"
        ) from exc
    return lance


def write_lance_federated(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
    dataset_id: str = "",
    *,
    _lance_writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict | None]:
    """Stream ChunkBundle tables to a single Lance dataset.

    On first call (_lance_writer_state is None): create the dataset.
    On subsequent calls: append to the existing dataset.
    On last call (_is_last_chunk=True): return None for writer_state.

    Parameters
    ----------
    bundle : ChunkBundle
        The translated chunk bundle from ``_translate_chunk()``.
    release_id : str
        Release identifier used for output file naming.
    matrix_root : Path
        Output directory for matrix artifacts.
    dataset_id : str, optional
        Dataset identifier (used for Lance metadata context only).
    _lance_writer_state : dict | None
        Writer state dict. None on first chunk — dataset is created.
    _is_last_chunk : bool
        True when this is the final chunk — returns None for writer_state.

    Returns
    -------
    tuple[dict[str, Path], dict | None]
        ``({"cells": lance_path}, writer_state_or_None)``.
        On last chunk the second element is None.
    """
    lance = _import_lance()
    matrix_root.mkdir(parents=True, exist_ok=True)
    lance_path = matrix_root / f"{release_id}.lance"

    if _lance_writer_state is None:
        mode = "create"
        _lance_writer_state = {"initialized": True, "lance_path": str(lance_path)}
    else:
        mode = "append"

    lance.write_dataset(
        bundle.table,
        str(lance_path),
        mode=mode,
        max_rows_per_group=4096,
    )

    if _is_last_chunk:
        return {"cells": lance_path}, None
    else:
        return {"cells": lance_path}, _lance_writer_state


def read_lance_cell(
    lance_path: Path,
    cell_index: int,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from a Lance dataset.

    Returns ``(expressed_gene_indices, expression_counts, size_factor)``.
    """
    import lance

    ds = lance.dataset(str(lance_path))
    row = ds.take([cell_index]).to_pylist()[0]

    indices = tuple(row["expressed_gene_indices"])
    counts = tuple(row["expression_counts"])

    sf = 1.0
    if size_factor_path is not None:
        import pyarrow.parquet as pq

        sf_table = pq.read_table(str(size_factor_path))
        sf = float(sf_table["size_factor"][cell_index].as_py())

    return (indices, counts, sf)


# ---------------------------------------------------------------------------
# Aggregate writer (Phase 2 — streaming)
# ---------------------------------------------------------------------------


def write_lance_aggregate(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, Path], dict | None]:
    """Stream ChunkBundle tables to a single aggregated Lance dataset.

    On first call (_writer_state is None): create the dataset.
    On subsequent calls: append to the existing dataset.
    On last call (_is_last_chunk=True): return None for writer_state.

    Parameters
    ----------
    bundle : ChunkBundle
        The translated chunk bundle from ``_translate_chunk()``.
    release_id : str
        Release identifier (used for metadata context only).
    matrix_root : Path
        Output directory for matrix artifacts.
    _writer_state : dict | None
        Writer state dict. None on first chunk — dataset is created.
    _is_last_chunk : bool
        True when this is the final chunk — returns None for writer_state.

    Returns
    -------
    tuple[dict[str, Path], dict | None]
        ``({"cells": lance_path}, writer_state_or_None)``.
        On last chunk the second element is None.
    """
    lance = _import_lance()
    matrix_root.mkdir(parents=True, exist_ok=True)
    lance_path = matrix_root / "aggregated-cells.lance"

    if _writer_state is None:
        # First chunk: create if dataset doesn't exist, otherwise append.
        # This supports incremental aggregate materialization across
        # separate CLI invocations where a new Stage2Materializer instance
        # does not carry forward the previous writer_state.
        if lance_path.exists():
            mode = "append"
        else:
            mode = "create"
        _writer_state = {"lance_path": str(lance_path), "initialized": True}
    else:
        mode = "append"

    lance.write_dataset(
        bundle.table,
        str(lance_path),
        mode=mode,
        max_rows_per_group=4096,
    )

    if _is_last_chunk:
        return {"cells": lance_path}, None
    else:
        return {"cells": lance_path}, _writer_state