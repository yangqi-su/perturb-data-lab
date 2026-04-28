"""Backend adapter: Arrow IPC federated and aggregate writers.

Phase 3: refactored to consume the shared ``_translate_chunk()`` translation
layer instead of per-backend sparse re-encoding.

Phase 4: adds aggregate topology writer for corpus-scoped single-file output
with deterministic global row indices across datasets.

Phase 5 (this file): thin serializer refactor — all writers accept ``ChunkBundle``
directly. No per-writer CSR logic, no legacy fallback, no ``_is_csr_dataset()``,
no ``use_translation_layer`` conditionals. Gene indices in ``ChunkBundle.indices``
are always dataset-local.

Produces:
- {release_id}-cells.arrow: heavy-row Arrow IPC file (global_row_index,
  expressed_gene_indices LIST<INT32>, expression_counts LIST<INT32>)
- Caller writes the separate size-factor Parquet sidecar.

Topology: federated (per-dataset files).
Backend name in registry: ``arrow-ipc``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from ..chunk_translation import ChunkBundle


def write_arrow_ipc_federated(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict | None]:
    """Stream ChunkBundle tables to a single Arrow IPC file.

    On first call (_writer_state is None): open writer and store in state dict.
    On subsequent calls: reuse the open writer from state.
    On last call (_is_last_chunk=True): close writer and return None for writer_state.

    Parameters
    ----------
    bundle : ChunkBundle
        The translated chunk bundle from ``_translate_chunk()``.
    release_id : str
        Release identifier used for output file naming.
    matrix_root : Path
        Output directory for matrix artifacts.
    _writer_state : dict | None
        Writer state dict carrying the open IPC writer across chunks.
        None on first chunk — a new writer is opened.
    _is_last_chunk : bool
        True when this is the final chunk — closes the writer and returns None.

    Returns
    -------
    tuple[dict[str, Path], dict | None]
        ``({"cells": cells_arrow_path}, writer_state_or_None)``.
        On last chunk the second element is None.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / f"{release_id}-cells.arrow"

    if _writer_state is None:
        writer = pa.ipc.new_file(str(cell_path), bundle.table.schema)
        _writer_state = {"writer": writer, "cell_path": cell_path}
    else:
        writer = _writer_state["writer"]

    batch = bundle.table.to_batches()[0]
    writer.write_batch(batch)

    if _is_last_chunk:
        writer.close()
        return {"cells": cell_path}, None
    else:
        return {"cells": cell_path}, _writer_state


def read_arrow_ipc_cell(
    arrow_path: Path,
    cell_index: int,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from an Arrow IPC file.

    Returns ``(expressed_gene_indices, expression_counts, size_factor)``.
    """
    with pa.memory_map(str(arrow_path), "r") as source:
        reader = pa_ipc.RecordBatchFileReader(source)
        running = 0
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            next_running = running + batch.num_rows
            if running <= cell_index < next_running:
                local_idx = cell_index - running
                indices = batch.column("expressed_gene_indices")[local_idx].as_py()
                counts = batch.column("expression_counts")[local_idx].as_py()
                break
            running = next_running
        else:
            raise IndexError(cell_index)

    if size_factor_path is not None and size_factor_path.exists():
        import pyarrow.parquet as pq

        sf_table = pq.read_table(str(size_factor_path))
        sf = float(sf_table["size_factor"][cell_index].as_py())
    else:
        sf = 1.0

    return (tuple(indices), tuple(counts), sf)

