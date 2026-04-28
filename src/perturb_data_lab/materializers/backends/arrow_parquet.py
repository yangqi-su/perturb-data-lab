"""Backend adapter: Arrow Parquet federated and aggregate writers.

Phase 3: refactored to consume the shared translation layer ``_translate_chunk()``
instead of duplicating CSR-to-Arrow translation logic in each writer.

Phase 4: adds aggregate topology writer for corpus-scoped single-file output
with deterministic global row indices across datasets.

Phase 5 (this file): thin serializer refactor — all writers accept ``ChunkBundle``
directly. ``Stage2Materializer`` loops over chunks and calls ``_translate_chunk()``.
No per-writer CSR logic, no legacy fallback, no ``_is_csr_dataset()``, no
``use_translation_layer`` conditionals. Gene indices in ``ChunkBundle.indices``
are always dataset-local.

Produces:
- {release_id}-cells.parquet: heavy-row Arrow table (global_row_index,
  expressed_gene_indices LIST<INT32>, expression_counts LIST<INT32>)
- Caller (Stage2Materializer) writes the separate size-factor Parquet sidecar.

Topology: federated (per-dataset files) and aggregate (corpus-scoped single file).
Backend name in registry: ``arrow-parquet``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..chunk_translation import ChunkBundle


def write_arrow_parquet_federated(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict | None]:
    """Stream ChunkBundle tables to a single Parquet file.

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
        Writer state dict carrying the open ParquetWriter across chunks.
        None on first chunk — a new writer is opened.
    _is_last_chunk : bool
        True when this is the final chunk — closes the writer and returns None.

    Returns
    -------
    tuple[dict[str, Path], dict | None]
        ``({"cells": cells_parquet_path}, writer_state_or_None)``.
        On last chunk the second element is None.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / f"{release_id}-cells.parquet"

    if _writer_state is None:
        # First chunk: open writer
        writer = pq.ParquetWriter(cell_path, bundle.table.schema)
        _writer_state = {"writer": writer, "cell_path": cell_path}
    else:
        writer = _writer_state["writer"]

    writer.write_table(bundle.table)

    if _is_last_chunk:
        writer.close()
        return {"cells": cell_path}, None
    else:
        return {"cells": cell_path}, _writer_state


def _append_arrow_parquet(
    cell_path: Path,
    table: pa.Table,
) -> None:
    """Append a single-chunk ``pa.Table`` to an existing Arrow Parquet file.

    Uses streaming write mode (no schema mutation allowed after first write).
    The file must already exist with a valid schema.
    """
    writer = pq.ParquetWriter(cell_path, table.schema, mode="a")
    writer.write_table(table)
    writer.close()


def read_arrow_parquet_cell(
    parquet_path: Path,
    cell_index: int,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from an Arrow parquet.

    Returns ``(expressed_gene_indices, expression_counts, size_factor)``.

    ``size_factor_path`` is the path to the separate size-factor parquet.
    If provided, size factors are read from there. Falls back to reading
    from the ``size_factor`` column in the cells parquet for backward
    compatibility with pre-separate-parquet artifacts.
    """
    table = pq.read_table(parquet_path)
    indices = table["expressed_gene_indices"][cell_index].as_py()
    counts = table["expression_counts"][cell_index].as_py()

    if size_factor_path is not None and size_factor_path.exists():
        sf_table = pq.read_table(str(size_factor_path))
        sf = float(sf_table["size_factor"][cell_index].as_py())
    elif "size_factor" in table.column_names:
        sf = float(table["size_factor"][cell_index].as_py())
    else:
        raise KeyError(
            "size_factor not found in cells parquet and size_factor_path "
            "not provided; artifact may predate the separate size-factor layout. "
            "Provide size_factor_path to read size factors from the separate parquet."
        )

    return (tuple(indices), tuple(counts), sf)


# ---------------------------------------------------------------------------
# Aggregate writer (Phase 4)
# ---------------------------------------------------------------------------


def write_arrow_parquet_aggregate(
    bundles: list[ChunkBundle],
    matrix_root: Path,
) -> dict[str, Path]:
    """Write aggregate sparse per-cell data as a single Arrow Parquet file.

    This is the ``arrow-parquet × aggregate`` thin serializer.
    It consumes an ordered list of ``ChunkBundle`` objects (one per dataset)
    and concatenates them into a single corpus-scoped Parquet file with
    deterministic global_row_index values.

    Parameters
    ----------
    bundles : list[ChunkBundle]
        Chunk bundles in corpus order (one per dataset). Each bundle's
        ``table`` must follow ``HEAVY_CELL_SCHEMA``.
    matrix_root : Path
        Output directory for matrix artifacts.

    Returns
    -------
    dict[str, Path]
        ``paths_dict`` containing ``{"cells": cells_parquet_path}``.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / "aggregated-cells.parquet"

    writer: pq.ParquetWriter | None = None

    for bundle in bundles:
        if writer is None:
            writer = pq.ParquetWriter(cell_path, bundle.table.schema)
        writer.write_table(bundle.table)

    if writer is not None:
        writer.close()

    return {"cells": cell_path}