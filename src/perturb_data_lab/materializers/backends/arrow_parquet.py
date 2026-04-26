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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..chunk_translation import ChunkBundle


def write_arrow_parquet_federated(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
) -> tuple[dict[str, Path], np.ndarray]:
    """Write a single-chunk ``ChunkBundle`` as an Arrow Parquet file.

    This is the ``arrow-parquet × federated`` thin serializer.
    It accepts a ``ChunkBundle`` from ``_translate_chunk()`` and writes the
    heavy-row ``table`` directly to Parquet. Size factors are returned to the
    caller for separate sidecar write.

    Parameters
    ----------
    bundle : ChunkBundle
        The translated chunk bundle from ``_translate_chunk()``.
    release_id : str
        Release identifier used for output file naming.
    matrix_root : Path
        Output directory for matrix artifacts.

    Returns
    -------
    tuple[dict[str, Path], np.ndarray]
        ``(paths_dict, size_factors_array)`` where paths_dict contains
        ``{"cells": cells_parquet_path}``. The size_factors_array contains
        median-normalized per-cell factors (clamped at >0, NaN-safe).
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / f"{release_id}-cells.parquet"

    writer = pq.ParquetWriter(cell_path, bundle.table.schema)
    writer.write_table(bundle.table)
    writer.close()

    return ({"cells": cell_path}, bundle.size_factors)


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
) -> tuple[dict[str, Path], list[np.ndarray]]:
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
    tuple[dict[str, Path], list[np.ndarray]]
        ``(paths_dict, size_factors_list)`` where paths_dict contains
        ``{"cells": cells_parquet_path}``. size_factors_list contains the
        computed size factors for each dataset.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / "aggregated-cells.parquet"

    writer: pq.ParquetWriter | None = None
    size_factors_out_list: list[np.ndarray] = []

    for bundle in bundles:
        if writer is None:
            writer = pq.ParquetWriter(cell_path, bundle.table.schema)
        writer.write_table(bundle.table)
        size_factors_out_list.append(bundle.size_factors)

    if writer is not None:
        writer.close()

    return ({"cells": cell_path}, size_factors_out_list)