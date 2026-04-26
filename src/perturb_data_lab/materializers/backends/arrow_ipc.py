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

Topology: federated (per-dataset files) and aggregate (corpus-scoped single file).
Backend name in registry: ``arrow-ipc``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from ..chunk_translation import ChunkBundle


def write_arrow_ipc_federated(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
) -> tuple[dict[str, Path], np.ndarray]:
    """Write a single-chunk ``ChunkBundle`` as an Arrow IPC file.

    This is the ``arrow-ipc × federated`` thin serializer.
    It accepts a ``ChunkBundle`` from ``_translate_chunk()`` and writes the
    heavy-row ``table`` directly to IPC format. Size factors are returned
    to the caller for separate sidecar write.

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
        ``{"cells": cells_arrow_path}``.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / f"{release_id}-cells.arrow"

    writer = pa.ipc.new_file(str(cell_path), bundle.table.schema)
    batch = bundle.table.to_batches()[0]
    writer.write_batch(batch)
    writer.close()

    return ({"cells": cell_path}, bundle.size_factors)


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


# ---------------------------------------------------------------------------
# Aggregate writer (Phase 4)
# ---------------------------------------------------------------------------


def write_arrow_ipc_aggregate(
    bundles: list[ChunkBundle],
    matrix_root: Path,
) -> tuple[dict[str, Path], list[np.ndarray]]:
    """Write aggregate sparse per-cell data as a single Arrow IPC file.

    This is the ``arrow-ipc × aggregate`` thin serializer.
    It consumes an ordered list of ``ChunkBundle`` objects (one per dataset)
    and concatenates them into a single corpus-scoped IPC file with
    deterministic global_row_index values.

    Parameters
    ----------
    bundles : list[ChunkBundle]
        Chunk bundles in corpus order (one per dataset).
    matrix_root : Path
        Output directory for matrix artifacts.

    Returns
    -------
    tuple[dict[str, Path], list[np.ndarray]]
        ``(paths_dict, size_factors_out_list)`` where paths_dict contains
        ``{"cells": cells_arrow_path}``.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)
    cell_path = matrix_root / "aggregated-cells.arrow"

    writer: pa.ipc.RecordBatchFileWriter | None = None
    size_factors_out_list: list[np.ndarray] = []

    for bundle in bundles:
        if writer is None:
            writer = pa.ipc.new_file(str(cell_path), bundle.table.schema)
        batch = bundle.table.to_batches()[0]
        writer.write_batch(batch)
        size_factors_out_list.append(bundle.size_factors)

    if writer is not None:
        writer.close()

    return ({"cells": cell_path}, size_factors_out_list)