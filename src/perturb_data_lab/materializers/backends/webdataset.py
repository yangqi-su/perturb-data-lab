"""Backend adapter: WebDataset federated and aggregate writers.

Phase 5 (this file): thin serializer refactor — all writers accept ``ChunkBundle``
directly. No per-writer CSR logic, no legacy fallback, no ``_is_csr_dataset()``.
Gene indices in ``ChunkBundle.indices`` are always dataset-local.

Each dataset is written as a set of .tar shards containing per-cell
pickle records with the heavy-row data (expressed_gene_indices,
expression_counts) and size_factor.

Topology: federated (per-dataset files) and aggregate (corpus-scoped single shard set).
Backend name in registry: ``webdataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import io
import pickle
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..chunk_translation import ChunkBundle


def write_webdataset_federated(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
    cell_ids: tuple[str, ...] | None = None,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict | None]:
    """Stream ChunkBundle to WebDataset shard format with stateful tarfile.

    On first call (_writer_state is None): create state dict (no tar opened yet).
    On each chunk: open tar in write mode for first chunk, append mode for
    subsequent chunks (tarfile "a"), write all cells, close tar.
    On last call (_is_last_chunk=True): after writing cells, return None
    for writer_state.

    No bundles are accumulated in memory — each chunk is written directly
    to the tar shard as it arrives. The tar is opened and closed per chunk
    to keep state management simple (no filesystem handle held across chunks).

    Parameters
    ----------
    bundle : ChunkBundle
        The translated chunk bundle from ``_translate_chunk()``.
    release_id : str
        Release identifier used for shard file naming.
    matrix_root : Path
        Output directory for shard artifacts.
    cell_ids : tuple[str, ...] | None
        Cell IDs in order, one per row in the chunk.
    _writer_state : dict | None
        State dict holding shard path and cell offset.
        None on first chunk — new state is created.
    _is_last_chunk : bool
        True when this is the final chunk — returned writer_state is None.

    Returns
    -------
    tuple[dict[str, Path], dict | None]
        ``({"shard_path": ...}, state_or_None)``.
        On last chunk the second element is None.
    """
    import io
    import pickle
    import tarfile

    matrix_root.mkdir(parents=True, exist_ok=True)
    shard_path = matrix_root / f"{release_id}-cells.tar"

    if _writer_state is None:
        _writer_state = {
            "shard_path": shard_path,
            "cell_offset": 0,
        }

    table = bundle.table
    n_rows = bundle.row_count

    # Open tar in write mode for first chunk, append mode for subsequent chunks.
    mode = "w" if _writer_state["cell_offset"] == 0 else "a"
    with tarfile.open(str(shard_path), mode) as tar:
        for i in range(n_rows):
            global_idx = int(table["global_row_index"][i].as_py())
            row_indices = table["expressed_gene_indices"][i].as_py()
            row_counts = table["expression_counts"][i].as_py()

            cell_record = {
                "expressed_gene_indices": np.array(row_indices, dtype=np.int32),
                "expression_counts": np.array(row_counts, dtype=np.int32),
                "global_row_index": global_idx,
            }

            data_bytes = pickle.dumps(cell_record, protocol=pickle.HIGHEST_PROTOCOL)
            info = tarfile.TarInfo(name=f"cell_{global_idx:08d}.pkl")
            info.size = len(data_bytes)
            tar.addfile(info, io.BytesIO(data_bytes))

    _writer_state["cell_offset"] += n_rows

    paths = {"shard_path": shard_path}

    if _is_last_chunk:
        return paths, None
    else:
        return paths, _writer_state


def read_webdataset_cell(
    shard_path: Path,
    cell_member_name: str,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from a WebDataset shard.

    Returns ``(expressed_gene_indices, expression_counts, size_factor)``.
    """
    with tarfile.open(str(shard_path), "r") as tar:
        extracted = tar.extractfile(cell_member_name)
        if extracted is None:
            raise FileNotFoundError(cell_member_name)
        record = pickle.loads(extracted.read())

    indices = tuple(record["expressed_gene_indices"])
    counts = tuple(record["expression_counts"])

    sf = float(record.get("size_factor", 1.0))
    if size_factor_path is not None and size_factor_path.exists():
        import pyarrow.parquet as pq

        sf_table = pq.read_table(str(size_factor_path))
        sf = float(sf_table["size_factor"][record["global_row_index"]].as_py())

    return (indices, counts, sf)


# ---------------------------------------------------------------------------
# Aggregate writer (Phase 2 — streaming)
# ---------------------------------------------------------------------------

_AGGREGATE_SHARD_SIZE = 10_000


def write_webdataset_aggregate(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, Path], dict | None]:
    """Stream ChunkBundle to WebDataset aggregate shard format.

    On first call (_writer_state is None): create state dict with shard
    tracking counters and meta accumulator.
    On each call: iterate over the bundle's rows, open/create tar shards
    as needed, write each cell as a pickle record.
    On last call (_is_last_chunk=True): finalize the current shard, write
    the shard-index Parquet, and return None for writer_state.

    Parameters
    ----------
    bundle : ChunkBundle
        The translated chunk bundle from ``_translate_chunk()``.
    release_id : str
        Release identifier (used for metadata context only).
    matrix_root : Path
        Output directory for shard artifacts.
    _writer_state : dict | None
        State dict holding shard tracking info and accumulated meta rows.
        None on first chunk — new state is created.
    _is_last_chunk : bool
        True when this is the final chunk — returned writer_state is None.

    Returns
    -------
    tuple[dict[str, Path], dict | None]
        ``({"shard_paths": list[Path], "shard_index": Path}, state_or_None)``.
        On last chunk the second element is None.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)

    if _writer_state is None:
        _writer_state = {
            "current_shard_idx": 0,
            "cells_in_shard": 0,
            "meta_rows": [],  # list of dicts accumulated across chunks
            "shard_paths": [],  # list of Paths to all created shards
        }

    table = bundle.table
    n_rows = bundle.row_count

    # Group rows by shard so we can open/write/close the tar once per shard
    # rather than per row. This avoids O(n) tar open/close overhead.
    rows_by_shard: dict[int, list[tuple[int, Any, Any]]] = {}
    for i in range(n_rows):
        # Advance to the next shard if current one is full.
        if _writer_state["cells_in_shard"] >= _AGGREGATE_SHARD_SIZE:
            _writer_state["current_shard_idx"] += 1
            _writer_state["cells_in_shard"] = 0

        shard_idx = _writer_state["current_shard_idx"]
        global_idx = int(table["global_row_index"][i].as_py())
        row_indices = table["expressed_gene_indices"][i].as_py()
        row_counts = table["expression_counts"][i].as_py()

        rows_by_shard.setdefault(shard_idx, []).append(
            (global_idx, row_indices, row_counts)
        )
        _writer_state["meta_rows"].append({
            "global_row_index": global_idx,
            "shard": shard_idx,
        })
        _writer_state["cells_in_shard"] += 1

    # Sort shard indices for deterministic write order.
    for shard_idx in sorted(rows_by_shard):
        shard_path = matrix_root / f"aggregate-{shard_idx:05d}.tar"
        if shard_idx >= len(_writer_state["shard_paths"]):
            _writer_state["shard_paths"].append(shard_path)

        rows = rows_by_shard[shard_idx]

        # Determine write mode: "w" for new shard, "a" for existing.
        # A shard is "new" if it was just created (shard_idx == len(shard_paths) - 1
        # and it was appended above in this call).
        mode = "a" if shard_path.exists() else "w"
        with tarfile.open(str(shard_path), mode) as tar:
            for global_idx, row_indices, row_counts in rows:
                cell_record = {
                    "global_row_index": global_idx,
                    "expressed_gene_indices": np.array(row_indices, dtype=np.int32),
                    "expression_counts": np.array(row_counts, dtype=np.int32),
                }
                data_bytes = pickle.dumps(
                    cell_record, protocol=pickle.HIGHEST_PROTOCOL
                )
                info = tarfile.TarInfo(name=f"cell_{global_idx:08d}.pkl")
                info.size = len(data_bytes)
                tar.addfile(info, io.BytesIO(data_bytes))

    paths = {
        "shard_paths": _writer_state["shard_paths"],
        "shard_index": matrix_root / "aggregate-shard-index.parquet",
    }

    if _is_last_chunk:
        # Write shard-index Parquet for efficient random access by global_row_index.
        meta_rows = _writer_state["meta_rows"]
        shard_index_path = matrix_root / "aggregate-shard-index.parquet"
        shard_index_table = pa.table({
            "global_row_index": pa.array(
                [r["global_row_index"] for r in meta_rows], type=pa.int64()
            ),
            "shard": pa.array(
                [r["shard"] for r in meta_rows], type=pa.int32()
            ),
        })
        pq.write_table(shard_index_table, shard_index_path)
        return paths, None
    else:
        return paths, _writer_state