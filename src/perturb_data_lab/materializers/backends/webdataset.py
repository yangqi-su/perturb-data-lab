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
# Aggregate writer (Phase 4)
# ---------------------------------------------------------------------------


def write_webdataset_aggregate(
    bundles: list[ChunkBundle],
    matrix_root: Path,
) -> dict[str, Path]:
    """Write aggregate sparse per-cell data in WebDataset shard format.

    This is the ``webdataset × aggregate`` thin serializer.
    It consumes an ordered list of ``ChunkBundle`` objects and writes them
    as a single corpus-scoped set of .tar shards with deterministic
    global_row_index values spanning all datasets. Size factors are in a
    separate Parquet sidecar written by the caller after all chunks.

    Parameters
    ----------
    bundles : list[ChunkBundle]
        Chunk bundles in corpus order (one per dataset).
    matrix_root : Path
        Output directory.

    Returns
    -------
    dict[str, Path]
        ``paths_dict`` containing ``{"shard_paths": list[Path], "meta": meta_parquet_path}``.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)

    total_cells = sum(b.row_count for b in bundles)
    shard_size = 10_000
    n_shards = (total_cells + shard_size - 1) // shard_size

    shard_paths: list[Path] = []
    meta_rows: list[dict[str, Any]] = []

    cell_global_index = 0  # running global row index across all datasets

    for shard_idx in range(n_shards):
        shard_path = matrix_root / f"aggregate-{shard_idx:05d}.tar"
        shard_paths.append(shard_path)

        with tarfile.open(str(shard_path), "w") as tar:
            for _ in range(shard_size):
                if cell_global_index >= total_cells:
                    break

                # Find which dataset this cell belongs to
                ds_idx = 0
                cum_rows = bundles[0].row_count
                while ds_idx < len(bundles) - 1 and cell_global_index >= cum_rows:
                    ds_idx += 1
                    cum_rows += bundles[ds_idx].row_count

                bundle = bundles[ds_idx]
                local_index = cell_global_index - (cum_rows - bundle.row_count)

                table = bundle.table
                global_row_indices = table["global_row_index"].to_numpy()
                global_idx = int(global_row_indices[local_index])
                row_indices = table["expressed_gene_indices"][local_index].as_py()
                row_counts = table["expression_counts"][local_index].as_py()

                cell_record = {
                    "global_row_index": global_idx,
                    "expressed_gene_indices": np.array(row_indices, dtype=np.int32),
                    "expression_counts": np.array(row_counts, dtype=np.int32),
                    "dataset_index": ds_idx,
                }

                data_bytes = pickle.dumps(cell_record, protocol=pickle.HIGHEST_PROTOCOL)
                info = tarfile.TarInfo(name=f"cell_{global_idx:08d}.pkl")
                info.size = len(data_bytes)
                tar.addfile(info, io.BytesIO(data_bytes))

                meta_rows.append({
                    "global_row_index": global_idx,
                    "dataset_index": ds_idx,
                    "shard": shard_idx,
                })

                cell_global_index += 1

    # Write meta as Parquet for efficient random access.
    meta_path = matrix_root / "aggregate-meta.parquet"
    meta_table = pa.table({
        "global_row_index": pa.array([r["global_row_index"] for r in meta_rows], type=pa.int64()),
        "dataset_index": pa.array([r["dataset_index"] for r in meta_rows], type=pa.int32()),
        "shard": pa.array([r["shard"] for r in meta_rows], type=pa.int32()),
    })
    pq.write_table(meta_table, meta_path)

    return {"shard_paths": shard_paths, "meta": meta_path}