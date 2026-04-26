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
) -> dict[str, Path]:
    """Write a ``ChunkBundle`` as a WebDataset shard (.tar with .pkl per cell).

    This is the ``webdataset × federated`` thin serializer.
    It accepts a ``ChunkBundle`` from ``_translate_chunk()`` and writes each
    cell as a ``cell_{global_row_index:08d}.pkl`` pickle record. The meta
    parquet (cell_id ↔ shard mapping) is written separately by the caller
    (``Stage2Materializer``).

    Parameters
    ----------
    bundle : ChunkBundle
        The translated chunk bundle from ``_translate_chunk()``.
    release_id : str
        Release identifier used for shard file naming.
    matrix_root : Path
        Output directory for shard artifacts.
    cell_ids : tuple[str, ...] | None
        Cell IDs in order, one per row in the chunk. If None, cells are
        named by their global_row_index.

    Returns a dict with keys: ``{"shard_path": Path, "meta": Path}``.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)

    n_rows = bundle.row_count
    table = bundle.table
    global_row_indices = table["global_row_index"].to_numpy()
    size_factors = bundle.size_factors

    shard_path = matrix_root / f"{release_id}-cells.tar"
    meta_rows: list[dict[str, Any]] = []

    with tarfile.open(str(shard_path), "w") as tar:
        for i in range(n_rows):
            global_idx = int(global_row_indices[i])
            row_indices = table["expressed_gene_indices"][i].as_py()
            row_counts = table["expression_counts"][i].as_py()
            cell_sf = float(size_factors[i])

            cell_record = {
                "expressed_gene_indices": np.array(row_indices, dtype=np.int32),
                "expression_counts": np.array(row_counts, dtype=np.int32),
                "size_factor": cell_sf,
                "global_row_index": global_idx,
            }
            if cell_ids is not None:
                cell_record["cell_id"] = cell_ids[i]

            data_bytes = pickle.dumps(cell_record, protocol=pickle.HIGHEST_PROTOCOL)
            info = tarfile.TarInfo(name=f"cell_{global_idx:08d}.pkl")
            info.size = len(data_bytes)
            tar.addfile(info, io.BytesIO(data_bytes))

            cell_id = cell_ids[i] if cell_ids is not None else str(global_idx)
            meta_rows.append({
                "cell_id": cell_id,
                "size_factor": cell_sf,
            })

    # Write meta as Parquet for efficient random access.
    meta_path = matrix_root / f"{release_id}-meta.parquet"
    meta_table = pa.table({
        "cell_id": pa.array([r["cell_id"] for r in meta_rows], type=pa.string()),
        "size_factor": pa.array([r["size_factor"] for r in meta_rows], type=pa.float64()),
    })
    pq.write_table(meta_table, meta_path)

    return {"shard_path": shard_path, "meta": meta_path}


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
        sf = float(sf_table["size_factor"][cell_index].as_py())

    return (indices, counts, sf)


# ---------------------------------------------------------------------------
# Aggregate writer (Phase 4)
# ---------------------------------------------------------------------------


def write_webdataset_aggregate(
    bundles: list[ChunkBundle],
    matrix_root: Path,
) -> tuple[dict[str, Path], list[np.ndarray]]:
    """Write aggregate sparse per-cell data in WebDataset shard format.

    This is the ``webdataset × aggregate`` thin serializer.
    It consumes an ordered list of ``ChunkBundle`` objects and writes them
    as a single corpus-scoped set of .tar shards with deterministic
    global_row_index values spanning all datasets.

    Parameters
    ----------
    bundles : list[ChunkBundle]
        Chunk bundles in corpus order (one per dataset).
    matrix_root : Path
        Output directory.

    Returns
    -------
    tuple[dict[str, Path], list[np.ndarray]]
        ``(paths_dict, size_factors_out_list)`` where paths_dict contains
        ``{"shard_paths": list[Path], "meta": meta_parquet_path}``.
    """
    matrix_root.mkdir(parents=True, exist_ok=True)

    total_cells = sum(b.row_count for b in bundles)
    shard_size = 10_000
    n_shards = (total_cells + shard_size - 1) // shard_size

    shard_paths: list[Path] = []
    meta_rows: list[dict[str, Any]] = []

    cell_global_index = 0  # running global row index across all datasets
    size_factors_out_list: list[np.ndarray] = []

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
                cell_sf = float(bundle.size_factors[local_index])

                cell_record = {
                    "global_row_index": global_idx,
                    "expressed_gene_indices": np.array(row_indices, dtype=np.int32),
                    "expression_counts": np.array(row_counts, dtype=np.int32),
                    "size_factor": cell_sf,
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
                    "size_factor": cell_sf,
                })

                cell_global_index += 1

        size_factors_out_list.append(bundle.size_factors)

    # Write meta as Parquet for efficient random access.
    meta_path = matrix_root / "aggregate-meta.parquet"
    meta_table = pa.table({
        "global_row_index": pa.array([r["global_row_index"] for r in meta_rows], type=pa.int64()),
        "dataset_index": pa.array([r["dataset_index"] for r in meta_rows], type=pa.int32()),
        "shard": pa.array([r["shard"] for r in meta_rows], type=pa.int32()),
        "size_factor": pa.array([r["size_factor"] for r in meta_rows], type=pa.float64()),
    })
    pq.write_table(meta_table, meta_path)

    return ({"shard_paths": shard_paths, "meta": meta_path}, size_factors_out_list)