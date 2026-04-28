"""Backend adapter: Zarr federated and aggregate writers.

Phase 5 (this file): thin serializer refactor — all writers accept ``ChunkBundle``
directly. No per-writer CSR logic, no legacy fallback, no ``_is_csr_dataset()``.
Gene indices in ``ChunkBundle.indices`` are always dataset-local.

Zarr layout (federated):
- {release_id}-indices.zarr: 1D flat buffer of all gene indices
- {release_id}-counts.zarr: 1D flat buffer of all counts
- {release_id}-row-offsets.zarr: row offset boundaries
- {release_id}-meta.json: cell metadata + size_factor path reference

Zarr layout (aggregate):
- aggregated-indices.zarr: 1D flat buffer of all gene indices across datasets
- aggregated-counts.zarr: 1D flat buffer of all counts across datasets
- aggregated-row-offsets.zarr: row offsets including dataset boundaries
- aggregated-meta.json: corpus-level metadata with dataset offsets

Topology: federated (per-dataset files) and aggregate (corpus-scoped single file).
Backend name in registry: ``zarr``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from ..chunk_translation import ChunkBundle


def write_zarr_federated(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict | None]:
    """Stream ChunkBundle to Zarr format with stateful open-zarr groups.

    On first call (_writer_state is None): create zarr arrays with initial
    size based on first chunk's nnz, store zarr group references in state.
    On subsequent calls: resize zarr arrays as needed, write chunk data
    via slice assignment.
    On last call (_is_last_chunk=True): resize arrays to exact final sizes,
    write meta.json, and return None for writer_state.

    No bundles are accumulated in memory — each chunk is written directly
    to the open zarr arrays as it arrives.

    Parameters
    ----------
    bundle : ChunkBundle
        The translated chunk bundle from ``_translate_chunk()``.
    release_id : str
        Release identifier used for output file naming.
    matrix_root : Path
        Output directory for matrix artifacts.
    _writer_state : dict | None
        State dict holding open zarr group references, accumulated nnz/row
        counts, and output paths. None on first chunk — new state is created.
    _is_last_chunk : bool
        True when this is the final chunk — triggers final resize,
        meta.json write, and returns None for writer_state.

    Returns
    -------
    tuple[dict[str, Path], dict | None]
        ``({"indices": ..., "counts": ..., "row_offsets": ..., "meta": ...}, state_or_None)``.
        On last chunk the second element is None.
    """
    import zarr

    matrix_root.mkdir(parents=True, exist_ok=True)
    indices_path = matrix_root / f"{release_id}-indices.zarr"
    counts_path = matrix_root / f"{release_id}-counts.zarr"
    row_offsets_path = matrix_root / f"{release_id}-row-offsets.zarr"

    if _writer_state is None:
        # First chunk: create zarr arrays with initial size.
        initial_size = max(len(bundle.indices), 1)

        indices_zarr = zarr.open(str(indices_path), mode="w")
        counts_zarr = zarr.open(str(counts_path), mode="w")
        row_offsets_zarr = zarr.open(str(row_offsets_path), mode="w")

        indices_zarr.create_dataset("indices", shape=(initial_size,), dtype="i4")
        counts_zarr.create_dataset("counts", shape=(initial_size,), dtype="i4")
        row_offsets_zarr.create_dataset("row_offsets", shape=(bundle.row_count + 1,), dtype="i8")

        _writer_state = {
            "indices_zarr": indices_zarr,
            "counts_zarr": counts_zarr,
            "row_offsets_zarr": row_offsets_zarr,
            "indices_path": indices_path,
            "counts_path": counts_path,
            "row_offsets_path": row_offsets_path,
            "global_nnz": 0,
            "row_count": 0,
        }

    # Verify indptr starts at 0 — otherwise the offset logic below
    # (adding current_nnz to all indptr values) would silently corrupt data.
    assert bundle.indptr[0] == 0, f"chunk indptr[0] == {bundle.indptr[0]}, expected 0"

    current_nnz = _writer_state["global_nnz"]
    current_rows = _writer_state["row_count"]
    chunk_nnz = len(bundle.indices)
    chunk_rows = bundle.row_count

    # Write row_offsets (one per row + 1).
    ro_start = current_rows
    ro_end = current_rows + chunk_rows + 1
    if ro_end > _writer_state["row_offsets_zarr"]["row_offsets"].shape[0]:
        _writer_state["row_offsets_zarr"]["row_offsets"].resize((ro_end + 1000,))
    _writer_state["row_offsets_zarr"]["row_offsets"][ro_start:ro_end] = bundle.indptr + current_nnz

    # Ensure indices/counts arrays are large enough for this chunk.
    needed_nnz = current_nnz + chunk_nnz
    if needed_nnz > _writer_state["indices_zarr"]["indices"].shape[0]:
        _writer_state["indices_zarr"]["indices"].resize((needed_nnz + 1000,))
        _writer_state["counts_zarr"]["counts"].resize((needed_nnz + 1000,))

    # Write chunk data via slice assignment.
    _writer_state["indices_zarr"]["indices"][current_nnz:current_nnz + chunk_nnz] = bundle.indices
    _writer_state["counts_zarr"]["counts"][current_nnz:current_nnz + chunk_nnz] = bundle.counts

    # Update accumulated state.
    _writer_state["global_nnz"] = needed_nnz
    _writer_state["row_count"] += chunk_rows

    paths = {
        "indices": indices_path,
        "counts": counts_path,
        "row_offsets": row_offsets_path,
        "meta": matrix_root / f"{release_id}-meta.json",
    }

    if _is_last_chunk:
        # Resize to exact final sizes.
        _writer_state["indices_zarr"]["indices"].resize((_writer_state["global_nnz"],))
        _writer_state["counts_zarr"]["counts"].resize((_writer_state["global_nnz"],))

        # Write meta.json with final dimensions.
        with open(matrix_root / f"{release_id}-meta.json", "w") as f:
            json.dump({
                "release_id": release_id,
                "n_obs": _writer_state["row_count"],
                "nnz": _writer_state["global_nnz"],
                "indices_path": str(indices_path),
                "counts_path": str(counts_path),
                "row_offsets_path": str(row_offsets_path),
            }, f, indent=2)

        return paths, None
    else:
        return paths, _writer_state


def read_zarr_cell(
    indices_path: Path,
    counts_path: Path,
    row_index: int,
    row_offsets_path: Path | None = None,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from Zarr storage.

    Returns ``(expressed_gene_indices, expression_counts, size_factor)``.
    """
    import zarr

    indices_group = zarr.open(str(indices_path), mode="r")
    counts_group = zarr.open(str(counts_path), mode="r")

    if row_offsets_path is not None:
        row_offsets = zarr.open(str(row_offsets_path), mode="r")["row_offsets"][:]
    else:
        # Fallback: use the row_offsets stored inside the indices zarr group
        row_offsets = indices_group["row_offsets"][:]

    start = int(row_offsets[row_index])
    stop = int(row_offsets[row_index + 1])

    gene_indices = tuple(indices_group["indices"][start:stop].astype(np.int32).tolist())
    expr_counts = tuple(counts_group["counts"][start:stop].astype(np.int32).tolist())

    sf = 1.0
    if size_factor_path is not None:
        import pyarrow.parquet as pq

        sf_table = pq.read_table(str(size_factor_path))
        sf = float(sf_table["size_factor"][row_index].as_py())

    return (gene_indices, expr_counts, sf)


# ---------------------------------------------------------------------------
# Aggregate writer (Phase 4)
# ---------------------------------------------------------------------------


def write_zarr_aggregate(
    bundles: list[ChunkBundle],
    matrix_root: Path,
) -> dict[str, Path]:
    """Write aggregate sparse per-cell data in Zarr format.

    This is the ``zarr × aggregate`` thin serializer.
    It consumes an ordered list of ``ChunkBundle`` objects and produces
    a single corpus-scoped Zarr store with flat 1D buffers. Size factors
    are in a separate Parquet sidecar written by the caller after all chunks.

    Parameters
    ----------
    bundles : list[ChunkBundle]
        Chunk bundles in corpus order (one per dataset).
    matrix_root : Path
        Output directory.

    Returns
    -------
    dict[str, Path]
        ``paths_dict`` containing ``{"indices": ..., "counts": ..., "row_offsets": ..., "meta": ...}``.
    """
    import zarr

    matrix_root.mkdir(parents=True, exist_ok=True)

    # Compute total sizes across all bundles.
    total_rows = sum(b.row_count for b in bundles)
    total_nnz = sum(len(b.indices) for b in bundles)

    # Build flat arrays across all datasets.
    all_indices = np.empty(total_nnz, dtype=np.int32)
    all_counts = np.zeros(total_nnz, dtype=np.int32)
    row_offsets = [0]
    global_row_offset = 0

    for bundle in bundles:
        nnz = len(bundle.indices)
        if nnz > 0:
            all_indices[global_row_offset:global_row_offset + nnz] = bundle.indices
            all_counts[global_row_offset:global_row_offset + nnz] = bundle.counts
        global_row_offset += nnz
        row_offsets.append(global_row_offset)

    row_offsets_arr = np.array(row_offsets, dtype=np.int64)

    # Write Zarr arrays.
    indices_path = matrix_root / "aggregated-indices.zarr"
    counts_path = matrix_root / "aggregated-counts.zarr"
    row_offsets_path = matrix_root / "aggregated-row-offsets.zarr"

    indices_zarr = zarr.open(str(indices_path), mode="w")
    counts_zarr = zarr.open(str(counts_path), mode="w")
    row_offsets_zarr = zarr.open(str(row_offsets_path), mode="w")

    indices_zarr.create_dataset("indices", data=all_indices, dtype="i4")
    counts_zarr.create_dataset("counts", data=all_counts, dtype="i4")
    row_offsets_zarr.create_dataset("row_offsets", data=row_offsets_arr, dtype="i8")

    # Write corpus-level metadata JSON.
    meta_path = matrix_root / "aggregated-meta.json"
    dataset_offsets = []
    cum_rows = 0
    for ds_idx, bundle in enumerate(bundles):
        dataset_offsets.append({
            "dataset_index": ds_idx,
            "rows": bundle.row_count,
        })
        cum_rows += bundle.row_count

    with open(meta_path, "w") as f:
        json.dump(
            {
                "total_rows": int(total_rows),
                "total_nnz": int(total_nnz),
                "indices_path": str(indices_path),
                "counts_path": str(counts_path),
                "row_offsets_path": str(row_offsets_path),
                "datasets": dataset_offsets,
            },
            f,
            indent=2,
        )

    return {
        "indices": indices_path,
        "counts": counts_path,
        "row_offsets": row_offsets_path,
        "meta": meta_path,
    }