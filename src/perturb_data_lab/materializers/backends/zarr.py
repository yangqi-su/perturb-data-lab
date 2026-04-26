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
import json

import numpy as np

from ..chunk_translation import ChunkBundle


def write_zarr_federated(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
) -> dict[str, Path]:
    """Write a ``ChunkBundle`` as Zarr flat 1D buffers.

    This is the ``zarr × federated`` thin serializer.
    It accepts a ``ChunkBundle`` and writes flat 1D Zarr arrays for
    indices, counts, and row_offsets.

    Parameters
    ----------
    bundle : ChunkBundle
        The translated chunk bundle from ``_translate_chunk()``.
    release_id : str
        Release identifier used for output file naming.
    matrix_root : Path
        Output directory for matrix artifacts.

    Returns a dict with keys: ``{"indices": ..., "counts": ..., "row_offsets": ..., "meta": ...}``.
    """
    import zarr

    matrix_root.mkdir(parents=True, exist_ok=True)

    indices_path = matrix_root / f"{release_id}-indices.zarr"
    counts_path = matrix_root / f"{release_id}-counts.zarr"
    row_offsets_path = matrix_root / f"{release_id}-row-offsets.zarr"

    # Build flat arrays from the bundle's CSR buffers.
    all_indices = bundle.indices
    all_counts = bundle.counts
    row_offsets_arr = bundle.indptr

    # Write as zarr 1D arrays.
    indices_zarr = zarr.open(str(indices_path), mode="w")
    counts_zarr = zarr.open(str(counts_path), mode="w")
    row_offsets_zarr = zarr.open(str(row_offsets_path), mode="w")

    indices_zarr.create_dataset("indices", data=all_indices, dtype="i4")
    counts_zarr.create_dataset("counts", data=all_counts, dtype="i4")
    row_offsets_zarr.create_dataset("row_offsets", data=row_offsets_arr, dtype="i8")

    # Write metadata JSON.
    meta_path = matrix_root / f"{release_id}-meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "release_id": release_id,
                "n_obs": bundle.row_count,
                "size_factor_path": str(matrix_root / f"{release_id}-size-factor.zarr"),
                "indices_path": str(indices_path),
                "counts_path": str(counts_path),
                "row_offsets_path": str(row_offsets_path),
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
) -> tuple[dict[str, Path], list[np.ndarray]]:
    """Write aggregate sparse per-cell data in Zarr format.

    This is the ``zarr × aggregate`` thin serializer.
    It consumes an ordered list of ``ChunkBundle`` objects and produces
    a single corpus-scoped Zarr store with flat 1D buffers.

    Parameters
    ----------
    bundles : list[ChunkBundle]
        Chunk bundles in corpus order (one per dataset).
    matrix_root : Path
        Output directory.

    Returns
    -------
    tuple[dict[str, Path], list[np.ndarray]]
        ``(paths_dict, size_factors_out_list)``.
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

    size_factors_out_list = [b.size_factors for b in bundles]

    return (
        {
            "indices": indices_path,
            "counts": counts_path,
            "row_offsets": row_offsets_path,
            "meta": meta_path,
        },
        size_factors_out_list,
    )