"""Zarr backend writers for federated and aggregate materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..chunk_translation import ChunkBundle


def _open_zarr_state(
    paths: dict[str, Path],
    bundle: ChunkBundle,
    *,
    append_existing: bool,
) -> dict[str, Any]:
    import zarr

    existing = [paths[name].exists() for name in ("indices", "counts", "row_offsets")]
    if append_existing and any(existing):
        if not all(existing):
            raise FileNotFoundError(
                f"incomplete aggregate Zarr artifacts in {paths['indices'].parent}"
            )
        indices_zarr = zarr.open(str(paths["indices"]), mode="a")
        counts_zarr = zarr.open(str(paths["counts"]), mode="a")
        row_offsets_zarr = zarr.open(str(paths["row_offsets"]), mode="a")
        indices = indices_zarr["indices"]
        counts = counts_zarr["counts"]
        row_offsets = row_offsets_zarr["row_offsets"]
        current_nnz = int(indices.shape[0])
        current_rows = int(row_offsets.shape[0]) - 1
        if counts.shape[0] != current_nnz or current_rows < 0:
            raise ValueError("aggregate Zarr artifacts have inconsistent shapes")
        if int(row_offsets[-1]) != current_nnz:
            raise ValueError("aggregate Zarr row_offsets[-1] does not match stored nnz")
    else:
        initial_nnz = max(len(bundle.indices), 1)
        indices_zarr = zarr.open(str(paths["indices"]), mode="w")
        counts_zarr = zarr.open(str(paths["counts"]), mode="w")
        row_offsets_zarr = zarr.open(str(paths["row_offsets"]), mode="w")
        indices_zarr.create_dataset("indices", shape=(initial_nnz,), dtype="i4")
        counts_zarr.create_dataset("counts", shape=(initial_nnz,), dtype="i4")
        row_offsets_zarr.create_dataset(
            "row_offsets", shape=(bundle.row_count + 1,), dtype="i8"
        )
        current_nnz = 0
        current_rows = 0

    return {
        "indices_zarr": indices_zarr,
        "counts_zarr": counts_zarr,
        "row_offsets_zarr": row_offsets_zarr,
        "global_nnz": current_nnz,
        "row_count": current_rows,
    }


def _write_zarr(
    *,
    bundle: ChunkBundle,
    paths: dict[str, Path],
    _writer_state: dict[str, Any] | None,
    _is_last_chunk: bool,
    append_existing: bool,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    paths["indices"].parent.mkdir(parents=True, exist_ok=True)
    if _writer_state is None:
        _writer_state = _open_zarr_state(paths, bundle, append_existing=append_existing)

    assert bundle.indptr[0] == 0, f"chunk indptr[0] == {bundle.indptr[0]}, expected 0"
    current_nnz = _writer_state["global_nnz"]
    current_rows = _writer_state["row_count"]
    chunk_nnz = len(bundle.indices)
    chunk_rows = bundle.row_count
    if append_existing and chunk_rows and int(bundle.global_row_index[0]) != current_rows:
        raise ValueError(
            "aggregate Zarr append expected next global row "
            f"{current_rows}, got {int(bundle.global_row_index[0])}"
        )

    row_offsets = _writer_state["row_offsets_zarr"]["row_offsets"]
    ro_end = current_rows + chunk_rows + 1
    if ro_end > row_offsets.shape[0]:
        row_offsets.resize((ro_end,))
    row_offsets[current_rows:ro_end] = bundle.indptr + current_nnz

    needed_nnz = current_nnz + chunk_nnz
    indices = _writer_state["indices_zarr"]["indices"]
    counts = _writer_state["counts_zarr"]["counts"]
    if needed_nnz > indices.shape[0]:
        indices.resize((needed_nnz,))
        counts.resize((needed_nnz,))
    indices[current_nnz:needed_nnz] = bundle.indices
    counts[current_nnz:needed_nnz] = bundle.counts

    _writer_state["global_nnz"] = needed_nnz
    _writer_state["row_count"] = current_rows + chunk_rows

    if _is_last_chunk:
        indices.resize((_writer_state["global_nnz"],))
        counts.resize((_writer_state["global_nnz"],))
        row_offsets.resize((_writer_state["row_count"] + 1,))
        return paths, None
    return paths, _writer_state


def write_zarr_federated(
    bundle: ChunkBundle,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    paths = {
        "indices": matrix_root / f"{dataset_id}-indices.zarr",
        "counts": matrix_root / f"{dataset_id}-counts.zarr",
        "row_offsets": matrix_root / f"{dataset_id}-row-offsets.zarr",
    }
    return _write_zarr(
        bundle=bundle,
        paths=paths,
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
        append_existing=False,
    )


def write_zarr_aggregate(
    bundle: ChunkBundle,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    paths = {
        "indices": matrix_root / "aggregated-indices.zarr",
        "counts": matrix_root / "aggregated-counts.zarr",
        "row_offsets": matrix_root / "aggregated-row-offsets.zarr",
    }
    return _write_zarr(
        bundle=bundle,
        paths=paths,
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
        append_existing=True,
    )
