"""Backend adapter: TileDB aggregate writer.

Phase 2 introduces the first TileDB-backed aggregate materializer route.
The layout is a native sparse 2D TileDB array storing
``(global_row_index, local_gene_index) -> count`` with dataset-local gene
coordinates preserved exactly as emitted by ``ChunkBundle``.

Topology: aggregate only (corpus-scoped single store).
Backend name in registry: ``tiledb``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from ..chunk_translation import ChunkBundle


_ROW_TILE_EXTENT = 4096
_GENE_TILE_EXTENT = 1024
_ROW_DOMAIN_MAX = np.int64(np.iinfo(np.int64).max - 1)
_GENE_DOMAIN_MAX = np.int32(np.iinfo(np.int32).max - 1)


def _import_tiledb():
    try:
        import tiledb
    except ImportError as exc:
        raise ImportError(
            "tiledb backend requires the TileDB Python package; "
            "install tiledb in the selected runtime"
        ) from exc
    return tiledb


def _fragment_count(array_uri: Path) -> int:
    fragments_dir = array_uri / "__fragments"
    if not fragments_dir.exists():
        return 0
    return sum(1 for child in fragments_dir.iterdir() if child.is_dir())


def _load_existing_state(meta_path: Path, array_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        raise FileNotFoundError(
            "existing TileDB aggregate array found without aggregated-meta.json: "
            f"{array_path}"
        )
    meta = json.loads(meta_path.read_text())
    return {
        "array_path": array_path,
        "meta_path": meta_path,
        "total_rows": int(meta.get("total_rows", 0)),
        "total_nnz": int(meta.get("total_nnz", 0)),
        "max_observed_local_vocabulary_size": int(
            meta.get("max_observed_local_vocabulary_size", 0)
        ),
    }


def _write_meta(meta_path: Path, state: dict[str, Any]) -> None:
    array_path = Path(state["array_path"])
    fragment_count = _fragment_count(array_path)
    meta = {
        "layout_kind": "aggregate_native_sparse_tiledb",
        "layout_version": 1,
        "array_path": str(array_path),
        "row_index_space": "corpus_global",
        "gene_index_space": "dataset_local",
        "total_rows": int(state["total_rows"]),
        "total_nnz": int(state["total_nnz"]),
        "max_observed_local_vocabulary_size": int(
            state["max_observed_local_vocabulary_size"]
        ),
        "sparse": True,
        "allows_duplicates": False,
        "domain_strategy": "broad_append_safe",
        "domains": {
            "global_row_index": [0, int(_ROW_DOMAIN_MAX)],
            "local_gene_index": [0, int(_GENE_DOMAIN_MAX)],
        },
        "tile_extents": {
            "global_row_index": _ROW_TILE_EXTENT,
            "local_gene_index": _GENE_TILE_EXTENT,
        },
        "consolidation": {
            "status": "not_consolidated",
            "fragment_count": fragment_count,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def _create_sparse_array(tiledb: Any, array_path: Path) -> None:
    dim_row = tiledb.Dim(
        name="global_row_index",
        domain=(0, _ROW_DOMAIN_MAX),
        tile=_ROW_TILE_EXTENT,
        dtype=np.int64,
    )
    dim_gene = tiledb.Dim(
        name="local_gene_index",
        domain=(0, _GENE_DOMAIN_MAX),
        tile=_GENE_TILE_EXTENT,
        dtype=np.int32,
    )
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(dim_row, dim_gene),
        sparse=True,
        attrs=[tiledb.Attr(name="count", dtype=np.int32)],
        allows_duplicates=False,
    )
    tiledb.SparseArray.create(str(array_path), schema)


def _validate_local_gene_indices(
    bundle: ChunkBundle,
    *,
    dataset_id: str,
    local_vocabulary_size: int | None,
) -> None:
    if local_vocabulary_size is None:
        raise ValueError(
            "tiledb aggregate writer requires local_vocabulary_size for "
            f"dataset '{dataset_id}'"
        )
    if local_vocabulary_size <= 0:
        raise ValueError(
            "tiledb aggregate writer requires local_vocabulary_size > 0 for "
            f"dataset '{dataset_id}', got {local_vocabulary_size}"
        )
    if bundle.indices.size == 0:
        return
    min_index = int(bundle.indices.min())
    max_index = int(bundle.indices.max())
    if min_index < 0 or max_index >= local_vocabulary_size:
        raise ValueError(
            "tiledb aggregate writer found local gene indices outside "
            f"[0, {local_vocabulary_size}) for dataset '{dataset_id}'; "
            f"observed min={min_index}, max={max_index}"
        )


def _validate_duplicate_free_coordinates(
    bundle: ChunkBundle,
    *,
    dataset_id: str,
    global_rows: np.ndarray,
) -> None:
    if bundle.indices.size == 0:
        return
    row_lengths = np.diff(bundle.indptr).astype(np.int64, copy=False)
    cursor = 0
    for row_pos, row_length in enumerate(row_lengths):
        row_stop = cursor + int(row_length)
        if row_length > 1:
            row_indices = np.asarray(bundle.indices[cursor:row_stop], dtype=np.int32)
            unique_indices, counts = np.unique(row_indices, return_counts=True)
            duplicate_indices = unique_indices[counts > 1]
            if duplicate_indices.size > 0:
                duplicate_preview = ", ".join(
                    str(int(index)) for index in duplicate_indices[:5]
                )
                raise ValueError(
                    "tiledb aggregate writer requires duplicate-free local gene "
                    "indices per row; "
                    f"dataset '{dataset_id}' global_row_index="
                    f"{int(global_rows[row_pos])} repeats local_gene_index "
                    f"values [{duplicate_preview}]"
                )
        cursor = row_stop


def _global_rows(bundle: ChunkBundle) -> np.ndarray:
    column = bundle.table.column("global_row_index").combine_chunks()
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)


def write_tiledb_aggregate(
    bundle: ChunkBundle,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
    local_vocabulary_size: int | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Path], dict | None]:
    """Stream ChunkBundle data into a native sparse aggregate TileDB array."""
    del kwargs
    tiledb = _import_tiledb()
    matrix_root.mkdir(parents=True, exist_ok=True)

    array_path = matrix_root / "aggregated-cells.tiledb"
    meta_path = matrix_root / "aggregated-meta.json"
    _validate_local_gene_indices(
        bundle,
        dataset_id=dataset_id,
        local_vocabulary_size=local_vocabulary_size,
    )
    global_rows = _global_rows(bundle)
    _validate_duplicate_free_coordinates(
        bundle,
        dataset_id=dataset_id,
        global_rows=global_rows,
    )

    if _writer_state is None:
        if array_path.exists():
            _writer_state = _load_existing_state(meta_path, array_path)
        else:
            _create_sparse_array(tiledb, array_path)
            _writer_state = {
                "array_path": array_path,
                "meta_path": meta_path,
                "total_rows": 0,
                "total_nnz": 0,
                "max_observed_local_vocabulary_size": 0,
            }

    if bundle.indices.size > 0:
        row_lengths = np.diff(bundle.indptr).astype(np.int64, copy=False)
        repeated_rows = np.repeat(global_rows, row_lengths)
        with tiledb.open(str(array_path), mode="w") as array:
            array[repeated_rows, bundle.indices] = {"count": bundle.counts}

    if global_rows.size > 0:
        _writer_state["total_rows"] = max(
            int(_writer_state["total_rows"]),
            int(global_rows[-1]) + 1,
        )
    _writer_state["total_nnz"] = int(_writer_state["total_nnz"]) + int(bundle.indices.size)
    _writer_state["max_observed_local_vocabulary_size"] = max(
        int(_writer_state["max_observed_local_vocabulary_size"]),
        int(local_vocabulary_size),
    )

    _write_meta(meta_path, _writer_state)

    paths = {
        "cells": array_path,
        "meta": meta_path,
    }
    if _is_last_chunk:
        return paths, None
    return paths, _writer_state
