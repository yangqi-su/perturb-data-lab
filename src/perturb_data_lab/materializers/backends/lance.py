"""Lance backend writers for federated and aggregate materialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa

from ..chunk_translation import ChunkBundle


HEAVY_CELL_SCHEMA = pa.schema(
    [
        pa.field("global_row_index", pa.int64()),
        pa.field("expressed_gene_indices", pa.list_(pa.int32())),
        pa.field("expression_counts", pa.list_(pa.int32())),
    ]
)


def _import_lance():
    try:
        import lance
    except ImportError as exc:
        raise ImportError(
            "lance backend requires the Lance library; install lance in the selected runtime"
        ) from exc
    return lance


def _bundle_to_table(bundle: ChunkBundle) -> pa.Table:
    offsets = pa.array(bundle.indptr.astype("int32", copy=False), type=pa.int32())
    return pa.table(
        {
            "global_row_index": pa.array(bundle.global_row_index, type=pa.int64()),
            "expressed_gene_indices": pa.ListArray.from_arrays(
                offsets,
                pa.array(bundle.indices, type=pa.int32()),
            ),
            "expression_counts": pa.ListArray.from_arrays(
                offsets,
                pa.array(bundle.counts.astype("int32", copy=False), type=pa.int32()),
            ),
        },
        schema=HEAVY_CELL_SCHEMA,
    )


def _write_lance(
    *,
    bundle: ChunkBundle,
    lance_path: Path,
    _writer_state: dict[str, Any] | None,
    _is_last_chunk: bool,
    append_existing: bool,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    lance = _import_lance()
    lance_path.parent.mkdir(parents=True, exist_ok=True)

    if _writer_state is None:
        mode = "append" if append_existing and lance_path.exists() else "create"
        _writer_state = {"lance_path": str(lance_path)}
    else:
        mode = "append"

    lance.write_dataset(
        _bundle_to_table(bundle),
        str(lance_path),
        mode=mode,
        max_rows_per_group=4096,
    )
    return {"cells": lance_path}, None if _is_last_chunk else _writer_state


def write_lance_federated(
    bundle: ChunkBundle,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    return _write_lance(
        bundle=bundle,
        lance_path=matrix_root / f"{dataset_id}.lance",
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
        append_existing=False,
    )


def write_lance_aggregate(
    bundle: ChunkBundle,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    return _write_lance(
        bundle=bundle,
        lance_path=matrix_root / "aggregated-cells.lance",
        _writer_state=_writer_state,
        _is_last_chunk=_is_last_chunk,
        append_existing=True,
    )
