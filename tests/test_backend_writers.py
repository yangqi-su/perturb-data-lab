from __future__ import annotations

from pathlib import Path

import numpy as np

from perturb_data_lab.materializers.backends.zarr import write_zarr_aggregate
from perturb_data_lab.materializers.chunk_translation import ChunkBundle


def _bundle(
    global_start: int,
    indptr: list[int],
    indices: list[int],
    counts: list[int],
) -> ChunkBundle:
    return ChunkBundle(
        global_row_index=np.arange(global_start, global_start + len(indptr) - 1, dtype=np.int64),
        row_sums=np.asarray(
            [sum(counts[indptr[i] : indptr[i + 1]]) for i in range(len(indptr) - 1)],
            dtype=np.float64,
        ),
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int32),
        counts=np.asarray(counts, dtype=np.int32),
        row_count=len(indptr) - 1,
    )


def test_zarr_aggregate_appends_after_previous_cli_finalization(tmp_path: Path) -> None:
    first = _bundle(0, [0, 2, 3], [1, 4, 2], [7, 8, 9])
    write_zarr_aggregate(
        bundle=first,
        dataset_id="ds0",
        matrix_root=tmp_path,
        _writer_state=None,
        _is_last_chunk=True,
    )

    second = _bundle(2, [0, 1, 3], [0, 3, 5], [2, 4, 6])
    paths, state = write_zarr_aggregate(
        bundle=second,
        dataset_id="ds1",
        matrix_root=tmp_path,
        _writer_state=None,
        _is_last_chunk=True,
    )

    import zarr

    assert state is None
    np.testing.assert_array_equal(
        zarr.open(str(paths["row_offsets"]), mode="r")["row_offsets"][:],
        np.asarray([0, 2, 3, 4, 6], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        zarr.open(str(paths["indices"]), mode="r")["indices"][:],
        np.asarray([1, 4, 2, 0, 3, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        zarr.open(str(paths["counts"]), mode="r")["counts"][:],
        np.asarray([7, 8, 9, 2, 4, 6], dtype=np.int32),
    )
