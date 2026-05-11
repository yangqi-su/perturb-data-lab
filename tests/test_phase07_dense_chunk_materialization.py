from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from perturb_data_lab.materializers.core import _slice_matrix_chunk_as_csr


def test_slice_matrix_chunk_as_csr_converts_dense_numpy_rows() -> None:
    dense = np.array(
        [
            [0.0, 2.0, 0.0, 3.0],
            [1.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )

    chunk = _slice_matrix_chunk_as_csr(dense, 1, 3)

    assert isinstance(chunk, csr_matrix)
    assert chunk.shape == (2, 4)
    np.testing.assert_array_equal(
        chunk.toarray(),
        dense[1:3],
    )


def test_slice_matrix_chunk_as_csr_preserves_sparse_chunks() -> None:
    sparse = csr_matrix(
        np.array(
            [
                [0, 1, 0],
                [2, 0, 3],
                [0, 0, 4],
            ],
            dtype=np.int32,
        )
    )

    chunk = _slice_matrix_chunk_as_csr(sparse, 0, 2)

    assert isinstance(chunk, csr_matrix)
    np.testing.assert_array_equal(chunk.toarray(), sparse[0:2].toarray())
