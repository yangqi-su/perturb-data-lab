from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pytest
import yaml

from perturb_data_lab.cli import _infer_backend_topology_from_corpus
from perturb_data_lab.loaders.expression import (
    AggregateTileDBReader,
    DatasetEntry,
    build_expression_reader,
)
from perturb_data_lab.materializers.backends import build_backend_fn
from perturb_data_lab.materializers.chunk_translation import ChunkBundle


def _bundle_for_rows(
    global_start: int,
    rows: list[tuple[list[int], list[int]]],
) -> ChunkBundle:
    indptr = [0]
    gene_indices: list[int] = []
    counts: list[int] = []
    row_sums: list[float] = []
    expressed_gene_indices = [genes for genes, _ in rows]
    expression_counts = [values for _, values in rows]

    for genes, values in rows:
        gene_indices.extend(genes)
        counts.extend(values)
        indptr.append(indptr[-1] + len(genes))
        row_sums.append(float(sum(values)))

    return ChunkBundle(
        table=pa.table(
            {
                "global_row_index": pa.array(
                    np.arange(global_start, global_start + len(rows), dtype=np.int64),
                    type=pa.int64(),
                ),
                "expressed_gene_indices": pa.array(
                    expressed_gene_indices,
                    type=pa.list_(pa.int32()),
                ),
                "expression_counts": pa.array(
                    expression_counts,
                    type=pa.list_(pa.int32()),
                ),
            }
        ),
        row_sums=np.asarray(row_sums, dtype=np.float64),
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(gene_indices, dtype=np.int32),
        counts=np.asarray(counts, dtype=np.int32),
        row_count=len(rows),
    )


def test_build_backend_fn_exposes_tiledb_aggregate() -> None:
    writer = build_backend_fn("tiledb", "aggregate")
    assert callable(writer)
    assert writer.__name__ == "write_tiledb_aggregate"


def test_infer_backend_topology_defaults_tiledb_to_aggregate(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "corpus-index.yaml").write_text(
        yaml.safe_dump(
            {
                "kind": "corpus-index",
                "contract_version": "0.3.0",
                "corpus_id": "mock-tiledb",
                "global_metadata": {"backend": "tiledb"},
                "datasets": [],
            }
        )
    )

    backend, topology = _infer_backend_topology_from_corpus(corpus_root)
    assert backend == "tiledb"
    assert topology == "aggregate"


def test_tiledb_aggregate_writer_streams_chunks_and_updates_meta(tmp_path: Path) -> None:
    tiledb = pytest.importorskip("tiledb")
    writer = build_backend_fn("tiledb", "aggregate")
    matrix_root = tmp_path / "matrix"

    dataset0_rows = [([0, 2], [5, 7]), ([], []), ([1], [3])]
    dataset1_rows = [([0, 4], [1, 2]), ([2, 5, 6], [4, 5, 6])]

    writer_state = None
    paths = None
    write_plan = [
        ("mock_00", 3, 0, dataset0_rows[:2], False),
        ("mock_00", 3, 2, dataset0_rows[2:], False),
        ("mock_01", 7, 3, dataset1_rows[:1], False),
        ("mock_01", 7, 4, dataset1_rows[1:], True),
    ]

    for dataset_id, local_vocabulary_size, global_start, rows, is_last_chunk in write_plan:
        paths, writer_state = writer(
            bundle=_bundle_for_rows(global_start, rows),
            dataset_id=dataset_id,
            matrix_root=matrix_root,
            _writer_state=writer_state,
            _is_last_chunk=is_last_chunk,
            local_vocabulary_size=local_vocabulary_size,
        )

    assert writer_state is None
    assert paths is not None
    assert paths["cells"].exists()
    assert paths["meta"].exists()

    meta = json.loads(paths["meta"].read_text())
    assert meta["layout_kind"] == "aggregate_native_sparse_tiledb"
    assert meta["layout_version"] == 1
    assert meta["row_index_space"] == "corpus_global"
    assert meta["gene_index_space"] == "dataset_local"
    assert meta["total_rows"] == 5
    assert meta["total_nnz"] == 8
    assert meta["max_observed_local_vocabulary_size"] == 7
    assert meta["consolidation"]["status"] == "not_consolidated"
    assert meta["consolidation"]["fragment_count"] >= 1

    with tiledb.open(str(paths["cells"]), mode="r") as array:
        result = array.query(attrs=["count"], coords=True)[0:5, 0:7]

    triples = sorted(
        zip(
            result["global_row_index"].tolist(),
            result["local_gene_index"].tolist(),
            result["count"].tolist(),
        )
    )
    assert triples == [
        (0, 0, 5),
        (0, 2, 7),
        (2, 1, 3),
        (3, 0, 1),
        (3, 4, 2),
        (4, 2, 4),
        (4, 5, 5),
        (4, 6, 6),
    ]


def test_tiledb_aggregate_writer_rejects_out_of_bounds_local_gene_index(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tiledb")
    writer = build_backend_fn("tiledb", "aggregate")

    with pytest.raises(ValueError, match=r"outside \[0, 4\).+mock_bad"):
        writer(
            bundle=_bundle_for_rows(0, [([4], [1])]),
            dataset_id="mock_bad",
            matrix_root=tmp_path / "matrix",
            _writer_state=None,
            _is_last_chunk=True,
            local_vocabulary_size=4,
        )


def test_build_expression_reader_exposes_aggregate_tiledb_reader(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tiledb")
    writer = build_backend_fn("tiledb", "aggregate")
    matrix_root = tmp_path / "matrix"
    paths, writer_state = writer(
        bundle=_bundle_for_rows(0, [([1, 3], [4, 6])]),
        dataset_id="mock_00",
        matrix_root=matrix_root,
        _writer_state=None,
        _is_last_chunk=True,
        local_vocabulary_size=5,
    )

    assert writer_state is None
    assert paths is not None

    reader = build_expression_reader(
        "tiledb",
        "aggregate",
        [DatasetEntry("mock_00", 0, 1)],
        tiledb_path=str(paths["cells"]),
        tiledb_meta_path=str(paths["meta"]),
    )

    assert isinstance(reader, AggregateTileDBReader)


def test_aggregate_tiledb_reader_preserves_row_order_and_empty_rows(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tiledb")
    writer = build_backend_fn("tiledb", "aggregate")
    matrix_root = tmp_path / "matrix"

    writer_state = None
    paths = None
    for dataset_id, local_vocabulary_size, global_start, rows, is_last_chunk in [
        ("mock_00", 3, 0, [([0, 2], [5, 7]), ([], []), ([1], [3])], False),
        ("mock_01", 7, 3, [([0, 4], [1, 2]), ([2, 5, 6], [4, 5, 6])], True),
    ]:
        paths, writer_state = writer(
            bundle=_bundle_for_rows(global_start, rows),
            dataset_id=dataset_id,
            matrix_root=matrix_root,
            _writer_state=writer_state,
            _is_last_chunk=is_last_chunk,
            local_vocabulary_size=local_vocabulary_size,
        )

    assert paths is not None
    reader = build_expression_reader(
        "tiledb",
        "aggregate",
        [
            DatasetEntry("mock_00", 0, 3),
            DatasetEntry("mock_01", 3, 5),
        ],
        tiledb_path=str(paths["cells"]),
        tiledb_meta_path=str(paths["meta"]),
    )

    batch = reader.read_expression_flat([4, 1, 0, 3])

    assert batch.batch_size == 4
    np.testing.assert_array_equal(batch.global_row_index, [4, 1, 0, 3])
    np.testing.assert_array_equal(batch.row_offsets, [0, 3, 3, 5, 7])
    np.testing.assert_array_equal(batch.row_gene_indices(0), [2, 5, 6])
    np.testing.assert_array_equal(batch.row_counts(0), [4, 5, 6])
    np.testing.assert_array_equal(batch.row_gene_indices(1), [])
    np.testing.assert_array_equal(batch.row_counts(1), [])
    np.testing.assert_array_equal(batch.row_gene_indices(2), [0, 2])
    np.testing.assert_array_equal(batch.row_counts(2), [5, 7])
    np.testing.assert_array_equal(batch.row_gene_indices(3), [0, 4])
    np.testing.assert_array_equal(batch.row_counts(3), [1, 2])
