"""Flat-only tests for ``AggregateCsrMemmapReader``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from perturb_data_lab.loaders.expression import (
    AggregateCsrMemmapReader,
    CsrMemmapShardEntry,
    build_expression_reader,
)
from perturb_data_lab.loaders.loaders import ExpressionBatch
from perturb_data_lab.materializers.backends.csr_memmap import CsrMemmapWriter


def _make_cell(*pairs: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    gis, cnts = zip(*pairs)
    return np.array(gis, dtype=np.int32), np.array(cnts, dtype=np.int32)


def _make_cells(*cells: tuple[np.ndarray, np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    gi_list, cnt_list = zip(*cells)
    return list(gi_list), list(cnt_list)


def _build_corpus(
    tmpdir: Path,
    shard_n_cells: int,
    gene_idx_pairs: list[list[tuple[int, int]]],
) -> tuple[Path, list[CsrMemmapShardEntry]]:
    writer = CsrMemmapWriter(output_dir=tmpdir, shard_n_cells=shard_n_cells)
    gi_list, cnt_list = [], []
    for cell_pairs in gene_idx_pairs:
        gi, cnt = _make_cell(*cell_pairs)
        gi_list.append(gi)
        cnt_list.append(cnt)
    writer.append_cells(gi_list, cnt_list)
    manifest_path = writer.finalize()

    import yaml

    with open(manifest_path, "r") as fh:
        doc = yaml.safe_load(fh)
    entries: list[CsrMemmapShardEntry] = []
    for shard in doc["shards"]:
        shard_dir = tmpdir / shard["path"]
        entries.append(
            CsrMemmapShardEntry(
                dataset_id=shard["path"],
                global_start=int(shard["global_start"]),
                global_end=int(shard["global_end"]),
                shard_id=int(shard["shard_id"]),
                shard_path=shard_dir,
                row_offsets_path=shard_dir / "row_offsets.npy",
                gene_indices_path=shard_dir / "gene_indices.npy",
                counts_path=shard_dir / "counts.npy",
                n_cells=int(shard["n_cells"]),
            )
        )
    entries.sort(key=lambda e: e.global_start)
    return tmpdir, entries


def _expected_cell_data(
    cell_data: list[list[tuple[int, int]]], global_idx: int
) -> tuple[list[int], list[int]]:
    pairs = cell_data[global_idx]
    if not pairs:
        return [], []
    gis, cnts = zip(*pairs)
    return list(gis), list(cnts)


def _assert_batch(batch: ExpressionBatch, indices: list[int]) -> None:
    assert isinstance(batch, ExpressionBatch)
    assert batch.batch_size == len(indices)
    np.testing.assert_array_equal(batch.global_row_index, indices)
    assert len(batch.row_offsets) == len(indices) + 1
    assert batch.row_offsets[0] == 0
    assert batch.row_offsets[-1] == len(batch.expressed_gene_indices)
    assert len(batch.expressed_gene_indices) == len(batch.expression_counts)


@pytest.fixture
def corpus_5_cells_2_shards():
    cell_data = [
        [(0, 10), (1, 20)],
        [(2, 30), (3, 40)],
        [(4, 50)],
        [(1, 60), (5, 70)],
        [],
    ]
    with tempfile.TemporaryDirectory() as td:
        yield _build_corpus(Path(td), shard_n_cells=3, gene_idx_pairs=cell_data), cell_data


@pytest.fixture
def corpus_10_cells_1_shard():
    cell_data = [
        [(0, 1), (1, 2)],
        [(2, 3)],
        [(0, 4), (3, 5), (5, 6)],
        [(1, 7)],
        [(2, 8), (4, 9)],
        [(0, 10), (5, 11)],
        [(3, 12), (4, 13), (6, 14)],
        [(1, 15)],
        [(2, 16), (5, 17)],
        [(0, 18), (3, 19), (4, 20)],
    ]
    with tempfile.TemporaryDirectory() as td:
        yield _build_corpus(Path(td), shard_n_cells=10, gene_idx_pairs=cell_data), cell_data


class TestAggregateCsrMemmapReaderFlatOnly:
    def test_basic_reads(self, corpus_5_cells_2_shards):
        (_corpus_root, entries), cell_data = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        indices = [0, 1, 3, 4]
        batch = reader.read_expression_flat(indices)
        _assert_batch(batch, indices)
        for row_position, global_idx in enumerate(indices):
            expected_gi, expected_cnt = _expected_cell_data(cell_data, global_idx)
            np.testing.assert_array_equal(batch.row_gene_indices(row_position), expected_gi)
            np.testing.assert_array_equal(batch.row_counts(row_position), expected_cnt)

    def test_order_preservation(self, corpus_10_cells_1_shard):
        (_corpus_root, entries), cell_data = corpus_10_cells_1_shard
        reader = AggregateCsrMemmapReader(entries)
        indices = [9, 5, 2, 0, 9]
        batch = reader.read_expression_flat(indices)
        _assert_batch(batch, indices)
        for row_position, global_idx in enumerate(indices):
            expected_gi, expected_cnt = _expected_cell_data(cell_data, global_idx)
            np.testing.assert_array_equal(batch.row_gene_indices(row_position), expected_gi)
            np.testing.assert_array_equal(batch.row_counts(row_position), expected_cnt)

    def test_empty_input(self, corpus_5_cells_2_shards):
        (_corpus_root, entries), _cell_data = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        batch = reader.read_expression_flat([])
        _assert_batch(batch, [])

    def test_out_of_range(self, corpus_5_cells_2_shards):
        (_corpus_root, entries), _cell_data = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        with pytest.raises(IndexError):
            reader.read_expression_flat([-1])
        with pytest.raises(IndexError):
            reader.read_expression_flat([5])

    def test_factory_integration(self, corpus_5_cells_2_shards):
        (_corpus_root, entries), _cell_data = corpus_5_cells_2_shards
        reader = build_expression_reader("csr_memmap", "aggregate", list(entries))
        assert isinstance(reader, AggregateCsrMemmapReader)
        batch = reader.read_expression_flat([0, 3])
        _assert_batch(batch, [0, 3])

    def test_factory_rejects_federated_csr(self, corpus_5_cells_2_shards):
        (_corpus_root, entries), _cell_data = corpus_5_cells_2_shards
        with pytest.raises(ValueError, match="only supports aggregate"):
            build_expression_reader("csr_memmap", "federated", list(entries))

    def test_factory_rejects_unknown_kwargs(self, corpus_5_cells_2_shards):
        (_corpus_root, entries), _cell_data = corpus_5_cells_2_shards
        with pytest.raises(TypeError, match="Unexpected keyword"):
            build_expression_reader(
                "csr_memmap",
                "aggregate",
                list(entries),
                lance_path="/nonexistent",
            )

    def test_shard_opening_is_lazy(self, corpus_5_cells_2_shards):
        (_corpus_root, entries), _cell_data = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        assert len(reader._mmaps) == 0
        reader.read_expression_flat([0])
        assert len(reader._mmaps) == 1
        reader.read_expression_flat([3])
        assert len(reader._mmaps) == 2


class TestRoundTrip:
    def test_round_trip_multi_shard(self):
        cell_data = [
            [(0, 1), (1, 2)],
            [(2, 3)],
            [(0, 4), (3, 5), (5, 6)],
            [(1, 7)],
            [(2, 8), (4, 9)],
            [(0, 10), (5, 11)],
            [(3, 12), (4, 13), (6, 14)],
        ]
        with tempfile.TemporaryDirectory() as td:
            (tmpdir, entries) = _build_corpus(Path(td), shard_n_cells=3, gene_idx_pairs=cell_data)
            reader = AggregateCsrMemmapReader(entries)
            indices = [6, 0, 4, 2, 5, 1, 3]
            batch = reader.read_expression_flat(indices)
            _assert_batch(batch, indices)
            for row_position, global_idx in enumerate(indices):
                expected_gi, expected_cnt = _expected_cell_data(cell_data, global_idx)
                np.testing.assert_array_equal(batch.row_gene_indices(row_position), expected_gi)
                np.testing.assert_array_equal(batch.row_counts(row_position), expected_cnt)

    def test_round_trip_empty_cells(self):
        cell_data = [[(0, 5)], [], [(1, 10), (2, 15)], [], [(3, 20)]]
        with tempfile.TemporaryDirectory() as td:
            (_tmpdir, entries) = _build_corpus(Path(td), shard_n_cells=10, gene_idx_pairs=cell_data)
            reader = AggregateCsrMemmapReader(entries)
            batch = reader.read_expression_flat([0, 1, 2, 3, 4])
            _assert_batch(batch, [0, 1, 2, 3, 4])
            np.testing.assert_array_equal(batch.row_gene_indices(1), [])
            np.testing.assert_array_equal(batch.row_gene_indices(3), [])
