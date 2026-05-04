"""Tests for AggregateCsrMemmapReader and CsrMemmapShardEntry.

Validates:
- Correct expression data read from CSR memmap shards
- Order-preserving reads (sorted and unsorted indices)
- Fast path read_expression_flat() producing correct ExpressionBatch
- Fast path equivalence with legacy ExpressionRow path
- Multi-shard cross-boundary reads
- Edge cases: empty indices, single cell, empty cells (zero genes)
- Error handling: out-of-range indices
- Round-trip correctness from CsrMemmapWriter to AggregateCsrMemmapReader
- build_expression_reader factory integration
"""

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


# ---------------------------------------------------------------------------
# Toy data helpers
# ---------------------------------------------------------------------------


def _make_cell(*pairs: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Create one cell's data from (gene_index, count) pairs."""
    if not pairs:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
    gis, cnts = zip(*pairs)
    return np.array(gis, dtype=np.int32), np.array(cnts, dtype=np.int32)


def _make_cells(
    *cells: tuple[np.ndarray, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Unzip a sequence of (gene_indices, counts) pairs into two lists."""
    gi_list, cnt_list = zip(*cells)
    return list(gi_list), list(cnt_list)


def _build_corpus(
    tmpdir: Path,
    shard_n_cells: int,
    gene_idx_pairs: list[list[tuple[int, int]]],
) -> tuple[Path, list[CsrMemmapShardEntry]]:
    """Write a CSR corpus from toy data and return (corpus_root, entries).

    Each element of *gene_idx_pairs* is a list of (gene_index, count)
    tuples representing one cell.
    """
    writer = CsrMemmapWriter(output_dir=tmpdir, shard_n_cells=shard_n_cells)
    gi_list, cnt_list = [], []
    for cell_pairs in gene_idx_pairs:
        gi, cnt = _make_cell(*cell_pairs)
        gi_list.append(gi)
        cnt_list.append(cnt)
    writer.append_cells(gi_list, cnt_list)
    manifest_path = writer.finalize()

    # Build entries from the manifest
    import yaml

    with open(manifest_path, "r") as fh:
        doc = yaml.safe_load(fh)
    entries: list[CsrMemmapShardEntry] = []
    for s in doc["shards"]:
        shard_dir = tmpdir / s["path"]
        entries.append(
            CsrMemmapShardEntry(
                dataset_id=s["path"],
                global_start=int(s["global_start"]),
                global_end=int(s["global_end"]),
                shard_id=int(s["shard_id"]),
                shard_path=shard_dir,
                row_offsets_path=shard_dir / "row_offsets.npy",
                gene_indices_path=shard_dir / "gene_indices.npy",
                counts_path=shard_dir / "counts.npy",
                n_cells=int(s["n_cells"]),
            )
        )
    entries.sort(key=lambda e: e.global_start)
    return tmpdir, entries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus_5_cells_2_shards():
    """Corpus with 5 cells across 2 shards (shard_n_cells=3).

    Cell data:
      cell 0: genes [0,,1] counts [10, 20]
      cell 1: genes [2, 3] counts [30, 40]
      cell 2: genes [4]    counts [50]          ← shard 0 (cells 0–2)
      cell 3: genes [1, 5] counts [60, 70]
      cell 4: genes []     counts []            ← shard 1 (cells 3–4, cell 4 empty)
    """
    cell_data = [
        [(0, 10), (1, 20)],
        [(2, 30), (3, 40)],
        [(4, 50)],
        [(1, 60), (5, 70)],
        [],  # empty cell
    ]
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        yield _build_corpus(tmpdir, shard_n_cells=3, gene_idx_pairs=cell_data)


@pytest.fixture
def corpus_10_cells_1_shard():
    """Corpus with 10 cells in a single shard."""
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
        tmpdir = Path(td)
        yield _build_corpus(tmpdir, shard_n_cells=10, gene_idx_pairs=cell_data)


# ---------------------------------------------------------------------------
# Helper: get expected gene indices and counts for a cell
# ---------------------------------------------------------------------------


def _expected_cell_data(
    cell_data: list[list[tuple[int, int]]], global_idx: int
) -> tuple[list[int], list[int]]:
    """Return (gene_indices, counts) for a cell from the cell_data list."""
    if global_idx < 0 or global_idx >= len(cell_data):
        raise IndexError(f"global_idx {global_idx} out of range")
    pairs = cell_data[global_idx]
    if not pairs:
        return [], []
    gis, cnts = zip(*pairs)
    return list(gis), list(cnts)


# ---------------------------------------------------------------------------
# Test class: Basic Reads
# ---------------------------------------------------------------------------


class TestBasicReads:
    """Tests for reading single cells and small batches."""

    def test_read_single_cell_first(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        rows = reader.read_expression([0])
        assert len(rows) == 1
        assert rows[0].global_row_index == 0
        np.testing.assert_array_equal(
            rows[0].expressed_gene_indices, [0, 1]
        )
        np.testing.assert_array_equal(
            rows[0].expression_counts, [10, 20]
        )

    def test_read_single_cell_empty(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        rows = reader.read_expression([4])  # empty cell
        assert len(rows) == 1
        assert rows[0].global_row_index == 4
        assert len(rows[0].expressed_gene_indices) == 0
        assert len(rows[0].expression_counts) == 0

    def test_read_multiple_cells_same_shard(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        rows = reader.read_expression([0, 1, 2])
        assert len(rows) == 3
        assert [r.global_row_index for r in rows] == [0, 1, 2]
        np.testing.assert_array_equal(
            rows[0].expressed_gene_indices, [0, 1]
        )
        np.testing.assert_array_equal(
            rows[1].expressed_gene_indices, [2, 3]
        )
        np.testing.assert_array_equal(
            rows[2].expressed_gene_indices, [4]
        )

    def test_read_cross_shard(self, corpus_5_cells_2_shards):
        """Read cells spanning two shards."""
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        rows = reader.read_expression([1, 3])  # cell 1 in shard 0, cell 3 in shard 1
        assert len(rows) == 2
        assert rows[0].global_row_index == 1
        assert rows[1].global_row_index == 3
        np.testing.assert_array_equal(
            rows[0].expressed_gene_indices, [2, 3]
        )
        np.testing.assert_array_equal(
            rows[1].expressed_gene_indices, [1, 5]
        )

    def test_read_empty_indices(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        rows = reader.read_expression([])
        assert rows == []


# ---------------------------------------------------------------------------
# Test class: Order Preservation
# ---------------------------------------------------------------------------


class TestOrderPreservation:
    """Tests that output order matches input order exactly."""

    def test_sorted_order(self, corpus_10_cells_1_shard):
        _, entries = corpus_10_cells_1_shard
        reader = AggregateCsrMemmapReader(entries)
        indices = [0, 2, 5, 9]
        rows = reader.read_expression(indices)
        assert [r.global_row_index for r in rows] == indices

    def test_reverse_order(self, corpus_10_cells_1_shard):
        _, entries = corpus_10_cells_1_shard
        reader = AggregateCsrMemmapReader(entries)
        indices = [9, 5, 2, 0]
        rows = reader.read_expression(indices)
        assert [r.global_row_index for r in rows] == indices

    def test_scrambled_order(self, corpus_10_cells_1_shard):
        _, entries = corpus_10_cells_1_shard
        reader = AggregateCsrMemmapReader(entries)
        indices = [3, 7, 0, 9, 1, 5]
        rows = reader.read_expression(indices)
        assert [r.global_row_index for r in rows] == indices

    def test_duplicate_indices(self, corpus_10_cells_1_shard):
        """Read the same cell twice — each should return correct data."""
        _, entries = corpus_10_cells_1_shard
        reader = AggregateCsrMemmapReader(entries)
        indices = [3, 3, 5, 5]
        rows = reader.read_expression(indices)
        assert len(rows) == 4
        assert [r.global_row_index for r in rows] == indices
        for i in range(4):
            assert rows[i].global_row_index == indices[i]

    def test_cross_shard_order(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        # Indices jump between shards and are in non-monotonic order
        indices = [4, 0, 3, 1, 2]
        rows = reader.read_expression(indices)
        assert [r.global_row_index for r in rows] == indices


# ---------------------------------------------------------------------------
# Test class: Fast Path (read_expression_flat)
# ---------------------------------------------------------------------------


class TestFastPath:
    """Tests for read_expression_flat() correctness."""

    def test_single_cell_flat(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        batch = reader.read_expression_flat([0])
        assert batch.batch_size == 1
        np.testing.assert_array_equal(batch.global_row_index, [0])
        np.testing.assert_array_equal(batch.row_offsets, [0, 2])
        np.testing.assert_array_equal(
            batch.expressed_gene_indices, [0, 1]
        )
        np.testing.assert_array_equal(
            batch.expression_counts, [10, 20]
        )

    def test_multiple_cells_flat(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        batch = reader.read_expression_flat([0, 1, 2])
        assert batch.batch_size == 3
        np.testing.assert_array_equal(
            batch.global_row_index, [0, 1, 2]
        )
        # row_offsets: [0, 2, 4, 5]
        np.testing.assert_array_equal(
            batch.row_offsets, [0, 2, 4, 5]
        )
        np.testing.assert_array_equal(
            batch.expressed_gene_indices, [0, 1, 2, 3, 4]
        )
        np.testing.assert_array_equal(
            batch.expression_counts, [10, 20, 30, 40, 50]
        )

    def test_flat_cross_shard(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        batch = reader.read_expression_flat([1, 3, 4])
        assert batch.batch_size == 3
        np.testing.assert_array_equal(
            batch.global_row_index, [1, 3, 4]
        )
        # cell 1: 2 genes  → offsets [0, 2]
        # cell 3: 2 genes  → offsets [2, 4]
        # cell 4: 0 genes  → offsets [4, 4]
        np.testing.assert_array_equal(
            batch.row_offsets, [0, 2, 4, 4]
        )
        np.testing.assert_array_equal(
            batch.expressed_gene_indices, [2, 3, 1, 5]
        )
        np.testing.assert_array_equal(
            batch.expression_counts, [30, 40, 60, 70]
        )

    def test_flat_empty_indices(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        batch = reader.read_expression_flat([])
        assert batch.batch_size == 0
        assert len(batch.global_row_index) == 0
        assert len(batch.row_offsets) == 1  # always [0]
        np.testing.assert_array_equal(batch.row_offsets, [0])
        assert len(batch.expressed_gene_indices) == 0
        assert len(batch.expression_counts) == 0

    def test_flat_order_preserved(self, corpus_10_cells_1_shard):
        """Flat read must preserve unordered input indices."""
        _, entries = corpus_10_cells_1_shard
        reader = AggregateCsrMemmapReader(entries)
        indices = [5, 0, 9, 2]
        batch = reader.read_expression_flat(indices)
        np.testing.assert_array_equal(batch.global_row_index, indices)


# ---------------------------------------------------------------------------
# Test class: Fast Path vs Legacy Row Path Equivalence
# ---------------------------------------------------------------------------


class TestFastPathEquivalence:
    """Verify that read_expression_flat() produces the same expression data
    as read_expression() for all test patterns."""

    def _check_equivalence(
        self, reader: AggregateCsrMemmapReader, indices: list[int]
    ) -> None:
        """Check that flat and row paths return identical expression data."""
        rows = reader.read_expression(indices)
        batch = reader.read_expression_flat(indices)

        assert batch.batch_size == len(indices)

        # Reconstruct flat arrays from rows and compare
        n = len(indices)
        expected_offsets = np.zeros(n + 1, dtype=np.int64)
        expected_gi_parts = []
        expected_cnt_parts = []
        for i, row in enumerate(rows):
            expected_gi_parts.append(row.expressed_gene_indices)
            expected_cnt_parts.append(row.expression_counts)
            expected_offsets[i + 1] = (
                expected_offsets[i] + len(row.expressed_gene_indices)
            )

        np.testing.assert_array_equal(batch.row_offsets, expected_offsets)
        np.testing.assert_array_equal(
            batch.expressed_gene_indices,
            np.concatenate(expected_gi_parts) if expected_gi_parts
            else np.array([], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            batch.expression_counts,
            np.concatenate(expected_cnt_parts) if expected_cnt_parts
            else np.array([], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            batch.global_row_index,
            np.array(indices, dtype=np.int64),
        )

    def test_equivalence_single(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        self._check_equivalence(reader, [0])
        self._check_equivalence(reader, [4])  # empty cell

    def test_equivalence_multi_same_shard(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        self._check_equivalence(reader, [0, 1, 2])

    def test_equivalence_cross_shard(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        self._check_equivalence(reader, [0, 1, 3, 4])  # spans both shards

    def test_equivalence_scrambled(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        self._check_equivalence(reader, [4, 0, 3, 1, 2])

    def test_equivalence_empty(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        self._check_equivalence(reader, [])

    def test_equivalence_large(self, corpus_10_cells_1_shard):
        _, entries = corpus_10_cells_1_shard
        reader = AggregateCsrMemmapReader(entries)
        self._check_equivalence(reader, [0, 2, 4, 6, 8])
        self._check_equivalence(reader, [9, 0, 5, 1, 7, 3])


# ---------------------------------------------------------------------------
# Test class: Error Handling
# ---------------------------------------------------------------------------


class TestErrors:
    """Tests for expected error conditions."""

    def test_out_of_range_negative(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        with pytest.raises(IndexError):
            reader.read_expression([-1])

    def test_out_of_range_too_large(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        with pytest.raises(IndexError):
            reader.read_expression([5])  # 5 cells (0–4)

    def test_out_of_range_flat(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        with pytest.raises(IndexError):
            reader.read_expression_flat([999])


# ---------------------------------------------------------------------------
# Test class: Row Slicing (ExpressionBatch) in Fast Path
# ---------------------------------------------------------------------------


class TestRowSlicing:
    """Tests that ExpressionBatch row_slice and row accessors work correctly."""

    def test_row_slice_single_row(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        batch = reader.read_expression_flat([0, 2])
        # Row 0 (cells 0)
        np.testing.assert_array_equal(batch.row_gene_indices(0), [0, 1])
        np.testing.assert_array_equal(batch.row_counts(0), [10, 20])
        # Row 2 (cell 2)
        np.testing.assert_array_equal(batch.row_gene_indices(1), [4])
        np.testing.assert_array_equal(batch.row_counts(1), [50])

    def test_row_slice_cross_shard(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        batch = reader.read_expression_flat([1, 3, 4])
        # Row 0: cell 1 → genes [2, 3] counts [30, 40]
        np.testing.assert_array_equal(batch.row_gene_indices(0), [2, 3])
        np.testing.assert_array_equal(batch.row_counts(0), [30, 40])
        # Row 1: cell 3 → genes [1, 5] counts [60, 70]
        np.testing.assert_array_equal(batch.row_gene_indices(1), [1, 5])
        np.testing.assert_array_equal(batch.row_counts(1), [60, 70])
        # Row 2: cell 4 → empty
        np.testing.assert_array_equal(batch.row_gene_indices(2), [])
        np.testing.assert_array_equal(batch.row_counts(2), [])


# ---------------------------------------------------------------------------
# Test class: build_expression_reader Factory Integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    """Tests that build_expression_reader() creates the correct reader."""

    def test_factory_creates_csr_reader(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = build_expression_reader(
            "csr_memmap", "aggregate", list(entries)
        )
        assert isinstance(reader, AggregateCsrMemmapReader)
        # Verify it can read
        rows = reader.read_expression([0, 3])
        assert len(rows) == 2

    def test_factory_rejects_federated_csr(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        with pytest.raises(ValueError, match="only supports aggregate"):
            build_expression_reader("csr_memmap", "federated", list(entries))

    def test_factory_with_cache_config(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            reader = build_expression_reader(
                "csr_memmap", "aggregate", list(entries),
                cache_config={
                    "enabled": True,
                    "cache_root": td,
                    "max_bytes": 100_000_000,
                },
            )
        assert isinstance(reader, AggregateCsrMemmapReader)
        # Phase 4: cache is constructed when enabled and config is valid
        assert reader._cache is not None
        assert reader._cache_config is not None

    def test_factory_rejects_unknown_kwargs(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        with pytest.raises(TypeError, match="Unexpected keyword"):
            build_expression_reader(
                "csr_memmap", "aggregate", list(entries),
                lance_path="/nonexistent",
            )


# ---------------------------------------------------------------------------
# Test class: Round-Trip from Writer
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Verify that data written by CsrMemmapWriter is correctly read back."""

    def test_round_trip_single_cell(self):
        cell_data = [[(0, 5), (1, 10), (3, 15)]]
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            writer = CsrMemmapWriter(output_dir=tmpdir, shard_n_cells=10)
            gi_list, cnt_list = _make_cells(_make_cell((0, 5), (1, 10), (3, 15)))
            writer.append_cells(gi_list, cnt_list)
            writer.finalize()

            # Build entries
            import yaml

            manifest_path = tmpdir / "csr-corpus-manifest.yaml"
            with open(manifest_path, "r") as fh:
                doc = yaml.safe_load(fh)
            entries: list[CsrMemmapShardEntry] = []
            for s in doc["shards"]:
                shard_dir = tmpdir / s["path"]
                entries.append(
                    CsrMemmapShardEntry(
                        dataset_id=s["path"],
                        global_start=int(s["global_start"]),
                        global_end=int(s["global_end"]),
                        shard_id=int(s["shard_id"]),
                        shard_path=shard_dir,
                        row_offsets_path=shard_dir / "row_offsets.npy",
                        gene_indices_path=shard_dir / "gene_indices.npy",
                        counts_path=shard_dir / "counts.npy",
                        n_cells=int(s["n_cells"]),
                    )
                )

            reader = AggregateCsrMemmapReader(entries)
            rows = reader.read_expression([0])
            assert rows[0].global_row_index == 0
            np.testing.assert_array_equal(
                rows[0].expressed_gene_indices, [0, 1, 3]
            )
            np.testing.assert_array_equal(
                rows[0].expression_counts, [5, 10, 15]
            )

    def test_round_trip_multi_shard(self):
        """7 cells across 3 shards (cap=3)."""
        cell_data = [
            [(0, 1), (1, 2)],        # cell 0
            [(2, 3)],                 # cell 1
            [(0, 4), (3, 5), (5, 6)], # cell 2 → shard 0 end
            [(1, 7)],                 # cell 3
            [(2, 8), (4, 9)],        # cell 4
            [(0, 10), (5, 11)],      # cell 5 → shard 1 end
            [(3, 12), (4, 13), (6, 14)], # cell 6 → shard 2
        ]
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            writer = CsrMemmapWriter(output_dir=tmpdir, shard_n_cells=3)
            gi_list, cnt_list = _make_cells(
                _make_cell((0, 1), (1, 2)),
                _make_cell((2, 3)),
                _make_cell((0, 4), (3, 5), (5, 6)),
                _make_cell((1, 7)),
                _make_cell((2, 8), (4, 9)),
                _make_cell((0, 10), (5, 11)),
                _make_cell((3, 12), (4, 13), (6, 14)),
            )
            writer.append_cells(gi_list, cnt_list)
            writer.finalize()

            import yaml

            manifest_path = tmpdir / "csr-corpus-manifest.yaml"
            with open(manifest_path, "r") as fh:
                doc = yaml.safe_load(fh)
            entries: list[CsrMemmapShardEntry] = []
            for s in doc["shards"]:
                shard_dir = tmpdir / s["path"]
                entries.append(
                    CsrMemmapShardEntry(
                        dataset_id=s["path"],
                        global_start=int(s["global_start"]),
                        global_end=int(s["global_end"]),
                        shard_id=int(s["shard_id"]),
                        shard_path=shard_dir,
                        row_offsets_path=shard_dir / "row_offsets.npy",
                        gene_indices_path=shard_dir / "gene_indices.npy",
                        counts_path=shard_dir / "counts.npy",
                        n_cells=int(s["n_cells"]),
                    )
                )
            entries.sort(key=lambda e: e.global_start)

            reader = AggregateCsrMemmapReader(entries)

            # Read all cells in random order across all shards
            indices = [6, 0, 4, 2, 5, 1, 3]
            batch = reader.read_expression_flat(indices)
            assert batch.batch_size == 7

            for i, idx in enumerate(indices):
                expected_gi, expected_cnt = _expected_cell_data(cell_data, idx)
                np.testing.assert_array_equal(
                    batch.row_gene_indices(i), expected_gi
                )
                np.testing.assert_array_equal(
                    batch.row_counts(i), expected_cnt
                )

    def test_round_trip_empty_cells(self):
        """Cells with zero expressed genes."""
        cell_data = [
            [(0, 5)],
            [],  # empty
            [(1, 10), (2, 15)],
            [],  # empty
            [(3, 20)],
        ]
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            writer = CsrMemmapWriter(output_dir=tmpdir, shard_n_cells=10)
            gi_list, cnt_list = _make_cells(
                _make_cell((0, 5)),
                _make_cell(),
                _make_cell((1, 10), (2, 15)),
                _make_cell(),
                _make_cell((3, 20)),
            )
            writer.append_cells(gi_list, cnt_list)
            writer.finalize()

            import yaml

            manifest_path = tmpdir / "csr-corpus-manifest.yaml"
            with open(manifest_path, "r") as fh:
                doc = yaml.safe_load(fh)
            entries: list[CsrMemmapShardEntry] = []
            for s in doc["shards"]:
                shard_dir = tmpdir / s["path"]
                entries.append(
                    CsrMemmapShardEntry(
                        dataset_id=s["path"],
                        global_start=int(s["global_start"]),
                        global_end=int(s["global_end"]),
                        shard_id=int(s["shard_id"]),
                        shard_path=shard_dir,
                        row_offsets_path=shard_dir / "row_offsets.npy",
                        gene_indices_path=shard_dir / "gene_indices.npy",
                        counts_path=shard_dir / "counts.npy",
                        n_cells=int(s["n_cells"]),
                    )
                )
            entries.sort(key=lambda e: e.global_start)

            reader = AggregateCsrMemmapReader(entries)
            rows = reader.read_expression([1, 3])  # empty cells
            assert len(rows[0].expressed_gene_indices) == 0
            assert len(rows[1].expressed_gene_indices) == 0

            # Flat path
            batch = reader.read_expression_flat([0, 1, 2, 3, 4])
            assert batch.batch_size == 5
            np.testing.assert_array_equal(batch.row_gene_indices(1), [])
            np.testing.assert_array_equal(batch.row_gene_indices(3), [])


# ---------------------------------------------------------------------------
# Test class: Shard Memmap Opening (lazy, isolated)
# ---------------------------------------------------------------------------


class TestShardOpening:
    """Verify shard memmap opening behavior."""

    def test_lazy_opening(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        reader = AggregateCsrMemmapReader(entries)
        # No shards opened yet
        assert len(reader._mmaps) == 0
        # Opening one shard
        reader.read_expression([0])
        assert len(reader._mmaps) == 1
        # Still only one shard opened (cell 0 is in shard 0)
        reader.read_expression([1, 2])
        assert len(reader._mmaps) == 1
        # Reading from shard 1 opens it
        reader.read_expression([3])
        assert len(reader._mmaps) == 2

    def test_no_cross_contamination(self, corpus_5_cells_2_shards):
        """Verify that each reader instance has independent memmap handles."""
        _, entries = corpus_5_cells_2_shards
        r1 = AggregateCsrMemmapReader(entries)
        r2 = AggregateCsrMemmapReader(entries)
        r1.read_expression([0])
        assert len(r2._mmaps) == 0  # r2's handles are independent


# ---------------------------------------------------------------------------
# Test class: DatasetEntry properties
# ---------------------------------------------------------------------------


class TestEntryProperties:
    """Verify that CsrMemmapShardEntry fields are correct."""

    def test_entry_fields(self, corpus_5_cells_2_shards):
        _, entries = corpus_5_cells_2_shards
        assert len(entries) == 2

        # First shard: cells 0–2
        e0 = entries[0]
        assert e0.shard_id == 0
        assert e0.global_start == 0
        assert e0.global_end == 3
        assert e0.n_cells == 3
        assert e0.dataset_id == "shard_000000"
        assert e0.row_offsets_path.name == "row_offsets.npy"
        assert e0.gene_indices_path.name == "gene_indices.npy"
        assert e0.counts_path.name == "counts.npy"

        # Second shard: cells 3–4
        e1 = entries[1]
        assert e1.shard_id == 1
        assert e1.global_start == 3
        assert e1.global_end == 5
        assert e1.n_cells == 2
        assert e1.dataset_id == "shard_000001"
