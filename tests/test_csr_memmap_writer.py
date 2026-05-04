"""Tests for CsrMemmapWriter and CSR shard manifest generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

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


def _assert_shard_on_disk(shard_dir: Path, expected_n_cells: int) -> None:
    """Verify that a shard directory has all required files."""
    assert shard_dir.is_dir(), f"shard dir does not exist: {shard_dir}"

    ro_path = shard_dir / "row_offsets.npy"
    gi_path = shard_dir / "gene_indices.npy"
    cnt_path = shard_dir / "counts.npy"
    manifest_path = shard_dir / "shard-manifest.yaml"

    assert ro_path.is_file(), f"missing row_offsets.npy in {shard_dir}"
    assert gi_path.is_file(), f"missing gene_indices.npy in {shard_dir}"
    assert cnt_path.is_file(), f"missing counts.npy in {shard_dir}"
    assert manifest_path.is_file(), f"missing shard-manifest.yaml in {shard_dir}"

    ro = np.load(str(ro_path))
    gi = np.load(str(gi_path))
    cnt = np.load(str(cnt_path))

    assert ro.dtype == np.int64
    assert gi.dtype == np.int32
    assert cnt.dtype == np.int32
    assert ro.shape == (expected_n_cells + 1,)
    assert gi.shape == cnt.shape

    # Verify manifest
    with open(manifest_path) as fh:
        manifest = yaml.safe_load(fh)
    assert manifest["kind"] == "csr-shard-manifest"
    assert manifest["contract_version"] == "0.1.0"
    assert manifest["n_cells"] == expected_n_cells
    assert manifest["total_nnz"] == int(len(gi))
    assert manifest["row_offsets_dtype"] == "int64"
    assert manifest["gene_indices_dtype"] == "int32"
    assert manifest["counts_dtype"] == "int32"


def _read_shard_cell(
    shard_dir: Path, local_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read a single cell's (gene_indices, counts) from a finalized shard."""
    ro = np.load(str(shard_dir / "row_offsets.npy"))
    gi = np.load(str(shard_dir / "gene_indices.npy"))
    cnt = np.load(str(shard_dir / "counts.npy"))
    s = slice(int(ro[local_idx]), int(ro[local_idx + 1]))
    return gi[s], cnt[s]


# ---------------------------------------------------------------------------
# Tests: basic write
# ---------------------------------------------------------------------------


class TestBasicWrite:
    """Basic write-then-finalize with one shard."""

    def test_single_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)
            gi, cnt = _make_cell((0, 5), (3, 2))
            ids = writer.append_cells([gi], [cnt])
            assert ids == [0]
            assert writer.total_cells_written == 1
            assert writer.n_shards == 0

            manifest_path = writer.finalize()
            assert writer.total_cells_written == 1
            assert writer.n_shards == 1
            assert manifest_path.exists()
            assert manifest_path.name == "csr-corpus-manifest.yaml"

            _assert_shard_on_disk(out / "shard_000000", 1)

            gi_out, cnt_out = _read_shard_cell(out / "shard_000000", 0)
            np.testing.assert_array_equal(gi_out, gi)
            np.testing.assert_array_equal(cnt_out, cnt)

    def test_several_cells_one_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=100)

            cells = [
                _make_cell((0, 1), (2, 3)),
                _make_cell((5, 1)),
                _make_cell((7, 2), (9, 4), (10, 1)),
            ]
            gi_list, cnt_list = _make_cells(*cells)
            ids = writer.append_cells(gi_list, cnt_list)
            assert ids == [0, 1, 2]

            manifest_path = writer.finalize()
            assert writer.total_cells_written == 3
            assert writer.n_shards == 1

            _assert_shard_on_disk(out / "shard_000000", 3)

            # Verify each cell round-trips
            for local_idx, (gi_exp, cnt_exp) in enumerate(cells):
                gi_out, cnt_out = _read_shard_cell(out / "shard_000000", local_idx)
                np.testing.assert_array_equal(gi_out, gi_exp)
                np.testing.assert_array_equal(cnt_out, cnt_exp)

    def test_empty_cell_zero_genes(self):
        """Cells with zero expressed genes should be supported."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)

            gi_empty = np.array([], dtype=np.int32)
            cnt_empty = np.array([], dtype=np.int32)
            gi_norm, cnt_norm = _make_cell((0, 3))

            ids = writer.append_cells(
                [gi_empty, gi_norm],
                [cnt_empty, cnt_norm],
            )
            assert ids == [0, 1]
            writer.finalize()

            gi_out, cnt_out = _read_shard_cell(out / "shard_000000", 0)
            assert len(gi_out) == 0
            assert len(cnt_out) == 0

            gi_out, cnt_out = _read_shard_cell(out / "shard_000000", 1)
            np.testing.assert_array_equal(gi_out, gi_norm)
            np.testing.assert_array_equal(cnt_out, cnt_norm)


# ---------------------------------------------------------------------------
# Tests: multi-shard
# ---------------------------------------------------------------------------


class TestMultiShard:
    """Write across multiple shards."""

    def test_exact_boundary(self):
        """Writing exactly shard_n_cells should flush the shard immediately,
        and the next cell goes to a new shard."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=3)

            # Write exactly 3 cells — shard is flushed eagerly
            cells_a = [
                _make_cell((0, 1)),
                _make_cell((1, 2)),
                _make_cell((2, 3)),
            ]
            gi_a, cnt_a = _make_cells(*cells_a)
            ids_a = writer.append_cells(gi_a, cnt_a)
            assert ids_a == [0, 1, 2]
            assert writer.n_shards == 1  # flushed as soon as full

            # Next cell starts a new (second) shard
            gi_b, cnt_b = _make_cell((3, 4))
            ids_b = writer.append_cells([gi_b], [cnt_b])
            assert ids_b == [3]
            assert writer.n_shards == 1  # second shard not full yet

            writer.finalize()
            assert writer.n_shards == 2

            _assert_shard_on_disk(out / "shard_000000", 3)
            _assert_shard_on_disk(out / "shard_000001", 1)

    def test_batch_across_boundary(self):
        """A batch that crosses the shard boundary should split correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=2)

            # Fill first shard with 2 cells — flushed eagerly on reaching capacity
            cells_a = _make_cells(
                _make_cell((0, 1)),
                _make_cell((1, 1)),
            )
            ids_a = writer.append_cells(*cells_a)
            assert ids_a == [0, 1]
            assert writer.n_shards == 1  # flushed when full

            # Now write 3 more cells in one batch.
            # Shard 1 starts empty; cells go there.
            # After 2 cells, shard 1 is flushed; remaining 1 cell goes to shard 2.
            cells_b = _make_cells(
                _make_cell((2, 1)),
                _make_cell((3, 1)),
                _make_cell((4, 1)),
            )
            ids_b = writer.append_cells(*cells_b)
            assert ids_b == [2, 3, 4]

            # Shards 0 and 1 are already finalized; shard 2 is still in buffer
            manifest_path = writer.finalize()

            _assert_shard_on_disk(out / "shard_000000", 2)
            _assert_shard_on_disk(out / "shard_000001", 2)
            _assert_shard_on_disk(out / "shard_000002", 1)

            # Verify global indices
            with open(manifest_path) as fh:
                corpus_manifest = yaml.safe_load(fh)
            assert corpus_manifest["total_cells"] == 5
            assert corpus_manifest["n_shards"] == 3
            shard_ranges = [
                (s["global_start"], s["global_end"], s["n_cells"])
                for s in corpus_manifest["shards"]
            ]
            assert shard_ranges == [
                (0, 2, 2),
                (2, 4, 2),
                (4, 5, 1),
            ]

    def test_global_indices_contiguous(self):
        """Global indices should be contiguous across shard boundaries."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=5)

            all_ids: list[int] = []
            for batch_idx in range(3):
                cells = [
                    _make_cell((batch_idx * 10 + j, 1))
                    for j in range(4)
                ]
                ids = writer.append_cells(*_make_cells(*cells))
                all_ids.extend(ids)

            assert all_ids == list(range(12))
            writer.finalize()

            # Check corpus manifest ranges
            with open(out / "csr-corpus-manifest.yaml") as fh:
                manifest = yaml.safe_load(fh)
            assert manifest["total_cells"] == 12
            # Shard 0: cells 0-4, Shard 1: cells 5-9, Shard 2: cells 10-11
            for s in manifest["shards"]:
                assert s["global_end"] - s["global_start"] == s["n_cells"]

            total = sum(s["n_cells"] for s in manifest["shards"])
            assert total == 12

    def test_row_offset_chain_across_shards(self):
        """Row offsets within each shard must correctly index into
        gene_indices and counts."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=3)

            # Cell 0: 2 genes
            # Cell 1: 1 gene
            # Cell 2: 3 genes
            cells = _make_cells(
                _make_cell((0, 10), (1, 20)),
                _make_cell((2, 30)),
                _make_cell((3, 40), (4, 50), (5, 60)),
            )
            writer.append_cells(*cells)

            # Cell 3: 0 genes, Cell 4: 2 genes (triggers new shard)
            cells2 = _make_cells(
                _make_cell(),
                _make_cell((6, 70), (7, 80)),
            )
            writer.append_cells(*cells2)
            writer.finalize()

            # Shard 0: cells 0-2, total nnz = 2+1+3 = 6
            shard0 = out / "shard_000000"
            ro0 = np.load(str(shard0 / "row_offsets.npy"))
            gi0 = np.load(str(shard0 / "gene_indices.npy"))
            cnt0 = np.load(str(shard0 / "counts.npy"))
            assert ro0.shape == (4,)  # 3 cells + 1
            # offsets: [0, 2, 3, 6]
            np.testing.assert_array_equal(ro0, [0, 2, 3, 6])
            assert len(gi0) == 6
            assert len(cnt0) == 6

            # Cell 0 in shard0
            s0_c0_gi, s0_c0_cnt = _read_shard_cell(shard0, 0)
            np.testing.assert_array_equal(s0_c0_gi, [0, 1])
            np.testing.assert_array_equal(s0_c0_cnt, [10, 20])

            # Cell 2 in shard0
            s0_c2_gi, s0_c2_cnt = _read_shard_cell(shard0, 2)
            np.testing.assert_array_equal(s0_c2_gi, [3, 4, 5])
            np.testing.assert_array_equal(s0_c2_cnt, [40, 50, 60])

            # Shard 1: cells 3-4
            shard1 = out / "shard_000001"
            ro1 = np.load(str(shard1 / "row_offsets.npy"))
            np.testing.assert_array_equal(ro1, [0, 0, 2])

            s1_c0_gi, s1_c0_cnt = _read_shard_cell(shard1, 0)
            assert len(s1_c0_gi) == 0  # empty cell

            s1_c1_gi, s1_c1_cnt = _read_shard_cell(shard1, 1)
            np.testing.assert_array_equal(s1_c1_gi, [6, 7])


# ---------------------------------------------------------------------------
# Tests: validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Input validation."""

    def test_reject_non_int32_gene_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)
            gi_bad = np.array([0, 1], dtype=np.int64)
            cnt = np.array([5, 3], dtype=np.int32)
            with pytest.raises(TypeError, match="gene_indices"):
                writer.append_cells([gi_bad], [cnt])

    def test_reject_non_int32_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)
            gi = np.array([0, 1], dtype=np.int32)
            cnt_bad = np.array([5, 3], dtype=np.int64)
            with pytest.raises(TypeError, match="counts"):
                writer.append_cells([gi], [cnt_bad])

    def test_reject_float_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)
            gi = np.array([0, 1], dtype=np.int32)
            cnt_float = np.array([5.0, 3.0], dtype=np.float32)
            with pytest.raises(TypeError):
                writer.append_cells([gi], [cnt_float])

    def test_reject_mismatched_list_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)
            gi = np.array([0], dtype=np.int32)
            with pytest.raises(ValueError, match="same length"):
                writer.append_cells([gi, gi], [gi])

    def test_reject_mismatched_element_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)
            gi = np.array([0, 1, 2], dtype=np.int32)
            cnt = np.array([5, 3], dtype=np.int32)
            with pytest.raises(ValueError, match="Mismatched lengths"):
                writer.append_cells([gi], [cnt])

    def test_reject_append_after_finalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)
            writer.finalize()
            gi = np.array([0], dtype=np.int32)
            cnt = np.array([5], dtype=np.int32)
            with pytest.raises(RuntimeError, match="finalized"):
                writer.append_cells([gi], [cnt])

    def test_finalize_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)
            p1 = writer.finalize()
            p2 = writer.finalize()
            assert p1 == p2
            assert p1.exists()

    def test_reject_zero_shard_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            with pytest.raises(ValueError, match="positive"):
                CsrMemmapWriter(out, shard_n_cells=0)


# ---------------------------------------------------------------------------
# Tests: manifest content
# ---------------------------------------------------------------------------


class TestManifests:
    """Verify manifest content correctness."""

    def test_shard_manifest_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)
            cells = _make_cells(
                _make_cell((0, 1), (1, 2)),
                _make_cell((2, 3)),
            )
            writer.append_cells(*cells)
            writer.finalize()

            with open(out / "shard_000000" / "shard-manifest.yaml") as fh:
                m = yaml.safe_load(fh)

            assert m["kind"] == "csr-shard-manifest"
            assert m["contract_version"] == "0.1.0"
            assert m["shard_id"] == 0
            assert m["n_cells"] == 2
            assert m["total_nnz"] == 3
            assert m["row_offsets_shape"] == [3]  # n_cells + 1
            assert m["gene_indices_shape"] == [3]
            assert m["counts_shape"] == [3]

    def test_corpus_manifest_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            src_root = Path("/fake/source/corpus")
            writer = CsrMemmapWriter(
                out, shard_n_cells=3,
                source_corpus_root=src_root,
            )
            cells = _make_cells(
                _make_cell((0, 1)),
                _make_cell((1, 1)),
                _make_cell((2, 1)),
                _make_cell((3, 1)),
            )
            writer.append_cells(*cells)
            p = writer.finalize()

            with open(p) as fh:
                m = yaml.safe_load(fh)

            assert m["kind"] == "csr-corpus-manifest"
            assert m["contract_version"] == "0.1.0"
            assert m["total_cells"] == 4
            assert m["total_nnz"] == 4
            assert m["shard_n_cells_target"] == 3
            assert m["n_shards"] == 2
            assert m["source_corpus_root"] == str(src_root)

            assert len(m["shards"]) == 2
            s0, s1 = m["shards"]

            assert s0["shard_id"] == 0
            assert s0["path"] == "shard_000000"
            assert s0["global_start"] == 0
            assert s0["global_end"] == 3
            assert s0["n_cells"] == 3

            assert s1["shard_id"] == 1
            assert s1["path"] == "shard_000001"
            assert s1["global_start"] == 3
            assert s1["global_end"] == 4
            assert s1["n_cells"] == 1

    def test_corpus_manifest_without_source_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=5)
            writer.append_cells(*_make_cells(_make_cell((0, 1))))
            p = writer.finalize()
            with open(p) as fh:
                m = yaml.safe_load(fh)
            assert "source_corpus_root" not in m

    def test_no_phantom_shard_on_empty_finalize(self):
        """finalize() with no cells should produce 0 shards."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=10)
            writer.finalize()
            with open(out / "csr-corpus-manifest.yaml") as fh:
                m = yaml.safe_load(fh)
            assert m["total_cells"] == 0
            assert m["n_shards"] == 0
            assert m["shards"] == []


# ---------------------------------------------------------------------------
# Tests: output directory isolation
# ---------------------------------------------------------------------------


class TestOutputIsolation:
    """Verify outputs are written only to the specified directory."""

    def test_shard_dirs_naming(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=1)
            for i in range(5):
                gi = np.array([i], dtype=np.int32)
                cnt = np.array([i + 100], dtype=np.int32)
                writer.append_cells([gi], [cnt])
            writer.finalize()

            entries = sorted(out.iterdir())
            entry_names = {e.name for e in entries}
            expected = {
                "shard_000000",
                "shard_000001",
                "shard_000002",
                "shard_000003",
                "shard_000004",
                "csr-corpus-manifest.yaml",
            }
            assert entry_names == expected

    def test_no_writes_outside_output_dir(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            ext = Path(tmp) / "not-csr"
            ext.mkdir()

            writer = CsrMemmapWriter(out, shard_n_cells=5)
            cells = _make_cells(
                _make_cell((0, 1)),
                _make_cell((1, 1)),
                _make_cell((2, 1)),
            )
            writer.append_cells(*cells)
            writer.finalize()

            # The external directory should be untouched
            ext_files = list(ext.iterdir())
            assert ext_files == []

            # All files should be under out/
            for root, _dirs, files in os.walk(str(out)):
                for fname in files:
                    full = Path(root) / fname
                    assert str(full).startswith(str(out))


# ---------------------------------------------------------------------------
# Tests: large-ish cell count
# ---------------------------------------------------------------------------


class TestLargeCellCount:
    """Test with larger numbers to exercise memory and index logic."""

    def test_1000_cells_small_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=200)

            gi_list: list[np.ndarray] = []
            cnt_list: list[np.ndarray] = []
            for i in range(1000):
                gi = np.array([i % 100], dtype=np.int32)
                cnt = np.array([i + 1], dtype=np.int32)
                gi_list.append(gi)
                cnt_list.append(cnt)

            ids = writer.append_cells(gi_list, cnt_list)
            assert ids == list(range(1000))
            assert writer.total_cells_written == 1000
            writer.finalize()

            # Should produce 5 shards of 200 each (1000 / 200)
            assert writer.n_shards == 5
            with open(out / "csr-corpus-manifest.yaml") as fh:
                m = yaml.safe_load(fh)
            assert m["n_shards"] == 5
            for s in m["shards"]:
                assert s["n_cells"] == 200

            # Spot-check: cell 500 should be shard 2, local index 100
            shard_id = 500 // 200  # = 2
            local_idx = 500 % 200  # = 100
            shard_dir = out / f"shard_{shard_id:06d}"
            gi_out, cnt_out = _read_shard_cell(shard_dir, local_idx)
            assert int(gi_out[0]) == 500 % 100
            assert int(cnt_out[0]) == 501

    def test_mixed_size_cells(self):
        """Cells with highly variable nnz should be handled correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "csr"
            writer = CsrMemmapWriter(out, shard_n_cells=5)

            cells = [
                _make_cell(*[(j, j) for j in range(100)]),  # 100 genes
                _make_cell(),                                # 0 genes
                _make_cell((0, 1)),                          # 1 gene
                _make_cell(*[(j, j) for j in range(200)]),  # 200 genes
                _make_cell((99, 999)),                       # 1 gene
            ]
            ids = writer.append_cells(*_make_cells(*cells))
            assert ids == [0, 1, 2, 3, 4]
            writer.finalize()

            # Verify each cell round-trips
            shard_dir = out / "shard_000000"
            for local_idx, (gi_exp, cnt_exp) in enumerate(cells):
                gi_out, cnt_out = _read_shard_cell(shard_dir, local_idx)
                np.testing.assert_array_equal(gi_out, gi_exp)
                np.testing.assert_array_equal(cnt_out, cnt_exp)
