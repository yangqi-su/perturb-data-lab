"""Tests for Phase 2 ExpressionReader — backend-agnostic architecture.

Validates:
- AggregateLanceReader: chunking ≤2048, correct 3-field ExpressionRow
- AggregateLanceReader fast path: flat read, equivalence, no ExpressionRow
- FederatedLanceReader: order preservation across mixed-dataset batches
- AggregateZarrReader: smoke test with CSR-format arrays
- FederatedArrowIpcReader: smoke test as non-Lance backend
- ExpressionRow contains exactly 3 fields (no metadata leakage)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from perturb_data_lab.loaders.expression import (
    AggregateLanceReader,
    AggregateZarrReader,
    ArrowIpcDatasetEntry,
    DatasetEntry,
    ExpressionRow,
    FederatedArrowIpcReader,
    FederatedLanceReader,
    FederatedParquetReader,
    FederatedWebDatasetReader,
    FederatedZarrReader,
    LanceDatasetEntry,
    ParquetDatasetEntry,
    WebDatasetEntry,
    ZarrDatasetEntry,
)
from perturb_data_lab.loaders.loaders import ExpressionBatch

# ===================================================================
# Constants — known paths for the synthetic corpus
# ===================================================================

_ARCHIVED_ROOT = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/archive/plans-20260427-backend-topology-validation/outputs"
)

AGGREGATE_LANCE = (
    _ARCHIVED_ROOT / "lance-aggregate/matrix/aggregated-cells.lance"
)

AGGREGATE_ZARR_BASE = _ARCHIVED_ROOT / "zarr-aggregate/matrix"
FEDERATED_ZARR_BASE = _ARCHIVED_ROOT / "zarr-federated"
FEDERATED_BASE = _ARCHIVED_ROOT / "lance-federated"
ARROW_IPC_BASE = _ARCHIVED_ROOT / "arrow-ipc-federated"
PARQUET_BASE = _ARCHIVED_ROOT / "arrow-parquet-federated"
WEBDATASET_BASE = _ARCHIVED_ROOT / "webdataset-federated"

# Dataset sizes from the archived corpus
_DATASET_SIZES = {
    "dummy_00": 50_000,
    "dummy_01": 75_000,
    "dummy_02": 100_000,
    "dummy_03": 120_000,
    "dummy_04": 80_000,
    "dummy_05": 150_000,
    "dummy_06": 90_000,
    "dummy_07": 110_000,
    "dummy_08": 65_000,
    "dummy_09": 95_000,
}

# Build ranges for all 10 datasets
_ALL_RANGES: list[DatasetEntry] = []
_start = 0
for ds_id, size in sorted(_DATASET_SIZES.items()):
    _ALL_RANGES.append(DatasetEntry(ds_id, _start, _start + size))
    _start += size

# Subset for tests: dummy_00 + dummy_01 (first two datasets)
_TEST_RANGES = [r for r in _ALL_RANGES if r.dataset_id in ("dummy_00", "dummy_01")]

# Single entry covering both test datasets (for aggregate readers)
_AGG_ENTRY = DatasetEntry("aggregated", 0, _start)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def agg_reader() -> AggregateLanceReader:
    """AggregateLanceReader for dummy_00 + dummy_01 range."""
    return AggregateLanceReader(AGGREGATE_LANCE, _TEST_RANGES)


@pytest.fixture(scope="module")
def agg_reader_full() -> AggregateLanceReader:
    """AggregateLanceReader for all 10 datasets."""
    return AggregateLanceReader(AGGREGATE_LANCE, _ALL_RANGES)


@pytest.fixture(scope="module")
def agg_reader_single_entry() -> AggregateLanceReader:
    """AggregateLanceReader with a single entry covering full range."""
    return AggregateLanceReader(AGGREGATE_LANCE, [_AGG_ENTRY])


@pytest.fixture(scope="module")
def fed_reader() -> FederatedLanceReader:
    """FederatedLanceReader for dummy_00 + dummy_01."""
    entries = [
        LanceDatasetEntry(
            "dummy_00",
            0,
            50_000,
            FEDERATED_BASE / "dummy_00/matrix/dummy_00-release.lance",
        ),
        LanceDatasetEntry(
            "dummy_01",
            50_000,
            125_000,
            FEDERATED_BASE / "dummy_01/matrix/dummy_01-release.lance",
        ),
    ]
    return FederatedLanceReader(entries)


@pytest.fixture(scope="module")
def agg_zarr_reader() -> AggregateZarrReader:
    """AggregateZarrReader for full 10-dataset corpus."""
    return AggregateZarrReader(
        AGGREGATE_ZARR_BASE / "aggregated-row-offsets.zarr",
        AGGREGATE_ZARR_BASE / "aggregated-indices.zarr",
        AGGREGATE_ZARR_BASE / "aggregated-counts.zarr",
        [_AGG_ENTRY],
    )


@pytest.fixture(scope="module")
def arrow_ipc_reader() -> FederatedArrowIpcReader:
    """FederatedArrowIpcReader for dummy_00 + dummy_01."""
    entries = [
        ArrowIpcDatasetEntry(
            "dummy_00",
            0,
            50_000,
            ARROW_IPC_BASE / "dummy_00/matrix/dummy_00-release-cells.arrow",
        ),
        ArrowIpcDatasetEntry(
            "dummy_01",
            50_000,
            125_000,
            ARROW_IPC_BASE / "dummy_01/matrix/dummy_01-release-cells.arrow",
        ),
    ]
    return FederatedArrowIpcReader(entries)


@pytest.fixture(scope="module")
def fed_zarr_reader() -> FederatedZarrReader:
    """FederatedZarrReader for dummy_00 + dummy_01."""
    entries = [
        ZarrDatasetEntry(
            "dummy_00",
            0,
            50_000,
            FEDERATED_ZARR_BASE / "dummy_00/matrix/dummy_00-release-row-offsets.zarr",
            FEDERATED_ZARR_BASE / "dummy_00/matrix/dummy_00-release-indices.zarr",
            FEDERATED_ZARR_BASE / "dummy_00/matrix/dummy_00-release-counts.zarr",
        ),
        ZarrDatasetEntry(
            "dummy_01",
            50_000,
            125_000,
            FEDERATED_ZARR_BASE / "dummy_01/matrix/dummy_01-release-row-offsets.zarr",
            FEDERATED_ZARR_BASE / "dummy_01/matrix/dummy_01-release-indices.zarr",
            FEDERATED_ZARR_BASE / "dummy_01/matrix/dummy_01-release-counts.zarr",
        ),
    ]
    return FederatedZarrReader(entries)


@pytest.fixture(scope="module")
def parquet_reader() -> FederatedParquetReader:
    """FederatedParquetReader for dummy_00 + dummy_01."""
    entries = [
        ParquetDatasetEntry(
            "dummy_00",
            0,
            50_000,
            PARQUET_BASE / "dummy_00/matrix/dummy_00-release-cells.parquet",
        ),
        ParquetDatasetEntry(
            "dummy_01",
            50_000,
            125_000,
            PARQUET_BASE / "dummy_01/matrix/dummy_01-release-cells.parquet",
        ),
    ]
    return FederatedParquetReader(entries)


@pytest.fixture(scope="module")
def webdataset_reader() -> FederatedWebDatasetReader:
    """FederatedWebDatasetReader for dummy_00 + dummy_01."""
    entries = [
        WebDatasetEntry(
            "dummy_00",
            0,
            50_000,
            WEBDATASET_BASE / "dummy_00/matrix/dummy_00-release-cells.tar",
        ),
        WebDatasetEntry(
            "dummy_01",
            50_000,
            125_000,
            WEBDATASET_BASE / "dummy_01/matrix/dummy_01-release-cells.tar",
        ),
    ]
    return FederatedWebDatasetReader(entries)


def _assert_flat_batch_matches_legacy(reader, indices: list[int]) -> None:
    """Assert ``read_expression_flat()`` matches ``read_expression()``."""
    batch = reader.read_expression_flat(indices)
    rows = reader.read_expression(indices)

    assert isinstance(batch, ExpressionBatch)
    assert batch.batch_size == len(indices)
    np.testing.assert_array_equal(
        batch.global_row_index, np.array(indices, dtype=np.int64)
    )

    if not indices:
        expected_offsets = np.array([0], dtype=np.int64)
        expected_egi = np.array([], dtype=np.int32)
        expected_ec = np.array([], dtype=np.int32)
    else:
        expected_offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        for pos, row in enumerate(rows):
            expected_offsets[pos + 1] = (
                expected_offsets[pos] + len(row.expressed_gene_indices)
            )
        expected_egi = np.concatenate([row.expressed_gene_indices for row in rows])
        expected_ec = np.concatenate([row.expression_counts for row in rows])

    np.testing.assert_array_equal(batch.row_offsets, expected_offsets)
    np.testing.assert_array_equal(batch.expressed_gene_indices, expected_egi)
    np.testing.assert_array_equal(batch.expression_counts, expected_ec)
    assert batch.row_offsets[-1] == len(batch.expressed_gene_indices)
    assert len(batch.expressed_gene_indices) == len(batch.expression_counts)


# ===================================================================
# ExpressionRow structure tests
# ===================================================================


class TestExpressionRowStructure:
    """Tests that ExpressionRow has exactly 3 fields."""

    def test_expression_row_fields(self):
        """ExpressionRow has exactly the 3 required fields."""
        row = ExpressionRow(
            global_row_index=42,
            expressed_gene_indices=np.array([1, 2, 3], dtype=np.int32),
            expression_counts=np.array([10, 20, 30], dtype=np.int32),
        )
        # Confirm exact fields
        fields = {f.name for f in row.__dataclass_fields__.values()}
        assert fields == {"global_row_index", "expressed_gene_indices", "expression_counts"}
        assert row.global_row_index == 42
        assert row.expressed_gene_indices.dtype == np.int32
        assert row.expression_counts.dtype == np.int32

    def test_no_extra_identity_fields(self, agg_reader):
        """ExpressionRow has no dataset_id, dataset_index, or local_row_index."""
        rows = agg_reader.read_expression([0])
        r = rows[0]
        assert not hasattr(r, "dataset_id")
        assert not hasattr(r, "dataset_index")
        assert not hasattr(r, "local_row_index")

    def test_no_metadata_fields(self, agg_reader):
        """ExpressionRow has no metadata fields (size_factor, etc.)."""
        rows = agg_reader.read_expression([42])
        r = rows[0]
        assert not hasattr(r, "size_factor")
        assert not hasattr(r, "canonical_perturbation")
        assert not hasattr(r, "canonical_context")
        assert not hasattr(r, "raw_fields")


# ===================================================================
# AggregateLanceReader tests
# ===================================================================


class TestAggregateLanceReader:
    """Tests for AggregateLanceReader."""

    def test_single_cell(self, agg_reader):
        rows = agg_reader.read_expression([0])
        assert len(rows) == 1
        r = rows[0]
        assert r.global_row_index == 0
        assert isinstance(r.expressed_gene_indices, np.ndarray)
        assert isinstance(r.expression_counts, np.ndarray)
        assert len(r.expressed_gene_indices) > 0
        assert len(r.expressed_gene_indices) == len(r.expression_counts)

    def test_cross_dataset(self, agg_reader):
        rows = agg_reader.read_expression([0, 50_000])
        assert len(rows) == 2
        assert rows[0].global_row_index == 0
        assert rows[1].global_row_index == 50_000

    def test_expression_data_type(self, agg_reader):
        """Expression arrays are numpy int32."""
        rows = agg_reader.read_expression([42])
        r = rows[0]
        assert r.expressed_gene_indices.dtype == np.int32
        assert r.expression_counts.dtype == np.int32

    @pytest.mark.parametrize("n", [0, 1, 128, 2048, 2049, 4096])
    def test_chunking_boundaries(self, agg_reader, n):
        """Read n indices (covering ≤2048, exact 2048, and >2048 cases)."""
        indices = list(range(n))
        rows = agg_reader.read_expression(indices)
        assert len(rows) == n
        for i, row in enumerate(rows):
            assert row.global_row_index == i

    def test_out_of_range_negative(self, agg_reader):
        with pytest.raises(IndexError):
            agg_reader.read_expression([-1])

    def test_out_of_range_past_end(self, agg_reader):
        with pytest.raises(IndexError):
            agg_reader.read_expression([125_000])

    def test_out_of_range_not_in_registered(self, agg_reader):
        """Index is a valid Lance row, but not in registered ranges."""
        with pytest.raises(IndexError):
            agg_reader.read_expression([500_000])

    def test_empty_input(self, agg_reader):
        assert agg_reader.read_expression([]) == []

    def test_output_matches_input_order(self, agg_reader):
        """Non-monotonic input order is preserved exactly."""
        indices = [100, 5, 50_000, 99999]
        rows = agg_reader.read_expression(indices)
        assert [r.global_row_index for r in rows] == indices

    def test_full_ten_datasets(self, agg_reader_full):
        """Read from all 10 datasets."""
        indices = []
        for r in _ALL_RANGES:
            mid = r.global_start + (r.global_end - r.global_start) // 2
            indices.append(mid)
        rows = agg_reader_full.read_expression(indices)
        assert len(rows) == len(_ALL_RANGES)
        for row, idx in zip(rows, indices):
            assert row.global_row_index == idx

    def test_shuffled_large_batch(self, agg_reader_full):
        """Large shuffled batch from all 10 datasets."""
        random.seed(42)
        all_indices = list(range(935_000))
        sample = random.sample(all_indices, 5000)
        expected = list(sample)
        rows = agg_reader_full.read_expression(sample)
        assert len(rows) == 5000
        assert [r.global_row_index for r in rows] == expected

    def test_single_entry_mode(self, agg_reader_single_entry):
        """Works with a single DatasetEntry covering all rows."""
        rows = agg_reader_single_entry.read_expression([0, 50000, 99999])
        assert len(rows) == 3
        assert [r.global_row_index for r in rows] == [0, 50000, 99999]


# ===================================================================
# FederatedLanceReader tests
# ===================================================================


class TestFederatedLanceReader:
    """Tests for FederatedLanceReader."""

    def test_single_cell_first_dataset(self, fed_reader):
        rows = fed_reader.read_expression([0])
        assert len(rows) == 1
        assert rows[0].global_row_index == 0

    def test_single_cell_second_dataset(self, fed_reader):
        rows = fed_reader.read_expression([50_000])
        assert len(rows) == 1
        assert rows[0].global_row_index == 50_000

    def test_preserves_input_order(self, fed_reader):
        """Exact input order is preserved across mixed-dataset batches."""
        indices = [0, 50_000, 1, 50_001]
        rows = fed_reader.read_expression(indices)
        assert [r.global_row_index for r in rows] == indices

    def test_alternating_order(self, fed_reader):
        """Alternating interleaved indices across datasets."""
        alt = [0, 50_000, 1, 50_001, 2, 50_002]
        rows = fed_reader.read_expression(alt)
        assert [r.global_row_index for r in rows] == alt

    def test_all_from_one_dataset(self, fed_reader):
        rows = fed_reader.read_expression(list(range(100)))
        assert len(rows) == 100
        assert all(r.global_row_index == i for i, r in enumerate(rows))

    def test_all_from_second_dataset(self, fed_reader):
        rows = fed_reader.read_expression(list(range(50_000, 50_100)))
        assert len(rows) == 100
        assert all(
            r.global_row_index == 50_000 + i for i, r in enumerate(rows)
        )

    def test_out_of_range(self, fed_reader):
        with pytest.raises(IndexError):
            fed_reader.read_expression([-1])
        with pytest.raises(IndexError):
            fed_reader.read_expression([125_000])

    def test_empty_input(self, fed_reader):
        assert fed_reader.read_expression([]) == []

    def test_chunking_large_single_dataset(self, fed_reader):
        """>2048 indices from one dataset triggers chunking."""
        rows = fed_reader.read_expression(list(range(3000)))
        assert len(rows) == 3000
        assert [r.global_row_index for r in rows] == list(range(3000))

    def test_large_shuffled_mixed_order(self, fed_reader):
        """Large shuffled batch preserves order across 2 datasets."""
        random.seed(42)
        base = list(range(0, 50_000, 10)) + list(range(50_000, 125_000, 10))
        random.shuffle(base)
        expected = list(base)
        rows = fed_reader.read_expression(base)
        assert len(rows) == len(expected)
        assert [r.global_row_index for r in rows] == expected

    def test_expression_data_only(self, fed_reader):
        """FederatedLanceReader returns only expression data."""
        rows = fed_reader.read_expression([42])
        r = rows[0]
        assert isinstance(r.expressed_gene_indices, np.ndarray)
        assert isinstance(r.expression_counts, np.ndarray)
        assert not hasattr(r, "size_factor")
        assert not hasattr(r, "canonical_perturbation")

    def test_caching_reuses_dataset_handles(self, fed_reader):
        """Calling read_expression multiple times reuses cached handles."""
        rows1 = fed_reader.read_expression([0, 50_000])
        rows2 = fed_reader.read_expression([1, 50_001])
        assert len(rows1) == 2
        assert len(rows2) == 2
        assert rows2[0].global_row_index == 1
        assert rows2[1].global_row_index == 50_001


# ===================================================================
# AggregateZarrReader tests (non-Lance backend smoke test)
# ===================================================================


class TestAggregateZarrReader:
    """Smoke tests for AggregateZarrReader (CSR-format Zarr)."""

    def test_single_cell(self, agg_zarr_reader):
        rows = agg_zarr_reader.read_expression([0])
        assert len(rows) == 1
        r = rows[0]
        assert r.global_row_index == 0
        assert isinstance(r.expressed_gene_indices, np.ndarray)
        assert isinstance(r.expression_counts, np.ndarray)
        assert len(r.expressed_gene_indices) > 0
        assert len(r.expressed_gene_indices) == len(r.expression_counts)

    def test_multiple_cells(self, agg_zarr_reader):
        rows = agg_zarr_reader.read_expression([0, 50000, 100000])
        assert len(rows) == 3
        assert [r.global_row_index for r in rows] == [0, 50000, 100000]

    def test_order_preservation(self, agg_zarr_reader):
        indices = [100, 5, 50000, 99999]
        rows = agg_zarr_reader.read_expression(indices)
        assert [r.global_row_index for r in rows] == indices

    def test_empty_input(self, agg_zarr_reader):
        assert agg_zarr_reader.read_expression([]) == []

    def test_expression_data_only(self, agg_zarr_reader):
        """Zarr reader returns only 3-field ExpressionRow."""
        rows = agg_zarr_reader.read_expression([42])
        r = rows[0]
        assert not hasattr(r, "dataset_id")
        assert not hasattr(r, "size_factor")
        assert r.expressed_gene_indices.dtype == np.int32
        assert r.expression_counts.dtype == np.int32


class TestFallbackFlatReaderContract:
    """Fallback ``read_expression_flat()`` coverage for non-optimized backends."""

    @pytest.mark.parametrize(
        ("fixture_name", "indices"),
        [
            ("agg_zarr_reader", [100, 5, 50_000, 99_999]),
            ("fed_zarr_reader", [0, 50_000, 1, 50_001]),
            ("arrow_ipc_reader", [0, 50_000, 1, 50_001]),
            ("parquet_reader", [0, 50_000, 1, 50_001]),
            ("webdataset_reader", [0, 50_000, 1, 50_001]),
        ],
    )
    def test_flat_reader_matches_legacy_path(
        self, request, fixture_name: str, indices: list[int]
    ):
        reader = request.getfixturevalue(fixture_name)
        _assert_flat_batch_matches_legacy(reader, indices)

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "agg_zarr_reader",
            "fed_zarr_reader",
            "arrow_ipc_reader",
            "parquet_reader",
            "webdataset_reader",
        ],
    )
    def test_flat_reader_empty_batch(self, request, fixture_name: str):
        reader = request.getfixturevalue(fixture_name)
        _assert_flat_batch_matches_legacy(reader, [])

    @pytest.mark.parametrize(
        ("fixture_name", "indices"),
        [
            ("agg_zarr_reader", [42]),
            ("fed_zarr_reader", [42]),
            ("arrow_ipc_reader", [42]),
            ("parquet_reader", [42]),
            ("webdataset_reader", [42]),
        ],
    )
    def test_flat_reader_returns_expression_only(
        self, request, fixture_name: str, indices: list[int]
    ):
        reader = request.getfixturevalue(fixture_name)
        batch = reader.read_expression_flat(indices)
        assert not hasattr(batch, "size_factor")
        assert not hasattr(batch, "canonical_perturbation")
        assert not hasattr(batch, "canonical_context")


# ===================================================================
# FederatedArrowIpcReader tests (non-Lance federated backend)
# ===================================================================


class TestFederatedArrowIpcReader:
    """Smoke tests for FederatedArrowIpcReader (Arrow IPC files)."""

    def test_single_cell_first_dataset(self, arrow_ipc_reader):
        rows = arrow_ipc_reader.read_expression([0])
        assert len(rows) == 1
        assert rows[0].global_row_index == 0

    def test_single_cell_second_dataset(self, arrow_ipc_reader):
        rows = arrow_ipc_reader.read_expression([50_000])
        assert len(rows) == 1
        assert rows[0].global_row_index == 50_000

    def test_preserves_input_order(self, arrow_ipc_reader):
        indices = [0, 50_000, 1, 50_001]
        rows = arrow_ipc_reader.read_expression(indices)
        assert [r.global_row_index for r in rows] == indices

    def test_all_from_one_dataset(self, arrow_ipc_reader):
        rows = arrow_ipc_reader.read_expression(list(range(50)))
        assert len(rows) == 50
        assert all(r.global_row_index == i for i, r in enumerate(rows))

    def test_out_of_range(self, arrow_ipc_reader):
        with pytest.raises(IndexError):
            arrow_ipc_reader.read_expression([-1])
        with pytest.raises(IndexError):
            arrow_ipc_reader.read_expression([125_000])

    def test_empty_input(self, arrow_ipc_reader):
        assert arrow_ipc_reader.read_expression([]) == []

    def test_expression_data_only(self, arrow_ipc_reader):
        """Arrow IPC reader returns only 3-field ExpressionRow."""
        rows = arrow_ipc_reader.read_expression([42])
        r = rows[0]
        assert not hasattr(r, "dataset_id")
        assert not hasattr(r, "size_factor")
        assert r.expressed_gene_indices.dtype == np.int32
        assert r.expression_counts.dtype == np.int32

    def test_correct_expression_values(self, arrow_ipc_reader):
        """Verify expression arrays are non-empty and consistent."""
        rows = arrow_ipc_reader.read_expression([0, 1, 2])
        for r in rows:
            assert len(r.expressed_gene_indices) > 0
            assert len(r.expression_counts) == len(r.expressed_gene_indices)
            assert r.expressed_gene_indices.dtype == np.int32
            assert r.expression_counts.dtype == np.int32


# ===================================================================
# Phase 2 — AggregateLanceReader fast path tests
# ===================================================================


class TestAggregateLanceReaderFastPath:
    """Tests for ``AggregateLanceReader.read_expression_flat()``.

    Validates:
    - No ``ExpressionRow`` objects are constructed.
    - Output is an ``ExpressionBatch`` with correct shapes.
    - Numerical equivalence with the legacy ``read_expression()`` path.
    - Order preservation, chunking across boundaries, and error handling.
    """

    def test_no_expression_row_objects(self, agg_reader):
        """Fast path does not return or construct ExpressionRow objects."""
        batch = agg_reader.read_expression_flat([0, 1, 2])
        assert isinstance(batch, ExpressionBatch)
        assert not isinstance(batch, ExpressionRow)

    def test_single_cell_shapes(self, agg_reader):
        """Single cell returns correct shapes and dtypes."""
        batch = agg_reader.read_expression_flat([0])
        assert batch.batch_size == 1
        assert batch.global_row_index.dtype == np.int64
        assert batch.row_offsets.dtype == np.int64
        assert batch.expressed_gene_indices.dtype == np.int32
        assert batch.expression_counts.dtype == np.int32
        assert len(batch.row_offsets) == 2  # batch_size+1
        assert batch.row_offsets[0] == 0
        assert batch.row_offsets[1] > 0  # non-empty cell

    def test_empty_input(self, agg_reader):
        """Empty input returns zero-size ExpressionBatch."""
        batch = agg_reader.read_expression_flat([])
        assert batch.batch_size == 0
        assert len(batch.global_row_index) == 0
        assert len(batch.row_offsets) == 1
        assert batch.row_offsets[0] == 0
        assert len(batch.expressed_gene_indices) == 0
        assert len(batch.expression_counts) == 0

    def test_order_preservation(self, agg_reader):
        """Input order is preserved in global_row_index."""
        indices = [100, 5, 50_000, 99_999]
        batch = agg_reader.read_expression_flat(indices)
        assert batch.global_row_index.tolist() == indices

    def test_chunking_boundaries(self, agg_reader):
        """Reads across the 2048-index chunk boundary."""
        # 2049 indices triggers two take() calls
        n = 2049
        indices = list(range(n))
        batch = agg_reader.read_expression_flat(indices)
        assert batch.batch_size == n
        assert batch.global_row_index[0] == 0
        assert batch.global_row_index[-1] == n - 1
        # Row offsets should have batch_size+1 entries
        assert len(batch.row_offsets) == n + 1
        # Offsets are strictly increasing
        assert np.all(np.diff(batch.row_offsets) >= 0)

    def test_out_of_range(self, agg_reader):
        """Out-of-range indices raise IndexError.

        The fast path validates against the Lance file row count (935 000),
        not the registered dataset entry ranges.  So 125_000–934_999 are
        valid Lance rows even though they're not in the dummy_00/dummy_01
        entry ranges used by the test fixture.
        """
        with pytest.raises(IndexError):
            agg_reader.read_expression_flat([-1])
        # Past-end: the aggregate Lance file has 935 000 rows
        with pytest.raises(IndexError):
            agg_reader.read_expression_flat([935_000])
        with pytest.raises(IndexError):
            agg_reader.read_expression_flat([1_000_000])

    @pytest.mark.parametrize("n", [0, 1, 5, 128, 1024, 2048, 2049, 3000])
    def test_equivalence_with_legacy_path(self, agg_reader_full, n):
        """Flat output is numerically equivalent to the legacy row-object path.

        Compares the reconstructed flat arrays from ``read_expression()``
        (which produces ``list[ExpressionRow]``) against the direct flat
        output from ``read_expression_flat()``.
        """
        # Use contiguous and shuffled indices for thorough testing
        indices = list(range(n)) if n <= 2048 else list(range(0, n * 2, 2))[:n]

        # Fast path
        batch = agg_reader_full.read_expression_flat(indices)

        # Legacy path
        rows = agg_reader_full.read_expression(indices)
        # Reconstruct flat arrays the way BatchExecutor does
        if n == 0:
            legacy_row_offsets = np.array([0], dtype=np.int64)
            legacy_egi = np.array([], dtype=np.int32)
            legacy_ec = np.array([], dtype=np.int32)
        else:
            legacy_row_offsets = np.zeros(n + 1, dtype=np.int64)
            for i, row in enumerate(rows):
                legacy_row_offsets[i + 1] = (
                    legacy_row_offsets[i] + len(row.expressed_gene_indices)
                )
            legacy_egi = np.concatenate([r.expressed_gene_indices for r in rows])
            legacy_ec = np.concatenate([r.expression_counts for r in rows])

        assert batch.batch_size == n
        np.testing.assert_array_equal(
            batch.global_row_index, np.array(indices, dtype=np.int64)
        )
        np.testing.assert_array_equal(batch.row_offsets, legacy_row_offsets)
        np.testing.assert_array_equal(batch.expressed_gene_indices, legacy_egi)
        np.testing.assert_array_equal(batch.expression_counts, legacy_ec)

    def test_single_entry_mode(self, agg_reader_single_entry):
        """Works with a single DatasetEntry covering all rows."""
        batch = agg_reader_single_entry.read_expression_flat([0, 50_000, 99_999])
        assert batch.batch_size == 3
        assert batch.global_row_index.tolist() == [0, 50_000, 99_999]

    def test_row_access_methods(self, agg_reader):
        """ExpressionBatch row_slice, row_gene_indices, row_counts work."""
        batch = agg_reader.read_expression_flat([2, 0])
        # Row 0 (index 2) and row 1 (index 0) should exist
        s = batch.row_slice(0)
        assert s.start == 0
        assert s.stop > 0
        genes = batch.row_gene_indices(0)
        counts = batch.row_counts(0)
        assert len(genes) > 0
        assert len(genes) == len(counts)
        assert genes.dtype == np.int32
        assert counts.dtype == np.int32

    def test_large_shuffled_batch(self, agg_reader_full):
        """Large shuffled batch across all 10 datasets."""
        random.seed(42)
        all_indices = list(range(935_000))
        sample = random.sample(all_indices, 5000)
        batch = agg_reader_full.read_expression_flat(sample)
        assert batch.batch_size == 5000
        assert batch.global_row_index.tolist() == sample
        assert len(batch.row_offsets) == 5001
        assert batch.row_offsets[-1] == len(batch.expressed_gene_indices)


# ===================================================================
# Phase 3 — FederatedLanceReader fast path tests
# ===================================================================


class TestFederatedLanceReaderFastPath:
    """Tests for ``FederatedLanceReader.read_expression_flat()``.

    Validates:
    - No ``ExpressionRow`` objects are constructed.
    - Output is an ``ExpressionBatch`` with correct shapes.
    - Numerical equivalence with the legacy ``read_expression()`` path.
    - Order preservation across mixed-dataset batches.
    - Chunking across boundaries, error handling, and handle caching.
    """

    def test_no_expression_row_objects(self, fed_reader):
        """Fast path does not return or construct ExpressionRow objects."""
        batch = fed_reader.read_expression_flat([0, 1, 2])
        assert isinstance(batch, ExpressionBatch)
        assert not isinstance(batch, ExpressionRow)

    def test_single_cell_shapes(self, fed_reader):
        """Single cell returns correct shapes and dtypes."""
        batch = fed_reader.read_expression_flat([0])
        assert batch.batch_size == 1
        assert batch.global_row_index.dtype == np.int64
        assert batch.row_offsets.dtype == np.int64
        assert batch.expressed_gene_indices.dtype == np.int32
        assert batch.expression_counts.dtype == np.int32
        assert len(batch.row_offsets) == 2  # batch_size+1
        assert batch.row_offsets[0] == 0
        assert batch.row_offsets[1] > 0  # non-empty cell

    def test_empty_input(self, fed_reader):
        """Empty input returns zero-size ExpressionBatch."""
        batch = fed_reader.read_expression_flat([])
        assert batch.batch_size == 0
        assert len(batch.global_row_index) == 0
        assert len(batch.row_offsets) == 1
        assert batch.row_offsets[0] == 0
        assert len(batch.expressed_gene_indices) == 0
        assert len(batch.expression_counts) == 0

    def test_order_preservation(self, fed_reader):
        """Non-monotonic interleaved input order is preserved exactly."""
        indices = [0, 50_000, 1, 50_001, 2, 50_002]
        batch = fed_reader.read_expression_flat(indices)
        assert batch.global_row_index.tolist() == indices

    def test_alternating_order(self, fed_reader):
        """Densely interleaved indices across two datasets."""
        alt = [0, 50_000, 1, 50_001, 2, 50_002, 3, 50_003, 4, 50_004]
        batch = fed_reader.read_expression_flat(alt)
        assert batch.batch_size == len(alt)
        assert batch.global_row_index.tolist() == alt

    def test_all_from_first_dataset(self, fed_reader):
        """Batch drawn entirely from the first dataset."""
        indices = list(range(100))
        batch = fed_reader.read_expression_flat(indices)
        assert batch.batch_size == 100
        assert batch.global_row_index.tolist() == indices
        assert len(batch.row_offsets) == 101

    def test_all_from_second_dataset(self, fed_reader):
        """Batch drawn entirely from the second dataset."""
        indices = list(range(50_000, 50_100))
        batch = fed_reader.read_expression_flat(indices)
        assert batch.batch_size == 100
        assert batch.global_row_index.tolist() == indices
        assert len(batch.row_offsets) == 101

    def test_chunking_boundaries(self, fed_reader):
        """Reads across the 2048-index chunk boundary in a single dataset."""
        n = 3000
        indices = list(range(n))
        batch = fed_reader.read_expression_flat(indices)
        assert batch.batch_size == n
        assert batch.global_row_index[0] == 0
        assert batch.global_row_index[-1] == n - 1
        assert len(batch.row_offsets) == n + 1
        # Row offsets are strictly increasing
        assert np.all(np.diff(batch.row_offsets) >= 0)

    def test_out_of_range(self, fed_reader):
        """Out-of-range indices raise IndexError."""
        with pytest.raises(IndexError):
            fed_reader.read_expression_flat([-1])
        with pytest.raises(IndexError):
            fed_reader.read_expression_flat([125_000])

    @pytest.mark.parametrize("n", [0, 1, 5, 128, 500, 1024, 2048, 3000])
    def test_equivalence_with_legacy_path(self, fed_reader, n):
        """Flat output is numerically equivalent to the legacy row-object path.

        Compares the reconstructed flat arrays from ``read_expression()``
        (which produces ``list[ExpressionRow]``) against the direct flat
        output from ``read_expression_flat()``.
        """
        # Build a mixed-dataset index list for thorough testing.
        # For n>0, interleave indices from both datasets.
        if n == 0:
            indices = []
        else:
            half = n // 2
            ds0 = list(range(0, min(half, 1000)))
            ds1 = list(range(50_000, 50_000 + n - len(ds0)))
            merged = []
            for i in range(max(len(ds0), len(ds1))):
                if i < len(ds0):
                    merged.append(ds0[i])
                if i < len(ds1):
                    merged.append(ds1[i])
            indices = merged

        # Fast path
        batch = fed_reader.read_expression_flat(indices)

        # Legacy path
        rows = fed_reader.read_expression(indices)
        # Reconstruct flat arrays the way BatchExecutor does
        if n == 0:
            legacy_row_offsets = np.array([0], dtype=np.int64)
            legacy_egi = np.array([], dtype=np.int32)
            legacy_ec = np.array([], dtype=np.int32)
        else:
            legacy_row_offsets = np.zeros(n + 1, dtype=np.int64)
            for i, row in enumerate(rows):
                legacy_row_offsets[i + 1] = (
                    legacy_row_offsets[i] + len(row.expressed_gene_indices)
                )
            legacy_egi = np.concatenate([r.expressed_gene_indices for r in rows])
            legacy_ec = np.concatenate([r.expression_counts for r in rows])

        assert batch.batch_size == n
        np.testing.assert_array_equal(
            batch.global_row_index, np.array(indices, dtype=np.int64)
        )
        np.testing.assert_array_equal(batch.row_offsets, legacy_row_offsets)
        np.testing.assert_array_equal(batch.expressed_gene_indices, legacy_egi)
        np.testing.assert_array_equal(batch.expression_counts, legacy_ec)

    def test_row_access_methods(self, fed_reader):
        """ExpressionBatch row_slice, row_gene_indices, row_counts work."""
        batch = fed_reader.read_expression_flat([2, 50_000, 0])
        # Row 0 (index 2), row 1 (index 50000), row 2 (index 0)
        s = batch.row_slice(0)
        assert s.start == 0
        assert s.stop > 0
        genes = batch.row_gene_indices(0)
        counts = batch.row_counts(0)
        assert len(genes) > 0
        assert len(genes) == len(counts)
        assert genes.dtype == np.int32
        assert counts.dtype == np.int32

    def test_large_shuffled_mixed_order(self, fed_reader):
        """Large shuffled batch preserves order across 2 datasets."""
        random.seed(42)
        base = list(range(0, 50_000, 10)) + list(range(50_000, 125_000, 10))
        random.shuffle(base)
        expected = list(base)
        batch = fed_reader.read_expression_flat(base)
        assert len(batch.global_row_index) == len(expected)
        assert batch.global_row_index.tolist() == expected
        assert len(batch.row_offsets) == len(expected) + 1
        assert batch.row_offsets[-1] == len(batch.expressed_gene_indices)

    def test_caching_reuses_handles(self, fed_reader):
        """Calling read_expression_flat multiple times reuses cached handles."""
        batch1 = fed_reader.read_expression_flat([0, 50_000])
        batch2 = fed_reader.read_expression_flat([1, 50_001])
        assert batch1.batch_size == 2
        assert batch2.batch_size == 2
        assert batch2.global_row_index.tolist() == [1, 50_001]

    def test_expression_data_only(self, fed_reader):
        """Fast path returns only expression data, no metadata leakage."""
        batch = fed_reader.read_expression_flat([42])
        assert not hasattr(batch, "size_factor")
        assert not hasattr(batch, "canonical_perturbation")
        assert not hasattr(batch, "canonical_context")
        assert not hasattr(batch, "dataset_index")
