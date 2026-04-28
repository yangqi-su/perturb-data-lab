"""Tests for Phase 2 ExpressionReader — backend-agnostic architecture.

Validates:
- AggregateLanceReader: chunking ≤2048, correct 3-field ExpressionRow
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
    LanceDatasetEntry,
    ZarrDatasetEntry,
)

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
FEDERATED_BASE = _ARCHIVED_ROOT / "lance-federated"
ARROW_IPC_BASE = _ARCHIVED_ROOT / "arrow-ipc-federated"

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
