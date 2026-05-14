"""Flat-only tests for backend-agnostic expression readers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from perturb_data_lab.loaders.expression import (
    AggregateLanceReader,
    AggregateZarrReader,
    DatasetEntry,
    FederatedLanceReader,
    FederatedZarrReader,
    LanceDatasetEntry,
    ZarrDatasetEntry,
    build_expression_reader,
)
from perturb_data_lab.loaders.loaders import ExpressionBatch

_ARCHIVED_ROOT = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/archive/plans-20260427-backend-topology-validation/outputs"
)

AGGREGATE_LANCE = _ARCHIVED_ROOT / "lance-aggregate/matrix/aggregated-cells.lance"
AGGREGATE_ZARR_BASE = _ARCHIVED_ROOT / "zarr-aggregate/matrix"
FEDERATED_ZARR_BASE = _ARCHIVED_ROOT / "zarr-federated"
FEDERATED_BASE = _ARCHIVED_ROOT / "lance-federated"
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

_ALL_RANGES: list[DatasetEntry] = []
_start = 0
for ds_id, size in sorted(_DATASET_SIZES.items()):
    _ALL_RANGES.append(DatasetEntry(ds_id, _start, _start + size))
    _start += size

_TEST_RANGES = [r for r in _ALL_RANGES if r.dataset_id in ("dummy_00", "dummy_01")]
_AGG_ENTRY = DatasetEntry("aggregated", 0, _start)


def _assert_batch_core(batch: ExpressionBatch, indices: list[int]) -> None:
    assert isinstance(batch, ExpressionBatch)
    assert batch.batch_size == len(indices)
    np.testing.assert_array_equal(batch.global_row_index, np.array(indices, dtype=np.int64))
    assert len(batch.row_offsets) == len(indices) + 1
    assert batch.row_offsets[0] == 0
    assert batch.row_offsets[-1] == len(batch.expressed_gene_indices)
    assert len(batch.expressed_gene_indices) == len(batch.expression_counts)
    assert batch.expressed_gene_indices.dtype == np.int32
    assert batch.expression_counts.dtype == np.int32


def _assert_row(batch: ExpressionBatch, row_position: int, *, nonempty: bool = True) -> None:
    genes = batch.row_gene_indices(row_position)
    counts = batch.row_counts(row_position)
    assert genes.dtype == np.int32
    assert counts.dtype == np.int32
    assert len(genes) == len(counts)
    if nonempty:
        assert len(genes) > 0


@pytest.fixture(scope="module")
def agg_reader() -> AggregateLanceReader:
    return AggregateLanceReader(AGGREGATE_LANCE, _TEST_RANGES)


@pytest.fixture(scope="module")
def agg_reader_full() -> AggregateLanceReader:
    return AggregateLanceReader(AGGREGATE_LANCE, _ALL_RANGES)


@pytest.fixture(scope="module")
def agg_reader_single_entry() -> AggregateLanceReader:
    return AggregateLanceReader(AGGREGATE_LANCE, [_AGG_ENTRY])


@pytest.fixture(scope="module")
def fed_reader() -> FederatedLanceReader:
    return FederatedLanceReader(
        [
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
    )


@pytest.fixture(scope="module")
def agg_zarr_reader() -> AggregateZarrReader:
    return AggregateZarrReader(
        AGGREGATE_ZARR_BASE / "aggregated-row-offsets.zarr",
        AGGREGATE_ZARR_BASE / "aggregated-indices.zarr",
        AGGREGATE_ZARR_BASE / "aggregated-counts.zarr",
        [_AGG_ENTRY],
    )


@pytest.fixture(scope="module")
def fed_zarr_reader() -> FederatedZarrReader:
    return FederatedZarrReader(
        [
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
    )


class TestAggregateLanceReaderFlatOnly:
    def test_single_cell(self, agg_reader):
        batch = agg_reader.read_expression_flat([0])
        _assert_batch_core(batch, [0])
        _assert_row(batch, 0)

    def test_cross_dataset_order(self, agg_reader):
        indices = [0, 50_000]
        batch = agg_reader.read_expression_flat(indices)
        _assert_batch_core(batch, indices)
        _assert_row(batch, 0)
        _assert_row(batch, 1)

    @pytest.mark.parametrize("n", [0, 1, 128, 2048, 2049, 4096])
    def test_chunking_boundaries(self, agg_reader, n):
        indices = list(range(n))
        batch = agg_reader.read_expression_flat(indices)
        _assert_batch_core(batch, indices)
        assert np.all(np.diff(batch.row_offsets) >= 0)

    def test_out_of_range(self, agg_reader):
        with pytest.raises(IndexError):
            agg_reader.read_expression_flat([-1])
        with pytest.raises(IndexError):
            agg_reader.read_expression_flat([935_000])
        with pytest.raises(IndexError):
            agg_reader.read_expression_flat([1_000_000])

    def test_full_ten_datasets(self, agg_reader_full):
        indices = [r.global_start + (r.global_end - r.global_start) // 2 for r in _ALL_RANGES]
        batch = agg_reader_full.read_expression_flat(indices)
        _assert_batch_core(batch, indices)
        for row_position in range(len(indices)):
            _assert_row(batch, row_position)

    def test_large_shuffled_batch(self, agg_reader_full):
        random.seed(42)
        sample = random.sample(list(range(935_000)), 5000)
        batch = agg_reader_full.read_expression_flat(sample)
        _assert_batch_core(batch, sample)

    def test_single_entry_mode(self, agg_reader_single_entry):
        indices = [0, 50_000, 99_999]
        batch = agg_reader_single_entry.read_expression_flat(indices)
        _assert_batch_core(batch, indices)

    def test_row_accessors(self, agg_reader):
        batch = agg_reader.read_expression_flat([2, 0])
        _assert_row(batch, 0)
        _assert_row(batch, 1)


class TestFederatedLanceReaderFlatOnly:
    def test_single_cell(self, fed_reader):
        batch = fed_reader.read_expression_flat([0])
        _assert_batch_core(batch, [0])
        _assert_row(batch, 0)

    def test_mixed_dataset_order(self, fed_reader):
        indices = [0, 50_000, 1, 50_001, 2, 50_002]
        batch = fed_reader.read_expression_flat(indices)
        _assert_batch_core(batch, indices)

    def test_empty_input(self, fed_reader):
        batch = fed_reader.read_expression_flat([])
        _assert_batch_core(batch, [])

    def test_large_single_dataset_chunking(self, fed_reader):
        indices = list(range(3000))
        batch = fed_reader.read_expression_flat(indices)
        _assert_batch_core(batch, indices)
        assert np.all(np.diff(batch.row_offsets) >= 0)

    def test_out_of_range(self, fed_reader):
        with pytest.raises(IndexError):
            fed_reader.read_expression_flat([-1])
        with pytest.raises(IndexError):
            fed_reader.read_expression_flat([125_000])

    def test_large_shuffled_batch(self, fed_reader):
        random.seed(42)
        indices = list(range(0, 50_000, 10)) + list(range(50_000, 125_000, 10))
        random.shuffle(indices)
        batch = fed_reader.read_expression_flat(indices)
        _assert_batch_core(batch, indices)

    def test_caching_reuses_handles(self, fed_reader):
        batch1 = fed_reader.read_expression_flat([0, 50_000])
        batch2 = fed_reader.read_expression_flat([1, 50_001])
        _assert_batch_core(batch1, [0, 50_000])
        _assert_batch_core(batch2, [1, 50_001])


class TestSlimReaderFlatContract:
    @pytest.mark.parametrize(
        ("fixture_name", "indices"),
        [
            ("agg_zarr_reader", [100, 5, 50_000, 99_999]),
            ("fed_zarr_reader", [0, 50_000, 1, 50_001]),
        ],
    )
    def test_flat_reader_smoke(self, request, fixture_name: str, indices: list[int]):
        reader = request.getfixturevalue(fixture_name)
        batch = reader.read_expression_flat(indices)
        _assert_batch_core(batch, indices)
        for row_position in range(len(indices)):
            _assert_row(batch, row_position)

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "agg_zarr_reader",
            "fed_zarr_reader",
        ],
    )
    def test_flat_reader_empty_input(self, request, fixture_name: str):
        reader = request.getfixturevalue(fixture_name)
        batch = reader.read_expression_flat([])
        _assert_batch_core(batch, [])

    @pytest.mark.parametrize(
        "backend",
        ["arrow_ipc", "hf_datasets", "parquet", "webdataset", "tiledb", "csr_memmap"],
    )
    def test_removed_backends_raise_clear_error(self, backend: str) -> None:
        with pytest.raises(ValueError, match="not supported in slim main"):
            build_expression_reader(backend, "federated", [DatasetEntry("dummy", 0, 1)])
