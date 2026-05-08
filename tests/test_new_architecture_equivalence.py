"""Phase 4 smoke test: equivalence between aggregate and federated readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perturb_data_lab.loaders import (
    AggregateLanceReader,
    DatasetEntry,
    FederatedLanceReader,
    LanceDatasetEntry,
    MetadataIndex,
)
from perturb_data_lab.loaders.corpus_loader import _read_raw_batch

_ARCHIVED_ROOT = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/archive/plans-20260427-backend-topology-validation/outputs"
)

AGGREGATE_LANCE = str(
    _ARCHIVED_ROOT / "lance-aggregate/matrix/aggregated-cells.lance"
)

FEDERATED_BASE = _ARCHIVED_ROOT / "lance-federated"


@pytest.fixture(scope="module")
def meta() -> MetadataIndex:
    return MetadataIndex.from_dummy_data()


@pytest.fixture(scope="module")
def agg_reader(meta: MetadataIndex) -> AggregateLanceReader:
    return AggregateLanceReader(
        AGGREGATE_LANCE,
        [DatasetEntry("aggregated", 0, len(meta))],
    )


@pytest.fixture(scope="module")
def fed_reader() -> FederatedLanceReader:
    return FederatedLanceReader([
        LanceDatasetEntry(
            "dummy_00", 0, 50_000,
            FEDERATED_BASE / "dummy_00/matrix/dummy_00-release.lance",
        ),
        LanceDatasetEntry(
            "dummy_01", 50_000, 125_000,
            FEDERATED_BASE / "dummy_01/matrix/dummy_01-release.lance",
        ),
    ])


def _sample_100_indices(meta: MetadataIndex, seed: int = 42) -> list[int]:
    sampled = meta.sample(100, seed=seed)
    return sorted(sampled["global_row_index"].to_list())


class TestAggregateFederatedEquivalence:
    def test_expression_equivalence(
        self,
        agg_reader: AggregateLanceReader,
        fed_reader: FederatedLanceReader,
        meta: MetadataIndex,
    ) -> None:
        indices = _sample_100_indices(meta, seed=42)

        agg_batch = agg_reader.read_expression_flat(indices)
        fed_batch = fed_reader.read_expression_flat(indices)

        assert agg_batch.batch_size == fed_batch.batch_size == 100
        np.testing.assert_array_equal(agg_batch.global_row_index, fed_batch.global_row_index)
        np.testing.assert_array_equal(agg_batch.expressed_gene_indices, fed_batch.expressed_gene_indices)
        np.testing.assert_array_equal(agg_batch.expression_counts, fed_batch.expression_counts)
        np.testing.assert_array_equal(agg_batch.row_offsets, fed_batch.row_offsets)

    def test_raw_batch_equivalence(
        self,
        agg_reader: AggregateLanceReader,
        fed_reader: FederatedLanceReader,
        meta: MetadataIndex,
    ) -> None:
        indices = _sample_100_indices(meta, seed=42)

        agg_batch = _read_raw_batch(
            agg_reader,
            meta,
            indices,
            metadata_columns=["perturb_label", "cell_id"],
        )
        fed_batch = _read_raw_batch(
            fed_reader,
            meta,
            indices,
            metadata_columns=["perturb_label", "cell_id"],
        )

        assert agg_batch["batch_size"] == fed_batch["batch_size"] == 100
        for key in (
            "global_row_index",
            "dataset_index",
            "local_row_index",
            "size_factor",
            "row_offsets",
            "expressed_gene_indices",
            "expression_counts",
        ):
            np.testing.assert_array_equal(agg_batch[key], fed_batch[key])
        assert agg_batch["meta_columns"] == fed_batch["meta_columns"]

    def test_expression_at_boundary(
        self,
        agg_reader: AggregateLanceReader,
        fed_reader: FederatedLanceReader,
    ) -> None:
        boundary_indices = [49_998, 49_999, 50_000, 50_001]
        agg_expr = agg_reader.read_expression_flat(boundary_indices)
        fed_expr = fed_reader.read_expression_flat(boundary_indices)

        np.testing.assert_array_equal(agg_expr.expressed_gene_indices, fed_expr.expressed_gene_indices)
        np.testing.assert_array_equal(agg_expr.expression_counts, fed_expr.expression_counts)
        np.testing.assert_array_equal(agg_expr.global_row_index, fed_expr.global_row_index)

    def test_expression_distribution_consistency(
        self,
        agg_reader: AggregateLanceReader,
        fed_reader: FederatedLanceReader,
        meta: MetadataIndex,
    ) -> None:
        indices = _sample_100_indices(meta, seed=99)

        agg_expr = agg_reader.read_expression_flat(indices)
        fed_expr = fed_reader.read_expression_flat(indices)

        np.testing.assert_array_equal(np.diff(agg_expr.row_offsets), np.diff(fed_expr.row_offsets))

    def test_repeated_reads_are_deterministic(
        self,
        agg_reader: AggregateLanceReader,
        fed_reader: FederatedLanceReader,
        meta: MetadataIndex,
    ) -> None:
        indices = _sample_100_indices(meta, seed=7)

        agg_first = agg_reader.read_expression_flat(indices)
        agg_second = agg_reader.read_expression_flat(indices)
        fed_first = fed_reader.read_expression_flat(indices)
        fed_second = fed_reader.read_expression_flat(indices)

        for first, second in ((agg_first, agg_second), (fed_first, fed_second)):
            np.testing.assert_array_equal(first.global_row_index, second.global_row_index)
            np.testing.assert_array_equal(first.row_offsets, second.row_offsets)
            np.testing.assert_array_equal(first.expressed_gene_indices, second.expressed_gene_indices)
            np.testing.assert_array_equal(first.expression_counts, second.expression_counts)
