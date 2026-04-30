"""Phase 4 smoke test: equivalence between aggregate and federated readers.

Validates that BatchExecutor produces identical read_batch() / read_expression_batch()
output regardless of which reader backend is used (aggregate vs federated Lance)
for the same global indices.

Also validates deterministic reproducibility on repeated reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perturb_data_lab.loaders import (
    AggregateLanceReader,
    BatchExecutor,
    DatasetEntry,
    FederatedLanceReader,
    LanceDatasetEntry,
    MetadataIndex,
)

# ===================================================================
# Constants
# ===================================================================

_ARCHIVED_ROOT = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/archive/plans-20260427-backend-topology-validation/outputs"
)

AGGREGATE_LANCE = str(
    _ARCHIVED_ROOT / "lance-aggregate/matrix/aggregated-cells.lance"
)

FEDERATED_BASE = _ARCHIVED_ROOT / "lance-federated"


# ===================================================================
# Module-scoped fixtures
# ===================================================================


@pytest.fixture(scope="module")
def meta() -> MetadataIndex:
    return MetadataIndex.from_dummy_data()


@pytest.fixture(scope="module")
def agg_executor(meta: MetadataIndex) -> BatchExecutor:
    agg_reader = AggregateLanceReader(
        AGGREGATE_LANCE,
        [DatasetEntry("aggregated", 0, len(meta))],
    )
    return BatchExecutor(agg_reader, meta)


@pytest.fixture(scope="module")
def fed_executor(meta: MetadataIndex) -> BatchExecutor:
    fed_reader = FederatedLanceReader([
        LanceDatasetEntry(
            "dummy_00", 0, 50_000,
            FEDERATED_BASE / "dummy_00/matrix/dummy_00-release.lance",
        ),
        LanceDatasetEntry(
            "dummy_01", 50_000, 125_000,
            FEDERATED_BASE / "dummy_01/matrix/dummy_01-release.lance",
        ),
    ])
    return BatchExecutor(fed_reader, meta)


# ===================================================================
# Helper
# ===================================================================


def _sample_100_indices(meta: MetadataIndex, seed: int = 42) -> list[int]:
    sampled = meta.sample(100, seed=seed)
    return sorted(sampled["global_row_index"].to_list())


# ===================================================================
# Tests — aggregate vs federated expression equivalence
# ===================================================================


class TestAggregateFederatedEquivalence:
    """Validate that aggregate and federated readers produce identical read_batch output."""

    def test_expression_equivalence(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
        meta: MetadataIndex,
    ):
        """100 random indices: aggregate and federated produce identical expression data."""
        indices = _sample_100_indices(meta, seed=42)

        agg_batch = agg_executor.read_expression_batch(indices)
        fed_batch = fed_executor.read_expression_batch(indices)

        assert agg_batch.batch_size == fed_batch.batch_size == 100

        np.testing.assert_array_equal(
            agg_batch.global_row_index, fed_batch.global_row_index
        )
        np.testing.assert_array_equal(
            agg_batch.expressed_gene_indices, fed_batch.expressed_gene_indices
        )
        np.testing.assert_array_equal(
            agg_batch.expression_counts, fed_batch.expression_counts
        )
        np.testing.assert_array_equal(
            agg_batch.row_offsets, fed_batch.row_offsets
        )

    def test_metadata_equivalence(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
        meta: MetadataIndex,
    ):
        """Metadata fields are identical between aggregate and federated readers."""
        indices = _sample_100_indices(meta, seed=42)

        agg_meta = agg_executor.read_metadata_batch(indices)
        fed_meta = fed_executor.read_metadata_batch(indices)

        np.testing.assert_array_equal(
            agg_meta["global_row_index"], fed_meta["global_row_index"]
        )
        np.testing.assert_array_equal(
            agg_meta["dataset_index"], fed_meta["dataset_index"]
        )
        np.testing.assert_array_equal(
            agg_meta["size_factor"], fed_meta["size_factor"]
        )
        assert agg_meta["dataset_id"] == fed_meta["dataset_id"]
        assert agg_meta["cell_id"] == fed_meta["cell_id"]
        assert agg_meta["canonical_perturbation"] == fed_meta["canonical_perturbation"]
        assert agg_meta["canonical_context"] == fed_meta["canonical_context"]

    def test_read_batch_equivalence(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
        meta: MetadataIndex,
    ):
        """read_batch() returns identical results from aggregate and federated readers."""
        indices = _sample_100_indices(meta, seed=42)

        agg_batch = agg_executor.read_batch(indices)
        fed_batch = fed_executor.read_batch(indices)

        assert agg_batch["batch_size"] == fed_batch["batch_size"]
        np.testing.assert_array_equal(
            agg_batch["global_row_index"], fed_batch["global_row_index"]
        )
        np.testing.assert_array_equal(
            agg_batch["expressed_gene_indices"], fed_batch["expressed_gene_indices"]
        )
        np.testing.assert_array_equal(
            agg_batch["expression_counts"], fed_batch["expression_counts"]
        )
        np.testing.assert_array_equal(
            agg_batch["row_offsets"], fed_batch["row_offsets"]
        )
        np.testing.assert_array_equal(
            agg_batch["dataset_index"], fed_batch["dataset_index"]
        )
        np.testing.assert_array_equal(
            agg_batch["size_factor"], fed_batch["size_factor"]
        )
        assert agg_batch["dataset_id"] == fed_batch["dataset_id"]
        assert agg_batch["cell_id"] == fed_batch["cell_id"]

    def test_expression_at_boundary(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
    ):
        """Cells at the dataset boundary are identical."""
        boundary_indices = [49998, 49999, 50000, 50001]
        agg_expr = agg_executor.read_expression_batch(boundary_indices)
        fed_expr = fed_executor.read_expression_batch(boundary_indices)

        np.testing.assert_array_equal(
            agg_expr.expressed_gene_indices, fed_expr.expressed_gene_indices
        )
        np.testing.assert_array_equal(
            agg_expr.expression_counts, fed_expr.expression_counts
        )
        np.testing.assert_array_equal(
            agg_expr.global_row_index, fed_expr.global_row_index
        )

    def test_expression_distribution_consistency(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
        meta: MetadataIndex,
    ):
        """Distribution of expressed gene counts is consistent across readers."""
        indices = _sample_100_indices(meta, seed=99)

        agg_expr = agg_executor.read_expression_batch(indices)
        fed_expr = fed_executor.read_expression_batch(indices)

        # row_offsets encode per-cell gene counts
        agg_n_genes = np.diff(agg_expr.row_offsets)
        fed_n_genes = np.diff(fed_expr.row_offsets)
        np.testing.assert_array_equal(agg_n_genes, fed_n_genes)

    def test_large_consistent_batch(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
        meta: MetadataIndex,
    ):
        """A larger batch (200 cells) is consistent between readers."""
        indices = _sample_100_indices(meta, seed=7) + _sample_100_indices(meta, seed=77)
        indices = sorted(set(indices))

        agg_batch = agg_executor.read_batch(indices)
        fed_batch = fed_executor.read_batch(indices)

        np.testing.assert_array_equal(
            agg_batch["expressed_gene_indices"], fed_batch["expressed_gene_indices"]
        )
        np.testing.assert_array_equal(
            agg_batch["expression_counts"], fed_batch["expression_counts"]
        )
        np.testing.assert_array_equal(
            agg_batch["row_offsets"], fed_batch["row_offsets"]
        )


# ===================================================================
# Tests — deterministic reproducibility
# ===================================================================


class TestDeterministicReproducibility:
    """Validate that repeated read_batch calls produce identical results."""

    def test_aggregate_deterministic(
        self, agg_executor: BatchExecutor, meta: MetadataIndex
    ):
        """Aggregate reader: 100 indices, 3 repeated read_batch calls, identical results."""
        indices = _sample_100_indices(meta, seed=42)
        ref_batch = agg_executor.read_batch(indices)

        for run in range(3):
            batch = agg_executor.read_batch(indices)
            np.testing.assert_array_equal(
                ref_batch["expressed_gene_indices"], batch["expressed_gene_indices"]
            )
            np.testing.assert_array_equal(
                ref_batch["expression_counts"], batch["expression_counts"]
            )
            np.testing.assert_array_equal(
                ref_batch["row_offsets"], batch["row_offsets"]
            )
            assert ref_batch["dataset_id"] == batch["dataset_id"]

    def test_federated_deterministic(
        self, fed_executor: BatchExecutor, meta: MetadataIndex
    ):
        """Federated reader: 100 indices, 3 repeated read_batch calls, identical results."""
        indices = _sample_100_indices(meta, seed=42)
        ref_batch = fed_executor.read_batch(indices)

        for run in range(3):
            batch = fed_executor.read_batch(indices)
            np.testing.assert_array_equal(
                ref_batch["expressed_gene_indices"], batch["expressed_gene_indices"]
            )
            np.testing.assert_array_equal(
                ref_batch["expression_counts"], batch["expression_counts"]
            )
            np.testing.assert_array_equal(
                ref_batch["row_offsets"], batch["row_offsets"]
            )
            assert ref_batch["dataset_id"] == batch["dataset_id"]

    def test_expression_batch_deterministic(
        self, agg_executor: BatchExecutor, meta: MetadataIndex
    ):
        """read_expression_batch produces identical ExpressionBatch on repeated calls."""
        indices = _sample_100_indices(meta, seed=42)
        ref_expr = agg_executor.read_expression_batch(indices)

        for run in range(3):
            expr = agg_executor.read_expression_batch(indices)
            np.testing.assert_array_equal(
                ref_expr.expressed_gene_indices, expr.expressed_gene_indices
            )
            np.testing.assert_array_equal(
                ref_expr.expression_counts, expr.expression_counts
            )
            np.testing.assert_array_equal(
                ref_expr.row_offsets, expr.row_offsets
            )
            assert ref_expr.batch_size == expr.batch_size


# ===================================================================
# Tests — global/local index correctness
# ===================================================================


class TestIndexCorrectness:
    """Validate no IndexError or ValueError from index mismatch."""

    def test_no_oor_on_boundary(self, agg_executor: BatchExecutor):
        """Read exactly at the boundary does not raise out-of-range."""
        batch = agg_executor.read_batch([124999])
        assert batch["batch_size"] == 1
        assert batch["global_row_index"][0] == 124999

    def test_no_crash_large_random_batch(
        self, agg_executor: BatchExecutor, meta: MetadataIndex
    ):
        """A large random batch across the full range does not crash."""
        sampled = meta.sample(500, seed=123)
        indices = sampled["global_row_index"].to_list()
        batch = agg_executor.read_batch(indices)
        assert batch["batch_size"] == 500

    def test_no_crash_interleaved_large(
        self, fed_executor: BatchExecutor
    ):
        """Large interleaved batch via federated reader does not crash."""
        rng = np.random.default_rng(42)
        d00 = rng.integers(0, 50_000, size=100).tolist()
        d01 = rng.integers(50_000, 125_000, size=100).tolist()
        all_indices = d00 + d01
        rng.shuffle(all_indices)
        batch = fed_executor.read_batch(all_indices)
        assert batch["batch_size"] == 200

    def test_sequential_indices(self, agg_executor: BatchExecutor):
        """Sequential indices across the full range work correctly."""
        indices = list(range(0, 100))
        batch = agg_executor.read_batch(indices)
        assert batch["batch_size"] == 100
        np.testing.assert_array_equal(
            batch["global_row_index"],
            np.arange(100, dtype=np.int64),
        )

    def test_sequential_mid_range(self, fed_executor: BatchExecutor):
        """Sequential indices in the middle of dummy_01 via federated reader."""
        indices = list(range(60000, 60100))
        batch = fed_executor.read_batch(indices)
        assert batch["batch_size"] == 100
        for ds_id in batch["dataset_id"]:
            assert ds_id == "dummy_01"
