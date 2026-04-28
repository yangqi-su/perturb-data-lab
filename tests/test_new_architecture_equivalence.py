"""Phase 4 smoke test: equivalence between aggregate and federated readers.

Validates that the new BatchExecutor produces identical CellState objects
(expression data, metadata) regardless of which reader backend is used
(aggregate vs federated Lance) for the same global indices.

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
    """MetadataIndex for dummy_00 + dummy_01."""
    return MetadataIndex.from_dummy_data()


@pytest.fixture(scope="module")
def agg_executor(meta: MetadataIndex) -> BatchExecutor:
    """BatchExecutor with AggregateLanceReader."""
    agg_reader = AggregateLanceReader(
        AGGREGATE_LANCE,
        [DatasetEntry("aggregated", 0, len(meta))],
    )
    return BatchExecutor(agg_reader, meta)


@pytest.fixture(scope="module")
def fed_executor(meta: MetadataIndex) -> BatchExecutor:
    """BatchExecutor with FederatedLanceReader."""
    fed_reader = FederatedLanceReader([
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
    ])
    return BatchExecutor(fed_reader, meta)


# ===================================================================
# Helper: generate 100 fixed random indices
# ===================================================================


def _sample_100_indices(meta: MetadataIndex, seed: int = 42) -> list[int]:
    """Return 100 random global indices from the metadata index."""
    sampled = meta.sample(100, seed=seed)
    return sorted(sampled["global_row_index"].to_list())


# ===================================================================
# Tests — aggregate vs federated expression equivalence
# ===================================================================


class TestAggregateFederatedEquivalence:
    """Validate that aggregate and federated readers produce identical results."""

    def test_expression_equivalence(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
        meta: MetadataIndex,
    ):
        """100 random indices: aggregate and federated produce identical CellState."""
        indices = _sample_100_indices(meta, seed=42)

        agg_cells = agg_executor.read_cells(indices)
        fed_cells = fed_executor.read_cells(indices)

        assert len(agg_cells) == len(fed_cells) == 100

        mismatches = []
        for i, (a, f) in enumerate(zip(agg_cells, fed_cells)):
            if a.expressed_gene_indices != f.expressed_gene_indices:
                mismatches.append(
                    f"index {indices[i]} gene_indices mismatch: "
                    f"agg={a.expressed_gene_indices[:5]}..., "
                    f"fed={f.expressed_gene_indices[:5]}..."
                )
            elif a.expression_counts != f.expression_counts:
                mismatches.append(
                    f"index {indices[i]} counts mismatch"
                )

        assert len(mismatches) == 0, (
            f"Found {len(mismatches)} mismatches between aggregate and federated:\n"
            + "\n".join(mismatches[:5])
        )

    def test_metadata_equivalence(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
        meta: MetadataIndex,
    ):
        """Metadata fields are identical between aggregate and federated readers."""
        indices = _sample_100_indices(meta, seed=42)

        agg_cells = agg_executor.read_cells(indices)
        fed_cells = fed_executor.read_cells(indices)

        for i, (a, f) in enumerate(zip(agg_cells, fed_cells)):
            assert a.global_row_index == f.global_row_index, f"index {i}"
            assert a.dataset_id == f.dataset_id, f"index {i}"
            assert a.dataset_index == f.dataset_index, f"index {i}"
            assert a.cell_id == f.cell_id, f"index {i}"
            assert a.size_factor == f.size_factor, f"index {i}"
            assert a.canonical_perturbation == f.canonical_perturbation, f"index {i}"
            assert a.canonical_context == f.canonical_context, f"index {i}"

    def test_expression_at_boundary(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
    ):
        """Cells at the dataset boundary are identical."""
        boundary_indices = [49998, 49999, 50000, 50001]
        agg_cells = agg_executor.read_cells(boundary_indices)
        fed_cells = fed_executor.read_cells(boundary_indices)

        for a, f in zip(agg_cells, fed_cells):
            assert a.expressed_gene_indices == f.expressed_gene_indices
            assert a.expression_counts == f.expression_counts
            assert a.dataset_id == f.dataset_id
            assert a.global_row_index == f.global_row_index

    def test_expression_distribution_consistency(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
        meta: MetadataIndex,
    ):
        """Distribution of expressed gene counts is consistent across readers."""
        indices = _sample_100_indices(meta, seed=99)

        agg_cells = agg_executor.read_cells(indices)
        fed_cells = fed_executor.read_cells(indices)

        agg_n_genes = [len(c.expressed_gene_indices) for c in agg_cells]
        fed_n_genes = [len(c.expressed_gene_indices) for c in fed_cells]

        assert agg_n_genes == fed_n_genes, (
            "Expressed gene counts differ between aggregate and federated readers"
        )

    def test_large_consistent_batch(
        self,
        agg_executor: BatchExecutor,
        fed_executor: BatchExecutor,
        meta: MetadataIndex,
    ):
        """A larger batch (200 cells) is consistent between readers."""
        indices = _sample_100_indices(meta, seed=7) + _sample_100_indices(meta, seed=77)
        indices = sorted(set(indices))  # dedupe

        agg_cells = agg_executor.read_cells(indices)
        fed_cells = fed_executor.read_cells(indices)

        for a, f in zip(agg_cells, fed_cells):
            assert a.expressed_gene_indices == f.expressed_gene_indices
            assert a.expression_counts == f.expression_counts


# ===================================================================
# Tests — deterministic reproducibility
# ===================================================================


class TestDeterministicReproducibility:
    """Validate that repeated reads produce identical results."""

    def test_aggregate_deterministic(
        self, agg_executor: BatchExecutor, meta: MetadataIndex
    ):
        """Aggregate reader: 100 indices, 3 repeated reads, identical results."""
        indices = _sample_100_indices(meta, seed=42)
        ref_cells = agg_executor.read_cells(indices)

        for run in range(3):
            cells = agg_executor.read_cells(indices)
            for i, (ref, cur) in enumerate(zip(ref_cells, cells)):
                assert ref.expressed_gene_indices == cur.expressed_gene_indices, (
                    f"run {run}, index {i}"
                )
                assert ref.expression_counts == cur.expression_counts, (
                    f"run {run}, index {i}"
                )

    def test_federated_deterministic(
        self, fed_executor: BatchExecutor, meta: MetadataIndex
    ):
        """Federated reader: 100 indices, 3 repeated reads, identical results."""
        indices = _sample_100_indices(meta, seed=42)
        ref_cells = fed_executor.read_cells(indices)

        for run in range(3):
            cells = fed_executor.read_cells(indices)
            for i, (ref, cur) in enumerate(zip(ref_cells, cells)):
                assert ref.expressed_gene_indices == cur.expressed_gene_indices, (
                    f"run {run}, index {i}"
                )
                assert ref.expression_counts == cur.expression_counts, (
                    f"run {run}, index {i}"
                )

    def test_collate_deterministic(
        self, agg_executor: BatchExecutor, meta: MetadataIndex
    ):
        """collate_sparse_batch produces identical payloads on repeated calls."""
        indices = _sample_100_indices(meta, seed=42)
        ref_payload = agg_executor.collate_sparse_batch(indices)

        for run in range(3):
            payload = agg_executor.collate_sparse_batch(indices)
            assert np.array_equal(ref_payload.expressed_gene_indices, payload.expressed_gene_indices)
            assert np.array_equal(ref_payload.expression_counts, payload.expression_counts)
            assert np.array_equal(ref_payload.row_offsets, payload.row_offsets)
            assert ref_payload.batch_size == payload.batch_size


# ===================================================================
# Tests — global/local index correctness
# ===================================================================


class TestIndexCorrectness:
    """Validate no IndexError or ValueError from index mismatch."""

    def test_no_oor_on_boundary(self, agg_executor: BatchExecutor):
        """Read exactly at the boundary does not raise out-of-range."""
        # Last valid index
        cells = agg_executor.read_cells([124999])
        assert len(cells) == 1
        assert cells[0].global_row_index == 124999

    def test_no_crash_large_random_batch(
        self, agg_executor: BatchExecutor, meta: MetadataIndex
    ):
        """A large random batch across the full range does not crash."""
        sampled = meta.sample(500, seed=123)
        indices = sampled["global_row_index"].to_list()
        cells = agg_executor.read_cells(indices)
        assert len(cells) == 500

    def test_no_crash_interleaved_large(
        self, fed_executor: BatchExecutor, meta: MetadataIndex
    ):
        """Large interleaved batch via federated reader does not crash."""
        rng = np.random.default_rng(42)
        # Mix dummy_00 and dummy_01 indices
        d00 = rng.integers(0, 50_000, size=100).tolist()
        d01 = rng.integers(50_000, 125_000, size=100).tolist()
        all_indices = d00 + d01
        rng.shuffle(all_indices)
        cells = fed_executor.read_cells(all_indices)
        assert len(cells) == 200

    def test_sequential_indices(
        self, agg_executor: BatchExecutor
    ):
        """Sequential indices across the full range work correctly."""
        indices = list(range(0, 100))  # first 100
        cells = agg_executor.read_cells(indices)
        assert len(cells) == 100
        for i, c in enumerate(cells):
            assert c.global_row_index == i

    def test_sequential_mid_range(
        self, fed_executor: BatchExecutor
    ):
        """Sequential indices in the middle of dummy_01."""
        indices = list(range(60000, 60100))
        cells = fed_executor.read_cells(indices)
        assert len(cells) == 100
        assert all(c.dataset_id == "dummy_01" for c in cells)
