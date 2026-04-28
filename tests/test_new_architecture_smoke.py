"""Phase 4 smoke test: aggregate Lance corpus with all three sampler modes.

Exercises the full loader stack (MetadataIndex + ExpressionReader → BatchExecutor
→ PerturbDataLoader/PerturbIterableDataset) on the aggregate Lance topology.

Runs 10 batches per sampler mode and validates:
- Batch sizes and shapes
- canonical_perturbation and canonical_context populated correctly
- No IndexError or ValueError from global/local index mismatch
- Both map-style (PerturbDataLoader) and streaming (PerturbIterableDataset) APIs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perturb_data_lab.loaders import (
    AggregateLanceReader,
    BatchExecutor,
    DatasetEntry,
    MetadataIndex,
    PerturbDataLoader,
    PerturbIterableDataset,
    SparseBatchCollator,
    SparseBatchPayload,
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

# ===================================================================
# Module-scoped fixtures (loaded once)
# ===================================================================


@pytest.fixture(scope="module")
def meta() -> MetadataIndex:
    """MetadataIndex for dummy_00 (50K) + dummy_01 (75K)."""
    return MetadataIndex.from_dummy_data()


@pytest.fixture(scope="module")
def agg_reader() -> AggregateLanceReader:
    """AggregateLanceReader limited to dummy_00 + dummy_01 range."""
    start = 0
    end = 50_000 + 75_000  # 125_000
    return AggregateLanceReader(
        AGGREGATE_LANCE, [DatasetEntry("aggregated", start, end)]
    )


@pytest.fixture(scope="module")
def executor(agg_reader: AggregateLanceReader, meta: MetadataIndex) -> BatchExecutor:
    """BatchExecutor composing AggregateLanceReader + MetadataIndex."""
    return BatchExecutor(agg_reader, meta)


# ===================================================================
# HVG set helper — load from sidecar files
# ===================================================================


def _build_hvg_set() -> tuple[int, ...]:
    """Load HVG gene indices from dummy_00 + dummy_01 sidecar npy files."""
    hvg_paths = [
        _ARCHIVED_ROOT / "lance-federated/dummy_00/metadata/hvg_sidecar/hvg.npy",
        _ARCHIVED_ROOT / "lance-federated/dummy_01/metadata/hvg_sidecar/hvg.npy",
    ]
    hvg_all: set[int] = set()
    for p in hvg_paths:
        arr = np.load(p)
        hvg_all.update(int(x) for x in arr)
    return tuple(sorted(hvg_all))


# ===================================================================
# Shared validation helpers
# ===================================================================


def _validate_batch_item(item: dict, sampler_mode: str) -> None:
    """Validate a single item dict returned by PerturbDataLoader / PerturbIterableDataset."""
    # Required keys
    assert "expressed_gene_indices" in item
    assert "expression_counts" in item
    assert "context_indices" in item
    assert "cell_id" in item
    assert "dataset_id" in item
    assert "dataset_index" in item
    assert "global_row_index" in item
    assert "size_factor" in item
    assert "canonical_perturbation" in item
    assert "canonical_context" in item

    # Type checks
    assert isinstance(item["cell_id"], str)
    assert isinstance(item["dataset_id"], str)
    assert item["dataset_index"] in (0, 1)
    assert 0 <= item["global_row_index"] < 125_000
    assert item["size_factor"] > 0.0

    # Expression arrays match in length
    assert isinstance(item["expressed_gene_indices"], np.ndarray)
    assert isinstance(item["expression_counts"], np.ndarray)
    assert len(item["expressed_gene_indices"]) == len(item["expression_counts"])
    assert len(item["expressed_gene_indices"]) > 0

    # Context indices
    assert isinstance(item["context_indices"], np.ndarray)
    if sampler_mode == "random_context":
        assert len(item["context_indices"]) > 0
    elif sampler_mode == "expressed_zeros":
        assert len(item["context_indices"]) > 0
    elif sampler_mode == "hvg_random":
        assert len(item["context_indices"]) > 0

    # Canonical metadata populated
    assert isinstance(item["canonical_perturbation"], dict)
    assert isinstance(item["canonical_context"], dict)
    # Dummy data should have at least guide_1 in perturbation
    assert "guide_1" in item["canonical_perturbation"]


def _validate_sparse_payload(payload: SparseBatchPayload, batch_size: int) -> None:
    """Validate a SparseBatchPayload produced by collate_sparse_batch."""
    assert isinstance(payload, SparseBatchPayload)
    assert payload.batch_size == batch_size
    assert payload.global_row_index.shape == (batch_size,)
    assert payload.dataset_index.shape == (batch_size,)
    assert payload.size_factor.shape == (batch_size,)
    assert len(payload.dataset_id) == batch_size
    assert len(payload.cell_id) == batch_size

    # Row offsets must be valid
    assert payload.row_offsets[0] == 0
    assert payload.row_offsets[-1] == len(payload.expressed_gene_indices)

    # Each row slice must be valid
    for i in range(batch_size):
        s = payload.row_slice(i)
        assert (
            payload.expressed_gene_indices[s].size
            == payload.expression_counts[s].size
        )


# ===================================================================
# Tests — PerturbDataLoader with all three sampler modes
# ===================================================================


class TestAggregateLanceSmoke:
    """Smoke test: aggregate Lance corpus via PerturbDataLoader."""

    @pytest.mark.parametrize(
        "sampler_mode,kwargs",
        [
            ("random_context", {"context_size": 100}),
            # max_context=None uses default (n_genes-based) sizing to avoid
            # negative-size errors when n_expressed > max_context
            ("expressed_zeros", {"max_context": None}),
            ("hvg_random", {"max_context": None, "hvg_set": _build_hvg_set()}),
        ],
    )
    def test_dataloader_10_batches(
        self,
        executor: BatchExecutor,
        sampler_mode: str,
        kwargs: dict,
    ):
        """Run 10 batches via PerturbDataLoader.__getitems__ for each sampler mode."""
        dl = PerturbDataLoader(
            executor,
            n_genes=5000,
            sampler_mode=sampler_mode,
            seed=42,
            **kwargs,
        )

        rng = np.random.default_rng(42)
        for batch_num in range(10):
            batch_size = 32
            indices = rng.integers(0, len(executor), size=batch_size).tolist()
            items = dl.__getitems__(indices)
            assert len(items) == batch_size, (
                f"batch {batch_num}: expected {batch_size}, got {len(items)}"
            )
            for item in items:
                _validate_batch_item(item, sampler_mode)

    @pytest.mark.parametrize(
        "sampler_mode,kwargs",
        [
            ("random_context", {"context_size": 100}),
            ("expressed_zeros", {"max_context": None}),
            ("hvg_random", {"max_context": None, "hvg_set": _build_hvg_set()}),
        ],
    )
    def test_iterable_dataset_10_batches(
        self,
        executor: BatchExecutor,
        sampler_mode: str,
        kwargs: dict,
    ):
        """Iterate 10 cells via PerturbIterableDataset for each sampler mode."""
        ds = PerturbIterableDataset(
            executor,
            n_genes=5000,
            sampler_mode=sampler_mode,
            shuffle=False,
            seed=42,
            **kwargs,
        )

        count = 0
        for item in ds:
            _validate_batch_item(item, sampler_mode)
            count += 1
            if count >= 10:
                break
        assert count == 10


# ===================================================================
# Tests — collate_sparse_batch & direct BatchExecutor usage
# ===================================================================


class TestAggregateLanceCollation:
    """Smoke test: collate_sparse_batch and direct BatchExecutor.read_cells."""

    def test_collate_10_random_batches(self, executor: BatchExecutor, meta: MetadataIndex):
        """Collate 10 random batches into SparseBatchPayload."""
        rng = np.random.default_rng(123)
        for batch_num in range(10):
            batch_size = 32
            sampled = meta.sample(batch_size, seed=123 + batch_num)
            indices = sampled["global_row_index"].to_list()
            payload = executor.collate_sparse_batch(indices)
            _validate_sparse_payload(payload, batch_size)

    def test_read_cells_100_random(self, executor: BatchExecutor, meta: MetadataIndex):
        """Read 100 random cells and validate CellState consistency."""
        sampled = meta.sample(100, seed=99)
        indices = sampled["global_row_index"].to_list()
        cells = executor.read_cells(indices)
        assert len(cells) == 100

        for c in cells:
            assert c.global_row_index in indices
            assert c.dataset_id in ("dummy_00", "dummy_01")
            assert c.dataset_index in (0, 1)
            assert c.size_factor > 0.0
            assert len(c.expressed_gene_indices) == len(c.expression_counts)
            assert len(c.expressed_gene_indices) > 0
            assert isinstance(c.canonical_perturbation, dict)
            assert isinstance(c.canonical_context, dict)
            assert "guide_1" in c.canonical_perturbation

    def test_read_cells_deterministic(self, executor: BatchExecutor):
        """Same indices produce identical CellState objects on repeated reads."""
        indices = [0, 1, 2, 50000, 50001, 99999, 124999]
        cells_1 = executor.read_cells(indices)
        cells_2 = executor.read_cells(indices)
        assert len(cells_1) == len(cells_2) == len(indices)

        for c1, c2 in zip(cells_1, cells_2):
            assert c1.global_row_index == c2.global_row_index
            assert c1.dataset_id == c2.dataset_id
            assert c1.dataset_index == c2.dataset_index
            assert c1.cell_id == c2.cell_id
            assert c1.expressed_gene_indices == c2.expressed_gene_indices
            assert c1.expression_counts == c2.expression_counts
            assert c1.size_factor == c2.size_factor
            assert c1.canonical_perturbation == c2.canonical_perturbation
            assert c1.canonical_context == c2.canonical_context


# ===================================================================
# Tests — canonical_perturbation and canonical_context
# ===================================================================


class TestCanonicalMetadata:
    """Validate canonical_perturbation and canonical_context correctness."""

    def test_perturbation_fields(self, executor: BatchExecutor):
        """canonical_perturbation contains expected keys from dummy data."""
        cells = executor.read_cells([0, 10, 50000, 75000])
        for c in cells:
            pert = c.canonical_perturbation
            # Dummy data has: guide_1, guide_2, treatment, site, genotype
            assert "guide_1" in pert
            assert isinstance(pert["guide_1"], str)

    def test_context_fields(self, executor: BatchExecutor):
        """canonical_context contains expected keys from dummy data."""
        cells = executor.read_cells([0, 10, 50000, 75000])
        for c in cells:
            ctx = c.canonical_context
            assert "cell_type" in ctx
            assert isinstance(ctx["cell_type"], str)

    def test_raw_fields_not_empty(self, executor: BatchExecutor):
        """raw_fields contains non-canonical columns."""
        cells = executor.read_cells([0])
        c = cells[0]
        assert isinstance(c.raw_fields, dict)
        # raw_fields should have some non-canonical entries
        # at minimum: doublet_score, pct_mito, n_counts, n_features, scrublet_score, etc.
        assert len(c.raw_fields) >= 1


# ===================================================================
# Tests — edge cases
# ===================================================================


class TestEdgeCases:
    """Smoke test: edge cases for the new architecture."""

    def test_empty_batch(self, executor: BatchExecutor):
        """Empty index list returns empty payload."""
        payload = executor.collate_sparse_batch([])
        assert payload.batch_size == 0

    def test_single_cell_batch(self, executor: BatchExecutor):
        """Single cell batch produces valid payload."""
        payload = executor.collate_sparse_batch([0])
        assert payload.batch_size == 1

    def test_cross_dataset_batch(self, executor: BatchExecutor, meta: MetadataIndex):
        """Batch spanning dummy_00 and dummy_01 boundary."""
        # Indices near the 50K boundary
        indices = [49998, 49999, 50000, 50001]
        cells = executor.read_cells(indices)
        assert len(cells) == 4
        assert cells[0].dataset_id == "dummy_00"
        assert cells[1].dataset_id == "dummy_00"
        assert cells[2].dataset_id == "dummy_01"
        assert cells[3].dataset_id == "dummy_01"

    def test_dataset_boundary_indices(self, executor: BatchExecutor):
        """First and last cell of each dataset."""
        indices = [0, 49999, 50000, 124999]
        cells = executor.read_cells(indices)
        assert cells[0].global_row_index == 0
        assert cells[1].global_row_index == 49999
        assert cells[2].global_row_index == 50000
        assert cells[3].global_row_index == 124999

    def test_hvg_sampler_has_hvg_set(self, executor: BatchExecutor):
        """HVGRandomSampler produces context containing at least some HVG genes."""
        hvg_set = _build_hvg_set()
        dl = PerturbDataLoader(
            executor,
            n_genes=5000,
            sampler_mode="hvg_random",
            seed=42,
            max_context=None,  # default n_genes-based sizing to avoid negative-size errors
            hvg_set=hvg_set,
        )
        # Sample multiple cells to check HVG coverage
        rng = np.random.default_rng(99)
        hvg_in_context_count = 0
        for _ in range(20):
            idx = int(rng.integers(0, len(executor)))
            item = dl[idx]
            context = set(item["context_indices"])
            hvg_overlap = context & set(hvg_set)
            if hvg_overlap:
                hvg_in_context_count += 1
        # At least some cells should have HVG genes in context
        assert hvg_in_context_count > 0, "HVG sampler should include HVG genes"
