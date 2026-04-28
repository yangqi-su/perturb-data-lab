"""Phase 4 smoke test: federated Lance corpus with all three sampler modes.

Exercises the full loader stack (MetadataIndex + FederatedLanceReader → BatchExecutor
→ PerturbDataLoader/PerturbIterableDataset) on the federated Lance topology.

Runs 10 batches per sampler mode and validates:
- Batch sizes and shapes
- canonical_perturbation and canonical_context populated correctly
- No IndexError or ValueError from global/local index mismatch
- Interleaved dataset batches are handled correctly
- Both map-style and streaming APIs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perturb_data_lab.loaders import (
    BatchExecutor,
    FederatedLanceReader,
    LanceDatasetEntry,
    MetadataIndex,
    PerturbDataLoader,
    PerturbIterableDataset,
    SparseBatchPayload,
)

# ===================================================================
# Constants
# ===================================================================

_ARCHIVED_ROOT = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/archive/plans-20260427-backend-topology-validation/outputs"
)

FEDERATED_BASE = _ARCHIVED_ROOT / "lance-federated"


def _fed_entries() -> list[LanceDatasetEntry]:
    """Federated Lance entries for dummy_00 (50K) and dummy_01 (75K)."""
    return [
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


# ===================================================================
# Module-scoped fixtures
# ===================================================================


@pytest.fixture(scope="module")
def meta() -> MetadataIndex:
    """MetadataIndex for dummy_00 (50K) + dummy_01 (75K)."""
    return MetadataIndex.from_dummy_data()


@pytest.fixture(scope="module")
def fed_reader() -> FederatedLanceReader:
    """FederatedLanceReader for dummy_00 + dummy_01."""
    return FederatedLanceReader(_fed_entries())


@pytest.fixture(scope="module")
def executor(
    fed_reader: FederatedLanceReader, meta: MetadataIndex
) -> BatchExecutor:
    """BatchExecutor composing FederatedLanceReader + MetadataIndex."""
    return BatchExecutor(fed_reader, meta)


# ===================================================================
# HVG set helper
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
    """Validate a single item dict returned by PerturbDataLoader."""
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

    assert isinstance(item["cell_id"], str)
    assert isinstance(item["dataset_id"], str)
    assert item["dataset_index"] in (0, 1)
    assert 0 <= item["global_row_index"] < 125_000
    assert item["size_factor"] > 0.0
    assert len(item["expressed_gene_indices"]) == len(item["expression_counts"])
    assert len(item["expressed_gene_indices"]) > 0
    assert len(item["context_indices"]) > 0
    assert isinstance(item["canonical_perturbation"], dict)
    assert isinstance(item["canonical_context"], dict)
    assert "guide_1" in item["canonical_perturbation"]


def _validate_sparse_payload(payload: SparseBatchPayload, batch_size: int) -> None:
    """Validate a SparseBatchPayload."""
    assert isinstance(payload, SparseBatchPayload)
    assert payload.batch_size == batch_size
    assert payload.global_row_index.shape == (batch_size,)
    assert payload.dataset_index.shape == (batch_size,)
    assert payload.size_factor.shape == (batch_size,)
    assert payload.row_offsets[0] == 0
    assert payload.row_offsets[-1] == len(payload.expressed_gene_indices)
    for i in range(batch_size):
        s = payload.row_slice(i)
        assert (
            payload.expressed_gene_indices[s].size
            == payload.expression_counts[s].size
        )


# ===================================================================
# Tests — PerturbDataLoader with all three sampler modes
# ===================================================================


class TestFederatedLanceSmoke:
    """Smoke test: federated Lance corpus via PerturbDataLoader."""

    @pytest.mark.parametrize(
        "sampler_mode,kwargs",
        [
            ("random_context", {"context_size": 100}),
            # max_context=None uses default n_genes-based sizing
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
    def test_iterable_dataset_10_cells(
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
# Tests — collation and direct BatchExecutor
# ===================================================================


class TestFederatedLanceCollation:
    """Smoke test: collate_sparse_batch and direct BatchExecutor for federated."""

    def test_collate_10_random_batches(self, executor: BatchExecutor, meta: MetadataIndex):
        """Collate 10 random batches via federated reader."""
        rng = np.random.default_rng(456)
        for batch_num in range(10):
            batch_size = 32
            sampled = meta.sample(batch_size, seed=456 + batch_num)
            indices = sampled["global_row_index"].to_list()
            payload = executor.collate_sparse_batch(indices)
            _validate_sparse_payload(payload, batch_size)


# ===================================================================
# Tests — interleaved dataset batches (federated-specific)
# ===================================================================


class TestFederatedInterleavedBatches:
    """Federated-specific: interleaved batches spanning multiple datasets."""

    def test_interleaved_order_preserved(self, executor: BatchExecutor):
        """Output order matches input order for interleaved dataset indices."""
        indices = [0, 50000, 1, 50001, 2, 50002]
        cells = executor.read_cells(indices)
        assert len(cells) == 6
        assert [c.global_row_index for c in cells] == indices

    def test_interleaved_dataset_ids(self, executor: BatchExecutor):
        """Dataset IDs are correct for interleaved indices."""
        indices = [0, 50000, 1, 50001]
        cells = executor.read_cells(indices)
        assert cells[0].dataset_id == "dummy_00"
        assert cells[1].dataset_id == "dummy_01"
        assert cells[2].dataset_id == "dummy_00"
        assert cells[3].dataset_id == "dummy_01"

    def test_interleaved_expression_content(self, executor: BatchExecutor):
        """Each cell in an interleaved batch has valid expression data."""
        indices = [0, 50000, 10, 50010, 100, 50100]
        cells = executor.read_cells(indices)
        for c in cells:
            assert len(c.expressed_gene_indices) > 0
            assert len(c.expression_counts) > 0
            assert len(c.expressed_gene_indices) == len(c.expression_counts)
            assert all(isinstance(g, int) for g in c.expressed_gene_indices)
            assert all(isinstance(cnt, int) for cnt in c.expression_counts)

    def test_batch_across_interleaved_boundary(self, executor: BatchExecutor, meta: MetadataIndex):
        """Large interleaved batch near the dataset boundary."""
        # Pick 64 indices mixing dummy_00 and dummy_01 near boundary
        rng = np.random.default_rng(777)
        indices_d00 = rng.integers(49_000, 50_000, size=32).tolist()
        indices_d01 = rng.integers(50_000, 51_000, size=32).tolist()
        all_indices = indices_d00 + indices_d01
        rng.shuffle(all_indices)

        payload = executor.collate_sparse_batch(all_indices)
        _validate_sparse_payload(payload, 64)


# ===================================================================
# Tests — edge cases
# ===================================================================


class TestFederatedEdgeCases:
    """Smoke test: edge cases for federated topology."""

    def test_empty_batch_federated(self, executor: BatchExecutor):
        """Empty index list returns empty payload."""
        payload = executor.collate_sparse_batch([])
        assert payload.batch_size == 0

    def test_single_dataset_only(self, executor: BatchExecutor):
        """Batch with only dummy_00 cells."""
        indices = list(range(10))
        cells = executor.read_cells(indices)
        assert len(cells) == 10
        for c in cells:
            assert c.dataset_id == "dummy_00"
            assert c.dataset_index == 0

    def test_second_dataset_only(self, executor: BatchExecutor):
        """Batch with only dummy_01 cells."""
        indices = list(range(50000, 50010))
        cells = executor.read_cells(indices)
        assert len(cells) == 10
        for c in cells:
            assert c.dataset_id == "dummy_01"
            assert c.dataset_index == 1

    def test_dataset_boundary_indices_federated(self, executor: BatchExecutor):
        """First and last cell of each dataset via federated reader."""
        indices = [0, 49999, 50000, 124999]
        cells = executor.read_cells(indices)
        assert cells[0].global_row_index == 0
        assert cells[1].global_row_index == 49999
        assert cells[2].global_row_index == 50000
        assert cells[3].global_row_index == 124999

    def test_all_single_dataset_federated(self, executor: BatchExecutor):
        """Multiple cells all from dummy_01."""
        rng = np.random.default_rng(42)
        indices = rng.integers(50000, 125000, size=50).tolist()
        cells = executor.read_cells(indices)
        assert len(cells) == 50
        for c in cells:
            assert c.dataset_id == "dummy_01"

    def test_read_cells_deterministic_federated(self, executor: BatchExecutor):
        """Same indices produce identical CellState on repeated reads via federated."""
        indices = [0, 50000, 1, 50001, 99999, 124999]
        cells_1 = executor.read_cells(indices)
        cells_2 = executor.read_cells(indices)

        for c1, c2 in zip(cells_1, cells_2):
            assert c1.global_row_index == c2.global_row_index
            assert c1.dataset_id == c2.dataset_id
            assert c1.expressed_gene_indices == c2.expressed_gene_indices
            assert c1.expression_counts == c2.expression_counts
