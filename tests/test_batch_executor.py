"""Phase 3 integration tests: BatchExecutor, samplers, PerturbDataLoader.

Validates the full pipeline:
- MetadataIndex + ExpressionReader → BatchExecutor
- CorpusRandomBatchSampler (MetadataIndex-backed)
- DatasetBatchSampler (MetadataIndex-backed)
- DatasetContextBatchSampler (MetadataIndex-backed)
- PerturbDataLoader with BatchExecutor
- SparseBatchCollator integration
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perturb_data_lab.loaders import (
    AggregateLanceReader,
    BatchExecutor,
    CellIdentity,
    CellState,
    CorpusRandomBatchSampler,
    DatasetBatchSampler,
    DatasetContextBatchSampler,
    DatasetEntry,
    MetadataIndex,
    PerturbDataLoader,
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
# Fixtures
# ===================================================================


def _meta() -> MetadataIndex:
    if not hasattr(_meta, "_cache"):
        _meta._cache = MetadataIndex.from_dummy_data()
    return _meta._cache


def _executor() -> BatchExecutor:
    if not hasattr(_executor, "_cache"):
        meta = _meta()
        agg_entry = DatasetEntry("aggregated", 0, len(meta))
        reader = AggregateLanceReader(AGGREGATE_LANCE, [agg_entry])
        _executor._cache = BatchExecutor(reader, meta)
    return _executor._cache


# ===================================================================
# Tests — BatchExecutor
# ===================================================================


class TestBatchExecutor:
    """Core BatchExecutor integration tests."""

    def test_len(self):
        """len(executor) matches MetadataIndex."""
        exec_ = _executor()
        meta = _meta()
        assert len(exec_) == len(meta) == 125_000

    def test_read_cells_empty(self):
        """Empty input returns empty list."""
        cells = _executor().read_cells([])
        assert cells == []

    def test_read_cells_single(self):
        """Read a single cell produces one CellState."""
        cells = _executor().read_cells([0])
        assert len(cells) == 1
        c = cells[0]
        assert isinstance(c, CellState)
        assert c.global_row_index == 0
        assert c.dataset_id == "dummy_00"
        assert isinstance(c.expressed_gene_indices, tuple)
        assert isinstance(c.expression_counts, tuple)
        assert len(c.expressed_gene_indices) == len(c.expression_counts)
        assert c.size_factor > 0.0

    def test_read_cells_preserves_order(self):
        """Output CellState list preserves input index order."""
        indices = [50001, 1, 99999, 0]
        cells = _executor().read_cells(indices)
        assert len(cells) == 4
        assert [c.global_row_index for c in cells] == indices

    def test_read_cells_32_random(self):
        """32 random cells produce 32 valid CellState objects."""
        meta = _meta()
        sampled_df = meta.sample(32, seed=42)
        indices = sampled_df["global_row_index"].to_list()
        cells = _executor().read_cells(indices)
        assert len(cells) == 32
        for c in cells:
            assert isinstance(c.cell_id, str)
            assert isinstance(c.dataset_id, str)
            assert c.dataset_index in (0, 1)
            assert 0 <= c.global_row_index < 125_000
            assert c.size_factor > 0.0
            assert len(c.expressed_gene_indices) > 0
            assert len(c.expressed_gene_indices) == len(c.expression_counts)

    def test_read_cells_metadata_canonical(self):
        """CellState has populated canonical_perturbation and canonical_context."""
        meta = _meta()
        # Pick a cell with known perturbation fields
        cells = _executor().read_cells([0])
        c = cells[0]
        assert isinstance(c.canonical_perturbation, dict)
        assert isinstance(c.canonical_context, dict)
        assert isinstance(c.raw_fields, dict)
        # Dummy data should have guide_1 in perturbation
        assert "guide_1" in c.canonical_perturbation

    def test_collate_sparse_batch(self):
        """collate_sparse_batch produces valid SparseBatchPayload."""
        indices = [0, 1, 2, 50000, 50001]
        payload = _executor().collate_sparse_batch(indices)
        assert isinstance(payload, SparseBatchPayload)
        assert payload.batch_size == 5
        assert payload.global_row_index.shape == (5,)
        assert payload.dataset_index.shape == (5,)
        assert payload.size_factor.shape == (5,)
        assert len(payload.dataset_id) == 5
        assert len(payload.cell_id) == 5
        # Row offsets should be valid
        assert payload.row_offsets[0] == 0
        assert payload.row_offsets[-1] == len(payload.expressed_gene_indices)
        # Slices should work
        for i in range(5):
            s = payload.row_slice(i)
            assert payload.expressed_gene_indices[s].size == payload.expression_counts[s].size


# ===================================================================
# Tests — Samplers
# ===================================================================


class TestCorpusRandomBatchSampler:
    """CorpusRandomBatchSampler with MetadataIndex."""

    def test_len(self):
        meta = _meta()
        sampler = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, drop_last=True
        )
        assert len(sampler) == 125_000 // 128  # 976

    def test_produces_valid_indices(self):
        meta = _meta()
        sampler = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, seed=42
        )
        batch = next(iter(sampler))
        assert len(batch) == 128
        for idx in batch:
            assert 0 <= idx < 125_000

    def test_epoch_variation(self):
        """Different epochs produce different batches."""
        meta = _meta()
        sampler = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=64, seed=42
        )
        batch_epoch0 = next(iter(sampler))
        sampler.set_epoch(1)
        batch_epoch1 = next(iter(sampler))
        # Extremely unlikely to produce identical batches
        assert batch_epoch0 != batch_epoch1


class TestDatasetBatchSampler:
    """DatasetBatchSampler with MetadataIndex."""

    def test_len(self):
        meta = _meta()
        sampler = DatasetBatchSampler(
            metadata_index=meta, dataset_index=0, batch_size=128
        )
        assert len(sampler) == 50_000 // 128  # 390

    def test_indices_in_dataset(self):
        meta = _meta()
        sampler = DatasetBatchSampler(
            metadata_index=meta, dataset_index=0, batch_size=128, seed=1
        )
        batch = next(iter(sampler))
        assert len(batch) == 128
        # All indices should be in dummy_00's range [0, 50000)
        for idx in batch:
            assert 0 <= idx < 50_000

    def test_dataset_index_1(self):
        meta = _meta()
        sampler = DatasetBatchSampler(
            metadata_index=meta, dataset_index=1, batch_size=128, seed=1
        )
        batch = next(iter(sampler))
        # All indices should be in dummy_01's range [50000, 125000)
        for idx in batch:
            assert 50000 <= idx < 125_000


class TestDatasetContextBatchSampler:
    """DatasetContextBatchSampler with MetadataIndex."""

    def test_produces_batches(self):
        meta = _meta()
        sampler = DatasetContextBatchSampler(
            metadata_index=meta,
            batch_size=8,
            context_field="raw_cell_type",
            seed=42,
        )
        batches = list(iter(sampler))
        assert len(batches) > 0
        for batch in batches:
            assert len(batch) == 8

    def test_single_dataset_filter(self):
        meta = _meta()
        sampler = DatasetContextBatchSampler(
            metadata_index=meta,
            batch_size=8,
            context_field="raw_cell_type",
            dataset_index=0,
            seed=42,
        )
        batches = list(iter(sampler))
        for batch in batches:
            for idx in batch:
                assert 0 <= idx < 50_000  # dummy_00 only

    def test_len(self):
        meta = _meta()
        sampler = DatasetContextBatchSampler(
            metadata_index=meta,
            batch_size=8,
            context_field="raw_cell_type",
            seed=42,
        )
        assert len(sampler) > 0


# ===================================================================
# Tests — PerturbDataLoader
# ===================================================================


class TestPerturbDataLoader:
    """PerturbDataLoader with BatchExecutor."""

    def test_len(self):
        dl = PerturbDataLoader(_executor(), n_genes=5000)
        assert len(dl) == 125_000

    def test_getitem(self):
        dl = PerturbDataLoader(_executor(), n_genes=5000)
        item = dl[0]
        assert "expressed_gene_indices" in item
        assert "expression_counts" in item
        assert "context_indices" in item
        assert "cell_id" in item
        assert "size_factor" in item

    def test_getitem_with_context_size(self):
        dl = PerturbDataLoader(
            _executor(), n_genes=5000, context_size=100, sampler_mode="random_context"
        )
        item = dl[0]
        assert len(item["context_indices"]) == 100

    def test_getitems(self):
        dl = PerturbDataLoader(_executor(), n_genes=5000)
        items = dl.__getitems__([0, 1, 2])
        assert len(items) == 3
        for item in items:
            assert "expressed_gene_indices" in item
