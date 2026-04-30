"""Phase 3 integration tests: BatchExecutor hot-path API.

Validates the flat-batch API:
- read_expression_batch() → ExpressionBatch
- read_metadata_batch() → columnar dicts with numpy arrays
- read_batch() → combined dict for GPU pipeline
- CorpusRandomBatchSampler (MetadataIndex-backed)
- DatasetBatchSampler (MetadataIndex-backed)
- DatasetContextBatchSampler (MetadataIndex-backed)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from perturb_data_lab.loaders import (
    AggregateLanceReader,
    BatchExecutor,
    CorpusRandomBatchSampler,
    DatasetBatchSampler,
    DatasetContextBatchSampler,
    DatasetEntry,
    ExpressionBatch,
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
# Tests — BatchExecutor hot-path API
# ===================================================================


class TestBatchExecutor:
    """Core BatchExecutor integration tests using the flat-batch API."""

    def test_len(self):
        """len(executor) matches MetadataIndex."""
        exec_ = _executor()
        meta = _meta()
        assert len(exec_) == len(meta) == 125_000

    # -- read_expression_batch -------------------------------------------

    def test_read_expression_batch_empty(self):
        """Empty input returns empty ExpressionBatch."""
        expr = _executor().read_expression_batch([])
        assert isinstance(expr, ExpressionBatch)
        assert expr.batch_size == 0
        assert len(expr.global_row_index) == 0

    def test_read_expression_batch_single(self):
        """Read a single cell produces ExpressionBatch with correct fields."""
        expr = _executor().read_expression_batch([0])
        assert isinstance(expr, ExpressionBatch)
        assert expr.batch_size == 1
        assert expr.global_row_index.shape == (1,)
        assert expr.global_row_index[0] == 0
        # row_offsets: [0, n_genes_for_cell_0]
        assert expr.row_offsets[0] == 0
        assert expr.row_offsets[-1] > 0
        assert len(expr.expressed_gene_indices) > 0
        assert len(expr.expression_counts) > 0
        assert len(expr.expressed_gene_indices) == len(expr.expression_counts)

    def test_read_expression_batch_preserves_order(self):
        """Output ExpressionBatch preserves input index order."""
        indices = [50001, 1, 99999, 0]
        expr = _executor().read_expression_batch(indices)
        assert expr.batch_size == 4
        np.testing.assert_array_equal(
            expr.global_row_index, np.array(indices, dtype=np.int64)
        )

    def test_read_expression_batch_32_random(self):
        """32 random cells produce valid ExpressionBatch."""
        meta = _meta()
        sampled_df = meta.sample(32, seed=42)
        indices = sampled_df["global_row_index"].to_list()
        expr = _executor().read_expression_batch(indices)
        assert expr.batch_size == 32
        assert expr.row_offsets[-1] == len(expr.expressed_gene_indices)
        # Each row should have expression data
        for i in range(32):
            s = expr.row_slice(i)
            assert expr.expressed_gene_indices[s].size > 0
            assert expr.expressed_gene_indices[s].size == expr.expression_counts[s].size

    def test_read_expression_batch_row_slicing(self):
        """row_slice() returns valid slices per cell."""
        expr = _executor().read_expression_batch([0, 1, 2])
        for i in range(3):
            s = expr.row_slice(i)
            gene_idx = expr.row_gene_indices(i)
            counts = expr.row_counts(i)
            assert len(gene_idx) > 0
            assert len(gene_idx) == len(counts)

    # -- read_metadata_batch ---------------------------------------------

    def test_read_metadata_batch_empty(self):
        """Empty input returns empty dicts with correct dtypes."""
        meta = _executor().read_metadata_batch([])
        assert meta["global_row_index"].dtype == np.int64
        assert meta["dataset_index"].dtype == np.int32
        assert meta["local_row_index"].dtype == np.int64
        assert meta["size_factor"].dtype == np.float32
        assert meta["dataset_id"] == ()
        assert meta["cell_id"] == ()
        assert meta["canonical_perturbation"] == ()
        assert meta["canonical_context"] == ()

    def test_read_metadata_batch_single(self):
        """Read a single cell produces columnar metadata."""
        meta = _executor().read_metadata_batch([0])
        assert meta["global_row_index"].shape == (1,)
        assert meta["global_row_index"][0] == 0
        assert meta["dataset_index"].shape == (1,)
        assert meta["dataset_index"][0] == 0
        assert meta["size_factor"].shape == (1,)
        assert meta["size_factor"][0] > 0.0
        assert len(meta["dataset_id"]) == 1
        assert len(meta["cell_id"]) == 1
        assert len(meta["canonical_perturbation"]) == 1
        assert len(meta["canonical_context"]) == 1
        assert isinstance(meta["canonical_perturbation"][0], dict)
        assert isinstance(meta["canonical_context"][0], dict)
        # Dummy data has guide_1 in perturbation
        assert "guide_1" in meta["canonical_perturbation"][0]

    def test_read_metadata_batch_canonical_fields(self):
        """Metadata has populated canonical_perturbation and canonical_context."""
        meta = _executor().read_metadata_batch([0])
        assert isinstance(meta["canonical_perturbation"][0], dict)
        assert isinstance(meta["canonical_context"][0], dict)
        assert "guide_1" in meta["canonical_perturbation"][0]
        assert "cell_type" in meta["canonical_context"][0]

    def test_read_metadata_batch_preserves_order(self):
        """Output metadata preserves input index order."""
        indices = [50001, 1, 99999, 0]
        meta = _executor().read_metadata_batch(indices)
        np.testing.assert_array_equal(
            meta["global_row_index"], np.array(indices, dtype=np.int64)
        )
        assert meta["dataset_id"][0] == "dummy_01"
        assert meta["dataset_id"][1] == "dummy_00"

    # -- read_batch ------------------------------------------------------

    def test_read_batch_empty(self):
        """Empty input returns dict with zero-sized arrays."""
        batch = _executor().read_batch([])
        assert batch["batch_size"] == 0
        assert batch["global_row_index"].dtype == np.int64
        assert batch["dataset_index"].dtype == np.int32
        assert batch["size_factor"].dtype == np.float32

    def test_read_batch_single(self):
        """read_batch returns all 12 expected keys."""
        batch = _executor().read_batch([0])
        expected_keys = {
            "batch_size", "global_row_index", "row_offsets",
            "expressed_gene_indices", "expression_counts",
            "dataset_index", "local_row_index", "size_factor",
            "dataset_id", "cell_id",
            "canonical_perturbation", "canonical_context",
        }
        assert set(batch.keys()) == expected_keys
        assert batch["batch_size"] == 1
        assert batch["global_row_index"][0] == 0

    def test_read_batch_consistent(self):
        """Expression and metadata are consistent within read_batch output."""
        batch = _executor().read_batch([0, 1, 50000])
        assert batch["batch_size"] == 3
        # Expression and metadata have matching global_row_index
        np.testing.assert_array_equal(
            batch["global_row_index"],
            np.array([0, 1, 50000], dtype=np.int64),
        )
        # row_offsets matches expression arrays
        assert batch["row_offsets"][0] == 0
        assert batch["row_offsets"][-1] == len(batch["expressed_gene_indices"])
        assert len(batch["expressed_gene_indices"]) == len(batch["expression_counts"])

    def test_read_batch_deterministic(self):
        """Same indices produce identical read_batch output."""
        indices = [0, 1, 2, 50000, 50001]
        batch1 = _executor().read_batch(indices)
        batch2 = _executor().read_batch(indices)
        assert batch1["batch_size"] == batch2["batch_size"]
        np.testing.assert_array_equal(batch1["global_row_index"], batch2["global_row_index"])
        np.testing.assert_array_equal(batch1["expressed_gene_indices"], batch2["expressed_gene_indices"])
        np.testing.assert_array_equal(batch1["expression_counts"], batch2["expression_counts"])
        np.testing.assert_array_equal(batch1["row_offsets"], batch2["row_offsets"])

    def test_read_batch_cross_dataset(self):
        """Batch spanning dummy_00 and dummy_01 boundary."""
        indices = [49998, 49999, 50000, 50001]
        batch = _executor().read_batch(indices)
        assert batch["batch_size"] == 4
        assert batch["dataset_id"][0] == "dummy_00"
        assert batch["dataset_id"][1] == "dummy_00"
        assert batch["dataset_id"][2] == "dummy_01"
        assert batch["dataset_id"][3] == "dummy_01"

    def test_read_batch_boundary_indices(self):
        """First and last cell of each dataset."""
        batch = _executor().read_batch([0, 49999, 50000, 124999])
        np.testing.assert_array_equal(
            batch["global_row_index"],
            np.array([0, 49999, 50000, 124999], dtype=np.int64),
        )


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
        for idx in batch:
            assert 0 <= idx < 50_000

    def test_dataset_index_1(self):
        meta = _meta()
        sampler = DatasetBatchSampler(
            metadata_index=meta, dataset_index=1, batch_size=128, seed=1
        )
        batch = next(iter(sampler))
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
                assert 0 <= idx < 50_000

    def test_len(self):
        meta = _meta()
        sampler = DatasetContextBatchSampler(
            metadata_index=meta,
            batch_size=8,
            context_field="raw_cell_type",
            seed=42,
        )
        assert len(sampler) > 0
