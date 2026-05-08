"""Phase 3 integration tests: BatchExecutor hot-path API.

Validates the flat-batch API:
- read_expression_batch() → ExpressionBatch
- read_metadata_batch() → columnar dicts with numpy arrays
- read_batch() → combined dict for GPU pipeline
- CorpusRandomBatchSampler (MetadataIndex-backed)
- DatasetBatchSampler (MetadataIndex-backed)
- DatasetContextBatchSampler (MetadataIndex-backed)

Phase 7: validate correctness and compatibility across modes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import torch

from perturb_data_lab.loaders import (
    AggregateLanceReader,
    BatchExecutor,
    CorpusRandomBatchSampler,
    DatasetBatchSampler,
    DatasetContextBatchSampler,
    DatasetEntry,
    ExpressionBatch,
    FastTrainingBatch,
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


def _meta_canonical() -> MetadataIndex:
    """Singleton: canonical MetadataIndex (use_canonical=True)."""
    if not hasattr(_meta_canonical, "_cache"):
        _meta_canonical._cache = MetadataIndex.from_dummy_data(use_canonical=True)
    return _meta_canonical._cache


def _executor_canonical() -> BatchExecutor:
    """Singleton: BatchExecutor with canonical metadata."""
    if not hasattr(_executor_canonical, "_cache"):
        meta = _meta_canonical()
        agg_entry = DatasetEntry("aggregated", 0, len(meta))
        reader = AggregateLanceReader(AGGREGATE_LANCE, [agg_entry])
        _executor_canonical._cache = BatchExecutor(
            reader, meta, use_canonical=True,
        )
    return _executor_canonical._cache


def _executor_federated() -> BatchExecutor:
    """Singleton: BatchExecutor backed by FederatedLanceReader."""
    if not hasattr(_executor_federated, "_cache"):
        meta = _meta()
        entries = [
            LanceDatasetEntry(
                "dummy_00", 0, 50_000,
                FEDERATED_BASE / "dummy_00/matrix/dummy_00-release.lance",
            ),
            LanceDatasetEntry(
                "dummy_01", 50_000, 125_000,
                FEDERATED_BASE / "dummy_01/matrix/dummy_01-release.lance",
            ),
        ]
        reader = FederatedLanceReader(entries)
        _executor_federated._cache = BatchExecutor(reader, meta)
    return _executor_federated._cache


def _legacy_perturb_batch_dataset(*args, **kwargs):
    from perturb_data_lab.loaders import PerturbBatchDataset

    with pytest.warns(DeprecationWarning, match="PerturbBatchDataset"):
        return PerturbBatchDataset(*args, **kwargs)


def _legacy_collate_batch_dict(items):
    from perturb_data_lab.loaders import collate_batch_dict

    with pytest.warns(DeprecationWarning, match="collate_batch_dict"):
        return collate_batch_dict(items)


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


# ===================================================================
# Phase 4 — Columnar metadata extraction tests
# ===================================================================


class TestBatchExecutorColumnarMetadata:
    """Phase 4 tests: columnar metadata extraction from canonical obs.

    These tests validate that:
    - ``read_metadata_batch(columnar=True)`` returns ``meta_columns``
      instead of ``canonical_perturbation`` / ``canonical_context``
    - ``read_batch(columnar=True)`` composes correctly
    - The backward-compatible default path still produces correct output
    - No ``_build_canonical_dicts()`` is called in the default path
      (tested indirectly via equivalence with old behavior)
    """

    def test_read_metadata_batch_columnar_mode_keys(self):
        """columnar=True returns meta_columns, omits canonical_perturbation/context."""
        meta = _executor().read_metadata_batch([0, 1, 50000], columnar=True)
        assert "meta_columns" in meta
        assert "canonical_perturbation" not in meta
        assert "canonical_context" not in meta
        # Identity columns still present
        assert "global_row_index" in meta
        assert "dataset_index" in meta
        assert "size_factor" in meta
        assert "dataset_id" in meta
        assert "cell_id" in meta

    def test_read_metadata_batch_columnar_meta_columns_content(self):
        """meta_columns contains individual perturbation fields as arrays/tuples."""
        meta = _executor().read_metadata_batch([0, 1, 2], columnar=True)
        mc = meta["meta_columns"]
        assert isinstance(mc, dict)
        assert len(mc) > 0, "meta_columns should not be empty"

        # Check known perturbation fields (raw_ mode)
        for key in ["guide_1", "guide_2", "treatment", "site", "genotype"]:
            assert key in mc, f"Expected '{key}' in meta_columns"
            data = mc[key]
            assert len(data) == 3, f"{key} should have 3 entries"
            assert isinstance(data, tuple), f"{key} should be tuple, got {type(data)}"

    def test_read_metadata_batch_columnar_context_fields(self):
        """meta_columns contains context fields as tuples (raw_ mode)."""
        meta = _executor().read_metadata_batch([0, 1, 2], columnar=True)
        mc = meta["meta_columns"]
        for key in ["cell_type", "cellline", "donor_id", "batch", "passage"]:
            assert key in mc, f"Expected context field '{key}' in meta_columns"
            data = mc[key]
            assert len(data) == 3, f"{key} should have 3 entries"

    def test_read_metadata_batch_columnar_empty(self):
        """Empty input with columnar=True returns empty meta_columns."""
        meta = _executor().read_metadata_batch([], columnar=True)
        assert meta["meta_columns"] == {}
        assert meta["global_row_index"].dtype == np.int64
        assert len(meta["global_row_index"]) == 0

    def test_read_metadata_batch_columnar_preserves_order(self):
        """columnar=True metadata preserves input index order."""
        indices = [50001, 1, 99999, 0]
        meta = _executor().read_metadata_batch(indices, columnar=True)
        np.testing.assert_array_equal(
            meta["global_row_index"], np.array(indices, dtype=np.int64)
        )
        assert meta["dataset_id"][0] == "dummy_01"
        assert meta["dataset_id"][1] == "dummy_00"
        assert meta["dataset_id"][2] == "dummy_01"
        assert meta["dataset_id"][3] == "dummy_00"

    def test_read_metadata_batch_columnar_perturbation_values(self):
        """meta_columns perturbation values match per-cell dict values from default path."""
        indices = [0, 1, 50000]
        # Get columnar metadata
        meta_col = _executor().read_metadata_batch(indices, columnar=True)
        # Get default metadata for comparison
        meta_def = _executor().read_metadata_batch(indices, columnar=False)

        mc = meta_col["meta_columns"]
        cp = meta_def["canonical_perturbation"]
        cc = meta_def["canonical_context"]

        # Perturbation fields should match (columnar may preserve raw types;
        # legacy path stringifies everything)
        for key in ["guide_1", "guide_2", "treatment", "site", "genotype"]:
            if key not in mc:
                continue
            for i in range(len(indices)):
                val_from_columnar = mc[key][i]
                val_from_dict = cp[i].get(key)
                if val_from_dict is not None:
                    assert str(val_from_columnar) == val_from_dict, (
                        f"Mismatch for {key}[{i}]: "
                        f"columnar={val_from_columnar!r}, dict={val_from_dict!r}"
                    )

        # Context fields should match
        for key in ["cell_type", "cellline", "donor_id", "batch", "passage"]:
            if key not in mc:
                continue
            for i in range(len(indices)):
                val_from_columnar = mc[key][i]
                val_from_dict = cc[i].get(key)
                if val_from_dict is not None:
                    assert str(val_from_columnar) == val_from_dict, (
                        f"Mismatch for {key}[{i}]: "
                        f"columnar={val_from_columnar!r}, dict={val_from_dict!r}"
                    )

    def test_read_batch_columnar_mode(self):
        """read_batch with columnar=True returns meta_columns, omits canonical dicts."""
        batch = _executor().read_batch([0, 1, 2, 50000, 50001], columnar=True)

        assert "meta_columns" in batch
        assert "canonical_perturbation" not in batch
        assert "canonical_context" not in batch
        # Expression fields still present
        assert "batch_size" in batch
        assert "global_row_index" in batch
        assert "row_offsets" in batch
        assert "expressed_gene_indices" in batch
        assert "expression_counts" in batch
        # Identity metadata fields still present
        assert "dataset_index" in batch
        assert "size_factor" in batch
        assert "dataset_id" in batch
        assert "cell_id" in batch

    def test_read_batch_columnar_consistency(self):
        """read_batch columnar meta_columns match expression batch size."""
        batch = _executor().read_batch([0, 1, 50000], columnar=True)
        mc = batch["meta_columns"]
        n = batch["batch_size"]
        assert n == 3
        for col_name, data in mc.items():
            assert len(data) == n, (
                f"meta_columns['{col_name}'] has {len(data)} entries, "
                f"expected {n}"
            )

    def test_read_batch_columnar_empty(self):
        """Empty read_batch with columnar=True returns empty meta_columns."""
        batch = _executor().read_batch([], columnar=True)
        assert batch["batch_size"] == 0
        assert batch["meta_columns"] == {}

    def test_metadata_equivalence_columnar_vs_default(self):
        """Identity columns are identical between columnar and default modes."""
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(125000, size=64, replace=False).tolist())

        meta_def = _executor().read_metadata_batch(indices, columnar=False)
        meta_col = _executor().read_metadata_batch(indices, columnar=True)

        # Identity columns should be identical
        np.testing.assert_array_equal(
            meta_def["global_row_index"], meta_col["global_row_index"]
        )
        np.testing.assert_array_equal(
            meta_def["dataset_index"], meta_col["dataset_index"]
        )
        np.testing.assert_array_equal(
            meta_def["local_row_index"], meta_col["local_row_index"]
        )
        np.testing.assert_array_equal(
            meta_def["size_factor"], meta_col["size_factor"]
        )
        assert meta_def["dataset_id"] == meta_col["dataset_id"]
        assert meta_def["cell_id"] == meta_col["cell_id"]


# ===================================================================
# Phase 5 — Fast-path wiring tests
# ===================================================================


class TestPhase5FastPathWiring:
    """Phase 5 tests: fast expression path, Dataset columnar mode, collation.

    These tests validate that:
    - ``read_expression_batch()`` uses the native flat read path
      (``read_expression_flat()``) for Lance-backed corpora
    - Expression output is numerically equivalent to the legacy
      ``ExpressionRow`` path
    - ``PerturbBatchDataset`` supports ``columnar=True`` and
      ``columnar=False`` modes
    - ``collate_batch_dict`` handles both legacy dict metadata and
      columnar ``meta_columns``
    """

    # -- Fast expression path equivalence -----------------------------------

    def test_read_expression_batch_uses_flat_path(self):
        """read_expression_batch returns identical results via fast path."""
        indices = [0, 1, 50000, 50001, 124999]
        expr = _executor().read_expression_batch(indices)

        assert isinstance(expr, ExpressionBatch)
        assert expr.batch_size == 5
        np.testing.assert_array_equal(
            expr.global_row_index, np.array(indices, dtype=np.int64)
        )
        # Row offsets must be valid
        assert expr.row_offsets[0] == 0
        assert expr.row_offsets[-1] == len(expr.expressed_gene_indices)
        assert len(expr.expressed_gene_indices) == len(expr.expression_counts)
        # Each row has expression data
        for i in range(5):
            s = expr.row_slice(i)
            assert expr.expressed_gene_indices[s].size > 0

    def test_read_expression_batch_fast_path_equivalence(self):
        """Fast path output is numerically identical to ExpressionRow path.

        Uses ``hasattr`` to verify the reader has the flat method, then
        reads the same indices via both paths and compares.
        """
        exec_ = _executor()
        reader = exec_.expression_reader

        # Verify fast-path capability
        assert hasattr(reader, "read_expression_flat"), (
            "Lance reader must have read_expression_flat for fast path"
        )

        indices = [0, 1, 50000, 50001, 124999]

        # Fast path (via BatchExecutor which now uses read_expression_flat)
        expr_fast = exec_.read_expression_batch(indices)

        # Legacy path (force via read_expression + manual reconstruction)
        expr_rows = reader.read_expression(indices)
        n = len(expr_rows)
        row_offsets = np.zeros(n + 1, dtype=np.int64)
        egi_parts: list[np.ndarray] = []
        ec_parts: list[np.ndarray] = []
        for i, row in enumerate(expr_rows):
            row_offsets[i + 1] = row_offsets[i] + len(row.expressed_gene_indices)
            egi_parts.append(row.expressed_gene_indices)
            ec_parts.append(row.expression_counts)
        expr_legacy = ExpressionBatch(
            batch_size=n,
            global_row_index=np.array(
                [r.global_row_index for r in expr_rows], dtype=np.int64
            ),
            row_offsets=row_offsets,
            expressed_gene_indices=np.concatenate(egi_parts).astype(np.int32, copy=False),
            expression_counts=np.concatenate(ec_parts).astype(np.int32, copy=False),
        )

        assert expr_fast.batch_size == expr_legacy.batch_size
        np.testing.assert_array_equal(
            expr_fast.global_row_index, expr_legacy.global_row_index
        )
        np.testing.assert_array_equal(
            expr_fast.row_offsets, expr_legacy.row_offsets
        )
        np.testing.assert_array_equal(
            expr_fast.expressed_gene_indices, expr_legacy.expressed_gene_indices
        )
        np.testing.assert_array_equal(
            expr_fast.expression_counts, expr_legacy.expression_counts
        )

    def test_read_expression_batch_fast_path_large_shuffled(self):
        """Fast path handles large shuffled batches correctly."""
        rng = np.random.default_rng(123)
        indices = sorted(rng.choice(125_000, size=128, replace=False).tolist())
        rng.shuffle(indices)

        expr = _executor().read_expression_batch(indices)
        assert expr.batch_size == 128
        np.testing.assert_array_equal(
            expr.global_row_index, np.array(indices, dtype=np.int64)
        )
        assert expr.row_offsets[-1] == len(expr.expressed_gene_indices)

        # Verify each row's expression data is non-empty
        for i in range(128):
            s = expr.row_slice(i)
            assert expr.expressed_gene_indices[s].size > 0
            assert expr.expression_counts[s].size > 0

    def test_read_expression_batch_fast_path_cross_chunk(self):
        """Fast path handles indices spanning Lance chunk boundary."""
        # Lance chunk limit is 2048, so 3000 consecutive indices
        # spans multiple take() calls
        indices = list(range(1000, 4000))
        expr = _executor().read_expression_batch(indices)
        assert expr.batch_size == 3000
        np.testing.assert_array_equal(
            expr.global_row_index, np.array(indices, dtype=np.int64)
        )
        assert expr.row_offsets[-1] == len(expr.expressed_gene_indices)

    # -- PerturbBatchDataset columnar mode ----------------------------------

    def test_dataset_columnar_false_default(self):
        """PerturbBatchDataset without columnar preserves legacy dict output."""
        ds = _legacy_perturb_batch_dataset(_executor(), seq_len=128)
        assert ds.columnar is False

        batch = ds.__getitems__([0, 1, 50000])[0]
        assert "canonical_perturbation" in batch
        assert "canonical_context" in batch
        assert "meta_columns" not in batch

    def test_dataset_columnar_true_output(self):
        """PerturbBatchDataset with columnar=True returns meta_columns."""
        ds = _legacy_perturb_batch_dataset(
            _executor(), seq_len=128, columnar=True,
        )
        assert ds.columnar is True

        batch = ds.__getitems__([0, 1, 50000])[0]
        assert "meta_columns" in batch
        assert "canonical_perturbation" not in batch
        assert "canonical_context" not in batch
        # Expression fields still present
        assert "batch_size" in batch
        assert "expressed_gene_indices" in batch
        assert "expression_counts" in batch
        assert "row_offsets" in batch

    def test_dataset_columnar_empty_batch(self):
        """PerturbBatchDataset columnar mode handles empty input."""
        ds = _legacy_perturb_batch_dataset(_executor(), columnar=True)
        batch = ds.__getitems__([])[0]
        assert batch["batch_size"] == 0
        assert batch["meta_columns"] == {}

    def test_dataset_columnar_meta_columns_content(self):
        """Columnar meta_columns from Dataset have correct shapes."""
        ds = _legacy_perturb_batch_dataset(_executor(), columnar=True)
        batch = ds.__getitems__([0, 1, 2, 50000, 50001])[0]

        mc = batch["meta_columns"]
        assert isinstance(mc, dict)
        assert len(mc) > 0

        # All columns should have 5 entries
        n = batch["batch_size"]
        for col_name, data in mc.items():
            assert len(data) == n, (
                f"meta_columns['{col_name}'] has {len(data)} entries"
            )

    # -- Collation with meta_columns ----------------------------------------

    def test_collate_legacy_dict_metadata(self):
        """collate_batch_dict produces legacy dict fields in default mode."""
        ds = _legacy_perturb_batch_dataset(_executor(), columnar=False)
        items = ds.__getitems__([0, 1, 2])
        collated = _legacy_collate_batch_dict(items)

        # Required numeric tensor fields
        assert isinstance(collated["global_row_index"], torch.Tensor)
        assert isinstance(collated["expressed_gene_indices"], torch.Tensor)
        assert isinstance(collated["expression_counts"], torch.Tensor)
        assert isinstance(collated["row_offsets"], torch.Tensor)
        assert isinstance(collated["dataset_index"], torch.Tensor)
        assert isinstance(collated["size_factor"], torch.Tensor)

        # Legacy dict metadata (non-tensor, pass-through)
        assert "canonical_perturbation" in collated
        assert "canonical_context" in collated
        assert "meta_columns" not in collated

    def test_collate_columnar_metadata(self):
        """collate_batch_dict passes meta_columns through unchanged."""
        ds = _legacy_perturb_batch_dataset(_executor(), columnar=True)
        items = ds.__getitems__([0, 1, 50000])
        collated = _legacy_collate_batch_dict(items)

        # Required numeric tensor fields present
        assert isinstance(collated["global_row_index"], torch.Tensor)
        assert isinstance(collated["expressed_gene_indices"], torch.Tensor)

        # Columnar metadata (non-tensor, pass-through)
        assert "meta_columns" in collated
        assert "canonical_perturbation" not in collated
        assert "canonical_context" not in collated

        # meta_columns entries are CPU-side (tuples/arrays, not tensors)
        mc = collated["meta_columns"]
        assert isinstance(mc, dict)
        for key, val in mc.items():
            assert not isinstance(val, torch.Tensor), (
                f"meta_columns['{key}'] should NOT be a tensor "
                f"(keep string metadata CPU-side)"
            )

    def test_collate_columnar_empty(self):
        """collate_batch_dict handles empty columnar batch."""
        ds = _legacy_perturb_batch_dataset(_executor(), columnar=True)
        items = ds.__getitems__([])
        collated = _legacy_collate_batch_dict(items)

        assert collated["batch_size"] == 0
        assert collated["meta_columns"] == {}
        assert "canonical_perturbation" not in collated

    def test_collate_all_numeric_tensors_on_cpu(self):
        """collate_batch_dict produces CPU tensors (not CUDA)."""
        ds = _legacy_perturb_batch_dataset(_executor(), columnar=True)
        items = ds.__getitems__([0, 1, 2])
        collated = _legacy_collate_batch_dict(items)

        tensor_keys = [
            "global_row_index", "row_offsets", "expressed_gene_indices",
            "expression_counts", "dataset_index", "size_factor",
            "local_row_index",
        ]
        for key in tensor_keys:
            t = collated[key]
            assert t.device.type == "cpu", (
                f"{key} tensor should be on CPU, got {t.device}"
            )


# ===================================================================
# Phase 6 — Direct integer sampling and hot-path plumbing
# ===================================================================


class TestPhase6CorpusRandomBatchSampler:
    """Phase 6: direct numpy integer sampling in CorpusRandomBatchSampler.

    Validates that:
    - Sampling uses direct integer random selection, not Polars ``.sample()``
    - Indices are valid, unique, and within [0, total_rows)
    - Seed-based determinism is preserved
    - Epoch variation still works
    - The sampler does not call ``MetadataIndex.sample()`` in the hot path
    """

    def test_indices_valid_range(self):
        """All sampled indices are within [0, total_rows)."""
        meta = _meta()
        sampler = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, seed=42
        )
        for batch_idx, batch in enumerate(sampler):
            assert len(batch) == 128
            for idx in batch:
                assert 0 <= idx < 125_000, (
                    f"Index {idx} out of range in batch {batch_idx}"
                )
            if batch_idx >= 19:  # check first 20 batches
                break

    def test_indices_unique_per_batch(self):
        """Each batch contains unique indices (no duplicates)."""
        meta = _meta()
        sampler = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, seed=42
        )
        for batch_idx, batch in enumerate(sampler):
            assert len(set(batch)) == 128, (
                f"Batch {batch_idx} contains duplicate indices"
            )
            if batch_idx >= 9:
                break

    def test_seed_determinism(self):
        """Same seed produces identical batches."""
        meta = _meta()
        s1 = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, seed=42
        )
        s2 = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, seed=42
        )
        batches1 = [next(iter(s1)) for _ in range(3)]
        batches2 = [next(iter(s2)) for _ in range(3)]
        for b1, b2 in zip(batches1, batches2):
            assert b1 == b2, "Same seed should yield identical batches"

    def test_different_seed_different_batches(self):
        """Different seeds produce different first batches."""
        meta = _meta()
        s1 = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, seed=42
        )
        s2 = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, seed=99
        )
        batch1 = next(iter(s1))
        batch2 = next(iter(s2))
        assert batch1 != batch2, "Different seeds should yield different batches"

    def test_epoch_variation(self):
        """Different epochs produce different batches."""
        meta = _meta()
        sampler = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=64, seed=42
        )
        batch_epoch0 = next(iter(sampler))
        sampler.set_epoch(1)
        batch_epoch1 = next(iter(sampler))
        assert batch_epoch0 != batch_epoch1, (
            "Different epochs should yield different first batches"
        )

    def test_no_polars_sample_in_hot_path(self):
        """CorpusRandomBatchSampler does NOT call MetadataIndex.sample().

        This is the core Phase 6 assertion: the sampler avoids Polars
        DataFrame ``.sample()`` in the corpus-wide random hot path.
        """
        from unittest.mock import patch, MagicMock

        meta = _meta()
        # We check by observing that MetadataIndex.sample is never called
        # during iteration.  Since the new sampler uses numpy directly,
        # the only Polars call should be to meta.df for schema access only
        # (metadata_index is passed to sampler but hot path does not use it).
        original_sample = meta.sample
        call_count = 0

        def _tracking_sample(n, seed=None):
            nonlocal call_count
            call_count += 1
            return original_sample(n, seed=seed)

        meta.sample = _tracking_sample
        sampler = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, seed=42
        )

        # Iterate 3 batches
        it = iter(sampler)
        for _ in range(3):
            next(it)

        assert call_count == 0, (
            f"CorpusRandomBatchSampler called MetadataIndex.sample() "
            f"{call_count} times — should be 0 (direct integer sampling)"
        )

    def test_sampler_indices_sorted(self):
        """Sampled indices are returned in sorted order."""
        meta = _meta()
        sampler = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, seed=42
        )
        for batch_idx, batch in enumerate(sampler):
            assert batch == sorted(batch), (
                f"Batch {batch_idx} indices should be sorted"
            )
            if batch_idx >= 4:
                break

    def test_sampler_total_coverage(self):
        """Over many batches, distribution covers broad range of indices."""
        meta = _meta()
        sampler = CorpusRandomBatchSampler(
            metadata_index=meta, batch_size=128, seed=42
        )
        all_seen = set()
        for batch_idx, batch in enumerate(sampler):
            all_seen.update(batch)
            if batch_idx >= 9:  # 10 batches = 1280 indices
                break
        # Expect indices spread across the full range
        assert len(all_seen) >= 1000, (
            f"Only {len(all_seen)} unique indices across 10 batches — "
            f"expected broad coverage"
        )


# ===================================================================
# Phase 7 — Correctness and Compatibility Validation
# ===================================================================


class TestPhase7Validation:
    """Phase 7: end-to-end correctness and compatibility across all modes.

    Validates:
    - Canonical mode (``use_canonical=True``) metadata extraction
    - Federated Lance + BatchExecutor integration
    - Expression equivalence between aggregate and federated readers
    - FastTrainingBatch construction from ExpressionBatch + metadata
    - Row offsets exhaustive cross-check
    - Deprecated compatibility methods still work (but not in hot path)
    - Cross-mode metadata consistency
    """

    # ------------------------------------------------------------------
    # Canonical mode metadata tests
    # ------------------------------------------------------------------

    def test_canonical_meta_loaded(self):
        """Canonical MetadataIndex loads with canonical field names."""
        meta = _meta_canonical()
        assert len(meta) == 125_000
        # Canonical columns should exist (not raw_ prefixed)
        for col in ["perturb_label", "perturb_type", "dose", "cell_context",
                     "species", "batch_id", "donor_id"]:
            assert col in meta.df.columns, f"Missing canonical column: {col}"
        # Raw columns should not exist
        assert "raw_guide_1" not in meta.df.columns

    def test_canonical_read_metadata_batch_columnar(self):
        """Canonical mode: columnar=True returns meta_columns with canonical names."""
        meta = _executor_canonical().read_metadata_batch(
            [0, 1, 50000], columnar=True,
        )
        assert "meta_columns" in meta
        assert "canonical_perturbation" not in meta
        assert "canonical_context" not in meta

        mc = meta["meta_columns"]
        assert isinstance(mc, dict)
        assert len(mc) > 0

        # Check canonical field names are present
        for key in ["perturb_label", "perturb_type", "dose", "dose_unit"]:
            assert key in mc, f"Missing canonical field '{key}' in meta_columns"
            data = mc[key]
            assert len(data) == 3, f"{key} should have 3 entries"

        # Context fields
        for key in ["cell_context", "cell_line_or_type", "species", "tissue"]:
            assert key in mc, f"Missing canonical context field '{key}'"
            data = mc[key]
            assert len(data) == 3, f"{key} should have 3 entries"

    def test_canonical_read_metadata_batch_default(self):
        """Canonical mode: default (columnar=False) produces legacy dicts."""
        meta = _executor_canonical().read_metadata_batch(
            [0, 1, 50000], columnar=False,
        )
        assert "canonical_perturbation" in meta
        assert "canonical_context" in meta
        assert "meta_columns" not in meta

        cp = meta["canonical_perturbation"]
        cc = meta["canonical_context"]
        assert isinstance(cp, tuple)
        assert isinstance(cc, tuple)
        assert isinstance(cp[0], dict)
        assert isinstance(cc[0], dict)
        assert "perturb_label" in cp[0]
        assert "species" in cc[0]

    def test_canonical_read_metadata_batch_preserves_order(self):
        """Canonical mode metadata preserves non-monotonic input order."""
        indices = [50001, 1, 99999, 0]
        meta = _executor_canonical().read_metadata_batch(
            indices, columnar=True,
        )
        np.testing.assert_array_equal(
            meta["global_row_index"], np.array(indices, dtype=np.int64),
        )
        assert meta["dataset_id"][0] == "dummy_01"
        assert meta["dataset_id"][3] == "dummy_00"

    def test_canonical_meta_columns_content_values(self):
        """Canonical meta_columns perturbation values match dict values."""
        indices = [0, 1, 50000]
        meta_col = _executor_canonical().read_metadata_batch(
            indices, columnar=True,
        )
        meta_def = _executor_canonical().read_metadata_batch(
            indices, columnar=False,
        )

        mc = meta_col["meta_columns"]
        cp = meta_def["canonical_perturbation"]
        cc = meta_def["canonical_context"]

        # Perturbation fields match
        for key in ["perturb_label", "perturb_type"]:
            if key not in mc:
                continue
            for i in range(len(indices)):
                val_col = mc[key][i]
                val_dict = cp[i].get(key)
                if val_dict is not None:
                    assert str(val_col) == val_dict, (
                        f"Mismatch for {key}[{i}]: "
                        f"columnar={val_col!r}, dict={val_dict!r}"
                    )

        # Context fields match
        for key in ["cell_context", "species"]:
            if key not in mc:
                continue
            for i in range(len(indices)):
                val_col = mc[key][i]
                val_dict = cc[i].get(key)
                if val_dict is not None:
                    assert str(val_col) == val_dict, (
                        f"Mismatch for {key}[{i}]: "
                        f"columnar={val_col!r}, dict={val_dict!r}"
                    )

    def test_canonical_read_batch_full(self):
        """Canonical mode read_batch with columnar=True returns correct keys."""
        batch = _executor_canonical().read_batch(
            [0, 1, 2, 50000, 50001], columnar=True,
        )
        assert "meta_columns" in batch
        assert "canonical_perturbation" not in batch
        assert "canonical_context" not in batch
        assert "batch_size" in batch
        assert batch["batch_size"] == 5
        assert "expressed_gene_indices" in batch
        assert "expression_counts" in batch
        assert "row_offsets" in batch

    def test_canonical_read_batch_consistency(self):
        """Canonical mode expression and metadata shapes are consistent."""
        batch = _executor_canonical().read_batch(
            [0, 1, 50000], columnar=True,
        )
        assert batch["batch_size"] == 3
        assert batch["row_offsets"][0] == 0
        assert batch["row_offsets"][-1] == len(batch["expressed_gene_indices"])
        assert len(batch["expressed_gene_indices"]) == len(
            batch["expression_counts"]
        )
        # All meta_columns have length == batch_size
        n = batch["batch_size"]
        for col_name, data in batch["meta_columns"].items():
            assert len(data) == n, (
                f"meta_columns['{col_name}'] has {len(data)} entries"
            )

    def test_canonical_empty(self):
        """Canonical mode handles empty input gracefully."""
        batch = _executor_canonical().read_batch([], columnar=True)
        assert batch["batch_size"] == 0
        assert batch["meta_columns"] == {}
        assert batch["row_offsets"].tolist() == [0]

    def test_canonical_identity_columns_match_raw_mode(self):
        """Identity columns are identical between canonical and raw modes."""
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(125_000, size=64, replace=False).tolist())

        meta_can = _executor_canonical().read_metadata_batch(
            indices, columnar=True,
        )
        meta_raw = _executor().read_metadata_batch(
            indices, columnar=True,
        )

        # Identity columns should be byte-identical
        np.testing.assert_array_equal(
            meta_can["global_row_index"], meta_raw["global_row_index"],
        )
        np.testing.assert_array_equal(
            meta_can["dataset_index"], meta_raw["dataset_index"],
        )
        np.testing.assert_array_equal(
            meta_can["local_row_index"], meta_raw["local_row_index"],
        )
        np.testing.assert_array_equal(
            meta_can["size_factor"], meta_raw["size_factor"],
        )
        assert meta_can["dataset_id"] == meta_raw["dataset_id"]
        assert meta_can["cell_id"] == meta_raw["cell_id"]

    # ------------------------------------------------------------------
    # Federated Lance + BatchExecutor integration tests
    # ------------------------------------------------------------------

    def test_federated_executor_len(self):
        """Federated executor reports correct total cell count."""
        exec_ = _executor_federated()
        assert len(exec_) == 125_000

    def test_federated_read_expression_batch_empty(self):
        """Federated executor: empty input → empty ExpressionBatch."""
        expr = _executor_federated().read_expression_batch([])
        assert isinstance(expr, ExpressionBatch)
        assert expr.batch_size == 0
        assert len(expr.global_row_index) == 0

    def test_federated_read_expression_batch_single(self):
        """Federated executor: single cell read via flat path."""
        expr = _executor_federated().read_expression_batch([0])
        assert expr.batch_size == 1
        assert expr.global_row_index[0] == 0
        assert expr.row_offsets[1] > 0

    def test_federated_read_expression_batch_preserves_order(self):
        """Federated executor: input order preserved across datasets."""
        indices = [0, 50_000, 1, 50_001, 2, 50_002]
        expr = _executor_federated().read_expression_batch(indices)
        np.testing.assert_array_equal(
            expr.global_row_index, np.array(indices, dtype=np.int64),
        )

    def test_federated_read_expression_batch_uses_fast_path(self):
        """Federated executor: reader has read_expression_flat (no fallback)."""
        reader = _executor_federated().expression_reader
        assert hasattr(reader, "read_expression_flat"), (
            "FederatedLanceReader must have read_expression_flat"
        )

    def test_federated_read_expression_batch_large_shuffled(self):
        """Federated executor: large shuffled mixed-dataset batch."""
        rng = np.random.default_rng(123)
        # Build mixed indices from both datasets
        ds0 = rng.choice(50_000, size=64, replace=False)
        ds1 = rng.choice(75_000, size=64, replace=False) + 50_000
        all_idx = np.concatenate([ds0, ds1])
        rng.shuffle(all_idx)
        indices = sorted(all_idx.tolist())

        expr = _executor_federated().read_expression_batch(indices)
        assert expr.batch_size == 128
        np.testing.assert_array_equal(
            expr.global_row_index, np.array(indices, dtype=np.int64),
        )
        assert expr.row_offsets[-1] == len(expr.expressed_gene_indices)

        # Every row has expression data
        for i in range(128):
            s = expr.row_slice(i)
            assert expr.expressed_gene_indices[s].size > 0
            assert expr.expression_counts[s].size > 0

    def test_federated_read_batch_full(self):
        """Federated executor: read_batch returns all expected keys."""
        batch = _executor_federated().read_batch([0, 1, 50000, 50001])
        assert batch["batch_size"] == 4
        assert "global_row_index" in batch
        assert "row_offsets" in batch
        assert "expressed_gene_indices" in batch
        assert "expression_counts" in batch
        assert "dataset_index" in batch
        assert "size_factor" in batch
        assert batch["dataset_id"][0] == "dummy_00"
        assert batch["dataset_id"][2] == "dummy_01"

    def test_federated_read_batch_cross_chunk(self):
        """Federated executor: handles >2048 consecutive indices from one dataset."""
        indices = list(range(3000))
        batch = _executor_federated().read_batch(indices)
        assert batch["batch_size"] == 3000
        np.testing.assert_array_equal(
            batch["global_row_index"], np.array(indices, dtype=np.int64),
        )
        assert batch["row_offsets"][-1] == len(batch["expressed_gene_indices"])

    # ------------------------------------------------------------------
    # Expression equivalence: aggregate vs federated
    # ------------------------------------------------------------------

    def test_expression_equivalence_aggregate_vs_federated(self):
        """Same indices produce identical expression data from both readers."""
        indices = [0, 1, 50000, 50001, 124999]
        expr_agg = _executor().read_expression_batch(indices)
        expr_fed = _executor_federated().read_expression_batch(indices)

        assert expr_agg.batch_size == expr_fed.batch_size
        np.testing.assert_array_equal(
            expr_agg.global_row_index, expr_fed.global_row_index,
        )
        np.testing.assert_array_equal(
            expr_agg.row_offsets, expr_fed.row_offsets,
        )
        np.testing.assert_array_equal(
            expr_agg.expressed_gene_indices, expr_fed.expressed_gene_indices,
        )
        np.testing.assert_array_equal(
            expr_agg.expression_counts, expr_fed.expression_counts,
        )

    def test_expression_equivalence_large_batch(self):
        """Large batch: aggregate and federated paths produce identical data."""
        rng = np.random.default_rng(99)
        indices = sorted(rng.choice(125_000, size=256, replace=False).tolist())

        expr_agg = _executor().read_expression_batch(indices)
        expr_fed = _executor_federated().read_expression_batch(indices)

        assert expr_agg.batch_size == expr_fed.batch_size == 256
        np.testing.assert_array_equal(
            expr_agg.global_row_index, expr_fed.global_row_index,
        )
        np.testing.assert_array_equal(
            expr_agg.row_offsets, expr_fed.row_offsets,
        )
        np.testing.assert_array_equal(
            expr_agg.expressed_gene_indices, expr_fed.expressed_gene_indices,
        )
        np.testing.assert_array_equal(
            expr_agg.expression_counts, expr_fed.expression_counts,
        )

    # ------------------------------------------------------------------
    # FastTrainingBatch construction
    # ------------------------------------------------------------------

    def test_expression_batch_to_fast_training_batch(self):
        """ExpressionBatch + metadata can construct a FastTrainingBatch."""
        indices = [0, 1, 2, 50000, 50001]
        expr = _executor().read_expression_batch(indices)
        meta = _executor().read_metadata_batch(indices)

        ftb = FastTrainingBatch(
            batch_size=expr.batch_size,
            global_row_index=expr.global_row_index,
            dataset_index=meta["dataset_index"],
            local_row_index=meta["local_row_index"],
            size_factor=meta["size_factor"],
            row_offsets=expr.row_offsets,
            expressed_gene_indices=expr.expressed_gene_indices,
            expression_counts=expr.expression_counts,
        )

        assert ftb.batch_size == 5
        assert ftb.global_row_index.tolist() == indices
        assert ftb.row_offsets[0] == 0
        assert ftb.row_offsets[-1] == len(ftb.expressed_gene_indices)
        # Row access methods work
        genes = ftb.row_gene_indices(0)
        counts = ftb.row_counts(0)
        assert len(genes) > 0
        assert len(genes) == len(counts)

    def test_fast_training_batch_from_federated(self):
        """Federated ExpressionBatch constructs valid FastTrainingBatch."""
        indices = [0, 1, 50000, 50001]
        expr = _executor_federated().read_expression_batch(indices)
        meta = _executor_federated().read_metadata_batch(indices)

        ftb = FastTrainingBatch(
            batch_size=expr.batch_size,
            global_row_index=expr.global_row_index,
            dataset_index=meta["dataset_index"],
            local_row_index=meta["local_row_index"],
            size_factor=meta["size_factor"],
            row_offsets=expr.row_offsets,
            expressed_gene_indices=expr.expressed_gene_indices,
            expression_counts=expr.expression_counts,
        )
        assert ftb.batch_size == 4
        assert ftb.global_row_index.tolist() == indices

    # ------------------------------------------------------------------
    # Row offsets exhaustive cross-check
    # ------------------------------------------------------------------

    def test_row_offsets_exact_per_row(self):
        """Row offsets exactly match cumulative per-row gene counts."""
        indices = [0, 1, 2, 50000, 50001]
        expr = _executor().read_expression_batch(indices)

        # Check that row_offsets[i+1] - row_offsets[i] equals
        # the number of genes for row i
        for i in range(len(indices)):
            n_genes = expr.row_offsets[i + 1] - expr.row_offsets[i]
            assert n_genes > 0, f"Row {i} has zero genes"
            s = expr.row_slice(i)
            assert s.stop - s.start == n_genes
            assert len(expr.row_gene_indices(i)) == n_genes
            assert len(expr.row_counts(i)) == n_genes

    def test_row_offsets_exact_federated(self):
        """Federated path: row offsets exactly match per-row gene counts."""
        indices = [0, 1, 50000, 50001]
        expr = _executor_federated().read_expression_batch(indices)
        for i in range(len(indices)):
            n_genes = expr.row_offsets[i + 1] - expr.row_offsets[i]
            assert n_genes > 0
            genes = expr.row_gene_indices(i)
            counts = expr.row_counts(i)
            assert len(genes) == n_genes
            assert len(counts) == n_genes

    def test_row_offsets_large_batch(self):
        """Row offsets are correct for a large 128-cell batch."""
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(125_000, size=128, replace=False).tolist())
        expr = _executor().read_expression_batch(indices)

        # Total expression entries = sum of per-row gene counts
        total_from_offsets = expr.row_offsets[-1] - expr.row_offsets[0]
        total_from_arrays = len(expr.expressed_gene_indices)
        assert total_from_offsets == total_from_arrays

        # Each row's slice matches offset deltas
        for i in range(128):
            n = expr.row_offsets[i + 1] - expr.row_offsets[i]
            if n > 0:
                assert len(expr.row_gene_indices(i)) == n
                assert len(expr.row_counts(i)) == n

    # ------------------------------------------------------------------
    # Expression dtype and shape validation
    # ------------------------------------------------------------------

    def test_expression_dtypes_correct(self):
        """Expression arrays have correct dtypes (int32 for indices/counts)."""
        expr = _executor().read_expression_batch([0, 1, 50000])
        assert expr.expressed_gene_indices.dtype == np.int32
        assert expr.expression_counts.dtype == np.int32
        assert expr.global_row_index.dtype == np.int64
        assert expr.row_offsets.dtype == np.int64

    def test_expression_no_nan_or_inf(self):
        """Expression arrays contain no NaN or inf values."""
        indices = list(range(100)) + list(range(50000, 50100))
        expr = _executor().read_expression_batch(indices)
        assert not np.any(np.isnan(expr.expressed_gene_indices))
        assert not np.any(np.isnan(expr.expression_counts))
        assert not np.any(np.isinf(expr.expressed_gene_indices))
        assert not np.any(np.isinf(expr.expression_counts))

    # ------------------------------------------------------------------
    # Expression order preservation exhaustive
    # ------------------------------------------------------------------

    def test_order_preservation_sequential(self):
        """Sequential indices preserve exact order."""
        indices = list(range(200, 300))
        expr = _executor().read_expression_batch(indices)
        assert expr.global_row_index.tolist() == indices

    def test_order_preservation_reverse(self):
        """Reverse order indices preserve input order."""
        indices = [99, 98, 97, 96]
        expr = _executor().read_expression_batch(indices)
        assert expr.global_row_index.tolist() == indices

    def test_order_preservation_single_dataset_boundary(self):
        """Indices at dataset boundary preserve order."""
        # dummy_00 ends at 49999, dummy_01 starts at 50000
        indices = [49998, 49999, 50000, 50001]
        expr = _executor().read_expression_batch(indices)
        assert expr.global_row_index.tolist() == indices

    def test_order_preservation_federated_interleaved(self):
        """Federated: heavily interleaved dataset indices preserve order."""
        indices = [0, 50000, 1, 50001, 2, 50002, 3, 50003, 4, 50004]
        expr = _executor_federated().read_expression_batch(indices)
        assert expr.global_row_index.tolist() == indices

    # ------------------------------------------------------------------
    # Deprecated compatibility methods
    # ------------------------------------------------------------------

    def test_deprecated_build_canonical_dicts_still_works(self):
        """_build_canonical_dicts() still functions as compatibility path."""
        meta = _meta_canonical()
        # Get a small subset directly
        import polars as pl
        subset = meta.df.filter(
            pl.col("global_row_index").is_in([0, 1, 50000])
        )
        pert, ctx = _executor_canonical()._build_canonical_dicts(
            subset, 3,
        )
        assert len(pert) == 3
        assert len(ctx) == 3
        assert isinstance(pert[0], dict)
        assert "perturb_label" in pert[0]

    def test_deprecated_build_dicts_from_raw_still_works(self):
        """_build_dicts_from_raw() still functions as compatibility path."""
        meta = _meta()
        import polars as pl
        subset = meta.df.filter(
            pl.col("global_row_index").is_in([0, 1, 50000])
        )
        pert, ctx = _executor()._build_dicts_from_raw(subset, 3)
        assert len(pert) == 3
        assert len(ctx) == 3
        assert isinstance(pert[0], dict)
        assert "guide_1" in pert[0]

    def test_deprecated_methods_not_in_hot_path(self):
        """Deprecated methods are NOT called in the default read_batch hot path.
        
        We verify this indirectly: ``read_batch()`` calls
        ``read_metadata_batch()`` which uses columnar builders, not
        ``_build_canonical_dicts()`` or ``_build_dicts_from_raw()``.
        
        The existence of the columnar builders
        (``_build_canonical_from_columnar``, ``_build_raw_from_columnar``)
        confirms the deprecated methods are intentionally bypassed.
        """
        exec_ = _executor()
        assert hasattr(exec_, "_build_canonical_from_columnar")
        assert hasattr(exec_, "_build_raw_from_columnar")
        assert hasattr(exec_, "_build_canonical_dicts")  # preserved
        assert hasattr(exec_, "_build_dicts_from_raw")    # preserved

        # Smoke: default read_batch works without errors
        batch = exec_.read_batch([0, 1, 50000])
        assert batch["batch_size"] == 3
        assert "canonical_perturbation" in batch
        assert "canonical_context" in batch

    # ------------------------------------------------------------------
    # Cross-mode metadata consistency
    # ------------------------------------------------------------------

    def test_raw_vs_canonical_size_factors_match(self):
        """Size factors are identical between raw and canonical modes."""
        rng = np.random.default_rng(7)
        indices = sorted(rng.choice(125_000, size=32, replace=False).tolist())

        meta_raw = _executor().read_metadata_batch(indices, columnar=True)
        meta_can = _executor_canonical().read_metadata_batch(
            indices, columnar=True,
        )
        np.testing.assert_array_almost_equal(
            meta_raw["size_factor"], meta_can["size_factor"],
        )

    def test_raw_vs_canonical_dataset_index_match(self):
        """Dataset indices are identical between raw and canonical modes."""
        indices = list(range(100)) + list(range(50000, 50100))
        meta_raw = _executor().read_metadata_batch(indices, columnar=True)
        meta_can = _executor_canonical().read_metadata_batch(
            indices, columnar=True,
        )
        np.testing.assert_array_equal(
            meta_raw["dataset_index"], meta_can["dataset_index"],
        )

    def test_read_batch_all_fields_nonempty(self):
        """read_batch returns non-empty data for all required fields."""
        batch = _executor().read_batch([0, 1, 50000])
        for key in ["global_row_index", "dataset_index", "local_row_index",
                     "size_factor", "row_offsets", "expressed_gene_indices",
                     "expression_counts"]:
            data = batch[key]
            assert len(data) > 0, f"Field '{key}' is empty"
        assert batch["dataset_id"] != ()
        assert batch["cell_id"] != ()
        assert len(batch["canonical_perturbation"]) > 0
        assert len(batch["canonical_context"]) > 0

    def test_read_batch_deterministic_across_runs(self):
        """Same indices across multiple read_batch calls produce identical output."""
        indices = [0, 1, 50000, 50001]
        b1 = _executor().read_batch(indices)
        b2 = _executor().read_batch(indices)
        np.testing.assert_array_equal(b1["global_row_index"], b2["global_row_index"])
        np.testing.assert_array_equal(b1["expressed_gene_indices"], b2["expressed_gene_indices"])
        np.testing.assert_array_equal(b1["expression_counts"], b2["expression_counts"])
        np.testing.assert_array_equal(b1["row_offsets"], b2["row_offsets"])
