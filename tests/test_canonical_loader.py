"""Phase 3 tests: canonical-file loading paths for MetadataIndex, BatchExecutor, and FeatureRegistry.

Validates:
- ``MetadataIndex.from_dummy_data(use_canonical=True)`` — canonical obs loading
- ``BatchExecutor(..., use_canonical=True)`` — direct canonical column reads
- ``FeatureRegistry.from_canonical_var_parquets()`` — canonical gene ID vocab
- Backward compatibility: raw-mode loading still works
- Output shape compatibility between canonical and raw modes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from perturb_data_lab.loaders import (
    AggregateLanceReader,
    BatchExecutor,
    DatasetEntry,
    FeatureRegistry,
    MetadataIndex,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARCHIVED_ROOT = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/archive/plans-20260427-backend-topology-validation/outputs"
)
AGGREGATE_LANCE = str(
    _ARCHIVED_ROOT / "lance-aggregate/matrix/aggregated-cells.lance"
)

_PLAN_RUN = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/active/plans-20260430-canonicalization-module-and-loader-adaptation"
    / "outputs"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _meta_canon() -> MetadataIndex:
    if not hasattr(_meta_canon, "_cache"):
        _meta_canon._cache = MetadataIndex.from_dummy_data(use_canonical=True)
    return _meta_canon._cache


def _meta_raw() -> MetadataIndex:
    if not hasattr(_meta_raw, "_cache"):
        _meta_raw._cache = MetadataIndex.from_dummy_data(use_canonical=False)
    return _meta_raw._cache


def _exec_canon() -> BatchExecutor:
    if not hasattr(_exec_canon, "_cache"):
        meta = _meta_canon()
        agg_entry = DatasetEntry("aggregated", 0, len(meta))
        reader = AggregateLanceReader(AGGREGATE_LANCE, [agg_entry])
        _exec_canon._cache = BatchExecutor(reader, meta, use_canonical=True)
    return _exec_canon._cache


def _exec_raw() -> BatchExecutor:
    if not hasattr(_exec_raw, "_cache"):
        meta = _meta_raw()
        agg_entry = DatasetEntry("aggregated", 0, len(meta))
        reader = AggregateLanceReader(AGGREGATE_LANCE, [agg_entry])
        _exec_raw._cache = BatchExecutor(reader, meta, use_canonical=False)
    return _exec_raw._cache


# ---------------------------------------------------------------------------
# MetadataIndex — canonical loading
# ---------------------------------------------------------------------------


class TestMetadataIndexCanonical:
    """MetadataIndex with use_canonical=True."""

    def test_loads_125k_rows(self):
        meta = _meta_canon()
        assert len(meta) == 125_000

    def test_has_canonical_columns(self):
        meta = _meta_canon()
        canonical_expected = {
            "perturb_label", "perturb_type", "dose", "dose_unit",
            "timepoint", "timepoint_unit", "cell_context", "cell_line_or_type",
            "species", "tissue", "assay", "condition", "batch_id",
            "donor_id", "sex", "disease_state",
        }
        actual = set(meta.df.columns)
        assert canonical_expected <= actual, (
            f"Missing canonical columns: {canonical_expected - actual}"
        )

    def test_no_raw_prefix_columns(self):
        """Canonical path should not have raw_ prefixed columns."""
        meta = _meta_canon()
        raw_cols = [c for c in meta.df.columns if c.startswith("raw_")]
        assert len(raw_cols) == 0, (
            f"Canonical MetadataIndex should not have raw_ columns: {raw_cols}"
        )

    def test_global_row_index_contiguous(self):
        meta = _meta_canon()
        gr = meta.df["global_row_index"].to_numpy()
        expected = np.arange(len(meta), dtype=np.int64)
        assert np.array_equal(gr, expected)

    def test_dataset_index_binary(self):
        meta = _meta_canon()
        dummy_00 = meta.filter(pl.col("dataset_id") == "dummy_00")
        assert dummy_00.df["dataset_index"].unique().to_list() == [0]
        dummy_01 = meta.filter(pl.col("dataset_id") == "dummy_01")
        assert dummy_01.df["dataset_index"].unique().to_list() == [1]

    def test_size_factors_populated(self):
        meta = _meta_canon()
        sf = meta.df["size_factor"].to_numpy()
        assert np.all(sf > 0.0)
        assert np.all(np.isfinite(sf))

    def test_na_columns_exist(self):
        """Columns with null strategy should contain NA strings."""
        meta = _meta_canon()
        dose_vals = meta.df["dose"].to_list()[:10]
        assert all(v == "NA" for v in dose_vals)

    def test_canonical_vs_raw_same_count(self):
        """Canonical and raw indices have the same total row count."""
        meta_canon = _meta_canon()
        meta_raw = _meta_raw()
        assert len(meta_canon) == len(meta_raw) == 125_000


# ---------------------------------------------------------------------------
# BatchExecutor — canonical mode
# ---------------------------------------------------------------------------


class TestBatchExecutorCanonical:
    """BatchExecutor with use_canonical=True."""

    def test_read_metadata_batch_has_canonical_keys(self):
        """Canonical perturbation dict uses canonical field names."""
        meta = _exec_canon().read_metadata_batch([0, 1])
        pert0 = meta["canonical_perturbation"][0]
        assert "perturb_label" in pert0
        assert "perturb_type" in pert0
        # Should NOT have raw_ keys
        assert "guide_1" not in pert0
        assert "treatment" not in pert0

    def test_read_metadata_batch_context_keys(self):
        meta = _exec_canon().read_metadata_batch([0])
        ctx0 = meta["canonical_context"][0]
        assert "cell_context" in ctx0
        assert "cell_line_or_type" in ctx0
        assert "species" in ctx0
        assert "tissue" in ctx0
        assert "assay" in ctx0
        assert "condition" in ctx0
        assert "batch_id" in ctx0
        assert "donor_id" in ctx0
        assert "sex" in ctx0
        assert "disease_state" in ctx0

    def test_na_values_excluded_from_dicts(self):
        """NA and None values are excluded from canonical dicts."""
        meta = _exec_canon().read_metadata_batch([0])
        pert0 = meta["canonical_perturbation"][0]
        # dose/timepoint are NA in dummy data — should be absent
        assert "dose" not in pert0
        assert "timepoint" not in pert0

    def test_read_batch_shape_compatible_with_raw(self):
        """Canonical read_batch has same keys and array shapes as raw."""
        indices = [0, 1, 50000, 50001]
        batch_canon = _exec_canon().read_batch(indices)
        batch_raw = _exec_raw().read_batch(indices)

        # Same keys
        assert set(batch_canon.keys()) == set(batch_raw.keys())

        # Same numeric array shapes
        for key in [
            "global_row_index", "dataset_index", "local_row_index",
            "size_factor", "expressed_gene_indices", "expression_counts",
            "row_offsets",
        ]:
            assert batch_canon[key].shape == batch_raw[key].shape, (
                f"Shape mismatch for {key}: "
                f"{batch_canon[key].shape} vs {batch_raw[key].shape}"
            )
            assert batch_canon[key].dtype == batch_raw[key].dtype, (
                f"Dtype mismatch for {key}: "
                f"{batch_canon[key].dtype} vs {batch_raw[key].dtype}"
            )

    def test_read_batch_cross_dataset(self):
        """Canonical batch spanning dummy_00 and dummy_01 boundary."""
        indices = [49998, 49999, 50000, 50001]
        batch = _exec_canon().read_batch(indices)
        assert batch["batch_size"] == 4
        assert batch["dataset_id"][0] == "dummy_00"
        assert batch["dataset_id"][2] == "dummy_01"

    def test_expression_data_identical_canonical_vs_raw(self):
        """Expression data should be identical regardless of metadata mode."""
        indices = [0, 1, 2]
        batch_canon = _exec_canon().read_batch(indices)
        batch_raw = _exec_raw().read_batch(indices)

        np.testing.assert_array_equal(
            batch_canon["expressed_gene_indices"],
            batch_raw["expressed_gene_indices"],
        )
        np.testing.assert_array_equal(
            batch_canon["expression_counts"],
            batch_raw["expression_counts"],
        )
        np.testing.assert_array_equal(
            batch_canon["row_offsets"],
            batch_raw["row_offsets"],
        )

    def test_perturbation_dict_type(self):
        """canonical_perturbation should be tuple of dicts in both modes."""
        batch_canon = _exec_canon().read_batch([0])
        batch_raw = _exec_raw().read_batch([0])
        assert isinstance(batch_canon["canonical_perturbation"], tuple)
        assert isinstance(batch_canon["canonical_perturbation"][0], dict)
        assert isinstance(batch_raw["canonical_perturbation"], tuple)
        assert isinstance(batch_raw["canonical_perturbation"][0], dict)


# ---------------------------------------------------------------------------
# FeatureRegistry — canonical var
# ---------------------------------------------------------------------------


class TestFeatureRegistryCanonical:
    """FeatureRegistry built from canonical var parquets."""

    def test_from_canonical_var_parquets_loads(self):
        var_paths = {
            "dummy_00": str(
                _PLAN_RUN / "dummy_00" / "dummy_00-release-canonical-var.parquet"
            ),
            "dummy_01": str(
                _PLAN_RUN / "dummy_01" / "dummy_01-release-canonical-var.parquet"
            ),
        }
        reg = FeatureRegistry.from_canonical_var_parquets(var_paths)
        assert len(reg.dataset_ids) == 2
        assert "dummy_00" in reg.dataset_ids
        assert "dummy_01" in reg.dataset_ids

    def test_global_vocab_size(self):
        var_paths = {
            "dummy_00": str(
                _PLAN_RUN / "dummy_00" / "dummy_00-release-canonical-var.parquet"
            ),
            "dummy_01": str(
                _PLAN_RUN / "dummy_01" / "dummy_01-release-canonical-var.parquet"
            ),
        }
        reg = FeatureRegistry.from_canonical_var_parquets(var_paths)
        # 22652 unique canonical_gene_ids across both datasets
        assert reg.global_vocab_size == 22652

    def test_max_local_vocab(self):
        var_paths = {
            "dummy_00": str(
                _PLAN_RUN / "dummy_00" / "dummy_00-release-canonical-var.parquet"
            ),
            "dummy_01": str(
                _PLAN_RUN / "dummy_01" / "dummy_01-release-canonical-var.parquet"
            ),
        }
        reg = FeatureRegistry.from_canonical_var_parquets(var_paths)
        assert reg.max_local_vocab == 15000

    def test_local_to_global_map(self):
        var_paths = {
            "dummy_00": str(
                _PLAN_RUN / "dummy_00" / "dummy_00-release-canonical-var.parquet"
            ),
        }
        reg = FeatureRegistry.from_canonical_var_parquets(var_paths)
        m = reg.local_to_global_map
        assert m.shape == (1, 10000)
        # First 5 entries should be consecutive 0..4
        np.testing.assert_array_equal(m[0, :5], np.array([0, 1, 2, 3, 4]))

    def test_canonical_vs_raw_vocab_differs(self):
        """Canonical_gene_id may differ from raw feature_id."""
        # Canonical has canonical_gene_id which may equal gene_id for dummy
        # but the concept is that the vocabulary uses harmonized IDs
        import polars as pl

        _ARCHIVED_ROOT_PATH = Path(
            "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2"
            "/copilot/plans/archive/plans-20260427-backend-topology-validation"
            "/outputs/lance-federated"
        )
        # Raw var
        raw_df_00 = pl.read_parquet(
            str(_ARCHIVED_ROOT_PATH / "dummy_00/metadata/dummy_00-release-raw-var.parquet")
        )
        raw_df_01 = pl.read_parquet(
            str(_ARCHIVED_ROOT_PATH / "dummy_01/metadata/dummy_01-release-raw-var.parquet")
        )

        raw_reg = FeatureRegistry({"dummy_00": raw_df_00, "dummy_01": raw_df_01})
        # For dummy data with identity mapping, canonical == raw vocab size
        var_paths = {
            "dummy_00": str(_PLAN_RUN / "dummy_00" / "dummy_00-release-canonical-var.parquet"),
            "dummy_01": str(_PLAN_RUN / "dummy_01" / "dummy_01-release-canonical-var.parquet"),
        }
        canon_reg = FeatureRegistry.from_canonical_var_parquets(var_paths)

        # For identity mapping, they should be the same
        assert canon_reg.global_vocab_size == raw_reg.global_vocab_size
