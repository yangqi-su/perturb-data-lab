"""Phase 3 tests: canonical-file loading paths for MetadataIndex and FeatureRegistry.

Validates:
- ``MetadataIndex.from_dummy_data(use_canonical=True)`` — canonical obs loading
- ``FeatureRegistry.from_canonical_var_parquets()`` — canonical gene ID vocab
- Backward compatibility: raw-mode loading still works
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from perturb_data_lab.loaders import FeatureRegistry, MetadataIndex

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARCHIVED_ROOT = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/archive/plans-20260427-backend-topology-validation/outputs"
)
_PLAN_RUN = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/active/plans-20260430-canonicalization-module-and-loader-adaptation"
    / "outputs"
)

_PLAN_RUN_ARCHIVE = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/archive/plans-20260430-canonicalization-module-and-loader-adaptation"
    / "outputs"
)


def _resolve_canonical_var_path(dataset_id: str) -> str:
    for base in (_PLAN_RUN, _PLAN_RUN_ARCHIVE):
        candidates = [
            base / dataset_id / "canonical-var.parquet",
            base / dataset_id / f"{dataset_id}-canonical-var.parquet",
            base / dataset_id / f"{dataset_id}-release-canonical-var.parquet",
            base / f"{dataset_id}-canonical-var.parquet",
            base / f"{dataset_id}-release-canonical-var.parquet",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
    raise FileNotFoundError(f"canonical var parquet not found for {dataset_id}")

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

    def test_structural_columns_are_typed(self):
        meta = _meta_canon()
        schema = meta.df.schema
        assert schema["global_row_index"] == pl.Int64
        assert schema["dataset_index"] == pl.Int32
        assert schema["local_row_index"] == pl.Int64
        assert schema["size_factor"] == pl.Float64

    def test_size_factors_populated(self):
        meta = _meta_canon()
        sf = meta.df["size_factor"].to_numpy()
        assert np.all(sf > 0.0)
        assert np.all(np.isfinite(sf))

    def test_safe_missing_columns_load_as_nulls(self):
        """Safe nullable canonical columns are normalized to real nulls."""
        meta = _meta_canon()
        dose_vals = meta.df["dose"].to_list()[:10]
        assert all(v is None for v in dose_vals)

    def test_canonical_vs_raw_same_count(self):
        """Canonical and raw indices have the same total row count."""
        meta_canon = _meta_canon()
        meta_raw = _meta_raw()
        assert len(meta_canon) == len(meta_raw) == 125_000

# ---------------------------------------------------------------------------
# FeatureRegistry — canonical var
# ---------------------------------------------------------------------------


class TestFeatureRegistryCanonical:
    """FeatureRegistry built from canonical var parquets."""

    def test_from_canonical_var_parquets_loads(self):
        var_paths = {
            "dummy_00": _resolve_canonical_var_path("dummy_00"),
            "dummy_01": _resolve_canonical_var_path("dummy_01"),
        }
        reg = FeatureRegistry.from_canonical_var_parquets(var_paths)
        assert len(reg.dataset_ids) == 2
        assert "dummy_00" in reg.dataset_ids
        assert "dummy_01" in reg.dataset_ids

    def test_global_vocab_size(self):
        var_paths = {
            "dummy_00": _resolve_canonical_var_path("dummy_00"),
            "dummy_01": _resolve_canonical_var_path("dummy_01"),
        }
        reg = FeatureRegistry.from_canonical_var_parquets(var_paths)
        # 22652 unique canonical_gene_ids across both datasets
        assert reg.global_vocab_size == 22652

    def test_max_local_vocab(self):
        var_paths = {
            "dummy_00": _resolve_canonical_var_path("dummy_00"),
            "dummy_01": _resolve_canonical_var_path("dummy_01"),
        }
        reg = FeatureRegistry.from_canonical_var_parquets(var_paths)
        assert reg.max_local_vocab == 15000

    def test_local_to_global_map(self):
        var_paths = {
            "dummy_00": _resolve_canonical_var_path("dummy_00"),
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
            "dummy_00": _resolve_canonical_var_path("dummy_00"),
            "dummy_01": _resolve_canonical_var_path("dummy_01"),
        }
        canon_reg = FeatureRegistry.from_canonical_var_parquets(var_paths)

        # For identity mapping, they should be the same
        assert canon_reg.global_vocab_size == raw_reg.global_vocab_size
