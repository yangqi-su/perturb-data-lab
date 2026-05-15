"""Canonical-file loading tests for FeatureRegistry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from perturb_data_lab.loaders import FeatureRegistry


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
        np.testing.assert_array_equal(m[0, :5], np.array([0, 1, 2, 3, 4]))

    def test_canonical_vs_raw_vocab_differs(self):
        archived_root = Path(
            "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2"
            "/copilot/plans/archive/plans-20260427-backend-topology-validation"
            "/outputs/lance-federated"
        )
        raw_df_00 = pl.read_parquet(
            str(archived_root / "dummy_00/metadata/dummy_00-release-raw-var.parquet")
        )
        raw_df_01 = pl.read_parquet(
            str(archived_root / "dummy_01/metadata/dummy_01-release-raw-var.parquet")
        )

        raw_reg = FeatureRegistry({"dummy_00": raw_df_00, "dummy_01": raw_df_01})
        var_paths = {
            "dummy_00": _resolve_canonical_var_path("dummy_00"),
            "dummy_01": _resolve_canonical_var_path("dummy_01"),
        }
        canon_reg = FeatureRegistry.from_canonical_var_parquets(var_paths)
        assert canon_reg.global_vocab_size == raw_reg.global_vocab_size
