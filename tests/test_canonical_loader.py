"""Canonical-file loading tests for FeatureRegistry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from perturb_data_lab.loaders import FeatureRegistry


def _write_canonical_var(path: Path, genes: list[str]) -> None:
    frame = pl.DataFrame(
        {
            "origin_index": np.arange(len(genes), dtype=np.int32),
            "gene_id": [f"ENSG_{gene}" for gene in genes],
            "canonical_gene_id": genes,
            "global_id": np.arange(len(genes), dtype=np.int32),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(path))


class TestFeatureRegistryCanonical:
    def test_from_canonical_var_parquets_loads(self, tmp_path: Path) -> None:
        ds0 = tmp_path / "ds0" / "canonical-var.parquet"
        ds1 = tmp_path / "ds1" / "canonical-var.parquet"
        _write_canonical_var(ds0, ["GENE_A", "GENE_B"])
        _write_canonical_var(ds1, ["GENE_B", "GENE_C"])

        reg = FeatureRegistry.from_canonical_var_parquets(
            {"ds0": ds0, "ds1": ds1},
            global_id_by_feature_id={"GENE_A": 0, "GENE_B": 1, "GENE_C": 2},
        )

        assert reg.dataset_ids == ("ds0", "ds1")
        assert reg.global_vocab_size == 3
        assert reg.max_local_vocab == 2
        np.testing.assert_array_equal(
            reg.local_to_global_map,
            np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        )

    def test_requires_tokenizer_mapping(self, tmp_path: Path) -> None:
        ds0 = tmp_path / "ds0" / "canonical-var.parquet"
        _write_canonical_var(ds0, ["GENE_A", "GENE_B"])

        with pytest.raises(ValueError, match="missing from persisted gene tokenizer"):
            FeatureRegistry.from_canonical_var_parquets(
                {"ds0": ds0},
                global_id_by_feature_id={"GENE_A": 0},
            )

    def test_requires_canonical_gene_id(self) -> None:
        with pytest.raises(ValueError, match="canonical_gene_id"):
            FeatureRegistry(
                {
                    "ds0": pl.DataFrame(
                        {
                            "origin_index": np.asarray([0, 1], dtype=np.int32),
                            "feature_id": ["GENE_A", "GENE_B"],
                        }
                    )
                },
                global_id_by_feature_id={"GENE_A": 0, "GENE_B": 1},
            )
