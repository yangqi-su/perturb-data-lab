"""HVG ranking artifact tests.

Tests cover:
- HVG ranking parquet schema and deterministic ranking semantics
- Manifest hvg_ranking_path wiring
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perturb_data_lab.contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL
from perturb_data_lab.materializers import (
    update_corpus_index,
)
from perturb_data_lab.materializers.chunk_translation import (
    HVG_RANKING_SCHEMA,
    _build_hvg_ranking_table,
)
from perturb_data_lab.materializers.models import (
    CountSourceSpec,
    DatasetJoinRecord,
    GlobalMetadataDocument,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
)

# ---------------------------------------------------------------------------
# HVG ranking parquet content tests
# ---------------------------------------------------------------------------


class TestHVGRankingParquet:
    def test_hvg_parquet_saved_and_loadable(self, tmp_path: Path):
        table = _build_hvg_ranking_table(
            sum_log1p=np.array([3.0, 1.5, 2.0], dtype=np.float64),
            sum_log1p_sq=np.array([5.0, 1.5, 2.8], dtype=np.float64),
            n_cells_total=3,
            feature_ids=["gene_a", "gene_b", "gene_c"],
            n_hvg=2,
        )
        path = tmp_path / "hvg.parquet"
        pq.write_table(table, path)
        loaded = pq.read_table(path)
        assert loaded.schema == HVG_RANKING_SCHEMA
        assert loaded.num_rows == 3

    def test_hvg_ranking_schema_and_default_selection_equivalence(self):
        table = _build_hvg_ranking_table(
            sum_log1p=np.array([6.0, 2.0, 5.0, 1.5], dtype=np.float64),
            sum_log1p_sq=np.array([12.0, 2.5, 9.0, 1.0], dtype=np.float64),
            n_cells_total=4,
            feature_ids=["g0", "g1", "g2", "g3"],
            n_hvg=2,
        )
        frame = table.to_pandas()
        assert frame.columns.tolist() == [field.name for field in HVG_RANKING_SCHEMA]
        assert frame["origin_index"].tolist() == [0, 1, 2, 3]
        assert frame["feature_id"].tolist() == ["g0", "g1", "g2", "g3"]
        np.testing.assert_array_equal(
            frame["selected_at_default_n_hvg"].to_numpy(dtype=bool),
            (frame["hvg_rank"].to_numpy(dtype=np.int32) <= 2),
        )

    def test_hvg_rank_tie_breaks_by_origin_index(self):
        table = _build_hvg_ranking_table(
            sum_log1p=np.array([4.0, 4.0, 4.0], dtype=np.float64),
            sum_log1p_sq=np.array([7.0, 7.0, 7.0], dtype=np.float64),
            n_cells_total=4,
            feature_ids=["g0", "g1", "g2"],
            n_hvg=2,
        )
        frame = table.to_pandas()
        assert frame["hvg_rank"].tolist() == [1, 2, 3]
        assert frame["selected_at_default_n_hvg"].tolist() == [True, True, False]

    def test_stage2_materializer_writes_hvg_parquet(self, tmp_path: Path):
        from perturb_data_lab.materializers import Stage2Materializer

        materializer = Stage2Materializer(
            source_path="/fake/source.h5ad",
            review_bundle_path="/fake/dataset-summary.yaml",
            output_roots=OutputRoots(
                metadata_root=str(tmp_path / "meta"),
                matrix_root=str(tmp_path / "matrix"),
            ),
            dataset_id="syn-ds",
            n_hvg=2,
        )
        path = materializer._write_hvg_ranking_parquet(
            sum_log1p=np.array([6.0, 2.0, 5.0], dtype=np.float64),
            sum_log1p_sq=np.array([12.0, 2.5, 9.0], dtype=np.float64),
            n_cells_total=4,
            feature_ids=("g0", "g1", "g2"),
            meta_root=tmp_path / "meta",
        )
        loaded = pq.read_table(path).to_pandas()
        assert path.name == "hvg.parquet"
        assert loaded["feature_id"].tolist() == ["g0", "g1", "g2"]
        np.testing.assert_array_equal(
            loaded["selected_at_default_n_hvg"].to_numpy(dtype=bool),
            (loaded["hvg_rank"].to_numpy(dtype=np.int32) <= 2),
        )


# ---------------------------------------------------------------------------
# MaterializationManifest HVG ranking wiring
# ---------------------------------------------------------------------------


class TestMaterializationManifestHVGRoundTrip:
    def test_manifest_hvg_ranking_path_set(self, tmp_path: Path):
        """MaterializationManifest can hold hvg_ranking_path and default_n_hvg."""
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version=CONTRACT_VERSION,
            dataset_id="ds1",
            route="create_new",
            backend="zarr",
            topology="federated",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=OutputRoots(
                metadata_root=str(tmp_path / "meta"),
                matrix_root=str(tmp_path / "matrix"),
            ),
            provenance=ProvenanceSpec(
                source_path="/path/to/source.h5ad",
                review_bundle="/path/to/dataset-summary.yaml",
            ),
            hvg_ranking_path=str(tmp_path / "meta/hvg.parquet"),
            default_n_hvg=2000,
        )
        path = tmp_path / "manifest.yaml"
        manifest.write_yaml(path)
        loaded = MaterializationManifest.from_yaml_file(path)
        assert loaded.hvg_ranking_path is not None
        assert loaded.hvg_ranking_path.endswith("hvg.parquet")
        assert loaded.default_n_hvg == 2000


# ---------------------------------------------------------------------------
# update_corpus_index global metadata
# ---------------------------------------------------------------------------


class TestUpdateCorpusIndexGlobalMetadata:
    def test_new_corpus_writes_global_metadata(self, tmp_path: Path):
        """Creating a new corpus index writes global-metadata.yaml."""
        corpus_index = tmp_path / "corpus-index.yaml"
        dataset_record = DatasetJoinRecord(
            dataset_id="ds1",
            join_mode="create_new",
            manifest_path="meta/materialization-manifest.yaml",
            cell_count=100,
            global_start=0,
            global_end=100,
        )
        update_corpus_index(
            corpus_index_path=corpus_index,
            new_dataset_record=dataset_record,
            backend="lance",
            topology="aggregate",
        )
        # Verify global-metadata.yaml was written
        global_meta_path = tmp_path / "global-metadata.yaml"
        assert global_meta_path.exists()
        import yaml

        meta = yaml.safe_load(global_meta_path.read_text())
        assert meta["backend"] == "lance"
        assert meta["topology"] == "aggregate"


# ---------------------------------------------------------------------------
# Verify hvg_ranking_path in manifest
# ---------------------------------------------------------------------------


class TestMaterializationManifestHVGRecord:
    def test_materialization_manifest_includes_hvg_ranking_path(self, tmp_path: Path):
        """After materialization, manifest.hvg_ranking_path records hvg.parquet."""
        output_roots = OutputRoots(
            metadata_root=str(tmp_path / "meta"),
            matrix_root=str(tmp_path / "matrix"),
        )
        # Verify the manifest model accepts hvg_ranking_path
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version=CONTRACT_VERSION,
            dataset_id="syn-ds",
            route="create_new",
            backend="zarr",
            topology="federated",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=output_roots,
            provenance=ProvenanceSpec(
                source_path="/fake/source.h5ad",
                review_bundle="/fake/dataset-summary.yaml",
            ),
            hvg_ranking_path=str(tmp_path / "meta/hvg.parquet"),
            default_n_hvg=1500,
        )
        manifest_path = tmp_path / "test-manifest.yaml"
        manifest.write_yaml(manifest_path)
        loaded = MaterializationManifest.from_yaml_file(manifest_path)
        assert loaded.hvg_ranking_path is not None
        assert loaded.hvg_ranking_path.endswith("hvg.parquet")
        assert loaded.default_n_hvg == 1500
