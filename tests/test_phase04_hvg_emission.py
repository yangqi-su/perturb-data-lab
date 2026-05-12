"""HVG ranking artifact and corpus emission spec tests.

Tests cover:
- CorpusEmissionSpec round-trip (YAML read/write)
- CorpusEmissionSpec field accessors
- HVG ranking parquet schema and deterministic ranking semantics
- Manifest hvg_ranking_path wiring
- Loader integration with emission spec metadata wiring
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perturb_data_lab.contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL
from perturb_data_lab.materializers import (
    CorpusEmissionSpec,
    CorpusTokenizer,
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
# CorpusEmissionSpec round-trip and field accessors
# ---------------------------------------------------------------------------


class TestCorpusEmissionSpecRoundTrip:
    def test_spec_to_dict(self):
        spec = CorpusEmissionSpec(
            corpus_id="test-corpus",
            perturbation_fields=("perturbation_label", "control_flag"),
            context_fields=("dataset_id", "cell_context"),
            output_convention="dict",
        )
        d = spec.to_dict()
        assert d["kind"] == "corpus-emission-spec"
        assert d["corpus_id"] == "test-corpus"
        assert d["perturbation_fields"] == ["perturbation_label", "control_flag"]
        assert d["context_fields"] == ["dataset_id", "cell_context"]

    def test_spec_write_read_round_trip(self, tmp_path: Path):
        spec = CorpusEmissionSpec(
            corpus_id="test-corpus",
            perturbation_fields=("perturbation_label", "perturbation_type"),
            context_fields=("dataset_id", "tissue"),
            output_convention="dict",
            hvg_ranking_path="metadata/ds1/hvg.parquet",
            default_n_hvg=2000,
        )
        path = tmp_path / "corpus-emission-spec.yaml"
        spec.write_yaml(path)
        loaded = CorpusEmissionSpec.from_yaml_file(path)
        assert loaded.corpus_id == "test-corpus"
        assert loaded.perturbation_fields == ("perturbation_label", "perturbation_type")
        assert loaded.context_fields == ("dataset_id", "tissue")
        assert loaded.hvg_ranking_path == "metadata/ds1/hvg.parquet"
        assert loaded.default_n_hvg == 2000

    def test_emitted_perturbation_fields(self):
        spec = CorpusEmissionSpec(
            corpus_id="c",
            perturbation_fields=("perturbation_label", "target_id", "control_flag"),
        )
        assert spec.emitted_perturbation_fields() == ("perturbation_label", "target_id", "control_flag")

    def test_emitted_context_fields(self):
        spec = CorpusEmissionSpec(
            corpus_id="c",
            context_fields=("dataset_id", "cell_context", "tissue"),
        )
        assert spec.emitted_context_fields() == ("dataset_id", "cell_context", "tissue")

    def test_default_fields(self):
        spec = CorpusEmissionSpec(corpus_id="c")
        # Should have the Phase 1 contract defaults
        assert "perturbation_label" in spec.perturbation_fields
        assert "dataset_id" in spec.context_fields


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
            backend="arrow-parquet",
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
# Emission spec loader integration
# ---------------------------------------------------------------------------


class TestEmissionSpecLoaderIntegration:
    """Test that loaders can read and apply emission spec."""

    def _make_reader_components(self, tmp_path: Path):
        """Create minimal Arrow/HF parquet + SQLite cell metadata for integration test."""
        cells_path = tmp_path / "cells.parquet"
        meta_path = tmp_path / "meta.parquet"
        sqlite_path = tmp_path / "cell_meta.sqlite"
        n_cells = 10
        n_genes = 50

        indices_list = []
        counts_list = []
        sf_list = []
        for i in range(n_cells):
            gene_indices = sorted(
                np.random.default_rng(i).choice(n_genes, size=5, replace=False).tolist()
            )
            gene_counts = np.random.default_rng(i + 1000).integers(1, 10, size=5).tolist()
            indices_list.append(gene_indices)
            counts_list.append(gene_counts)
            sf_list.append(float(np.random.default_rng(i).uniform(0.5, 2.0)))

        table = pa.table(
            {
                "expressed_gene_indices": pa.array(indices_list, type=pa.list_(pa.int32())),
                "expression_counts": pa.array(counts_list, type=pa.list_(pa.int32())),
                "size_factor": pa.array(sf_list, type=pa.float64()),
            }
        )
        pq.write_table(table, cells_path)

        cell_ids = [f"syn_cell_{i}" for i in range(n_cells)]
        meta_table = pa.table(
            {
                "cell_id": pa.array(cell_ids, type=pa.string()),
                "size_factor": pa.array(sf_list, type=pa.float64()),
                "raw_obs": pa.array([""] * n_cells, type=pa.string()),
            }
        )
        pq.write_table(meta_table, meta_path)

        # SQLite with canonical_perturbation/context
        conn = sqlite3.connect(str(sqlite_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cell_meta "
            "(cell_id TEXT, dataset_id TEXT, dataset_release TEXT, "
            "size_factor REAL, canonical_perturbation TEXT, "
            "canonical_context TEXT, raw_obs TEXT)"
        )
        for i in range(n_cells):
            conn.execute(
                "INSERT INTO cell_meta VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"syn_cell_{i}",
                    "syn_ds",
                    "v0",
                    sf_list[i],
                    json.dumps({"perturbation_label": f"gene_{i % 3}", "control_flag": "0"}),
                    json.dumps({"dataset_id": "syn_ds", "cell_context": "treat"}),
                    json.dumps({}),
                ),
            )
        conn.commit()
        conn.close()
        return cells_path, meta_path, sqlite_path

    @pytest.mark.skip(reason="ArrowHFCellReader removed in deprecated-loader-stack cleanup (Phase 1)")
    def test_arrow_hf_cell_reader_emits_fields_via_spec(self, tmp_path: Path):
        """ArrowHFCellReader returns CellState with canonical fields populated from SQLite."""
        from perturb_data_lab.loaders import ArrowHFCellReader  # noqa: F401

        cells_path, meta_path, sqlite_path = self._make_reader_components(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"
        feature_reg = tmp_path / "feature-registry.yaml"
        size_factor_path = tmp_path / "size-factor-manifest.yaml"

        # Write minimal feature registry (50 genes)
        from perturb_data_lab.materializers.models import (
            FeatureRegistryEntry,
            FeatureRegistryManifest,
        )

        reg = FeatureRegistryManifest(
            kind="feature-registry",
            contract_version=CONTRACT_VERSION,
            registry_id="syn-reg",
            append_only=True,
            namespace="test",
            feature_id_field="gene_id",
            feature_label_field="gene_symbol",
            default_missing_value=MISSING_VALUE_LITERAL,
            entries=tuple(
                FeatureRegistryEntry(
                    token_id=j,
                    feature_id=f"gene_{j}",
                    feature_label=f"Gene {j}",
                    namespace="test",
                )
                for j in range(50)
            ),
        )
        reg.write_yaml(feature_reg)

        from perturb_data_lab.materializers.models import (
            SizeFactorEntry,
            SizeFactorManifest,
        )

        sf_entries = [
            SizeFactorEntry(cell_id=f"syn_cell_{i}", size_factor=1.0) for i in range(10)
        ]
        sf_manifest = SizeFactorManifest(
            kind="size-factor-manifest",
            contract_version=CONTRACT_VERSION,
            method="sum",
            entries=tuple(sf_entries),
        )
        sf_manifest.write_yaml(size_factor_path)

        reader = ArrowHFCellReader(
            dataset_id="syn_ds",
            dataset_index=0,
            corpus_index_path=corpus_index,
            cells_parquet_path=cells_path,
            meta_parquet_path=meta_path,
            cell_meta_sqlite_path=sqlite_path,
            feature_registry_path=feature_reg,
            size_factor_manifest_path=size_factor_path,
        )

        cell = reader.read_cell(0)
        # Verify emission fields are populated from SQLite
        assert cell.canonical_perturbation.get("perturbation_label") is not None
        assert cell.canonical_context.get("dataset_id") == "syn_ds"
        assert cell.canonical_perturbation.get("control_flag") is not None


# ---------------------------------------------------------------------------
# update_corpus_index with emission spec path
# ---------------------------------------------------------------------------


class TestUpdateCorpusIndexWithEmissionSpec:
    def test_new_corpus_writes_global_metadata_with_emission_spec_path(self, tmp_path: Path):
        """Creating a new corpus index writes global-metadata.yaml with emission_spec_path."""
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
            emission_spec_path="corpus-emission-spec.yaml",
        )
        # Verify global-metadata.yaml was written
        global_meta_path = tmp_path / "global-metadata.yaml"
        assert global_meta_path.exists()
        import yaml

        meta = yaml.safe_load(global_meta_path.read_text())
        assert meta["emission_spec_path"] == "corpus-emission-spec.yaml"
        # tokenizer_path is None in Phase 3 (tokenizer-free architecture)
        assert meta.get("tokenizer_path") is None


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
            backend="arrow-parquet",
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
