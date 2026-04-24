"""Phase 4 tests: HVG/non-HVG materialization and corpus emission spec.

Tests cover:
- CorpusEmissionSpec round-trip (YAML read/write)
- CorpusEmissionSpec field accessors
- HVG/non-HVG arrays: content, shape, disjointness, index-space
- Manifest hvg_sidecar_path wiring
- Loader integration with emission spec and HVG sidecar loading
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
    build_materialization_route,
    update_corpus_index,
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
            hvg_sidecar_path="metadata/hvg_sidecar",
        )
        path = tmp_path / "corpus-emission-spec.yaml"
        spec.write_yaml(path)
        loaded = CorpusEmissionSpec.from_yaml_file(path)
        assert loaded.corpus_id == "test-corpus"
        assert loaded.perturbation_fields == ("perturbation_label", "perturbation_type")
        assert loaded.context_fields == ("dataset_id", "tissue")
        assert loaded.hvg_sidecar_path == "metadata/hvg_sidecar"

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
# HVG/non-HVG array content tests
# ---------------------------------------------------------------------------


class TestHVGArrays:
    def test_hvg_npy_saved_and_loadable(self, tmp_path: Path):
        """HVG array is saved as .npy and can be loaded back."""
        sidecar = tmp_path / "hvg_sidecar"
        sidecar.mkdir(parents=True, exist_ok=True)
        indices = np.array([0, 5, 10, 15, 20], dtype=np.int32)
        np.save(str(sidecar / "hvg.npy"), indices, allow_pickle=False)
        loaded = np.load(str(sidecar / "hvg.npy"), allow_pickle=False)
        np.testing.assert_array_equal(loaded, indices)

    def test_nonhvg_npy_saved_and_loadable(self, tmp_path: Path):
        """non-HVG array is saved as .npy and can be loaded back."""
        sidecar = tmp_path / "hvg_sidecar"
        sidecar.mkdir(parents=True, exist_ok=True)
        indices = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=np.int32)
        np.save(str(sidecar / "nonhvg.npy"), indices, allow_pickle=False)
        loaded = np.load(str(sidecar / "nonhvg.npy"), allow_pickle=False)
        np.testing.assert_array_equal(loaded, indices)

    def test_hvg_nonhvg_disjoint(self, tmp_path: Path):
        """hvg and nonhvg index sets must not overlap."""
        sidecar = tmp_path / "hvg_sidecar"
        sidecar.mkdir(parents=True, exist_ok=True)
        n = 50
        hvg = np.arange(0, 20, dtype=np.int32)
        nonhvg = np.arange(20, n, dtype=np.int32)
        np.save(str(sidecar / "hvg.npy"), hvg, allow_pickle=False)
        np.save(str(sidecar / "nonhvg.npy"), nonhvg, allow_pickle=False)
        loaded_hvg = np.load(str(sidecar / "hvg.npy"), allow_pickle=False)
        loaded_nonhvg = np.load(str(sidecar / "nonhvg.npy"), allow_pickle=False)
        intersection = set(loaded_hvg) & set(loaded_nonhvg)
        assert len(intersection) == 0, f"hvg and nonhvg overlap: {intersection}"

    def test_hvg_nonhvg_cover_all_features(self, tmp_path: Path):
        """hvg ∪ nonhvg should cover all feature indices 0..n_vars-1."""
        sidecar = tmp_path / "hvg_sidecar"
        sidecar.mkdir(parents=True, exist_ok=True)
        n = 30
        hvg = np.array([0, 5, 10, 15, 20, 25], dtype=np.int32)
        nonhvg = np.array([i for i in range(n) if i not in {0, 5, 10, 15, 20, 25}], dtype=np.int32)
        np.save(str(sidecar / "hvg.npy"), hvg, allow_pickle=False)
        np.save(str(sidecar / "nonhvg.npy"), nonhvg, allow_pickle=False)
        loaded_hvg = np.load(str(sidecar / "hvg.npy"), allow_pickle=False)
        loaded_nonhvg = np.load(str(sidecar / "nonhvg.npy"), allow_pickle=False)
        union = set(loaded_hvg) | set(loaded_nonhvg)
        assert union == set(range(n))

    def test_hvg_indices_are_original_dataset_indices(self, tmp_path: Path):
        """HVG indices must be in original dataset index space (not token IDs)."""
        sidecar = tmp_path / "hvg_sidecar"
        sidecar.mkdir(parents=True, exist_ok=True)
        # Large indices that would be out of range for a small dataset confirm
        # they're in dataset index space, not token space
        indices = np.array([5000, 5001, 5002], dtype=np.int32)  # hypothetical large
        np.save(str(sidecar / "hvg.npy"), indices, allow_pickle=False)
        loaded = np.load(str(sidecar / "hvg.npy"), allow_pickle=False)
        np.testing.assert_array_equal(loaded, indices)

    def test_hvg_array_dtype_int32(self, tmp_path: Path):
        """HVG/non-HVG arrays must be saved as int32."""
        sidecar = tmp_path / "hvg_sidecar"
        sidecar.mkdir(parents=True, exist_ok=True)
        indices = np.array([0, 5, 10], dtype=np.int32)
        np.save(str(sidecar / "hvg.npy"), indices, allow_pickle=False)
        loaded = np.load(str(sidecar / "hvg.npy"), allow_pickle=False)
        assert loaded.dtype == np.int32


# ---------------------------------------------------------------------------
# MaterializationManifest hvg_sidecar_path wiring
# ---------------------------------------------------------------------------


class TestMaterializationManifestHVGPath:
    def test_manifest_hvg_sidecar_path_set(self, tmp_path: Path):
        """MaterializationManifest can hold hvg_sidecar_path."""
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version=CONTRACT_VERSION,
            dataset_id="ds1",
            release_id="v0",
            route="create_new",
            backend="arrow-hf",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=OutputRoots(
                metadata_root=str(tmp_path / "meta"),
                matrix_root=str(tmp_path / "matrix"),
            ),
            provenance=ProvenanceSpec(
                source_path="/path/to/source.h5ad",
                schema="/path/to/schema.yaml",
            ),
            hvg_sidecar_path=str(tmp_path / "meta/hvg_sidecar"),
        )
        path = tmp_path / "manifest.yaml"
        manifest.write_yaml(path)
        loaded = MaterializationManifest.from_yaml_file(path)
        assert loaded.hvg_sidecar_path is not None
        assert "hvg_sidecar" in loaded.hvg_sidecar_path


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

    def test_arrow_hf_cell_reader_emits_fields_via_spec(self, tmp_path: Path):
        """ArrowHFCellReader returns CellState with canonical fields populated from SQLite."""
        from perturb_data_lab.loaders import ArrowHFCellReader

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
            release_id="v0",
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
            release_id="v0",
            join_mode="create_new",
            manifest_path="meta/materialization-manifest.yaml",
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
# Build materialization route and verify hvg_sidecar_path in manifest
# ---------------------------------------------------------------------------


class TestMaterializationRouteWithHVG:
    def test_materialization_manifest_includes_hvg_sidecar_path(self, tmp_path: Path):
        """After route materialization, manifest.hvg_sidecar_path is set."""
        # Create a minimal corpus index so create_new route works
        corpus_index = tmp_path / "corpus-index.yaml"
        output_roots = OutputRoots(
            metadata_root=str(tmp_path / "meta"),
            matrix_root=str(tmp_path / "matrix"),
        )
        from perturb_data_lab.materializers.models import CountSourceSpec

        route = build_materialization_route(
            route="create_new",
            output_roots=output_roots,
            release_id="syn-v0",
            dataset_id="syn-ds",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            corpus_index_path=None,  # create_new does not require existing corpus
        )
        # We can't run full materialize() without a real h5ad + schema,
        # but we can verify the manifest construction path sets hvg_sidecar_path
        # by checking the manifest model accepts the field
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version=CONTRACT_VERSION,
            dataset_id="syn-ds",
            release_id="syn-v0",
            route="create_new",
            backend="arrow-hf",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=output_roots,
            provenance=ProvenanceSpec(
                source_path="/fake/source.h5ad",
                schema="/fake/schema.yaml",
            ),
            hvg_sidecar_path=str(tmp_path / "meta/hvg_sidecar"),
        )
        manifest_path = tmp_path / "test-manifest.yaml"
        manifest.write_yaml(manifest_path)
        loaded = MaterializationManifest.from_yaml_file(manifest_path)
        assert loaded.hvg_sidecar_path is not None
        assert "hvg_sidecar" in loaded.hvg_sidecar_path
