"""Phase 3 materializer smoke tests using synthetic fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from perturb_data_lab.materializers import (
    CanonicalCellRecord,
    CreateNewRoute,
    AppendRoutedRoute,
    build_materialization_route,
    update_corpus_index,
)
from perturb_data_lab.materializers.models import (
    CountSourceSpec,
    DatasetJoinRecord,
    FeatureRegistryEntry,
    FeatureRegistryManifest,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
    SizeFactorEntry,
    SizeFactorManifest,
)


class TestCanonicalCellRecord:
    """Test the canonical sparse per-cell record contract."""

    def test_integer_sparse_check(self):
        record = CanonicalCellRecord(
            expressed_gene_indices=(0, 3, 7),
            expression_counts=(5, 2, 8),
            cell_id="cell_001",
            dataset_id="test_ds",
            dataset_release="v0",
            size_factor=1.0,
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )
        assert record.is_integer_sparse()

    def test_non_integer_fails_check(self):
        record = CanonicalCellRecord(
            expressed_gene_indices=(0, 3),
            expression_counts=(5.0, 2.1),  # non-integer
            cell_id="cell_001",
            dataset_id="test_ds",
            dataset_release="v0",
            size_factor=1.0,
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )
        assert not record.is_integer_sparse()

    def test_to_csr_matrix_parts(self):
        record = CanonicalCellRecord(
            expressed_gene_indices=(1, 4, 8),
            expression_counts=(3, 7, 1),
            cell_id="cell_001",
            dataset_id="test_ds",
            dataset_release="v0",
            size_factor=1.5,
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )
        data, indices, indptr = record.to_csr_matrix_parts(total_genes=10)
        np.testing.assert_array_equal(data, [3, 7, 1])
        np.testing.assert_array_equal(indices, [1, 4, 8])
        np.testing.assert_array_equal(indptr, [0, 3])


class TestBuildMaterializationRoute:
    """Test route factory."""

    def test_create_new_route(self):
        roots = OutputRoots(metadata_root="/tmp/meta", matrix_root="/tmp/matrix")
        route = build_materialization_route(
            "create_new",
            roots,
            "release-1",
            "ds-1",
            CountSourceSpec(selected=".X", integer_only=True),
        )
        assert isinstance(route, CreateNewRoute)
        assert route.route_name == "create_new"

    def test_append_routed_route(self):
        roots = OutputRoots(metadata_root="/tmp/meta", matrix_root="/tmp/matrix")
        route = build_materialization_route(
            "append_routed",
            roots,
            "release-3",
            "ds-3",
            CountSourceSpec(selected=".X", integer_only=True),
        )
        assert isinstance(route, AppendRoutedRoute)
        assert route.route_name == "append_routed"

    def test_unknown_route_raises(self):
        roots = OutputRoots(metadata_root="/tmp/meta", matrix_root="/tmp/matrix")
        with pytest.raises(ValueError, match="unknown route"):
            build_materialization_route(
                "unknown_route",
                roots,
                "release-x",
                "ds-x",
                CountSourceSpec(selected=".X", integer_only=True),
            )


class TestFeatureRegistryAppend:
    """Test append-only feature registry logic."""

    def test_new_registry_starts_at_zero(self):
        manifest = FeatureRegistryManifest(
            kind="feature-registry",
            contract_version="0.1.0",
            registry_id="test-reg",
            append_only=True,
            namespace="test",
            feature_id_field="gene_id",
            feature_label_field="gene_symbol",
            default_missing_value="NA",
            entries=(),
        )
        assert len(manifest.entries) == 0

    def test_registry_append_preserves_existing_ids(self):
        existing = FeatureRegistryManifest(
            kind="feature-registry",
            contract_version="0.1.0",
            registry_id="test-reg",
            append_only=True,
            namespace="test",
            feature_id_field="gene_id",
            feature_label_field="gene_symbol",
            default_missing_value="NA",
            entries=(
                FeatureRegistryEntry(
                    token_id=0,
                    feature_id="ENSG1",
                    feature_label="GENE1",
                    namespace="test",
                ),
                FeatureRegistryEntry(
                    token_id=1,
                    feature_id="ENSG2",
                    feature_label="GENE2",
                    namespace="test",
                ),
            ),
        )

        # Simulate appending new genes
        new_genes = ["ENSG3", "ENSG4"]
        existing_ids = {e.feature_id for e in existing.entries}
        start_token = max(e.token_id for e in existing.entries) + 1

        new_entries = []
        for token_id, gene_id in enumerate(new_genes, start=start_token):
            if gene_id not in existing_ids:
                new_entries.append(
                    FeatureRegistryEntry(
                        token_id=token_id,
                        feature_id=gene_id,
                        feature_label=gene_id,
                        namespace="test",
                    )
                )

        combined = FeatureRegistryManifest(
            kind="feature-registry",
            contract_version="0.1.0",
            registry_id="test-reg",
            append_only=True,
            namespace="test",
            feature_id_field="gene_id",
            feature_label_field="gene_symbol",
            default_missing_value="NA",
            entries=(*existing.entries, *new_entries),
        )

        assert len(combined.entries) == 4
        assert combined.entries[0].token_id == 0
        assert combined.entries[2].token_id == 2  # appended, not overwritten


class TestCorpusIndexUpdate:
    """Test corpus index append logic."""

    def test_new_corpus_index_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "corpus-index.yaml"
            record = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.1",
                join_mode="create_new",
                manifest_path="/tmp/meta/v0.1-manifest.yaml",
            )
            updated = update_corpus_index(idx_path, record)
            assert updated.corpus_id == "perturb-data-lab-v0"
            assert len(updated.datasets) == 1
            assert updated.datasets[0].dataset_id == "ds_001"

    def test_existing_corpus_index_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "corpus-index.yaml"

            # Create initial index
            record1 = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.1",
                join_mode="create_new",
                manifest_path="/tmp/v0.1.yaml",
            )
            update_corpus_index(idx_path, record1)

            # Append second dataset
            record2 = DatasetJoinRecord(
                dataset_id="ds_002",
                release_id="v0.2",
                join_mode="append_routed",
                manifest_path="/tmp/v0.2.yaml",
            )
            updated = update_corpus_index(idx_path, record2)

            assert len(updated.datasets) == 2
            assert updated.datasets[1].dataset_id == "ds_002"

    def test_duplicate_dataset_id_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "corpus-index.yaml"
            record = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.1",
                join_mode="create_new",
                manifest_path="/tmp/v0.1.yaml",
            )
            update_corpus_index(idx_path, record)

            duplicate = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.2",
                join_mode="append_routed",
                manifest_path="/tmp/v0.2.yaml",
            )
            with pytest.raises(ValueError, match="already exists in corpus index"):
                update_corpus_index(idx_path, duplicate)


class TestMaterializationManifest:
    """Test manifest YAML round-trip."""

    def test_manifest_round_trip(self):
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version="0.1.0",
            dataset_id="test_ds",
            release_id="v0.1",
            route="create_new",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            provenance=ProvenanceSpec(
                source_path="/data/test.h5ad",
                schema="/reviewed-schema.yaml",
            ),
            integer_verified=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            manifest.write_yaml(path)
            loaded = MaterializationManifest.from_yaml_file(path)
            assert loaded.dataset_id == "test_ds"
            assert loaded.route == "create_new"
            assert loaded.integer_verified is True
