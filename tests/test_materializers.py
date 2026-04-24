"""Phase 3 materializer smoke tests using synthetic fixtures."""

from __future__ import annotations

import json
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

from perturb_data_lab.materializers.backends.lancedb_aggregated import mark_lance_append_committed


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
                cell_count=1000,
            )
            updated = update_corpus_index(idx_path, record)
            assert updated.corpus_id == "perturb-data-lab-v0"
            assert len(updated.datasets) == 1
            assert updated.datasets[0].dataset_id == "ds_001"
            # For new corpus, global range = [0, cell_count)
            assert updated.datasets[0].cell_count == 1000
            assert updated.datasets[0].global_start == 0
            assert updated.datasets[0].global_end == 1000

    def test_existing_corpus_index_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "corpus-index.yaml"

            # Create initial index with cell_count
            record1 = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.1",
                join_mode="create_new",
                manifest_path="/tmp/v0.1.yaml",
                cell_count=500,
            )
            updated1 = update_corpus_index(idx_path, record1)
            assert updated1.datasets[0].global_start == 0
            assert updated1.datasets[0].global_end == 500

            # Append second dataset — global range is computed from existing total
            record2 = DatasetJoinRecord(
                dataset_id="ds_002",
                release_id="v0.2",
                join_mode="append_routed",
                manifest_path="/tmp/v0.2.yaml",
                cell_count=300,
            )
            updated2 = update_corpus_index(idx_path, record2)

            assert len(updated2.datasets) == 2
            assert updated2.datasets[1].dataset_id == "ds_002"
            # ds_002 starts where ds_001 ended
            assert updated2.datasets[1].global_start == 500
            assert updated2.datasets[1].global_end == 800  # 500 + 300

    def test_duplicate_dataset_id_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "corpus-index.yaml"
            record = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.1",
                join_mode="create_new",
                manifest_path="/tmp/v0.1.yaml",
                cell_count=1000,
            )
            update_corpus_index(idx_path, record)

            duplicate = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.2",
                join_mode="append_routed",
                manifest_path="/tmp/v0.2.yaml",
                cell_count=500,
            )
            with pytest.raises(ValueError, match="already exists in corpus index"):
                update_corpus_index(idx_path, duplicate)

    def test_global_range_contiguous_no_overlap(self):
        """Verify that appending multiple datasets produces contiguous non-overlapping ranges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "corpus-index.yaml"
            cell_counts = [1000, 500, 750, 200]
            for i, cc in enumerate(cell_counts):
                record = DatasetJoinRecord(
                    dataset_id=f"ds_{i:03d}",
                    release_id=f"v{i}",
                    join_mode="create_new" if i == 0 else "append_routed",
                    manifest_path=f"/tmp/v{i}.yaml",
                    cell_count=cc,
                )
                update_corpus_index(idx_path, record)

            # Reload and verify
            from perturb_data_lab.materializers.models import CorpusIndexDocument

            corpus = CorpusIndexDocument.from_yaml_file(idx_path)
            running_end = 0
            for ds in corpus.datasets:
                assert ds.global_start == running_end
                assert ds.global_end == running_end + ds.cell_count
                assert ds.global_end - ds.global_start == ds.cell_count
                running_end += ds.cell_count

    def test_lancedb_append_sidecar_is_marked_committed_after_index_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_root = Path(tmpdir)
            matrix_root = corpus_root / "matrix"
            matrix_root.mkdir()
            sidecar_path = matrix_root / "aggregated-corpus-append-log.json"
            sidecar_path.write_text(
                """
{
  "backend": "lancedb-aggregated",
  "table_name": "aggregated-corpus",
  "dataset_uri": "dummy",
  "db_root": "dummy-root",
  "entries": [
    {
      "dataset_index": 0,
      "dataset_id": "ds_001",
      "release_id": "v0.1",
      "global_row_start": 0,
      "global_row_end": 5,
      "cell_count": 5,
      "lance_version": 1,
      "status": "pending"
    }
  ]
}
                """.strip(),
                encoding="utf-8",
            )

            idx_path = corpus_root / "corpus-index.yaml"
            record = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.1",
                join_mode="create_new",
                manifest_path="/tmp/meta/v0.1-manifest.yaml",
                cell_count=5,
            )
            update_corpus_index(idx_path, record, backend="lancedb-aggregated")

            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            assert payload["entries"][0]["status"] == "committed"


class TestLanceSidecarHelpers:
    def test_mark_lance_append_committed_returns_none_when_sidecar_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_index_path = Path(tmpdir) / "corpus-index.yaml"
            corpus_index_path.parent.mkdir(parents=True, exist_ok=True)
            record = DatasetJoinRecord(
                dataset_id="ds_001",
                release_id="v0.1",
                join_mode="create_new",
                manifest_path="m.yaml",
                dataset_index=0,
                cell_count=1,
                global_start=0,
                global_end=1,
            )
            assert mark_lance_append_committed(corpus_index_path, record) is None

    def test_sidecar_status_stays_pending_until_corpus_index_commit(self):
        pytest.importorskip("lance")
        pytest.importorskip("lancedb")

        from perturb_data_lab.materializers.backends.lancedb_aggregated import (
            write_lancedb_aggregated,
        )

        class DummyObs:
            def __init__(self):
                self.index = ["c0", "c1"]

        class DummyAnnData:
            def __init__(self):
                self.n_obs = 2
                self.obs = DummyObs()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matrix_root = root / "dataset_matrix"
            corpus_root = root / "corpus"
            corpus_index_path = corpus_root / "corpus-index.yaml"
            corpus_root.mkdir()

            adata = DummyAnnData()
            count_matrix = np.array([[0, 4, 0], [1, 0, 2]], dtype=np.int32)
            size_factors = np.array([1.0, 1.5], dtype=np.float64)
            write_lancedb_aggregated(
                adata=adata,
                count_matrix=count_matrix,
                size_factors=size_factors,
                release_id="rel0",
                matrix_root=matrix_root,
                canonical_perturbation=({}, {}),
                canonical_context=({}, {}),
                raw_fields=({}, {}),
                dataset_id="ds0",
                corpus_index_path=corpus_index_path,
            )

            sidecar_path = corpus_root / "matrix" / "aggregated-corpus-append-log.json"
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            assert payload["entries"][0]["status"] == "pending"
            assert payload["entries"][0]["dataset_index"] == 0
            assert payload["entries"][0]["global_row_start"] == 0
            assert payload["entries"][0]["global_row_end"] == 2

    def test_true_lance_append_preserves_ranges_and_dataset_index(self):
        lance = pytest.importorskip("lance")
        pytest.importorskip("lancedb")

        from perturb_data_lab.materializers.backends.lancedb_aggregated import (
            write_lancedb_aggregated,
        )

        class DummyObs:
            def __init__(self, ids):
                self.index = ids

        class DummyAnnData:
            def __init__(self, ids):
                self.n_obs = len(ids)
                self.obs = DummyObs(ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            corpus_root = root / "corpus"
            corpus_root.mkdir()
            corpus_index_path = corpus_root / "corpus-index.yaml"

            first_matrix_root = root / "dataset0" / "matrix"
            second_matrix_root = root / "dataset1" / "matrix"

            write_lancedb_aggregated(
                adata=DummyAnnData(["a0", "a1"]),
                count_matrix=np.array([[3, 0, 1], [0, 5, 0]], dtype=np.int32),
                size_factors=np.array([1.0, 1.2], dtype=np.float64),
                release_id="rel0",
                matrix_root=first_matrix_root,
                canonical_perturbation=({}, {}),
                canonical_context=({}, {}),
                raw_fields=({}, {}),
                dataset_id="ds0",
                corpus_index_path=corpus_index_path,
            )
            update_corpus_index(
                corpus_index_path,
                DatasetJoinRecord(
                    dataset_id="ds0",
                    release_id="rel0",
                    join_mode="create_new",
                    manifest_path="meta0/materialization-manifest.yaml",
                    cell_count=2,
                ),
                backend="lancedb-aggregated",
            )

            write_lancedb_aggregated(
                adata=DummyAnnData(["b0", "b1"]),
                count_matrix=np.array([[0, 2, 0], [4, 0, 6]], dtype=np.int32),
                size_factors=np.array([0.9, 1.1], dtype=np.float64),
                release_id="rel1",
                matrix_root=second_matrix_root,
                canonical_perturbation=({}, {}),
                canonical_context=({}, {}),
                raw_fields=({}, {}),
                dataset_id="ds1",
                corpus_index_path=corpus_index_path,
            )

            shared_path = corpus_root / "matrix" / "aggregated-corpus.lance"
            rows = lance.dataset(shared_path).take([0, 1, 2, 3]).to_pylist()
            assert [row["global_row_index"] for row in rows] == [0, 1, 2, 3]
            assert [row["dataset_index"] for row in rows] == [0, 0, 1, 1]
            assert [row["local_row_index"] for row in rows] == [0, 1, 0, 1]

            sidecar_path = corpus_root / "matrix" / "aggregated-corpus-append-log.json"
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            assert payload["entries"][0]["status"] == "committed"
            assert payload["entries"][1]["status"] == "pending"
            assert payload["entries"][1]["dataset_index"] == 1
            assert payload["entries"][1]["global_row_start"] == 2
            assert payload["entries"][1]["global_row_end"] == 4


class TestMaterializationManifest:
    """Test manifest YAML round-trip."""

    def test_manifest_round_trip(self):
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version="0.2.0",
            dataset_id="test_ds",
            release_id="v0.1",
            route="create_new",
            backend="arrow-hf",
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
