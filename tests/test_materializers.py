"""Phase 3 materializer smoke tests using synthetic fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from perturb_data_lab.materializers import (
    CanonicalCellRecord,
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
            size_factor=1.5,
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )
        data, indices, indptr = record.to_csr_matrix_parts(total_genes=10)
        np.testing.assert_array_equal(data, [3, 7, 1])
        np.testing.assert_array_equal(indices, [1, 4, 8])
        np.testing.assert_array_equal(indptr, [0, 3])


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
                join_mode="create_new",
                manifest_path="/tmp/meta/materialization-manifest.yaml",
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
                join_mode="create_new",
                manifest_path="/tmp/ds_001.yaml",
                cell_count=500,
            )
            updated1 = update_corpus_index(idx_path, record1)
            assert updated1.datasets[0].global_start == 0
            assert updated1.datasets[0].global_end == 500

            # Append second dataset — global range is computed from existing total
            record2 = DatasetJoinRecord(
                dataset_id="ds_002",
                join_mode="append_routed",
                manifest_path="/tmp/ds_002.yaml",
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
                join_mode="create_new",
                manifest_path="/tmp/ds_001.yaml",
                cell_count=1000,
            )
            update_corpus_index(idx_path, record)

            duplicate = DatasetJoinRecord(
                dataset_id="ds_001",
                join_mode="append_routed",
                manifest_path="/tmp/ds_001_dup.yaml",
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
                    join_mode="create_new" if i == 0 else "append_routed",
                    manifest_path=f"/tmp/ds_{i:03d}.yaml",
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


class TestMaterializationManifest:
    """Test manifest YAML round-trip."""

    def test_manifest_round_trip(self):
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version="0.3.0",
            dataset_id="test_ds",
            route="create_new",
            backend="arrow-parquet",
            topology="federated",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            provenance=ProvenanceSpec(
                source_path="/data/test.h5ad",
                review_bundle="/reviewed-schema.yaml",
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

    def test_manifest_append_routed_round_trip(self):
        """Manifest with append_routed route round-trips correctly."""
        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version="0.3.0",
            dataset_id="ds_appended",
            route="append_routed",
            backend="lance",
            topology="aggregate",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            provenance=ProvenanceSpec(
                source_path="/data/appended.h5ad",
                review_bundle="/reviewed-schema.yaml",
            ),
            integer_verified=True,
            cell_count=500,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest-append.yaml"
            manifest.write_yaml(path)
            loaded = MaterializationManifest.from_yaml_file(path)
            assert loaded.dataset_id == "ds_appended"
            assert loaded.route == "append_routed"
            assert loaded.topology == "aggregate"
            assert loaded.cell_count == 500


class TestManifestToJoinRecordRoute:
    """Test manifest route propagation to DatasetJoinRecord and corpus ledger."""

    def test_create_new_route_propagates_to_join_record(self):
        """manifest_to_join_record preserves create_new route."""
        from perturb_data_lab.materializers import manifest_to_join_record

        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version="0.3.0",
            dataset_id="ds_create",
            route="create_new",
            backend="lance",
            topology="aggregate",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=OutputRoots(metadata_root="/meta/create", matrix_root="/matrix/create"),
            provenance=ProvenanceSpec(
                source_path="/data/create.h5ad",
                review_bundle="/reviewed-schema.yaml",
            ),
            raw_cell_meta_path="/meta/create/meta.parquet",
            provenance_spec_path="/meta/create/prov.parquet",
            cell_count=1000,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_root = Path(tmpdir) / "corpus"
            corpus_root.mkdir()
            record = manifest_to_join_record(manifest, corpus_root)
            assert record.join_mode == "create_new"
            assert record.dataset_id == "ds_create"

    def test_append_routed_route_propagates_to_join_record(self):
        """manifest_to_join_record preserves append_routed route."""
        from perturb_data_lab.materializers import manifest_to_join_record

        manifest = MaterializationManifest(
            kind="materialization-manifest",
            contract_version="0.3.0",
            dataset_id="ds_append",
            route="append_routed",
            backend="lance",
            topology="aggregate",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=OutputRoots(metadata_root="/meta/append", matrix_root="/matrix/append"),
            provenance=ProvenanceSpec(
                source_path="/data/append.h5ad",
                review_bundle="/reviewed-schema.yaml",
            ),
            raw_cell_meta_path="/meta/append/meta.parquet",
            provenance_spec_path="/meta/append/prov.parquet",
            cell_count=500,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_root = Path(tmpdir) / "corpus"
            corpus_root.mkdir()
            record = manifest_to_join_record(manifest, corpus_root)
            assert record.join_mode == "append_routed"
            assert record.dataset_id == "ds_append"

    def test_corpus_ledger_writes_append_routed_join_mode(self):
        """Corpus ledger YAML and Parquet record append_routed for appended datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "corpus-index.yaml"

            # Create corpus with first dataset
            record1 = DatasetJoinRecord(
                dataset_id="ds_first",
                join_mode="create_new",
                manifest_path="ds_first/manifest.yaml",
                cell_count=1000,
            )
            update_corpus_index(idx_path, record1, backend="lance", topology="aggregate")

            # Append second dataset
            record2 = DatasetJoinRecord(
                dataset_id="ds_second",
                join_mode="append_routed",
                manifest_path="ds_second/manifest.yaml",
                cell_count=500,
            )
            updated = update_corpus_index(idx_path, record2, backend="lance", topology="aggregate")

            # Verify YAML index
            assert len(updated.datasets) == 2
            assert updated.datasets[0].join_mode == "create_new"
            assert updated.datasets[1].join_mode == "append_routed"

            # Verify Parquet ledger
            import pyarrow.parquet as pq
            ledger_path = Path(tmpdir) / "corpus-ledger.parquet"
            assert ledger_path.exists()
            table = pq.read_table(str(ledger_path))
            join_modes = table.column("join_mode").to_pylist()
            assert join_modes == ["create_new", "append_routed"]


class TestStage2MaterializerConstructorApi:
    """Test Stage2Materializer constructor accepts mode, writer_state, _is_last_dataset."""

    def test_constructor_accepts_mode_create_default(self):
        """Default mode is 'create' and is stored as instance attribute."""
        try:
            from perturb_data_lab.materializers import Stage2Materializer
        except ImportError:
            pytest.skip("Stage2Materializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = Stage2Materializer(
            source_path="/fake/source.h5ad",
            review_bundle_path="/fake/bundle.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
        )
        assert mat.mode == "create"

    def test_constructor_accepts_mode_append(self):
        """mode='append' is accepted and stored."""
        try:
            from perturb_data_lab.materializers import Stage2Materializer
        except ImportError:
            pytest.skip("Stage2Materializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = Stage2Materializer(
            source_path="/fake/source.h5ad",
            review_bundle_path="/fake/bundle.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
            mode="append",
        )
        assert mat.mode == "append"

    def test_constructor_rejects_invalid_mode(self):
        """Invalid mode raises ValueError."""
        try:
            from perturb_data_lab.materializers import Stage2Materializer
        except ImportError:
            pytest.skip("Stage2Materializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        with pytest.raises(ValueError, match="mode must be 'create' or 'append'"):
            Stage2Materializer(
                source_path="/fake/source.h5ad",
                review_bundle_path="/fake/bundle.yaml",
                output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
                dataset_id="ds_test",
                mode="invalid",
            )

    def test_constructor_accepts_writer_state(self):
        """writer_state parameter is stored as instance attribute."""
        try:
            from perturb_data_lab.materializers import Stage2Materializer
        except ImportError:
            pytest.skip("Stage2Materializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        state = {"lance_path": "/corpus/cells.lance", "initialized": True}
        mat = Stage2Materializer(
            source_path="/fake/source.h5ad",
            review_bundle_path="/fake/bundle.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
            writer_state=state,
        )
        assert mat.writer_state == state

    def test_constructor_accepts_writer_state_none_default(self):
        """writer_state defaults to None."""
        try:
            from perturb_data_lab.materializers import Stage2Materializer
        except ImportError:
            pytest.skip("Stage2Materializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = Stage2Materializer(
            source_path="/fake/source.h5ad",
            review_bundle_path="/fake/bundle.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
        )
        assert mat.writer_state is None

    def test_constructor_accepts_is_last_dataset(self):
        """_is_last_dataset parameter is stored."""
        try:
            from perturb_data_lab.materializers import Stage2Materializer
        except ImportError:
            pytest.skip("Stage2Materializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = Stage2Materializer(
            source_path="/fake/source.h5ad",
            review_bundle_path="/fake/bundle.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
            _is_last_dataset=True,
        )
        assert mat._is_last_dataset is True

    def test_constructor_is_last_dataset_default_false(self):
        """_is_last_dataset defaults to False."""
        try:
            from perturb_data_lab.materializers import Stage2Materializer
        except ImportError:
            pytest.skip("Stage2Materializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = Stage2Materializer(
            source_path="/fake/source.h5ad",
            review_bundle_path="/fake/bundle.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
        )
        assert mat._is_last_dataset is False

    def test_dataset_index_and_global_row_start_defaults(self):
        """dataset_index and global_row_start have sensible defaults."""
        try:
            from perturb_data_lab.materializers import Stage2Materializer
        except ImportError:
            pytest.skip("Stage2Materializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = Stage2Materializer(
            source_path="/fake/source.h5ad",
            review_bundle_path="/fake/bundle.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
        )
        assert mat.dataset_index == 0
        assert mat.global_row_start == 0

    def test_constructor_accepts_dataset_index_and_global_row_start(self):
        """dataset_index and global_row_start can be set via constructor."""
        try:
            from perturb_data_lab.materializers import Stage2Materializer
        except ImportError:
            pytest.skip("Stage2Materializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = Stage2Materializer(
            source_path="/fake/source.h5ad",
            review_bundle_path="/fake/bundle.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
            dataset_index=3,
            global_row_start=1500,
        )
        assert mat.dataset_index == 3
        assert mat.global_row_start == 1500
