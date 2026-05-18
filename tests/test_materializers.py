"""Materializer smoke tests using synthetic fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from perturb_data_lab.materializers import update_corpus_index
from perturb_data_lab.materializers.models import (
    CountSourceSpec,
    DatasetJoinRecord,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
)


class TestCorpusIndexUpdate:
    """Test corpus index append logic."""

    def test_new_corpus_index_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "corpus-index.yaml"
            record = DatasetJoinRecord(
                dataset_id="ds_001",
                join_mode="create_new",
                manifest_path="meta/materialization-manifest.yaml",
                cell_count=1000,
            )
            updated = update_corpus_index(
                idx_path,
                record,
                corpus_id="test-v0",
                backend="lance",
                topology="aggregate",
            )
            assert updated.corpus_id == "test-v0"
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
                manifest_path="ds_001.yaml",
                cell_count=500,
            )
            updated1 = update_corpus_index(
                idx_path,
                record1,
                corpus_id="test-v0",
                backend="lance",
                topology="aggregate",
            )
            assert updated1.datasets[0].global_start == 0
            assert updated1.datasets[0].global_end == 500

            # Append second dataset — global range is computed from existing total
            record2 = DatasetJoinRecord(
                dataset_id="ds_002",
                join_mode="append_routed",
                manifest_path="ds_002.yaml",
                cell_count=300,
            )
            updated2 = update_corpus_index(idx_path, record2, backend="lance", topology="aggregate")

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
                manifest_path="ds_001.yaml",
                cell_count=1000,
            )
            update_corpus_index(
                idx_path,
                record,
                corpus_id="test-v0",
                backend="lance",
                topology="aggregate",
            )

            duplicate = DatasetJoinRecord(
                dataset_id="ds_001",
                join_mode="append_routed",
                manifest_path="ds_001_dup.yaml",
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
                    manifest_path=f"ds_{i:03d}.yaml",
                    cell_count=cc,
                )
                update_corpus_index(
                    idx_path,
                    record,
                    corpus_id="test-v0",
                    backend="lance",
                    topology="aggregate",
                )

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
            backend="zarr",
            topology="federated",
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            outputs=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            provenance=ProvenanceSpec(
                source_path="/data/test.h5ad",
                inspection_summary_path="/dataset-summary.yaml",
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
                inspection_summary_path="/dataset-summary.yaml",
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

    def test_manifest_rejects_legacy_route_alias(self, tmp_path: Path):
        payload = {
            "kind": "materialization-manifest",
            "contract_version": "0.3.0",
            "dataset_id": "ds_appended",
            "route": "append",
            "backend": "lance",
            "topology": "aggregate",
            "count_source": {"selected": ".X", "integer_only": True},
            "outputs": {"metadata_root": "meta/ds", "matrix_root": "matrix"},
            "provenance": {"source_path": "/data/ds.h5ad", "inspection_summary_path": "meta/ds/dataset-summary.yaml"},
        }
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="invalid route"):
            MaterializationManifest.from_yaml_file(path)

    def test_manifest_rejects_legacy_backend_alias(self, tmp_path: Path):
        payload = {
            "kind": "materialization-manifest",
            "contract_version": "0.3.0",
            "dataset_id": "ds_appended",
            "route": "append_routed",
            "backend": "lancedb-aggregated",
            "topology": "aggregate",
            "count_source": {"selected": ".X", "integer_only": True},
            "outputs": {"metadata_root": "meta/ds", "matrix_root": "matrix"},
            "provenance": {"source_path": "/data/ds.h5ad", "inspection_summary_path": "meta/ds/dataset-summary.yaml"},
        }
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="invalid backend"):
            MaterializationManifest.from_yaml_file(path)


class TestCorpusIndexRoutes:
    """Test route propagation to corpus-index records."""

    def test_corpus_index_writes_append_routed_join_mode(self):
        """Corpus index records append_routed for appended datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "corpus-index.yaml"

            # Create corpus with first dataset
            record1 = DatasetJoinRecord(
                dataset_id="ds_first",
                join_mode="create_new",
                manifest_path="ds_first/manifest.yaml",
                cell_count=1000,
            )
            update_corpus_index(
                idx_path,
                record1,
                corpus_id="test-v0",
                backend="lance",
                topology="aggregate",
            )

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

            assert [d.join_mode for d in updated.datasets] == ["create_new", "append_routed"]


class TestDatasetMaterializerConstructorApi:
    """Test DatasetMaterializer constructor options."""

    def test_constructor_accepts_mode_create_default(self):
        """Default mode is 'create' and is stored as instance attribute."""
        try:
            from perturb_data_lab.materializers import DatasetMaterializer
        except ImportError:
            pytest.skip("DatasetMaterializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = DatasetMaterializer(
            source_path="/fake/source.h5ad",
            inspection_summary_path="/fake/dataset-summary.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
        )
        assert mat.mode == "create"

    def test_constructor_accepts_mode_append(self):
        """mode='append' is accepted and stored."""
        try:
            from perturb_data_lab.materializers import DatasetMaterializer
        except ImportError:
            pytest.skip("DatasetMaterializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = DatasetMaterializer(
            source_path="/fake/source.h5ad",
            inspection_summary_path="/fake/dataset-summary.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
            mode="append",
        )
        assert mat.mode == "append"

    def test_constructor_rejects_invalid_mode(self):
        """Invalid mode raises ValueError."""
        try:
            from perturb_data_lab.materializers import DatasetMaterializer
        except ImportError:
            pytest.skip("DatasetMaterializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        with pytest.raises(ValueError, match="mode must be 'create' or 'append'"):
            DatasetMaterializer(
                source_path="/fake/source.h5ad",
                inspection_summary_path="/fake/dataset-summary.yaml",
                output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
                dataset_id="ds_test",
                mode="invalid",
            )

    def test_constructor_accepts_writer_state(self):
        """writer_state parameter is stored as instance attribute."""
        try:
            from perturb_data_lab.materializers import DatasetMaterializer
        except ImportError:
            pytest.skip("DatasetMaterializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        state = {"lance_path": "/corpus/cells.lance", "initialized": True}
        mat = DatasetMaterializer(
            source_path="/fake/source.h5ad",
            inspection_summary_path="/fake/dataset-summary.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
            writer_state=state,
        )
        assert mat.writer_state == state

    def test_constructor_accepts_writer_state_none_default(self):
        """writer_state defaults to None."""
        try:
            from perturb_data_lab.materializers import DatasetMaterializer
        except ImportError:
            pytest.skip("DatasetMaterializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = DatasetMaterializer(
            source_path="/fake/source.h5ad",
            inspection_summary_path="/fake/dataset-summary.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
        )
        assert mat.writer_state is None

    def test_constructor_accepts_is_last_dataset(self):
        """_is_last_dataset parameter is stored."""
        try:
            from perturb_data_lab.materializers import DatasetMaterializer
        except ImportError:
            pytest.skip("DatasetMaterializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = DatasetMaterializer(
            source_path="/fake/source.h5ad",
            inspection_summary_path="/fake/dataset-summary.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
            _is_last_dataset=True,
        )
        assert mat._is_last_dataset is True

    def test_constructor_is_last_dataset_default_false(self):
        """_is_last_dataset defaults to False."""
        try:
            from perturb_data_lab.materializers import DatasetMaterializer
        except ImportError:
            pytest.skip("DatasetMaterializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = DatasetMaterializer(
            source_path="/fake/source.h5ad",
            inspection_summary_path="/fake/dataset-summary.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
        )
        assert mat._is_last_dataset is False

    def test_dataset_index_and_global_row_start_defaults(self):
        """dataset_index and global_row_start have sensible defaults."""
        try:
            from perturb_data_lab.materializers import DatasetMaterializer
        except ImportError:
            pytest.skip("DatasetMaterializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = DatasetMaterializer(
            source_path="/fake/source.h5ad",
            inspection_summary_path="/fake/dataset-summary.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
        )
        assert mat.dataset_index == 0
        assert mat.global_row_start == 0

    def test_constructor_accepts_dataset_index_and_global_row_start(self):
        """dataset_index and global_row_start can be set via constructor."""
        try:
            from perturb_data_lab.materializers import DatasetMaterializer
        except ImportError:
            pytest.skip("DatasetMaterializer import unavailable (anndata not installed)")
        from perturb_data_lab.materializers.models import OutputRoots

        mat = DatasetMaterializer(
            source_path="/fake/source.h5ad",
            inspection_summary_path="/fake/dataset-summary.yaml",
            output_roots=OutputRoots(metadata_root="/meta", matrix_root="/matrix"),
            dataset_id="ds_test",
            dataset_index=3,
            global_row_start=1500,
        )
        assert mat.dataset_index == 3
        assert mat.global_row_start == 1500


class TestDatasetMaterializerInspectionGate:
    """Test DatasetMaterializer respects non-pass inspection summaries."""

    def test_materialize_rejects_needs_review_before_loading_h5ad(
        self,
        tmp_path: Path,
    ):
        try:
            from perturb_data_lab.materializers import DatasetMaterializer
        except ImportError:
            pytest.skip("DatasetMaterializer import unavailable (anndata not installed)")

        inspection_summary = tmp_path / "dataset-summary.yaml"
        inspection_summary.write_text(
            yaml.safe_dump(
                {
                    "kind": "dataset-summary",
                    "contract_version": "0.3.0",
                    "dataset": {
                        "dataset_id": "fp32_ds",
                        "source_release": "fp32_ds",
                        "source_path": "/missing/source.h5ad",
                        "obs_rows": 10,
                        "var_rows": 5,
                        "obs_index_name": "index",
                        "var_index_name": "index",
                    },
                    "structure": {
                        "has_raw": False,
                        "raw_var_rows": 0,
                        "layers": ["X_binned"],
                    },
                    "obs_fields": [],
                    "var_fields": [],
                    "count_source_candidates": [
                        {
                            "candidate": ".layers[X_binned]",
                            "rank": 1,
                            "status": "fail",
                            "storage": "sparse",
                            "dtype": "float32",
                            "shape": [10, 5],
                            "sampled_rows": 3,
                            "sampled_nonzero_values": 6,
                            "sampled_density": 0.1,
                            "fraction_noninteger_nonzero": 1.0,
                            "max_abs_integer_deviation": 0.49,
                            "nonnegative": True,
                            "inferred_transform": "binned",
                            "recovery_policy": "disallowed",
                            "notes": ["candidate name suggests binned data"],
                        }
                    ],
                    "count_source_decision": {
                        "selected_candidate": ".layers[X_binned]",
                        "status": "needs-review",
                        "confidence": "medium",
                        "recovery_policy": "disallowed",
                        "rationale": "explicit approval required",
                        "uses_recovery": False,
                        "pass_mode": None,
                    },
                    "materialization_readiness": "needs-review",
                    "inspector_notes": [],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        materializer = DatasetMaterializer(
            source_path="/missing/source.h5ad",
            inspection_summary_path=str(inspection_summary),
            output_roots=OutputRoots(
                metadata_root=str(tmp_path / "meta"),
                matrix_root=str(tmp_path / "matrix"),
            ),
            dataset_id="fp32_ds",
        )

        with pytest.raises(ValueError, match="materialization_readiness is 'needs-review'"):
            materializer.materialize()
