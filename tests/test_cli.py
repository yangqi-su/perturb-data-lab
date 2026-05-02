"""Phase 2 tests for unified materialize CLI and corpus-gc command."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest
import yaml

from perturb_data_lab.cli import (
    _DatasetInput,
    _materialize_dataset,
    _parse_input_list_csv,
    _resolve_effective_topology,
    _scan_input_dir,
    _resolve_inputs,
    build_parser,
)


# ---------------------------------------------------------------------------
# Parser construction tests
# ---------------------------------------------------------------------------


class TestParserConstruction:
    """Verify the CLI parser accepts all expected subcommands."""

    def test_parser_builds(self):
        parser = build_parser()
        assert parser is not None
        assert isinstance(parser, argparse.ArgumentParser)

    def test_inspect_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(
            ["inspect", "--config", "c.yaml", "--workers", "2"]
        )
        assert ns.command == "inspect"
        assert ns.config == "c.yaml"
        assert ns.workers == 2

    def test_materialize_single_dataset(self):
        parser = build_parser()
        ns = parser.parse_args([
            "materialize",
            "--mode", "create",
            "--source", "/data/test.h5ad",
            "--dataset-id", "ds1",
            "--review-bundle", "/data/summary.yaml",
            "--output-corpus", "/corpus",
            "--backend", "arrow-parquet",
        ])
        assert ns.command == "materialize"
        assert ns.mode == "create"
        assert ns.source == "/data/test.h5ad"
        assert ns.dataset_id == "ds1"
        assert ns.review_bundle == "/data/summary.yaml"
        assert ns.output_corpus == "/corpus"
        assert ns.backend == "arrow-parquet"
        assert ns.topology == "federated"
        assert ns.dry_run is False

    def test_materialize_with_input_list(self):
        parser = build_parser()
        ns = parser.parse_args([
            "materialize",
            "--mode", "append",
            "--input-list", "/data/datasets.csv",
            "--output-corpus", "/corpus",
            "--backend", "lance",
            "--topology", "aggregate",
            "--corpus-id", "my-corpus",
        ])
        assert ns.command == "materialize"
        assert ns.mode == "append"
        assert ns.input_list == "/data/datasets.csv"
        assert ns.output_corpus == "/corpus"
        assert ns.backend == "lance"
        assert ns.topology == "aggregate"
        assert ns.corpus_id == "my-corpus"

    def test_materialize_with_input_dir(self):
        parser = build_parser()
        ns = parser.parse_args([
            "materialize",
            "--mode", "create",
            "--input-dir", "/data/h5ad/",
            "--review-bundle-dir", "/data/reviews/",
            "--output-corpus", "/corpus",
            "--backend", "arrow-ipc",
            "--dry-run",
        ])
        assert ns.command == "materialize"
        assert ns.mode == "create"
        assert ns.input_dir == "/data/h5ad/"
        assert ns.review_bundle_dir == "/data/reviews/"
        assert ns.dry_run is True

    def test_materialize_parses_ambiguous_input(self):
        """--source + --input-list are both accepted by parser
        (validation happens in _resolve_inputs)."""
        parser = build_parser()
        ns = parser.parse_args([
            "materialize",
            "--mode", "create",
            "--source", "/data/test.h5ad",
            "--input-list", "/data/list.csv",
            "--dataset-id", "ds1",
            "--review-bundle", "/data/summary.yaml",
            "--output-corpus", "/corpus",
            "--backend", "arrow-parquet",
        ])
        assert ns.source == "/data/test.h5ad"
        assert ns.input_list == "/data/list.csv"

    def test_materialize_source_accepts_dataset_id(self):
        """--source with --dataset-id parses fine."""
        parser = build_parser()
        ns = parser.parse_args([
            "materialize",
            "--mode", "create",
            "--source", "/data/test.h5ad",
            "--dataset-id", "ds1",
            "--review-bundle", "/data/summary.yaml",
            "--output-corpus", "/corpus",
            "--backend", "arrow-parquet",
        ])
        assert ns.dataset_id == "ds1"

    def test_corpus_validate_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(
            ["corpus-validate", "/c.yaml", "--backend", "webdataset"]
        )
        assert ns.command == "corpus-validate"
        assert ns.corpus_index == "/c.yaml"
        assert ns.backend == "webdataset"

    def test_corpus_gc_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(
            ["corpus-gc", "/corpus", "--dry-run"]
        )
        assert ns.command == "corpus-gc"
        assert ns.corpus_root == "/corpus"
        assert ns.dry_run is True

    def test_corpus_gc_default_dry_run_false(self):
        parser = build_parser()
        ns = parser.parse_args(["corpus-gc", "/corpus"])
        assert ns.dry_run is False

    def test_unknown_subcommand_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["unknown-cmd"])

    def test_removed_subcommands_not_present(self):
        """Verify stage2-materialize, corpus-create, corpus-append are gone."""
        parser = build_parser()
        choices = parser._subparsers._group_actions[0].choices
        assert "stage2-materialize" not in choices
        assert "corpus-create" not in choices
        assert "corpus-append" not in choices


class TestEffectiveTopologyResolution:
    """Ensure backend/topology routing resolves aggregate correctly."""

    def test_lance_forces_aggregate_even_if_requested_federated(self):
        assert _resolve_effective_topology("lance", "federated") == "aggregate"

    def test_non_lance_respects_requested_topology(self):
        assert _resolve_effective_topology("arrow-parquet", "federated") == "federated"
        assert _resolve_effective_topology("arrow-parquet", "aggregate") == "aggregate"


class TestMaterializeDatasetRouting:
    """Verify _materialize_dataset routes aggregate matrix_root correctly."""

    def test_lance_uses_corpus_wide_matrix_root(self, tmp_path: Path, monkeypatch):
        import perturb_data_lab.materializers as materializers_mod

        captured: dict[str, object] = {}

        class _DummyMaterializer:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.writer_state = {"dummy": "state"}

            def materialize(self):
                class _Manifest:
                    cell_count = 7
                    feature_count = 3
                    route = "append_routed"
                    corpus_registration = None

                return _Manifest()

        monkeypatch.setattr(materializers_mod, "Stage2Materializer", _DummyMaterializer)

        ds = _DatasetInput(
            source=str(tmp_path / "dummy.h5ad"),
            dataset_id="dummy_00",
            review_bundle=str(tmp_path / "dummy-summary.yaml"),
        )
        args = argparse.Namespace(
            backend="lance",
            topology="federated",  # parser default
            rerun_stage1=False,
            n_hvg=2000,
            corpus_id="test-corpus",
        )
        corpus_root = tmp_path / "corpus"

        next_global, next_writer_state = _materialize_dataset(
            ds,
            args,
            corpus_root=corpus_root,
            effective_topology=_resolve_effective_topology(args.backend, args.topology),
            mode="append",
            dataset_index=1,
            global_row_start=10,
            writer_state={"prev": "state"},
            is_last_dataset=False,
            total_datasets=2,
            dry_run=False,
        )

        output_roots = captured["output_roots"]
        assert str(output_roots.matrix_root) == str(corpus_root / "matrix")
        assert str(output_roots.metadata_root) == str(corpus_root / "meta" / "dummy_00")
        assert captured["topology"] == "aggregate"
        assert next_global == 17
        assert next_writer_state == {"dummy": "state"}


# ---------------------------------------------------------------------------
# Input parsing tests
# ---------------------------------------------------------------------------


class TestParseInputListCsv:
    """Test _parse_input_list_csv function."""

    def test_parses_basic_csv(self, tmp_path: Path):
        csv_path = tmp_path / "datasets.csv"
        csv_path.write_text(
            "source,dataset_id,review_bundle\n"
            "/data/ds1.h5ad,ds1,/reviews/ds1-summary.yaml\n"
            "/data/ds2.h5ad,ds2,/reviews/ds2-summary.yaml\n"
        )
        inputs = _parse_input_list_csv(csv_path)
        assert len(inputs) == 2
        assert inputs[0].dataset_id == "ds1"
        assert inputs[1].dataset_id == "ds2"

    def test_raises_on_missing_columns(self, tmp_path: Path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("source,dataset_id\n/data/ds1.h5ad,ds1\n")
        with pytest.raises(ValueError, match="missing required columns"):
            _parse_input_list_csv(csv_path)

    def test_raises_on_empty_csv(self, tmp_path: Path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("source,dataset_id,review_bundle\n")
        with pytest.raises(ValueError, match="no dataset rows"):
            _parse_input_list_csv(csv_path)

    def test_strips_whitespace(self, tmp_path: Path):
        csv_path = tmp_path / "datasets.csv"
        csv_path.write_text(
            "source,dataset_id,review_bundle\n"
            " /data/ds1.h5ad , ds1 , /reviews/ds1-summary.yaml \n"
        )
        inputs = _parse_input_list_csv(csv_path)
        assert inputs[0].source == "/data/ds1.h5ad"
        assert inputs[0].dataset_id == "ds1"


class TestScanInputDir:
    """Test _scan_input_dir function."""

    def test_scans_h5ad_files(self, tmp_path: Path):
        # Create h5ad files and review bundles
        (tmp_path / "ds_a.h5ad").touch()
        (tmp_path / "ds_b.h5ad").touch()
        (tmp_path / "ds_c.txt").touch()  # should be ignored
        # Review bundles
        (tmp_path / "ds_a-summary.yaml").write_text("kind: review")
        (tmp_path / "ds_b-summary.yaml").write_text("kind: review")

        inputs = _scan_input_dir(tmp_path)
        assert len(inputs) == 2
        assert {d.dataset_id for d in inputs} == {"ds_a", "ds_b"}

    def test_raises_when_review_bundle_missing(self, tmp_path: Path):
        (tmp_path / "ds_a.h5ad").touch()
        with pytest.raises(FileNotFoundError, match="Review bundle not found"):
            _scan_input_dir(tmp_path)

    def test_raises_on_empty_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No .h5ad files"):
            _scan_input_dir(tmp_path)

    def test_uses_alternate_review_bundle_dir(self, tmp_path: Path):
        h5ad_dir = tmp_path / "h5ad"
        review_dir = tmp_path / "reviews"
        h5ad_dir.mkdir()
        review_dir.mkdir()
        (h5ad_dir / "ds_a.h5ad").touch()
        (review_dir / "ds_a-summary.yaml").write_text("kind: review")

        inputs = _scan_input_dir(h5ad_dir, review_bundle_dir=review_dir)
        assert len(inputs) == 1
        assert inputs[0].dataset_id == "ds_a"
        assert inputs[0].review_bundle == str(review_dir / "ds_a-summary.yaml")


# ---------------------------------------------------------------------------
# Resolve inputs tests (validation logic)
# ---------------------------------------------------------------------------


class TestResolveInputs:
    """Test _resolve_inputs validation logic."""

    def test_rejects_no_input_mode(self):
        args = argparse.Namespace(
            source=None, input_list=None, input_dir=None,
            rerun_stage1=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _resolve_inputs(args)
        assert exc_info.value.code == 1

    def test_rejects_missing_source_file(self, tmp_path: Path):
        args = argparse.Namespace(
            source=str(tmp_path / "nonexistent.h5ad"),
            input_list=None,
            input_dir=None,
            dataset_id="ds1",
            review_bundle=str(tmp_path / "summary.yaml"),
            rerun_stage1=False,
        )
        # Need the review_bundle to exist or it will fail on that first
        (tmp_path / "summary.yaml").write_text("kind: review")
        with pytest.raises(SystemExit) as exc_info:
            _resolve_inputs(args)
        assert exc_info.value.code == 1

    def test_rejects_missing_review_bundle(self, tmp_path: Path):
        h5ad_path = tmp_path / "test.h5ad"
        h5ad_path.touch()
        args = argparse.Namespace(
            source=str(h5ad_path),
            input_list=None,
            input_dir=None,
            dataset_id="ds1",
            review_bundle=str(tmp_path / "nonexistent-summary.yaml"),
            rerun_stage1=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _resolve_inputs(args)
        assert exc_info.value.code == 1

    def test_accepts_valid_single_input(self, tmp_path: Path):
        h5ad_path = tmp_path / "test.h5ad"
        h5ad_path.touch()
        bundle_path = tmp_path / "summary.yaml"
        bundle_path.write_text("kind: review")
        args = argparse.Namespace(
            source=str(h5ad_path),
            input_list=None,
            input_dir=None,
            dataset_id="ds1",
            review_bundle=str(bundle_path),
            rerun_stage1=False,
        )
        inputs = _resolve_inputs(args)
        assert len(inputs) == 1
        assert inputs[0].dataset_id == "ds1"
        assert inputs[0].source == str(h5ad_path)

    def test_detects_duplicate_dataset_ids(self, tmp_path: Path):
        """Duplicate detection within batch happens in _resolve_inputs."""
        csv_path = tmp_path / "datasets.csv"
        csv_path.write_text(
            "source,dataset_id,review_bundle\n"
            "/data/ds1.h5ad,dup_id,/reviews/ds1-summary.yaml\n"
            "/data/ds2.h5ad,dup_id,/reviews/ds2-summary.yaml\n"
        )
        args = argparse.Namespace(
            source=None,
            input_list=str(csv_path),
            input_dir=None,
            rerun_stage1=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _resolve_inputs(args)
        assert exc_info.value.code == 1

    def test_skips_review_check_with_rerun_stage1(self, tmp_path: Path):
        """With rerun_stage1=True, missing review bundle is tolerated."""
        h5ad_path = tmp_path / "test.h5ad"
        h5ad_path.touch()
        args = argparse.Namespace(
            source=str(h5ad_path),
            input_list=None,
            input_dir=None,
            dataset_id="ds1",
            review_bundle=str(tmp_path / "nonexistent-summary.yaml"),
            rerun_stage1=True,
        )
        inputs = _resolve_inputs(args)
        assert len(inputs) == 1
        assert inputs[0].dataset_id == "ds1"


# ---------------------------------------------------------------------------
# Corpus-validate command tests (kept from Phase 1)
# ---------------------------------------------------------------------------


class TestCorpusValidateCmd:
    """Test corpus-validate command."""

    def test_validate_rejects_missing_corpus_index(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_validate

        args = argparse.Namespace(
            corpus_index=str(tmp_path / "nonexistent.yaml"),
            backend=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_corpus_validate(args)
        assert exc_info.value.code == 1

    def test_validate_detects_missing_manifest(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_validate

        idx_path = tmp_path / "corpus-index.yaml"
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.2.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [
                {
                    "dataset_id": "ds1",
                    "join_mode": "create_new",
                    "manifest_path": str(tmp_path / "missing-manifest.yaml"),
                }
            ],
        }
        idx_path.write_text(yaml.safe_dump(idx_data), encoding="utf-8")

        args = argparse.Namespace(corpus_index=str(idx_path), backend=None)
        with pytest.raises(SystemExit) as exc_info:
            _cmd_corpus_validate(args)
        assert exc_info.value.code == 1

    def test_validate_passes_clean_corpus(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_validate

        idx_path = tmp_path / "corpus-index.yaml"

        # Write a valid corpus index with a valid manifest
        manifest_path = tmp_path / "materialization-manifest.yaml"
        manifest_data = {
            "kind": "materialization-manifest",
            "contract_version": "0.3.0",
            "dataset_id": "ds1",
            "route": "create_new",
            "backend": "arrow-parquet",
            "topology": "federated",
            "count_source": {"selected": ".X", "integer_only": True},
            "outputs": {
                "metadata_root": str(tmp_path / "meta"),
                "matrix_root": str(tmp_path / "matrix"),
            },
            "provenance": {
                "source_path": "/fake.h5ad",
                "review_bundle": "/fake.yaml",
            },
        }
        manifest_path.write_text(yaml.safe_dump(manifest_data), encoding="utf-8")

        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.2.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [
                {
                    "dataset_id": "ds1",
                    "join_mode": "create_new",
                    "manifest_path": str(manifest_path),
                }
            ],
        }
        idx_path.write_text(yaml.safe_dump(idx_data), encoding="utf-8")

        args = argparse.Namespace(corpus_index=str(idx_path), backend=None)
        _cmd_corpus_validate(args)
        captured = capsys.readouterr()
        assert "PASS" in captured.out


# ---------------------------------------------------------------------------
# Corpus-gc command tests (new)
# ---------------------------------------------------------------------------


class TestCorpusGcCmd:
    """Test corpus-gc command."""

    def test_gc_rejects_missing_corpus_index(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_gc

        args = argparse.Namespace(
            corpus_root=str(tmp_path / "nonexistent"),
            dry_run=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_corpus_gc(args)
        assert exc_info.value.code == 1

    def test_gc_no_orphans_prints_message(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_gc

        # Create corpus index with one dataset
        idx_path = tmp_path / "corpus-index.yaml"
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.2.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [
                {
                    "dataset_id": "ds1",
                    "join_mode": "create_new",
                    "manifest_path": "ds1/meta/materialization-manifest.yaml",
                }
            ],
        }
        idx_path.write_text(yaml.safe_dump(idx_data), encoding="utf-8")

        # Create the registered dataset directory
        (tmp_path / "ds1" / "meta").mkdir(parents=True)

        args = argparse.Namespace(corpus_root=str(tmp_path), dry_run=False)
        _cmd_corpus_gc(args)
        captured = capsys.readouterr()
        assert "no orphaned dataset directories" in captured.out

    def test_gc_detects_orphans(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_gc

        # Create corpus index with registered dataset
        idx_path = tmp_path / "corpus-index.yaml"
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.2.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [
                {
                    "dataset_id": "ds1",
                    "join_mode": "create_new",
                    "manifest_path": "ds1/meta/materialization-manifest.yaml",
                }
            ],
        }
        idx_path.write_text(yaml.safe_dump(idx_data), encoding="utf-8")

        # Create registered dataset directory
        (tmp_path / "ds1" / "meta").mkdir(parents=True)

        # Create orphan directory
        orphan_dir = tmp_path / "ds_orphan"
        orphan_dir.mkdir()
        (orphan_dir / "meta").mkdir()
        (orphan_dir / "matrix").mkdir()

        args = argparse.Namespace(corpus_root=str(tmp_path), dry_run=False)
        _cmd_corpus_gc(args)
        captured = capsys.readouterr()

        # The orphan should have been detected and removed
        assert "found" in captured.out or "orphaned" in captured.out.lower()
        # ds_orphan should be removed
        assert not orphan_dir.exists()

    def test_gc_dry_run_does_not_remove(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_gc

        idx_path = tmp_path / "corpus-index.yaml"
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.2.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [],
        }
        idx_path.write_text(yaml.safe_dump(idx_data), encoding="utf-8")

        # Create orphan directory
        orphan_dir = tmp_path / "ds_orphan"
        orphan_dir.mkdir()
        (orphan_dir / "meta").mkdir()

        args = argparse.Namespace(corpus_root=str(tmp_path), dry_run=True)
        _cmd_corpus_gc(args)
        captured = capsys.readouterr()

        # On dry-run, orphans are NOT removed
        assert "dry-run" in captured.out
        assert orphan_dir.exists()


# ---------------------------------------------------------------------------
# CLI help output tests
# ---------------------------------------------------------------------------


class TestCLIHelp:
    """Test that CLI help output is correct."""

    def test_top_level_help(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])

    def test_subcommand_list(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        captured = capsys.readouterr()
        # Current commands should appear
        for cmd in ["inspect", "materialize", "canonicalize", "corpus-validate", "corpus-gc"]:
            assert cmd in captured.out
        # Removed commands should NOT appear
        for cmd in ["stage2-materialize", "corpus-create", "corpus-append"]:
            assert cmd not in captured.out


# ---------------------------------------------------------------------------
# Canonicalize CLI tests (Phase 3)
# ---------------------------------------------------------------------------


class TestCanonicalizeParser:
    """Verify the canonicalize CLI parser accepts all expected flag combinations."""

    def test_canonicalize_bulk_mode(self):
        """Bulk mode with --schema-dir."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--schema-dir", "/schemas/final/",
            "--corpus", "/corpus",
            "--output-dir", "/canonical",
        ])
        assert ns.command == "canonicalize"
        assert ns.schema_dir == "/schemas/final/"
        assert ns.corpus == "/corpus"
        assert ns.output_dir == "/canonical"
        assert ns.dataset_id is None
        assert ns.schema is None
        assert ns.dry_run is False

    def test_canonicalize_incremental_mode(self):
        """Incremental mode with --dataset-id and --schema."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--dataset-id", "dummy_00",
            "--schema", "/schemas/dummy_00-schema.yaml",
            "--corpus", "/corpus",
            "--output-dir", "/canonical",
        ])
        assert ns.command == "canonicalize"
        assert ns.dataset_id == "dummy_00"
        assert ns.schema == "/schemas/dummy_00-schema.yaml"
        assert ns.corpus == "/corpus"
        assert ns.output_dir == "/canonical"
        assert ns.schema_dir is None

    def test_canonicalize_bulk_with_dry_run(self):
        """Bulk mode with --dry-run."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--schema-dir", "/schemas/",
            "--corpus", "/corpus",
            "--output-dir", "/canonical",
            "--dry-run",
        ])
        assert ns.dry_run is True

    def test_canonicalize_incremental_with_dry_run(self):
        """Incremental mode with --dry-run."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--dataset-id", "dummy_00",
            "--schema", "/schemas/schema.yaml",
            "--corpus", "/corpus",
            "--output-dir", "/canonical",
            "--dry-run",
        ])
        assert ns.dry_run is True
        assert ns.dataset_id == "dummy_00"

    def test_canonicalize_default_dry_run_false(self):
        """--dry-run defaults to False."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--schema-dir", "/schemas/",
            "--corpus", "/corpus",
            "--output-dir", "/canonical",
        ])
        assert ns.dry_run is False

    def test_canonicalize_parser_accepts_all_combined(self):
        """Parser accepts --schema-dir + --dataset-id at parse time
        (validation happens in _cmd_canonicalize)."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--schema-dir", "/schemas/",
            "--dataset-id", "dummy_00",
            "--schema", "/schemas/schema.yaml",
            "--corpus", "/corpus",
            "--output-dir", "/canonical",
        ])
        assert ns.schema_dir == "/schemas/"
        assert ns.dataset_id == "dummy_00"
        assert ns.schema == "/schemas/schema.yaml"


class TestCanonicalizeDiscovery:
    """Test schema discovery and sidecar resolution helpers."""

    def test_discover_schemas_from_dir(self, tmp_path: Path):
        from perturb_data_lab.cli import _discover_schemas_from_dir

        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()

        # Write two schemas with different dataset_ids
        (schema_dir / "final-dummy_00-canonicalization-schema.yaml").write_text(
            yaml.safe_dump({
                "kind": "canonicalization-schema",
                "contract_version": "0.3.0",
                "dataset_id": "dummy_00",
                "status": "ready",
                "description": "",
                "gene_mapping": {"enabled": False, "engine": "identity"},
            })
        )
        (schema_dir / "final-dummy_01-canonicalization-schema.yaml").write_text(
            yaml.safe_dump({
                "kind": "canonicalization-schema",
                "contract_version": "0.3.0",
                "dataset_id": "dummy_01",
                "status": "ready",
                "description": "",
                "gene_mapping": {"enabled": False, "engine": "identity"},
            })
        )
        # Non-schema files should be ignored
        (schema_dir / "notes.txt").write_text("ignored")

        schemas = _discover_schemas_from_dir(schema_dir)
        assert len(schemas) == 2
        assert schemas[0][0] == "dummy_00"
        assert schemas[1][0] == "dummy_01"

    def test_discover_schemas_skips_duplicate_dataset_id(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _discover_schemas_from_dir

        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()

        # Two schema files with the same dataset_id
        (schema_dir / "dummy_00-a-canonicalization-schema.yaml").write_text(
            yaml.safe_dump({
                "kind": "canonicalization-schema",
                "contract_version": "0.3.0",
                "dataset_id": "dummy_00",
                "status": "ready",
                "description": "",
                "gene_mapping": {"enabled": False, "engine": "identity"},
            })
        )
        (schema_dir / "dummy_00-b-canonicalization-schema.yaml").write_text(
            yaml.safe_dump({
                "kind": "canonicalization-schema",
                "contract_version": "0.3.0",
                "dataset_id": "dummy_00",
                "status": "ready",
                "description": "",
                "gene_mapping": {"enabled": False, "engine": "identity"},
            })
        )

        schemas = _discover_schemas_from_dir(schema_dir)
        assert len(schemas) == 1
        captured = capsys.readouterr()
        assert "duplicate dataset_id" in captured.err.lower()

    def test_discover_schemas_raises_on_empty_dir(self, tmp_path: Path):
        from perturb_data_lab.cli import _discover_schemas_from_dir

        schema_dir = tmp_path / "empty"
        schema_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No .*canonicalization-schema.yaml"):
            _discover_schemas_from_dir(schema_dir)

    def test_discover_schemas_raises_on_nonexistent_dir(self, tmp_path: Path):
        from perturb_data_lab.cli import _discover_schemas_from_dir

        schema_dir = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            _discover_schemas_from_dir(schema_dir)

    def test_resolve_sidecars_from_corpus(self, tmp_path: Path):
        from perturb_data_lab.cli import _resolve_sidecars_from_corpus

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()

        # Create manifest
        manifest_dir = corpus_root / "meta" / "dummy_00"
        manifest_dir.mkdir(parents=True)
        manifest_path = manifest_dir / "materialization-manifest.yaml"
        manifest_path.write_text(yaml.safe_dump({
            "kind": "materialization-manifest",
            "contract_version": "0.3.0",
            "dataset_id": "dummy_00",
            "route": "create_new",
            "backend": "lance",
            "topology": "aggregate",
            "count_source": {"selected": ".X", "integer_only": True},
            "outputs": {
                "metadata_root": str(manifest_dir),
                "matrix_root": str(corpus_root / "matrix"),
            },
            "provenance": {
                "source_path": "/fake.h5ad",
                "review_bundle": "/fake.yaml",
            },
            "raw_cell_meta_path": str(manifest_dir / "raw-obs.parquet"),
            "raw_feature_meta_path": str(manifest_dir / "raw-var.parquet"),
            "size_factor_parquet_path": str(manifest_dir / "size-factor.parquet"),
        }), encoding="utf-8")

        # Create corpus index
        idx_path = corpus_root / "corpus-index.yaml"
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [{
                "dataset_id": "dummy_00",
                "join_mode": "create_new",
                "manifest_path": str(manifest_path.relative_to(corpus_root)),
                "dataset_index": 0,
                "cell_count": 100,
                "global_start": 0,
                "global_end": 100,
            }],
        }
        idx_path.write_text(yaml.safe_dump(idx_data))

        raw_obs, raw_var, size_factor = _resolve_sidecars_from_corpus(
            "dummy_00", corpus_root
        )
        assert raw_obs == str(manifest_dir / "raw-obs.parquet")
        assert raw_var == str(manifest_dir / "raw-var.parquet")
        assert size_factor == str(manifest_dir / "size-factor.parquet")

    def test_resolve_sidecars_no_size_factor(self, tmp_path: Path):
        from perturb_data_lab.cli import _resolve_sidecars_from_corpus

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()

        manifest_dir = corpus_root / "meta" / "dummy_00"
        manifest_dir.mkdir(parents=True)
        manifest_path = manifest_dir / "materialization-manifest.yaml"
        manifest_path.write_text(yaml.safe_dump({
            "kind": "materialization-manifest",
            "contract_version": "0.3.0",
            "dataset_id": "dummy_00",
            "route": "create_new",
            "backend": "lance",
            "topology": "aggregate",
            "count_source": {"selected": ".X", "integer_only": True},
            "outputs": {
                "metadata_root": str(manifest_dir),
                "matrix_root": str(corpus_root / "matrix"),
            },
            "provenance": {
                "source_path": "/fake.h5ad",
                "review_bundle": "/fake.yaml",
            },
            "raw_cell_meta_path": str(manifest_dir / "raw-obs.parquet"),
            "raw_feature_meta_path": str(manifest_dir / "raw-var.parquet"),
        }), encoding="utf-8")

        idx_path = corpus_root / "corpus-index.yaml"
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [{
                "dataset_id": "dummy_00",
                "join_mode": "create_new",
                "manifest_path": str(manifest_path.relative_to(corpus_root)),
                "dataset_index": 0,
                "cell_count": 100,
                "global_start": 0,
                "global_end": 100,
            }],
        }
        idx_path.write_text(yaml.safe_dump(idx_data))

        raw_obs, raw_var, size_factor = _resolve_sidecars_from_corpus(
            "dummy_00", corpus_root
        )
        assert raw_obs == str(manifest_dir / "raw-obs.parquet")
        assert raw_var == str(manifest_dir / "raw-var.parquet")
        assert size_factor is None

    def test_resolve_sidecars_raises_on_missing_corpus(self, tmp_path: Path):
        from perturb_data_lab.cli import _resolve_sidecars_from_corpus

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        # No corpus-index.yaml

        with pytest.raises(FileNotFoundError, match="corpus-index.yaml not found"):
            _resolve_sidecars_from_corpus("dummy_00", corpus_root)

    def test_resolve_sidecars_raises_on_missing_dataset(self, tmp_path: Path):
        from perturb_data_lab.cli import _resolve_sidecars_from_corpus

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()

        idx_path = corpus_root / "corpus-index.yaml"
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [{
                "dataset_id": "dummy_00",
                "join_mode": "create_new",
                "manifest_path": "dummy_00/manifest.yaml",
                "dataset_index": 0,
                "cell_count": 100,
                "global_start": 0,
                "global_end": 100,
            }],
        }
        idx_path.write_text(yaml.safe_dump(idx_data))

        with pytest.raises(ValueError, match="not found in corpus"):
            _resolve_sidecars_from_corpus("dummy_99", corpus_root)

    def test_resolve_sidecars_raises_on_missing_manifest(self, tmp_path: Path):
        from perturb_data_lab.cli import _resolve_sidecars_from_corpus

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()

        idx_path = corpus_root / "corpus-index.yaml"
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [{
                "dataset_id": "dummy_00",
                "join_mode": "create_new",
                "manifest_path": "dummy_00/missing-manifest.yaml",
                "dataset_index": 0,
                "cell_count": 100,
                "global_start": 0,
                "global_end": 100,
            }],
        }
        idx_path.write_text(yaml.safe_dump(idx_data))

        with pytest.raises(FileNotFoundError, match="materialization manifest not found"):
            _resolve_sidecars_from_corpus("dummy_00", corpus_root)


class TestCanonicalizeCmd:
    """Test _cmd_canonicalize behavior via direct invocation."""

    def test_cmd_rejects_no_mode(self, tmp_path: Path, capsys):
        """Neither --schema-dir nor --dataset-id+--schema."""
        from perturb_data_lab.cli import _cmd_canonicalize

        args = argparse.Namespace(
            schema_dir=None,
            dataset_id=None,
            schema=None,
            corpus=str(tmp_path),
            output_dir=str(tmp_path / "output"),
            dry_run=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_canonicalize(args)
        assert exc_info.value.code == 1

    def test_cmd_rejects_schema_dir_plus_dataset_id(self, tmp_path: Path, capsys):
        """--schema-dir combined with --dataset-id."""
        from perturb_data_lab.cli import _cmd_canonicalize

        args = argparse.Namespace(
            schema_dir=str(tmp_path),
            dataset_id="dummy_00",
            schema=None,
            corpus=str(tmp_path),
            output_dir=str(tmp_path / "output"),
            dry_run=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_canonicalize(args)
        assert exc_info.value.code == 1

    def test_cmd_accepts_incremental_mode_flags(self, tmp_path: Path, capsys):
        """Incremental mode validates schema existence."""
        from perturb_data_lab.cli import _cmd_canonicalize

        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()

        # Create manifest and corpus needed for sidecar resolution
        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        manifest_dir = corpus_root / "meta" / "dummy_00"
        manifest_dir.mkdir(parents=True)
        manifest_path = manifest_dir / "materialization-manifest.yaml"
        manifest_path.write_text(yaml.safe_dump({
            "kind": "materialization-manifest",
            "contract_version": "0.3.0",
            "dataset_id": "dummy_00",
            "route": "create_new",
            "backend": "lance",
            "topology": "aggregate",
            "count_source": {"selected": ".X", "integer_only": True},
            "outputs": {
                "metadata_root": str(manifest_dir),
                "matrix_root": str(corpus_root / "matrix"),
            },
            "provenance": {
                "source_path": "/fake.h5ad",
                "review_bundle": "/fake.yaml",
            },
            "raw_cell_meta_path": str(manifest_dir / "raw-obs.parquet"),
            "raw_feature_meta_path": str(manifest_dir / "raw-var.parquet"),
        }), encoding="utf-8")

        idx_path = corpus_root / "corpus-index.yaml"
        idx_path.write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [{
                "dataset_id": "dummy_00",
                "join_mode": "create_new",
                "manifest_path": str(manifest_path.relative_to(corpus_root)),
                "dataset_index": 0,
                "cell_count": 100,
                "global_start": 0,
                "global_end": 100,
            }],
        }))

        schema_path = schema_dir / "dummy_00-canonicalization-schema.yaml"
        schema_path.write_text(yaml.safe_dump({
            "kind": "canonicalization-schema",
            "contract_version": "0.3.0",
            "dataset_id": "dummy_00",
            "status": "ready",
            "description": "",
            "gene_mapping": {"enabled": False, "engine": "identity"},
        }))

        output_dir = tmp_path / "output"
        args = argparse.Namespace(
            schema_dir=None,
            dataset_id="dummy_00",
            schema=str(schema_path),
            corpus=str(corpus_root),
            output_dir=str(output_dir),
            dry_run=True,
        )
        # Should not raise - dry-run succeeds without actual sidecar files
        _cmd_canonicalize(args)
        captured = capsys.readouterr()
        assert "incremental mode" in captured.out
        assert "dry-run" in captured.out.lower()
        assert "dummy_00" in captured.out

    def test_cmd_rejects_missing_schema_dir(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_canonicalize

        args = argparse.Namespace(
            schema_dir=str(tmp_path / "nonexistent"),
            dataset_id=None,
            schema=None,
            corpus=str(tmp_path),
            output_dir=str(tmp_path / "output"),
            dry_run=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_canonicalize(args)
        assert exc_info.value.code == 1

    def test_cmd_rejects_missing_schema_file(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_canonicalize

        args = argparse.Namespace(
            schema_dir=None,
            dataset_id="dummy_00",
            schema=str(tmp_path / "nonexistent.yaml"),
            corpus=str(tmp_path),
            output_dir=str(tmp_path / "output"),
            dry_run=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_canonicalize(args)
        assert exc_info.value.code == 1
