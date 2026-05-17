"""Phase 2 tests for unified materialize CLI and corpus-gc command."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest
import yaml

from perturb_data_lab.cli import (
    _DatasetInput,
    _infer_backend_topology_from_corpus,
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

    def test_inspect_direct_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args([
            "inspect",
            "--source", "/data/test.h5ad",
            "--dataset-id", "dummy_00",
            "--output-dir", "/inspection",
        ])
        assert ns.command == "inspect"
        assert ns.source == "/data/test.h5ad"
        assert ns.dataset_id == "dummy_00"
        assert ns.output_dir == "/inspection"

    def test_materialize_single_dataset(self):
        parser = build_parser()
        ns = parser.parse_args([
            "materialize",
            "--mode", "create",
            "--source", "/data/test.h5ad",
            "--dataset-id", "ds1",
            "--review-bundle", "/data/summary.yaml",
            "--output-corpus", "/corpus",
            "--backend", "zarr",
        ])
        assert ns.command == "materialize"
        assert ns.mode == "create"
        assert ns.source == "/data/test.h5ad"
        assert ns.dataset_id == "ds1"
        assert ns.review_bundle == "/data/summary.yaml"
        assert ns.output_corpus == "/corpus"
        assert ns.backend == "zarr"
        assert ns.topology is None
        assert ns.dry_run is False

    def test_materialize_append_can_omit_backend_and_topology(self):
        parser = build_parser()
        ns = parser.parse_args([
            "materialize",
            "--mode", "append",
            "--input-list", "/data/datasets.csv",
            "--output-corpus", "/corpus",
        ])
        assert ns.command == "materialize"
        assert ns.mode == "append"
        assert ns.backend is None
        assert ns.topology is None

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
            "--backend", "zarr",
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
            "--backend", "zarr",
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
            "--backend", "zarr",
        ])
        assert ns.dataset_id == "ds1"

    def test_corpus_validate_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(
            ["corpus-validate", "/c.yaml", "--backend", "zarr"]
        )
        assert ns.command == "corpus-validate"
        assert ns.corpus_index == "/c.yaml"
        assert ns.backend == "zarr"

    def test_removed_backend_choice_is_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "materialize",
                "--mode", "create",
                "--source", "/data/test.h5ad",
                "--dataset-id", "ds1",
                "--review-bundle", "/data/summary.yaml",
                "--output-corpus", "/corpus",
                "--backend", "arrow-parquet",
            ])

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

    def test_backfill_hvg_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args([
            "backfill-hvg",
            "--corpus-root", "/corpus",
            "--dataset-id", "ds1",
            "--chunk-rows", "1234",
            "--summary-json", "/tmp/summary.json",
        ])
        assert ns.command == "backfill-hvg"
        assert ns.corpus_root == "/corpus"
        assert ns.dataset_id == ["ds1"]
        assert ns.chunk_rows == 1234
        assert ns.summary_json == "/tmp/summary.json"
        assert ns.update_manifests is True

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
    """Ensure backend/topology routing respects the requested topology."""

    def test_lance_respects_requested_federated(self):
        assert _resolve_effective_topology("lance", "federated") == "federated"

    def test_lance_respects_requested_aggregate(self):
        assert _resolve_effective_topology("lance", "aggregate") == "aggregate"

    def test_non_lance_respects_requested_topology(self):
        assert _resolve_effective_topology("zarr", "federated") == "federated"
        assert _resolve_effective_topology("zarr", "aggregate") == "aggregate"


class TestMaterializeDatasetRouting:
    """Verify _materialize_dataset routes aggregate matrix_root correctly."""

    def test_lance_aggregate_uses_corpus_wide_matrix_root(self, tmp_path: Path, monkeypatch):
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
            topology="aggregate",
            rerun_stage1=False,
            n_hvg=2000,
            corpus_id="test-corpus",
        )
        corpus_root = tmp_path / "corpus"

        next_global, next_writer_state = _materialize_dataset(
            ds,
            args,
            backend="lance",
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

    def test_lance_federated_uses_dataset_matrix_root(self, tmp_path: Path, monkeypatch):
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
            topology="federated",
            rerun_stage1=False,
            n_hvg=2000,
            corpus_id="test-corpus",
        )
        corpus_root = tmp_path / "corpus"

        next_global, next_writer_state = _materialize_dataset(
            ds,
            args,
            backend="lance",
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
        assert str(output_roots.matrix_root) == str(corpus_root / "dummy_00" / "matrix")
        assert str(output_roots.metadata_root) == str(corpus_root / "dummy_00" / "meta")
        assert captured["topology"] == "federated"
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
            "backend": "zarr",
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
        for cmd in ["inspect", "materialize", "draft-schema", "canonicalize", "backfill-hvg", "corpus-validate", "corpus-gc"]:
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
        """Bulk mode with only --corpus."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--corpus", "/corpus",
        ])
        assert ns.command == "canonicalize"
        assert ns.corpus == "/corpus"
        assert ns.dataset_id is None
        assert ns.dry_run is False

    def test_canonicalize_incremental_mode(self):
        """Incremental mode with --dataset-id."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--dataset-id", "dummy_00",
            "--corpus", "/corpus",
        ])
        assert ns.command == "canonicalize"
        assert ns.dataset_id == "dummy_00"
        assert ns.corpus == "/corpus"

    def test_canonicalize_bulk_with_dry_run(self):
        """Bulk mode with --dry-run."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--corpus", "/corpus",
            "--dry-run",
        ])
        assert ns.dry_run is True

    def test_canonicalize_incremental_with_dry_run(self):
        """Incremental mode with --dry-run."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--dataset-id", "dummy_00",
            "--corpus", "/corpus",
            "--dry-run",
        ])
        assert ns.dry_run is True
        assert ns.dataset_id == "dummy_00"

    def test_canonicalize_default_dry_run_false(self):
        """--dry-run defaults to False."""
        parser = build_parser()
        ns = parser.parse_args([
            "canonicalize",
            "--corpus", "/corpus",
        ])
        assert ns.dry_run is False

    def test_canonicalize_rejects_missing_corpus(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["canonicalize"])


class TestDraftSchemaParser:
    """Verify draft-schema parser behavior."""

    def test_draft_schema_basic(self):
        parser = build_parser()
        ns = parser.parse_args(["draft-schema", "--corpus", "/corpus"])
        assert ns.command == "draft-schema"
        assert ns.corpus == "/corpus"
        assert ns.force_all is False

    def test_draft_schema_force_all(self):
        parser = build_parser()
        ns = parser.parse_args(["draft-schema", "--corpus", "/corpus", "--force-all"])
        assert ns.force_all is True


class TestCanonicalizeDiscovery:
    """Test corpus metadata and sidecar resolution helpers."""

    def test_infer_backend_topology_from_global_metadata(self, tmp_path: Path):
        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {
                "backend": "lance",
                "topology": "aggregate",
            },
            "datasets": [],
        }))

        backend, topology = _infer_backend_topology_from_corpus(corpus_root)
        assert backend == "lance"
        assert topology == "aggregate"

    def test_infer_backend_topology_requires_global_metadata(self, tmp_path: Path):
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
        }))

        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
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

        with pytest.raises(ValueError, match="global_metadata.backend is missing"):
            _infer_backend_topology_from_corpus(corpus_root)

    def test_infer_backend_topology_raises_when_unknown(self, tmp_path: Path):
        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [],
        }))
        with pytest.raises(ValueError, match="global_metadata.backend is missing"):
            _infer_backend_topology_from_corpus(corpus_root)

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

    def test_cmd_rejects_when_no_final_schemas_discovered(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_canonicalize

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {"backend": "lance", "topology": "aggregate"},
            "datasets": [{
                "dataset_id": "dummy_00",
                "join_mode": "create_new",
                "manifest_path": "meta/dummy_00/materialization-manifest.yaml",
                "dataset_index": 0,
                "cell_count": 100,
                "global_start": 0,
                "global_end": 100,
            }],
        }))

        args = argparse.Namespace(
            dataset_id=None,
            corpus=str(corpus_root),
            dry_run=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_canonicalize(args)
        assert exc_info.value.code == 1

    def test_cmd_rejects_unknown_dataset_id(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_canonicalize

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {"backend": "lance", "topology": "aggregate"},
            "datasets": [],
        }))

        args = argparse.Namespace(
            dataset_id="missing_ds",
            corpus=str(corpus_root),
            dry_run=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_canonicalize(args)
        assert exc_info.value.code == 1

    def test_cmd_accepts_incremental_mode_flags(self, tmp_path: Path, capsys):
        """Incremental mode with final-schema in corpus meta succeeds in dry-run."""
        from perturb_data_lab.cli import _cmd_canonicalize

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
            "global_metadata": {"backend": "lance", "topology": "aggregate"},
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

        schema_path = manifest_dir / "final-schema.yaml"
        schema_path.write_text(yaml.safe_dump({
            "kind": "canonicalization-schema",
            "contract_version": "0.3.0",
            "dataset_id": "dummy_00",
            "status": "ready",
            "description": "",
            "gene_mapping": {"enabled": False, "engine": "identity"},
        }))

        args = argparse.Namespace(
            dataset_id="dummy_00",
            corpus=str(corpus_root),
            dry_run=True,
        )
        # Should not raise - dry-run succeeds without actual sidecar files
        _cmd_canonicalize(args)
        captured = capsys.readouterr()
        assert "dry-run" in captured.out.lower()
        assert "dummy_00" in captured.out

    def test_cmd_incremental_canonicalize_does_not_write_corpus_vocab(
        self,
        tmp_path: Path,
        monkeypatch,
        capsys,
    ):
        from perturb_data_lab.canonical.runner import CanonicalizationResult
        from perturb_data_lab.cli import _cmd_canonicalize

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        meta_root_0 = corpus_root / "meta" / "dummy_00"
        meta_root_1 = corpus_root / "meta" / "dummy_01"
        canonical_root_1 = meta_root_1 / "canonical_meta"
        canonical_root_1.mkdir(parents=True)

        for meta_root, dataset_id in [(meta_root_0, "dummy_00"), (meta_root_1, "dummy_01")]:
            meta_root.mkdir(parents=True, exist_ok=True)
            (meta_root / "final-schema.yaml").write_text(
                yaml.safe_dump({
                    "kind": "canonicalization-schema",
                    "contract_version": "0.3.0",
                    "dataset_id": dataset_id,
                    "status": "ready",
                    "description": "",
                    "gene_mapping": {"enabled": False, "engine": "identity"},
                }),
                encoding="utf-8",
            )

        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {"backend": "lance", "topology": "aggregate"},
            "datasets": [
                {
                    "dataset_id": "dummy_00",
                    "join_mode": "create_new",
                    "manifest_path": "meta/dummy_00/materialization-manifest.yaml",
                    "dataset_index": 0,
                    "cell_count": 1,
                    "global_start": 0,
                    "global_end": 1,
                },
                {
                    "dataset_id": "dummy_01",
                    "join_mode": "append_routed",
                    "manifest_path": "meta/dummy_01/materialization-manifest.yaml",
                    "dataset_index": 1,
                    "cell_count": 1,
                    "global_start": 1,
                    "global_end": 2,
                },
            ],
        }))

        def _fake_sidecars(dataset_id: str, corpus_root_arg: Path):
            assert dataset_id == "dummy_01"
            return ("/fake/raw-obs.parquet", "/fake/raw-var.parquet", None)

        def _fake_run_canonicalization(**kwargs):
            return CanonicalizationResult(
                dataset_id="dummy_01",
                obs_path=canonical_root_1 / "canonical-obs.parquet",
                var_path=canonical_root_1 / "canonical-var.parquet",
                obs_rows=1,
                var_rows=1,
            )

        monkeypatch.setattr(
            "perturb_data_lab.cli._resolve_sidecars_from_corpus",
            _fake_sidecars,
        )
        monkeypatch.setattr(
            "perturb_data_lab.canonical.run_canonicalization",
            _fake_run_canonicalization,
        )

        args = argparse.Namespace(
            dataset_id="dummy_01",
            corpus=str(corpus_root),
            dry_run=False,
        )
        _cmd_canonicalize(args)
        captured = capsys.readouterr()

        assert not (corpus_root / "corpus-vocab.yaml").exists()
        assert "merged vocab" not in captured.out.lower()


class TestInspectCmd:
    """Test _cmd_inspect behavior for batch and direct modes."""

    def test_cmd_inspect_rejects_mixed_batch_and_direct_flags(self):
        from perturb_data_lab.cli import _cmd_inspect

        args = argparse.Namespace(
            config="/tmp/config.yaml",
            source="/tmp/source.h5ad",
            dataset_id="dummy_00",
            output_dir="/tmp/out",
            workers=1,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_inspect(args)
        assert exc_info.value.code == 1

    def test_cmd_inspect_direct_mode(self, tmp_path: Path, monkeypatch):
        from perturb_data_lab.cli import _cmd_inspect

        source = tmp_path / "dummy.h5ad"
        source.write_text("not-real-h5ad")

        captured: dict[str, object] = {}

        class _Artifacts:
            review_bundle = tmp_path / "out" / "dummy_00" / "dataset-summary.yaml"

        def _fake_inspect_target(target, output_root):
            captured["dataset_id"] = target.dataset_id
            captured["source_path"] = target.source_path
            captured["source_release"] = target.source_release
            captured["output_root"] = output_root
            return _Artifacts()

        monkeypatch.setattr(
            "perturb_data_lab.inspectors.workflow.inspect_target",
            _fake_inspect_target,
        )

        args = argparse.Namespace(
            config=None,
            source=str(source),
            dataset_id="dummy_00",
            output_dir=str(tmp_path / "out"),
            workers=1,
        )
        _cmd_inspect(args)

        assert captured["dataset_id"] == "dummy_00"
        assert captured["source_path"] == str(source)
        assert captured["source_release"] == "dummy_00"
        assert str(captured["output_root"]) == str(tmp_path / "out")


class TestDraftSchemaCmd:
    """Test draft-schema command behavior."""

    def test_cmd_draft_schema_writes_draft_for_uncanonicalized_dataset(
        self,
        tmp_path: Path,
    ):
        from perturb_data_lab.cli import _cmd_draft_schema

        corpus_root = tmp_path / "corpus"
        meta_root = corpus_root / "meta" / "dummy_00"
        meta_root.mkdir(parents=True)

        (meta_root / "dataset-summary.yaml").write_text(yaml.safe_dump({
            "kind": "dataset-summary",
            "contract_version": "0.3.0",
            "dataset": {
                "dataset_id": "dummy_00",
                "source_release": "dummy_00",
                "source_path": "/fake.h5ad",
                "obs_rows": 10,
                "var_rows": 5,
                "obs_index_name": "index",
                "var_index_name": "index",
            },
            "structure": {
                "has_raw": False,
                "raw_var_rows": 0,
                "layers": [],
            },
            "obs_fields": [
                {
                    "name": "cell_id",
                    "dtype": "object",
                    "null_count": 0,
                    "sampled_unique_values": 2,
                    "examples": ["c1", "c2"],
                }
            ],
            "var_fields": [
                {
                    "name": "gene_id",
                    "dtype": "object",
                    "null_count": 0,
                    "sampled_unique_values": 2,
                    "examples": ["g1", "g2"],
                }
            ],
            "count_source_candidates": [],
            "count_source_decision": {
                "selected_candidate": ".X",
                "status": "pass",
                "confidence": "high",
                "recovery_policy": "not-needed",
                "rationale": "test",
                "uses_recovery": False,
            },
            "materialization_readiness": "pass",
            "inspector_notes": [],
        }))

        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {"backend": "lance", "topology": "aggregate"},
            "datasets": [{
                "dataset_id": "dummy_00",
                "join_mode": "create_new",
                "manifest_path": "meta/dummy_00/materialization-manifest.yaml",
                "dataset_index": 0,
                "cell_count": 10,
                "global_start": 0,
                "global_end": 10,
            }],
        }))

        args = argparse.Namespace(corpus=str(corpus_root), force_all=False)
        _cmd_draft_schema(args)

        assert (meta_root / "draft-schema.yaml").exists()

    def test_cmd_draft_schema_uses_inspection_suggestions(self, tmp_path: Path):
        from perturb_data_lab.canonical.contract import CanonicalizationSchema
        from perturb_data_lab.cli import _cmd_draft_schema

        corpus_root = tmp_path / "corpus"
        meta_root = corpus_root / "meta" / "dummy_00"
        meta_root.mkdir(parents=True)

        (meta_root / "dataset-summary.yaml").write_text(yaml.safe_dump({
            "kind": "dataset-summary",
            "contract_version": "0.3.0",
            "dataset": {
                "dataset_id": "dummy_00",
                "source_release": "dummy_00",
                "source_path": "/fake.h5ad",
                "obs_rows": 10,
                "var_rows": 5,
                "obs_index_name": "index",
                "var_index_name": "index",
            },
            "structure": {
                "has_raw": False,
                "raw_var_rows": 0,
                "layers": [],
            },
            "obs_fields": [
                {
                    "name": "perturbation",
                    "dtype": "object",
                    "null_count": 0,
                    "sampled_unique_values": 2,
                    "examples": [" NTC ", "STAT1"],
                },
                {
                    "name": "cell_id",
                    "dtype": "object",
                    "null_count": 0,
                    "sampled_unique_values": 2,
                    "examples": ["c1", "c2"],
                },
            ],
            "var_fields": [
                {
                    "name": "feature_id",
                    "dtype": "object",
                    "null_count": 0,
                    "sampled_unique_values": 2,
                    "examples": ["ENSG00000141510.18", "ENSG00000139618.12"],
                }
            ],
            "control_label_candidates": [
                {
                    "column": "perturbation",
                    "candidate_values": ["NTC"],
                    "suggested_output": "ctrl",
                    "confidence": "high",
                    "reason": "explicit control labels",
                }
            ],
            "count_source_candidates": [],
            "count_source_decision": {
                "selected_candidate": ".X",
                "status": "pass",
                "confidence": "high",
                "recovery_policy": "not-needed",
                "rationale": "test",
                "uses_recovery": False,
            },
            "materialization_readiness": "pass",
            "inspector_notes": [],
        }))

        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {"backend": "lance", "topology": "aggregate"},
            "datasets": [{
                "dataset_id": "dummy_00",
                "join_mode": "create_new",
                "manifest_path": "meta/dummy_00/materialization-manifest.yaml",
                "dataset_index": 0,
                "cell_count": 10,
                "global_start": 0,
                "global_end": 10,
            }],
        }))

        args = argparse.Namespace(corpus=str(corpus_root), force_all=False)
        _cmd_draft_schema(args)

        schema = CanonicalizationSchema.from_yaml_file(meta_root / "draft-schema.yaml")
        perturb_mapping = {m.canonical_name: m for m in schema.obs_column_mappings}["perturb_label"]
        assert any(transform.name == "map_control_labels" for transform in perturb_mapping.transforms)
        gene_mapping = {m.canonical_name: m for m in schema.var_column_mappings}["gene_id"]
        assert any(transform.name == "strip_ensembl_version" for transform in gene_mapping.transforms)


class TestMaterializeCmd:
    """Test _cmd_materialize backend/topology auto-detection behavior."""

    def test_cmd_materialize_create_defaults_lance_to_federated(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from perturb_data_lab.cli import _cmd_materialize

        h5ad = tmp_path / "dummy_00.h5ad"
        h5ad.write_text("fake")
        review = tmp_path / "dummy_00-summary.yaml"
        review.write_text("kind: dataset-summary")
        corpus_root = tmp_path / "corpus"

        args = argparse.Namespace(
            mode="create",
            source=str(h5ad),
            input_list=None,
            input_dir=None,
            dataset_id="dummy_00",
            review_bundle=str(review),
            review_bundle_dir=None,
            output_corpus=str(corpus_root),
            backend="lance",
            topology=None,
            corpus_id="test-v0",
            rerun_stage1=False,
            n_hvg=2000,
            dry_run=True,
        )

        captured: dict[str, object] = {}

        def _fake_materialize_dataset(ds, parsed_args, **kwargs):
            captured["backend"] = kwargs["backend"]
            captured["effective_topology"] = kwargs["effective_topology"]
            return (0, None)

        monkeypatch.setattr(
            "perturb_data_lab.cli._materialize_dataset",
            _fake_materialize_dataset,
        )

        _cmd_materialize(args)

        assert captured["backend"] == "lance"
        assert captured["effective_topology"] == "federated"

    def test_cmd_materialize_append_autodetects_backend_and_topology(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from perturb_data_lab.cli import _cmd_materialize

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {"backend": "lance", "topology": "aggregate"},
            "datasets": [],
        }))

        h5ad = tmp_path / "dummy_00.h5ad"
        h5ad.write_text("fake")
        review = tmp_path / "dummy_00-summary.yaml"
        review.write_text("kind: dataset-summary")

        args = argparse.Namespace(
            mode="append",
            source=str(h5ad),
            input_list=None,
            input_dir=None,
            dataset_id="dummy_00",
            review_bundle=str(review),
            review_bundle_dir=None,
            output_corpus=str(corpus_root),
            backend=None,
            topology=None,
            corpus_id="test-v0",
            rerun_stage1=False,
            n_hvg=2000,
            dry_run=True,
        )

        captured: dict[str, object] = {}

        def _fake_materialize_dataset(ds, parsed_args, **kwargs):
            captured["backend"] = kwargs["backend"]
            captured["effective_topology"] = kwargs["effective_topology"]
            return (0, None)

        monkeypatch.setattr(
            "perturb_data_lab.cli._materialize_dataset",
            _fake_materialize_dataset,
        )

        _cmd_materialize(args)

        assert captured["backend"] == "lance"
        assert captured["effective_topology"] == "aggregate"

    def test_cmd_materialize_append_seeds_existing_offsets(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from perturb_data_lab.cli import _cmd_materialize

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {"backend": "lance", "topology": "aggregate"},
            "datasets": [{
                "dataset_id": "dummy_00",
                "join_mode": "create_new",
                "manifest_path": "meta/dummy_00/materialization-manifest.yaml",
                "dataset_index": 0,
                "cell_count": 10,
                "global_start": 0,
                "global_end": 10,
            }],
        }))

        h5ad = tmp_path / "dummy_01.h5ad"
        h5ad.write_text("fake")
        review = tmp_path / "dummy_01-summary.yaml"
        review.write_text("kind: dataset-summary")

        args = argparse.Namespace(
            mode="append",
            source=str(h5ad),
            input_list=None,
            input_dir=None,
            dataset_id="dummy_01",
            review_bundle=str(review),
            review_bundle_dir=None,
            output_corpus=str(corpus_root),
            backend=None,
            topology=None,
            corpus_id="test-v0",
            rerun_stage1=False,
            n_hvg=2000,
            dry_run=True,
        )

        captured: dict[str, object] = {}

        def _fake_materialize_dataset(ds, parsed_args, **kwargs):
            captured["dataset_index"] = kwargs["dataset_index"]
            captured["global_row_start"] = kwargs["global_row_start"]
            return (kwargs["global_row_start"], None)

        monkeypatch.setattr(
            "perturb_data_lab.cli._materialize_dataset",
            _fake_materialize_dataset,
        )

        _cmd_materialize(args)

        assert captured["dataset_index"] == 1
        assert captured["global_row_start"] == 10

    def test_cmd_materialize_append_bulk_continues_existing_offsets(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from perturb_data_lab.cli import _cmd_materialize

        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        (corpus_root / "corpus-index.yaml").write_text(yaml.safe_dump({
            "kind": "corpus-index",
            "contract_version": "0.3.0",
            "corpus_id": "test-v0",
            "global_metadata": {"backend": "lance", "topology": "aggregate"},
            "datasets": [{
                "dataset_id": "dummy_00",
                "join_mode": "create_new",
                "manifest_path": "meta/dummy_00/materialization-manifest.yaml",
                "dataset_index": 0,
                "cell_count": 10,
                "global_start": 0,
                "global_end": 10,
            }],
        }))

        input_list = tmp_path / "inputs.csv"
        input_list.write_text(
            "source,dataset_id,review_bundle\n"
            f"{tmp_path / 'dummy_01.h5ad'},dummy_01,{tmp_path / 'dummy_01-summary.yaml'}\n"
            f"{tmp_path / 'dummy_02.h5ad'},dummy_02,{tmp_path / 'dummy_02-summary.yaml'}\n",
            encoding="utf-8",
        )
        (tmp_path / "dummy_01.h5ad").write_text("fake")
        (tmp_path / "dummy_02.h5ad").write_text("fake")
        (tmp_path / "dummy_01-summary.yaml").write_text("kind: dataset-summary")
        (tmp_path / "dummy_02-summary.yaml").write_text("kind: dataset-summary")

        args = argparse.Namespace(
            mode="append",
            source=None,
            input_list=str(input_list),
            input_dir=None,
            dataset_id=None,
            review_bundle=None,
            review_bundle_dir=None,
            output_corpus=str(corpus_root),
            backend=None,
            topology=None,
            corpus_id="test-v0",
            rerun_stage1=False,
            n_hvg=2000,
            dry_run=False,
        )

        calls: list[tuple[str, int, int]] = []
        fake_cell_counts = {"dummy_01": 7, "dummy_02": 5}

        def _fake_materialize_dataset(ds, parsed_args, **kwargs):
            calls.append(
                (ds.dataset_id, kwargs["dataset_index"], kwargs["global_row_start"])
            )
            return (kwargs["global_row_start"] + fake_cell_counts[ds.dataset_id], None)

        monkeypatch.setattr(
            "perturb_data_lab.cli._materialize_dataset",
            _fake_materialize_dataset,
        )

        _cmd_materialize(args)

        assert calls == [
            ("dummy_01", 1, 10),
            ("dummy_02", 2, 17),
        ]
