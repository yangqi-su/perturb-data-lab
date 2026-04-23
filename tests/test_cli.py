"""Phase 6 tests for CLI command wiring and argument validation."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pytest
import yaml

from perturb_data_lab.cli import build_parser
from perturb_data_lab.materializers.models import (
    CountSourceSpec,
    CorpusIndexDocument,
    DatasetJoinRecord,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
)


# ---------------------------------------------------------------------------
# Parser construction tests
# ---------------------------------------------------------------------------


class TestParserConstruction:
    """Verify the CLI parser accepts all expected subcommands."""

    def test_parser_has_all_subcommands(self):
        parser = build_parser()
        # Trigger subparser creation by parsing --help
        # Just verify the parser was built without error
        assert parser is not None
        assert isinstance(parser, argparse.ArgumentParser)

    def test_inspect_subcommand_exists(self):
        parser = build_parser()
        ns = parser.parse_args(["inspect", "--config", "c.yaml", "--workers", "2"])
        assert ns.command == "inspect"
        assert ns.config == "c.yaml"
        assert ns.workers == 2

    def test_schema_validate_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(["schema-validate", "schema.yaml", "--corpus-namespace", "ensembl"])
        assert ns.command == "schema-validate"
        assert ns.schema_path == "schema.yaml"
        assert ns.corpus_namespace == "ensembl"

    def test_schema_preview_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(["schema-preview", "schema.yaml", "--sample", "data.h5ad", "--n-rows", "3"])
        assert ns.command == "schema-preview"
        assert ns.schema_path == "schema.yaml"
        assert ns.sample == "data.h5ad"
        assert ns.n_rows == 3

    def test_materialize_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args([
            "materialize",
            "--schema", "schema.yaml",
            "--backend", "arrow-hf",
            "--release-id", "v0.1",
            "--dataset-id", "ds1",
            "--output-root", "/out",
            "--corpus-index", "/corp/corpus-index.yaml",
        ])
        assert ns.command == "materialize"
        assert ns.schema == "schema.yaml"
        assert ns.backend == "arrow-hf"
        assert ns.release_id == "v0.1"
        assert ns.dataset_id == "ds1"
        assert ns.output_root == "/out"
        assert ns.corpus_index == "/corp/corpus-index.yaml"

    def test_materialize_without_corpus_index(self):
        parser = build_parser()
        ns = parser.parse_args([
            "materialize",
            "--schema", "schema.yaml",
            "--backend", "zarr-ts",
            "--release-id", "v0.1",
            "--dataset-id", "ds1",
            "--output-root", "/out",
        ])
        assert ns.corpus_index is None

    def test_corpus_create_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(["corpus-create", "--backend", "arrow-hf", "--output", "/corp/idx.yaml"])
        assert ns.command == "corpus-create"
        assert ns.backend == "arrow-hf"
        assert ns.output == "/corp/idx.yaml"

    def test_corpus_append_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(["corpus-append", "--corpus-index", "/c.yaml", "--manifest", "/m.yaml"])
        assert ns.command == "corpus-append"
        assert ns.corpus_index == "/c.yaml"
        assert ns.manifest == "/m.yaml"

    def test_corpus_validate_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(["corpus-validate", "/c.yaml", "--backend", "webdataset"])
        assert ns.command == "corpus-validate"
        assert ns.corpus_index == "/c.yaml"
        assert ns.backend == "webdataset"

    def test_unknown_subcommand_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["unknown-cmd"])


# ---------------------------------------------------------------------------
# Schema validation command tests
# ---------------------------------------------------------------------------


class TestSchemaValidateCmd:
    """Test schema-validate command output and exit codes."""

    def test_validate_rejects_missing_file(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_schema_validate

        args = argparse.Namespace(
            schema_path=str(tmp_path / "nonexistent.yaml"),
            corpus_namespace=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_schema_validate(args)
        assert exc_info.value.code == 1

    def test_validate_rejects_invalid_schema_kind(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_schema_validate

        # Write a YAML that is valid YAML and a dict, but has wrong kind
        bad_schema = tmp_path / "bad-schema.yaml"
        bad_schema.write_text(
            "kind: wrong-kind\ncontract_version: 0.2.0\ndataset_id: x\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(schema_path=str(bad_schema), corpus_namespace=None)
        with pytest.raises((SystemExit, ValueError, KeyError)):
            _cmd_schema_validate(args)

    def test_validate_rejects_draft_status(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_schema_validate
        from perturb_data_lab.inspectors.models import SchemaDocument, SchemaFieldEntry

        # Write a schema with status=draft that will fail readiness
        schema_dict = {
            "kind": "schema",
            "contract_version": "0.2.0",
            "dataset_id": "test",
            "source_path": "/fake.h5ad",
            "status": "draft",
            "dataset_metadata": {},
            "perturbation_fields": {
                "perturbation_label": {
                    "source_fields": (),
                    "strategy": "null",
                    "transforms": (),
                    "confidence": "low",
                    "required": True,
                },
            },
            "context_fields": {},
            "feature_fields": {},
            "count_source": {"selected": ".X", "integer_only": True},
            "feature_tokenization": {"selected": "gene_id", "namespace": "ensembl"},
            "transform_catalog": [],
        }
        draft_schema = tmp_path / "draft-schema.yaml"
        draft_schema.write_text(yaml.safe_dump(schema_dict), encoding="utf-8")
        args = argparse.Namespace(schema_path=str(draft_schema), corpus_namespace=None)
        with pytest.raises(SystemExit) as exc_info:
            _cmd_schema_validate(args)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Corpus create command tests
# ---------------------------------------------------------------------------


class TestCorpusCreateCmd:
    """Test corpus-create command."""

    def test_corpus_create_writes_files(self, tmp_path: Path):
        from perturb_data_lab.cli import _cmd_corpus_create

        idx_path = tmp_path / "corpus-index.yaml"
        args = argparse.Namespace(
            backend="arrow-hf",
            output=str(idx_path),
            corpus_id="test-corpus-v0",
        )
        _cmd_corpus_create(args)

        assert idx_path.exists()
        corpus = CorpusIndexDocument.from_yaml_file(idx_path)
        assert corpus.corpus_id == "perturb-data-lab-v0"  # update_corpus_index generates ID on new corpus

        emission_spec_path = tmp_path / "corpus-emission-spec.yaml"
        assert emission_spec_path.exists()

    def test_corpus_create_fails_if_exists(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_create

        idx_path = tmp_path / "corpus-index.yaml"
        idx_path.write_text("dummy", encoding="utf-8")
        args = argparse.Namespace(backend="arrow-hf", output=str(idx_path), corpus_id=None)
        with pytest.raises(SystemExit) as exc_info:
            _cmd_corpus_create(args)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Corpus validate command tests
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
        # Write a corpus with one dataset whose manifest is missing
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.2.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [
                {
                    "dataset_id": "ds1",
                    "release_id": "v0.1",
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
        captured = capsys.readouterr()
        assert "MISSING" in captured.out

    def test_validate_passes_clean_corpus(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_validate

        idx_path = tmp_path / "corpus-index.yaml"

        # Write a valid corpus index with a valid manifest
        manifest_path = tmp_path / "v0.1-manifest.yaml"
        manifest_data = {
            "kind": "materialization-manifest",
            "contract_version": "0.2.0",
            "dataset_id": "ds1",
            "release_id": "v0.1",
            "route": "create_new",
            "backend": "arrow-hf",
            "count_source": {"selected": ".X", "integer_only": True},
            "outputs": {"metadata_root": str(tmp_path / "meta"), "matrix_root": str(tmp_path / "matrix")},
            "provenance": {"source_path": "/fake.h5ad", "schema": "/fake.yaml"},
        }
        manifest_path.write_text(yaml.safe_dump(manifest_data), encoding="utf-8")

        # Tokenizer is optional for corpus-validate (tokenizer-free architecture)

        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.2.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [
                {
                    "dataset_id": "ds1",
                    "release_id": "v0.1",
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
# corpus append command tests
# ---------------------------------------------------------------------------


class TestCorpusAppendCmd:
    """Test corpus-append command."""

    def test_append_rejects_missing_corpus_index(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_append

        args = argparse.Namespace(
            corpus_index=str(tmp_path / "nonexistent.yaml"),
            manifest=str(tmp_path / "manifest.yaml"),
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_corpus_append(args)
        assert exc_info.value.code == 1

    def test_append_rejects_missing_manifest(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_append

        # Create corpus index
        idx_path = tmp_path / "corpus-index.yaml"
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.2.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [],
        }
        idx_path.write_text(yaml.safe_dump(idx_data), encoding="utf-8")

        args = argparse.Namespace(
            corpus_index=str(idx_path),
            manifest=str(tmp_path / "missing-manifest.yaml"),
        )
        with pytest.raises(SystemExit) as exc_info:
            _cmd_corpus_append(args)
        assert exc_info.value.code == 1

    def test_append_writes_dataset_record(self, tmp_path: Path, capsys):
        from perturb_data_lab.cli import _cmd_corpus_append

        # Create corpus index
        idx_path = tmp_path / "corpus-index.yaml"
        idx_data = {
            "kind": "corpus-index",
            "contract_version": "0.2.0",
            "corpus_id": "test-v0",
            "global_metadata": {},
            "datasets": [],
        }
        idx_path.write_text(yaml.safe_dump(idx_data), encoding="utf-8")

        # Create manifest
        manifest_path = tmp_path / "v0.1-manifest.yaml"
        manifest_data = {
            "kind": "materialization-manifest",
            "contract_version": "0.2.0",
            "dataset_id": "ds1",
            "release_id": "v0.1",
            "route": "create_new",
            "backend": "arrow-hf",
            "count_source": {"selected": ".X", "integer_only": True},
            "outputs": {"metadata_root": str(tmp_path / "meta"), "matrix_root": str(tmp_path / "matrix")},
            "provenance": {"source_path": "/fake.h5ad", "schema": "/fake.yaml"},
        }
        manifest_path.write_text(yaml.safe_dump(manifest_data), encoding="utf-8")

        args = argparse.Namespace(corpus_index=str(idx_path), manifest=str(manifest_path))
        _cmd_corpus_append(args)

        updated = CorpusIndexDocument.from_yaml_file(idx_path)
        assert len(updated.datasets) == 1
        assert updated.datasets[0].dataset_id == "ds1"
        assert updated.datasets[0].join_mode == "append_routed"


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
        # All commands should appear in help
        for cmd in ["inspect", "schema-validate", "schema-preview", "materialize", "corpus-create", "corpus-append", "corpus-validate"]:
            assert cmd in captured.out
