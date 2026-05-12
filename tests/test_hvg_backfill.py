"""Focused tests for existing-corpus HVG backfill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from perturb_data_lab.cli import _cmd_backfill_hvg, build_parser
from perturb_data_lab.materializers import backfill_hvg_rankings_for_corpus
from perturb_data_lab.materializers.models import (
    CountSourceSpec,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
)


def _write_feature_meta(meta_root: Path, feature_ids: list[str]) -> Path:
    path = meta_root / "raw-var.parquet"
    table = pa.table(
        {
            "origin_index": pa.array(list(range(len(feature_ids))), type=pa.int32()),
            "feature_id": pa.array(feature_ids, type=pa.string()),
            "raw_var": pa.array(
                [f'{{"feature_id": "{feature_id}"}}' for feature_id in feature_ids],
                type=pa.string(),
            ),
        }
    )
    pq.write_table(table, path)
    return path


def _write_manifest(
    *,
    corpus_root: Path,
    dataset_id: str,
    dataset_index: int,
    cell_count: int,
    feature_count: int,
    manifest_path: Path,
    raw_feature_meta_path: str,
) -> None:
    manifest = MaterializationManifest(
        kind="materialization-manifest",
        contract_version="0.3.0",
        dataset_id=dataset_id,
        route="create_new" if dataset_index == 0 else "append_routed",
        backend="lance",
        topology="aggregate",
        count_source=CountSourceSpec(selected=".X", integer_only=True),
        outputs=OutputRoots(
            metadata_root=str(corpus_root / "meta" / dataset_id),
            matrix_root=str(corpus_root / "matrix"),
        ),
        provenance=ProvenanceSpec(
            source_path=f"/fake/{dataset_id}.h5ad",
            review_bundle=f"/fake/{dataset_id}-summary.yaml",
        ),
        raw_feature_meta_path=raw_feature_meta_path,
        hvg_sidecar_path=str(corpus_root / "meta" / dataset_id / "hvg_sidecar"),
        cell_count=cell_count,
        feature_count=feature_count,
    )
    manifest.write_yaml(manifest_path)


def _build_mock_aggregate_lance_corpus(corpus_root: Path) -> None:
    corpus_root.mkdir(parents=True, exist_ok=True)
    matrix_root = corpus_root / "matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "dataset_id": "ds_a",
            "dataset_index": 0,
            "global_start": 0,
            "global_end": 3,
            "feature_ids": ["ga", "gb", "gc"],
            "rows": [
                {"expressed_gene_indices": [0, 2], "expression_counts": [2, 1]},
                {"expressed_gene_indices": [1], "expression_counts": [4]},
                {"expressed_gene_indices": [0, 1], "expression_counts": [1, 1]},
            ],
        },
        {
            "dataset_id": "ds_b",
            "dataset_index": 1,
            "global_start": 3,
            "global_end": 5,
            "feature_ids": ["ha", "hb", "hc"],
            "rows": [
                {"expressed_gene_indices": [0], "expression_counts": [3]},
                {"expressed_gene_indices": [1, 2], "expression_counts": [1, 2]},
            ],
        },
    ]

    index_payload = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "corpus_id": "mock-corpus",
        "global_metadata": {"backend": "lance", "topology": "aggregate"},
        "datasets": [],
    }
    lance_rows = []
    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        meta_root = corpus_root / "meta" / dataset_id
        meta_root.mkdir(parents=True, exist_ok=True)
        raw_var_path = _write_feature_meta(meta_root, dataset["feature_ids"])
        manifest_path = meta_root / "materialization-manifest.yaml"
        raw_feature_meta_path = str(raw_var_path)
        if dataset_id == "ds_a":
            raw_feature_meta_path = "/nonexistent/old-plan/raw-var.parquet"
        _write_manifest(
            corpus_root=corpus_root,
            dataset_id=dataset_id,
            dataset_index=dataset["dataset_index"],
            cell_count=dataset["global_end"] - dataset["global_start"],
            feature_count=len(dataset["feature_ids"]),
            manifest_path=manifest_path,
            raw_feature_meta_path=raw_feature_meta_path,
        )
        index_payload["datasets"].append(
            {
                "dataset_id": dataset_id,
                "join_mode": "create_new" if dataset["dataset_index"] == 0 else "append_routed",
                "manifest_path": f"meta/{dataset_id}/materialization-manifest.yaml",
                "dataset_index": dataset["dataset_index"],
                "cell_count": dataset["global_end"] - dataset["global_start"],
                "global_start": dataset["global_start"],
                "global_end": dataset["global_end"],
            }
        )
        lance_rows.extend(dataset["rows"])

    (corpus_root / "corpus-index.yaml").write_text(
        yaml.safe_dump(index_payload),
        encoding="utf-8",
    )
    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [row["expressed_gene_indices"] for row in lance_rows],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [row["expression_counts"] for row in lance_rows],
                type=pa.list_(pa.int32()),
            ),
        }
    )
    lance.write_dataset(
        table,
        str(matrix_root / "aggregated-cells.lance"),
        mode="overwrite",
    )


class TestHVGBackfillParser:
    def test_backfill_hvg_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(["backfill-hvg", "--corpus-root", "/corpus"])
        assert ns.command == "backfill-hvg"
        assert ns.corpus_root == "/corpus"
        assert ns.chunk_rows == 50_000
        assert ns.update_manifests is True


class TestHVGBackfill:
    def test_backfill_recomputes_dataset_local_hvg_artifacts(self, tmp_path: Path):
        _build_mock_aggregate_lance_corpus(tmp_path)

        summary = backfill_hvg_rankings_for_corpus(
            tmp_path,
            chunk_rows=2,
            n_hvg=2,
        )

        assert summary.dataset_count == 2
        assert summary.topology == "aggregate"
        assert [dataset.dataset_id for dataset in summary.datasets] == ["ds_a", "ds_b"]

        for dataset in summary.datasets:
            output_path = Path(dataset.output_path)
            manifest_path = Path(dataset.manifest_path)
            table = pq.read_table(output_path).to_pandas()
            manifest = MaterializationManifest.from_yaml_file(manifest_path)

            assert output_path == tmp_path / "meta" / dataset.dataset_id / "hvg.parquet"
            assert dataset.row_count == dataset.feature_count == 3
            assert table["feature_id"].tolist()
            assert manifest.hvg_ranking_path == str(output_path)
            assert manifest.default_n_hvg == 2
            assert dataset.manifest_updated is True
            assert dataset.sha256

    def test_backfill_cli_writes_summary_json(self, tmp_path: Path):
        _build_mock_aggregate_lance_corpus(tmp_path)
        summary_json = tmp_path / "summary.json"

        args = argparse.Namespace(
            corpus_root=str(tmp_path),
            dataset_id=None,
            output_root=None,
            chunk_rows=2,
            n_hvg=2,
            summary_json=str(summary_json),
            overwrite=False,
            update_manifests=True,
        )
        _cmd_backfill_hvg(args)

        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        assert payload["dataset_count"] == 2
        assert sorted(dataset["dataset_id"] for dataset in payload["datasets"]) == ["ds_a", "ds_b"]
