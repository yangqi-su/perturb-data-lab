"""Focused tests for current-corpus HVG recalculation."""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from perturb_data_lab.cli import _cmd_recalc_hvg, build_parser
from perturb_data_lab.loaders import load_corpus
from perturb_data_lab.materializers.models import (
    CountSourceSpec,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
)
from perturb_data_lab.pp import calculate_hvgs, recalc_hvg


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


def _write_canonical_obs(
    canonical_meta_root: Path,
    *,
    dataset_id: str,
    global_start: int,
    cell_count: int,
) -> None:
    canonical_meta_root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "global_row_index": np.arange(global_start, global_start + cell_count, dtype=np.int64),
            "cell_id": [f"{dataset_id}_cell_{i}" for i in range(cell_count)],
            "dataset_id": [dataset_id] * cell_count,
            "local_row_index": np.arange(cell_count, dtype=np.int64),
            "size_factor": np.ones(cell_count, dtype=np.float64),
            "perturb_label": ["control"] * cell_count,
        }
    ).write_parquet(canonical_meta_root / "canonical-obs.parquet")


def _write_canonical_var(canonical_meta_root: Path, feature_ids: list[str]) -> None:
    canonical_meta_root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "origin_index": np.arange(len(feature_ids), dtype=np.int32),
            "gene_id": feature_ids,
            "canonical_gene_id": feature_ids,
            "global_id": np.arange(len(feature_ids), dtype=np.int32),
        }
    ).write_parquet(canonical_meta_root / "canonical-var.parquet")


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
        canonical_meta_root = meta_root / "canonical_meta"
        raw_var_path = _write_feature_meta(meta_root, dataset["feature_ids"])
        _write_canonical_obs(
            canonical_meta_root,
            dataset_id=dataset_id,
            global_start=dataset["global_start"],
            cell_count=dataset["global_end"] - dataset["global_start"],
        )
        _write_canonical_var(canonical_meta_root, dataset["feature_ids"])
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


class TestRecalcHvgParser:
    def test_recalc_hvg_subcommand(self):
        parser = build_parser()
        ns = parser.parse_args(["recalc-hvg", "--corpus-root", "/corpus"])
        assert ns.command == "recalc-hvg"
        assert ns.corpus_root == "/corpus"
        assert ns.batch_size == 1024
        assert ns.update_manifests is True


class TestRecalcHvg:
    def test_recalc_writes_dataset_local_hvg_artifacts(self, tmp_path: Path):
        _build_mock_aggregate_lance_corpus(tmp_path)

        summary = recalc_hvg(
            tmp_path,
            batch_size=2,
            n_hvg=2,
        )

        assert summary.dataset_count == 2
        assert [dataset.dataset_id for dataset in summary.datasets] == ["ds_a", "ds_b"]

        for dataset in summary.datasets:
            output_path = Path(dataset.output_path)
            manifest_path = output_path.parent / "materialization-manifest.yaml"
            table = pq.read_table(output_path).to_pandas()
            manifest = MaterializationManifest.from_yaml_file(manifest_path)

            assert output_path == tmp_path / "meta" / dataset.dataset_id / "hvg.parquet"
            assert dataset.row_count == 3
            assert table.columns.tolist() == [
                "origin_index",
                "feature_id",
                "mean_log1p_expr",
                "variance_log1p_expr",
                "dispersion_log",
                "dispersion_norm",
                "hvg_rank",
                "selected_at_default_n_hvg",
            ]
            assert manifest.hvg_ranking_path == f"meta/{dataset.dataset_id}/hvg.parquet"
            assert manifest.default_n_hvg == 2
            assert dataset.manifest_updated is True

    def test_recalc_cli_writes_hvg_artifacts(self, tmp_path: Path):
        _build_mock_aggregate_lance_corpus(tmp_path)
        parser = build_parser()
        args = parser.parse_args([
            "recalc-hvg",
            "--corpus-root", str(tmp_path),
            "--batch-size", "2",
            "--n-hvg", "2",
        ])
        _cmd_recalc_hvg(args)

        assert (tmp_path / "meta" / "ds_a" / "hvg.parquet").exists()
        assert (tmp_path / "meta" / "ds_b" / "hvg.parquet").exists()

    def test_recalc_writes_same_ranked_metrics_as_pp_calculate_hvgs(self, tmp_path: Path):
        _build_mock_aggregate_lance_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        pp_hvgs = calculate_hvgs(corpus, batch_size=2, n_hvg=2).select(
            [
                "origin_index",
                "feature_id",
                "mean_log1p_expr",
                "variance_log1p_expr",
                "dispersion_log",
                "dispersion_norm",
                "hvg_rank",
                "selected_at_default_n_hvg",
            ]
        )

        summary = recalc_hvg(tmp_path, batch_size=2, n_hvg=2)
        written = pl.concat(
            [pl.read_parquet(dataset.output_path) for dataset in summary.datasets],
            rechunk=True,
        ).sort(["feature_id", "origin_index"])
        expected = pp_hvgs.sort(["feature_id", "origin_index"])

        assert written.to_dict(as_series=False) == expected.to_dict(as_series=False)
