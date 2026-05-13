"""Focused tests for the Phase 6 streamed ``perturb_data_lab.pp`` helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
from scipy import sparse
import yaml

from perturb_data_lab.loaders import load_corpus
from perturb_data_lab.materializers.backends import build_backend_fn
from perturb_data_lab.materializers.chunk_translation import ChunkBundle
from perturb_data_lab.pp import (
    build_pp_provenance,
    calculate_hvgs,
    calculate_lognorm_stats,
    iter_dataset_batches,
    log1p_size_factor_batch,
    prepare_pp_output,
    write_pp_provenance,
)

N_GENES = 100


def _make_obs_df(
    dataset_id: str,
    dataset_index: int,
    n_cells: int,
    global_start: int,
) -> pl.DataFrame:
    rows = []
    rng = np.random.RandomState(dataset_index * 10_000 + global_start + n_cells)
    for i in range(n_cells):
        rows.append(
            {
                "cell_id": f"{dataset_id}_cell_{i:04d}",
                "dataset_id": dataset_id,
                "dataset_index": dataset_index,
                "global_row_index": global_start + i,
                "local_row_index": i,
                "size_factor": round(float(rng.uniform(0.5, 2.0)), 4),
                "perturb_label": "CRISPR_control" if i % 3 == 0 else "CRISPR_geneX",
                "perturb_type": "CRISPR",
                "dose": "1.0",
                "dose_unit": "MOI",
                "timepoint": "7",
                "timepoint_unit": "days",
                "cell_context": "",
                "cell_line_or_type": "K562",
                "species": "Homo sapiens",
                "tissue": "bone marrow",
                "assay": "Perturb-seq",
                "condition": "control",
                "batch_id": f"batch_{i // 5}",
                "donor_id": "donor_01",
                "sex": "unknown",
                "disease_state": "healthy",
            }
        )
    return pl.DataFrame(rows)


def _make_var_df(dataset_id: str, n_genes: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_index": np.arange(n_genes, dtype=np.int32),
            "gene_id": [f"ENSG_{dataset_id}_{i:05d}" for i in range(n_genes)],
            "canonical_gene_id": [f"GENE_{i:05d}" for i in range(n_genes)],
            "global_id": np.arange(n_genes, dtype=np.int32),
        }
    )


def _make_expression_rows(n_cells: int) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.RandomState(42 + n_cells)
    for _ in range(n_cells):
        n_nonzero = rng.randint(5, min(20, N_GENES))
        gene_indices = rng.choice(N_GENES, size=n_nonzero, replace=False).astype(np.int32)
        gene_indices.sort()
        counts = rng.poisson(2, size=n_nonzero).astype(np.int32)
        rows.append(
            {
                "expressed_gene_indices": list(gene_indices),
                "expression_counts": list(counts),
            }
        )
    return rows


def _bundle_for_rows(global_start: int, rows: list[dict[str, Any]]) -> ChunkBundle:
    indptr = [0]
    gene_indices: list[int] = []
    counts: list[int] = []
    row_sums: list[float] = []
    expressed_gene_indices = [row["expressed_gene_indices"] for row in rows]
    expression_counts = [row["expression_counts"] for row in rows]
    for genes, values in zip(expressed_gene_indices, expression_counts, strict=True):
        gene_indices.extend(genes)
        counts.extend(values)
        indptr.append(indptr[-1] + len(genes))
        row_sums.append(float(np.sum(values)))

    return ChunkBundle(
        table=pa.table(
            {
                "global_row_index": pa.array(
                    np.arange(global_start, global_start + len(rows), dtype=np.int64),
                    type=pa.int64(),
                ),
                "expressed_gene_indices": pa.array(expressed_gene_indices, type=pa.list_(pa.int32())),
                "expression_counts": pa.array(expression_counts, type=pa.list_(pa.int32())),
            }
        ),
        row_sums=np.asarray(row_sums, dtype=np.float64),
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(gene_indices, dtype=np.int32),
        counts=np.asarray(counts, dtype=np.int32),
        row_count=len(rows),
    )


def _build_mock_federated_lance_corpus(corpus_root: Path) -> None:
    corpus_root.mkdir(parents=True, exist_ok=True)
    datasets = [
        {"dataset_id": "mock_00", "cell_count": 10, "global_start": 0, "global_end": 10, "dataset_index": 0},
        {"dataset_id": "mock_01", "cell_count": 15, "global_start": 10, "global_end": 25, "dataset_index": 1},
    ]
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {"backend": "lance", "topology": "federated"},
        "datasets": [
            {
                "dataset_id": ds["dataset_id"],
                "join_mode": "create_new" if i == 0 else "append_routed",
                "dataset_index": ds["dataset_index"],
                "cell_count": ds["cell_count"],
                "global_start": ds["global_start"],
                "global_end": ds["global_end"],
            }
            for i, ds in enumerate(datasets)
        ],
    }
    (corpus_root / "corpus-index.yaml").write_text(
        yaml.safe_dump(index_doc),
        encoding="utf-8",
    )

    writer = build_backend_fn("lance", "federated")
    for ds in datasets:
        dataset_id = ds["dataset_id"]
        meta_dir = corpus_root / dataset_id / "meta" / "canonical_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        _make_obs_df(
            dataset_id,
            ds["dataset_index"],
            ds["cell_count"],
            ds["global_start"],
        ).write_parquet(meta_dir / "canonical-obs.parquet")
        _make_var_df(dataset_id, N_GENES).write_parquet(meta_dir / "canonical-var.parquet")

        matrix_dir = corpus_root / dataset_id / "matrix"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        writer(
            bundle=_bundle_for_rows(ds["global_start"], _make_expression_rows(ds["cell_count"])),
            dataset_id=dataset_id,
            matrix_root=matrix_dir,
            _writer_state=None,
            _is_last_chunk=True,
        )


def test_iter_dataset_batches_supports_single_dataset_scope(tmp_path: Path) -> None:
    _build_mock_federated_lance_corpus(tmp_path)
    corpus = load_corpus(str(tmp_path))

    batches = list(iter_dataset_batches(corpus, dataset_id="mock_00", batch_size=4))

    assert [batch.batch_size for batch in batches] == [4, 4, 2]
    assert {batch.dataset_id for batch in batches} == {"mock_00"}
    assert {batch.dataset_index for batch in batches} == {0}
    np.testing.assert_array_equal(
        np.concatenate([batch.global_row_index for batch in batches]),
        np.arange(10, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.local_row_index for batch in batches]),
        np.arange(10, dtype=np.int64),
    )

    first_batch = batches[0]
    assert sparse.isspmatrix_csr(first_batch.expression)
    assert first_batch.expression.shape == (4, N_GENES)
    assert first_batch.feature_context.n_features == N_GENES
    np.testing.assert_array_equal(
        first_batch.feature_context.local_to_global,
        np.arange(N_GENES, dtype=np.int32),
    )
    assert first_batch.feature_context.local_feature_ids[0] == "GENE_00000"

    lognorm = log1p_size_factor_batch(first_batch, dtype=np.float64)
    row_nonzero = np.diff(first_batch.expression.indptr)
    expected = np.log1p(
        first_batch.expression.data.astype(np.float64)
        / np.repeat(first_batch.size_factor.astype(np.float64), row_nonzero)
    )
    np.testing.assert_allclose(lognorm.data, expected)
    np.testing.assert_array_equal(lognorm.indices, first_batch.expression.indices)
    np.testing.assert_array_equal(lognorm.indptr, first_batch.expression.indptr)


def test_iter_dataset_batches_streams_all_datasets_without_crossing_boundaries(tmp_path: Path) -> None:
    _build_mock_federated_lance_corpus(tmp_path)
    corpus = load_corpus(str(tmp_path))

    batches = list(iter_dataset_batches(corpus, batch_size=7))

    assert [(batch.dataset_id, batch.batch_size) for batch in batches] == [
        ("mock_00", 7),
        ("mock_00", 3),
        ("mock_01", 7),
        ("mock_01", 7),
        ("mock_01", 1),
    ]
    np.testing.assert_array_equal(
        np.concatenate([batch.global_row_index for batch in batches]),
        np.arange(25, dtype=np.int64),
    )
    assert all(len(set(batch.global_row_index.tolist())) == batch.batch_size for batch in batches)


def test_pp_artifact_helpers_write_dataset_local_provenance(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _build_mock_federated_lance_corpus(corpus_root)
    corpus = load_corpus(str(corpus_root))

    output = prepare_pp_output(
        tmp_path / "pp-output",
        dataset_id="mock_01",
        artifact_name="lognorm-stats",
        suffix="parquet",
    )
    provenance = build_pp_provenance(
        corpus,
        operation="calculate_lognorm_stats",
        dataset_id="mock_01",
        parameters={"batch_size": 32},
    )

    assert output.artifact_path == tmp_path / "pp-output" / "mock_01" / "lognorm-stats.parquet"
    assert provenance["dataset_index"] == 1
    assert provenance["backend"] == "lance"
    assert provenance["topology"] == "federated"

    provenance_path = write_pp_provenance(
        output,
        corpus=corpus,
        operation="calculate_lognorm_stats",
        parameters={"batch_size": 32},
    )
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance_path == output.provenance_path
    assert payload["operation"] == "calculate_lognorm_stats"
    assert payload["dataset_id"] == "mock_01"
    assert payload["parameters"] == {"batch_size": 32}
    assert payload["extra"]["artifact_path"] == str(output.artifact_path)


def _dense_reference_lognorm_stats(corpus, dataset_id: str) -> dict[str, np.ndarray | int]:
    matrices = []
    for batch in iter_dataset_batches(corpus, dataset_id=dataset_id, batch_size=3):
        dense = batch.expression.toarray().astype(np.float64, copy=False)
        dense = np.log1p(dense / batch.size_factor[:, None].astype(np.float64, copy=False))
        matrices.append(dense)

    combined = np.vstack(matrices)
    return {
        "mean": combined.mean(axis=0),
        "var": combined.var(axis=0),
        "std": combined.std(axis=0),
        "n_obs": int(combined.shape[0]),
        "n_nonzero": np.count_nonzero(combined, axis=0).astype(np.int64, copy=False),
    }


def _dense_reference_hvgs(corpus, dataset_id: str) -> dict[str, np.ndarray | int]:
    matrices = []
    for batch in iter_dataset_batches(corpus, dataset_id=dataset_id, batch_size=3):
        matrices.append(batch.expression.toarray().astype(np.float64, copy=False))

    combined = np.vstack(matrices)
    log1p = np.log1p(combined)
    return {
        "mean": log1p.mean(axis=0),
        "var": log1p.var(axis=0, ddof=1),
        "n_cells_detected": np.count_nonzero(combined > 0, axis=0).astype(np.int64, copy=False),
        "n_obs": int(combined.shape[0]),
    }


def test_calculate_lognorm_stats_matches_dense_reference_for_single_dataset(
    tmp_path: Path,
) -> None:
    _build_mock_federated_lance_corpus(tmp_path)
    corpus = load_corpus(str(tmp_path))

    stats = calculate_lognorm_stats(corpus, dataset_id="mock_00", batch_size=4)
    reference = _dense_reference_lognorm_stats(corpus, "mock_00")

    assert stats.columns == [
        "dataset_id",
        "dataset_index",
        "gene_id",
        "global_feature_id",
        "mean_lognorm",
        "var_lognorm",
        "std_lognorm",
        "n_obs",
        "n_nonzero",
    ]
    assert stats.shape == (N_GENES, 9)
    assert stats["dataset_id"].unique().to_list() == ["mock_00"]
    assert stats["dataset_index"].unique().to_list() == [0]
    assert stats["n_obs"].unique().to_list() == [10]
    np.testing.assert_array_equal(
        stats["global_feature_id"].to_numpy(),
        np.arange(N_GENES, dtype=np.int64),
    )
    np.testing.assert_allclose(stats["mean_lognorm"].to_numpy(), reference["mean"])
    np.testing.assert_allclose(stats["var_lognorm"].to_numpy(), reference["var"])
    np.testing.assert_allclose(stats["std_lognorm"].to_numpy(), reference["std"])
    np.testing.assert_array_equal(stats["n_nonzero"].to_numpy(), reference["n_nonzero"])


def test_calculate_lognorm_stats_streams_all_datasets_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "corpus"
    _build_mock_federated_lance_corpus(corpus_root)
    corpus = load_corpus(str(corpus_root))
    output_dir = tmp_path / "pp-output"

    first = calculate_lognorm_stats(corpus, batch_size=6, output_dir=output_dir)
    second = calculate_lognorm_stats(
        corpus,
        batch_size=6,
        output_dir=output_dir,
        overwrite=True,
    )

    assert first.sort(["dataset_index", "global_feature_id"]).to_dict(as_series=False) == second.sort(
        ["dataset_index", "global_feature_id"]
    ).to_dict(as_series=False)
    assert first["dataset_id"].unique().sort().to_list() == ["mock_00", "mock_01"]

    for dataset_id, expected_n_obs in (("mock_00", 10), ("mock_01", 15)):
        reference = _dense_reference_lognorm_stats(corpus, dataset_id)
        subset = first.filter(pl.col("dataset_id") == dataset_id).sort("global_feature_id")
        np.testing.assert_allclose(subset["mean_lognorm"].to_numpy(), reference["mean"])
        np.testing.assert_allclose(subset["var_lognorm"].to_numpy(), reference["var"])
        np.testing.assert_allclose(subset["std_lognorm"].to_numpy(), reference["std"])
        np.testing.assert_array_equal(subset["n_nonzero"].to_numpy(), reference["n_nonzero"])
        assert subset["n_obs"].unique().to_list() == [expected_n_obs]

        stats_path = output_dir / dataset_id / "lognorm-stats.parquet"
        provenance_path = output_dir / dataset_id / "lognorm-stats.provenance.json"
        assert stats_path.exists()
        assert provenance_path.exists()
        assert pl.read_parquet(stats_path).sort("global_feature_id").to_dict(as_series=False) == subset.to_dict(
            as_series=False
        )

        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        assert payload["operation"] == "calculate_lognorm_stats"
        assert payload["dataset_id"] == dataset_id
        assert payload["parameters"]["dataset_scope"] == "dataset"
        assert payload["parameters"]["normalization"] == "log1p(count / size_factor)"
        assert payload["parameters"]["size_factor_source"] == "canonical_obs.size_factor"
        assert payload["parameters"]["variance_ddof"] == 0
        assert payload["extra"]["n_obs"] == expected_n_obs


def test_calculate_hvgs_matches_dense_reference_for_single_dataset(tmp_path: Path) -> None:
    _build_mock_federated_lance_corpus(tmp_path)
    corpus = load_corpus(str(tmp_path))

    hvgs = calculate_hvgs(corpus, dataset_id="mock_00", batch_size=4, n_hvg=7)
    reference = _dense_reference_hvgs(corpus, "mock_00")

    assert hvgs.columns == [
        "dataset_id",
        "dataset_index",
        "origin_index",
        "gene_id",
        "feature_id",
        "mean",
        "variance",
        "dispersion",
        "dispersion_log",
        "dispersion_norm",
        "hvg_rank",
        "is_hvg",
        "selected_at_default_n_hvg",
        "n_cells_detected",
        "mean_log1p_expr",
        "variance_log1p_expr",
        "global_feature_id",
    ]
    assert hvgs.shape == (N_GENES, 17)
    assert hvgs["dataset_id"].unique().to_list() == ["mock_00"]
    assert hvgs["dataset_index"].unique().to_list() == [0]
    np.testing.assert_array_equal(hvgs["origin_index"].to_numpy(), np.arange(N_GENES, dtype=np.int32))
    np.testing.assert_array_equal(hvgs["global_feature_id"].to_numpy(), np.arange(N_GENES, dtype=np.int64))
    np.testing.assert_allclose(hvgs["mean"].to_numpy(), reference["mean"])
    np.testing.assert_allclose(hvgs["variance"].to_numpy(), reference["var"])
    np.testing.assert_allclose(hvgs["mean_log1p_expr"].to_numpy(), reference["mean"])
    np.testing.assert_allclose(hvgs["variance_log1p_expr"].to_numpy(), reference["var"])
    np.testing.assert_array_equal(hvgs["n_cells_detected"].to_numpy(), reference["n_cells_detected"])
    np.testing.assert_array_equal(
        np.sort(hvgs["hvg_rank"].to_numpy()),
        np.arange(1, N_GENES + 1, dtype=np.int32),
    )
    assert hvgs["is_hvg"].sum() == 7
    assert hvgs["selected_at_default_n_hvg"].sum() == 7
