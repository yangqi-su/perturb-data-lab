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
