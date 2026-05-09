"""Focused tests for the experimental federated HF datasets backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import yaml

from perturb_data_lab.loaders.corpus_loader import load_corpus
from perturb_data_lab.loaders.expression import HfDatasetsDatasetEntry
from perturb_data_lab.materializers.backends import build_backend_fn
from perturb_data_lab.materializers.chunk_translation import ChunkBundle


N_GENES = 6
LOADER_SEQ_LEN = 4
DATASET_ROWS: dict[str, list[tuple[list[int], list[int]]]] = {
    "mock_00": [([0, 2], [5, 7]), ([1], [3])],
    "mock_01": [([3, 4], [9, 1]), ([0, 4, 5], [2, 4, 6])],
}
SIZE_FACTORS: dict[str, list[float]] = {
    "mock_00": [1.0, 1.5],
    "mock_01": [0.75, 2.0],
}


def _require_datasets() -> None:
    pytest.importorskip("datasets")


def _make_var_df(dataset_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_index": [str(i) for i in range(N_GENES)],
            "gene_id": [f"ENSG_{dataset_id}_{i:05d}" for i in range(N_GENES)],
            "canonical_gene_id": [f"GENE_{i:05d}" for i in range(N_GENES)],
            "global_id": [str(i) for i in range(N_GENES)],
        }
    )


def _make_obs_df(dataset_id: str, dataset_index: int, size_factors: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cell_id": [f"{dataset_id}_cell_{i:04d}" for i in range(len(size_factors))],
            "dataset_id": [dataset_id] * len(size_factors),
            "size_factor": size_factors,
            "perturb_label": [f"{dataset_id}_perturb_{i}" for i in range(len(size_factors))],
            "cell_line_or_type": ["K562"] * len(size_factors),
            "dataset_index": [dataset_index] * len(size_factors),
        }
    )


def _bundle_for_rows(global_start: int, rows: list[tuple[list[int], list[int]]]) -> ChunkBundle:
    indptr = [0]
    gene_indices: list[int] = []
    counts: list[int] = []
    row_sums: list[float] = []
    for genes, values in rows:
        gene_indices.extend(genes)
        counts.extend(values)
        indptr.append(indptr[-1] + len(genes))
        row_sums.append(float(sum(values)))

    return ChunkBundle(
        table=pa.table(
            {
                "global_row_index": pa.array(
                    np.arange(global_start, global_start + len(rows), dtype=np.int64),
                    type=pa.int64(),
                ),
                "expressed_gene_indices": pa.array(rows and [genes for genes, _ in rows] or [], type=pa.list_(pa.int32())),
                "expression_counts": pa.array(rows and [values for _, values in rows] or [], type=pa.list_(pa.int32())),
            }
        ),
        row_sums=np.asarray(row_sums, dtype=np.float64),
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(gene_indices, dtype=np.int32),
        counts=np.asarray(counts, dtype=np.int32),
        row_count=len(rows),
    )


def _write_matrix_backend(
    *,
    backend: str,
    matrix_root: Path,
    dataset_id: str,
    global_start: int,
    rows: list[tuple[list[int], list[int]]],
) -> None:
    writer = build_backend_fn(backend, "federated")
    writer_state = None
    chunk_start = 0
    chunks = [rows[:1], rows[1:]] if len(rows) > 1 else [rows]
    for chunk_index, chunk_rows in enumerate(chunks):
        if not chunk_rows:
            continue
        _, writer_state = writer(
            bundle=_bundle_for_rows(global_start + chunk_start, chunk_rows),
            dataset_id=dataset_id,
            matrix_root=matrix_root,
            _writer_state=writer_state,
            _is_last_chunk=chunk_index == len(chunks) - 1,
        )
        chunk_start += len(chunk_rows)


def _build_mock_corpus(root: Path, *, backend: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    datasets = []
    global_start = 0
    for dataset_index, dataset_id in enumerate(DATASET_ROWS):
        rows = DATASET_ROWS[dataset_id]
        cell_count = len(rows)
        dataset_root = root / dataset_id
        meta_root = dataset_root / "meta" / "canonical_meta"
        matrix_root = dataset_root / "matrix"
        meta_root.mkdir(parents=True, exist_ok=True)
        matrix_root.mkdir(parents=True, exist_ok=True)

        _make_obs_df(dataset_id, dataset_index, SIZE_FACTORS[dataset_id]).write_parquet(
            str(meta_root / "canonical-obs.parquet")
        )
        _make_var_df(dataset_id).write_parquet(str(meta_root / "canonical-var.parquet"))
        _write_matrix_backend(
            backend=backend,
            matrix_root=matrix_root,
            dataset_id=dataset_id,
            global_start=global_start,
            rows=rows,
        )

        datasets.append(
            {
                "dataset_id": dataset_id,
                "join_mode": "create_new" if dataset_index == 0 else "append_routed",
                "dataset_index": dataset_index,
                "cell_count": cell_count,
                "global_start": global_start,
                "global_end": global_start + cell_count,
            }
        )
        global_start += cell_count

    with open(root / "corpus-index.yaml", "w") as fh:
        yaml.safe_dump(
            {
                "kind": "corpus-index",
                "contract_version": "0.3.0",
                "global_metadata": {"backend": backend, "topology": "federated"},
                "datasets": datasets,
            },
            fh,
        )


def _expected_expression(indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_rows = [row for rows in DATASET_ROWS.values() for row in rows]
    row_offsets = [0]
    genes: list[int] = []
    counts: list[int] = []
    for index in indices:
        row_genes, row_counts = flat_rows[index]
        genes.extend(row_genes)
        counts.extend(row_counts)
        row_offsets.append(row_offsets[-1] + len(row_genes))
    return (
        np.asarray(row_offsets, dtype=np.int64),
        np.asarray(genes, dtype=np.int32),
        np.asarray(counts, dtype=np.int32),
    )


def test_hf_datasets_backend_is_registered() -> None:
    writer = build_backend_fn("hf-datasets", "federated")
    assert callable(writer)


def test_load_corpus_supports_federated_hf_datasets(tmp_path: Path) -> None:
    _require_datasets()
    _build_mock_corpus(tmp_path, backend="hf-datasets")

    corpus = load_corpus(str(tmp_path))

    assert corpus.backend == "hf_datasets"
    assert corpus.topology == "federated"
    assert len(corpus.dataset_entries) == 2
    for entry in corpus.dataset_entries:
        assert isinstance(entry, HfDatasetsDatasetEntry)
        assert Path(str(entry.dataset_path)).is_dir()
        assert str(entry.dataset_path).endswith(f"{entry.dataset_id}-hf-dataset")

    indices = [3, 0, 1]
    batch = corpus.read_expression(indices)
    expected_offsets, expected_genes, expected_counts = _expected_expression(indices)
    np.testing.assert_array_equal(batch.global_row_index, np.asarray(indices, dtype=np.int64))
    np.testing.assert_array_equal(batch.row_offsets, expected_offsets)
    np.testing.assert_array_equal(batch.expressed_gene_indices, expected_genes)
    np.testing.assert_array_equal(batch.expression_counts, expected_counts)

    inspected = corpus.inspect_batch(indices, metadata_columns=["perturb_label"])
    np.testing.assert_array_equal(inspected["global_row_index"], np.asarray(indices, dtype=np.int64))
    assert list(inspected["meta_columns"]["perturb_label"]) == [
        "mock_01_perturb_1",
        "mock_00_perturb_0",
        "mock_00_perturb_1",
    ]


def test_hf_public_api_matches_arrow_ipc_on_tiny_corpus(tmp_path: Path) -> None:
    _require_datasets()
    hf_root = tmp_path / "hf"
    arrow_root = tmp_path / "arrow"
    _build_mock_corpus(hf_root, backend="hf-datasets")
    _build_mock_corpus(arrow_root, backend="arrow-ipc")

    hf_corpus = load_corpus(str(hf_root))
    arrow_corpus = load_corpus(str(arrow_root))
    indices = [0, 3, 1, 2]

    hf_expr = hf_corpus.read_expression(indices)
    arrow_expr = arrow_corpus.read_expression(indices)
    np.testing.assert_array_equal(hf_expr.global_row_index, arrow_expr.global_row_index)
    np.testing.assert_array_equal(hf_expr.row_offsets, arrow_expr.row_offsets)
    np.testing.assert_array_equal(hf_expr.expressed_gene_indices, arrow_expr.expressed_gene_indices)
    np.testing.assert_array_equal(hf_expr.expression_counts, arrow_expr.expression_counts)

    hf_inspect = hf_corpus.inspect_batch(indices, metadata_columns=["perturb_label"])
    arrow_inspect = arrow_corpus.inspect_batch(indices, metadata_columns=["perturb_label"])
    np.testing.assert_array_equal(hf_inspect["global_row_index"], arrow_inspect["global_row_index"])
    np.testing.assert_array_equal(hf_inspect["dataset_index"], arrow_inspect["dataset_index"])
    np.testing.assert_array_equal(
        hf_inspect["meta_columns"]["perturb_label"],
        arrow_inspect["meta_columns"]["perturb_label"],
    )

    hf_corpus.set_sampler(batch_size=2, seed=7)
    arrow_corpus.set_sampler(batch_size=2, seed=7)
    hf_batch = next(hf_corpus.loader(processing="cpu", seq_len=LOADER_SEQ_LEN, metadata_columns=["perturb_label"]))
    arrow_batch = next(arrow_corpus.loader(processing="cpu", seq_len=LOADER_SEQ_LEN, metadata_columns=["perturb_label"]))
    np.testing.assert_array_equal(hf_batch["global_row_index"], arrow_batch["global_row_index"])
    np.testing.assert_array_equal(hf_batch["dataset_index"], arrow_batch["dataset_index"])
    np.testing.assert_array_equal(
        hf_batch["meta_columns"]["perturb_label"],
        arrow_batch["meta_columns"]["perturb_label"],
    )


def test_hf_load_corpus_dependency_error_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from perturb_data_lab.loaders import expression as expression_mod

    dataset_root = tmp_path / "mock_00"
    meta_root = dataset_root / "meta" / "canonical_meta"
    matrix_root = dataset_root / "matrix" / "mock_00-hf-dataset"
    meta_root.mkdir(parents=True, exist_ok=True)
    matrix_root.mkdir(parents=True, exist_ok=True)
    _make_obs_df("mock_00", 0, [1.0]).write_parquet(str(meta_root / "canonical-obs.parquet"))
    _make_var_df("mock_00").write_parquet(str(meta_root / "canonical-var.parquet"))
    with open(tmp_path / "corpus-index.yaml", "w") as fh:
        yaml.safe_dump(
            {
                "kind": "corpus-index",
                "contract_version": "0.3.0",
                "global_metadata": {"backend": "hf-datasets", "topology": "federated"},
                "datasets": [
                    {
                        "dataset_id": "mock_00",
                        "join_mode": "create_new",
                        "dataset_index": 0,
                        "cell_count": 1,
                        "global_start": 0,
                        "global_end": 1,
                    }
                ],
            },
            fh,
        )

    def _boom():
        raise ImportError(
            "hf_datasets backend requires the optional 'datasets' package; "
            "install perturb-data-lab[hf-datasets] or pip install datasets"
        )

    monkeypatch.setattr(expression_mod, "_import_hf_datasets", _boom)
    with pytest.raises(ImportError, match="optional 'datasets' package"):
        load_corpus(str(tmp_path))
