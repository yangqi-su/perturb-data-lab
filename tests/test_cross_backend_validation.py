"""Regression tests for the shared cross-backend public-API contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import yaml

from perturb_data_lab.loaders.validation import validate_cross_backend_contract
from perturb_data_lab.materializers.backends import build_backend_fn
from perturb_data_lab.materializers.chunk_translation import ChunkBundle
import perturb_data_lab.loaders.validation as validation_mod


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

    with open(root / "corpus-index.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "kind": "corpus-index",
                "contract_version": "0.3.0",
                "corpus_id": f"mock-{backend}-corpus",
                "global_metadata": {"backend": backend, "topology": "federated"},
                "datasets": datasets,
            },
            fh,
        )


def test_validate_cross_backend_contract_reports_all_federated_backends(tmp_path: Path) -> None:
    _require_datasets()
    lance_root = tmp_path / "lance"
    zarr_root = tmp_path / "zarr"
    arrow_root = tmp_path / "arrow"
    hf_root = tmp_path / "hf"
    parquet_root = tmp_path / "parquet"
    _build_mock_corpus(lance_root, backend="lance")
    _build_mock_corpus(zarr_root, backend="zarr")
    _build_mock_corpus(arrow_root, backend="arrow-ipc")
    _build_mock_corpus(hf_root, backend="hf-datasets")
    _build_mock_corpus(parquet_root, backend="arrow-parquet")

    report = validate_cross_backend_contract(
        {
            "lance": lance_root,
            "zarr": zarr_root,
            "arrow-ipc": arrow_root,
            "hf-datasets": hf_root,
            "arrow-parquet": parquet_root,
        },
        sample_indices=[0, 3, 1, 2],
        metadata_columns=["perturb_label"],
        loader_batch_size=2,
        seq_len=LOADER_SEQ_LEN,
    )

    assert report["comparisons"]["baseline_backend"] == "lance"
    assert list(report["backends"]) == ["lance", "zarr", "arrow_ipc", "hf_datasets", "parquet"]
    for backend_report in report["backends"].values():
        assert backend_report["artifact_checks"]["status"] == "success"
        assert backend_report["artifact_checks"]["dataset_count"] == 2
        assert backend_report["corpus_index"]["total_cells"] == 4
        assert backend_report["load_corpus"]["status"] == "success"
    assert set(report["comparisons"]["pairs"]) == {
        "zarr_vs_lance",
        "arrow_ipc_vs_lance",
        "hf_datasets_vs_lance",
        "parquet_vs_lance",
    }
    for pair_report in report["comparisons"]["pairs"].values():
        assert pair_report["status"] == "success"
        assert pair_report["layers"] == {
            "corpus_index": "success",
            "read_expression": "success",
            "dataset": "success",
            "take_metadata": "success",
            "inspect_batch": "success",
            "loader": "success",
        }


def test_validate_cross_backend_contract_failure_mentions_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lance_root = tmp_path / "lance"
    arrow_root = tmp_path / "arrow"
    _build_mock_corpus(lance_root, backend="lance")
    _build_mock_corpus(arrow_root, backend="arrow-ipc")

    original_load_corpus = validation_mod.load_corpus

    def _patched_load_corpus(corpus_root: str | Path, *, use_canonical: bool = True):
        corpus = original_load_corpus(corpus_root, use_canonical=use_canonical)
        if Path(corpus_root).resolve() == arrow_root.resolve():
            original_inspect_batch = corpus.inspect_batch

            def _broken_inspect_batch(indices, *, metadata_columns=None):
                batch = original_inspect_batch(indices, metadata_columns=metadata_columns)
                broken = dict(batch)
                broken["dataset_index"] = batch["dataset_index"] + 1
                return broken

            corpus.inspect_batch = _broken_inspect_batch  # type: ignore[method-assign]
        return corpus

    monkeypatch.setattr(validation_mod, "load_corpus", _patched_load_corpus)

    with pytest.raises(AssertionError, match=r"arrow_ipc vs lance failed at inspect_batch"):
        validate_cross_backend_contract(
            {
                "lance": lance_root,
                "arrow-ipc": arrow_root,
            },
            sample_indices=[0, 3, 1, 2],
            metadata_columns=["perturb_label"],
            loader_batch_size=2,
            seq_len=LOADER_SEQ_LEN,
        )
