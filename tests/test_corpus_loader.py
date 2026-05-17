from __future__ import annotations

from pathlib import Path
from typing import Any

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from perturb_data_lab.loaders import (
    CorpusRandomBatchSampler,
    ContextBatchSampler,
    ExpressionBatchDataset,
    build_loader,
    collate_expression_batch,
)
from perturb_data_lab.loaders.corpus_loader import Corpus, load_corpus
from perturb_data_lab.loaders.validation import validate_corpus_structure


N_GENES = 8
LOADER_SEQ_LEN = 4

DATASETS = (
    {"dataset_id": "mock_00", "dataset_index": 0, "global_start": 0, "cell_count": 4},
    {"dataset_id": "mock_01", "dataset_index": 1, "global_start": 4, "cell_count": 5},
)


def _obs_frame(dataset_id: str, dataset_index: int, global_start: int, n_cells: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "global_row_index": np.arange(global_start, global_start + n_cells, dtype=np.int64),
            "cell_id": [f"{dataset_id}_cell_{idx}" for idx in range(n_cells)],
            "dataset_id": [dataset_id] * n_cells,
            "dataset_index": np.full(n_cells, dataset_index, dtype=np.int32),
            "local_row_index": np.arange(n_cells, dtype=np.int64),
            "size_factor": np.asarray([1.0 + 0.1 * idx for idx in range(n_cells)], dtype=np.float64),
            "perturb_label": ["ctrl" if idx % 2 == 0 else "treat" for idx in range(n_cells)],
            "perturb_type": ["CRISPR"] * n_cells,
            "dose": [None] * n_cells,
            "dose_unit": [None] * n_cells,
            "timepoint": [None] * n_cells,
            "timepoint_unit": [None] * n_cells,
            "cell_context": ["K562"] * n_cells,
            "cell_line_or_type": ["K562"] * n_cells,
            "species": ["Homo sapiens"] * n_cells,
            "tissue": ["bone marrow"] * n_cells,
            "assay": ["Perturb-seq"] * n_cells,
            "condition": ["mock"] * n_cells,
            "batch_id": [f"batch_{dataset_index}"] * n_cells,
            "donor_id": ["donor_0"] * n_cells,
            "sex": ["NA"] * n_cells,
            "disease_state": ["healthy"] * n_cells,
        }
    )


def _var_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_index": np.arange(N_GENES, dtype=np.int32),
            "gene_id": [f"ENSG{i:05d}" for i in range(N_GENES)],
            "canonical_gene_id": [f"GENE{i:05d}" for i in range(N_GENES)],
            "global_id": np.arange(N_GENES, dtype=np.int32),
        }
    )


def _expression_rows(n_cells: int, *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _ in range(n_cells):
        genes = np.sort(rng.choice(N_GENES, size=3, replace=False).astype(np.int32))
        counts = rng.integers(1, 5, size=3, dtype=np.int32)
        rows.append(
            {
                "expressed_gene_indices": genes.tolist(),
                "expression_counts": counts.tolist(),
            }
        )
    return rows


def _write_lance_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [row["expressed_gene_indices"] for row in rows],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [row["expression_counts"] for row in rows],
                type=pa.list_(pa.int32()),
            ),
        }
    )
    lance.write_dataset(table, str(path), mode="overwrite")


def _write_corpus_index(root: Path, *, topology: str) -> None:
    doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {"backend": "lance", "topology": topology},
        "datasets": [
            {
                "dataset_id": item["dataset_id"],
                "join_mode": "create_new" if idx == 0 else "append_routed",
                "dataset_index": item["dataset_index"],
                "cell_count": item["cell_count"],
                "global_start": item["global_start"],
                "global_end": item["global_start"] + item["cell_count"],
            }
            for idx, item in enumerate(DATASETS)
        ],
    }
    with open(root / "corpus-index.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle)


def _write_metadata(root: Path, *, topology: str) -> None:
    for item in DATASETS:
        if topology == "aggregate":
            meta_root = root / "meta" / item["dataset_id"] / "canonical_meta"
        else:
            meta_root = root / item["dataset_id"] / "meta" / "canonical_meta"
        meta_root.mkdir(parents=True, exist_ok=True)
        _obs_frame(
            item["dataset_id"],
            item["dataset_index"],
            item["global_start"],
            item["cell_count"],
        ).write_parquet(meta_root / "canonical-obs.parquet")
        _var_frame().write_parquet(meta_root / "canonical-var.parquet")


def _build_aggregate_lance_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_corpus_index(root, topology="aggregate")
    _write_metadata(root, topology="aggregate")
    rows: list[dict[str, Any]] = []
    for item in DATASETS:
        rows.extend(_expression_rows(item["cell_count"], seed=100 + item["dataset_index"]))
    _write_lance_rows(root / "matrix" / "aggregated-cells.lance", rows)


def _build_federated_lance_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_corpus_index(root, topology="federated")
    _write_metadata(root, topology="federated")
    for item in DATASETS:
        rows = _expression_rows(item["cell_count"], seed=200 + item["dataset_index"])
        _write_lance_rows(
            root / item["dataset_id"] / "matrix" / f"{item['dataset_id']}.lance",
            rows,
        )


def _assert_processed_batch(batch: dict[str, Any], *, batch_size: int) -> None:
    assert batch["batch_size"] == batch_size
    assert batch["seq_len"] == LOADER_SEQ_LEN
    for key in (
        "sampled_gene_ids",
        "sampled_counts",
        "valid_mask",
        "exact_match_mask",
        "dataset_index",
        "global_row_index",
    ):
        assert isinstance(batch[key], torch.Tensor)
    assert "row_offsets" not in batch
    assert "expressed_gene_indices" not in batch
    assert "expression_counts" not in batch


def test_load_corpus_builds_components_without_runtime_loader_state(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)

    corpus = load_corpus(tmp_path)

    assert isinstance(corpus, Corpus)
    assert corpus.topology == "aggregate"
    assert corpus.backend == "lance"
    assert len(corpus.metadata_index) == 9
    assert corpus.dataset_index_by_id == {"mock_00": 0, "mock_01": 1}
    assert corpus.feature_registry.global_vocab_size == N_GENES
    assert not hasattr(corpus, "loader")
    assert not hasattr(corpus, "dataset")
    assert not hasattr(corpus, "read_expression")
    assert not hasattr(corpus, "set_sampler")
    assert not hasattr(corpus, "select_obs_indices")


def test_expression_reader_and_take_metadata_use_global_rows(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    expression = corpus.expression_reader.read_expression_flat([0, 4, 8])
    metadata = corpus.take_metadata([0, 4, 8], columns=["dataset_id", "dataset_index"])

    np.testing.assert_array_equal(expression.global_row_index, [0, 4, 8])
    assert expression.batch_size == 3
    assert metadata["dataset_id"] == ("mock_00", "mock_01", "mock_01")
    np.testing.assert_array_equal(metadata["dataset_index"], [0, 1, 1])


def test_validate_corpus_structure_checks_aggregate_corpus(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)

    report = validate_corpus_structure(tmp_path, sample_n=4, seed=1)

    assert report["status"] == "success"
    assert report["backend"] == "lance"
    assert report["topology"] == "aggregate"
    assert report["dataset_count"] == 2
    assert report["total_rows"] == 9


def test_validate_corpus_structure_checks_federated_corpus(tmp_path: Path) -> None:
    _build_federated_lance_corpus(tmp_path)

    report = validate_corpus_structure(tmp_path, sample_n=4, seed=1)

    assert report["status"] == "success"
    assert report["topology"] == "federated"
    assert report["matrix"]["checked_layout"] == "federated"


def test_validate_corpus_structure_rejects_bad_ranges(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    index_path = tmp_path / "corpus-index.yaml"
    with open(index_path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    doc["datasets"][1]["global_start"] = 5
    with open(index_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle)

    with pytest.raises(AssertionError, match="global ranges"):
        validate_corpus_structure(tmp_path)


def test_expression_dataset_is_expression_only(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)
    dataset = ExpressionBatchDataset(corpus.expression_reader, total_rows=len(corpus.metadata_index))

    raw = dataset.__getitems__([0, 4, 8])[0]
    assert raw.batch_size == 3
    np.testing.assert_array_equal(raw.global_row_index, [0, 4, 8])

    sampler = CorpusRandomBatchSampler(
        metadata_index=corpus.metadata_index,
        batch_size=3,
        drop_last=False,
        seed=5,
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=collate_expression_batch,
                num_workers=0,
            )
        )
    )
    assert isinstance(batch["global_row_index"], torch.Tensor)
    assert "dataset_index" not in batch
    assert "size_factor" not in batch


def test_build_loader_attaches_metadata_from_metadata_index(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    batch = next(
        build_loader(
            corpus,
            batch_size=3,
            seq_len=LOADER_SEQ_LEN,
            seed=2,
            device="cpu",
            metadata_columns=["perturb_label", "size_factor"],
        )
    )

    _assert_processed_batch(batch, batch_size=3)
    expected = corpus.take_metadata(
        batch["global_row_index"].cpu().numpy(),
        columns=["dataset_index", "size_factor", "perturb_label"],
    )
    np.testing.assert_array_equal(batch["dataset_index"].cpu().numpy(), expected["dataset_index"])
    np.testing.assert_allclose(batch["size_factor"].cpu().numpy(), expected["size_factor"])
    assert batch["meta_columns"]["perturb_label"] == expected["perturb_label"]
    np.testing.assert_allclose(batch["meta_columns"]["size_factor"], expected["size_factor"])


def test_build_loader_respects_context_sampler_and_row_indices(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    batch = next(
        build_loader(
            corpus,
            sampler="context",
            context_columns=("dataset_id",),
            row_indices=[4, 5, 6],
            batch_size=5,
            drop_last=False,
            shuffle=False,
            seq_len=LOADER_SEQ_LEN,
            device="cpu",
        )
    )

    _assert_processed_batch(batch, batch_size=3)
    np.testing.assert_array_equal(batch["global_row_index"].cpu().numpy(), [4, 5, 6])
    np.testing.assert_array_equal(batch["dataset_index"].cpu().numpy(), [1, 1, 1])


def test_build_loader_context_sampler_keeps_context_group(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    batch = next(
        build_loader(
            corpus,
            sampler="context",
            context_columns=("dataset_id", "perturb_label"),
            row_indices=[4, 5, 6, 7, 8],
            batch_size=2,
            seq_len=LOADER_SEQ_LEN,
            device="cpu",
            metadata_columns=["perturb_label"],
        )
    )

    _assert_processed_batch(batch, batch_size=2)
    assert len(set(batch["meta_columns"]["perturb_label"])) == 1


def test_context_sampler_exhausts_each_context_group(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    sampler = ContextBatchSampler(
        metadata_index=corpus.metadata_index,
        context_columns=("dataset_id", "perturb_label"),
        row_indices=[4, 5, 6, 7, 8],
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    assert list(sampler) == [[4, 6], [8], [5, 7]]
    assert len(sampler) == 3


def test_load_corpus_federated_reader_uses_dataset_files(tmp_path: Path) -> None:
    _build_federated_lance_corpus(tmp_path)
    corpus = load_corpus(tmp_path)

    batch = corpus.expression_reader.read_expression_flat([0, 4, 8])

    assert corpus.topology == "federated"
    np.testing.assert_array_equal(batch.global_row_index, [0, 4, 8])
    assert batch.batch_size == 3


def test_load_corpus_rejects_unknown_backend(tmp_path: Path) -> None:
    _build_aggregate_lance_corpus(tmp_path)
    index_path = tmp_path / "corpus-index.yaml"
    with open(index_path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    doc["global_metadata"]["backend"] = "unknown_backend"
    with open(index_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(doc, handle)

    with pytest.raises(ValueError, match="backend"):
        load_corpus(tmp_path)
