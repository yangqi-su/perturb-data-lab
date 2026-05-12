from pathlib import Path

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import torch
import yaml

from perturb_data_lab.loaders import (
    PertTFAdapterConfig,
    PertTFPairedBatchBuilder,
    PerturbationPairSampler,
    load_corpus,
)


def _write_canonical_obs(path: Path) -> None:
    frame = pl.DataFrame(
        {
            "cell_id": ["ds0_wt", "ds0_ko"],
            "dataset_id": ["ds0", "ds0"],
            "size_factor": [1.0, 1.2],
            "cell_context": ["T_cell", "T_cell"],
            "perturb_label": ["WT", "KO_TP53"],
            "batch_id": ["batch_0", "batch_1"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(path))


def _write_canonical_var(path: Path, gene_names: list[str]) -> None:
    frame = pl.DataFrame(
        {
            "origin_index": [str(idx) for idx in range(len(gene_names))],
            "gene_id": [f"ENSG_{gene_name}" for gene_name in gene_names],
            "canonical_gene_id": gene_names,
            "global_id": [str(idx) for idx in range(len(gene_names))],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(path))


def _write_hvg_ranking(path: Path, gene_names: list[str]) -> None:
    frame = pl.DataFrame(
        {
            "origin_index": np.arange(len(gene_names), dtype=np.int32),
            "feature_id": gene_names,
            "mean_log1p_expr": np.linspace(0.1, 1.0, len(gene_names), dtype=np.float32),
            "variance_log1p_expr": np.linspace(1.0, 2.0, len(gene_names), dtype=np.float32),
            "dispersion_log": np.linspace(0.5, 1.5, len(gene_names), dtype=np.float32),
            "dispersion_norm": np.linspace(1.5, 0.5, len(gene_names), dtype=np.float32),
            "hvg_rank": np.arange(1, len(gene_names) + 1, dtype=np.int32),
            "selected_at_default_n_hvg": np.arange(len(gene_names), dtype=np.int32) < 2,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(path))


def _write_aggregate_lance(path: Path) -> None:
    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [[0, 2], [1, 3]],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [[5, 1], [3, 2]],
                type=pa.list_(pa.int32()),
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(path), mode="overwrite")


def _build_small_pair_corpus(tmp_path: Path) -> Path:
    corpus_root = tmp_path / "perttf-paired-corpus"
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "corpus_id": "perttf-paired-corpus",
        "global_metadata": {"backend": "lance", "topology": "aggregate"},
        "datasets": [
            {
                "dataset_id": "ds0",
                "join_mode": "create_new",
                "dataset_index": 0,
                "cell_count": 2,
                "global_start": 0,
                "global_end": 2,
            }
        ],
    }
    corpus_root.mkdir(parents=True, exist_ok=True)
    with open(corpus_root / "corpus-index.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(index_doc, handle, sort_keys=False)

    dataset_root = corpus_root / "meta" / "ds0"
    _write_canonical_obs(dataset_root / "canonical_meta" / "canonical-obs.parquet")
    _write_canonical_var(
        dataset_root / "canonical_meta" / "canonical-var.parquet",
        ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
    )
    _write_hvg_ranking(dataset_root / "hvg.parquet", ["GENE_A", "GENE_B", "GENE_C", "GENE_D"])
    _write_aggregate_lance(corpus_root / "matrix" / "aggregated-cells.lance")
    return corpus_root


def _build_mixed_union_pair_corpus(tmp_path: Path) -> Path:
    corpus_root = tmp_path / "perttf-union-pair-corpus"
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "corpus_id": "perttf-union-pair-corpus",
        "global_metadata": {"backend": "lance", "topology": "aggregate"},
        "datasets": [
            {
                "dataset_id": "ds0",
                "join_mode": "create_new",
                "dataset_index": 0,
                "cell_count": 2,
                "global_start": 0,
                "global_end": 2,
            },
            {
                "dataset_id": "ds1",
                "join_mode": "append_routed",
                "dataset_index": 1,
                "cell_count": 2,
                "global_start": 2,
                "global_end": 4,
            },
        ],
    }
    corpus_root.mkdir(parents=True, exist_ok=True)
    with open(corpus_root / "corpus-index.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(index_doc, handle, sort_keys=False)

    ds0_root = corpus_root / "meta" / "ds0"
    _write_canonical_obs(ds0_root / "canonical_meta" / "canonical-obs.parquet")
    _write_canonical_var(
        ds0_root / "canonical_meta" / "canonical-var.parquet",
        ["GENE_A", "GENE_B"],
    )
    _write_hvg_ranking(ds0_root / "hvg.parquet", ["GENE_A", "GENE_B"])

    ds1_root = corpus_root / "meta" / "ds1"
    frame = pl.DataFrame(
        {
            "cell_id": ["ds1_wt", "ds1_ko"],
            "dataset_id": ["ds1", "ds1"],
            "size_factor": [0.9, 1.1],
            "cell_context": ["T_cell", "T_cell"],
            "perturb_label": ["WT", "KO_STAT1"],
            "batch_id": ["batch_2", "batch_3"],
        }
    )
    (ds1_root / "canonical_meta").mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(ds1_root / "canonical_meta" / "canonical-obs.parquet"))
    _write_canonical_var(
        ds1_root / "canonical_meta" / "canonical-var.parquet",
        ["GENE_B", "GENE_C"],
    )
    _write_hvg_ranking(ds1_root / "hvg.parquet", ["GENE_B", "GENE_C"])

    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [[0], [1], [0], [1]],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [[5], [4], [2], [7]],
                type=pa.list_(pa.int32()),
            ),
        }
    )
    matrix_path = corpus_root / "matrix" / "aggregated-cells.lance"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(matrix_path), mode="overwrite")
    return corpus_root


def _dense_row(row_index: int) -> np.ndarray:
    if row_index == 0:
        return np.asarray([5.0, 0.0, 1.0, 0.0], dtype=np.float32)
    if row_index == 1:
        return np.asarray([0.0, 3.0, 0.0, 2.0], dtype=np.float32)
    raise AssertionError(f"unexpected row_index {row_index}")


def test_paired_batch_builder_reconstructs_target_values_at_source_sampled_genes(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0)
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    pair_batch = PerturbationPairSampler(
        corpus.metadata_index,
        batch_size=2,
        config=config,
        seed=7,
    ).pair_source_indices([0, 1], seed=11)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=3, config=config)

    batch = builder.build_paired_batch(
        pair_batch,
        seed=5,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=2,
    )

    expected_keys = {
        "gene_ids",
        "next_gene_ids",
        "values",
        "target_values",
        "target_values_next",
        "batch_labels",
        "celltype_labels",
        "perturbation_labels",
        "celltype_labels_next",
        "perturbation_labels_next",
        "ps",
        "ps_next",
        "sf",
        "sf_next",
        "index",
        "next_index",
    }
    assert expected_keys.issubset(batch.keys())
    assert batch["gene_ids"].shape == (2, 4)
    assert torch.equal(batch["gene_ids"], batch["next_gene_ids"])
    assert torch.equal(batch["values"], batch["target_values"])
    assert batch["sf"].shape == (2, 1)
    assert batch["sf_next"].shape == (2, 1)
    torch.testing.assert_close(
        batch["sf"],
        torch.tensor([[1.0], [1.2]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        batch["sf_next"],
        torch.tensor([[1.2], [1.2]], dtype=torch.float32),
    )
    assert batch["perturbation_labels_next"].tolist() == [1, 0]
    assert batch["next_index"].tolist() == [1, 1]
    assert torch.count_nonzero(batch["ps"]) == 0
    assert torch.count_nonzero(batch["ps_next"]) == 0

    pad_token_id = builder.adapter.vocab.to_simple_vocab_stoi()[config.pad_token]
    cls_token_id = builder.adapter.vocab.to_simple_vocab_stoi()[config.cls_token]
    offset = builder.adapter.vocab.special_token_offset
    assert torch.all(batch["gene_ids"][:, 0] == cls_token_id)
    assert torch.all(batch["target_values"][:, 0] == float(config.cls_value))
    assert torch.all(batch["target_values_next"][:, 0] == float(config.cls_value))

    for batch_row, source_index in enumerate(pair_batch.source_indices.tolist()):
        source_dense = _dense_row(source_index)
        target_dense = _dense_row(int(pair_batch.target_indices[batch_row]))
        for col in range(1, batch["gene_ids"].shape[1]):
            token_id = int(batch["gene_ids"][batch_row, col])
            if token_id == pad_token_id:
                assert batch["target_values"][batch_row, col].item() == float(config.pad_value)
                assert batch["target_values_next"][batch_row, col].item() == float(config.pad_value)
                continue
            global_id = token_id - offset
            assert batch["target_values"][batch_row, col].item() == source_dense[global_id]
            assert batch["target_values_next"][batch_row, col].item() == target_dense[global_id]


def test_paired_batch_builder_accepts_precomputed_sampled_gene_ids_and_preserves_padding(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT",), mask_ratio=1.0)
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    pair_batch = PerturbationPairSampler(
        corpus.metadata_index,
        batch_size=1,
        config=config,
        seed=3,
    ).pair_source_indices([0], seed=13)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=3, config=config)

    batch = builder.build_paired_batch(
        pair_batch,
        seed=17,
        sampled_gene_ids=torch.tensor([[0, 3, -1]], dtype=torch.long),
    )

    pad_token_id = builder.adapter.vocab.to_simple_vocab_stoi()[config.pad_token]
    cls_token_id = builder.adapter.vocab.to_simple_vocab_stoi()[config.cls_token]
    assert batch["gene_ids"].shape == (1, 4)
    assert batch["gene_ids"][0].tolist() == [cls_token_id, 4, 7, pad_token_id]
    assert batch["values"][0].tolist() == [
        float(config.cls_value),
        float(config.mask_value),
        float(config.mask_value),
        float(config.pad_value),
    ]
    assert batch["target_values"][0].tolist() == [
        float(config.cls_value),
        5.0,
        0.0,
        float(config.pad_value),
    ]
    assert batch["target_values_next"][0].tolist() == [
        float(config.cls_value),
        0.0,
        2.0,
        float(config.pad_value),
    ]


def test_paired_batch_builder_emits_union_full_expression_masks_for_mixed_datasets(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(
        control_labels=("WT",),
        mask_ratio=0.0,
        include_full_expr=True,
    )
    corpus = load_corpus(str(_build_mixed_union_pair_corpus(tmp_path)))
    pair_batch = PerturbationPairSampler(
        corpus.metadata_index,
        batch_size=2,
        config=config,
        seed=19,
    ).pair_source_indices([0, 2], seed=23)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=2, config=config)

    batch = builder.build_paired_batch(
        pair_batch,
        seed=29,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=1,
    )

    cls_token_id = builder.adapter.vocab.to_simple_vocab_stoi()[config.cls_token]
    assert batch["full_gene_ids"].shape == (2, 4)
    assert batch["full_gene_ids"][0].tolist() == [cls_token_id, 4, 5, 6]
    assert torch.equal(batch["full_gene_ids"][0], batch["full_gene_ids"][1])

    assert batch["full_expr"].shape == (2, 4)
    assert batch["full_expr_next"].shape == (2, 4)
    assert batch["full_expr_mask"].dtype == torch.bool
    assert batch["full_expr_next_mask"].dtype == torch.bool

    assert batch["index"].tolist() == [0, 2]
    assert batch["next_index"].tolist() == [1, 3]

    assert batch["full_expr"][0].tolist() == [float(config.cls_value), 5.0, 0.0, 0.0]
    assert batch["full_expr_mask"][0].tolist() == [True, True, True, False]
    assert batch["full_expr_next"][0].tolist() == [float(config.cls_value), 0.0, 4.0, 0.0]
    assert batch["full_expr_next_mask"][0].tolist() == [True, True, True, False]

    assert batch["full_expr"][1].tolist() == [float(config.cls_value), 0.0, 2.0, 0.0]
    assert batch["full_expr_mask"][1].tolist() == [True, False, True, True]
    assert batch["full_expr_next"][1].tolist() == [float(config.cls_value), 0.0, 0.0, 7.0]
    assert batch["full_expr_next_mask"][1].tolist() == [True, False, True, True]
