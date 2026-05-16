from pathlib import Path

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import torch
import yaml

from perturb_data_lab.loaders import FeatureRegistry, GPUSparsePipeline, build_loader, load_corpus


def _write_canonical_obs(
    path: Path,
    *,
    dataset_id: str,
    n_cells: int,
) -> None:
    frame = pl.DataFrame(
        {
            "cell_id": [f"{dataset_id}_cell_{idx}" for idx in range(n_cells)],
            "dataset_id": [dataset_id] * n_cells,
            "size_factor": [1.0 + 0.1 * idx for idx in range(n_cells)],
            "perturb_label": ["ctrl"] * n_cells,
            "raw_cell_type": ["mock_type"] * n_cells,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(path))


def _write_canonical_var(path: Path, gene_names: list[str]) -> None:
    frame = pl.DataFrame(
        {
            "origin_index": np.arange(len(gene_names), dtype=np.int32),
            "gene_id": [f"ENSG_{gene_name}" for gene_name in gene_names],
            "canonical_gene_id": gene_names,
            "global_id": np.arange(len(gene_names), dtype=np.int32),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(path))


def _write_hvg_ranking(
    path: Path,
    *,
    gene_names: list[str],
    ranks: list[int],
    default_n_hvg: int,
) -> None:
    frame = pl.DataFrame(
        {
            "origin_index": np.arange(len(gene_names), dtype=np.int32),
            "feature_id": gene_names,
            "mean_log1p_expr": np.linspace(0.1, 1.0, len(gene_names), dtype=np.float32),
            "variance_log1p_expr": np.linspace(1.0, 2.0, len(gene_names), dtype=np.float32),
            "dispersion_log": np.linspace(0.5, 1.5, len(gene_names), dtype=np.float32),
            "dispersion_norm": np.linspace(1.5, 0.5, len(gene_names), dtype=np.float32),
            "hvg_rank": np.asarray(ranks, dtype=np.int32),
            "selected_at_default_n_hvg": np.asarray(ranks, dtype=np.int32) <= int(default_n_hvg),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(path))


def _write_aggregate_lance(path: Path, rows: list[dict[str, list[int]]]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(path), mode="overwrite")


def _build_runtime_hvg_corpus(tmp_path: Path) -> Path:
    corpus_root = tmp_path / "runtime-hvg-corpus"
    datasets = [
        {
            "dataset_id": "ds0",
            "dataset_index": 0,
            "global_start": 0,
            "global_end": 2,
            "genes": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
            "hvg_ranks": [1, 2, 3, 4],
            "rows": [
                {"expressed_gene_indices": [0, 2], "expression_counts": [5, 1]},
                {"expressed_gene_indices": [1, 3], "expression_counts": [3, 2]},
            ],
        },
        {
            "dataset_id": "ds1",
            "dataset_index": 1,
            "global_start": 2,
            "global_end": 4,
            "genes": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
            "hvg_ranks": [4, 3, 2, 1],
            "rows": [
                {"expressed_gene_indices": [0, 1], "expression_counts": [2, 4]},
                {"expressed_gene_indices": [2, 3], "expression_counts": [6, 7]},
            ],
        },
    ]

    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "corpus_id": "runtime-hvg-corpus",
        "global_metadata": {"backend": "lance", "topology": "aggregate"},
        "datasets": [
            {
                "dataset_id": dataset["dataset_id"],
                "join_mode": "create_new" if dataset["dataset_index"] == 0 else "append_routed",
                "dataset_index": dataset["dataset_index"],
                "cell_count": dataset["global_end"] - dataset["global_start"],
                "global_start": dataset["global_start"],
                "global_end": dataset["global_end"],
            }
            for dataset in datasets
        ],
    }
    corpus_root.mkdir(parents=True, exist_ok=True)
    with open(corpus_root / "corpus-index.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(index_doc, handle, sort_keys=False)

    all_rows: list[dict[str, list[int]]] = []
    for dataset in datasets:
        dataset_root = corpus_root / "meta" / dataset["dataset_id"]
        _write_canonical_obs(
            dataset_root / "canonical_meta" / "canonical-obs.parquet",
            dataset_id=dataset["dataset_id"],
            n_cells=dataset["global_end"] - dataset["global_start"],
        )
        _write_canonical_var(
            dataset_root / "canonical_meta" / "canonical-var.parquet",
            dataset["genes"],
        )
        _write_hvg_ranking(
            dataset_root / "hvg.parquet",
            gene_names=dataset["genes"],
            ranks=dataset["hvg_ranks"],
            default_n_hvg=2,
        )
        all_rows.extend(dataset["rows"])

    _write_aggregate_lance(corpus_root / "matrix" / "aggregated-cells.lance", all_rows)
    return corpus_root


def _build_feature_registry_with_partial_vocab(tmp_path: Path) -> FeatureRegistry:
    base = tmp_path / "registry"
    ds0_var = base / "meta" / "ds0" / "canonical-var.parquet"
    ds1_var = base / "meta" / "ds1" / "canonical-var.parquet"
    ds0_hvg = base / "meta" / "ds0" / "hvg.parquet"
    ds1_hvg = base / "meta" / "ds1" / "hvg.parquet"

    _write_canonical_var(ds0_var, ["GENE_A", "GENE_B"])
    _write_canonical_var(ds1_var, ["GENE_B", "GENE_C", "GENE_D"])
    _write_hvg_ranking(ds0_hvg, gene_names=["GENE_A", "GENE_B"], ranks=[1, 2], default_n_hvg=1)
    _write_hvg_ranking(
        ds1_hvg,
        gene_names=["GENE_B", "GENE_C", "GENE_D"],
        ranks=[3, 2, 1],
        default_n_hvg=2,
    )

    return FeatureRegistry.from_canonical_var_parquets(
        {
            "ds0": ds0_var,
            "ds1": ds1_var,
        },
        dataset_order=["ds0", "ds1"],
    )


def test_load_corpus_reads_hvg_rank_matrix_from_parquet(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_runtime_hvg_corpus(tmp_path)))
    registry = corpus.feature_registry

    np.testing.assert_array_equal(
        registry.hvg_rank_matrix,
        np.asarray(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        registry.hvg_mask,
        np.asarray(
            [
                [True, True, False, False],
                [False, False, True, True],
            ],
            dtype=bool,
        ),
    )


def test_feature_registry_hvg_without_default_selection_is_top_k_only(tmp_path: Path) -> None:
    var_path = tmp_path / "meta" / "ds0" / "canonical_meta" / "canonical-var.parquet"
    hvg_path = tmp_path / "meta" / "ds0" / "hvg.parquet"
    _write_canonical_var(var_path, ["GENE_A", "GENE_B"])
    pl.DataFrame(
        {
            "origin_index": np.asarray([0, 1], dtype=np.int32),
            "hvg_rank": np.asarray([1, 2], dtype=np.int32),
        }
    ).write_parquet(hvg_path)

    registry = FeatureRegistry.from_canonical_var_parquets(
        {"ds0": var_path},
    )

    np.testing.assert_array_equal(registry.hvg_rank_matrix, np.asarray([[1, 2]], dtype=np.int32))
    np.testing.assert_array_equal(registry.hvg_mask, np.asarray([[False, False]], dtype=bool))


def test_dynamic_hvg_top_k_changes_weighted_probs_without_rematerialization(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_runtime_hvg_corpus(tmp_path)))
    pipeline = GPUSparsePipeline(corpus.feature_registry, seq_len=2)
    device = torch.device("cpu")
    cached = pipeline._cached_tensors(device)
    dataset_indices = torch.tensor([0], dtype=torch.long, device=device)

    probs_top1 = pipeline._build_weighted_probs(
        device,
        1,
        dataset_indices,
        sampling_mode="hvg",
        hvg_weight=3.0,
        hvg_top_k=1,
        has_gene_t=cached["has_gene"],
        hvg_t=cached["hvg_mask"],
        hvg_rank_t=cached["hvg_rank"],
    )
    probs_top3 = pipeline._build_weighted_probs(
        device,
        1,
        dataset_indices,
        sampling_mode="hvg",
        hvg_weight=3.0,
        hvg_top_k=3,
        has_gene_t=cached["has_gene"],
        hvg_t=cached["hvg_mask"],
        hvg_rank_t=cached["hvg_rank"],
    )

    np.testing.assert_allclose(
        probs_top1.cpu().numpy()[0],
        np.asarray([4.0, 1.0, 1.0, 1.0], dtype=np.float32) / 7.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        probs_top3.cpu().numpy()[0],
        np.asarray([4.0, 4.0, 4.0, 1.0], dtype=np.float32) / 13.0,
        rtol=1e-6,
    )


def test_hvg_weighted_probs_zero_absent_genes(tmp_path: Path) -> None:
    registry = _build_feature_registry_with_partial_vocab(tmp_path)
    pipeline = GPUSparsePipeline(registry, seq_len=2)
    device = torch.device("cpu")
    cached = pipeline._cached_tensors(device)

    probs = pipeline._build_weighted_probs(
        device,
        1,
        torch.tensor([0], dtype=torch.long, device=device),
        sampling_mode="hvg",
        hvg_weight=3.0,
        hvg_top_k=5,
        has_gene_t=cached["has_gene"],
        hvg_t=cached["hvg_mask"],
        hvg_rank_t=cached["hvg_rank"],
    )

    np.testing.assert_allclose(
        probs.cpu().numpy()[0],
        np.asarray([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
        rtol=1e-6,
    )


def test_cpu_and_gpu_loader_routes_share_hvg_top_k_semantics(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_runtime_hvg_corpus(tmp_path)))
    loader_kwargs = {
        "batch_size": 2,
        "drop_last": False,
        "sampler": "context",
        "context_columns": ("dataset_id",),
        "row_indices": [0, 1],
        "shuffle": False,
        "seed": 17,
        "seq_len": 2,
        "sampling_mode": "hvg",
        "hvg_top_k": 1,
    }

    torch.manual_seed(123)
    first_batch = next(
        build_loader(
            corpus,
            device="cpu",
            num_workers=0,
            **loader_kwargs,
        )
    )
    torch.manual_seed(123)
    second_batch = next(
        build_loader(
            corpus,
            device="cpu",
            num_workers=0,
            **loader_kwargs,
        )
    )

    for key in (
        "sampled_gene_ids",
        "sampled_counts",
        "valid_mask",
        "exact_match_mask",
        "dataset_index",
        "global_row_index",
    ):
        assert torch.equal(first_batch[key], second_batch[key])
