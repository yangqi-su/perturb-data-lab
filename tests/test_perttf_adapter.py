from pathlib import Path

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import yaml

import perturb_data_lab.loaders as loader_exports
import perturb_data_lab.loaders.adapters as adapter_exports
from perturb_data_lab.loaders import (
    FeatureRegistry,
    MetadataIndex,
    PertTFAdapterConfig,
    PertTFCorpusAdapter,
    load_corpus,
)


def _build_feature_registry() -> FeatureRegistry:
    return FeatureRegistry(
        {
            "ds0": pl.DataFrame(
                {
                    "origin_index": [0, 1],
                    "feature_id": ["GENE_B", "GENE_A"],
                }
            ),
            "ds1": pl.DataFrame(
                {
                    "origin_index": [0, 1],
                    "feature_id": ["GENE_A", "GENE_C"],
                }
            ),
        },
        dataset_order=["ds0", "ds1"],
    )


def _build_metadata_index() -> MetadataIndex:
    return MetadataIndex(
        pl.DataFrame(
            {
                "global_row_index": np.arange(4, dtype=np.int64),
                "cell_id": [f"cell_{idx}" for idx in range(4)],
                "dataset_id": ["ds0", "ds0", "ds1", "ds1"],
                "dataset_index": np.asarray([0, 0, 1, 1], dtype=np.int32),
                "local_row_index": np.asarray([0, 1, 0, 1], dtype=np.int64),
                "size_factor": np.asarray([1.0, 1.1, 0.9, 1.2], dtype=np.float32),
                "cell_context": ["T_cell", "B_cell", "T_cell", "NK_cell"],
                "perturb_label": ["KO_TP53", "KO_TP53", "KO_GATA1", "KO_GATA1"],
                "batch_id": ["batch_b", "batch_a", "batch_b", "batch_c"],
            }
        )
    )


def _write_canonical_obs(path: Path, *, dataset_id: str, global_start: int) -> None:
    frame = pl.DataFrame(
        {
            "cell_id": [f"{dataset_id}_cell_0", f"{dataset_id}_cell_1"],
            "dataset_id": [dataset_id, dataset_id],
            "size_factor": [1.0, 1.2],
            "cell_context": ["T_cell", "B_cell"],
            "perturb_label": ["KO_TP53", "KO_GATA1"],
            "batch_id": [f"batch_{global_start}", f"batch_{global_start + 1}"],
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


def _write_aggregate_lance(path: Path) -> None:
    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [[0, 1], [1], [0], [0, 1]],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [[5, 3], [2], [4], [1, 6]],
                type=pa.list_(pa.int32()),
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(path), mode="overwrite")


def _build_small_corpus(tmp_path: Path) -> Path:
    corpus_root = tmp_path / "perttf-adapter-corpus"
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "corpus_id": "perttf-adapter-corpus",
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

    _write_canonical_obs(
        corpus_root / "meta" / "ds0" / "canonical_meta" / "canonical-obs.parquet",
        dataset_id="ds0",
        global_start=0,
    )
    _write_canonical_obs(
        corpus_root / "meta" / "ds1" / "canonical_meta" / "canonical-obs.parquet",
        dataset_id="ds1",
        global_start=2,
    )
    _write_canonical_var(
        corpus_root / "meta" / "ds0" / "canonical_meta" / "canonical-var.parquet",
        ["GENE_B", "GENE_A"],
    )
    _write_canonical_var(
        corpus_root / "meta" / "ds1" / "canonical_meta" / "canonical-var.parquet",
        ["GENE_A", "GENE_C"],
    )
    _write_aggregate_lance(corpus_root / "matrix" / "aggregated-cells.lance")
    return corpus_root


def test_corpus_adapter_preserves_feature_registry_order_with_reserved_offset(
    tmp_path: Path,
) -> None:
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(corpus)

    assert corpus.feature_registry.global_feature_ids == ("GENE_B", "GENE_A", "GENE_C")
    assert adapter.special_token_offset == 4
    assert adapter.tokens_in_order == (
        "<pad>",
        "<cls>",
        "<unk>",
        "<eos>",
        "GENE_B",
        "GENE_A",
        "GENE_C",
    )
    np.testing.assert_array_equal(
        adapter.gene_token_ids,
        np.asarray([4, 5, 6], dtype=np.int64),
    )
    assert adapter.token_id_for_global_id(0) == 4
    assert adapter.token_id_for_global_id(1) == 5
    assert adapter.feature_id_for_token_id(6) == "GENE_C"


def test_corpus_adapter_emits_simple_vocab_compatible_mapping(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(corpus)
    stoi = adapter.to_simple_vocab_stoi()

    assert list(stoi.keys()) == list(adapter.tokens_in_order)
    assert list(stoi.values()) == list(range(len(stoi)))


def test_corpus_adapter_builds_deterministic_maps_and_control_slots(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT", "CTRL"))
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(corpus, config)

    assert adapter.cell_context_labels == ("T_cell", "B_cell")
    assert adapter.perturbation_labels == ("WT", "CTRL", "KO_TP53", "KO_GATA1")
    assert adapter.batch_labels == ("batch_0", "batch_1", "batch_2", "batch_3")
    assert adapter.control_label_ids == (0, 1)


def test_corpus_adapter_unknown_label_falls_back_when_configured(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(unknown_label="__unknown__")
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(corpus, config)

    assert adapter.encode_cell_context("missing_context") == adapter.encode_cell_context("__unknown__")
    assert adapter.encode_batch("missing_batch") == adapter.encode_batch("__unknown__")


def test_corpus_adapter_builds_from_loaded_small_corpus(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(
        corpus,
        PertTFAdapterConfig(control_labels=("WT",), cell_context_column="cell_context"),
    )

    reference = adapter.to_reference_dict()
    assert reference["genes"] == ["GENE_B", "GENE_A", "GENE_C"]
    np.testing.assert_array_equal(
        reference["gene_token_ids"],
        np.asarray([4, 5, 6], dtype=np.int64),
    )
    assert reference["simple_vocab_stoi"]["GENE_B"] == 4
    assert reference["cell_context_to_index"] == {"T_cell": 0, "B_cell": 1}
    assert reference["perturbation_to_index"] == {"WT": 0, "KO_TP53": 1, "KO_GATA1": 2}


def test_removed_mapping_helpers_are_no_longer_public_exports() -> None:
    for name in ("CategoricalLabelMap", "PertTFLabelAdapter", "PertTFVocabAdapter"):
        assert name not in loader_exports.__all__
        assert name not in adapter_exports.__all__
        assert not hasattr(loader_exports, name)
        assert not hasattr(adapter_exports, name)
