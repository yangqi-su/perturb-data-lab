from pathlib import Path

import numpy as np
import polars as pl
import torch

from perturb_data_lab.loaders import FeatureRegistry, GeneTokenMapper


def _registry() -> FeatureRegistry:
    return FeatureRegistry(
        {
            "ds0": pl.DataFrame(
                {
                    "origin_index": np.arange(3, dtype=np.int32),
                    "canonical_gene_id": ["GENE_A", "GENE_B", "GENE_C"],
                }
            ),
            "ds1": pl.DataFrame(
                {
                    "origin_index": np.arange(2, dtype=np.int32),
                    "canonical_gene_id": ["GENE_B", "GENE_D"],
                }
            ),
        },
        dataset_order=["ds0", "ds1"],
    )


def test_de_novo_mapper_encodes_global_ids_with_special_offset() -> None:
    mapper = GeneTokenMapper.from_feature_registry(_registry())

    assert mapper.tokens_in_order == (
        "<pad>",
        "<cls>",
        "<unk>",
        "<eos>",
        "GENE_A",
        "GENE_B",
        "GENE_C",
        "GENE_D",
    )
    np.testing.assert_array_equal(mapper.global_to_token_id, [4, 5, 6, 7])
    np.testing.assert_array_equal(mapper.tokenizable_by_global_id, [True, True, True, True])

    token_ids, token_mask = mapper.encode_global_ids(
        torch.tensor([[0, 2, -1]], dtype=torch.long),
        torch.tensor([[True, True, False]], dtype=torch.bool),
    )

    assert token_ids.tolist() == [[4, 6, 0]]
    assert token_mask.tolist() == [[True, True, False]]


def test_custom_mapper_marks_missing_genes_and_serializes(tmp_path: Path) -> None:
    mapper = GeneTokenMapper.from_tokenizer_stoi(
        _registry(),
        {"<pad>": 0, "<cls>": 1, "<unk>": 2, "GENE_A": 101, "GENE_D": 404},
    )

    np.testing.assert_array_equal(mapper.global_to_token_id, [101, 0, 0, 404])
    np.testing.assert_array_equal(mapper.tokenizable_by_global_id, [True, False, False, True])
    assert mapper.coverage_summary(_registry())["tokenizable_by_dataset"] == {
        "ds0": 1,
        "ds1": 1,
    }

    token_ids, token_mask = mapper.encode_global_ids(
        torch.tensor([[0, 1, 3]], dtype=torch.long),
        torch.ones((1, 3), dtype=torch.bool),
    )
    assert token_ids.tolist() == [[101, 0, 404]]
    assert token_mask.tolist() == [[True, False, True]]

    output_path = tmp_path / "gene-token-mapper.json"
    mapper.to_json(output_path)
    loaded = GeneTokenMapper.from_json(output_path)
    np.testing.assert_array_equal(loaded.global_to_token_id, mapper.global_to_token_id)
    np.testing.assert_array_equal(loaded.tokenizable_by_global_id, mapper.tokenizable_by_global_id)
