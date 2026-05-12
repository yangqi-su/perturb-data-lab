"""Unit tests for the append-stable persisted gene tokenizer."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from perturb_data_lab.loaders import GeneTokenizer


def _write_canonical_var(path: Path, genes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "origin_index": list(range(len(genes))),
            "canonical_gene_id": genes,
        }
    ).write_parquet(str(path))


class TestGeneTokenizer:
    def test_build_uses_corpus_order_not_alphabetic_order(self, tmp_path: Path) -> None:
        old_path = tmp_path / "z_old.parquet"
        new_path = tmp_path / "a_new.parquet"
        _write_canonical_var(old_path, ["GENE_B", "GENE_C"])
        _write_canonical_var(new_path, ["GENE_A", "GENE_C", "GENE_D"])

        tokenizer = GeneTokenizer.build_from_canonical_var_parquets(
            corpus_id="test-corpus",
            named_var_paths={"z_old": old_path, "a_new": new_path},
            dataset_order=["z_old", "a_new"],
        )

        assert tokenizer.dataset_build_order == ("z_old", "a_new")
        assert tokenizer.to_id("GENE_B") == 0
        assert tokenizer.to_id("GENE_C") == 1
        assert tokenizer.to_id("GENE_A") == 2
        assert tokenizer.to_id("GENE_D") == 3
        assert tokenizer.dataset_token_spans[0].to_dict() == {
            "dataset_id": "z_old",
            "new_token_start": 0,
            "new_token_end": 2,
            "new_token_count": 2,
        }
        assert tokenizer.dataset_token_spans[1].to_dict() == {
            "dataset_id": "a_new",
            "new_token_start": 2,
            "new_token_end": 4,
            "new_token_count": 2,
        }

    def test_append_preserves_existing_ids_for_alphabetically_earlier_dataset(self) -> None:
        tokenizer = GeneTokenizer.empty(corpus_id="test-corpus")
        tokenizer = tokenizer.append_dataset("z_old", ["GENE_B", "GENE_C"])
        updated = tokenizer.append_dataset("a_new", ["GENE_A", "GENE_C", "GENE_D"])

        assert updated.to_id("GENE_B") == 0
        assert updated.to_id("GENE_C") == 1
        assert updated.to_id("GENE_A") == 2
        assert updated.to_id("GENE_D") == 3

    def test_json_round_trip_preserves_mapping(self, tmp_path: Path) -> None:
        tokenizer = GeneTokenizer.empty(corpus_id="test-corpus")
        tokenizer = tokenizer.append_dataset("z_old", ["GENE_B", "GENE_C"])
        tokenizer = tokenizer.append_dataset("a_new", ["GENE_A", "GENE_C", "GENE_D"])

        path = tmp_path / "gene-tokenizer.json"
        tokenizer.to_json(path)
        loaded = GeneTokenizer.from_json(path)

        assert loaded.to_dict() == tokenizer.to_dict()
