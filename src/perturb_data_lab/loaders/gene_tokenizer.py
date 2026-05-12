"""Persisted append-stable gene tokenizer for corpus loaders."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl

__all__ = ["DatasetTokenSpan", "GeneTokenizer"]


_CONTRACT_VERSION = "0.1.0"
_TOKEN_NAMESPACE = "canonical_gene_id"


@dataclass(frozen=True)
class DatasetTokenSpan:
    """Audit record for the token IDs introduced by one dataset append."""

    dataset_id: str
    new_token_start: int
    new_token_end: int
    new_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "new_token_start": self.new_token_start,
            "new_token_end": self.new_token_end,
            "new_token_count": self.new_token_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetTokenSpan":
        return cls(
            dataset_id=str(data["dataset_id"]),
            new_token_start=int(data["new_token_start"]),
            new_token_end=int(data["new_token_end"]),
            new_token_count=int(data["new_token_count"]),
        )


class GeneTokenizer:
    """Append-stable canonical gene tokenizer.

    Token IDs are assigned in corpus build order:
    datasets are processed in ``corpus-index.yaml`` order and unseen
    ``canonical_gene_id`` values are appended in each dataset's local
    ``origin_index`` order.
    """

    __slots__ = (
        "kind",
        "contract_version",
        "corpus_id",
        "token_namespace",
        "_token_to_id",
        "_dataset_build_order",
        "_dataset_token_spans",
    )

    def __init__(
        self,
        *,
        corpus_id: str,
        token_to_id: Mapping[str, int] | None = None,
        dataset_build_order: Sequence[str] = (),
        dataset_token_spans: Sequence[DatasetTokenSpan] = (),
        contract_version: str = _CONTRACT_VERSION,
        token_namespace: str = _TOKEN_NAMESPACE,
        kind: str = "gene-tokenizer",
    ) -> None:
        self.kind = kind
        self.contract_version = contract_version
        self.corpus_id = corpus_id
        self.token_namespace = token_namespace
        self._token_to_id = dict(token_to_id or {})
        self._dataset_build_order = tuple(dataset_build_order)
        self._dataset_token_spans = tuple(dataset_token_spans)
        self._validate()

    def _validate(self) -> None:
        if self.kind != "gene-tokenizer":
            raise ValueError(f"expected kind='gene-tokenizer', got {self.kind!r}")
        if self.token_namespace != _TOKEN_NAMESPACE:
            raise ValueError(
                f"expected token_namespace={_TOKEN_NAMESPACE!r}, got {self.token_namespace!r}"
            )
        ids = sorted(self._token_to_id.values())
        if ids != list(range(len(ids))):
            raise ValueError("gene tokenizer IDs must be contiguous 0..N-1")
        if len(set(self._dataset_build_order)) != len(self._dataset_build_order):
            raise ValueError("dataset_build_order contains duplicates")

    @property
    def token_to_id(self) -> dict[str, int]:
        return dict(self._token_to_id)

    @property
    def dataset_build_order(self) -> tuple[str, ...]:
        return self._dataset_build_order

    @property
    def dataset_token_spans(self) -> tuple[DatasetTokenSpan, ...]:
        return self._dataset_token_spans

    @property
    def global_vocab_size(self) -> int:
        return len(self._token_to_id)

    def to_id(self, canonical_gene_id: str) -> int:
        return self._token_to_id[canonical_gene_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "contract_version": self.contract_version,
            "corpus_id": self.corpus_id,
            "token_namespace": self.token_namespace,
            "dataset_build_order": list(self._dataset_build_order),
            "dataset_token_spans": [span.to_dict() for span in self._dataset_token_spans],
            "token_to_id": self._token_to_id,
        }

    def to_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneTokenizer":
        return cls(
            kind=str(data.get("kind", "gene-tokenizer")),
            contract_version=str(data.get("contract_version", _CONTRACT_VERSION)),
            corpus_id=str(data["corpus_id"]),
            token_namespace=str(data.get("token_namespace", _TOKEN_NAMESPACE)),
            dataset_build_order=tuple(str(item) for item in data.get("dataset_build_order", [])),
            dataset_token_spans=tuple(
                DatasetTokenSpan.from_dict(item)
                for item in data.get("dataset_token_spans", [])
            ),
            token_to_id={
                str(key): int(value)
                for key, value in dict(data.get("token_to_id", {})).items()
            },
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "GeneTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object in {path}")
        return cls.from_dict(payload)

    def append_dataset(
        self,
        dataset_id: str,
        canonical_gene_ids: Iterable[str],
    ) -> "GeneTokenizer":
        if dataset_id in self._dataset_build_order:
            raise ValueError(f"dataset_id {dataset_id!r} is already recorded in gene tokenizer")

        token_to_id = dict(self._token_to_id)
        next_id = len(token_to_id)
        seen_in_dataset: set[str] = set()
        new_token_start = next_id

        for canonical_gene_id in canonical_gene_ids:
            gene_id = str(canonical_gene_id)
            if gene_id in seen_in_dataset:
                continue
            seen_in_dataset.add(gene_id)
            if gene_id in token_to_id:
                continue
            token_to_id[gene_id] = next_id
            next_id += 1

        span = DatasetTokenSpan(
            dataset_id=dataset_id,
            new_token_start=new_token_start,
            new_token_end=next_id,
            new_token_count=next_id - new_token_start,
        )
        return GeneTokenizer(
            corpus_id=self.corpus_id,
            token_to_id=token_to_id,
            dataset_build_order=(*self._dataset_build_order, dataset_id),
            dataset_token_spans=(*self._dataset_token_spans, span),
            contract_version=self.contract_version,
            token_namespace=self.token_namespace,
            kind=self.kind,
        )

    @classmethod
    def empty(cls, corpus_id: str) -> "GeneTokenizer":
        return cls(corpus_id=corpus_id)

    @classmethod
    def build_from_canonical_var_parquets(
        cls,
        *,
        corpus_id: str,
        named_var_paths: Mapping[str, str | Path],
        dataset_order: Sequence[str] | None = None,
    ) -> "GeneTokenizer":
        order = list(dataset_order) if dataset_order is not None else list(named_var_paths.keys())
        missing = [dataset_id for dataset_id in order if dataset_id not in named_var_paths]
        if missing:
            raise ValueError(
                f"dataset_order contains dataset IDs missing from named_var_paths: {missing}"
            )

        tokenizer = cls.empty(corpus_id=corpus_id)
        for dataset_id in order:
            var_df = pl.read_parquet(str(named_var_paths[dataset_id]))
            required = {"origin_index", "canonical_gene_id"}
            missing_cols = required - set(var_df.columns)
            if missing_cols:
                raise ValueError(
                    f"canonical var parquet for dataset {dataset_id!r} missing columns: {sorted(missing_cols)}"
                )
            if var_df["origin_index"].dtype == pl.Utf8:
                var_df = var_df.with_columns(pl.col("origin_index").cast(pl.Int32))
            ordered = var_df.sort("origin_index")
            tokenizer = tokenizer.append_dataset(
                dataset_id,
                ordered["canonical_gene_id"].to_list(),
            )
        return tokenizer
