"""pertTF-specific vocab and label adapters built inside perturb-data-lab.

This module prepares pertTF-compatible mapping objects from a loaded
``perturb-data-lab`` corpus without importing or modifying the external
``pertTF`` repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from ..feature_registry import FeatureRegistry
from ..index import MetadataIndex

if TYPE_CHECKING:
    from ..corpus_loader import Corpus

__all__ = [
    "CategoricalLabelMap",
    "PertTFAdapterConfig",
    "PertTFCorpusAdapter",
    "PertTFLabelAdapter",
    "PertTFVocabAdapter",
]


_DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<cls>", "<unk>", "<eos>")


def _normalize_label(value: Any, *, column: str) -> str:
    if value is None:
        raise ValueError(f"metadata column '{column}' contains null labels")
    return str(value)


def _ordered_unique(labels: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return tuple(ordered)


@dataclass(frozen=True)
class PertTFAdapterConfig:
    """Configuration for pertTF-local adapter surfaces."""

    cell_context_column: str = "cell_context"
    perturbation_column: str = "perturb_label"
    batch_column: str = "batch_id"
    control_labels: tuple[str, ...] = ("WT",)
    special_tokens: tuple[str, ...] = _DEFAULT_SPECIAL_TOKENS
    pad_token: str = "<pad>"
    cls_token: str = "<cls>"
    unk_token: str = "<unk>"
    eos_token: str = "<eos>"
    pad_value: int = -2
    mask_value: int = -1
    cls_value: int = -3
    unknown_label: str | None = None

    def __post_init__(self) -> None:
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")
        for token_name, token_value in (
            ("pad_token", self.pad_token),
            ("cls_token", self.cls_token),
            ("unk_token", self.unk_token),
            ("eos_token", self.eos_token),
        ):
            if token_value not in self.special_tokens:
                raise ValueError(
                    f"{token_name}={token_value!r} must appear in special_tokens"
                )


class CategoricalLabelMap:
    """Deterministic string-label mapping for one metadata field."""

    def __init__(
        self,
        *,
        name: str,
        column: str,
        labels: Sequence[str],
        unknown_label: str | None = None,
    ) -> None:
        ordered = _ordered_unique(str(label) for label in labels)
        if not ordered:
            raise ValueError(f"label map '{name}' must contain at least one label")
        self.name = name
        self.column = column
        self._labels = ordered
        self._unknown_label = unknown_label
        self._label_to_index = {label: idx for idx, label in enumerate(self._labels)}
        if unknown_label is not None and unknown_label not in self._label_to_index:
            raise ValueError(
                f"unknown_label {unknown_label!r} is not present in map '{name}'"
            )

    @classmethod
    def from_metadata_column(
        cls,
        metadata_index: MetadataIndex,
        *,
        name: str,
        column: str,
        prepend_labels: Sequence[str] = (),
        append_labels: Sequence[str] = (),
        unknown_label: str | None = None,
    ) -> "CategoricalLabelMap":
        values = metadata_index.get_column(column)
        if values is None:
            raise ValueError(f"metadata column '{column}' is not available")
        observed = tuple(_normalize_label(value, column=column) for value in values)
        labels = _ordered_unique(
            [
                *[str(label) for label in prepend_labels],
                *observed,
                *[str(label) for label in append_labels],
            ]
        )
        return cls(
            name=name,
            column=column,
            labels=labels,
            unknown_label=unknown_label,
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def label_to_index(self) -> dict[str, int]:
        return dict(self._label_to_index)

    @property
    def unknown_label(self) -> str | None:
        return self._unknown_label

    def encode(self, label: Any) -> int:
        normalized = _normalize_label(label, column=self.column)
        if normalized in self._label_to_index:
            return self._label_to_index[normalized]
        if self._unknown_label is not None:
            return self._label_to_index[self._unknown_label]
        raise KeyError(
            f"unknown label {normalized!r} for map '{self.name}' (column '{self.column}')"
        )

    def encode_many(self, labels: Sequence[Any]) -> np.ndarray:
        return np.asarray([self.encode(label) for label in labels], dtype=np.int64)

    def decode(self, index: int) -> str:
        return self._labels[int(index)]

    def __len__(self) -> int:
        return len(self._labels)


@dataclass(frozen=True)
class PertTFVocabAdapter:
    """SimpleVocab-compatible token ordering over the feature registry."""

    special_tokens: tuple[str, ...]
    feature_ids_by_global_id: tuple[str, ...]
    _stoi: dict[str, int]

    @classmethod
    def from_feature_registry(
        cls,
        feature_registry: FeatureRegistry,
        *,
        special_tokens: Sequence[str] = _DEFAULT_SPECIAL_TOKENS,
    ) -> "PertTFVocabAdapter":
        normalized_specials = tuple(str(token) for token in special_tokens)
        if len(set(normalized_specials)) != len(normalized_specials):
            raise ValueError("special_tokens must be unique")
        feature_ids = feature_registry.global_feature_ids
        overlap = set(normalized_specials).intersection(feature_ids)
        if overlap:
            raise ValueError(
                "feature IDs overlap reserved special tokens: "
                f"{sorted(overlap)}"
            )
        stoi: dict[str, int] = {}
        for token_id, token in enumerate(normalized_specials):
            stoi[token] = token_id
        next_id = len(normalized_specials)
        for feature_id in feature_ids:
            stoi[feature_id] = next_id
            next_id += 1
        return cls(
            special_tokens=normalized_specials,
            feature_ids_by_global_id=feature_ids,
            _stoi=stoi,
        )

    @property
    def special_token_offset(self) -> int:
        return len(self.special_tokens)

    @property
    def stoi(self) -> dict[str, int]:
        return dict(self._stoi)

    @property
    def gene_token_ids(self) -> np.ndarray:
        start = self.special_token_offset
        stop = start + len(self.feature_ids_by_global_id)
        return np.arange(start, stop, dtype=np.int64)

    @property
    def tokens_in_order(self) -> tuple[str, ...]:
        return tuple(self._stoi.keys())

    def token_id_for_global_id(self, global_id: int) -> int:
        if global_id < 0 or global_id >= len(self.feature_ids_by_global_id):
            raise IndexError(f"global_id {global_id} out of range")
        return self.special_token_offset + int(global_id)

    def global_id_for_token_id(self, token_id: int) -> int | None:
        if token_id < self.special_token_offset:
            return None
        global_id = int(token_id) - self.special_token_offset
        if global_id >= len(self.feature_ids_by_global_id):
            raise IndexError(f"token_id {token_id} out of range")
        return global_id

    def feature_id_for_token_id(self, token_id: int) -> str:
        global_id = self.global_id_for_token_id(token_id)
        if global_id is None:
            raise KeyError(f"token_id {token_id} is reserved for a special token")
        return self.feature_ids_by_global_id[global_id]

    def to_simple_vocab_stoi(self) -> dict[str, int]:
        """Return the raw ordered mapping expected by pertTF SimpleVocab."""
        return dict(self._stoi)

    def to_simple_vocab_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_simple_vocab_stoi(), indent=2, sort_keys=False),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class PertTFLabelAdapter:
    """pertTF label mappings derived from corpus metadata."""

    config: PertTFAdapterConfig
    cell_context: CategoricalLabelMap
    perturbation: CategoricalLabelMap
    batch: CategoricalLabelMap

    @classmethod
    def from_metadata_index(
        cls,
        metadata_index: MetadataIndex,
        config: PertTFAdapterConfig | None = None,
    ) -> "PertTFLabelAdapter":
        resolved = config or PertTFAdapterConfig()
        unknown_label = resolved.unknown_label
        append_unknown = (unknown_label,) if unknown_label is not None else ()
        return cls(
            config=resolved,
            cell_context=CategoricalLabelMap.from_metadata_column(
                metadata_index,
                name="cell_context",
                column=resolved.cell_context_column,
                append_labels=append_unknown,
                unknown_label=unknown_label,
            ),
            perturbation=CategoricalLabelMap.from_metadata_column(
                metadata_index,
                name="perturbation",
                column=resolved.perturbation_column,
                prepend_labels=resolved.control_labels,
                append_labels=append_unknown,
                unknown_label=unknown_label,
            ),
            batch=CategoricalLabelMap.from_metadata_column(
                metadata_index,
                name="batch",
                column=resolved.batch_column,
                append_labels=append_unknown,
                unknown_label=unknown_label,
            ),
        )

    @property
    def control_label_ids(self) -> tuple[int, ...]:
        return tuple(
            self.perturbation.encode(label)
            for label in self.config.control_labels
        )

    def to_reference_dict(self) -> dict[str, Any]:
        return {
            "cell_context_to_index": self.cell_context.label_to_index,
            "perturbation_to_index": self.perturbation.label_to_index,
            "batch_to_index": self.batch.label_to_index,
            "control_label_ids": self.control_label_ids,
        }


@dataclass(frozen=True)
class PertTFCorpusAdapter:
    """Bundle pertTF-local vocab and label mappings for one loaded corpus."""

    config: PertTFAdapterConfig
    vocab: PertTFVocabAdapter
    labels: PertTFLabelAdapter

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        config: PertTFAdapterConfig | None = None,
    ) -> "PertTFCorpusAdapter":
        resolved = config or PertTFAdapterConfig()
        return cls(
            config=resolved,
            vocab=PertTFVocabAdapter.from_feature_registry(
                corpus.feature_registry,
                special_tokens=resolved.special_tokens,
            ),
            labels=PertTFLabelAdapter.from_metadata_index(
                corpus.metadata_index,
                resolved,
            ),
        )

    def to_reference_dict(self) -> dict[str, Any]:
        return {
            "genes": list(self.vocab.feature_ids_by_global_id),
            "gene_token_ids": self.vocab.gene_token_ids.copy(),
            "simple_vocab_stoi": self.vocab.to_simple_vocab_stoi(),
            **self.labels.to_reference_dict(),
        }
