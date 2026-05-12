"""pertTF-specific vocab and label adapters built inside perturb-data-lab.

This module prepares pertTF-compatible mapping objects from a loaded
``perturb-data-lab`` corpus without importing or modifying the external
``pertTF`` repository.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import torch

from ..feature_registry import FeatureRegistry
from ..gpu_pipeline import GPUSparsePipeline
from ..index import MetadataIndex

if TYPE_CHECKING:
    from ..corpus_loader import Corpus

__all__ = [
    "CategoricalLabelMap",
    "PertTFAdapterConfig",
    "PertTFPairedBatchBuilder",
    "PerturbationPairBatch",
    "PerturbationPairSampler",
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
    append_cls: bool = True
    mask_ratio: float = 0.15
    ps_width: int = 1
    include_full_expr: bool = False
    full_expr_mode: str = "union_padded"

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
        if not 0.0 <= float(self.mask_ratio) <= 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        if int(self.ps_width) <= 0:
            raise ValueError("ps_width must be positive")
        if self.full_expr_mode != "union_padded":
            raise ValueError("full_expr_mode must be 'union_padded'")


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
class PerturbationPairBatch:
    """Paired source/target metadata spec for pertTF-style perturbation batches."""

    source_indices: np.ndarray
    target_indices: np.ndarray
    source_dataset_indices: np.ndarray
    target_dataset_indices: np.ndarray
    source_cell_context_ids: np.ndarray
    target_cell_context_ids: np.ndarray
    source_perturbation_ids: np.ndarray
    target_perturbation_ids: np.ndarray
    source_batch_ids: np.ndarray
    target_batch_ids: np.ndarray
    source_cell_context_labels: tuple[str, ...]
    target_cell_context_labels: tuple[str, ...]
    source_perturbation_labels: tuple[str, ...]
    target_perturbation_labels: tuple[str, ...]
    source_batch_labels: tuple[str, ...]
    target_batch_labels: tuple[str, ...]


class PerturbationPairSampler:
    """Sample same-dataset/context source-target perturbation pairs."""

    def __init__(
        self,
        metadata_index: MetadataIndex,
        *,
        batch_size: int,
        config: PertTFAdapterConfig | None = None,
        label_adapter: PertTFLabelAdapter | None = None,
        seed: int = 0,
        drop_last: bool = True,
        perturbed_target_policy: str = "self_to_control_label",
        missing_target_policy: str = "error",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if perturbed_target_policy not in {
            "self_to_control_label",
            "matched_control_cell",
        }:
            raise ValueError(
                "perturbed_target_policy must be 'self_to_control_label' or "
                "'matched_control_cell'"
            )
        if missing_target_policy not in {"error", "warn_skip"}:
            raise ValueError(
                "missing_target_policy must be 'error' or 'warn_skip'"
            )

        resolved_config = config or PertTFAdapterConfig()
        resolved_labels = label_adapter or PertTFLabelAdapter.from_metadata_index(
            metadata_index,
            resolved_config,
        )
        self._meta = metadata_index
        self.config = resolved_config
        self.labels = resolved_labels
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.perturbed_target_policy = perturbed_target_policy
        self.missing_target_policy = missing_target_policy
        self._control_labels = tuple(str(label) for label in self.config.control_labels)
        self._control_label_set = frozenset(self._control_labels)
        if not self._control_labels:
            raise ValueError("control_labels must contain at least one label")

        dataset_index = metadata_index.get_column("dataset_index")
        context_labels = metadata_index.get_column(self.config.cell_context_column)
        perturbation_labels = metadata_index.get_column(self.config.perturbation_column)
        batch_labels = metadata_index.get_column(self.config.batch_column)
        if dataset_index is None:
            raise ValueError("metadata_index is missing required column 'dataset_index'")
        if context_labels is None:
            raise ValueError(
                f"metadata_index is missing required column {self.config.cell_context_column!r}"
            )
        if perturbation_labels is None:
            raise ValueError(
                f"metadata_index is missing required column {self.config.perturbation_column!r}"
            )
        if batch_labels is None:
            raise ValueError(
                f"metadata_index is missing required column {self.config.batch_column!r}"
            )

        self.total_rows = len(metadata_index)
        self._all_indices = np.arange(self.total_rows, dtype=np.int64)
        self._dataset_index = np.asarray(dataset_index, dtype=np.int32)
        self._context_labels = tuple(
            _normalize_label(value, column=self.config.cell_context_column)
            for value in context_labels
        )
        self._perturbation_labels = tuple(
            _normalize_label(value, column=self.config.perturbation_column)
            for value in perturbation_labels
        )
        self._batch_labels = tuple(
            _normalize_label(value, column=self.config.batch_column)
            for value in batch_labels
        )
        self._cell_context_ids = self.labels.cell_context.encode_many(self._context_labels)
        self._perturbation_ids = self.labels.perturbation.encode_many(
            self._perturbation_labels
        )
        self._batch_ids = self.labels.batch.encode_many(self._batch_labels)
        self._pool_by_key = self._build_pool_by_key()
        self._perturbations_by_context = self._build_perturbations_by_context()
        self._control_pool_by_context = self._build_control_pool_by_context()

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_rows // self.batch_size
        return (self.total_rows + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if self.total_rows == 0:
            return
        rng = np.random.default_rng(self.seed + self.epoch)
        source_order = self._all_indices.copy()
        rng.shuffle(source_order)
        for batch_idx, start in enumerate(range(0, self.total_rows, self.batch_size)):
            source_indices = source_order[start : start + self.batch_size]
            if len(source_indices) < self.batch_size and self.drop_last:
                continue
            batch_seed = self.seed + self.epoch * 10000 + batch_idx
            yield self.pair_source_indices(source_indices, seed=batch_seed)

    def pair_source_indices(
        self,
        source_indices: Sequence[int] | np.ndarray,
        *,
        seed: int | None = None,
    ) -> PerturbationPairBatch:
        source_array = np.asarray(source_indices, dtype=np.int64)
        if source_array.ndim != 1:
            raise ValueError("source_indices must be a 1-D sequence")
        if source_array.size == 0:
            return self._assemble_batch(
                source_array,
                np.asarray([], dtype=np.int64),
                (),
            )
        if np.any(source_array < 0) or np.any(source_array >= self.total_rows):
            raise IndexError("source_indices contain out-of-range rows")

        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        target_indices: list[int] = []
        target_perturbation_labels: list[str] = []
        kept_sources: list[int] = []

        for source_idx in source_array.tolist():
            target = self._sample_target_for_source(source_idx, rng)
            if target is None:
                continue
            target_idx, target_perturbation_label = target
            kept_sources.append(source_idx)
            target_indices.append(target_idx)
            target_perturbation_labels.append(target_perturbation_label)

        return self._assemble_batch(
            np.asarray(kept_sources, dtype=np.int64),
            np.asarray(target_indices, dtype=np.int64),
            tuple(target_perturbation_labels),
        )

    def _build_pool_by_key(self) -> dict[tuple[int, str, str], np.ndarray]:
        buckets: dict[tuple[int, str, str], list[int]] = {}
        for row_idx in range(self.total_rows):
            key = (
                int(self._dataset_index[row_idx]),
                self._context_labels[row_idx],
                self._perturbation_labels[row_idx],
            )
            buckets.setdefault(key, []).append(row_idx)
        return {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in buckets.items()
        }

    def _build_perturbations_by_context(
        self,
    ) -> dict[tuple[int, str], tuple[str, ...]]:
        context_to_labels: dict[tuple[int, str], list[str]] = {}
        for dataset_index, context_label, perturbation_label in self._pool_by_key:
            if perturbation_label in self._control_label_set:
                continue
            key = (dataset_index, context_label)
            labels = context_to_labels.setdefault(key, [])
            if perturbation_label not in labels:
                labels.append(perturbation_label)
        return {
            key: tuple(labels)
            for key, labels in context_to_labels.items()
        }

    def _build_control_pool_by_context(
        self,
    ) -> dict[tuple[int, str], np.ndarray]:
        context_to_indices: dict[tuple[int, str], list[np.ndarray]] = {}
        for (dataset_index, context_label, perturbation_label), indices in self._pool_by_key.items():
            if perturbation_label not in self._control_label_set:
                continue
            context_to_indices.setdefault((dataset_index, context_label), []).append(indices)
        return {
            key: np.concatenate(chunks).astype(np.int64, copy=False)
            for key, chunks in context_to_indices.items()
        }

    def _sample_target_for_source(
        self,
        source_idx: int,
        rng: np.random.Generator,
    ) -> tuple[int, str] | None:
        dataset_index = int(self._dataset_index[source_idx])
        context_label = self._context_labels[source_idx]
        perturbation_label = self._perturbation_labels[source_idx]
        context_key = (dataset_index, context_label)
        if perturbation_label in self._control_label_set:
            candidate_perturbations = self._perturbations_by_context.get(context_key, ())
            if not candidate_perturbations:
                return self._handle_missing_target(
                    source_idx,
                    reason=(
                        "no treated target pool exists for control source within "
                        f"dataset_index={dataset_index}, context={context_label!r}"
                    ),
                )
            target_perturbation = str(rng.choice(candidate_perturbations))
            pool = self._pool_by_key[(dataset_index, context_label, target_perturbation)]
            target_idx = int(rng.choice(pool))
            return target_idx, target_perturbation

        if self.perturbed_target_policy == "self_to_control_label":
            return source_idx, self._control_labels[0]

        control_pool = self._control_pool_by_context.get(context_key)
        if control_pool is None or len(control_pool) == 0:
            return self._handle_missing_target(
                source_idx,
                reason=(
                    "no matched control pool exists for perturbed source within "
                    f"dataset_index={dataset_index}, context={context_label!r}"
                ),
            )
        target_idx = int(rng.choice(control_pool))
        return target_idx, self._perturbation_labels[target_idx]

    def _handle_missing_target(
        self,
        source_idx: int,
        *,
        reason: str,
    ) -> tuple[int, str] | None:
        message = f"unable to pair source row {source_idx}: {reason}"
        if self.missing_target_policy == "warn_skip":
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return None
        raise RuntimeError(message)

    def _assemble_batch(
        self,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        target_perturbation_labels: tuple[str, ...],
    ) -> PerturbationPairBatch:
        source_dataset_indices = self._dataset_index[source_indices]
        target_dataset_indices = self._dataset_index[target_indices]
        source_context_ids = self._cell_context_ids[source_indices]
        target_context_ids = self._cell_context_ids[target_indices]
        source_perturbation_ids = self._perturbation_ids[source_indices]
        target_perturbation_ids = self.labels.perturbation.encode_many(
            target_perturbation_labels
        )
        source_batch_ids = self._batch_ids[source_indices]
        target_batch_ids = self._batch_ids[target_indices]
        source_context_labels = tuple(self._context_labels[int(idx)] for idx in source_indices)
        target_context_labels = tuple(self._context_labels[int(idx)] for idx in target_indices)
        source_perturbation_labels = tuple(
            self._perturbation_labels[int(idx)] for idx in source_indices
        )
        source_batch_labels = tuple(self._batch_labels[int(idx)] for idx in source_indices)
        target_batch_labels = tuple(self._batch_labels[int(idx)] for idx in target_indices)
        return PerturbationPairBatch(
            source_indices=source_indices,
            target_indices=target_indices,
            source_dataset_indices=source_dataset_indices,
            target_dataset_indices=target_dataset_indices,
            source_cell_context_ids=source_context_ids,
            target_cell_context_ids=target_context_ids,
            source_perturbation_ids=source_perturbation_ids,
            target_perturbation_ids=target_perturbation_ids,
            source_batch_ids=source_batch_ids,
            target_batch_ids=target_batch_ids,
            source_cell_context_labels=source_context_labels,
            target_cell_context_labels=target_context_labels,
            source_perturbation_labels=source_perturbation_labels,
            target_perturbation_labels=target_perturbation_labels,
            source_batch_labels=source_batch_labels,
            target_batch_labels=target_batch_labels,
        )


class PertTFPairedBatchBuilder:
    """Build pertTF-compatible paired source/target batches from corpus rows.

    When ``include_full_expr=True``, this also emits union-vocabulary dense
    expression tensors plus per-row presence masks. Those masks are intended
    for future pertTF-side loss changes and are emitted here without modifying
    the external ``pertTF`` repository.
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        seq_len: int,
        config: PertTFAdapterConfig | None = None,
        adapter: "PertTFCorpusAdapter" | None = None,
        device: torch.device | str | None = "cpu",
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        resolved_adapter = adapter or PertTFCorpusAdapter.from_corpus(corpus, config)
        self.corpus = corpus
        self.adapter = resolved_adapter
        self.config = resolved_adapter.config
        self.seq_len = int(seq_len)
        self.device = torch.device("cpu" if device is None else device)
        self._pipeline = GPUSparsePipeline(corpus.feature_registry, seq_len=self.seq_len)
        stoi = self.adapter.vocab.to_simple_vocab_stoi()
        self._pad_token_id = int(stoi[self.config.pad_token])
        self._cls_token_id = int(stoi[self.config.cls_token])
        self._special_token_offset = self.adapter.vocab.special_token_offset

    def build_paired_batch(
        self,
        pair_batch: PerturbationPairBatch,
        *,
        seed: int | None = None,
        sampled_gene_ids: torch.Tensor | None = None,
        sampling_mode: str = "hvg",
        expressed_weight: float = 3.0,
        hvg_weight: float = 3.0,
        hvg_top_k: int | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_pair_batch(pair_batch)

        source_raw = self.corpus.inspect_batch(pair_batch.source_indices)
        target_raw = self.corpus.inspect_batch(pair_batch.target_indices)

        source_processed = self._pipeline.process_batch(
            source_raw,
            device=self.device,
            generator=self._torch_generator(seed),
            sampled_gene_ids=sampled_gene_ids,
            sampling_mode=sampling_mode,
            expressed_weight=expressed_weight,
            hvg_weight=hvg_weight,
            hvg_top_k=hvg_top_k,
        )
        target_processed = self._pipeline.process_batch(
            target_raw,
            device=self.device,
            sampled_gene_ids=source_processed["sampled_gene_ids"],
            sampling_mode=sampling_mode,
            expressed_weight=expressed_weight,
            hvg_weight=hvg_weight,
            hvg_top_k=hvg_top_k,
        )

        source_sampled_gene_ids = source_processed["sampled_gene_ids"]
        target_sampled_gene_ids = target_processed["sampled_gene_ids"]
        source_valid_mask = source_processed["valid_mask"]
        target_valid_mask = target_processed["valid_mask"]
        if not torch.equal(source_sampled_gene_ids, target_sampled_gene_ids):
            raise RuntimeError("target reconstruction changed the source-sampled gene IDs")
        if not torch.equal(source_valid_mask, target_valid_mask):
            raise RuntimeError(
                "source and target valid masks diverged despite same-dataset pairing"
            )

        gene_ids = self._to_token_ids(source_sampled_gene_ids, source_valid_mask)
        source_values = self._to_value_tensor(
            source_processed["sampled_counts"],
            source_valid_mask,
        )
        target_values_next = self._to_value_tensor(
            target_processed["sampled_counts"],
            target_valid_mask,
        )

        batch_size = int(source_processed["batch_size"])
        masked_values = self._mask_values(
            source_values,
            generator=self._torch_generator(None if seed is None else int(seed) + 1),
        )

        batch = {
            "gene_ids": gene_ids,
            "next_gene_ids": gene_ids.clone(),
            "values": masked_values,
            "target_values": source_values,
            "target_values_next": target_values_next,
            "batch_labels": torch.as_tensor(
                pair_batch.source_batch_ids,
                dtype=torch.long,
                device=self.device,
            ),
            "celltype_labels": torch.as_tensor(
                pair_batch.source_cell_context_ids,
                dtype=torch.long,
                device=self.device,
            ),
            "perturbation_labels": torch.as_tensor(
                pair_batch.source_perturbation_ids,
                dtype=torch.long,
                device=self.device,
            ),
            "celltype_labels_next": torch.as_tensor(
                pair_batch.target_cell_context_ids,
                dtype=torch.long,
                device=self.device,
            ),
            "perturbation_labels_next": torch.as_tensor(
                pair_batch.target_perturbation_ids,
                dtype=torch.long,
                device=self.device,
            ),
            "sf": self._size_factor_tensor(source_processed, batch_size),
            "sf_next": self._size_factor_tensor(target_processed, batch_size),
            "index": torch.as_tensor(
                pair_batch.source_indices,
                dtype=torch.long,
                device=self.device,
            ),
            "next_index": torch.as_tensor(
                pair_batch.target_indices,
                dtype=torch.long,
                device=self.device,
            ),
            "ps": torch.zeros(
                (batch_size, int(self.config.ps_width)),
                dtype=torch.float32,
                device=self.device,
            ),
            "ps_next": torch.zeros(
                (batch_size, int(self.config.ps_width)),
                dtype=torch.float32,
                device=self.device,
            ),
        }
        if self.config.include_full_expr:
            full_expr, full_expr_mask = self._build_full_expression_tensor(source_raw)
            full_expr_next, full_expr_next_mask = self._build_full_expression_tensor(
                target_raw
            )
            batch["full_gene_ids"] = self._full_gene_ids(batch_size)
            batch["full_expr"] = full_expr
            batch["full_expr_next"] = full_expr_next
            batch["full_expr_mask"] = full_expr_mask
            batch["full_expr_next_mask"] = full_expr_next_mask
        return batch

    __call__ = build_paired_batch

    def _validate_pair_batch(self, pair_batch: PerturbationPairBatch) -> None:
        if pair_batch.source_indices.shape != pair_batch.target_indices.shape:
            raise ValueError("pair_batch source and target indices must have matching shapes")
        if not np.array_equal(
            pair_batch.source_dataset_indices,
            pair_batch.target_dataset_indices,
        ):
            raise ValueError("pair_batch must preserve same-dataset pairing")
        if not np.array_equal(
            pair_batch.source_cell_context_ids,
            pair_batch.target_cell_context_ids,
        ):
            raise ValueError("pair_batch must preserve same-context pairing")

    def _torch_generator(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        if self.device.type == "cuda":
            generator = torch.Generator(device=self.device)
        else:
            generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator

    def _to_token_ids(
        self,
        sampled_gene_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_ids = torch.where(
            valid_mask,
            sampled_gene_ids.to(dtype=torch.long) + self._special_token_offset,
            torch.full_like(sampled_gene_ids, self._pad_token_id, dtype=torch.long),
        )
        return self._prepend_cls_gene_ids(token_ids)

    def _to_value_tensor(
        self,
        sampled_counts: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.where(
            valid_mask,
            sampled_counts.to(dtype=torch.float32),
            torch.full_like(
                sampled_counts,
                float(self.config.pad_value),
                dtype=torch.float32,
            ),
        )
        return self._prepend_cls_values(values)

    def _prepend_cls_gene_ids(self, gene_ids: torch.Tensor) -> torch.Tensor:
        if not self.config.append_cls:
            return gene_ids
        cls_column = torch.full(
            (gene_ids.shape[0], 1),
            self._cls_token_id,
            dtype=torch.long,
            device=gene_ids.device,
        )
        return torch.cat([cls_column, gene_ids], dim=1)

    def _prepend_cls_values(self, values: torch.Tensor) -> torch.Tensor:
        if not self.config.append_cls:
            return values
        cls_column = torch.full(
            (values.shape[0], 1),
            float(self.config.cls_value),
            dtype=torch.float32,
            device=values.device,
        )
        return torch.cat([cls_column, values], dim=1)

    def _prepend_cls_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.config.append_cls:
            return mask
        cls_column = torch.ones(
            (mask.shape[0], 1),
            dtype=torch.bool,
            device=mask.device,
        )
        return torch.cat([cls_column, mask], dim=1)

    def _full_gene_ids(self, batch_size: int) -> torch.Tensor:
        full_gene_ids = torch.as_tensor(
            self.adapter.vocab.gene_token_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        full_gene_ids = self._prepend_cls_gene_ids(full_gene_ids)
        return full_gene_ids.expand(batch_size, -1).clone()

    def _build_full_expression_tensor(
        self,
        raw_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(raw_batch["batch_size"])
        global_vocab = self.corpus.feature_registry.global_vocab_size
        full_expr = torch.zeros(
            (batch_size, global_vocab),
            dtype=torch.float32,
            device=self.device,
        )
        row_offsets = np.asarray(raw_batch["row_offsets"], dtype=np.int64)
        local_gene_ids = np.asarray(raw_batch["expressed_gene_indices"], dtype=np.int64)
        counts = np.asarray(raw_batch["expression_counts"], dtype=np.float32)
        dataset_indices = np.asarray(raw_batch["dataset_index"], dtype=np.int64)

        local_to_global = self.corpus.feature_registry.local_to_global_map
        dataset_has_gene = self.corpus.feature_registry.dataset_has_gene
        full_mask = torch.as_tensor(
            dataset_has_gene[dataset_indices],
            dtype=torch.bool,
            device=self.device,
        )

        for row_idx in range(batch_size):
            start = int(row_offsets[row_idx])
            stop = int(row_offsets[row_idx + 1])
            if start >= stop:
                continue
            dataset_index = int(dataset_indices[row_idx])
            row_local_gene_ids = local_gene_ids[start:stop]
            row_global_gene_ids = local_to_global[dataset_index, row_local_gene_ids]
            if np.any(row_global_gene_ids < 0):
                raise RuntimeError(
                    "raw batch contained unmapped local gene IDs during full_expr build"
                )
            full_expr[
                row_idx,
                torch.as_tensor(
                    row_global_gene_ids,
                    dtype=torch.long,
                    device=self.device,
                ),
            ] = torch.as_tensor(
                counts[start:stop],
                dtype=torch.float32,
                device=self.device,
            )

        return self._prepend_cls_values(full_expr), self._prepend_cls_mask(full_mask)

    def _mask_values(
        self,
        values: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self.config.mask_ratio <= 0.0:
            return values.clone()
        masked = values.clone()
        pad_value = float(self.config.pad_value)
        cls_value = float(self.config.cls_value)
        mask_value = float(self.config.mask_value)
        for row_idx in range(masked.shape[0]):
            eligible = torch.nonzero(
                (masked[row_idx] != pad_value) & (masked[row_idx] != cls_value),
                as_tuple=False,
            ).flatten()
            if eligible.numel() == 0:
                continue
            n_mask = int(eligible.numel() * float(self.config.mask_ratio))
            if n_mask <= 0:
                continue
            chosen = eligible[
                torch.randperm(
                    eligible.numel(),
                    generator=generator,
                    device=eligible.device,
                )[:n_mask]
            ]
            masked[row_idx, chosen] = mask_value
        return masked

    def _size_factor_tensor(
        self,
        processed_batch: dict[str, Any],
        batch_size: int,
    ) -> torch.Tensor:
        raw_size_factor = processed_batch.get("size_factor")
        if raw_size_factor is None:
            return torch.ones(
                (batch_size, 1),
                dtype=torch.float32,
                device=self.device,
            )
        size_factor = torch.as_tensor(
            raw_size_factor,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)
        if size_factor.shape[0] != batch_size:
            raise RuntimeError("size_factor length does not match batch size")
        return size_factor


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
