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
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..feature_registry import FeatureRegistry
from ..gpu_pipeline import GPUSparsePipeline
from ..index import MetadataIndex
from ..loaders import DatasetRoutingTable, _normalize_candidate_row_indices

if TYPE_CHECKING:
    from ..corpus_loader import Corpus

__all__ = [
    "PertTFAdapterConfig",
    "PertTFNullLabelFilterStats",
    "PertTFPairedBatchLoader",
    "PertTFPairedBatchBuilder",
    "PerturbationPairBatch",
    "PerturbationPairSampler",
    "PertTFCorpusAdapter",
]


_DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<cls>", "<unk>", "<eos>")


@dataclass(frozen=True)
class PertTFNullLabelFilterStats:
    """Compact summary of required-label null filtering."""

    policy: str
    required_columns: tuple[str, ...]
    checked_row_count: int
    kept_row_count: int
    dropped_row_count: int
    per_column_null_counts: dict[str, int]


@dataclass(frozen=True)
class _PertTFNullLabelSelection:
    """Resolved row subset after required-label null handling."""

    candidate_row_indices: np.ndarray
    effective_row_indices: np.ndarray
    stats: PertTFNullLabelFilterStats


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


def _derive_perttf_stream_seed(
    base_seed: int,
    *,
    epoch: int,
    batch_index: int,
    stream_id: int,
) -> int:
    seed_sequence = np.random.SeedSequence(
        [
            int(base_seed),
            int(epoch),
            int(batch_index),
            int(stream_id),
        ]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


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
    null_label_policy: str = "drop"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "null_label_policy",
            _normalize_null_label_policy(self.null_label_policy),
        )
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


def _normalize_null_label_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized == "strict":
        normalized = "error"
    if normalized not in {"drop", "error"}:
        raise ValueError("null_label_policy must be 'drop', 'error', or 'strict'")
    return normalized


def _required_label_columns(config: PertTFAdapterConfig) -> tuple[str, ...]:
    return (
        config.cell_context_column,
        config.perturbation_column,
        config.batch_column,
    )


def _format_null_label_counts(stats: PertTFNullLabelFilterStats) -> str:
    nonzero_counts = [
        f"{column}={count}"
        for column, count in stats.per_column_null_counts.items()
        if count > 0
    ]
    if not nonzero_counts:
        return "none"
    return ", ".join(nonzero_counts)


def _format_null_label_policy_message(
    warning_owner: str,
    stats: PertTFNullLabelFilterStats,
) -> str:
    counts = _format_null_label_counts(stats)
    if stats.policy == "drop":
        return (
            f"{warning_owner} dropped {stats.dropped_row_count} of "
            f"{stats.checked_row_count} rows with null required pertTF labels "
            f"({counts})"
        )
    return (
        f"{warning_owner} found null required pertTF labels in "
        f"{stats.dropped_row_count} of {stats.checked_row_count} rows while "
        "null_label_policy='error' "
        f"({counts})"
    )


def _select_non_null_label_rows(
    metadata_index: MetadataIndex,
    *,
    config: PertTFAdapterConfig,
    row_indices: Sequence[int] | np.ndarray | None,
    warning_owner: str,
    emit_warning: bool,
    warning_stacklevel: int,
) -> _PertTFNullLabelSelection:
    candidate_row_indices = _normalize_candidate_row_indices(metadata_index, row_indices)
    if candidate_row_indices is None:
        candidate_row_indices = np.arange(len(metadata_index), dtype=np.int64)
    else:
        candidate_row_indices = candidate_row_indices.copy()

    required_columns = _required_label_columns(config)
    missing_columns = [
        column for column in required_columns if column not in metadata_index.df.columns
    ]
    if missing_columns:
        missing_list = ", ".join(repr(column) for column in missing_columns)
        raise ValueError(f"metadata_index is missing required column(s): {missing_list}")

    selected_metadata = metadata_index.take(
        candidate_row_indices,
        columns=list(required_columns),
    )
    null_mask = np.zeros(candidate_row_indices.size, dtype=bool)
    per_column_null_counts: dict[str, int] = {}
    for column in required_columns:
        values = selected_metadata[column]
        column_null_mask = np.fromiter(
            (value is None for value in values),
            dtype=bool,
            count=candidate_row_indices.size,
        )
        per_column_null_counts[column] = int(np.count_nonzero(column_null_mask))
        null_mask |= column_null_mask

    dropped_row_count = int(np.count_nonzero(null_mask))
    effective_row_indices = candidate_row_indices[~null_mask].copy()
    stats = PertTFNullLabelFilterStats(
        policy=config.null_label_policy,
        required_columns=required_columns,
        checked_row_count=int(candidate_row_indices.size),
        kept_row_count=int(effective_row_indices.size),
        dropped_row_count=dropped_row_count,
        per_column_null_counts=per_column_null_counts,
    )
    if dropped_row_count > 0:
        message = _format_null_label_policy_message(warning_owner, stats)
        if config.null_label_policy == "error":
            raise ValueError(message)
        if emit_warning:
            warnings.warn(message, RuntimeWarning, stacklevel=warning_stacklevel)
    return _PertTFNullLabelSelection(
        candidate_row_indices=candidate_row_indices,
        effective_row_indices=effective_row_indices,
        stats=stats,
    )


def _raise_empty_effective_rows(warning_owner: str, pool_name: str) -> None:
    raise ValueError(
        f"{warning_owner} resolved no usable {pool_name} after dropping rows with "
        "null required pertTF labels"
    )


class _CategoricalLabelMap:
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
        row_indices: Sequence[int] | np.ndarray | None = None,
    ) -> "_CategoricalLabelMap":
        if column not in metadata_index.df.columns:
            raise ValueError(f"metadata column '{column}' is not available")
        normalized_row_indices = _normalize_candidate_row_indices(
            metadata_index,
            row_indices,
        )
        if normalized_row_indices is None:
            unique_values = metadata_index.df[column].unique(maintain_order=True).to_list()
        else:
            selected_values = metadata_index.take(
                normalized_row_indices,
                columns=[column],
            )[column]
            unique_values = list(selected_values)
        observed = tuple(
            _normalize_label(value, column=column)
            for value in unique_values
        )
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
class _PertTFVocabMapping:
    """Internal SimpleVocab-compatible token ordering over the feature registry."""

    special_tokens: tuple[str, ...]
    feature_ids_by_global_id: tuple[str, ...]
    _stoi: dict[str, int]

    @classmethod
    def from_feature_registry(
        cls,
        feature_registry: FeatureRegistry,
        *,
        special_tokens: Sequence[str] = _DEFAULT_SPECIAL_TOKENS,
    ) -> "_PertTFVocabMapping":
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
class _PertTFLabelMappings:
    """Internal pertTF label mappings derived from corpus metadata."""

    config: PertTFAdapterConfig
    cell_context: _CategoricalLabelMap
    perturbation: _CategoricalLabelMap
    batch: _CategoricalLabelMap
    null_label_filter_stats: PertTFNullLabelFilterStats | None = None

    @classmethod
    def from_metadata_index(
        cls,
        metadata_index: MetadataIndex,
        config: PertTFAdapterConfig | None = None,
        row_indices: Sequence[int] | np.ndarray | None = None,
        *,
        _emit_null_label_warning: bool = True,
        _null_label_warning_owner: str = "PertTFCorpusAdapter",
        _null_label_warning_stacklevel: int = 3,
    ) -> "_PertTFLabelMappings":
        resolved = config or PertTFAdapterConfig()
        unknown_label = resolved.unknown_label
        append_unknown = (unknown_label,) if unknown_label is not None else ()
        selection = _select_non_null_label_rows(
            metadata_index,
            config=resolved,
            row_indices=row_indices,
            warning_owner=_null_label_warning_owner,
            emit_warning=_emit_null_label_warning,
            warning_stacklevel=_null_label_warning_stacklevel,
        )
        if selection.effective_row_indices.size == 0:
            _raise_empty_effective_rows(_null_label_warning_owner, "label row pool")
        return cls(
            config=resolved,
            cell_context=_CategoricalLabelMap.from_metadata_column(
                metadata_index,
                name="cell_context",
                column=resolved.cell_context_column,
                append_labels=append_unknown,
                unknown_label=unknown_label,
                row_indices=selection.effective_row_indices,
            ),
            perturbation=_CategoricalLabelMap.from_metadata_column(
                metadata_index,
                name="perturbation",
                column=resolved.perturbation_column,
                prepend_labels=resolved.control_labels,
                append_labels=append_unknown,
                unknown_label=unknown_label,
                row_indices=selection.effective_row_indices,
            ),
            batch=_CategoricalLabelMap.from_metadata_column(
                metadata_index,
                name="batch",
                column=resolved.batch_column,
                append_labels=append_unknown,
                unknown_label=unknown_label,
                row_indices=selection.effective_row_indices,
            ),
            null_label_filter_stats=selection.stats,
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


@dataclass(frozen=True)
class _PertTFPairBatchPlan:
    """Deterministic source-batch plan for one pertTF paired request.

    Future distributed support can shard this plan stream by ``batch_index``
    before worker reads without changing the worker dataset or tensor-builder
    contracts.
    """

    source_indices: np.ndarray
    batch_index: int
    seed: int
    epoch: int


class PerturbationPairSampler:
    """Sample same-dataset/context source-target perturbation pairs.

    Optional source and target candidate pools let callers restrict which
    corpus-global rows can appear as sampled sources and paired targets while
    preserving the existing same-dataset and same-context pairing invariants.
    """

    def __init__(
        self,
        metadata_index: MetadataIndex,
        *,
        batch_size: int,
        config: PertTFAdapterConfig | None = None,
        adapter: "PertTFCorpusAdapter" | None = None,
        seed: int = 0,
        drop_last: bool = True,
        perturbed_target_policy: str = "self_to_control_label",
        missing_target_policy: str = "error",
        source_indices: Sequence[int] | np.ndarray | None = None,
        target_candidate_indices: Sequence[int] | np.ndarray | None = None,
        _null_label_warning_owner: str = "PerturbationPairSampler",
        _null_label_warning_stacklevel: int = 3,
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
        if source_indices is None and target_candidate_indices is not None:
            raise ValueError(
                "target_candidate_indices requires source_indices"
            )

        resolved_config = adapter.config if adapter is not None else (config or PertTFAdapterConfig())
        self._meta = metadata_index
        self.config = resolved_config
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
        normalized_source_indices = _normalize_candidate_row_indices(
            metadata_index,
            source_indices,
        )
        normalized_target_candidate_indices = _normalize_candidate_row_indices(
            metadata_index,
            target_candidate_indices,
        )

        self.total_rows = len(metadata_index)
        self._all_indices = np.arange(self.total_rows, dtype=np.int64)
        if normalized_source_indices is None:
            candidate_source_indices = self._all_indices.copy()
        else:
            candidate_source_indices = normalized_source_indices
        if normalized_target_candidate_indices is None:
            if normalized_source_indices is None:
                candidate_target_candidate_indices = self._all_indices.copy()
            else:
                candidate_target_candidate_indices = candidate_source_indices.copy()
        else:
            candidate_target_candidate_indices = normalized_target_candidate_indices
        candidate_label_row_indices = np.unique(
            np.concatenate(
                [candidate_source_indices, candidate_target_candidate_indices]
            ).astype(np.int64, copy=False)
        )
        label_row_selection = _select_non_null_label_rows(
            metadata_index,
            config=resolved_config,
            row_indices=candidate_label_row_indices,
            warning_owner=_null_label_warning_owner,
            emit_warning=True,
            warning_stacklevel=_null_label_warning_stacklevel,
        )
        source_selection = _select_non_null_label_rows(
            metadata_index,
            config=resolved_config,
            row_indices=candidate_source_indices,
            warning_owner=_null_label_warning_owner,
            emit_warning=False,
            warning_stacklevel=_null_label_warning_stacklevel,
        )
        target_selection = _select_non_null_label_rows(
            metadata_index,
            config=resolved_config,
            row_indices=candidate_target_candidate_indices,
            warning_owner=_null_label_warning_owner,
            emit_warning=False,
            warning_stacklevel=_null_label_warning_stacklevel,
        )
        if source_selection.effective_row_indices.size == 0:
            _raise_empty_effective_rows(_null_label_warning_owner, "source row pool")
        if target_selection.effective_row_indices.size == 0:
            _raise_empty_effective_rows(
                _null_label_warning_owner,
                "target candidate row pool",
            )
        if label_row_selection.effective_row_indices.size == 0:
            _raise_empty_effective_rows(_null_label_warning_owner, "label row pool")
        resolved_labels = adapter._labels if adapter is not None else _PertTFLabelMappings.from_metadata_index(
            metadata_index,
            resolved_config,
            row_indices=label_row_selection.effective_row_indices,
            _emit_null_label_warning=False,
            _null_label_warning_owner=_null_label_warning_owner,
            _null_label_warning_stacklevel=_null_label_warning_stacklevel,
        )
        self._label_mappings = resolved_labels
        self.null_label_filter_stats = label_row_selection.stats
        self.source_null_label_filter_stats = source_selection.stats
        self.target_candidate_null_label_filter_stats = target_selection.stats
        self.effective_label_row_indices = label_row_selection.effective_row_indices.copy()
        self.effective_source_indices = source_selection.effective_row_indices.copy()
        self.effective_target_candidate_indices = (
            target_selection.effective_row_indices.copy()
        )

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

        self._source_indices = self.effective_source_indices.copy()
        self._target_candidate_indices = self.effective_target_candidate_indices.copy()
        self._source_index_mask = np.zeros(self.total_rows, dtype=bool)
        self._source_index_mask[self._source_indices] = True
        self._target_candidate_mask = np.zeros(self.total_rows, dtype=bool)
        self._target_candidate_mask[self._target_candidate_indices] = True
        self._source_row_count = int(self._source_indices.size)
        self._dataset_index = np.asarray(dataset_index, dtype=np.int32)
        relevant_indices = self.effective_label_row_indices.copy()
        relevant_metadata = metadata_index.take(
            relevant_indices,
            columns=[
                self.config.cell_context_column,
                self.config.perturbation_column,
                self.config.batch_column,
            ],
        )
        context_labels_by_row = [""] * self.total_rows
        perturbation_labels_by_row = [""] * self.total_rows
        batch_labels_by_row = [""] * self.total_rows
        cell_context_ids = np.full(self.total_rows, -1, dtype=np.int64)
        perturbation_ids = np.full(self.total_rows, -1, dtype=np.int64)
        batch_ids = np.full(self.total_rows, -1, dtype=np.int64)

        relevant_context_labels = tuple(
            _normalize_label(value, column=self.config.cell_context_column)
            for value in relevant_metadata[self.config.cell_context_column]
        )
        relevant_perturbation_labels = tuple(
            _normalize_label(value, column=self.config.perturbation_column)
            for value in relevant_metadata[self.config.perturbation_column]
        )
        relevant_batch_labels = tuple(
            _normalize_label(value, column=self.config.batch_column)
            for value in relevant_metadata[self.config.batch_column]
        )
        try:
            relevant_context_ids = self._label_mappings.cell_context.encode_many(
                relevant_context_labels
            )
            relevant_perturbation_ids = self._label_mappings.perturbation.encode_many(
                relevant_perturbation_labels
            )
            relevant_batch_ids = self._label_mappings.batch.encode_many(relevant_batch_labels)
        except KeyError as exc:
            raise ValueError(
                "adapter mappings are missing labels required by the effective "
                "source/target row pools"
            ) from exc

        for offset, row_idx in enumerate(relevant_indices.tolist()):
            context_labels_by_row[row_idx] = relevant_context_labels[offset]
            perturbation_labels_by_row[row_idx] = relevant_perturbation_labels[offset]
            batch_labels_by_row[row_idx] = relevant_batch_labels[offset]
            cell_context_ids[row_idx] = int(relevant_context_ids[offset])
            perturbation_ids[row_idx] = int(relevant_perturbation_ids[offset])
            batch_ids[row_idx] = int(relevant_batch_ids[offset])

        self._context_labels = tuple(context_labels_by_row)
        self._perturbation_labels = tuple(perturbation_labels_by_row)
        self._batch_labels = tuple(batch_labels_by_row)
        self._cell_context_ids = cell_context_ids
        self._perturbation_ids = perturbation_ids
        self._batch_ids = batch_ids
        self._pool_by_key = self._build_pool_by_key()
        self._perturbations_by_context = self._build_perturbations_by_context()
        self._control_pool_by_context = self._build_control_pool_by_context()

    def __len__(self) -> int:
        if self.drop_last:
            return self._source_row_count // self.batch_size
        return (self._source_row_count + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        for _, pair_batch in self._iter_planned_batches():
            yield pair_batch

    def _iter_planned_batches(
        self,
    ) -> Iterator[tuple[_PertTFPairBatchPlan, PerturbationPairBatch]]:
        for batch_plan in self._iter_batch_plans():
            yield batch_plan, self.pair_source_indices(
                batch_plan.source_indices,
                seed=batch_plan.seed,
            )

    def _iter_batch_plans(self) -> Iterator[_PertTFPairBatchPlan]:
        if self._source_row_count == 0:
            return
        epoch = int(self.epoch)
        source_order = self._source_indices.copy()
        if self._source_row_count > 1:
            rng = np.random.default_rng(self._shuffle_seed(epoch))
            rng.shuffle(source_order)
        for batch_index, start in enumerate(
            range(0, self._source_row_count, self.batch_size)
        ):
            source_indices = source_order[start : start + self.batch_size]
            if len(source_indices) < self.batch_size and self.drop_last:
                continue
            yield _PertTFPairBatchPlan(
                source_indices=source_indices.copy(),
                batch_index=batch_index,
                seed=self._batch_seed(epoch, batch_index),
                epoch=epoch,
            )

    def _shuffle_seed(self, epoch: int) -> int:
        return _derive_perttf_stream_seed(
            self.seed,
            epoch=epoch,
            batch_index=0,
            stream_id=0,
        )

    def _batch_seed(self, epoch: int, batch_index: int) -> int:
        return _derive_perttf_stream_seed(
            self.seed,
            epoch=epoch,
            batch_index=batch_index,
            stream_id=1,
        )

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
        if not np.all(self._source_index_mask[source_array]):
            raise ValueError(
                "source_indices must come from the configured source_indices pool"
            )

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
        for row_idx in self._target_candidate_indices.tolist():
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
            if not self._target_candidate_mask[source_idx]:
                return self._handle_missing_target(
                    source_idx,
                    reason=(
                        "self_to_control_label target row is not present in the "
                        "configured target pool"
                    ),
                )
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
        target_perturbation_ids = self._label_mappings.perturbation.encode_many(
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


@dataclass(frozen=True)
class _PertTFPairReadRequest:
    """Compact worker-facing request for one paired raw-expression read."""

    pair_batch: PerturbationPairBatch
    batch_index: int
    seed: int
    epoch: int

    @property
    def request_index(self) -> int:
        """Stable per-epoch request position for future sharding wrappers."""

        return self.batch_index


class _PertTFPairReadBatchSampler:
    """Wrap ``PerturbationPairSampler`` into pre-batched raw-read requests."""

    def __init__(self, pair_sampler: PerturbationPairSampler) -> None:
        self._pair_sampler = pair_sampler

    def __len__(self) -> int:
        return len(self._pair_sampler)

    def set_epoch(self, epoch: int) -> None:
        self._pair_sampler.set_epoch(epoch)

    def __iter__(self):
        for batch_plan, pair_batch in self._pair_sampler._iter_planned_batches():
            yield [
                _PertTFPairReadRequest(
                    pair_batch=pair_batch,
                    batch_index=batch_plan.batch_index,
                    seed=batch_plan.seed,
                    epoch=batch_plan.epoch,
                )
            ]


class _PertTFPairExpressionDataset:
    """Worker-light dataset that reads paired source/target raw expression only."""

    def __init__(
        self,
        expression_reader: Any,
        *,
        routing_table: DatasetRoutingTable,
        topology: str = "aggregate",
        backend: str = "lance",
    ) -> None:
        self._reader = expression_reader
        self._routing_table = routing_table
        self._topology = topology
        self._backend = backend

    @property
    def routing_table(self) -> DatasetRoutingTable:
        return self._routing_table

    @property
    def topology(self) -> str:
        return self._topology

    @property
    def backend(self) -> str:
        return self._backend

    def __len__(self) -> int:
        return self._routing_table.total_rows

    def __getitems__(
        self,
        requests: Sequence[_PertTFPairReadRequest],
    ) -> list[dict[str, Any]]:
        request = _unwrap_single_pair_read_request(requests)
        pair_batch = request.pair_batch
        return [
            {
                "request": request,
                "source_raw": self._read_raw_batch(pair_batch.source_indices),
                "target_raw": self._read_raw_batch(pair_batch.target_indices),
            }
        ]

    def _read_raw_batch(self, indices: np.ndarray) -> dict[str, Any]:
        index_array = np.asarray(indices, dtype=np.int64)
        expr = self._reader.read_expression_flat(index_array.tolist())
        dataset_index = self._routing_table.resolve_dataset_indices(index_array)
        batch: dict[str, Any] = {
            "batch_size": expr.batch_size,
            "global_row_index": expr.global_row_index,
            "dataset_index": dataset_index,
            "row_offsets": expr.row_offsets,
            "expressed_gene_indices": expr.expressed_gene_indices,
            "expression_counts": expr.expression_counts,
        }
        size_factor = self._routing_table.take_size_factor(index_array)
        if size_factor is not None:
            batch["size_factor"] = size_factor
        return batch


def _unwrap_single_pair_read_request(
    requests: Sequence[_PertTFPairReadRequest],
) -> _PertTFPairReadRequest:
    if not requests:
        raise ValueError("paired read dataset received empty request batch")
    if len(requests) != 1:
        raise ValueError(
            "paired read dataset expected a single pre-batched request, "
            f"got {len(requests)}"
        )
    request = requests[0]
    if not isinstance(request, _PertTFPairReadRequest):
        raise TypeError(
            "paired read dataset expected _PertTFPairReadRequest items, "
            f"got {type(request)!r}"
        )
    return request


def _unwrap_single_prebatched_pair_item(
    items: list[dict[str, Any]],
    *,
    collate_name: str,
) -> dict[str, Any]:
    if not items:
        raise ValueError(f"{collate_name} received empty list")
    if len(items) != 1:
        raise ValueError(
            f"{collate_name} expected a single pre-batched item, got {len(items)}"
        )
    return items[0]


def _collate_expression_like_raw_batch(raw_batch: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "batch_size": raw_batch["batch_size"],
        "global_row_index": torch.as_tensor(
            raw_batch["global_row_index"],
            dtype=torch.long,
        ),
        "dataset_index": torch.as_tensor(
            raw_batch["dataset_index"],
            dtype=torch.long,
        ),
        "row_offsets": torch.as_tensor(raw_batch["row_offsets"], dtype=torch.long),
        "expressed_gene_indices": torch.as_tensor(
            raw_batch["expressed_gene_indices"],
            dtype=torch.long,
        ),
        "expression_counts": torch.as_tensor(
            raw_batch["expression_counts"],
            dtype=torch.float32,
        ),
    }
    if "size_factor" in raw_batch:
        result["size_factor"] = torch.as_tensor(
            raw_batch["size_factor"],
            dtype=torch.float32,
        )
    return result


def _collate_perttf_raw_pair_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Unwrap one paired raw-read item for main-process pertTF assembly."""

    batch = _unwrap_single_prebatched_pair_item(
        items,
        collate_name="_collate_perttf_raw_pair_batch",
    )
    return {
        "request": batch["request"],
        "source_raw": _collate_expression_like_raw_batch(batch["source_raw"]),
        "target_raw": _collate_expression_like_raw_batch(batch["target_raw"]),
    }


def _validate_perttf_loader_kwargs(
    *,
    num_workers: int,
    multiprocessing_context: str | None,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> dict[str, Any]:
    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers must be >= 0")
    if not isinstance(pin_memory, bool):
        raise TypeError("pin_memory must be a bool")
    if not isinstance(persistent_workers, bool):
        raise TypeError("persistent_workers must be a bool")

    if workers == 0:
        if multiprocessing_context is not None:
            raise ValueError("multiprocessing_context requires num_workers > 0")
        if persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")
        if prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0")
        normalized_context = None
        normalized_prefetch = None
    else:
        if multiprocessing_context is None:
            normalized_context = None
        else:
            normalized_context = str(multiprocessing_context).strip().lower()
            if normalized_context not in {"fork", "spawn", "forkserver"}:
                raise ValueError(
                    "multiprocessing_context must be one of 'fork', 'spawn', "
                    f"or 'forkserver', got {multiprocessing_context!r}"
                )
        if prefetch_factor is None:
            normalized_prefetch = None
        else:
            normalized_prefetch = int(prefetch_factor)
            if normalized_prefetch <= 0:
                raise ValueError("prefetch_factor must be positive")

    return {
        "num_workers": workers,
        "multiprocessing_context": normalized_context,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": normalized_prefetch,
    }


def _build_perttf_loader_kwargs(
    *,
    num_workers: int,
    multiprocessing_context: str | None,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    backend: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0:
        kwargs["multiprocessing_context"] = (
            multiprocessing_context
            if multiprocessing_context is not None
            else ("spawn" if backend == "lance" else None)
        )
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


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
        stoi = self.adapter.to_simple_vocab_stoi()
        self._pad_token_id = int(stoi[self.config.pad_token])
        self._cls_token_id = int(stoi[self.config.cls_token])
        self._special_token_offset = self.adapter.special_token_offset

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
        source_raw = self.corpus.inspect_batch(pair_batch.source_indices)
        target_raw = self.corpus.inspect_batch(pair_batch.target_indices)

        return self.build_from_raw_pair_batch(
            pair_batch,
            source_raw,
            target_raw,
            seed=seed,
            sampled_gene_ids=sampled_gene_ids,
            sampling_mode=sampling_mode,
            expressed_weight=expressed_weight,
            hvg_weight=hvg_weight,
            hvg_top_k=hvg_top_k,
        )

    def build_from_raw_pair_batch(
        self,
        pair_batch: PerturbationPairBatch,
        source_raw: dict[str, Any],
        target_raw: dict[str, Any],
        *,
        seed: int | None = None,
        sampled_gene_ids: torch.Tensor | None = None,
        sampling_mode: str = "hvg",
        expressed_weight: float = 3.0,
        hvg_weight: float = 3.0,
        hvg_top_k: int | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_pair_batch(pair_batch)
        self._validate_raw_batch_alignment(
            "source_raw",
            source_raw,
            expected_indices=pair_batch.source_indices,
            expected_dataset_indices=pair_batch.source_dataset_indices,
        )
        self._validate_raw_batch_alignment(
            "target_raw",
            target_raw,
            expected_indices=pair_batch.target_indices,
            expected_dataset_indices=pair_batch.target_dataset_indices,
        )

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

    def _validate_raw_batch_alignment(
        self,
        name: str,
        raw_batch: dict[str, Any],
        *,
        expected_indices: np.ndarray,
        expected_dataset_indices: np.ndarray,
    ) -> None:
        batch_size = int(raw_batch.get("batch_size", -1))
        expected_size = int(expected_indices.shape[0])
        if batch_size != expected_size:
            raise ValueError(
                f"{name} batch_size {batch_size} does not match paired batch size {expected_size}"
            )

        raw_indices = self._raw_batch_numpy(raw_batch, "global_row_index", dtype=np.int64)
        if raw_indices.shape != expected_indices.shape or not np.array_equal(
            raw_indices,
            expected_indices,
        ):
            raise ValueError(
                f"{name} global_row_index does not match pair_batch ordering"
            )

        raw_dataset_indices = self._raw_batch_numpy(
            raw_batch,
            "dataset_index",
            dtype=np.int64,
        )
        expected_dataset_indices64 = np.asarray(expected_dataset_indices, dtype=np.int64)
        if (
            raw_dataset_indices.shape != expected_dataset_indices64.shape
            or not np.array_equal(raw_dataset_indices, expected_dataset_indices64)
        ):
            raise ValueError(
                f"{name} dataset_index does not match pair_batch dataset routing"
            )

    def _raw_batch_numpy(
        self,
        raw_batch: dict[str, Any],
        key: str,
        *,
        dtype: Any,
    ) -> np.ndarray:
        if key not in raw_batch:
            raise KeyError(f"raw batch is missing required key {key!r}")
        values = raw_batch[key]
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(dtype, copy=False)
        return np.asarray(values, dtype=dtype)

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
            self.adapter.gene_token_ids,
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
    _vocab: _PertTFVocabMapping
    _labels: _PertTFLabelMappings
    null_label_filter_stats: PertTFNullLabelFilterStats | None = None

    @classmethod
    def _from_parts(
        cls,
        *,
        feature_registry: FeatureRegistry,
        config: PertTFAdapterConfig,
        label_mappings: _PertTFLabelMappings,
        null_label_filter_stats: PertTFNullLabelFilterStats | None = None,
    ) -> "PertTFCorpusAdapter":
        return cls(
            config=config,
            _vocab=_PertTFVocabMapping.from_feature_registry(
                feature_registry,
                special_tokens=config.special_tokens,
            ),
            _labels=label_mappings,
            null_label_filter_stats=(
                label_mappings.null_label_filter_stats
                if null_label_filter_stats is None
                else null_label_filter_stats
            ),
        )

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        config: PertTFAdapterConfig | None = None,
        row_indices: Sequence[int] | np.ndarray | None = None,
        *,
        _emit_null_label_warning: bool = True,
        _null_label_warning_owner: str = "PertTFCorpusAdapter",
        _null_label_warning_stacklevel: int = 3,
    ) -> "PertTFCorpusAdapter":
        resolved = config or PertTFAdapterConfig()
        selection = _select_non_null_label_rows(
            corpus.metadata_index,
            config=resolved,
            row_indices=row_indices,
            warning_owner=_null_label_warning_owner,
            emit_warning=_emit_null_label_warning,
            warning_stacklevel=_null_label_warning_stacklevel,
        )
        if selection.effective_row_indices.size == 0:
            _raise_empty_effective_rows(_null_label_warning_owner, "label row pool")
        return cls._from_parts(
            feature_registry=corpus.feature_registry,
            config=resolved,
            label_mappings=_PertTFLabelMappings.from_metadata_index(
                corpus.metadata_index,
                resolved,
                row_indices=selection.effective_row_indices,
                _emit_null_label_warning=False,
                _null_label_warning_owner=_null_label_warning_owner,
                _null_label_warning_stacklevel=_null_label_warning_stacklevel,
            ),
            null_label_filter_stats=selection.stats,
        )

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return self._vocab.special_tokens

    @property
    def feature_ids_by_global_id(self) -> tuple[str, ...]:
        return self._vocab.feature_ids_by_global_id

    @property
    def special_token_offset(self) -> int:
        return self._vocab.special_token_offset

    @property
    def gene_token_ids(self) -> np.ndarray:
        return self._vocab.gene_token_ids

    @property
    def tokens_in_order(self) -> tuple[str, ...]:
        return self._vocab.tokens_in_order

    def token_id_for_global_id(self, global_id: int) -> int:
        return self._vocab.token_id_for_global_id(global_id)

    def global_id_for_token_id(self, token_id: int) -> int | None:
        return self._vocab.global_id_for_token_id(token_id)

    def feature_id_for_token_id(self, token_id: int) -> str:
        return self._vocab.feature_id_for_token_id(token_id)

    def to_simple_vocab_stoi(self) -> dict[str, int]:
        return self._vocab.to_simple_vocab_stoi()

    def to_simple_vocab_json(self, path: str | Path) -> None:
        self._vocab.to_simple_vocab_json(path)

    @property
    def cell_context_labels(self) -> tuple[str, ...]:
        return self._labels.cell_context.labels

    @property
    def perturbation_labels(self) -> tuple[str, ...]:
        return self._labels.perturbation.labels

    @property
    def batch_labels(self) -> tuple[str, ...]:
        return self._labels.batch.labels

    @property
    def cell_context_to_index(self) -> dict[str, int]:
        return self._labels.cell_context.label_to_index

    @property
    def perturbation_to_index(self) -> dict[str, int]:
        return self._labels.perturbation.label_to_index

    @property
    def batch_to_index(self) -> dict[str, int]:
        return self._labels.batch.label_to_index

    @property
    def control_label_ids(self) -> tuple[int, ...]:
        return self._labels.control_label_ids

    def encode_cell_context(self, label: Any) -> int:
        return self._labels.cell_context.encode(label)

    def encode_cell_context_many(self, labels: Sequence[Any]) -> np.ndarray:
        return self._labels.cell_context.encode_many(labels)

    def encode_perturbation(self, label: Any) -> int:
        return self._labels.perturbation.encode(label)

    def encode_perturbation_many(self, labels: Sequence[Any]) -> np.ndarray:
        return self._labels.perturbation.encode_many(labels)

    def encode_batch(self, label: Any) -> int:
        return self._labels.batch.encode(label)

    def encode_batch_many(self, labels: Sequence[Any]) -> np.ndarray:
        return self._labels.batch.encode_many(labels)

    def to_reference_dict(self) -> dict[str, Any]:
        return {
            "genes": list(self.feature_ids_by_global_id),
            "gene_token_ids": self.gene_token_ids.copy(),
            "simple_vocab_stoi": self.to_simple_vocab_stoi(),
            **self._labels.to_reference_dict(),
        }


class PertTFPairedBatchLoader:
    """Slim public iterable that yields final pertTF paired batches."""

    def __init__(
        self,
        corpus: Corpus,
        *,
        batch_size: int,
        seq_len: int,
        config: PertTFAdapterConfig | None = None,
        adapter: PertTFCorpusAdapter | None = None,
        row_indices: Sequence[int] | np.ndarray | None = None,
        seed: int = 0,
        drop_last: bool = True,
        perturbed_target_policy: str = "self_to_control_label",
        missing_target_policy: str = "error",
        source_indices: Sequence[int] | np.ndarray | None = None,
        target_candidate_indices: Sequence[int] | np.ndarray | None = None,
        sampling_mode: str = "hvg",
        expressed_weight: float = 3.0,
        hvg_weight: float = 3.0,
        hvg_top_k: int | None = None,
        num_workers: int = 0,
        multiprocessing_context: str | None = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        device: torch.device | str | None = "cpu",
    ) -> None:
        self.corpus = corpus
        resolved_config = adapter.config if adapter is not None else (config or PertTFAdapterConfig())
        normalized_row_indices = _normalize_candidate_row_indices(
            corpus.metadata_index,
            row_indices,
        )
        resolved_source_indices = source_indices
        if resolved_source_indices is None and normalized_row_indices is not None:
            resolved_source_indices = normalized_row_indices.copy()
        resolved_target_candidate_indices = target_candidate_indices
        if resolved_target_candidate_indices is None:
            resolved_target_candidate_indices = resolved_source_indices
        self.pair_sampler = PerturbationPairSampler(
            corpus.metadata_index,
            batch_size=batch_size,
            config=resolved_config,
            adapter=adapter,
            seed=seed,
            drop_last=drop_last,
            perturbed_target_policy=perturbed_target_policy,
            missing_target_policy=missing_target_policy,
            source_indices=resolved_source_indices,
            target_candidate_indices=resolved_target_candidate_indices,
            _null_label_warning_owner="PertTFPairedBatchLoader",
            _null_label_warning_stacklevel=4,
        )
        resolved_adapter = adapter
        if resolved_adapter is None:
            resolved_adapter = PertTFCorpusAdapter._from_parts(
                feature_registry=corpus.feature_registry,
                config=resolved_config,
                label_mappings=self.pair_sampler._label_mappings,
                null_label_filter_stats=self.pair_sampler.null_label_filter_stats,
            )
        self._builder = PertTFPairedBatchBuilder(
            corpus,
            seq_len=seq_len,
            config=resolved_config,
            adapter=resolved_adapter,
            device=device,
        )
        self.adapter = self._builder.adapter
        self.config = self._builder.config
        self.row_indices = None if normalized_row_indices is None else normalized_row_indices.copy()
        self.effective_label_row_indices = self.pair_sampler.effective_label_row_indices.copy()
        self.effective_source_indices = self.pair_sampler.effective_source_indices.copy()
        self.effective_target_candidate_indices = (
            self.pair_sampler.effective_target_candidate_indices.copy()
        )
        self.null_label_filter_stats = self.pair_sampler.null_label_filter_stats
        self.source_null_label_filter_stats = self.pair_sampler.source_null_label_filter_stats
        self.target_candidate_null_label_filter_stats = (
            self.pair_sampler.target_candidate_null_label_filter_stats
        )
        self._request_sampler = _PertTFPairReadBatchSampler(self.pair_sampler)
        self._dataset = _PertTFPairExpressionDataset(
            corpus.expression_reader,
            routing_table=corpus.routing_table,
            topology=corpus.topology,
            backend=corpus.backend,
        )
        validated_loader = _validate_perttf_loader_kwargs(
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        self._loader_kwargs = _build_perttf_loader_kwargs(
            backend=corpus.backend,
            **validated_loader,
        )
        self._sampling_mode = sampling_mode
        self._expressed_weight = float(expressed_weight)
        self._hvg_weight = float(hvg_weight)
        self._hvg_top_k = hvg_top_k
        self._data_loader = DataLoader(
            self._dataset,
            batch_sampler=self._request_sampler,
            collate_fn=_collate_perttf_raw_pair_batch,
            **self._loader_kwargs,
        )

    def __len__(self) -> int:
        return len(self._request_sampler)

    @property
    def epoch(self) -> int:
        return int(self.pair_sampler.epoch)

    def set_epoch(self, epoch: int) -> None:
        self._request_sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for raw_pair_batch in self._data_loader:
            request = raw_pair_batch["request"]
            yield self._builder.build_from_raw_pair_batch(
                request.pair_batch,
                raw_pair_batch["source_raw"],
                raw_pair_batch["target_raw"],
                seed=request.seed,
                sampling_mode=self._sampling_mode,
                expressed_weight=self._expressed_weight,
                hvg_weight=self._hvg_weight,
                hvg_top_k=self._hvg_top_k,
            )
