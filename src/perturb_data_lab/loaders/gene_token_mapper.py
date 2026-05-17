"""Map corpus-global gene IDs to model token IDs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .feature_registry import FeatureRegistry

__all__ = ["GeneTokenMapper"]

DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<cls>", "<unk>", "<eos>")


def _check_unique_strings(values: Sequence[str], *, field_name: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")


def _normalize_stoi(stoi: Mapping[str, int]) -> dict[str, int]:
    if not stoi:
        raise ValueError("tokenizer_stoi must not be empty")
    normalized: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in stoi.items():
        if not isinstance(token, str) or not token:
            raise ValueError("tokenizer_stoi keys must be non-empty strings")
        resolved_id = int(token_id)
        if resolved_id < 0:
            raise ValueError("tokenizer_stoi token IDs must be non-negative")
        if resolved_id in seen_ids:
            raise ValueError(f"duplicate tokenizer token ID {resolved_id}")
        normalized[token] = resolved_id
        seen_ids.add(resolved_id)
    return normalized


def _tokens_by_id(stoi: Mapping[str, int]) -> tuple[str, ...]:
    return tuple(token for token, _ in sorted(stoi.items(), key=lambda item: item[1]))


@dataclass
class GeneTokenMapper:
    """Dense lookup from corpus-global gene IDs to model token IDs."""

    feature_ids_by_global_id: tuple[str, ...]
    tokens_in_order: tuple[str, ...]
    stoi: dict[str, int]
    global_to_token_id: np.ndarray
    tokenizable_by_global_id: np.ndarray
    pad_token_id: int
    cls_token_id: int | None = None
    unk_token_id: int | None = None
    _device_caches: dict[str, dict[str, torch.Tensor]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        n_genes = len(self.feature_ids_by_global_id)
        self.global_to_token_id = np.asarray(self.global_to_token_id, dtype=np.int64)
        self.tokenizable_by_global_id = np.asarray(self.tokenizable_by_global_id, dtype=bool)
        if self.global_to_token_id.shape != (n_genes,):
            raise ValueError("global_to_token_id must have one entry per global gene")
        if self.tokenizable_by_global_id.shape != (n_genes,):
            raise ValueError("tokenizable_by_global_id must have one entry per global gene")

    @classmethod
    def from_feature_registry(
        cls,
        feature_registry: FeatureRegistry,
        *,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
        pad_token: str = "<pad>",
        cls_token: str | None = "<cls>",
        unk_token: str | None = "<unk>",
    ) -> "GeneTokenMapper":
        """Build a de novo tokenizer from corpus global feature IDs."""
        _check_unique_strings(special_tokens, field_name="special_tokens")
        normalized_specials = tuple(special_tokens)
        feature_ids = feature_registry.global_feature_ids
        overlap = set(normalized_specials).intersection(feature_ids)
        if overlap:
            raise ValueError(
                "feature IDs overlap reserved special tokens: "
                f"{sorted(overlap)}"
            )
        tokens_in_order = normalized_specials + feature_ids
        stoi = {token: token_id for token_id, token in enumerate(tokens_in_order)}
        if pad_token not in stoi:
            raise ValueError(f"pad_token={pad_token!r} is missing from special_tokens")
        if cls_token is not None and cls_token not in stoi:
            raise ValueError(f"cls_token={cls_token!r} is missing from special_tokens")
        if unk_token is not None and unk_token not in stoi:
            raise ValueError(f"unk_token={unk_token!r} is missing from special_tokens")
        offset = len(normalized_specials)
        return cls(
            feature_ids_by_global_id=feature_ids,
            tokens_in_order=tokens_in_order,
            stoi=stoi,
            global_to_token_id=np.arange(offset, offset + len(feature_ids), dtype=np.int64),
            tokenizable_by_global_id=np.ones(len(feature_ids), dtype=bool),
            pad_token_id=stoi[pad_token],
            cls_token_id=None if cls_token is None else stoi[cls_token],
            unk_token_id=None if unk_token is None else stoi[unk_token],
        )

    @classmethod
    def from_tokenizer_stoi(
        cls,
        feature_registry: FeatureRegistry,
        tokenizer_stoi: Mapping[str, int],
        *,
        pad_token: str = "<pad>",
        cls_token: str | None = None,
        unk_token: str | None = None,
        feature_id_to_token: Mapping[str, str] | None = None,
    ) -> "GeneTokenMapper":
        """Build lookup tables from an external token string -> token ID map."""
        stoi = _normalize_stoi(tokenizer_stoi)
        if pad_token not in stoi:
            raise ValueError(f"pad_token={pad_token!r} is missing from tokenizer_stoi")
        if cls_token is not None and cls_token not in stoi:
            raise ValueError(f"cls_token={cls_token!r} is missing from tokenizer_stoi")
        if unk_token is not None and unk_token not in stoi:
            raise ValueError(f"unk_token={unk_token!r} is missing from tokenizer_stoi")

        feature_ids = feature_registry.global_feature_ids
        global_to_token = np.full(len(feature_ids), int(stoi[pad_token]), dtype=np.int64)
        tokenizable = np.zeros(len(feature_ids), dtype=bool)
        for global_id, feature_id in enumerate(feature_ids):
            token = (
                feature_id_to_token.get(feature_id)
                if feature_id_to_token is not None
                else feature_id
            )
            if token in stoi:
                global_to_token[global_id] = int(stoi[token])
                tokenizable[global_id] = True

        return cls(
            feature_ids_by_global_id=feature_ids,
            tokens_in_order=_tokens_by_id(stoi),
            stoi=stoi,
            global_to_token_id=global_to_token,
            tokenizable_by_global_id=tokenizable,
            pad_token_id=stoi[pad_token],
            cls_token_id=None if cls_token is None else stoi[cls_token],
            unk_token_id=None if unk_token is None else stoi[unk_token],
        )

    @classmethod
    def from_tokenizer_json(
        cls,
        feature_registry: FeatureRegistry,
        path: str | Path,
        **kwargs: Any,
    ) -> "GeneTokenMapper":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        stoi = data.get("stoi", data)
        return cls.from_tokenizer_stoi(feature_registry, stoi, **kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> "GeneTokenMapper":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            feature_ids_by_global_id=tuple(data["feature_ids_by_global_id"]),
            tokens_in_order=tuple(data["tokens_in_order"]),
            stoi={token: int(token_id) for token, token_id in data["stoi"].items()},
            global_to_token_id=np.asarray(data["global_to_token_id"], dtype=np.int64),
            tokenizable_by_global_id=np.asarray(data["tokenizable_by_global_id"], dtype=bool),
            pad_token_id=int(data["pad_token_id"]),
            cls_token_id=None if data.get("cls_token_id") is None else int(data["cls_token_id"]),
            unk_token_id=None if data.get("unk_token_id") is None else int(data["unk_token_id"]),
        )

    def to_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_ids_by_global_id": list(self.feature_ids_by_global_id),
            "tokens_in_order": list(self.tokens_in_order),
            "stoi": dict(self.stoi),
            "global_to_token_id": self.global_to_token_id.tolist(),
            "tokenizable_by_global_id": self.tokenizable_by_global_id.tolist(),
            "pad_token_id": int(self.pad_token_id),
            "cls_token_id": self.cls_token_id,
            "unk_token_id": self.unk_token_id,
        }

    def check_feature_registry(self, feature_registry: FeatureRegistry) -> None:
        if self.feature_ids_by_global_id != feature_registry.global_feature_ids:
            raise ValueError("GeneTokenMapper was built for a different global gene order")

    def tensors_for_device(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        resolved = torch.device(device)
        key = str(resolved)
        if key not in self._device_caches:
            self._device_caches[key] = {
                "global_to_token_id": torch.as_tensor(
                    self.global_to_token_id,
                    dtype=torch.long,
                    device=resolved,
                ),
                "tokenizable_by_global_id": torch.as_tensor(
                    self.tokenizable_by_global_id,
                    dtype=torch.bool,
                    device=resolved,
                ),
            }
        return self._device_caches[key]

    def encode_global_ids(
        self,
        global_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized ``global gene ID -> model token ID`` encoding."""
        cached = self.tensors_for_device(global_ids.device)
        vocab_size = len(self.global_to_token_id)
        safe_ids = global_ids.clamp(0, vocab_size - 1)
        tokenizable = cached["tokenizable_by_global_id"].gather(
            0,
            safe_ids.reshape(-1),
        ).reshape_as(global_ids)
        in_vocab = (global_ids >= 0) & (global_ids < vocab_size)
        token_valid_mask = valid_mask & in_vocab & tokenizable
        token_ids = cached["global_to_token_id"].gather(
            0,
            safe_ids.reshape(-1),
        ).reshape_as(global_ids)
        token_ids = torch.where(
            token_valid_mask,
            token_ids,
            torch.full_like(token_ids, int(self.pad_token_id)),
        )
        return token_ids, token_valid_mask

    def coverage_summary(self, feature_registry: FeatureRegistry | None = None) -> dict[str, Any]:
        tokenizable = self.tokenizable_by_global_id
        summary: dict[str, Any] = {
            "global_vocab_size": len(tokenizable),
            "tokenizable_genes": int(tokenizable.sum()),
            "missing_genes": int((~tokenizable).sum()),
        }
        if feature_registry is not None:
            self.check_feature_registry(feature_registry)
            summary["tokenizable_by_dataset"] = {
                dataset_id: int((feature_registry.dataset_has_gene[idx] & tokenizable).sum())
                for idx, dataset_id in enumerate(feature_registry.dataset_ids)
            }
        return summary
