"""Feature registry for dataset-local to corpus-global gene coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl

__all__ = ["FeatureRegistry"]

_UNMAPPED: int = -1


class FeatureRegistry:
    """Map per-dataset local gene indices to shared global gene IDs.

    The gene tokenizer owns the global canonical-gene ordering. This registry
    turns each dataset's ``origin_index`` coordinates into those tokenizer IDs
    and keeps optional canonical HVG ranks for loader-side sampling.
    """

    @classmethod
    def from_canonical_var_parquets(
        cls,
        named_var_paths: Mapping[str, str | Path],
        *,
        global_id_by_feature_id: Mapping[str, int],
        dataset_order: Sequence[str] | None = None,
    ) -> "FeatureRegistry":
        order = list(dataset_order) if dataset_order is not None else list(named_var_paths.keys())
        named_var_dfs = {
            dataset_id: pl.read_parquet(str(named_var_paths[dataset_id])).sort("origin_index")
            for dataset_id in order
        }
        named_hvg_rank_dfs = {
            dataset_id: pl.read_parquet(str(hvg_path))
            for dataset_id in order
            if (hvg_path := cls._discover_hvg_path(Path(named_var_paths[dataset_id]))) is not None
        }
        return cls(
            named_var_dfs,
            dataset_order=order,
            global_id_by_feature_id=global_id_by_feature_id,
            named_hvg_rank_dfs=named_hvg_rank_dfs or None,
        )

    @staticmethod
    def _discover_hvg_path(var_path: Path) -> Path | None:
        meta_root = var_path.parent.parent if var_path.parent.name == "canonical_meta" else var_path.parent
        hvg_path = meta_root / "hvg.parquet"
        return hvg_path if hvg_path.exists() else None

    def __init__(
        self,
        named_var_dfs: Mapping[str, pl.DataFrame],
        *,
        global_id_by_feature_id: Mapping[str, int],
        dataset_order: Sequence[str] | None = None,
        named_hvg_rank_dfs: Mapping[str, pl.DataFrame] | None = None,
    ) -> None:
        if not named_var_dfs:
            raise ValueError("named_var_dfs must not be empty")

        self._dataset_ids = list(dataset_order) if dataset_order is not None else list(named_var_dfs.keys())
        if set(self._dataset_ids) != set(named_var_dfs.keys()):
            raise ValueError("dataset_order must contain exactly the named_var_dfs keys")
        self._n_datasets = len(self._dataset_ids)

        self._feature_id_to_global = dict(global_id_by_feature_id)
        ids = sorted(self._feature_id_to_global.values())
        if ids != list(range(len(ids))):
            raise ValueError("global_id_by_feature_id must define contiguous IDs 0..N-1")
        self._global_vocab_size = len(self._feature_id_to_global)

        self._local_to_global: dict[int, np.ndarray] = {}
        self._dataset_default_n_hvg: dict[int, int | None] = {}
        self._dataset_local_hvg_rank: dict[int, np.ndarray | None] = {}

        for ds_idx, ds_id in enumerate(self._dataset_ids):
            var_df = named_var_dfs[ds_id].sort("origin_index")
            self._validate_var_df(var_df, ds_id)
            canonical_gene_ids = var_df["canonical_gene_id"].to_list()
            mapping = np.empty(len(var_df), dtype=np.int32)
            for local_idx, canonical_gene_id in enumerate(canonical_gene_ids):
                if canonical_gene_id not in self._feature_id_to_global:
                    raise ValueError(
                        f"canonical_gene_id {canonical_gene_id!r} missing from persisted gene tokenizer"
                    )
                mapping[local_idx] = int(self._feature_id_to_global[canonical_gene_id])
            self._local_to_global[ds_idx] = mapping

            default_n_hvg = None
            hvg_rank = None
            if named_hvg_rank_dfs is not None and ds_id in named_hvg_rank_dfs:
                hvg_rank, inferred_default_n_hvg = self._parse_hvg_ranking_df(
                    named_hvg_rank_dfs[ds_id],
                    ds_id=ds_id,
                    n_vars=len(var_df),
                )
                if default_n_hvg is None:
                    default_n_hvg = inferred_default_n_hvg
            self._dataset_default_n_hvg[ds_idx] = default_n_hvg
            self._dataset_local_hvg_rank[ds_idx] = hvg_rank

        self._max_local_vocab = max(len(mapping) for mapping in self._local_to_global.values())
        self._dense_map = self._build_dense_map()
        self._dataset_has_gene: np.ndarray | None = None
        self._dataset_gene_prob: np.ndarray | None = None
        self._hvg_rank_matrix: np.ndarray | None = None
        self._hvg_mask: np.ndarray | None = None

    @staticmethod
    def _validate_var_df(var_df: pl.DataFrame, ds_id: str) -> None:
        missing = {"origin_index", "canonical_gene_id"} - set(var_df.columns)
        if missing:
            raise ValueError(f"canonical var parquet for dataset '{ds_id}' missing columns: {sorted(missing)}")
        n = len(var_df)
        origin_idx = var_df["origin_index"].to_numpy()
        if not np.array_equal(origin_idx, np.arange(n, dtype=origin_idx.dtype)):
            raise ValueError(
                f"var DataFrame for dataset '{ds_id}': origin_index is not contiguous 0..{n - 1}"
            )

    @staticmethod
    def _parse_hvg_ranking_df(
        hvg_df: pl.DataFrame,
        *,
        ds_id: str,
        n_vars: int,
    ) -> tuple[np.ndarray, int | None]:
        missing = {"origin_index", "hvg_rank"} - set(hvg_df.columns)
        if missing:
            raise ValueError(f"hvg.parquet for dataset '{ds_id}' missing columns: {sorted(missing)}")
        origin_index = np.asarray(hvg_df["origin_index"].to_numpy(), dtype=np.int64)
        hvg_rank = np.asarray(hvg_df["hvg_rank"].to_numpy(), dtype=np.int32)
        local_hvg_rank = np.zeros(n_vars, dtype=np.int32)
        valid = (origin_index >= 0) & (origin_index < n_vars) & (hvg_rank > 0)
        if not np.all(valid):
            raise ValueError(f"hvg.parquet for dataset '{ds_id}' contains invalid origin_index or hvg_rank rows")
        local_hvg_rank[origin_index] = hvg_rank
        inferred_default_n_hvg = None
        if "selected_at_default_n_hvg" in hvg_df.columns:
            inferred_default_n_hvg = int(np.asarray(hvg_df["selected_at_default_n_hvg"].to_numpy(), dtype=bool).sum())
        return local_hvg_rank, inferred_default_n_hvg

    def _build_dense_map(self) -> np.ndarray:
        dense = np.full((self._n_datasets, self._max_local_vocab), _UNMAPPED, dtype=np.int32)
        for ds_idx, mapping in self._local_to_global.items():
            dense[ds_idx, : len(mapping)] = mapping
        return dense

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(self._dataset_ids)

    @property
    def n_datasets(self) -> int:
        return self._n_datasets

    @property
    def global_vocab_size(self) -> int:
        return self._global_vocab_size

    @property
    def global_feature_ids(self) -> tuple[str, ...]:
        ordered = [""] * self._global_vocab_size
        for feature_id, global_id in self._feature_id_to_global.items():
            ordered[int(global_id)] = feature_id
        return tuple(ordered)

    @property
    def max_local_vocab(self) -> int:
        return self._max_local_vocab

    @property
    def local_to_global_map(self) -> np.ndarray:
        return self._dense_map

    @property
    def dataset_has_gene(self) -> np.ndarray:
        if self._dataset_has_gene is None:
            self._compute_gene_masks()
        assert self._dataset_has_gene is not None
        return self._dataset_has_gene

    @property
    def dataset_gene_prob(self) -> np.ndarray:
        if self._dataset_gene_prob is None:
            self._compute_gene_masks()
        assert self._dataset_gene_prob is not None
        return self._dataset_gene_prob

    @property
    def hvg_rank_matrix(self) -> np.ndarray:
        if self._hvg_rank_matrix is None:
            self._build_hvg_rank_matrix()
        assert self._hvg_rank_matrix is not None
        return self._hvg_rank_matrix

    @property
    def hvg_mask(self) -> np.ndarray:
        if self._hvg_mask is None:
            ranks = self.hvg_rank_matrix
            mask = np.zeros_like(ranks, dtype=bool)
            for ds_idx, default_n_hvg in self._dataset_default_n_hvg.items():
                if default_n_hvg is not None and default_n_hvg > 0:
                    mask[ds_idx] = (ranks[ds_idx] > 0) & (ranks[ds_idx] <= int(default_n_hvg))
            self._hvg_mask = mask
        return self._hvg_mask

    def _compute_gene_masks(self) -> None:
        has_gene = np.zeros((self._n_datasets, self._global_vocab_size), dtype=bool)
        for ds_idx, mapping in self._local_to_global.items():
            has_gene[ds_idx, mapping[mapping >= 0]] = True
        prob = np.zeros((self._n_datasets, self._global_vocab_size), dtype=np.float32)
        for ds_idx in range(self._n_datasets):
            n_valid = int(has_gene[ds_idx].sum())
            if n_valid > 0:
                prob[ds_idx, has_gene[ds_idx]] = 1.0 / n_valid
        self._dataset_has_gene = has_gene
        self._dataset_gene_prob = prob

    def _build_hvg_rank_matrix(self) -> None:
        matrix = np.zeros((self._n_datasets, self._global_vocab_size), dtype=np.int32)
        for ds_idx, local_ranks in self._dataset_local_hvg_rank.items():
            if local_ranks is None:
                continue
            local_to_global = self._local_to_global[ds_idx]
            ranked = np.flatnonzero(local_ranks > 0)
            matrix[ds_idx, local_to_global[ranked]] = local_ranks[ranked]
        self._hvg_rank_matrix = matrix

    def __len__(self) -> int:
        return self._n_datasets

    def __repr__(self) -> str:
        return (
            f"FeatureRegistry(n_datasets={self._n_datasets}, "
            f"global_vocab={self._global_vocab_size}, "
            f"max_local_vocab={self._max_local_vocab})"
        )
