"""Phase 2: Feature Registry and Local→Global Mapping.

``FeatureRegistry`` maps per-dataset local gene indices to global gene IDs.
It reads per-dataset ``var`` parquet data (``origin_index`` → ``feature_id``),
assigns a global integer ID to each unique ``feature_id``, and produces
GPU-ready dense tensors for local→global resolution.

``GlobalGeneSampler`` uses the registry to sample ``seq_len`` global gene IDs
per cell from each cell's own dataset's valid gene pool (uniform sampling).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureRegistry",
    "GlobalGeneSampler",
]

# Sentinel for unmapped local positions in the dense mapping tensor.
_UNMAPPED: int = -1


# ---------------------------------------------------------------------------
# FeatureRegistry
# ---------------------------------------------------------------------------


class FeatureRegistry:
    """On-the-fly feature registry for cross-dataset gene ID resolution.

    Builds from per-dataset ``var`` DataFrames that contain ``origin_index``
    (dataset-local gene index) and ``feature_id`` (string gene identifier).
    Assigns consecutive global integer IDs to unique ``feature_id`` values
    and produces dense mapping tensors suitable for GPU gather operations.

    When built from canonical var parquets (via ``from_canonical_var_parquets``),
    uses ``canonical_gene_id`` instead of ``feature_id`` for the global
    vocabulary, so ``global_vocab_size`` reflects canonical (harmonized) gene
    identifiers rather than raw feature IDs.

    Parameters
    ----------
    named_var_dfs : dict[str, pl.DataFrame]
        Mapping ``dataset_id → var_df``.  Each DataFrame must have columns
        ``origin_index`` (int) and ``feature_id`` (str).

    Attributes
    ----------
    dataset_ids : tuple[str, ...]
        Ordered dataset identifiers (sorted alphabetically for determinism).
    global_vocab_size : int
        Total number of unique genes across all registered datasets.
    max_local_vocab : int
        Maximum number of local genes (``n_vars``) across all datasets.
    """

    @classmethod
    def from_canonical_var_parquets(
        cls,
        named_var_paths: dict[str, str | Path],
    ) -> "FeatureRegistry":
        """Build a FeatureRegistry from canonical var parquet files.

        Reads ``{release_id}-canonical-var.parquet`` files produced by the
        canonicalization runner (Phase 2).  Uses ``canonical_gene_id``
        instead of raw ``feature_id`` as the global vocabulary key, so
        ``global_vocab_size`` reflects harmonized gene identifiers.

        The canonical var parquet must contain columns: ``origin_index``,
        ``gene_id``, ``canonical_gene_id``, ``global_id``.

        Parameters
        ----------
        named_var_paths : dict[str, str | Path]
            Mapping ``dataset_id → path`` to canonical var parquet files.

        Returns
        -------
        FeatureRegistry
            Registry with canonically-mapped gene identities.
        """
        from pathlib import Path

        named_var_dfs: dict[str, pl.DataFrame] = {}
        for ds_id, path in named_var_paths.items():
            df = pl.read_parquet(str(path))

            required = {"origin_index", "canonical_gene_id"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"Canonical var parquet for dataset '{ds_id}' missing "
                    f"columns: {sorted(missing)}"
                )

            # Canonical parquets store all columns as strings; cast to proper types
            if "origin_index" in df.columns and df["origin_index"].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col("origin_index").cast(pl.Int32)
                )
            if "global_id" in df.columns and df["global_id"].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col("global_id").cast(pl.Int32)
                )

            # Sort by origin_index to ensure contiguity
            df = df.sort("origin_index")

            # Rename canonical_gene_id → feature_id for internal compatibility
            if "canonical_gene_id" in df.columns and "feature_id" not in df.columns:
                df = df.rename({"canonical_gene_id": "feature_id"})

            named_var_dfs[ds_id] = df

        return cls(named_var_dfs)

    def __init__(
        self,
        named_var_dfs: dict[str, pl.DataFrame],
    ):
        if not named_var_dfs:
            raise ValueError("named_var_dfs must not be empty")

        # Sort for deterministic ordering
        self._dataset_ids: list[str] = sorted(named_var_dfs.keys())
        self._n_datasets: int = len(self._dataset_ids)

        # ---- Pass 1: discover all unique feature_ids and assign global IDs ----
        self._feature_id_to_global: dict[str, int] = {}
        self._dataset_var_entries: dict[int, list[dict[str, Any]]] = {}
        # Per-dataset local HVG flags (None = no HVG column found)
        self._dataset_local_hvg: dict[int, np.ndarray | None] = {}

        _hvg_column_names = ("is_hvg", "highly_variable", "hvg", "is_highly_variable")

        for ds_idx, ds_id in enumerate(self._dataset_ids):
            var_df = named_var_dfs[ds_id]
            self._validate_var_df(var_df, ds_id)

            # Detect HVG column (case-insensitive)
            hvg_col: str | None = None
            for candidate in _hvg_column_names:
                for col in var_df.columns:
                    if col.lower() == candidate.lower():
                        hvg_col = col
                        break
                if hvg_col is not None:
                    break

            local_hvg: np.ndarray | None = None
            if hvg_col is not None:
                try:
                    hvg_values = var_df[hvg_col].to_numpy()
                except Exception:
                    hvg_values = np.array(var_df[hvg_col].to_list())
                # Treat truthy values as HVG (supports bool, int 0/1, float)
                local_hvg = np.asarray(hvg_values, dtype=bool)
                if local_hvg.shape[0] != len(var_df):
                    logger.warning(
                        "Dataset '%s': HVG column '%s' length mismatch (%d vs %d); "
                        "HVG mask disabled.",
                        ds_id, hvg_col, local_hvg.shape[0], len(var_df),
                    )
                    local_hvg = None
            if local_hvg is None and hvg_col is not None:
                logger.warning(
                    "Dataset '%s': HVG column '%s' found but could not be parsed; "
                    "HVG mask disabled.",
                    ds_id, hvg_col,
                )
            self._dataset_local_hvg[ds_idx] = local_hvg

            entries: list[dict[str, Any]] = []
            # Determine gene identifier column: prefer canonical_gene_id if present
            gene_id_col = (
                "canonical_gene_id"
                if "canonical_gene_id" in var_df.columns
                else "feature_id"
            )
            for row in var_df.iter_rows(named=True):
                origin_idx = int(row["origin_index"])
                feature_id = str(row[gene_id_col])
                if feature_id not in self._feature_id_to_global:
                    self._feature_id_to_global[feature_id] = len(
                        self._feature_id_to_global
                    )
                global_id = self._feature_id_to_global[feature_id]
                entries.append(
                    {
                        "origin_index": origin_idx,
                        "feature_id": feature_id,
                        "global_id": global_id,
                    }
                )
            self._dataset_var_entries[ds_idx] = entries

        self._global_vocab_size: int = len(self._feature_id_to_global)

        # ---- Pass 2: build per-dataset local→global mapping arrays ----
        self._max_local_vocab: int = max(
            len(entries) for entries in self._dataset_var_entries.values()
        )

        self._local_to_global: dict[int, np.ndarray] = {}
        for ds_idx, entries in self._dataset_var_entries.items():
            n_vars = len(entries)
            mapping = np.full(n_vars, _UNMAPPED, dtype=np.int32)
            for entry in entries:
                mapping[entry["origin_index"]] = entry["global_id"]
            # Verify all positions were assigned
            if np.any(mapping == _UNMAPPED):
                missing = int(np.where(mapping == _UNMAPPED)[0][0])
                raise ValueError(
                    f"dataset '{self._dataset_ids[ds_idx]}': "
                    f"origin_index range gap — position {missing} not assigned"
                )
            self._local_to_global[ds_idx] = mapping

        # ---- Precompute dense (n_datasets, max_local_vocab) mapping tensor ----
        self._dense_map: np.ndarray = self._build_dense_map()

        # ---- Build HVG mask from per-dataset local HVG flags ----
        self._hvg_mask: np.ndarray | None = None
        self._build_hvg_mask()

        # Lazy-computed fields
        self._dataset_has_gene: np.ndarray | None = None
        self._dataset_gene_prob: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_var_df(var_df: pl.DataFrame, ds_id: str) -> None:
        """Validate that the var DataFrame has required columns.

        Accepts either ``feature_id`` or ``canonical_gene_id`` as the
        gene identifier column.
        """
        has_feature_id = "feature_id" in var_df.columns
        has_canonical = "canonical_gene_id" in var_df.columns
        if not has_feature_id and not has_canonical:
            raise ValueError(
                f"var DataFrame for dataset '{ds_id}' is missing "
                f"a gene identifier column (need 'feature_id' or 'canonical_gene_id')"
            )
        if "origin_index" not in var_df.columns:
            raise ValueError(
                f"var DataFrame for dataset '{ds_id}' is missing "
                f"required column: 'origin_index'"
            )
        # Check that origin_index is contiguous 0..n_vars-1
        n = len(var_df)
        origin_idx = var_df["origin_index"].to_numpy()
        expected = np.arange(n, dtype=origin_idx.dtype)
        if not np.array_equal(origin_idx, expected):
            raise ValueError(
                f"var DataFrame for dataset '{ds_id}': "
                f"origin_index is not contiguous 0..{n - 1}"
            )

    # ------------------------------------------------------------------
    # Dense tensor construction
    # ------------------------------------------------------------------

    def _build_dense_map(self) -> np.ndarray:
        """Build dense ``(n_datasets, max_local_vocab)`` int32 mapping tensor.

        Shorter datasets are right-padded with ``_UNMAPPED`` (-1).
        """
        dense = np.full(
            (self._n_datasets, self._max_local_vocab),
            _UNMAPPED,
            dtype=np.int32,
        )
        for ds_idx in range(self._n_datasets):
            local_map = self._local_to_global[ds_idx]
            n_vars = len(local_map)
            dense[ds_idx, :n_vars] = local_map
        return dense

    # ------------------------------------------------------------------
    # Incremental append
    # ------------------------------------------------------------------

    def append_dataset(self, dataset_id: str, var_df: pl.DataFrame) -> None:
        """Register a new dataset without rebuilding the entire registry.

        The new dataset is assigned the next available ``dataset_index``.
        Existing global gene IDs are **stable** — new ``feature_id`` values
        are appended at the end of the global vocabulary.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the new dataset.
        var_df : pl.DataFrame
            ``var`` data with ``origin_index`` and ``feature_id`` columns.
        """
        if dataset_id in self._dataset_ids:
            raise ValueError(
                f"dataset_id '{dataset_id}' is already registered"
            )

        self._validate_var_df(var_df, dataset_id)
        ds_idx = self._n_datasets

        # Determine gene identifier column
        gene_id_col = (
            "canonical_gene_id"
            if "canonical_gene_id" in var_df.columns
            else "feature_id"
        )
        # Process new dataset entries, extending global vocab
        entries: list[dict[str, Any]] = []
        for row in var_df.iter_rows(named=True):
            origin_idx = int(row["origin_index"])
            feature_id = str(row[gene_id_col])
            if feature_id not in self._feature_id_to_global:
                self._feature_id_to_global[feature_id] = len(
                    self._feature_id_to_global
                )
            global_id = self._feature_id_to_global[feature_id]
            entries.append(
                {
                    "origin_index": origin_idx,
                    "feature_id": feature_id,
                    "global_id": global_id,
                }
            )

        # Build local→global for the new dataset
        n_vars = len(entries)
        mapping = np.full(n_vars, _UNMAPPED, dtype=np.int32)
        for entry in entries:
            mapping[entry["origin_index"]] = entry["global_id"]
        if np.any(mapping == _UNMAPPED):
            missing = int(np.where(mapping == _UNMAPPED)[0][0])
            raise ValueError(
                f"dataset '{dataset_id}': "
                f"origin_index range gap — position {missing} not assigned"
            )

        self._dataset_ids.append(dataset_id)
        self._n_datasets += 1
        self._dataset_var_entries[ds_idx] = entries
        self._local_to_global[ds_idx] = mapping
        self._global_vocab_size = len(self._feature_id_to_global)

        # Update max_local_vocab and rebuild dense map
        self._max_local_vocab = max(self._max_local_vocab, n_vars)
        self._dense_map = self._build_dense_map()

        # Invalidate lazy caches
        self._dataset_has_gene = None
        self._dataset_gene_prob = None
        self._hvg_mask = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """Ordered dataset identifiers."""
        return tuple(self._dataset_ids)

    @property
    def n_datasets(self) -> int:
        """Number of registered datasets."""
        return self._n_datasets

    @property
    def global_vocab_size(self) -> int:
        """Total number of unique genes across all datasets."""
        return self._global_vocab_size

    @property
    def max_local_vocab(self) -> int:
        """Maximum local vocab size (``n_vars``) across all datasets."""
        return self._max_local_vocab

    @property
    def local_to_global_map(self) -> np.ndarray:
        """Dense ``(n_datasets, max_local_vocab)`` int32 mapping tensor.

        Each row maps a dataset's local gene indices (``origin_index``) to
        global gene IDs.  Positions beyond ``n_vars_dataset`` are filled
        with ``-1`` (unmapped sentinel).
        """
        return self._dense_map

    @property
    def dataset_has_gene(self) -> np.ndarray:
        """Boolean mask ``(n_datasets, global_vocab_size)``.

        ``True`` where a dataset has a given global gene, ``False`` otherwise.
        Computed lazily on first access.
        """
        if self._dataset_has_gene is None:
            self._compute_masks()
        assert self._dataset_has_gene is not None
        return self._dataset_has_gene

    @property
    def dataset_gene_prob(self) -> np.ndarray:
        """Uniform sampling probabilities ``(n_datasets, global_vocab_size)``.

        Each row sums to 1.0 over the genes the dataset actually has.
        Computed lazily on first access.
        """
        if self._dataset_gene_prob is None:
            self._compute_masks()
        assert self._dataset_gene_prob is not None
        return self._dataset_gene_prob

    @property
    def hvg_mask(self) -> np.ndarray:
        """Per-dataset HVG mask ``(n_datasets, global_vocab_size)`` bool.

        ``True`` where a global gene is classified as highly variable for a
        given dataset.  All-zero when no var table contains an HVG column.
        """
        if self._hvg_mask is None:
            self._build_hvg_mask()
        assert self._hvg_mask is not None
        return self._hvg_mask

    def _compute_masks(self) -> None:
        """Build dataset_has_gene and dataset_gene_prob arrays."""
        has_gene = np.zeros(
            (self._n_datasets, self._global_vocab_size), dtype=bool
        )
        for ds_idx in range(self._n_datasets):
            global_ids = self._local_to_global[ds_idx]
            valid_ids = global_ids[global_ids >= 0]
            has_gene[ds_idx, valid_ids] = True

        # Uniform probabilities over valid genes per dataset
        prob = np.zeros(
            (self._n_datasets, self._global_vocab_size), dtype=np.float32
        )
        for ds_idx in range(self._n_datasets):
            n_valid = int(has_gene[ds_idx].sum())
            if n_valid > 0:
                prob[ds_idx, has_gene[ds_idx]] = 1.0 / n_valid

        self._dataset_has_gene = has_gene
        self._dataset_gene_prob = prob

    def _build_hvg_mask(self) -> None:
        """Build per-dataset HVG mask ``(n_datasets, global_vocab_size)`` bool.

        Uses per-dataset local HVG flags (captured in ``__init__``) mapped
        through the local→global mapping.  If no dataset has an HVG column,
        the mask is all-zeros.
        """
        any_hvg = any(
            local_hvg is not None for local_hvg in self._dataset_local_hvg.values()
        )
        if not any_hvg:
            self._hvg_mask = np.zeros(
                (self._n_datasets, self._global_vocab_size), dtype=bool
            )
            return

        hvg_mask = np.zeros(
            (self._n_datasets, self._global_vocab_size), dtype=bool
        )
        for ds_idx in range(self._n_datasets):
            local_hvg = self._dataset_local_hvg.get(ds_idx)
            if local_hvg is None:
                continue
            local_to_global = self._local_to_global[ds_idx]
            n_vars = len(local_to_global)
            if len(local_hvg) > n_vars:
                local_hvg = local_hvg[:n_vars]
            elif len(local_hvg) < n_vars:
                local_hvg_padded = np.zeros(n_vars, dtype=bool)
                local_hvg_padded[:len(local_hvg)] = local_hvg
                local_hvg = local_hvg_padded

            hvg_local_ids = local_to_global[local_hvg]
            valid = hvg_local_ids >= 0
            hvg_mask[ds_idx, hvg_local_ids[valid]] = True

        self._hvg_mask = hvg_mask

    # ------------------------------------------------------------------
    # Reverse lookup helpers
    # ------------------------------------------------------------------

    def global_to_feature_id(self, global_id: int) -> str:
        """Return the ``feature_id`` string for a global gene ID."""
        for fid, gid in self._feature_id_to_global.items():
            if gid == global_id:
                return fid
        raise KeyError(f"global_id {global_id} not in registry")

    def feature_id_to_global(self, feature_id: str) -> int:
        """Return the global gene ID for a ``feature_id`` string."""
        return self._feature_id_to_global[feature_id]

    # ------------------------------------------------------------------
    # dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FeatureRegistry(n_datasets={self._n_datasets}, "
            f"global_vocab={self._global_vocab_size}, "
            f"max_local_vocab={self._max_local_vocab})"
        )

    def __len__(self) -> int:
        return self._n_datasets


# ---------------------------------------------------------------------------
# GlobalGeneSampler
# ---------------------------------------------------------------------------


class GlobalGeneSampler:
    """Sample global gene IDs from per-dataset valid gene pools.

    Given a batch of cells (each belonging to a specific dataset), samples
    ``seq_len`` unique global gene IDs per cell from that cell's dataset's
    valid gene pool.  Uses uniform probabilities — each valid gene is
    equally likely to be sampled.

    Parameters
    ----------
    registry : FeatureRegistry
        The feature registry providing per-dataset gene masks and probabilities.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.
    """

    def __init__(self, registry: FeatureRegistry, rng: np.random.Generator):
        self._registry = registry
        self._rng = rng

    @property
    def registry(self) -> FeatureRegistry:
        return self._registry

    def sample(
        self,
        seq_len: int,
        dataset_indices: np.ndarray,
    ) -> np.ndarray:
        """Sample ``seq_len`` global gene IDs per cell.

        Parameters
        ----------
        seq_len : int
            Number of genes to sample per cell.
        dataset_indices : np.ndarray
            1-D int array of shape ``(batch_size,)``, where each element is
            the ``dataset_index`` for the corresponding cell in the batch.

        Returns
        -------
        np.ndarray
            2-D int32 array of shape ``(batch_size, seq_len)`` containing
            sampled global gene IDs.  Unfilled positions (when a cell has
            fewer valid genes than ``seq_len``) are set to ``-1`` (pad).
        """
        batch_size = len(dataset_indices)
        # Use -1 as pad sentinel so that gene 0 is not ambiguous
        sampled = np.full((batch_size, seq_len), -1, dtype=np.int32)

        if batch_size == 0:
            return sampled

        prob = self._registry.dataset_gene_prob
        global_vocab = self._registry.global_vocab_size

        for i, ds_idx in enumerate(dataset_indices):
            ds_idx = int(ds_idx)
            if ds_idx < 0 or ds_idx >= self._registry.n_datasets:
                raise IndexError(
                    f"dataset_index {ds_idx} out of range "
                    f"[0, {self._registry.n_datasets})"
                )
            # Sample from the per-dataset probability distribution
            p_vec = prob[ds_idx]
            n_valid = int(np.sum(p_vec > 0.0))
            actual_len = min(seq_len, n_valid)
            # Use multinomial-style sampling via weighted choice
            candidates = np.arange(global_vocab, dtype=np.int32)[p_vec > 0.0]
            chosen = self._rng.choice(candidates, size=actual_len, replace=False)
            sampled[i, :actual_len] = chosen
            # Remaining positions stay -1 (pad sentinel)

        return sampled

    def sample_with_mask(
        self,
        seq_len: int,
        dataset_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample global gene IDs with a valid-position mask.

        Returns
        -------
        sampled : np.ndarray
            ``(batch_size, seq_len)`` int32 — sampled global gene IDs.
        valid_mask : np.ndarray
            ``(batch_size, seq_len)`` bool — ``True`` where a valid gene
            was sampled, ``False`` for pad positions (including cells
            with fewer valid genes than ``seq_len``).
        """
        sampled = self.sample(seq_len, dataset_indices)
        # Pad positions are -1; valid genes are >= 0
        valid_mask = sampled >= 0
        # Cross-check with has_gene for precision
        has_gene = self._registry.dataset_has_gene
        precise_mask = np.zeros_like(sampled, dtype=bool)
        for i, ds_idx in enumerate(dataset_indices):
            ds_idx = int(ds_idx)
            for j in range(seq_len):
                gid = sampled[i, j]
                if gid >= 0 and gid < self._registry.global_vocab_size:
                    if has_gene[ds_idx, gid]:
                        precise_mask[i, j] = True
        return sampled, precise_mask
