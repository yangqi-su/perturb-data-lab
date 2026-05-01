"""Phase 1: Queryable in-memory metadata index with flat columnar schema.

Replaces MetadataTable with a polars DataFrame-backed index that supports
vectorized filtering, stratified sampling, and O(1) global index lookup.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

__all__ = ["MetadataIndex", "MetadataRow"]


# ---------------------------------------------------------------------------
# Constants — known paths for the synthetic dummy corpus from Stage 2
# ---------------------------------------------------------------------------

_LANCE_FEDERATED_BASE = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/archive/plans-20260427-backend-topology-validation/outputs"
    / "lance-federated"
)

_DUMMY_DATASETS = {
    "dummy_00": {
        "n_obs": 50_000,
        "release_id": "dummy_00-release",
    },
    "dummy_01": {
        "n_obs": 75_000,
        "release_id": "dummy_01-release",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetadataRow:
    """A single row of metadata with flat attributes.

    ``raw_fields`` is the only sub-dict; all other dict fields are already
    flattened into individual top-level attributes (e.g. ``raw_guide_1``).
    """

    global_row_index: int
    cell_id: str
    dataset_id: str
    dataset_release: str
    dataset_index: int
    local_row_index: int
    size_factor: float
    raw_fields: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MetadataIndex
# ---------------------------------------------------------------------------


class MetadataIndex:
    """Queryable in-memory metadata index backed by a polars DataFrame.

    All columns are primitive types (int, float, str, bool). No Struct, Object,
    or List columns. Dict fields from raw observations are flattened into
    individual ``raw_<key>`` columns.

    Parameters
    ----------
    df : pl.DataFrame
        The underlying DataFrame with a flat schema. Must contain at least
        ``global_row_index``, ``cell_id``, ``dataset_id``, ``dataset_release``,
        ``dataset_index``, ``local_row_index``, and ``size_factor``.
    """

    def __init__(self, df: pl.DataFrame):
        self._validate_flat_schema(df)
        self.df = df
        self._global_index = df["global_row_index"].to_numpy()

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_flat_schema(df: pl.DataFrame) -> None:
        """Raise if any column has a non-primitive dtype."""
        for col_name in df.columns:
            dtype = df[col_name].dtype
            if dtype in (pl.Struct, pl.Object, pl.List):
                raise ValueError(
                    f"Column '{col_name}' has non-flat dtype {dtype}. "
                    "All columns must be primitive (int, float, str, bool)."
                )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet_files(
        cls,
        corpus_index_path: str | Path,
        *,
        use_canonical: bool = False,
    ) -> "MetadataIndex":
        """Load from a corpus-index YAML and per-dataset parquet files.

        When ``use_canonical=False`` (default), reads raw obs sidecars
        (``{release_id}-raw-obs.parquet``) with ``raw_fields`` JSON parsing
        and ``raw_``-prefixed columns.

        When ``use_canonical=True``, reads canonical obs sidecars
        (``{release_id}-canonical-obs.parquet``) with already-flat
        canonical columns.  No JSON parsing needed.

        Expected YAML structure (list of dataset entries)::

            datasets:
              - dataset_id: dummy_00
                release_id: dummy_00-release
                obs_path: /path/to/raw-obs.parquet
                size_factor_path: /path/to/size-factor.parquet
                n_obs: 50000
              - ...   (canonical_obs_path: /path/to/canonical-obs.parquet)

        Parameters
        ----------
        corpus_index_path:
            Path to a YAML file describing the corpus.
        use_canonical:
            When ``True``, reads canonical obs parquets.
        """
        import yaml

        with open(corpus_index_path) as f:
            index_doc = yaml.safe_load(f)

        datasets = index_doc.get("datasets", [])
        if not datasets:
            raise ValueError(
                f"No 'datasets' list found in corpus index: {corpus_index_path}"
            )

        entries = []
        for ds in datasets:
            obs_path = (
                ds.get("canonical_obs_path", ds["obs_path"])
                if use_canonical
                else ds["obs_path"]
            )
            entries.append(
                {
                    "dataset_id": ds["dataset_id"],
                    "release_id": ds.get(
                        "release_id", f"{ds['dataset_id']}-release"
                    ),
                    "obs_path": obs_path,
                    "size_factor_path": ds.get("size_factor_path"),
                    "n_obs": ds.get("n_obs"),
                }
            )

        if use_canonical:
            return cls._from_canonical_dataset_entries(entries)
        return cls._from_dataset_entries(entries)

    @classmethod
    def from_dummy_data(cls, use_canonical: bool = False) -> "MetadataIndex":
        """Convenience factory for the synthetic dummy_00 + dummy_01 corpus.

        Reads from the known Stage 2 lance-federated output paths.
        When ``use_canonical=True``, reads canonical obs parquets from
        Phase 2 outputs.
        """
        if use_canonical:
            _PLAN_RUN = (
                Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
                / "copilot/plans/active/plans-20260430-canonicalization-module-and-loader-adaptation"
                / "outputs"
            )
            entries = []
            for ds_id, info in sorted(_DUMMY_DATASETS.items()):
                entries.append(
                    {
                        "dataset_id": ds_id,
                        "release_id": info["release_id"],
                        "obs_path": str(
                            _PLAN_RUN / ds_id / f"{info['release_id']}-canonical-obs.parquet"
                        ),
                        "size_factor_path": None,  # size_factor is inline in canonical
                        "n_obs": info["n_obs"],
                    }
                )
            return cls._from_canonical_dataset_entries(entries)

        entries = []
        for ds_id, info in sorted(_DUMMY_DATASETS.items()):
            ds_dir = _LANCE_FEDERATED_BASE / ds_id / "metadata"
            entries.append(
                {
                    "dataset_id": ds_id,
                    "release_id": info["release_id"],
                    "obs_path": str(
                        ds_dir / f"{info['release_id']}-raw-obs.parquet"
                    ),
                    "size_factor_path": str(
                        ds_dir / f"{info['release_id']}-size-factor.parquet"
                    ),
                    "n_obs": info["n_obs"],
                }
            )
        return cls._from_dataset_entries(entries)

    # ------------------------------------------------------------------
    # Internal factory logic
    # ------------------------------------------------------------------

    @classmethod
    def _from_dataset_entries(
        cls, entries: list[dict[str, Any]]
    ) -> "MetadataIndex":
        """Build a MetadataIndex from a list of dataset descriptor dicts.

        Each entry must contain:
            dataset_id, release_id, obs_path, size_factor_path[, n_obs]
        """
        # --- First pass: read all datasets and collect all unique raw_fields keys ---
        dataset_dfs: list[pl.DataFrame] = []
        all_raw_keys: set[str] = set()

        for entry in entries:
            obs_path = entry["obs_path"]
            df = pl.read_parquet(obs_path)
            # Parse JSON raw_fields to discover keys
            parsed_keys = cls._discover_raw_keys(df)
            all_raw_keys.update(parsed_keys)
            dataset_dfs.append(df)

        # --- Second pass: parse and flatten each dataset ---
        processed_dfs: list[pl.DataFrame] = []
        global_row_offset = 0

        for ds_idx, (entry, raw_df) in enumerate(zip(entries, dataset_dfs)):
            dataset_id = entry["dataset_id"]
            release_id = entry["release_id"]
            n_obs = len(raw_df)

            # Parse JSON raw_fields into struct, unnest into columns
            df_flat = cls._flatten_json_fields(raw_df)

            # Ensure all known keys exist (fill missing with null).
            # Skip keys that duplicate base columns (cell_id, dataset_id, dataset_release).
            _base_cols = {"cell_id", "dataset_id", "dataset_release"}
            existing_cols = set(df_flat.columns)
            for key in all_raw_keys:
                col_name = f"raw_{key}"
                if key in _base_cols:
                    continue  # these are already top-level columns
                if col_name not in existing_cols:
                    df_flat = df_flat.with_columns(
                        pl.lit(None).alias(col_name)
                    )

            # Read size-factor
            sf_path = entry["size_factor_path"]
            sf_df = pl.read_parquet(sf_path)  # columns: cell_id, size_factor

            # Join size-factor on cell_id
            df_flat = df_flat.join(sf_df, on="cell_id", how="left")

            # Compute computed columns
            df_flat = df_flat.with_columns(
                pl.lit(ds_idx, dtype=pl.Int64).alias("dataset_index"),
                pl.lit(release_id, dtype=pl.Utf8).alias("dataset_release"),
            )

            # Add a temporary row index (local within dataset)
            df_flat = df_flat.with_row_index("local_row_index", offset=0)

            # Add global_row_index
            df_flat = df_flat.with_columns(
                (pl.col("local_row_index") + global_row_offset).alias(
                    "global_row_index"
                )
            )

            global_row_offset += n_obs
            processed_dfs.append(df_flat)

        # --- Concatenate all datasets ---
        combined = pl.concat(processed_dfs, how="diagonal_relaxed")

        # --- Sort by global_row_index to ensure contiguous ordering ---
        combined = combined.sort("global_row_index")

        # --- Validate contiguous indices ---
        cls._validate_contiguous(combined)

        # --- Reorder columns for a clean schema ---
        fixed_cols = [
            "global_row_index",
            "cell_id",
            "dataset_id",
            "dataset_release",
            "dataset_index",
            "local_row_index",
            "size_factor",
        ]
        raw_cols = sorted(
            [c for c in combined.columns if c.startswith("raw_")]
        )
        col_order = fixed_cols + raw_cols
        combined = combined.select(col_order)

        return cls(combined)

    # ------------------------------------------------------------------
    # Canonical parquet loading (use_canonical=True)
    # ------------------------------------------------------------------

    @classmethod
    def _from_canonical_dataset_entries(
        cls, entries: list[dict[str, Any]]
    ) -> "MetadataIndex":
        """Build a MetadataIndex from canonical obs parquets (already flat).

        Canonical obs parquets have all columns as flat strings — no
        ``raw_fields`` JSON parsing or ``raw_`` prefixing needed.

        Size factors are already inline in the canonical parquet (no
        separate join required).

        Column reordering and global_row_index validation still apply.
        """
        processed_dfs: list[pl.DataFrame] = []
        global_row_offset = 0

        for ds_idx, entry in enumerate(entries):
            obs_path = entry["obs_path"]
            dataset_id = entry["dataset_id"]
            release_id = entry["release_id"]

            df = pl.read_parquet(obs_path)
            n_obs = len(df)

            # The canonical obs columns may arrive as strings; cast numeric
            # identity columns to proper types for downstream consumers.
            _cast_map: dict[str, pl.DataType] = {
                "global_row_index": pl.Int64,
                "dataset_index": pl.Int32,
                "local_row_index": pl.Int64,
                "size_factor": pl.Float64,
            }
            for col_name, dtype in _cast_map.items():
                if col_name in df.columns:
                    try:
                        df = df.with_columns(pl.col(col_name).cast(dtype))
                    except Exception:
                        # If casting fails (e.g. "NA" strings), keep as string
                        pass

            # Ensure dataset_index and dataset_release are correct
            if "dataset_index" in df.columns:
                df = df.with_columns(
                    pl.lit(ds_idx, dtype=pl.Int32).alias("dataset_index")
                )
            else:
                df = df.with_columns(
                    pl.lit(ds_idx, dtype=pl.Int32).alias("dataset_index")
                )
            if "dataset_release" not in df.columns:
                df = df.with_columns(
                    pl.lit(release_id, dtype=pl.Utf8).alias("dataset_release")
                )

            # Override global_row_index with corpus-global range
            local_range = pl.int_range(0, n_obs, dtype=pl.Int64)
            df = df.with_columns(
                (local_range + global_row_offset).alias("global_row_index")
            )

            # Ensure local_row_index is correct
            if "local_row_index" in df.columns:
                df = df.with_columns(
                    pl.int_range(0, n_obs, dtype=pl.Int64).alias("local_row_index")
                )

            global_row_offset += n_obs
            processed_dfs.append(df)

        # Concatenate all datasets (diagonal_relaxed fills missing columns)
        combined = pl.concat(processed_dfs, how="diagonal_relaxed")

        # Ensure all must-have canonical obs columns exist
        _canonical_cols = [
            "global_row_index", "cell_id", "dataset_id", "dataset_release",
            "dataset_index", "local_row_index", "size_factor",
            "perturb_label", "perturb_type", "dose", "dose_unit",
            "timepoint", "timepoint_unit", "cell_context", "cell_line_or_type",
            "species", "tissue", "assay", "condition", "batch_id",
            "donor_id", "sex", "disease_state",
        ]
        for col in _canonical_cols:
            if col not in combined.columns:
                combined = combined.with_columns(
                    pl.lit(None, dtype=pl.Utf8).alias(col)
                )

        # Sort by global_row_index
        combined = combined.sort("global_row_index")

        # Reorder: structural columns first, then canonical content, then extensible
        structural = [
            "global_row_index", "cell_id", "dataset_id", "dataset_release",
            "dataset_index", "local_row_index", "size_factor",
        ]
        content = [
            c for c in _canonical_cols
            if c not in structural and c in combined.columns
        ]
        # Keep raw_ columns that may have been carried forward as extensible
        extensible = sorted(
            c for c in combined.columns
            if c not in structural and c not in content and not c.startswith("raw_")
        )
        raw_cols = sorted(
            c for c in combined.columns
            if c.startswith("raw_")
        )
        col_order = structural + content + extensible + raw_cols
        combined = combined.select(col_order)

        # Validate contiguous
        cls._validate_contiguous(combined)

        return cls(combined)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_raw_keys(df: pl.DataFrame) -> set[str]:
        """Return the set of JSON keys present in the raw_fields column.

        Scans the first row to discover keys (all cells in the corpus have
        the same keys, so scanning the full column is unnecessary).
        """
        sample = df["raw_fields"][0]
        if isinstance(sample, str):
            parsed = json.loads(sample)
        elif isinstance(sample, dict):
            parsed = sample
        else:
            return set()
        return set(parsed.keys())

    @staticmethod
    def _flatten_json_fields(df: pl.DataFrame) -> pl.DataFrame:
        """Parse ``raw_fields`` JSON string and unnest into ``raw_<key>`` cols.

        The original ``raw_fields`` column is dropped after extraction.

        Uses Python's ``json.loads`` + ``pl.from_dicts`` for explicit type
        inference instead of ``str.json_decode()`` (which requires an explicit
        ``dtype`` in polars >= 1.33).
        """
        import json

        # Parse all JSON strings into Python dicts
        parsed_dicts = [json.loads(s) for s in df["raw_fields"]]
        parsed = pl.from_dicts(parsed_dicts)

        # Columns that already exist as top-level columns in the raw-obs parquet.
        # These should NOT get a ``raw_`` prefix to avoid duplication.
        base_cols = {"cell_id", "dataset_id", "dataset_release"}

        # Only columns not in base_cols get the raw_ prefix
        json_cols = [c for c in parsed.columns if c not in base_cols]
        rename_map = {c: f"raw_{c}" for c in json_cols}
        parsed = parsed.rename(rename_map)

        # Select only the raw_ columns (the base ones are already in ``df``)
        raw_cols = [c for c in parsed.columns if c.startswith("raw_")]

        # Drop the source JSON column and horizontally concat
        df_no_raw = df.drop("raw_fields")
        return pl.concat([df_no_raw, parsed.select(raw_cols)], how="horizontal")

    @staticmethod
    def _validate_contiguous(df: pl.DataFrame) -> None:
        """Assert that global_row_index is 0..N-1 with no gaps."""
        gr = df["global_row_index"]
        n = len(df)
        expected = pl.Series("expected", range(n), dtype=pl.Int64)
        if not (gr.to_numpy() == expected.to_numpy()).all():
            mismatch = df.filter(
                pl.col("global_row_index") != pl.int_range(0, n)
            )
            first_bad = mismatch["global_row_index"][0] if len(mismatch) > 0 else None
            raise ValueError(
                f"global_row_index is not contiguous 0..{n - 1}. "
                f"First mismatch at index={first_bad}."
            )

    # ------------------------------------------------------------------
    # Container interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"MetadataIndex({len(self)} rows, "
            f"{len(self.df.columns)} columns)"
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def filter(self, *predicates: pl.Expr) -> "MetadataIndex":
        """Return a new ``MetadataIndex`` filtered by the given predicates.

        Parameters
        ----------
        *predicates:
            One or more polars boolean expressions (e.g.
            ``pl.col("dataset_id") == "dummy_00"``).

        Returns
        -------
        MetadataIndex
            A new index containing only the matching rows.
        """
        filtered = self.df.filter(*predicates)
        return MetadataIndex(filtered)

    def sample(self, n: int, seed: int | None = None) -> pl.DataFrame:
        """Random sample of ``n`` rows.

        Parameters
        ----------
        n : int
            Number of rows to sample.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pl.DataFrame
            Sampled rows as a polars DataFrame (not a MetadataIndex).
        """
        return self.df.sample(n=n, seed=seed)

    def sample_by(
        self,
        group_col: str,
        n_per_group: int,
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Stratified sampling: ``n_per_group`` rows from each group.

        Parameters
        ----------
        group_col : str
            Column name to group by.
        n_per_group : int
            Number of rows to sample per group.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pl.DataFrame
            Sampled rows with approximately ``n_per_group`` rows per group.
        """
        return self.df.group_by(group_col).map_groups(
            lambda g: g.sample(n=min(n_per_group, len(g)), seed=seed),
        )

    # ------------------------------------------------------------------
    # Index lookup
    # ------------------------------------------------------------------

    def __getitem__(self, global_indices: list[int]) -> "MetadataIndex":
        """Return a new ``MetadataIndex`` for the given positional indices.

        Since ``global_row_index`` is guaranteed to be contiguous 0..N-1,
        this is equivalent to ``df[global_indices]`` (positional indexing).
        """
        subset = self.df[global_indices]
        return MetadataIndex(subset)

    def get_indices(self, df: pl.DataFrame | None = None) -> list[int]:
        """Extract ``global_row_index`` values from a filtered/sampled DataFrame.

        Parameters
        ----------
        df : pl.DataFrame, optional
            The DataFrame to extract indices from. If None, use the full index.

        Returns
        -------
        list[int]
            List of global row indices.
        """
        src = df if df is not None else self.df
        return src["global_row_index"].to_list()

    def rows(self, global_indices: list[int]) -> list[MetadataRow]:
        """Extract ``MetadataRow`` objects for the given global indices.

        The output list preserves the order of the input indices.

        Parameters
        ----------
        global_indices : list[int]
            Global row indices to look up.

        Returns
        -------
        list[MetadataRow]
            One ``MetadataRow`` per input index, in the same order.
        """
        subset = self.df.filter(
            pl.col("global_row_index").is_in(global_indices)
        )
        # Preserve input order
        index_order = {idx: pos for pos, idx in enumerate(global_indices)}
        subset = subset.with_columns(
            pl.col("global_row_index")
            .map_elements(
                lambda x: index_order.get(x, -1),
                return_dtype=pl.Int64,
            )
            .alias("_order")
        ).sort("_order").drop("_order")

        return [self._row_to_metadata_row(r) for r in subset.to_dicts()]

    # ------------------------------------------------------------------
    # Internal row conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_metadata_row(row: dict[str, Any]) -> MetadataRow:
        """Convert a polars dict row to a ``MetadataRow``.

        ``raw_`` prefixed columns are collected into the ``raw_fields`` dict.
        """
        known_scalars = {
            "global_row_index", "cell_id", "dataset_id",
            "dataset_release", "dataset_index", "local_row_index",
            "size_factor",
        }

        raw_fields = {
            k.removeprefix("raw_"): v
            for k, v in row.items()
            if k.startswith("raw_") and v is not None
        }

        return MetadataRow(
            global_row_index=int(row["global_row_index"]),
            cell_id=str(row["cell_id"]),
            dataset_id=str(row["dataset_id"]),
            dataset_release=str(row.get("dataset_release", "")),
            dataset_index=int(row["dataset_index"]),
            local_row_index=int(row["local_row_index"]),
            size_factor=float(row["size_factor"]),
            raw_fields=raw_fields,
        )
