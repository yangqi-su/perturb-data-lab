"""Small Polars-backed metadata index used by corpus loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

__all__ = ["MetadataIndex"]


_CANONICAL_OBS_TYPED_DTYPES: dict[str, pl.DataType] = {
    "global_row_index": pl.Int64,
    "dataset_index": pl.Int32,
    "local_row_index": pl.Int64,
    "size_factor": pl.Float64,
}

_CANONICAL_OBS_STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "global_row_index",
    "cell_id",
    "dataset_id",
    "dataset_index",
    "local_row_index",
    "size_factor",
)

_CANONICAL_OBS_CONTENT_COLUMNS: tuple[str, ...] = (
    "perturb_label",
    "perturb_type",
    "dose",
    "dose_unit",
    "timepoint",
    "timepoint_unit",
    "cell_context",
    "cell_line_or_type",
    "species",
    "tissue",
    "assay",
    "condition",
    "batch_id",
    "donor_id",
    "sex",
    "disease_state",
)

_CANONICAL_OBS_CORE_COLUMNS: tuple[str, ...] = (
    _CANONICAL_OBS_STRUCTURAL_COLUMNS + _CANONICAL_OBS_CONTENT_COLUMNS
)

_CANONICAL_NULL_NORMALIZED_STRING_COLUMNS: frozenset[str] = frozenset(
    column_name
    for column_name in _CANONICAL_OBS_CORE_COLUMNS
    if column_name not in _CANONICAL_OBS_TYPED_DTYPES
)

_CANONICAL_NULL_LITERALS: frozenset[str] = frozenset(
    {"", "na", "none", "null", "nan", ".", "-"}
)

_STRINGLIKE_DTYPES: set[pl.DataType] = {pl.Utf8, pl.Categorical, pl.Enum}


class MetadataIndex:
    """Thin wrapper around a flat, corpus-global metadata DataFrame."""

    _NUMERIC_DTYPES: set[pl.DataType] = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Boolean,
    }

    def __init__(self, df: pl.DataFrame):
        self._validate_flat_schema(df)
        self.df = df

    @staticmethod
    def _validate_flat_schema(df: pl.DataFrame) -> None:
        for col_name in df.columns:
            dtype = df[col_name].dtype
            if dtype in (pl.Struct, pl.Object, pl.List):
                raise ValueError(
                    f"Column {col_name!r} has non-flat dtype {dtype}; metadata columns must be primitive"
                )

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"MetadataIndex({len(self)} rows, {len(self.df.columns)} columns)"

    def get_column(self, col_name: str) -> np.ndarray | tuple | None:
        """Return a full column, or None when absent."""
        if col_name not in self.df.columns:
            return None
        series = self.df[col_name]
        if series.dtype in self._NUMERIC_DTYPES:
            return series.to_numpy()
        return tuple(series.to_list())

    def gather_columns(
        self,
        indices: list[int] | np.ndarray,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray | tuple]:
        """Gather selected columns for corpus-global row positions."""
        indices_arr = np.asarray(indices, dtype=np.int64)
        resolved_columns = list(self.df.columns) if columns is None else list(columns)
        result: dict[str, np.ndarray | tuple] = {}
        for col_name in resolved_columns:
            series = self.df[col_name]
            if series.dtype in self._NUMERIC_DTYPES:
                result[col_name] = series.to_numpy()[indices_arr]
            else:
                result[col_name] = tuple(self.df[indices_arr, col_name].to_list())
        return result

    def take(
        self,
        indices: list[int] | np.ndarray,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray | tuple]:
        return self.gather_columns(indices, columns)


def _normalize_canonical_obs_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize typed and read-time-null-aware canonical obs columns."""
    expressions: list[pl.Expr] = []

    for col_name, dtype in _CANONICAL_OBS_TYPED_DTYPES.items():
        if col_name not in df.columns:
            continue
        expr = pl.col(col_name)
        if df[col_name].dtype in _STRINGLIKE_DTYPES:
            expr = _null_if_legacy_missing(expr)
        expressions.append(expr.cast(dtype, strict=True).alias(col_name))

    for col_name in _CANONICAL_NULL_NORMALIZED_STRING_COLUMNS:
        if col_name not in df.columns or df[col_name].dtype not in _STRINGLIKE_DTYPES:
            continue
        expressions.append(
            _null_if_legacy_missing(pl.col(col_name)).cast(pl.Utf8).alias(col_name)
        )

    if not expressions:
        return df
    return df.with_columns(expressions)


def _load_canonical_obs_frame(
    obs_path: str | Path,
    *,
    extra_metadata_columns: Sequence[str] | None = None,
    context: str | None = None,
) -> pl.DataFrame:
    """Read a canonical obs parquet with canonical-core projection by default."""
    path_str = str(obs_path)
    projection = _resolve_canonical_obs_projection(
        pl.read_parquet_schema(path_str).keys(),
        extra_metadata_columns=extra_metadata_columns,
        context=context or f"canonical obs parquet at {path_str}",
    )
    return _normalize_canonical_obs_dtypes(
        pl.read_parquet(path_str, columns=projection)
    )


def _resolve_canonical_obs_projection(
    available_columns: Sequence[str],
    *,
    extra_metadata_columns: Sequence[str] | None = None,
    context: str,
) -> list[str]:
    """Resolve canonical-core parquet projection plus optional extras."""
    extras = _normalize_extra_metadata_columns(extra_metadata_columns)
    available = set(available_columns)
    missing = [name for name in extras if name not in available]
    if missing:
        missing_str = ", ".join(repr(name) for name in missing)
        raise ValueError(
            f"{context} is missing requested extra metadata columns: {missing_str}"
        )

    projection = [
        column_name
        for column_name in _CANONICAL_OBS_CORE_COLUMNS
        if column_name in available
    ]
    projection.extend(
        column_name for column_name in extras if column_name not in projection
    )
    return projection


def _normalize_extra_metadata_columns(
    extra_metadata_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate and deduplicate requested extra metadata columns."""
    if extra_metadata_columns is None:
        return ()
    if isinstance(extra_metadata_columns, (str, bytes)):
        raise TypeError("extra_metadata_columns must be a sequence of column names")

    normalized: list[str] = []
    seen: set[str] = set()
    for name in extra_metadata_columns:
        if not isinstance(name, str) or not name:
            raise ValueError("extra_metadata_columns must contain non-empty strings")
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return tuple(normalized)


def _null_if_legacy_missing(expr: pl.Expr) -> pl.Expr:
    """Convert configured canonical null-like strings to true nulls."""
    normalized = expr.cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    return (
        pl.when(expr.is_null())
        .then(None)
        .when(normalized.is_in(_CANONICAL_NULL_LITERALS))
        .then(None)
        .otherwise(expr)
    )
