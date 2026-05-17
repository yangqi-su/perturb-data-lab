"""Parquet sidecar writers used by dataset materialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _safe_serialize(val: Any) -> Any:
    """Serialize common numpy/pandas values into JSON-safe scalars."""
    if val is None or (isinstance(val, float) and (val != val)):  # NaN check
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, pd.CategoricalDtype):
        return str(val)
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (str, int, float, bool)):
        return val
    return str(val)


def write_raw_cell_metadata_parquet(
    *,
    obs: pd.DataFrame,
    source_row_indices: np.ndarray | None,
    meta_root: Path,
    dataset_id: str,
) -> Path:
    """Write raw obs metadata as ``raw-obs.parquet``."""
    parquet_path = meta_root / "raw-obs.parquet"
    n = len(obs)

    cell_ids = [str(idx) for idx in obs.index]
    dataset_ids = [dataset_id] * n
    col_names = [
        col for col in obs.columns if col not in {"source_row_index", "source_obs_index"}
    ]
    col_lists = {col: obs[col].apply(_safe_serialize).to_list() for col in col_names}
    raw_fields = [
        json.dumps({col: col_lists[col][i] for col in col_names})
        for i in range(n)
    ]

    columns: dict[str, pa.Array] = {
        "cell_id": pa.array(cell_ids, type=pa.string()),
        "dataset_id": pa.array(dataset_ids, type=pa.string()),
        "raw_fields": pa.array(raw_fields, type=pa.string()),
    }
    if source_row_indices is not None:
        columns["source_row_index"] = pa.array(
            source_row_indices.tolist(), type=pa.int64()
        )
        columns["source_obs_index"] = pa.array(cell_ids, type=pa.string())

    pq.write_table(pa.table(columns), parquet_path)
    return parquet_path


def write_raw_feature_metadata_parquet(
    *,
    var_mem: pd.DataFrame,
    meta_root: Path,
) -> Path:
    """Write raw var metadata as ``raw-var.parquet``."""
    parquet_path = meta_root / "raw-var.parquet"
    n = len(var_mem)
    col_names = list(var_mem.columns)
    col_lists = {col: var_mem[col].apply(_safe_serialize).to_list() for col in col_names}
    raw_var = [
        json.dumps({col: col_lists[col][i] for col in col_names})
        for i in range(n)
    ]

    table = pa.table({
        "origin_index": pa.array(list(range(n)), type=pa.int32()),
        "feature_id": pa.array([str(idx) for idx in var_mem.index], type=pa.string()),
        "raw_var": pa.array(raw_var, type=pa.string()),
    })
    pq.write_table(table, parquet_path)
    return parquet_path


def write_feature_provenance_parquet(
    *,
    var_index: pd.Index,
    n_vars: int,
    meta_root: Path,
    count_source_selected: str,
    source_path: str,
) -> Path:
    """Write the dataset-local feature ordering and source count layer."""
    parquet_path = meta_root / "feature-provenance.parquet"
    table = pa.table({
        "origin_index": pa.array(list(range(n_vars)), type=pa.int32()),
        "feature_id": pa.array([str(var_index[i]) for i in range(n_vars)], type=pa.string()),
        "count_source": pa.array([count_source_selected] * n_vars, type=pa.string()),
        "source_path": pa.array([source_path] * n_vars, type=pa.string()),
    })
    pq.write_table(table, parquet_path)
    return parquet_path


def write_size_factor_parquet(
    *,
    size_factors: np.ndarray,
    cell_ids: pd.Index,
    meta_root: Path,
) -> Path:
    """Write median-normalized per-cell size factors."""
    parquet_path = meta_root / "size-factor.parquet"
    table = pa.table({
        "cell_id": pa.array([str(c) for c in cell_ids], type=pa.string()),
        "size_factor": pa.array(size_factors.tolist(), type=pa.float64()),
    })
    pq.write_table(table, parquet_path)
    return parquet_path


def write_hvg_ranking_parquet(
    *,
    sum_log1p: np.ndarray,
    sum_log1p_sq: np.ndarray,
    n_cells_total: int,
    feature_ids: Sequence[str],
    n_hvg: int,
    meta_root: Path,
) -> Path:
    """Write the per-dataset HVG ranking artifact."""
    from .chunk_translation import _build_hvg_ranking_table

    table = _build_hvg_ranking_table(
        sum_log1p=sum_log1p,
        sum_log1p_sq=sum_log1p_sq,
        n_cells_total=n_cells_total,
        feature_ids=feature_ids,
        n_hvg=n_hvg,
    )
    output_path = meta_root / "hvg.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return output_path
