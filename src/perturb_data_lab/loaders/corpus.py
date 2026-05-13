"""Legacy corpus-loader utilities and migration notes.

The legacy ``CorpusLoader``, ``DatasetReaderEntry``, and ``build_corpus_loader``
have been removed.  They are replaced by:

- ``load_corpus`` / ``Corpus`` (``corpus_loader.py``) — preferred user-facing
  corpus API for sampler setup, dataset access, and CPU/GPU loader iteration
- ``MetadataIndex``  (``index.py``)  — metadata query / sampling
- ``ExpressionReader`` (``expression.py``) — expression I/O

Utility functions ``read_raw_obs_parquet`` and ``read_raw_var_parquet`` are
preserved for ad-hoc metadata inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

__all__ = [
    "read_raw_obs_parquet",
    "read_raw_var_parquet",
]


def read_raw_obs_parquet(parquet_path: Path) -> list[dict[str, Any]]:
    """Read raw cell metadata from a Stage 2 Parquet sidecar.

    Parameters
    ----------
    parquet_path : Path
        Path to ``raw-obs.parquet`` written by ``Stage2Materializer``.
        Schema: cell_id (string), dataset_id (string), raw_fields
        (string, JSON-serialized dict), plus optional top-level provenance
        fields such as source_row_index/source_obs_index.

    Returns
    -------
    list[dict[str, Any]]
        List of row dicts.
    """
    table = pq.read_table(str(parquet_path))
    result: list[dict[str, Any]] = []
    for row in table.to_pylist():
        item = dict(row)
        item["cell_id"] = str(item["cell_id"])
        item["dataset_id"] = str(item["dataset_id"])
        item["raw_fields"] = json.loads(item["raw_fields"])
        result.append(item)
    return result


def read_raw_var_parquet(parquet_path: Path) -> list[dict[str, Any]]:
    """Read raw feature metadata from a Stage 2 Parquet sidecar.

    Parameters
    ----------
    parquet_path : Path
        Path to ``raw-var.parquet`` written by ``Stage2Materializer``.
        Schema: origin_index (int32), feature_id (string),
        raw_var (string, JSON-serialized dict).

    Returns
    -------
    list[dict[str, Any]]
        List of row dicts.
    """
    table = pq.read_table(str(parquet_path))
    result: list[dict[str, Any]] = []
    for row in table.to_pylist():
        result.append({
            "origin_index": int(row["origin_index"]),
            "feature_id": str(row["feature_id"]),
            "raw_fields": json.loads(row["raw_var"]),
        })
    return result
