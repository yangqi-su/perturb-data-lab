"""True Lance aggregated backend with corpus-scoped heavy storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pyarrow as pa

from .arrow_hf import _get_row_nonzero, write_arrow_hf_metadata_artifacts
from ..models import CorpusIndexDocument, DatasetJoinRecord

AGGREGATED_LANCE_TABLE_NAME = "aggregated-corpus"
AGGREGATED_LANCE_DATASET_NAME = f"{AGGREGATED_LANCE_TABLE_NAME}.lance"
AGGREGATED_LANCE_SIDECAR_NAME = f"{AGGREGATED_LANCE_TABLE_NAME}-append-log.json"


def _import_lance_stack() -> tuple[Any, Any]:
    try:
        import lance
        import lancedb
    except ImportError as exc:  # pragma: no cover - exercised in environment-specific runs
        raise ImportError(
            "lancedb-aggregated backend requires the true Lance stack; "
            "install lancedb==0.30.2 and deprecation==2.1.0 in the selected runtime"
        ) from exc
    return lance, lancedb


def _resolve_corpus_matrix_root(
    matrix_root: Path,
    corpus_index_path: Path | None,
) -> Path:
    if corpus_index_path is not None:
        return corpus_index_path.parent / "matrix"
    return matrix_root


def _reserve_append_range(
    corpus_index_path: Path | None,
    cell_count: int,
) -> tuple[int, int, int]:
    if corpus_index_path is not None and corpus_index_path.exists():
        corpus_index = CorpusIndexDocument.from_yaml_file(corpus_index_path)
        dataset_index = len(corpus_index.datasets)
        global_row_start = max((record.global_end for record in corpus_index.datasets), default=0)
    else:
        dataset_index = 0
        global_row_start = 0
    global_row_end = global_row_start + int(cell_count)
    return dataset_index, global_row_start, global_row_end


def _default_sidecar_payload(dataset_uri: Path, db_root: Path) -> dict[str, Any]:
    return {
        "backend": "lancedb-aggregated",
        "table_name": AGGREGATED_LANCE_TABLE_NAME,
        "dataset_uri": str(dataset_uri),
        "db_root": str(db_root),
        "entries": [],
    }


def _load_sidecar(sidecar_path: Path, dataset_uri: Path, db_root: Path) -> dict[str, Any]:
    if not sidecar_path.exists():
        return _default_sidecar_payload(dataset_uri, db_root)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid Lance append sidecar payload: {sidecar_path}")
    payload.setdefault("backend", "lancedb-aggregated")
    payload.setdefault("table_name", AGGREGATED_LANCE_TABLE_NAME)
    payload.setdefault("dataset_uri", str(dataset_uri))
    payload.setdefault("db_root", str(db_root))
    payload.setdefault("entries", [])
    return payload


def _write_sidecar(sidecar_path: Path, payload: dict[str, Any]) -> None:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _replace_sidecar_entry(payload: dict[str, Any], entry: dict[str, Any]) -> None:
    entries = list(payload.get("entries", []))
    replaced = False
    for idx, existing in enumerate(entries):
        if (
            int(existing.get("dataset_index", -1)) == int(entry["dataset_index"])
            and str(existing.get("dataset_id", "")) == str(entry["dataset_id"])
            and str(existing.get("release_id", "")) == str(entry["release_id"])
        ):
            entries[idx] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)
    payload["entries"] = entries


def _batch_tables(
    count_matrix: Any,
    size_factors: np.ndarray,
    *,
    dataset_index: int,
    global_row_start: int,
    cell_count: int,
    batch_size: int = 256,
) -> list[pa.Table]:
    batches: list[pa.Table] = []
    for start in range(0, cell_count, batch_size):
        stop = min(start + batch_size, cell_count)
        global_row_index: list[int] = []
        dataset_indices: list[int] = []
        local_row_index: list[int] = []
        expressed_gene_indices: list[list[int]] = []
        expression_counts: list[list[int]] = []
        size_factor_batch: list[float] = []

        for local_index in range(start, stop):
            indices, counts = _get_row_nonzero(count_matrix, local_index)
            global_row_index.append(global_row_start + local_index)
            dataset_indices.append(dataset_index)
            local_row_index.append(local_index)
            expressed_gene_indices.append(indices.tolist())
            expression_counts.append(counts.tolist())
            size_factor_batch.append(float(size_factors[local_index]))

        batches.append(
            pa.table(
                {
                    "global_row_index": pa.array(global_row_index, type=pa.int64()),
                    "dataset_index": pa.array(dataset_indices, type=pa.int32()),
                    "local_row_index": pa.array(local_row_index, type=pa.int64()),
                    "expressed_gene_indices": pa.array(
                        expressed_gene_indices,
                        type=pa.list_(pa.int32()),
                    ),
                    "expression_counts": pa.array(
                        expression_counts,
                        type=pa.list_(pa.int32()),
                    ),
                    "size_factor": pa.array(size_factor_batch, type=pa.float64()),
                }
            )
        )
    return batches


def write_lancedb_aggregated(
    adata: ad.AnnData,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
    dataset_id: str = "",
    corpus_index_path: Path | None = None,
) -> dict[str, Path]:
    """Write one dataset append into a corpus-scoped Lance dataset."""
    lance, lancedb = _import_lance_stack()

    cell_count = int(adata.n_obs)
    dataset_index, global_row_start, global_row_end = _reserve_append_range(
        corpus_index_path,
        cell_count,
    )
    corpus_matrix_root = _resolve_corpus_matrix_root(matrix_root, corpus_index_path)
    corpus_matrix_root.mkdir(parents=True, exist_ok=True)
    dataset_uri = corpus_matrix_root / AGGREGATED_LANCE_DATASET_NAME
    sidecar_path = corpus_matrix_root / AGGREGATED_LANCE_SIDECAR_NAME

    metadata_paths = write_arrow_hf_metadata_artifacts(
        adata=adata,
        size_factors=size_factors,
        release_id=release_id,
        matrix_root=matrix_root,
        canonical_perturbation=canonical_perturbation,
        canonical_context=canonical_context,
        raw_fields=raw_fields,
        dataset_id=dataset_id,
    )

    payload = _load_sidecar(sidecar_path, dataset_uri, corpus_matrix_root)
    entry: dict[str, Any] = {
        "dataset_index": dataset_index,
        "dataset_id": dataset_id,
        "release_id": release_id,
        "global_row_start": global_row_start,
        "global_row_end": global_row_end,
        "cell_count": cell_count,
        "lance_version": "none",
        "status": "pending",
    }
    _replace_sidecar_entry(payload, entry)
    _write_sidecar(sidecar_path, payload)

    try:
        db = lancedb.connect(corpus_matrix_root)
        table_exists = dataset_uri.exists()
        before_count = 0
        table = None
        if table_exists:
            table = db.open_table(AGGREGATED_LANCE_TABLE_NAME)
            before_count = int(table.count_rows())
            if before_count != global_row_start:
                raise ValueError(
                    "Lance append cannot reserve the requested global row range because "
                    f"the existing Lance row count ({before_count}) does not match the "
                    f"corpus-index reservation start ({global_row_start})"
                )
        elif global_row_start != 0:
            raise ValueError(
                "Lance append expected an existing aggregated dataset for nonzero "
                f"global_row_start={global_row_start}, but none was found"
            )

        batches = _batch_tables(
            count_matrix,
            size_factors,
            dataset_index=dataset_index,
            global_row_start=global_row_start,
            cell_count=cell_count,
        )
        if batches:
            if table is None:
                table = db.create_table(AGGREGATED_LANCE_TABLE_NAME, data=batches[0], mode="create")
                for batch in batches[1:]:
                    table.add(batch)
            else:
                for batch in batches:
                    table.add(batch)
        elif table is None:
            raise ValueError("lancedb-aggregated requires at least one cell to append")

        assert table is not None
        after_count = int(table.count_rows())
        if after_count - before_count != cell_count:
            raise ValueError(
                "Lance append row-count delta mismatch: "
                f"expected {cell_count}, observed {after_count - before_count}"
            )

        dataset_rows = int(table.count_rows(f"dataset_index = {dataset_index}"))
        if dataset_rows != cell_count:
            raise ValueError(
                "Lance append dataset_index row-count mismatch: "
                f"expected {cell_count}, observed {dataset_rows}"
            )

        if cell_count > 0:
            sampled = lance.dataset(dataset_uri).take([global_row_start, global_row_end - 1]).to_pylist()
            first_row = sampled[0]
            last_row = sampled[-1]
            if int(first_row["global_row_index"]) != global_row_start:
                raise ValueError("Lance append start global_row_index mismatch")
            if int(last_row["global_row_index"]) != global_row_end - 1:
                raise ValueError("Lance append end global_row_index mismatch")
            if int(first_row["dataset_index"]) != dataset_index or int(last_row["dataset_index"]) != dataset_index:
                raise ValueError("Lance append dataset_index mismatch at append boundaries")
            if int(first_row["local_row_index"]) != 0 or int(last_row["local_row_index"]) != cell_count - 1:
                raise ValueError("Lance append local_row_index mismatch at append boundaries")

        entry["lance_version"] = int(table.version)
        _replace_sidecar_entry(payload, entry)
        _write_sidecar(sidecar_path, payload)
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = str(exc)
        _replace_sidecar_entry(payload, entry)
        _write_sidecar(sidecar_path, payload)
        raise

    return {
        "cells": dataset_uri,
        **metadata_paths,
        "lance_append_sidecar": sidecar_path,
    }


def mark_lance_append_committed(
    corpus_index_path: Path,
    dataset_record: DatasetJoinRecord,
) -> Path | None:
    """Mark a pending Lance append sidecar record as committed."""
    sidecar_path = corpus_index_path.parent / "matrix" / AGGREGATED_LANCE_SIDECAR_NAME
    if not sidecar_path.exists():
        return None

    dataset_uri = corpus_index_path.parent / "matrix" / AGGREGATED_LANCE_DATASET_NAME
    payload = _load_sidecar(sidecar_path, dataset_uri, corpus_index_path.parent / "matrix")
    for entry in payload.get("entries", []):
        if (
            int(entry.get("dataset_index", -1)) == int(dataset_record.dataset_index)
            and str(entry.get("dataset_id", "")) == str(dataset_record.dataset_id)
            and str(entry.get("release_id", "")) == str(dataset_record.release_id)
        ):
            entry["status"] = "committed"
            break
    _write_sidecar(sidecar_path, payload)
    return sidecar_path
