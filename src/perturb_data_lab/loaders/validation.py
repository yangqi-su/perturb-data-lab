"""Simple structural validation for materialized corpora."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from ..materializers.paths import resolve_corpus_paths
from .corpus_loader import _normalize_backend, load_corpus

__all__ = ["validate_corpus_structure"]


def validate_corpus_structure(
    corpus_root: str | Path,
    *,
    sample_n: int = 128,
    seed: int = 0,
) -> dict[str, Any]:
    """Validate one materialized corpus without comparing against another backend.

    The checks focus on row-index consistency: corpus-index ranges, metadata
    rows, matrix row counts, and sampled expression/metadata bridge reads.
    """
    root = Path(corpus_root).resolve()
    index_doc = _read_corpus_index(root)
    datasets = _validate_index_ranges(index_doc)
    corpus = load_corpus(root)

    if corpus.backend != _normalize_backend(str(index_doc["global_metadata"]["backend"])):
        raise AssertionError("loaded corpus backend does not match corpus-index.yaml")
    if corpus.topology != str(index_doc["global_metadata"]["topology"]):
        raise AssertionError("loaded corpus topology does not match corpus-index.yaml")

    total_rows = datasets[-1]["global_end"]
    _validate_metadata_index(corpus.metadata_index.df, datasets, total_rows)

    matrix_report = _validate_matrix_storage(root, corpus.backend, corpus.topology, datasets)
    sample_indices = _sample_global_rows(total_rows, datasets, sample_n=sample_n, seed=seed)
    _validate_sampled_bridge(corpus, datasets, sample_indices)

    return {
        "status": "success",
        "corpus_root": str(root),
        "backend": corpus.backend,
        "topology": corpus.topology,
        "dataset_count": len(datasets),
        "total_rows": total_rows,
        "sample_count": len(sample_indices),
        "sample_indices": sample_indices.tolist(),
        "matrix": matrix_report,
    }


def _read_corpus_index(root: Path) -> dict[str, Any]:
    path = root / "corpus-index.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"corpus-index.yaml not found at {path}")
    with open(path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    if "global_metadata" not in doc or "datasets" not in doc:
        raise ValueError("corpus-index.yaml must contain global_metadata and datasets")
    return doc


def _validate_index_ranges(index_doc: dict[str, Any]) -> list[dict[str, Any]]:
    raw_datasets = list(index_doc["datasets"])
    if not raw_datasets:
        raise ValueError("corpus-index.yaml datasets cannot be empty")

    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    expected_start = 0
    datasets: list[dict[str, Any]] = []
    for position, item in enumerate(raw_datasets):
        dataset_id = str(item["dataset_id"])
        dataset_index = int(item.get("dataset_index", position))
        global_start = int(item["global_start"])
        global_end = int(item["global_end"])
        cell_count = int(item.get("cell_count", global_end - global_start))

        if dataset_id in seen_ids:
            raise AssertionError(f"duplicate dataset_id {dataset_id!r}")
        if dataset_index in seen_indices:
            raise AssertionError(f"duplicate dataset_index {dataset_index}")
        if dataset_index != position:
            raise AssertionError("dataset_index values must be contiguous in corpus-index order")
        if global_start != expected_start:
            raise AssertionError("global ranges must be contiguous and start at 0")
        if global_end <= global_start:
            raise AssertionError(f"dataset {dataset_id!r} has empty or negative global range")
        if cell_count != global_end - global_start:
            raise AssertionError(f"dataset {dataset_id!r} cell_count does not match global range")

        seen_ids.add(dataset_id)
        seen_indices.add(dataset_index)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "dataset_index": dataset_index,
                "global_start": global_start,
                "global_end": global_end,
                "cell_count": cell_count,
            }
        )
        expected_start = global_end
    return datasets


def _validate_metadata_index(
    df: pl.DataFrame,
    datasets: list[dict[str, Any]],
    total_rows: int,
) -> None:
    required = {"global_row_index", "dataset_id", "dataset_index", "local_row_index", "cell_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise AssertionError(f"metadata_index is missing required columns: {missing}")
    if len(df) != total_rows:
        raise AssertionError(f"metadata_index has {len(df)} rows, expected {total_rows}")
    np.testing.assert_array_equal(
        df["global_row_index"].to_numpy(),
        np.arange(total_rows, dtype=np.int64),
    )
    if df["cell_id"].null_count() > 0:
        raise AssertionError("metadata_index cell_id contains null values")

    for dataset in datasets:
        start = dataset["global_start"]
        end = dataset["global_end"]
        sub = df.slice(start, end - start)
        if set(sub["dataset_id"].unique().to_list()) != {dataset["dataset_id"]}:
            raise AssertionError(f"metadata dataset_id mismatch in range for {dataset['dataset_id']!r}")
        np.testing.assert_array_equal(
            sub["dataset_index"].to_numpy(),
            np.full(dataset["cell_count"], dataset["dataset_index"], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            sub["local_row_index"].to_numpy(),
            np.arange(dataset["cell_count"], dtype=np.int64),
        )


def _validate_matrix_storage(
    root: Path,
    backend: str,
    topology: str,
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    if topology == "aggregate":
        total_rows = datasets[-1]["global_end"]
        if backend == "lance":
            _validate_lance_rows(root / "matrix" / "aggregated-cells.lance", total_rows)
        else:
            _validate_zarr_arrays(
                root / "matrix" / "aggregated-row-offsets.zarr",
                root / "matrix" / "aggregated-indices.zarr",
                root / "matrix" / "aggregated-counts.zarr",
                total_rows,
            )
        return {"checked_layout": "aggregate", "row_count": total_rows}

    for dataset in datasets:
        paths = resolve_corpus_paths("federated", root, dataset["dataset_id"])
        if backend == "lance":
            _validate_lance_rows(
                paths.matrix_root / f"{dataset['dataset_id']}.lance",
                dataset["cell_count"],
            )
        else:
            _validate_zarr_arrays(
                paths.matrix_root / f"{dataset['dataset_id']}-row-offsets.zarr",
                paths.matrix_root / f"{dataset['dataset_id']}-indices.zarr",
                paths.matrix_root / f"{dataset['dataset_id']}-counts.zarr",
                dataset["cell_count"],
            )
    return {"checked_layout": "federated", "row_count": datasets[-1]["global_end"]}


def _validate_lance_rows(path: Path, expected_rows: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Lance matrix not found at {path}")
    import lance

    ds = lance.dataset(str(path))
    if int(ds.count_rows()) != expected_rows:
        raise AssertionError(f"{path} row count does not match corpus-index")
    names = set(ds.schema.names)
    missing = {"expressed_gene_indices", "expression_counts"} - names
    if missing:
        raise AssertionError(f"{path} is missing matrix columns: {sorted(missing)}")


def _validate_zarr_arrays(
    row_offsets_path: Path,
    indices_path: Path,
    counts_path: Path,
    expected_rows: int,
) -> None:
    if not row_offsets_path.is_dir() or not indices_path.is_dir() or not counts_path.is_dir():
        raise FileNotFoundError("Zarr row_offsets, indices, and counts artifacts must all exist")
    import zarr

    row_offsets = zarr.open(str(row_offsets_path), mode="r")["row_offsets"][:]
    indices = zarr.open(str(indices_path), mode="r")["indices"]
    counts = zarr.open(str(counts_path), mode="r")["counts"]
    if len(row_offsets) != expected_rows + 1:
        raise AssertionError("Zarr row_offsets length does not match expected rows")
    if int(row_offsets[0]) != 0:
        raise AssertionError("Zarr row_offsets must start at 0")
    if np.any(np.diff(row_offsets) < 0):
        raise AssertionError("Zarr row_offsets must be monotonic")
    if int(row_offsets[-1]) != int(indices.shape[0]) or int(indices.shape[0]) != int(counts.shape[0]):
        raise AssertionError("Zarr row_offsets, indices, and counts lengths disagree")


def _sample_global_rows(
    total_rows: int,
    datasets: list[dict[str, Any]],
    *,
    sample_n: int,
    seed: int,
) -> np.ndarray:
    if sample_n <= 0:
        raise ValueError("sample_n must be positive")
    boundary_rows: list[int] = [0, total_rows - 1]
    for dataset in datasets:
        boundary_rows.extend([dataset["global_start"], dataset["global_end"] - 1])
    if sample_n >= total_rows:
        return np.arange(total_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    random_rows = rng.choice(total_rows, size=min(sample_n, total_rows), replace=False).tolist()
    return np.asarray(sorted(set(boundary_rows + random_rows)), dtype=np.int64)


def _validate_sampled_bridge(corpus: Any, datasets: list[dict[str, Any]], sample_indices: np.ndarray) -> None:
    expression = corpus.expression_reader.read_expression_flat(sample_indices.tolist())
    metadata = corpus.take_metadata(
        sample_indices,
        columns=["global_row_index", "dataset_id", "dataset_index", "local_row_index"],
    )
    np.testing.assert_array_equal(expression.global_row_index, sample_indices)
    np.testing.assert_array_equal(metadata["global_row_index"], sample_indices)

    for position, global_row in enumerate(sample_indices.tolist()):
        dataset = _dataset_for_global_row(datasets, global_row)
        expected_local = global_row - dataset["global_start"]
        if metadata["dataset_id"][position] != dataset["dataset_id"]:
            raise AssertionError(f"sampled row {global_row} has wrong dataset_id")
        if int(metadata["dataset_index"][position]) != dataset["dataset_index"]:
            raise AssertionError(f"sampled row {global_row} has wrong dataset_index")
        if int(metadata["local_row_index"][position]) != expected_local:
            raise AssertionError(f"sampled row {global_row} has wrong local_row_index")

        genes = expression.row_gene_indices(position)
        counts = expression.row_counts(position)
        if len(genes) != len(counts):
            raise AssertionError(f"sampled row {global_row} has mismatched sparse arrays")
        if np.any(genes < 0):
            raise AssertionError(f"sampled row {global_row} has negative gene indices")
        local_feature_count = int(
            np.sum(corpus.feature_registry.local_to_global_map[dataset["dataset_index"]] >= 0)
        )
        if genes.size and int(genes.max()) >= local_feature_count:
            raise AssertionError(f"sampled row {global_row} has out-of-range gene indices")
        if np.any(counts < 0):
            raise AssertionError(f"sampled row {global_row} has negative counts")


def _dataset_for_global_row(datasets: list[dict[str, Any]], global_row: int) -> dict[str, Any]:
    for dataset in datasets:
        if dataset["global_start"] <= global_row < dataset["global_end"]:
            return dataset
    raise AssertionError(f"sampled row {global_row} is outside corpus ranges")
