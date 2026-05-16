"""Reusable slim-main public-API validation helpers.

These helpers exercise the federated corpus public API through a single
contract so Lance and Zarr parity checks stay consistent.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from ..materializers.paths import resolve_corpus_paths
from .corpus_loader import Corpus, load_corpus
from .loaders import build_loader

__all__ = ["validate_cross_backend_contract"]


def validate_cross_backend_contract(
    corpus_roots: Mapping[str, str | Path],
    *,
    sample_indices: Sequence[int],
    metadata_columns: Sequence[str] = ("perturb_label",),
    loader_batch_size: int = 128,
    seq_len: int = 8,
    loader_seed: int = 11,
    loader_rng_seed: int = 123,
) -> dict[str, Any]:
    """Validate federated corpus parity through one shared public-API contract.

    Parameters
    ----------
    corpus_roots:
        Mapping from backend label to corpus root. Backend labels may use dash
        or underscore forms such as ``"lance"`` and ``"zarr"``.
    sample_indices:
        Ordered global row indices used for deterministic cross-backend reads.
    metadata_columns:
        Metadata columns requested through ``build_loader(...)``.
    loader_batch_size:
        Batch size used for the CPU loader comparison.
    seq_len:
        Sequence length used for the CPU loader comparison.
    loader_seed:
        Sampler seed used for the CPU loader comparison.
    loader_rng_seed:
        Python / NumPy / Torch RNG seed reset before each loader read.

    Returns
    -------
    dict
        Structured per-backend validation report plus pairwise parity results.

    Raises
    ------
    AssertionError
        If any backend disagrees with the baseline backend on corpus-index
        ranges or on the public API contract layers.
    FileNotFoundError
        If any required artifact is missing.
    ValueError
        If a corpus is not a supported federated backend corpus.
    """
    if not corpus_roots:
        raise ValueError("validate_cross_backend_contract requires at least one backend corpus")

    normalized_indices = [int(index) for index in sample_indices]
    if not normalized_indices:
        raise ValueError("sample_indices must contain at least one global row index")
    if loader_batch_size <= 0:
        raise ValueError("loader_batch_size must be a positive integer")
    if seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")

    report: dict[str, Any] = {
        "sample_indices": normalized_indices,
        "metadata_columns": [str(column) for column in metadata_columns],
        "loader": {
            "batch_size": int(loader_batch_size),
            "seq_len": int(seq_len),
            "sampler_seed": int(loader_seed),
            "rng_seed": int(loader_rng_seed),
        },
        "backends": {},
        "comparisons": {
            "baseline_backend": "none",
            "pairs": {},
        },
    }

    backend_state: dict[str, dict[str, Any]] = {}
    for requested_backend, corpus_root in corpus_roots.items():
        backend = _normalize_backend_label(requested_backend)
        root = Path(corpus_root).resolve()
        index_doc = _read_corpus_index(root)
        artifact_report = _validate_backend_artifacts(backend, root, index_doc)
        corpus = _load_backend_corpus(backend, root)
        index_report = _validate_loaded_corpus_index(backend, corpus, index_doc)

        expression_batch = corpus.expression_reader.read_expression_flat(normalized_indices)
        taken_metadata = corpus.take_metadata(
            normalized_indices,
            columns=metadata_columns,
        )
        loader_batch = _read_loader_batch(
            corpus,
            metadata_columns=metadata_columns,
            batch_size=loader_batch_size,
            seq_len=seq_len,
            sampler_seed=loader_seed,
            rng_seed=loader_rng_seed,
        )

        report["backends"][backend] = {
            "corpus_root": str(root),
            "artifact_checks": artifact_report,
            "corpus_index": index_report,
            "load_corpus": {
                "status": "success",
                "backend": corpus.backend,
                "topology": corpus.topology,
                "dataset_entry_count": len(corpus.dataset_entries),
            },
            "expression_reader": {
                "batch_size": int(expression_batch.batch_size),
                "nonzero_count": int(expression_batch.expression_counts.size),
                "global_row_index_preview": expression_batch.global_row_index[:5].tolist(),
            },
            "take_metadata": {
                "columns": sorted(taken_metadata.keys()),
            },
            "loader": {
                "batch_size": int(loader_batch["batch_size"]),
                "seq_len": int(loader_batch["seq_len"]),
                "metadata_columns": sorted(loader_batch.get("meta_columns", {}).keys()),
                "global_row_index_preview": loader_batch["global_row_index"][:5].tolist(),
            },
        }
        backend_state[backend] = {
            "corpus": corpus,
            "expression": expression_batch,
            "take_metadata": taken_metadata,
            "loader": loader_batch,
            "index_report": index_report,
        }

    backend_order = list(report["backends"])
    baseline_backend = backend_order[0]
    report["comparisons"]["baseline_backend"] = baseline_backend
    baseline_state = backend_state[baseline_backend]

    for backend in backend_order[1:]:
        current_state = backend_state[backend]
        _assert_pair_report_equal(
            current_state["index_report"],
            baseline_state["index_report"],
            layer="corpus_index",
            backend=backend,
            baseline_backend=baseline_backend,
        )
        _assert_expression_batch_equal(
            current_state["expression"],
            baseline_state["expression"],
            layer="expression_reader",
            backend=backend,
            baseline_backend=baseline_backend,
        )
        _assert_meta_columns_equal(
            current_state["take_metadata"],
            baseline_state["take_metadata"],
            layer="take_metadata",
            backend=backend,
            baseline_backend=baseline_backend,
        )
        _assert_loader_batch_equal(
            current_state["loader"],
            baseline_state["loader"],
            layer="loader",
            backend=backend,
            baseline_backend=baseline_backend,
        )

        report["comparisons"]["pairs"][f"{backend}_vs_{baseline_backend}"] = {
            "status": "success",
            "layers": {
                "corpus_index": "success",
                "expression_reader": "success",
                "take_metadata": "success",
                "loader": "success",
            },
        }

    return report


def _load_backend_corpus(backend: str, corpus_root: Path) -> Corpus:
    try:
        corpus = load_corpus(corpus_root)
    except Exception as exc:  # pragma: no cover - exact exception type is preserved
        raise type(exc)(f"{backend} failed at load_corpus: {exc}") from exc
    if corpus.topology != "federated":
        raise ValueError(
            f"{backend} failed at load_corpus: expected topology 'federated', got '{corpus.topology}'"
        )
    if corpus.backend != backend:
        raise AssertionError(
            f"{backend} failed at load_corpus: loader returned backend '{corpus.backend}'"
        )
    return corpus


def _read_corpus_index(corpus_root: Path) -> dict[str, Any]:
    corpus_index_path = corpus_root / "corpus-index.yaml"
    if not corpus_index_path.is_file():
        raise FileNotFoundError(f"corpus-index.yaml not found at {corpus_index_path}")
    with open(corpus_index_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _validate_backend_artifacts(
    backend: str,
    corpus_root: Path,
    index_doc: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = index_doc.get("global_metadata", {})
    topology = str(metadata.get("topology", ""))
    if topology != "federated":
        raise ValueError(
            f"{backend} failed at artifact existence: expected federated topology, got '{topology}'"
        )

    index_backend = _normalize_backend_label(str(metadata.get("backend", "")))
    if index_backend != backend:
        raise AssertionError(
            f"{backend} failed at artifact existence: corpus-index backend is '{index_backend}'"
        )

    datasets = list(index_doc.get("datasets", []))
    if not datasets:
        raise ValueError(f"{backend} failed at artifact existence: no datasets found in corpus-index.yaml")

    dataset_reports: list[dict[str, Any]] = []
    for item in datasets:
        dataset_id = str(item["dataset_id"])
        paths = resolve_corpus_paths("federated", corpus_root, dataset_id)
        canonical_obs_path = paths.canonical_meta_root / "canonical-obs.parquet"
        canonical_var_path = paths.canonical_meta_root / "canonical-var.parquet"
        matrix_artifacts = _matrix_artifact_report(backend, paths.matrix_root, dataset_id)

        if not canonical_obs_path.is_file():
            raise FileNotFoundError(
                f"{backend} failed at artifact existence for dataset '{dataset_id}': "
                f"missing canonical obs parquet at {canonical_obs_path}"
            )
        if not canonical_var_path.is_file():
            raise FileNotFoundError(
                f"{backend} failed at artifact existence for dataset '{dataset_id}': "
                f"missing canonical var parquet at {canonical_var_path}"
            )
        dataset_reports.append(
            {
                "dataset_id": dataset_id,
                "dataset_index": int(item.get("dataset_index", len(dataset_reports))),
                "global_start": int(item.get("global_start", 0)),
                "global_end": int(item.get("global_end", 0)),
                "canonical_obs_path": str(canonical_obs_path),
                "canonical_var_path": str(canonical_var_path),
                **matrix_artifacts,
            }
        )

    return {
        "status": "success",
        "dataset_count": len(dataset_reports),
        "datasets": dataset_reports,
    }


def _validate_loaded_corpus_index(
    backend: str,
    corpus: Corpus,
    index_doc: Mapping[str, Any],
) -> dict[str, Any]:
    datasets = list(index_doc.get("datasets", []))
    if len(corpus.dataset_entries) != len(datasets):
        raise AssertionError(
            f"{backend} failed at corpus_index: dataset entry count {len(corpus.dataset_entries)} "
            f"!= corpus-index count {len(datasets)}"
        )

    dataset_reports: list[dict[str, Any]] = []
    previous_end: int | None = None
    total_cells = 0
    for index, (item, entry) in enumerate(zip(datasets, corpus.dataset_entries, strict=True)):
        dataset_id = str(item["dataset_id"])
        dataset_index = int(item.get("dataset_index", index))
        global_start = int(item.get("global_start", 0))
        global_end = int(item.get("global_end", 0))
        cell_count = int(item.get("cell_count", global_end - global_start))

        if entry.dataset_id != dataset_id:
            raise AssertionError(
                f"{backend} failed at corpus_index: dataset entry '{entry.dataset_id}' != '{dataset_id}'"
            )
        if entry.global_start != global_start or entry.global_end != global_end:
            raise AssertionError(
                f"{backend} failed at corpus_index for dataset '{dataset_id}': "
                f"entry range [{entry.global_start}, {entry.global_end}) != index range [{global_start}, {global_end})"
            )
        if corpus.dataset_index_by_id.get(dataset_id) != dataset_index:
            raise AssertionError(
                f"{backend} failed at corpus_index for dataset '{dataset_id}': "
                f"dataset_index {corpus.dataset_index_by_id.get(dataset_id)} != {dataset_index}"
            )
        if previous_end is not None and global_start != previous_end:
            raise AssertionError(
                f"{backend} failed at corpus_index: non-contiguous ranges at dataset '{dataset_id}'"
            )
        if global_end - global_start != cell_count:
            raise AssertionError(
                f"{backend} failed at corpus_index for dataset '{dataset_id}': "
                f"cell_count {cell_count} != range width {global_end - global_start}"
            )

        dataset_reports.append(
            {
                "dataset_id": dataset_id,
                "dataset_index": dataset_index,
                "cell_count": cell_count,
                "global_start": global_start,
                "global_end": global_end,
            }
        )
        total_cells += cell_count
        previous_end = global_end

    if len(corpus.metadata_index) != total_cells:
        raise AssertionError(
            f"{backend} failed at corpus_index: metadata_index length {len(corpus.metadata_index)} != {total_cells}"
        )

    return {
        "dataset_count": len(dataset_reports),
        "total_cells": total_cells,
        "datasets": dataset_reports,
    }


def _matrix_artifact_report(backend: str, matrix_root: Path, dataset_id: str) -> dict[str, Any]:
    if backend == "lance":
        matrix_path = matrix_root / f"{dataset_id}.lance"
        if not matrix_path.exists():
            raise FileNotFoundError(
                f"{backend} failed at artifact existence for dataset '{dataset_id}': "
                f"missing Lance dataset at {matrix_path}"
            )
        return {"matrix_path": str(matrix_path)}

    if backend == "zarr":
        artifacts = {
            "row_offsets_path": matrix_root / f"{dataset_id}-row-offsets.zarr",
            "indices_path": matrix_root / f"{dataset_id}-indices.zarr",
            "counts_path": matrix_root / f"{dataset_id}-counts.zarr",
        }
        for label, path in artifacts.items():
            if not path.is_dir():
                raise FileNotFoundError(
                    f"{backend} failed at artifact existence for dataset '{dataset_id}': "
                    f"missing {label} at {path}"
                )
        return {"matrix_paths": {label: str(path) for label, path in artifacts.items()}}

    raise ValueError(f"Unsupported backend label '{backend}'")


def _normalize_backend_label(label: str) -> str:
    normalized = str(label).strip().lower().replace("-", "_")
    aliases = {
        "zarr": "zarr",
        "lance": "lance",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported backend label '{label}'. Expected 'lance' or 'zarr'."
        )
    return resolved


def _read_loader_batch(
    corpus: Corpus,
    *,
    metadata_columns: Sequence[str],
    batch_size: int,
    seq_len: int,
    sampler_seed: int,
    rng_seed: int,
) -> dict[str, Any]:
    _reset_loader_rngs(rng_seed)
    batch = next(
        build_loader(
            corpus,
            batch_size=batch_size,
            seq_len=seq_len,
            seed=sampler_seed,
            num_workers=0,
            metadata_columns=metadata_columns,
        )
    )
    _assert_loader_contract(batch, batch_size=batch_size, seq_len=seq_len)
    return batch


def _reset_loader_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _assert_loader_contract(batch: dict[str, Any], *, batch_size: int, seq_len: int) -> None:
    if batch["batch_size"] != batch_size:
        raise AssertionError(
            f"loader contract expected batch_size={batch_size}, got {batch['batch_size']}"
        )
    if batch["seq_len"] != seq_len:
        raise AssertionError(
            f"loader contract expected seq_len={seq_len}, got {batch['seq_len']}"
        )
    required_tensor_keys = (
        "sampled_gene_ids",
        "sampled_counts",
        "valid_mask",
        "exact_match_mask",
        "dataset_index",
        "global_row_index",
    )
    for key in required_tensor_keys:
        if not isinstance(batch[key], torch.Tensor):
            raise AssertionError(f"loader contract expected torch.Tensor for '{key}'")
    for forbidden_key in ("row_offsets", "expressed_gene_indices", "expression_counts"):
        if forbidden_key in batch:
            raise AssertionError(f"loader contract should not expose '{forbidden_key}'")


def _assert_pair_report_equal(
    actual: Any,
    expected: Any,
    *,
    layer: str,
    backend: str,
    baseline_backend: str,
) -> None:
    if actual != expected:
        raise AssertionError(
            f"{backend} vs {baseline_backend} failed at {layer}: {actual!r} != {expected!r}"
        )


def _assert_expression_batch_equal(
    actual: Any,
    expected: Any,
    *,
    layer: str,
    backend: str,
    baseline_backend: str,
) -> None:
    if actual.batch_size != expected.batch_size:
        raise AssertionError(
            f"{backend} vs {baseline_backend} failed at {layer}.batch_size: "
            f"{actual.batch_size} != {expected.batch_size}"
        )
    _assert_array_equal(
        actual.global_row_index,
        expected.global_row_index,
        layer=f"{layer}.global_row_index",
        backend=backend,
        baseline_backend=baseline_backend,
    )
    _assert_array_equal(
        actual.row_offsets,
        expected.row_offsets,
        layer=f"{layer}.row_offsets",
        backend=backend,
        baseline_backend=baseline_backend,
    )
    _assert_array_equal(
        actual.expressed_gene_indices,
        expected.expressed_gene_indices,
        layer=f"{layer}.expressed_gene_indices",
        backend=backend,
        baseline_backend=baseline_backend,
    )
    _assert_array_equal(
        actual.expression_counts,
        expected.expression_counts,
        layer=f"{layer}.expression_counts",
        backend=backend,
        baseline_backend=baseline_backend,
    )


def _assert_loader_batch_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    layer: str,
    backend: str,
    baseline_backend: str,
) -> None:
    if actual.keys() != expected.keys():
        raise AssertionError(
            f"{backend} vs {baseline_backend} failed at {layer}.keys: {sorted(actual.keys())} != {sorted(expected.keys())}"
        )
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, torch.Tensor):
            if not torch.equal(actual_value, expected_value):
                raise AssertionError(
                    f"{backend} vs {baseline_backend} failed at {layer}.{key}: tensor values differ"
                )
            continue
        if isinstance(expected_value, dict):
            _assert_meta_columns_equal(
                actual_value,
                expected_value,
                layer=f"{layer}.{key}",
                backend=backend,
                baseline_backend=baseline_backend,
            )
            continue
        if actual_value != expected_value:
            raise AssertionError(
                f"{backend} vs {baseline_backend} failed at {layer}.{key}: {actual_value!r} != {expected_value!r}"
            )


def _assert_meta_columns_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    layer: str,
    backend: str,
    baseline_backend: str,
) -> None:
    if set(actual) != set(expected):
        raise AssertionError(
            f"{backend} vs {baseline_backend} failed at {layer}.keys: {sorted(actual)} != {sorted(expected)}"
        )
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, np.ndarray):
            _assert_array_equal(
                actual_value,
                expected_value,
                layer=f"{layer}.{key}",
                backend=backend,
                baseline_backend=baseline_backend,
            )
        elif actual_value != expected_value:
            raise AssertionError(
                f"{backend} vs {baseline_backend} failed at {layer}.{key}: {actual_value!r} != {expected_value!r}"
            )


def _assert_array_equal(
    actual: Any,
    expected: Any,
    *,
    layer: str,
    backend: str,
    baseline_backend: str,
) -> None:
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as exc:
        raise AssertionError(
            f"{backend} vs {baseline_backend} failed at {layer}: {exc}"
        ) from exc
