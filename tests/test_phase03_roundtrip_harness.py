"""Phase 3 dummy-data roundtrip harness for federated Lance and Arrow IPC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch

from perturb_data_lab.canonical import draft_canonicalization_schema, run_canonicalization
from perturb_data_lab.inspectors.models import DatasetSummaryDocument, InspectionBatchConfig, InspectionTarget
from perturb_data_lab.inspectors.workflow import run_batch
from perturb_data_lab.loaders.corpus_loader import Corpus, load_corpus
from perturb_data_lab.materializers import Stage2Materializer
from perturb_data_lab.materializers.models import CorpusIndexDocument, MaterializationManifest, OutputRoots
from perturb_data_lab.materializers.paths import resolve_corpus_paths


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DUMMY_DATA_ROOT = WORKSPACE_ROOT / "dummy_data"
DUMMY_MANIFEST_PATH = DUMMY_DATA_ROOT / "manifest.json"
SELECTED_DATASET_IDS = ("dummy_00", "dummy_01")
LOADER_SEQ_LEN = 16
LOADER_METADATA_COLUMNS = ("dataset_id", "perturb_label")
METADATA_COLUMNS = ("dataset_id", "dataset_index", "local_row_index", "perturb_label")


@dataclass(frozen=True)
class DummyDatasetSpec:
    dataset_id: str
    source_path: Path
    n_cells: int
    n_genes: int
    gene_order_fingerprint: tuple[str, ...]


def _load_selected_dummy_specs() -> list[DummyDatasetSpec]:
    payload = json.loads(DUMMY_MANIFEST_PATH.read_text(encoding="utf-8"))
    by_id = {str(item["dataset_id"]): item for item in payload}
    specs: list[DummyDatasetSpec] = []
    for dataset_id in SELECTED_DATASET_IDS:
        item = by_id[dataset_id]
        specs.append(
            DummyDatasetSpec(
                dataset_id=dataset_id,
                source_path=DUMMY_DATA_ROOT / str(item["filename"]),
                n_cells=int(item["n_cells"]),
                n_genes=int(item["n_genes"]),
                gene_order_fingerprint=tuple(str(v) for v in item.get("gene_order_fingerprint", [])),
            )
        )
    return specs


def _artifact_root(tmp_path: Path) -> Path:
    requested = os.environ.get("PDL_PHASE3_ARTIFACT_ROOT")
    if requested:
        root = Path(requested).resolve()
    else:
        root = (tmp_path / "phase03-roundtrip").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _assert_expression_batch_equal(actual: Any, expected: Any) -> None:
    assert actual.batch_size == expected.batch_size
    np.testing.assert_array_equal(actual.global_row_index, expected.global_row_index)
    np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
    np.testing.assert_array_equal(actual.expressed_gene_indices, expected.expressed_gene_indices)
    np.testing.assert_array_equal(actual.expression_counts, expected.expression_counts)


def _assert_columnar_value_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        return
    assert actual == expected


def _assert_raw_batch_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    assert actual["batch_size"] == expected["batch_size"]
    for key in (
        "global_row_index",
        "dataset_index",
        "local_row_index",
        "row_offsets",
        "expressed_gene_indices",
        "expression_counts",
        "size_factor",
    ):
        if key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    if "meta_columns" in expected:
        assert set(actual["meta_columns"]) == set(expected["meta_columns"])
        for key, value in expected["meta_columns"].items():
            _assert_columnar_value_equal(actual["meta_columns"][key], value)


def _assert_metadata_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        _assert_columnar_value_equal(actual[key], value)


def _assert_processed_loader_batch_contract(batch: dict[str, Any], *, batch_size: int) -> None:
    assert batch["batch_size"] == batch_size
    assert batch["seq_len"] == LOADER_SEQ_LEN
    assert isinstance(batch["sampled_gene_ids"], torch.Tensor)
    assert isinstance(batch["sampled_counts"], torch.Tensor)
    assert isinstance(batch["valid_mask"], torch.Tensor)
    assert isinstance(batch["exact_match_mask"], torch.Tensor)
    assert isinstance(batch["dataset_index"], torch.Tensor)
    assert isinstance(batch["global_row_index"], torch.Tensor)
    assert "row_offsets" not in batch
    assert "expressed_gene_indices" not in batch
    assert "expression_counts" not in batch


def _assert_processed_loader_batch_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, torch.Tensor):
            assert torch.equal(actual_value, expected_value), key
        elif isinstance(expected_value, dict):
            _assert_metadata_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def _expected_metadata(indices: list[int], specs: list[DummyDatasetSpec]) -> dict[str, np.ndarray]:
    dataset_starts = np.cumsum([0, *(spec.n_cells for spec in specs[:-1])], dtype=np.int64)
    expected_dataset_id: list[str] = []
    expected_dataset_index: list[int] = []
    expected_local_row_index: list[int] = []
    for index in indices:
        for dataset_index, (dataset_start, spec) in enumerate(zip(dataset_starts, specs)):
            dataset_end = int(dataset_start + spec.n_cells)
            if dataset_start <= index < dataset_end:
                expected_dataset_id.append(spec.dataset_id)
                expected_dataset_index.append(dataset_index)
                expected_local_row_index.append(int(index - dataset_start))
                break
        else:
            raise AssertionError(f"index {index} is out of range for selected datasets")
    return {
        "dataset_id": np.asarray(expected_dataset_id, dtype=object),
        "dataset_index": np.asarray(expected_dataset_index, dtype=np.int32),
        "local_row_index": np.asarray(expected_local_row_index, dtype=np.int64),
    }


def _build_inspection_config(specs: list[DummyDatasetSpec], output_root: Path) -> InspectionBatchConfig:
    return InspectionBatchConfig(
        output_root=str(output_root),
        datasets=tuple(
            InspectionTarget(
                dataset_id=spec.dataset_id,
                source_path=str(spec.source_path),
                source_release=spec.dataset_id,
            )
            for spec in specs
        ),
    )


def _run_inspection(specs: list[DummyDatasetSpec], inspection_root: Path) -> dict[str, Path]:
    manifest = run_batch(_build_inspection_config(specs, inspection_root), workers=1)
    return {
        record.dataset_id: Path(record.review_bundle)
        for record in manifest.records
    }


def _materialize_backend(
    backend: str,
    specs: list[DummyDatasetSpec],
    review_bundles: dict[str, Path],
    backend_root: Path,
) -> tuple[Path, dict[str, MaterializationManifest]]:
    corpus_root = backend_root / "corpus"
    manifests: dict[str, MaterializationManifest] = {}
    next_global_row_start = 0
    for dataset_index, spec in enumerate(specs):
        resolved = resolve_corpus_paths(
            topology="federated",
            corpus_root=corpus_root,
            dataset_id=spec.dataset_id,
        )
        materializer = Stage2Materializer(
            source_path=str(spec.source_path),
            review_bundle_path=str(review_bundles[spec.dataset_id]),
            output_roots=OutputRoots(
                metadata_root=str(resolved.meta_root),
                matrix_root=str(resolved.matrix_root),
            ),
            dataset_id=spec.dataset_id,
            backend=backend,
            topology="federated",
            corpus_index_path=str(corpus_root / "corpus-index.yaml"),
            corpus_id=f"phase03-{backend}-dummy-roundtrip",
            register=True,
            mode="create" if dataset_index == 0 else "append",
            dataset_index=dataset_index,
            global_row_start=next_global_row_start,
        )
        manifests[spec.dataset_id] = materializer.materialize()
        next_global_row_start += manifests[spec.dataset_id].cell_count
    return corpus_root, manifests


def _finalize_schema(
    spec: DummyDatasetSpec,
    *,
    dataset_index: int,
    meta_root: Path,
) -> Path:
    summary = DatasetSummaryDocument.from_yaml_file(meta_root / "dataset-summary.yaml")
    obs_columns = list(dict.fromkeys(["cell_id", "dataset_id", *[field.name for field in summary.obs_fields]]))
    var_columns = list(dict.fromkeys(["origin_index", "feature_id", *[field.name for field in summary.var_fields]]))
    schema = draft_canonicalization_schema(
        dataset_id=spec.dataset_id,
        obs_columns=obs_columns,
        var_columns=var_columns,
        hints={
            "dataset_index": dataset_index,
            "sampled_gene_ids": list(spec.gene_order_fingerprint),
            "assay": "Perturb-seq",
            "species": "human",
        },
    )
    final_schema = replace(
        schema,
        status="ready",
        description="Phase 3 dummy-data subset auto-finalized roundtrip schema",
    )
    schema_path = meta_root / "final-schema.yaml"
    final_schema.write_yaml(schema_path)
    return schema_path


def _canonicalize_backend(
    specs: list[DummyDatasetSpec],
    corpus_root: Path,
    manifests: dict[str, MaterializationManifest],
) -> None:
    for dataset_index, spec in enumerate(specs):
        resolved = resolve_corpus_paths(
            topology="federated",
            corpus_root=corpus_root,
            dataset_id=spec.dataset_id,
        )
        schema_path = _finalize_schema(spec, dataset_index=dataset_index, meta_root=resolved.meta_root)
        manifest = manifests[spec.dataset_id]
        result = run_canonicalization(
            dataset_id=spec.dataset_id,
            raw_obs_path=manifest.raw_cell_meta_path,
            raw_var_path=manifest.raw_feature_meta_path,
            size_factor_path=manifest.size_factor_parquet_path,
            schema_path=schema_path,
            output_root=resolved.canonical_meta_root,
        )
        assert result.obs_path.exists()
        assert result.var_path.exists()


def _case_indices(specs: list[DummyDatasetSpec]) -> dict[str, list[int]]:
    first_cells = specs[0].n_cells
    total_cells = sum(spec.n_cells for spec in specs)
    return {
        "first_last_per_dataset": [0, first_cells - 1, first_cells, total_cells - 1],
        "dataset_boundary_rows": [first_cells - 1, first_cells],
        "reversed_global_indices": [total_cells - 1, first_cells, first_cells - 1, 0],
        "mixed_cross_dataset_batch": [0, first_cells, 1, first_cells + 1, first_cells - 1, total_cells - 1],
        "batch_size_128": list(range(64)) + list(range(first_cells, first_cells + 64)),
    }


def _validate_corpus_index(corpus_root: Path, specs: list[DummyDatasetSpec]) -> dict[str, Any]:
    corpus_index = CorpusIndexDocument.from_yaml_file(corpus_root / "corpus-index.yaml")
    assert corpus_index.global_metadata["backend"] in {"lance", "arrow-ipc"}
    assert corpus_index.global_metadata["topology"] == "federated"
    assert [record.dataset_id for record in corpus_index.datasets] == list(SELECTED_DATASET_IDS)
    return {
        "corpus_id": corpus_index.corpus_id,
        "datasets": [
            {
                "dataset_id": record.dataset_id,
                "dataset_index": record.dataset_index,
                "cell_count": record.cell_count,
                "global_start": record.global_start,
                "global_end": record.global_end,
            }
            for record in corpus_index.datasets
        ],
        "expected_total_cells": sum(spec.n_cells for spec in specs),
    }


def _validate_public_api_case(
    *,
    name: str,
    indices: list[int],
    specs: list[DummyDatasetSpec],
    lance_corpus: Corpus,
    arrow_corpus: Corpus,
) -> dict[str, Any]:
    _assert_expression_batch_equal(
        arrow_corpus.read_expression(indices),
        lance_corpus.read_expression(indices),
    )
    _assert_raw_batch_equal(
        arrow_corpus.dataset().__getitems__(indices)[0],
        lance_corpus.dataset().__getitems__(indices)[0],
    )
    arrow_inspected = arrow_corpus.inspect_batch(indices, metadata_columns=LOADER_METADATA_COLUMNS)
    lance_inspected = lance_corpus.inspect_batch(indices, metadata_columns=LOADER_METADATA_COLUMNS)
    _assert_raw_batch_equal(arrow_inspected, lance_inspected)

    expected = _expected_metadata(indices, specs)
    np.testing.assert_array_equal(arrow_inspected["dataset_index"], expected["dataset_index"])
    np.testing.assert_array_equal(arrow_inspected["local_row_index"], expected["local_row_index"])

    arrow_metadata = arrow_corpus.take_metadata(indices, columns=METADATA_COLUMNS)
    lance_metadata = lance_corpus.take_metadata(indices, columns=METADATA_COLUMNS)
    _assert_metadata_equal(arrow_metadata, lance_metadata)
    np.testing.assert_array_equal(arrow_metadata["dataset_id"], expected["dataset_id"])
    np.testing.assert_array_equal(arrow_metadata["dataset_index"], expected["dataset_index"])
    np.testing.assert_array_equal(arrow_metadata["local_row_index"], expected["local_row_index"])

    return {
        "indices": indices,
        "batch_size": len(indices),
        "dataset_index": expected["dataset_index"].tolist(),
        "local_row_index": expected["local_row_index"].tolist(),
    }


def _validate_cpu_loader(lance_corpus: Corpus, arrow_corpus: Corpus) -> dict[str, Any]:
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    lance_batch = next(
        lance_corpus.loader(
            processing="cpu",
            seq_len=LOADER_SEQ_LEN,
            batch_size=128,
            seed=11,
            num_workers=0,
            metadata_columns=LOADER_METADATA_COLUMNS,
        )
    )
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    arrow_batch = next(
        arrow_corpus.loader(
            processing="cpu",
            seq_len=LOADER_SEQ_LEN,
            batch_size=128,
            seed=11,
            num_workers=0,
            metadata_columns=LOADER_METADATA_COLUMNS,
        )
    )
    _assert_processed_loader_batch_contract(arrow_batch, batch_size=128)
    _assert_processed_loader_batch_equal(arrow_batch, lance_batch)
    return {
        "batch_size": int(arrow_batch["batch_size"]),
        "seq_len": int(arrow_batch["seq_len"]),
        "sampled_global_row_index_preview": arrow_batch["global_row_index"][:8].tolist(),
    }


def test_dummy_subset_roundtrip_federated_lance_vs_arrow_ipc(tmp_path: Path) -> None:
    specs = _load_selected_dummy_specs()
    missing_sources = [str(spec.source_path) for spec in specs if not spec.source_path.exists()]
    if missing_sources:
        pytest.skip(f"dummy h5ad sources are missing: {missing_sources}")

    artifact_root = _artifact_root(tmp_path)
    report_path = artifact_root / "roundtrip-report.json"
    report: dict[str, Any] = {
        "status": "running",
        "selected_dataset_ids": list(SELECTED_DATASET_IDS),
        "selected_datasets": [
            {
                "dataset_id": spec.dataset_id,
                "source_path": str(spec.source_path),
                "n_cells": spec.n_cells,
                "n_genes": spec.n_genes,
            }
            for spec in specs
        ],
        "inspection": {},
        "backends": {},
        "comparison": {"cases": {}},
    }
    _write_report(report_path, report)

    print(f"[phase3] artifact root: {artifact_root}")
    print(f"[phase3] selected datasets: {', '.join(spec.dataset_id for spec in specs)}")

    try:
        inspection_root = artifact_root / "inspection"
        review_bundles = _run_inspection(specs, inspection_root)
        report["inspection"] = {
            "output_root": str(inspection_root),
            "review_bundles": {dataset_id: str(path) for dataset_id, path in review_bundles.items()},
        }
        _write_report(report_path, report)
        print("[phase3] inspection complete")

        corpora: dict[str, Corpus] = {}
        for backend in ("lance", "arrow-ipc"):
            backend_root = artifact_root / backend.replace("-", "_")
            corpus_root, manifests = _materialize_backend(backend, specs, review_bundles, backend_root)
            _canonicalize_backend(specs, corpus_root, manifests)
            corpora[backend] = load_corpus(str(corpus_root))
            report["backends"][backend] = {
                "corpus_root": str(corpus_root),
                "corpus_index": _validate_corpus_index(corpus_root, specs),
                "datasets": {
                    dataset_id: {
                        "manifest": str(resolve_corpus_paths("federated", corpus_root, dataset_id).meta_root / "materialization-manifest.yaml"),
                        "canonical_obs": str(resolve_corpus_paths("federated", corpus_root, dataset_id).canonical_meta_root / "canonical-obs.parquet"),
                        "canonical_var": str(resolve_corpus_paths("federated", corpus_root, dataset_id).canonical_meta_root / "canonical-var.parquet"),
                        "matrix_root": str(resolve_corpus_paths("federated", corpus_root, dataset_id).matrix_root),
                        "materialization": asdict(manifests[dataset_id].corpus_registration) if manifests[dataset_id].corpus_registration else None,
                    }
                    for dataset_id in SELECTED_DATASET_IDS
                },
            }
            _write_report(report_path, report)
            print(f"[phase3] {backend} materialization + canonicalization complete")

        lance_corpus = corpora["lance"]
        arrow_corpus = corpora["arrow-ipc"]
        for case_name, indices in _case_indices(specs).items():
            report["comparison"]["cases"][case_name] = _validate_public_api_case(
                name=case_name,
                indices=indices,
                specs=specs,
                lance_corpus=lance_corpus,
                arrow_corpus=arrow_corpus,
            )
            _write_report(report_path, report)
            print(f"[phase3] comparison case passed: {case_name}")

        report["comparison"]["cpu_loader"] = _validate_cpu_loader(lance_corpus, arrow_corpus)
        report["status"] = "success"
        _write_report(report_path, report)
        print("[phase3] cpu loader comparison complete")
        print(f"[phase3] report: {report_path}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_report(report_path, report)
        raise
