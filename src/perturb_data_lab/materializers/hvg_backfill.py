"""Backfill canonical HVG ranking tables for existing Lance corpora."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pyarrow.parquet as pq

from ..loaders.expression import DatasetEntry, LanceDatasetEntry, build_expression_reader
from ..pp.hvg import build_ranked_hvg_frame
from .models import CorpusIndexDocument, MaterializationManifest
from .paths import resolve_corpus_paths

DEFAULT_BACKFILL_CHUNK_ROWS = 50_000


@dataclass(frozen=True)
class HVGBackfillDatasetResult:
    dataset_id: str
    dataset_index: int
    manifest_path: str
    feature_meta_path: str
    output_path: str
    cell_count: int
    feature_count: int
    row_count: int
    default_n_hvg: int
    chunk_rows: int
    sha256: str
    manifest_updated: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_index": self.dataset_index,
            "manifest_path": self.manifest_path,
            "feature_meta_path": self.feature_meta_path,
            "output_path": self.output_path,
            "cell_count": self.cell_count,
            "feature_count": self.feature_count,
            "row_count": self.row_count,
            "default_n_hvg": self.default_n_hvg,
            "chunk_rows": self.chunk_rows,
            "sha256": self.sha256,
            "manifest_updated": self.manifest_updated,
        }


@dataclass(frozen=True)
class HVGBackfillSummary:
    corpus_root: str
    backend: str
    topology: str
    output_root: str
    dataset_count: int
    chunk_rows: int
    update_manifests: bool
    datasets: tuple[HVGBackfillDatasetResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_root": self.corpus_root,
            "backend": self.backend,
            "topology": self.topology,
            "output_root": self.output_root,
            "dataset_count": self.dataset_count,
            "chunk_rows": self.chunk_rows,
            "update_manifests": self.update_manifests,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
        }

    def write_json(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _resolve_manifest_path(corpus_root: Path, manifest_path: str) -> Path:
    path = Path(manifest_path)
    if not path.is_absolute():
        path = corpus_root / path
    return path


def _infer_backend_topology(
    corpus_root: Path,
    corpus: CorpusIndexDocument,
) -> tuple[str, str]:
    backend = corpus.global_metadata.get("backend")
    topology = corpus.global_metadata.get("topology")

    if (backend is None or topology is None) and corpus.datasets:
        first_manifest_path = _resolve_manifest_path(
            corpus_root,
            corpus.datasets[0].manifest_path,
        )
        if first_manifest_path.exists():
            manifest = MaterializationManifest.from_yaml_file(first_manifest_path)
            backend = backend or manifest.backend
            topology = topology or manifest.topology

    if backend is None:
        raise ValueError(
            f"cannot infer backend from {corpus_root / 'corpus-index.yaml'}"
        )
    if topology is None:
        topology = "aggregate" if backend == "lance" else "federated"
    return str(backend), str(topology)


def _build_lance_reader(
    corpus_root: Path,
    *,
    corpus: CorpusIndexDocument,
    topology: str,
):
    if topology == "aggregate":
        lance_path = corpus_root / "matrix" / "aggregated-cells.lance"
        if not lance_path.exists():
            raise FileNotFoundError(f"aggregate Lance file not found: {lance_path}")
        entries = [
            DatasetEntry(
                dataset_id=dataset.dataset_id,
                global_start=dataset.global_start,
                global_end=dataset.global_end,
            )
            for dataset in corpus.datasets
        ]
        return build_expression_reader(
            "lance",
            topology,
            entries,
            lance_path=str(lance_path),
        )

    if topology == "federated":
        entries = []
        for dataset in corpus.datasets:
            matrix_root = resolve_corpus_paths(
                topology,
                corpus_root,
                dataset.dataset_id,
            ).matrix_root
            lance_path = matrix_root / f"{dataset.dataset_id}.lance"
            if not lance_path.exists():
                raise FileNotFoundError(
                    f"federated Lance file not found for {dataset.dataset_id}: {lance_path}"
                )
            entries.append(
                LanceDatasetEntry(
                    dataset_id=dataset.dataset_id,
                    global_start=dataset.global_start,
                    global_end=dataset.global_end,
                    lance_path=str(lance_path),
                )
            )
        return build_expression_reader("lance", topology, entries)

    raise ValueError(
        f"unsupported topology '{topology}'; expected 'aggregate' or 'federated'"
    )


def _first_existing_path(paths: Sequence[Path]) -> Path | None:
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            return path
    return None


def _resolve_feature_meta_path(
    *,
    corpus_root: Path,
    topology: str,
    dataset_id: str,
    manifest: MaterializationManifest,
) -> Path:
    dataset_paths = resolve_corpus_paths(topology, corpus_root, dataset_id)
    candidates: list[Path] = []
    if manifest.raw_feature_meta_path:
        manifest_path = Path(manifest.raw_feature_meta_path)
        candidates.append(
            manifest_path if manifest_path.is_absolute() else corpus_root / manifest_path
        )
        candidates.append(dataset_paths.meta_root / manifest_path.name)
    candidates.append(dataset_paths.meta_root / "raw-var.parquet")
    candidates.append(dataset_paths.canonical_meta_root / "canonical-var.parquet")

    resolved = _first_existing_path(candidates)
    if resolved is None:
        rendered = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"feature metadata not found for {dataset_id}; tried: {rendered}"
        )
    return resolved


def _read_feature_metadata(feature_meta_path: Path) -> tuple[tuple[str, ...], np.ndarray | None]:
    schema = pq.read_schema(feature_meta_path)
    available = set(schema.names)
    gene_id_col = "feature_id" if "feature_id" in available else "canonical_gene_id"
    if gene_id_col not in available:
        raise ValueError(
            f"feature_id/canonical_gene_id column missing in {feature_meta_path}"
        )

    columns = [gene_id_col]
    has_global_id = "global_id" in available
    if has_global_id:
        columns.append("global_id")
    table = pq.read_table(feature_meta_path, columns=columns)
    gene_ids = tuple(str(value) for value in table.column(gene_id_col).to_pylist())
    global_feature_ids = None
    if has_global_id:
        global_feature_ids = np.asarray(table.column("global_id").to_pylist(), dtype=np.int64)
    return gene_ids, global_feature_ids


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def backfill_hvg_rankings_for_corpus(
    corpus_root: str | Path,
    *,
    dataset_ids: Sequence[str] | None = None,
    output_root: str | Path | None = None,
    chunk_rows: int = DEFAULT_BACKFILL_CHUNK_ROWS,
    n_hvg: int = 2000,
    overwrite: bool = False,
    update_manifests: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    progress_every_chunks: int = 10,
) -> HVGBackfillSummary:
    """Recompute per-dataset ``hvg.parquet`` files from an existing Lance corpus."""
    corpus_root = Path(corpus_root).resolve()
    output_root_path = Path(output_root).resolve() if output_root else corpus_root
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if n_hvg <= 0:
        raise ValueError("n_hvg must be positive")

    corpus_index_path = corpus_root / "corpus-index.yaml"
    if not corpus_index_path.exists():
        raise FileNotFoundError(f"corpus-index.yaml not found at {corpus_index_path}")

    corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)
    backend, topology = _infer_backend_topology(corpus_root, corpus)
    if backend != "lance":
        raise ValueError(
            f"backfill_hvg_rankings_for_corpus only supports Lance corpora; got '{backend}'"
        )

    selected_ids = set(dataset_ids) if dataset_ids is not None else None
    if selected_ids is not None:
        known_ids = {dataset.dataset_id for dataset in corpus.datasets}
        unknown = sorted(selected_ids - known_ids)
        if unknown:
            raise ValueError(f"unknown dataset_id(s): {', '.join(unknown)}")

    expression_reader = _build_lance_reader(
        corpus_root,
        corpus=corpus,
        topology=topology,
    )
    results: list[HVGBackfillDatasetResult] = []

    for dataset in corpus.datasets:
        if selected_ids is not None and dataset.dataset_id not in selected_ids:
            continue

        manifest_path = _resolve_manifest_path(corpus_root, dataset.manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"materialization manifest not found for {dataset.dataset_id}: {manifest_path}"
            )
        manifest = MaterializationManifest.from_yaml_file(manifest_path)
        feature_meta_path = _resolve_feature_meta_path(
            corpus_root=corpus_root,
            topology=topology,
            dataset_id=dataset.dataset_id,
            manifest=manifest,
        )
        feature_ids, global_feature_ids = _read_feature_metadata(feature_meta_path)
        feature_count = len(feature_ids)
        if feature_count == 0:
            raise ValueError(f"no feature_id rows found for {dataset.dataset_id}")

        effective_n_hvg = manifest.default_n_hvg or int(n_hvg)
        dataset_output_root = resolve_corpus_paths(
            topology,
            output_root_path,
            dataset.dataset_id,
        ).meta_root
        output_path = dataset_output_root / "hvg.parquet"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists for {dataset.dataset_id}: {output_path}"
            )

        sum_log1p = np.zeros(feature_count, dtype=np.float64)
        sum_log1p_sq = np.zeros(feature_count, dtype=np.float64)
        n_cells_detected = np.zeros(feature_count, dtype=np.int64)
        processed_cells = 0
        chunk_counter = 0

        if progress_callback is not None:
            progress_callback(
                f"[backfill-hvg] {dataset.dataset_id}: start cells={dataset.cell_count} features={feature_count}"
            )

        for global_start in range(dataset.global_start, dataset.global_end, chunk_rows):
            global_end = min(global_start + chunk_rows, dataset.global_end)
            batch = expression_reader.read_expression_flat(range(global_start, global_end))
            if batch.expressed_gene_indices.size:
                max_index = int(batch.expressed_gene_indices.max())
                if max_index >= feature_count:
                    raise ValueError(
                        f"dataset {dataset.dataset_id} contains local gene index {max_index} "
                        f"outside feature_count={feature_count}"
                    )
                log1p_counts = np.log1p(batch.expression_counts.astype(np.float64, copy=False))
                np.add.at(sum_log1p, batch.expressed_gene_indices, log1p_counts)
                np.add.at(sum_log1p_sq, batch.expressed_gene_indices, log1p_counts ** 2)
                detected_mask = batch.expression_counts > 0
                if np.any(detected_mask):
                    np.add.at(
                        n_cells_detected,
                        batch.expressed_gene_indices[detected_mask],
                        1,
                    )
            processed_cells += batch.batch_size
            chunk_counter += 1

            if (
                progress_callback is not None
                and progress_every_chunks > 0
                and (
                    chunk_counter % progress_every_chunks == 0
                    or global_end == dataset.global_end
                )
            ):
                progress_callback(
                    f"[backfill-hvg] {dataset.dataset_id}: processed {processed_cells}/{dataset.cell_count} cells"
                )

        if processed_cells != dataset.cell_count:
            raise ValueError(
                f"dataset {dataset.dataset_id} processed {processed_cells} cells; "
                f"expected {dataset.cell_count}"
            )

        frame = build_ranked_hvg_frame(
            dataset_id=dataset.dataset_id,
            dataset_index=dataset.dataset_index,
            gene_ids=feature_ids,
            global_feature_ids=global_feature_ids,
            sum_log1p=sum_log1p,
            sum_log1p_sq=sum_log1p_sq,
            n_cells_total=processed_cells,
            n_cells_detected=n_cells_detected,
            n_hvg=effective_n_hvg,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(output_path)

        manifest_updated = False
        if update_manifests:
            updated_manifest = replace(
                manifest,
                hvg_ranking_path=str(output_path),
                default_n_hvg=effective_n_hvg,
            )
            updated_manifest.write_yaml(manifest_path)
            manifest_updated = True

        result = HVGBackfillDatasetResult(
            dataset_id=dataset.dataset_id,
            dataset_index=dataset.dataset_index,
            manifest_path=str(manifest_path),
            feature_meta_path=str(feature_meta_path),
            output_path=str(output_path),
            cell_count=dataset.cell_count,
            feature_count=feature_count,
            row_count=frame.height,
            default_n_hvg=effective_n_hvg,
            chunk_rows=chunk_rows,
            sha256=_sha256_file(output_path),
            manifest_updated=manifest_updated,
        )
        results.append(result)

        if progress_callback is not None:
            progress_callback(
                f"[backfill-hvg] {dataset.dataset_id}: wrote {output_path} rows={frame.height} sha256={result.sha256[:12]}"
            )

    return HVGBackfillSummary(
        corpus_root=str(corpus_root),
        backend=backend,
        topology=topology,
        output_root=str(output_root_path),
        dataset_count=len(results),
        chunk_rows=chunk_rows,
        update_manifests=update_manifests,
        datasets=tuple(results),
    )
