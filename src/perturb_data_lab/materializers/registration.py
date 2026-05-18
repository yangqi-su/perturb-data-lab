"""Register materialized dataset manifests in ``corpus-index.yaml``."""

from __future__ import annotations

from pathlib import Path

from .corpus_index import update_corpus_index
from .models import DatasetJoinRecord, MaterializationManifest


def register_materialization(
    manifest: MaterializationManifest,
    manifest_path: Path,
    corpus_index_path: Path,
    *,
    corpus_id: str | None = None,
    backend: str,
    topology: str,
) -> DatasetJoinRecord:
    """Append a materialized dataset entry to ``corpus-index.yaml``."""
    corpus_root = corpus_index_path.parent
    manifest_path_relative = manifest_path.resolve().relative_to(corpus_root.resolve())
    join_record = DatasetJoinRecord(
        dataset_id=manifest.dataset_id,
        join_mode=manifest.route,
        manifest_path=str(manifest_path_relative),
        cell_count=manifest.cell_count,
    )
    updated_corpus = update_corpus_index(
        corpus_index_path=corpus_index_path,
        new_dataset_record=join_record,
        corpus_id=corpus_id,
        backend=backend,
        topology=topology,
    )
    return updated_corpus.datasets[-1]
