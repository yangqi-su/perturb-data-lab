"""Corpus index update helper for materialized datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL
from .models import (
    CorpusIndexDocument,
    DatasetJoinRecord,
    GlobalMetadataDocument,
)


def update_corpus_index(
    corpus_index_path: Path,
    new_dataset_record: DatasetJoinRecord,
    global_metadata: GlobalMetadataDocument | None = None,
    backend: str | None = None,
    topology: str | None = None,
) -> CorpusIndexDocument:
    """Append one dataset to ``corpus-index.yaml`` and write it back."""
    corpus_root = corpus_index_path.parent.resolve()

    manifest_path_input = Path(new_dataset_record.manifest_path)
    if manifest_path_input.is_absolute():
        manifest_path_relative = manifest_path_input.resolve().relative_to(corpus_root)
    else:
        manifest_path_relative = manifest_path_input

    if corpus_index_path.exists():
        corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)
        existing_ids = {d.dataset_id for d in corpus.datasets}
        if new_dataset_record.dataset_id in existing_ids:
            raise ValueError(
                f"dataset {new_dataset_record.dataset_id} already exists in corpus index; "
                "use a different dataset_id to avoid duplication."
            )
        total_existing_cells = sum(d.cell_count for d in corpus.datasets)
        new_record = DatasetJoinRecord(
            dataset_id=new_dataset_record.dataset_id,
            dataset_index=len(corpus.datasets),
            join_mode=new_dataset_record.join_mode,
            manifest_path=str(manifest_path_relative),
            cell_count=new_dataset_record.cell_count,
            global_start=total_existing_cells,
            global_end=total_existing_cells + new_dataset_record.cell_count,
        )
        datasets = list(corpus.datasets) + [new_record]
        corpus_id = corpus.corpus_id
        global_meta = corpus.global_metadata
    else:
        new_record = DatasetJoinRecord(
            dataset_id=new_dataset_record.dataset_id,
            dataset_index=0,
            join_mode=new_dataset_record.join_mode,
            manifest_path=str(manifest_path_relative),
            cell_count=new_dataset_record.cell_count,
            global_start=0,
            global_end=new_dataset_record.cell_count,
        )
        datasets = [new_record]
        corpus_id = "perturb-data-lab-v0"
        if global_metadata is None:
            if backend is None:
                raise ValueError("backend is required when creating a corpus index")
            if topology is None:
                raise ValueError("topology is required when creating a corpus index")
            assert backend in {"lance", "zarr"}, f"unknown backend: {backend}"
            assert topology in {"federated", "aggregate"}, f"unknown topology: {topology}"
        gmeta_dict: dict[str, Any] = {
            "kind": "global-metadata",
            "contract_version": CONTRACT_VERSION,
            "schema_version": CONTRACT_VERSION,
            "missing_value_literal": MISSING_VALUE_LITERAL,
            "raw_field_policy": "preserve-unchanged",
            "backend": backend,
            "topology": topology,
        }
        if global_metadata is not None:
            gmeta_dict = global_metadata.to_dict()
        global_meta = gmeta_dict
        if gmeta_dict:
            global_meta_path = corpus_index_path.parent / "global-metadata.yaml"
            GlobalMetadataDocument.from_dict(gmeta_dict).write_yaml(global_meta_path)

    updated = CorpusIndexDocument(
        kind="corpus-index",
        contract_version=CONTRACT_VERSION,
        corpus_id=corpus_id,
        global_metadata=global_meta,
        datasets=tuple(datasets),
    )
    updated.write_yaml(corpus_index_path)
    return updated
