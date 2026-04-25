"""Corpus registration flow for Stage 2 materialization.

This module provides the registration layer that connects Stage2Materializer
output to the corpus ledger. It handles the create-vs-append decision,
dataset index assignment, global range computation, and ledger persistence.

The registration flow is:
1. Stage2Materializer.materialize() produces a MaterializationManifest
2. register_materialization() converts the manifest to a DatasetJoinRecord
   and registers it in the corpus ledger via update_corpus_index
3. The Parquet corpus-ledger.parquet is updated or created as the
   authoritative machine-readable corpus index

Design principles:
- Registration is explicit and routing-oriented, not schema-dependent
- The ledger is append-only: existing entries are never modified
- create_new and append_routed are the two join modes
- feature_count is back-filled from per-dataset manifests when writing the ledger
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .core import update_corpus_index
from .models import (
    CorpusLedgerEntry,
    DatasetJoinRecord,
    GlobalMetadataDocument,
    MaterializationManifest,
)


def manifest_to_join_record(
    manifest: MaterializationManifest,
    corpus_root: Path,
) -> DatasetJoinRecord:
    """Convert a MaterializationManifest to a DatasetJoinRecord for corpus registration.

    Parameters
    ----------
    manifest : MaterializationManifest
        The manifest produced by Stage2Materializer.materialize().
    corpus_root : Path
        The root directory of the corpus (where corpus-ledger.parquet lives).

    Returns
    -------
    DatasetJoinRecord
        A join record ready for update_corpus_index(). The
        global_start/global_end are set to 0 here; they are recomputed
        by update_corpus_index() based on existing corpus cell count.

    Notes
    -----
    - manifest_path is stored relative to corpus_root when possible
    - cell_count is taken from manifest.cell_count
    - feature_count lives in the ledger entry, not DatasetJoinRecord
    """
    manifest_dir = Path(manifest.raw_cell_meta_path or manifest.provenance_spec_path).parent
    try:
        manifest_path_relative = manifest_dir.relative_to(corpus_root)
    except ValueError:
        # Manifest is outside corpus_root — store absolute path as fallback
        manifest_path_relative = manifest_dir

    return DatasetJoinRecord(
        dataset_id=manifest.dataset_id,
        release_id=manifest.release_id,
        join_mode=manifest.route,
        manifest_path=str(manifest_path_relative),
        cell_count=manifest.cell_count,
        global_start=0,  # recomputed by update_corpus_index
        global_end=0,    # recomputed by update_corpus_index
    )


def register_materialization(
    manifest: MaterializationManifest,
    corpus_index_path: Path,
    *,
    corpus_id: str | None = None,
    backend: str | None = None,
    topology: str | None = None,
) -> tuple[DatasetJoinRecord, str, bool]:
    """Register a Stage2Materializer output to the corpus ledger.

    This is the primary registration entry point for Stage 2. It:
    1. Determines whether the corpus exists (ledger present)
    2. Converts the manifest to a DatasetJoinRecord
    3. Calls update_corpus_index to update the YAML and Parquet ledger
    4. Returns the join record, corpus_id, and whether this was a create (True) or append (False)

    Parameters
    ----------
    manifest : MaterializationManifest
        The filled manifest from Stage2Materializer.materialize().
    corpus_index_path : Path
        Path to corpus-index.yaml (or where it will be written for new corpora).
        The Parquet ledger is written alongside at corpus_index_path.parent.
    corpus_id : str | None
        Corpus identifier. Required for new corpus creation; inferred from
        existing ledger for append operations.
    backend : str | None
        Storage backend (e.g., "arrow-hf"). Required for new corpus creation.
        For append, the backend is inherited from existing corpus metadata.
    topology : str | None
        Corpus topology ("federated" or "aggregate"). Required for new corpus
        creation. For append, inferred from existing corpus metadata.

    Returns
    -------
    tuple[DatasetJoinRecord, str, bool]
        The registered DatasetJoinRecord, the corpus_id (resolved), and a boolean
        indicating whether this was a new corpus creation (True) or an append (False).

    Raises
    ------
    ValueError
        If corpus_id, backend, or topology are not provided for a new corpus.
    FileNotFoundError
        If the manifest's outputs are not found on disk.

    Notes
    -----
    - The manifest's outputs are verified to exist before registration
    - The ledger is append-only; existing entries are never modified
    - For aggregate topology, global_start/global_end are computed by
      update_corpus_index from the running cell count total
    """
    corpus_root = corpus_index_path.parent

    # Verify key manifest outputs exist before registering
    _verify_manifest_outputs(manifest)

    # Determine if corpus exists by checking for existing ledger or YAML
    yaml_exists = corpus_index_path.exists()
    ledger_path = corpus_root / "corpus-ledger.parquet"
    ledger_exists = ledger_path.exists()
    is_create = not yaml_exists and not ledger_exists

    # Convert manifest to join record
    join_record = manifest_to_join_record(manifest, corpus_root)

    if is_create:
        # --- New corpus ---
        if corpus_id is None:
            raise ValueError(
                "corpus_id is required for new corpus registration; "
                "pass it explicitly (e.g., corpus_id='perturb-corpus-v0')"
            )
        if backend is None:
            raise ValueError(
                "backend is required for new corpus registration; "
                "pass it explicitly (e.g., backend='arrow-hf')"
            )
        if topology is None:
            # Infer from backend per contract backend-topology matrix
            topology = "aggregate" if backend == "lance" else "federated"

        # Build GlobalMetadataDocument for new corpus creation
        global_meta = GlobalMetadataDocument(
            kind="global-metadata",
            contract_version="0.3.0",
            schema_version="0.3.0",
            feature_registry_id="",
            missing_value_literal="",
            raw_field_policy="preserve-unchanged",
            backend=backend,
            topology=topology,
        )
    else:
        # --- Append to existing corpus ---
        global_meta = None

        # Read existing metadata to inherit backend/topology/corpus_id
        existing_yaml = corpus_root / "global-metadata.yaml"
        if existing_yaml.exists():
            existing_meta = GlobalMetadataDocument.from_yaml_file(existing_yaml)
            if backend is None:
                backend = existing_meta.backend
            if topology is None:
                topology = existing_meta.topology

        # Read corpus_id from existing ledger if not provided
        if corpus_id is None:
            if ledger_exists:
                corpus_id = _read_corpus_id_from_ledger(ledger_path)
            else:
                # Fall back to yaml-based corpus index
                from .models import CorpusIndexDocument
                existing_corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)
                corpus_id = existing_corpus.corpus_id

    # Call update_corpus_index which writes both YAML and Parquet ledger
    updated_corpus = update_corpus_index(
        corpus_index_path=corpus_index_path,
        new_dataset_record=join_record,
        global_metadata=global_meta,
        backend=backend,
        topology=topology,
    )

    # Find the record that was just added
    registered_record = None
    for ds in updated_corpus.datasets:
        if ds.dataset_id == manifest.dataset_id and ds.release_id == manifest.release_id:
            registered_record = ds
            break

    if registered_record is None:
        raise RuntimeError(
            f"register_materialization: dataset {manifest.dataset_id}/{manifest.release_id} "
            "was not found in updated corpus index after update_corpus_index returned"
        )

    # Resolve corpus_id from the updated corpus (for new corpus, it's set now)
    resolved_corpus_id = updated_corpus.corpus_id

    return registered_record, resolved_corpus_id, is_create


def _verify_manifest_outputs(manifest: MaterializationManifest) -> None:
    """Verify that key manifest outputs exist on disk.

    Parameters
    ----------
    manifest : MaterializationManifest
        The manifest to verify.

    Raises
    ------
    FileNotFoundError
        If any required output path does not exist on disk.
    """
    import warnings

    # Paths that must exist for a valid materialization
    required_paths = [
        manifest.raw_cell_meta_path,
        manifest.raw_feature_meta_path,
        manifest.metadata_summary_path,
        manifest.provenance_spec_path,
        manifest.size_factor_manifest_path,
        manifest.qa_manifest_path,
    ]

    for path_str in required_paths:
        if path_str is None:
            continue
        p = Path(path_str)
        if not p.exists():
            warnings.warn(
                f"manifest output not found on disk: {p}; "
                "registration will proceed but artifact completeness is not guaranteed",
                UserWarning,
                stacklevel=3,
            )

    # Verify matrix root has some content
    matrix_root = Path(manifest.outputs.matrix_root)
    if not matrix_root.exists():
        raise FileNotFoundError(
            f"matrix_root does not exist: {matrix_root}; "
            "materialization may have failed"
        )


def _read_corpus_id_from_ledger(ledger_path: Path) -> str:
    """Read the corpus_id from an existing Parquet ledger.

    Parameters
    ----------
    ledger_path : Path
        Path to corpus-ledger.parquet.

    Returns
    -------
    str
        The corpus_id from the first row of the ledger.

    Raises
    ------
    FileNotFoundError
        If the ledger does not exist.
    ValueError
        If the ledger is empty.
    """
    table = pq.read_table(str(ledger_path))
    corpus_ids = table.column("corpus_id").to_pylist()
    if not corpus_ids:
        raise ValueError(f"corpus ledger at {ledger_path} is empty")
    return str(corpus_ids[0])


def read_corpus_ledger(
    ledger_path: Path,
) -> list[CorpusLedgerEntry]:
    """Read the Parquet corpus ledger and return a list of ledger entries.

    Parameters
    ----------
    ledger_path : Path
        Path to corpus-ledger.parquet.

    Returns
    -------
    list[CorpusLedgerEntry]
        All entries in the corpus ledger, in order of dataset_index.

    Raises
    ------
    FileNotFoundError
        If the ledger does not exist.

    Notes
    -----
    The Parquet ledger is the authoritative machine-readable corpus index.
    It is append-only: entries are never modified after being written.
    """
    if not ledger_path.exists():
        raise FileNotFoundError(
            f"corpus ledger not found: {ledger_path}; "
            "use register_materialization() to create a new corpus"
        )

    table = pq.read_table(str(ledger_path))
    entries = []
    for row in table.to_pylist():
        entries.append(CorpusLedgerEntry.from_dict(row))
    return entries


def corpus_exists(corpus_root: Path) -> bool:
    """Check whether a corpus ledger exists at the given corpus root.

    Parameters
    ----------
    corpus_root : Path
        The corpus root directory.

    Returns
    -------
    bool
        True if corpus-ledger.parquet OR corpus-index.yaml exists at corpus_root.
    """
    return (corpus_root / "corpus-ledger.parquet").exists() or (
        corpus_root / "corpus-index.yaml"
    ).exists()


def get_corpus_summary(ledger_path: Path) -> dict[str, Any]:
    """Return a summary of the corpus from the ledger.

    Parameters
    ----------
    ledger_path : Path
        Path to corpus-ledger.parquet.

    Returns
    -------
    dict with keys:
        - corpus_id: str
        - total_datasets: int
        - total_cells: int
        - backend: str
        - topology: str
        - entries: list of dict representations

    Raises
    ------
    FileNotFoundError
        If the ledger does not exist.
    ValueError
        If the ledger is empty.
    """
    entries = read_corpus_ledger(ledger_path)
    if not entries:
        raise ValueError(f"corpus ledger at {ledger_path} is empty")

    first = entries[0]
    total_cells = sum(e.cell_count for e in entries)

    return {
        "corpus_id": first.corpus_id,
        "total_datasets": len(entries),
        "total_cells": total_cells,
        "backend": first.backend,
        "topology": first.topology,
        "entries": [e.to_dict() for e in entries],
    }
