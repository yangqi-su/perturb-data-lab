"""Corpus registration for materialized datasets.

This module connects a per-dataset materialization manifest to the corpus-level
``corpus-index.yaml`` used by downstream loaders. It handles create-vs-append,
dataset index assignment, and global row range computation.
"""

from __future__ import annotations

from pathlib import Path

from .corpus_index import update_corpus_index
from .models import (
    CorpusIndexDocument,
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
        The manifest produced by DatasetMaterializer.materialize().
    corpus_root : Path
        The root directory of the corpus.

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
    """
    corpus_root = corpus_root.resolve()
    manifest_artifact = manifest.raw_cell_meta_path or manifest.provenance_spec_path
    if manifest_artifact is None:
        raise ValueError("manifest must contain raw_cell_meta_path or provenance_spec_path")
    manifest_dir = _resolve_manifest_artifact(corpus_root, manifest_artifact).parent
    manifest_yaml = manifest_dir / "materialization-manifest.yaml"
    manifest_path_relative = manifest_yaml.relative_to(corpus_root)

    return DatasetJoinRecord(
        dataset_id=manifest.dataset_id,
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
    """Register a materialized dataset in the corpus index.

    This function:
    1. Determines whether the corpus index exists
    2. Converts the manifest to a DatasetJoinRecord
    3. Calls update_corpus_index to update corpus-index.yaml
    4. Returns the join record, corpus_id, and whether this was a create (True) or append (False)

    Parameters
    ----------
    manifest : MaterializationManifest
        The filled manifest from DatasetMaterializer.materialize().
    corpus_index_path : Path
        Path to corpus-index.yaml (or where it will be written for new corpora).
    corpus_id : str | None
        Corpus identifier. Required for new corpus creation.
    backend : str | None
        Storage backend ("lance" or "zarr"). Required for registration and
        checked against existing corpus metadata on append.
    topology : str | None
        Corpus topology ("federated" or "aggregate"). Required for new corpus
        creation and checked against existing corpus metadata on append.

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
    - For aggregate topology, global_start/global_end are computed by
      update_corpus_index from the running cell count total
    """
    corpus_root = corpus_index_path.parent

    # Verify current required artifacts exist before registering.
    _verify_manifest_outputs(manifest, corpus_root)

    # Determine if corpus exists by checking the corpus index.
    yaml_exists = corpus_index_path.exists()
    is_create = not yaml_exists

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
                "pass it explicitly (e.g., backend='lance')"
            )
        if topology is None:
            raise ValueError(
                "topology is required for new corpus registration; "
                "pass it explicitly (e.g., topology='federated')"
            )
        assert backend in {"lance", "zarr"}, f"unknown backend: {backend}"
        assert topology in {"federated", "aggregate"}, f"unknown topology: {topology}"

        # Build GlobalMetadataDocument for new corpus creation
        global_meta = GlobalMetadataDocument(
            kind="global-metadata",
            contract_version="0.3.0",
            schema_version="0.3.0",
            missing_value_literal="",
            raw_field_policy="preserve-unchanged",
            backend=backend,
            topology=topology,
        )
    else:
        # --- Append to existing corpus ---
        global_meta = None
        existing_corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)
        existing_backend = existing_corpus.global_metadata["backend"]
        existing_topology = existing_corpus.global_metadata["topology"]
        assert backend == existing_backend, (
            f"backend mismatch: selected {backend}, corpus has {existing_backend}"
        )
        assert topology == existing_topology, (
            f"topology mismatch: selected {topology}, corpus has {existing_topology}"
        )

        # Read corpus_id from the existing corpus index if not provided.
        if corpus_id is None:
            corpus_id = existing_corpus.corpus_id

    # Call update_corpus_index which writes corpus-index.yaml.
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
        if ds.dataset_id == manifest.dataset_id:
            registered_record = ds
            break

    if registered_record is None:
        raise RuntimeError(
            f"register_materialization: dataset {manifest.dataset_id} "
            "was not found in updated corpus index after update_corpus_index returned"
        )

    # Resolve corpus_id from the updated corpus (for new corpus, it's set now)
    resolved_corpus_id = updated_corpus.corpus_id

    return registered_record, resolved_corpus_id, is_create


def _resolve_manifest_artifact(corpus_root: Path, path_str: str) -> Path:
    corpus_root = corpus_root.resolve()
    path = Path(path_str)
    if not path.is_absolute():
        path = corpus_root / path
    path = path.resolve()
    path.relative_to(corpus_root)
    return path


def _verify_manifest_outputs(manifest: MaterializationManifest, corpus_root: Path) -> None:
    required_paths = {
        "raw_cell_meta_path": manifest.raw_cell_meta_path,
        "raw_feature_meta_path": manifest.raw_feature_meta_path,
        "provenance_spec_path": manifest.provenance_spec_path,
        "size_factor_parquet_path": manifest.size_factor_parquet_path,
        "hvg_ranking_path": manifest.hvg_ranking_path,
    }
    for name, path_str in required_paths.items():
        if path_str is None:
            raise ValueError(f"manifest is missing required artifact path: {name}")
        path = _resolve_manifest_artifact(corpus_root, path_str)
        if not path.exists():
            raise FileNotFoundError(f"manifest artifact not found for {name}: {path}")

    matrix_root = _resolve_manifest_artifact(corpus_root, manifest.outputs.matrix_root)
    if not matrix_root.exists():
        raise FileNotFoundError(f"matrix_root does not exist: {matrix_root}")
    if not any(matrix_root.iterdir()):
        raise FileNotFoundError(f"matrix_root is empty: {matrix_root}")


def corpus_exists(corpus_root: Path) -> bool:
    """Check whether a corpus index exists at the given corpus root.

    Parameters
    ----------
    corpus_root : Path
        The corpus root directory.

    Returns
    -------
    bool
        True if corpus-index.yaml exists at corpus_root.
    """
    return (corpus_root / "corpus-index.yaml").exists()
