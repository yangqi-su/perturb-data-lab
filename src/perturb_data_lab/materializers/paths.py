"""Corpus path helpers for topology-aware directory layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusPaths:
    meta_root: Path
    matrix_root: Path
    canonical_meta_root: Path


def resolve_corpus_paths(
    topology: str,
    corpus_root: str | Path,
    dataset_id: str,
) -> CorpusPaths:
    """Resolve dataset roots for federated or aggregate topology.

    Aggregate topology:
    - meta_root: ``{corpus}/meta/{dataset_id}``
    - matrix_root: ``{corpus}/matrix``
    - canonical_meta_root: ``{corpus}/meta/{dataset_id}/canonical_meta``

    Federated topology:
    - meta_root: ``{corpus}/{dataset_id}/meta``
    - matrix_root: ``{corpus}/{dataset_id}/matrix``
    - canonical_meta_root: ``{corpus}/{dataset_id}/meta/canonical_meta``
    """
    root = Path(corpus_root)
    if topology == "aggregate":
        return CorpusPaths(
            meta_root=root / "meta" / dataset_id,
            matrix_root=root / "matrix",
            canonical_meta_root=root / "meta" / dataset_id / "canonical_meta",
        )
    if topology == "federated":
        dataset_root = root / dataset_id
        return CorpusPaths(
            meta_root=dataset_root / "meta",
            matrix_root=dataset_root / "matrix",
            canonical_meta_root=dataset_root / "meta" / "canonical_meta",
        )
    raise ValueError(
        f"unsupported topology '{topology}'; expected 'federated' or 'aggregate'"
    )
