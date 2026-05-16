"""Corpus factory for materialized perturbation corpora.

``load_corpus()`` reconstructs a training-ready ``Corpus`` from a corpus
directory using canonical metadata as the source of truth. Slim main supports
only Lance and Zarr corpora in aggregate or federated topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from .expression import (
    DatasetEntry,
    ExpressionReader,
    LanceDatasetEntry,
    ZarrDatasetEntry,
    build_expression_reader,
)
from .feature_registry import FeatureRegistry
from .index import MetadataIndex

from ..materializers.paths import resolve_corpus_paths

__all__ = [
    "Corpus",
    "load_corpus",
]
# ---------------------------------------------------------------------------
# Corpus dataclass
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    """Loaded corpus components.

    Attributes
    ----------
    expression_reader : ExpressionReader
        Backend-aware flat expression reader.
    feature_registry : FeatureRegistry
        Per-dataset local→global gene ID mapping.
    metadata_index : MetadataIndex
        Polars-backed flat-schema metadata table for all cells.
    dataset_entries : list[DatasetEntry]
        Backend-aware dataset routing entries.
    topology : str
        Corpus topology: ``"aggregate"`` or ``"federated"``.
    backend : str
        Storage backend key (normalized to ``build_expression_reader`` form).
    corpus_root : Path
        Absolute path to the corpus root directory.
    """

    expression_reader: ExpressionReader
    feature_registry: FeatureRegistry
    metadata_index: MetadataIndex
    dataset_entries: list[DatasetEntry] = field(default_factory=list)
    dataset_index_by_id: dict[str, int] = field(default_factory=dict)
    topology: str = ""
    backend: str = ""
    corpus_root: Path = Path()

    def take_metadata(
        self,
        indices: np.ndarray | Sequence[int],
        *,
        columns: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray | tuple]:
        """Return columnar metadata for selected corpus-global row indices.

        Use this to recover provenance fields such as ``local_row_index`` or to
        inspect rich annotations without constructing a DataLoader.
        """
        normalized_indices = _normalize_batch_indices(indices)
        resolved_columns = _normalize_take_columns(self.metadata_index, columns)
        return self.metadata_index.take(normalized_indices, resolved_columns)

# ---------------------------------------------------------------------------
# Backend name normalisation
# ---------------------------------------------------------------------------

_BACKEND_NORMALIZE: dict[str, str] = {
    "lance": "lance",
    "zarr": "zarr",
    "lancedb": "lance",
}


def _normalize_backend(raw: str) -> str:
    """Map corpus-index backend strings to ``build_expression_reader`` keys."""
    if not raw:
        raise ValueError("corpus-index.yaml global_metadata.backend is empty")
    norm = _BACKEND_NORMALIZE.get(raw)
    if norm is None:
        raise ValueError(
            f"Unsupported corpus backend '{raw}'. "
            f"Supported: {sorted(_BACKEND_NORMALIZE.keys())}"
        )
    return norm


def _build_range_entries(
    global_ranges: Sequence[tuple[str, int, int, int]],
) -> list[DatasetEntry]:
    return [
        DatasetEntry(
            dataset_id=ds_id,
            global_start=g_start,
            global_end=g_end,
        )
        for ds_id, _dsi, g_start, g_end in global_ranges
    ]


def _build_aggregate_expression_components(
    root: Path,
    backend: str,
    global_ranges: Sequence[tuple[str, int, int, int]],
) -> tuple[list[DatasetEntry], ExpressionReader]:
    entries = _build_range_entries(global_ranges)
    if backend == "lance":
        lance_path = root / "matrix" / "aggregated-cells.lance"
        if not lance_path.exists():
            raise FileNotFoundError(
                f"Aggregate Lance file not found: {lance_path}"
            )
        return entries, build_expression_reader(
            backend,
            "aggregate",
            entries,
            lance_path=str(lance_path),
        )

    row_offsets_path = root / "matrix" / "aggregated-row-offsets.zarr"
    indices_path = root / "matrix" / "aggregated-indices.zarr"
    counts_path = root / "matrix" / "aggregated-counts.zarr"
    if not row_offsets_path.is_dir():
        raise FileNotFoundError(
            "Aggregate Zarr row-offsets artifact not found: "
            f"{row_offsets_path}"
        )
    if not indices_path.is_dir():
        raise FileNotFoundError(
            f"Aggregate Zarr indices artifact not found: {indices_path}"
        )
    if not counts_path.is_dir():
        raise FileNotFoundError(
            f"Aggregate Zarr counts artifact not found: {counts_path}"
        )
    return entries, build_expression_reader(
        backend,
        "aggregate",
        entries,
        offsets_path=str(row_offsets_path),
        indices_path=str(indices_path),
        counts_path=str(counts_path),
    )


def _build_federated_dataset_entry(
    root: Path,
    backend: str,
    dataset_id: str,
    global_start: int,
    global_end: int,
) -> LanceDatasetEntry | ZarrDatasetEntry:
    matrix_root = resolve_corpus_paths("federated", root, dataset_id).matrix_root
    if backend == "lance":
        lance_path = matrix_root / f"{dataset_id}.lance"
        if not lance_path.exists():
            raise FileNotFoundError(
                f"Lance file not found for dataset '{dataset_id}': "
                f"{lance_path}"
            )
        return LanceDatasetEntry(
            dataset_id=dataset_id,
            global_start=global_start,
            global_end=global_end,
            lance_path=str(lance_path),
        )

    row_offsets_path = matrix_root / f"{dataset_id}-row-offsets.zarr"
    indices_path = matrix_root / f"{dataset_id}-indices.zarr"
    counts_path = matrix_root / f"{dataset_id}-counts.zarr"
    if not row_offsets_path.is_dir():
        raise FileNotFoundError(
            f"Zarr row-offsets artifact not found for dataset '{dataset_id}': "
            f"{row_offsets_path}"
        )
    if not indices_path.is_dir():
        raise FileNotFoundError(
            f"Zarr indices artifact not found for dataset '{dataset_id}': "
            f"{indices_path}"
        )
    if not counts_path.is_dir():
        raise FileNotFoundError(
            f"Zarr counts artifact not found for dataset '{dataset_id}': "
            f"{counts_path}"
        )
    return ZarrDatasetEntry(
        dataset_id=dataset_id,
        global_start=global_start,
        global_end=global_end,
        offsets_path=str(row_offsets_path),
        indices_path=str(indices_path),
        counts_path=str(counts_path),
    )


def _build_federated_expression_components(
    root: Path,
    backend: str,
    global_ranges: Sequence[tuple[str, int, int, int]],
) -> tuple[list[DatasetEntry], ExpressionReader]:
    entries = [
        _build_federated_dataset_entry(root, backend, ds_id, g_start, g_end)
        for ds_id, _dsi, g_start, g_end in global_ranges
    ]
    return entries, build_expression_reader(backend, "federated", entries)


def _build_expression_components(
    root: Path,
    topology: str,
    backend: str,
    global_ranges: Sequence[tuple[str, int, int, int]],
) -> tuple[list[DatasetEntry], ExpressionReader]:
    if topology == "aggregate":
        return _build_aggregate_expression_components(root, backend, global_ranges)
    if topology == "federated":
        return _build_federated_expression_components(root, backend, global_ranges)
    raise ValueError(
        f"Unknown topology '{topology}'. "
        f"Expected 'aggregate' or 'federated'."
    )


def _normalize_take_columns(
    metadata_index: MetadataIndex,
    columns: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate columns for ``Corpus.take_metadata(...)``."""
    if columns is None:
        return tuple(metadata_index.df.columns)
    if isinstance(columns, (str, bytes)):
        raise TypeError("columns must be a sequence of column names")
    resolved: list[str] = []
    for name in columns:
        if not isinstance(name, str):
            raise TypeError("columns must contain strings")
        if name not in metadata_index.df.columns:
            raise ValueError(f"metadata column {name!r} not found")
        if name not in resolved:
            resolved.append(name)
    return tuple(resolved)


def _normalize_batch_indices(
    indices: np.ndarray | Sequence[int],
) -> np.ndarray:
    """Convert batch row indices to int64."""
    return np.asarray(indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# load_corpus factory
# ---------------------------------------------------------------------------


def load_corpus(
    corpus_root: str | Path,
    *,
    extra_metadata_columns: Sequence[str] | None = None,
) -> Corpus:
    """Load a training-ready ``Corpus`` from a corpus directory.

    Reads ``corpus-index.yaml``, locates canonical obs/var parquets via
    ``resolve_corpus_paths()``, and constructs expression, metadata, and
    feature-registry components.

    Parameters
    ----------
    corpus_root : str or Path
        Path to a corpus directory containing ``corpus-index.yaml``.
    extra_metadata_columns : sequence of str, optional
        Additional canonical-obs parquet columns to load into
        ``metadata_index`` beyond the default canonical/core projection.

    Returns
    -------
    Corpus
        Fully-constructed corpus components ready for ``build_loader(...)`` or
        direct expression-reader / ``take_metadata(...)`` inspection.

    Raises
    ------
    FileNotFoundError
        If ``corpus-index.yaml``, any canonical parquet file, or any required
        backend matrix artifact is missing.
    ValueError
        If the corpus topology or backend is unsupported.
    """
    root = Path(corpus_root).resolve()
    index_path = root / "corpus-index.yaml"
    if not index_path.exists():
        raise FileNotFoundError(
            f"corpus-index.yaml not found at {index_path}"
        )

    # ------------------------------------------------------------------
    # 1. Parse corpus-index.yaml
    # ------------------------------------------------------------------
    with open(index_path, encoding="utf-8") as handle:
        index_doc = yaml.safe_load(handle) or {}
    metadata = index_doc.get("global_metadata", {})
    topology = str(metadata.get("topology", ""))
    raw_backend = str(metadata.get("backend", ""))
    backend = _normalize_backend(raw_backend)
    datasets = index_doc.get("datasets", [])
    if not datasets:
        raise ValueError(
            f"No datasets list in corpus-index.yaml: {index_path}"
        )

    # ------------------------------------------------------------------
    # 2. Resolve canonical obs/var paths and matrix paths
    # ------------------------------------------------------------------
    canonical_obs_paths: dict[str, Path] = {}
    canonical_var_paths: dict[str, Path] = {}
    global_ranges: list[tuple[str, int, int, int]] = []
    # (dataset_id, dataset_index, global_start, global_end)

    for ds_entry in datasets:
        ds_id = str(ds_entry["dataset_id"])
        ds_index = int(ds_entry.get("dataset_index", 0))
        g_start = int(ds_entry.get("global_start", 0))
        g_end = int(ds_entry.get("global_end", 0))

        paths = resolve_corpus_paths(topology, root, ds_id)
        obs_path = paths.canonical_meta_root / "canonical-obs.parquet"
        var_path = paths.canonical_meta_root / "canonical-var.parquet"

        if not obs_path.exists():
            raise FileNotFoundError(
                f"canonical-obs.parquet not found for dataset '{ds_id}' "
                f"at {obs_path}"
            )
        if not var_path.exists():
            raise FileNotFoundError(
                f"canonical-var.parquet not found for dataset '{ds_id}' "
                f"at {var_path}"
            )

        canonical_obs_paths[ds_id] = obs_path
        canonical_var_paths[ds_id] = var_path
        global_ranges.append((ds_id, ds_index, g_start, g_end))

    # ------------------------------------------------------------------
    # 3. Build MetadataIndex from canonical obs parquets
    # ------------------------------------------------------------------
    metadata_index = MetadataIndex.from_canonical_obs_parquets(
        datasets_info=global_ranges,
        obs_paths=canonical_obs_paths,
        extra_metadata_columns=extra_metadata_columns,
    )

    # ------------------------------------------------------------------
    # 4. Build dataset entries and ExpressionReader
    # ------------------------------------------------------------------
    entries, expression_reader = _build_expression_components(
        root,
        topology,
        backend,
        global_ranges,
    )

    # ------------------------------------------------------------------
    # 5. Build FeatureRegistry from canonical var parquets
    # ------------------------------------------------------------------
    var_path_map: dict[str, str] = {
        ds_id: str(p) for ds_id, p in canonical_var_paths.items()
    }
    dataset_order = [ds_id for ds_id, *_ in global_ranges]
    feature_registry = FeatureRegistry.from_canonical_var_parquets(
        var_path_map,
        dataset_order=dataset_order,
    )

    # ------------------------------------------------------------------
    # 6. Return Corpus
    # ------------------------------------------------------------------
    return Corpus(
        expression_reader=expression_reader,
        feature_registry=feature_registry,
        metadata_index=metadata_index,
        dataset_entries=list(entries),
        dataset_index_by_id={ds_id: ds_index for ds_id, ds_index, *_ in global_ranges},
        topology=topology,
        backend=backend,
        corpus_root=root,
    )
