"""Corpus factory for materialized perturbation corpora.

``load_corpus()`` reconstructs a training-ready ``Corpus`` from a corpus
directory using canonical metadata as the source of truth. Slim main supports
only Lance and Zarr corpora in aggregate or federated topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch
import yaml

from .expression import (
    DatasetEntry,
    ExpressionReader,
    LanceDatasetEntry,
    ZarrDatasetEntry,
    build_expression_reader,
)
from .feature_registry import FeatureRegistry
from .gene_tokenizer import GeneTokenizer
from .index import (
    MetadataIndex,
    _CANONICAL_OBS_CONTENT_COLUMNS,
    _CANONICAL_OBS_STRUCTURAL_COLUMNS,
    _CANONICAL_OBS_TYPED_DTYPES,
    _load_canonical_obs_frame,
    _normalize_canonical_obs_dtypes,
)
from .loaders import ExpressionBatch

# Import path resolver (Phase 1 output — canonical_meta under meta/)
from ..materializers.models import MaterializationManifest
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

    def read_expression(
        self,
        indices: torch.Tensor | np.ndarray | Sequence[int],
    ) -> ExpressionBatch:
        """Read flat expression arrays for ad-hoc corpus inspection."""
        normalized_indices = _normalize_batch_indices(indices).tolist()
        return self.expression_reader.read_expression_flat(normalized_indices)

    def take_metadata(
        self,
        indices: torch.Tensor | np.ndarray | Sequence[int],
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
    # legacy / alternate names
    "lancedb": "lance",
}

_REMOVED_BACKEND_NAMES: frozenset[str] = frozenset(
    {
        "arrow-parquet",
        "arrow_parquet",
        "arrow-ipc",
        "arrow_ipc",
        "arrow_hf",
        "hf-datasets",
        "hf_datasets",
        "huggingface-datasets",
        "datasets",
        "parquet",
        "tiledb",
        "webdataset",
        "csr-memmap",
        "csr_memmap",
    }
)


def _normalize_backend(raw: str) -> str:
    """Map corpus-index backend strings to ``build_expression_reader`` keys."""
    if not raw:
        raise ValueError("corpus-index.yaml global_metadata.backend is empty")
    norm = _BACKEND_NORMALIZE.get(raw)
    if norm is None:
        if raw in _REMOVED_BACKEND_NAMES:
            raise ValueError(
                f"Backend '{raw}' is not supported in slim main. Only 'lance' "
                "and 'zarr' corpora can be loaded here; use the preserved "
                "experimental snapshot branch for removed backends."
            )
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
    indices: torch.Tensor | np.ndarray | Sequence[int],
) -> np.ndarray:
    """Convert batch row indices from torch/numpy/sequence form to int64."""
    if isinstance(indices, torch.Tensor):
        return indices.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# load_corpus factory
# ---------------------------------------------------------------------------


def load_corpus(
    corpus_root: str | Path,
    *,
    use_canonical: bool = True,
    extra_metadata_columns: Sequence[str] | None = None,
) -> Corpus:
    """Load a training-ready ``Corpus`` from a corpus directory.

    Reads ``corpus-index.yaml``, locates canonical obs/var parquets via
    ``resolve_corpus_paths()``, and constructs expression, metadata, tokenizer,
    and feature-registry components.

    Parameters
    ----------
    corpus_root : str or Path
        Path to a corpus directory containing ``corpus-index.yaml``.
    use_canonical : bool
        Whether to use canonical obs/var parquets.  Default ``True``.
        When ``True``, reads ``canonical-obs.parquet`` and
        ``canonical-var.parquet`` from ``meta/{id}/canonical_meta/``.
    extra_metadata_columns : sequence of str, optional
        Additional canonical-obs parquet columns to load into
        ``metadata_index`` beyond the default canonical/core projection.

    Returns
    -------
    Corpus
        Fully-constructed corpus components ready for ``build_loader(...)`` or
        direct ``read_expression(...)`` / ``take_metadata(...)`` inspection.

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
    index_doc = _read_yaml(index_path)
    metadata = index_doc.get("global_metadata", {})
    corpus_id = str(index_doc.get("corpus_id", root.name))
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
    hvg_rank_paths: dict[str, str | Path] = {}
    default_n_hvg_by_dataset: dict[str, int] = {}
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
        hvg_path, default_n_hvg = _resolve_dataset_hvg_inputs(
            corpus_root=root,
            topology=topology,
            dataset_id=ds_id,
            ds_entry=ds_entry,
        )
        if hvg_path is not None:
            hvg_rank_paths[ds_id] = hvg_path
        if default_n_hvg is not None:
            default_n_hvg_by_dataset[ds_id] = int(default_n_hvg)
        global_ranges.append((ds_id, ds_index, g_start, g_end))

    # ------------------------------------------------------------------
    # 3. Build MetadataIndex from canonical obs parquets
    # ------------------------------------------------------------------
    metadata_index = _build_metadata_index(
        datasets_info=global_ranges,
        obs_paths=canonical_obs_paths,
        use_canonical=use_canonical,
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
    tokenizer_path_value = metadata.get("tokenizer_path")
    tokenizer_path = (
        root / str(tokenizer_path_value)
        if tokenizer_path_value
        else root / "gene-tokenizer.json"
    )
    if tokenizer_path.exists():
        gene_tokenizer = GeneTokenizer.from_json(tokenizer_path)
        if gene_tokenizer.dataset_build_order != tuple(dataset_order):
            raise ValueError(
                "Persisted gene tokenizer dataset_build_order does not match corpus-index.yaml order"
            )
    else:
        gene_tokenizer = GeneTokenizer.build_from_canonical_var_parquets(
            corpus_id=corpus_id,
            named_var_paths=var_path_map,
            dataset_order=dataset_order,
        )
    feature_registry = FeatureRegistry.from_canonical_var_parquets(
        var_path_map,
        dataset_order=dataset_order,
        global_id_by_feature_id=gene_tokenizer.token_to_id,
        named_hvg_rank_paths=hvg_rank_paths or None,
        default_n_hvg_by_dataset=default_n_hvg_by_dataset or None,
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return the parsed dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_dataset_hvg_inputs(
    *,
    corpus_root: Path,
    topology: str,
    dataset_id: str,
    ds_entry: dict[str, Any],
) -> tuple[Path | None, int | None]:
    """Resolve an optional canonical ``hvg.parquet`` plus default top-k."""
    resolved_paths = resolve_corpus_paths(topology, corpus_root, dataset_id)
    manifest_path_value = ds_entry.get("manifest_path")
    manifest: MaterializationManifest | None = None

    if manifest_path_value:
        manifest_path = Path(str(manifest_path_value))
        if not manifest_path.is_absolute():
            manifest_path = corpus_root / manifest_path
        if manifest_path.exists():
            manifest = MaterializationManifest.from_yaml_file(manifest_path)

    default_n_hvg = manifest.default_n_hvg if manifest is not None else None
    candidates: list[Path] = []
    if manifest is not None and manifest.hvg_ranking_path:
        manifest_hvg_path = Path(manifest.hvg_ranking_path)
        if not manifest_hvg_path.is_absolute():
            manifest_hvg_path = corpus_root / manifest_hvg_path
        candidates.append(manifest_hvg_path)
    candidates.append(resolved_paths.meta_root / "hvg.parquet")

    for candidate in candidates:
        if candidate.exists():
            return candidate, default_n_hvg
    return None, default_n_hvg


def _build_metadata_index(
    datasets_info: list[tuple[str, int, int, int]],
    obs_paths: dict[str, Path],
    use_canonical: bool = True,
    extra_metadata_columns: Sequence[str] | None = None,
) -> MetadataIndex:
    """Build a ``MetadataIndex`` from per-dataset canonical obs parquets.

    Reads each parquet, assigns corpus-global ``global_row_index`` and
    ``dataset_index``, concatenates, and wraps in ``MetadataIndex``.

    Parameters
    ----------
    datasets_info : list of (dataset_id, dataset_index, global_start, global_end)
        Ordered per-dataset routing information from ``corpus-index.yaml``.
    obs_paths : dict[str, Path]
        Mapping ``dataset_id → canonical-obs.parquet`` path.
    use_canonical : bool
        When ``True``, canonical columns (``perturb_label``, etc.) are
        expected in the parquet.

    Returns
    -------
    MetadataIndex
    """
    if not use_canonical:
        raise NotImplementedError(
            "Non-canonical MetadataIndex loading is not yet supported in "
            "load_corpus(). Pass use_canonical=True."
        )

    processed: list[pl.DataFrame] = []

    for ds_id, ds_index, g_start, g_end in datasets_info:
        obs_path = obs_paths[ds_id]
        df = _load_canonical_obs_frame(
            obs_path,
            extra_metadata_columns=extra_metadata_columns,
            context=(
                f"canonical obs parquet for dataset '{ds_id}' at {obs_path}"
            ),
        )
        n_obs = len(df)

        # Override with corpus-global values
        df = df.with_columns(
            pl.lit(ds_index, dtype=pl.Int32).alias("dataset_index"),
        )
        df = df.with_columns(
            (pl.int_range(0, n_obs, dtype=pl.Int64) + g_start).alias(
                "global_row_index"
            ),
        )
        df = df.with_columns(
            pl.int_range(0, n_obs, dtype=pl.Int64).alias("local_row_index"),
        )

        processed.append(df)

    # Concatenate all datasets
    combined = pl.concat(processed, how="diagonal_relaxed")

    # Reorder: structural columns first, then canonical content, then the rest
    structural = list(_CANONICAL_OBS_STRUCTURAL_COLUMNS)
    # Ensure all structural columns exist (fill missing with null)
    for col in structural:
        if col not in combined.columns:
            fill_dtype = _CANONICAL_OBS_TYPED_DTYPES.get(col, pl.Utf8)
            combined = combined.with_columns(
                pl.lit(None, dtype=fill_dtype).alias(col)
            )

    combined = _normalize_canonical_obs_dtypes(combined)

    # Canonical content columns (may or may not all be present)
    content_cols = [c for c in _CANONICAL_OBS_CONTENT_COLUMNS if c in combined.columns]

    # Everything else (extensible / raw_ columns)
    existing = set(combined.columns)
    other_cols = sorted(
        c for c in combined.columns
        if c not in structural and c not in content_cols
    )

    col_order = structural + content_cols + other_cols
    combined = combined.select(col_order)

    # Sort by global_row_index for deterministic access
    combined = combined.sort("global_row_index")

    return MetadataIndex(combined)
