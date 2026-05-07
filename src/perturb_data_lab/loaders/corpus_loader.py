"""Phase 2: ``load_corpus()`` factory — reconstruct a training-ready ``Corpus``
from a corpus directory using canonical metadata as the source of truth.

The function reads ``corpus-index.yaml``, locates canonical obs/var parquets
via ``resolve_corpus_paths()``, and wires up ``MetadataIndex``,
``ExpressionReader``, ``BatchExecutor``, and ``FeatureRegistry`` so that
training scripts need zero manual path arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader

from .executor import BatchExecutor
from .expression import (
    CsrMemmapShardEntry,
    DatasetEntry,
    LanceDatasetEntry,
    build_expression_reader,
)
from .feature_registry import FeatureRegistry
from .index import MetadataIndex
from .loaders import (
    CorpusRandomBatchSampler,
    DatasetBatchSampler,
    DatasetContextBatchSampler,
    LanceExpressionBatchDataset,
    RawExpressionBatchDataset,
    collate_raw_expression_batch,
)

# Import path resolver (Phase 1 output — canonical_meta under meta/)
from ..materializers.paths import resolve_corpus_paths

__all__ = [
    "Corpus",
    "load_corpus",
]


_RAW_BATCH_RESERVED_KEYS: frozenset[str] = frozenset(
    {
        "batch_size",
        "global_row_index",
        "dataset_index",
        "local_row_index",
        "size_factor",
        "row_offsets",
        "expressed_gene_indices",
        "expression_counts",
        "meta_columns",
    }
)


# ---------------------------------------------------------------------------
# Corpus dataclass
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    """Ready-to-train corpus object.

    Contains everything needed to instantiate a ``PerturbBatchDataset``
    and a GPU/CPU pipeline — no manual path arithmetic required.

    Attributes
    ----------
    batch_executor : BatchExecutor
        Combined metadata + expression batch reader.
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

    batch_executor: BatchExecutor
    feature_registry: FeatureRegistry
    metadata_index: MetadataIndex
    dataset_entries: list[DatasetEntry] = field(default_factory=list)
    dataset_index_by_id: dict[str, int] = field(default_factory=dict)
    topology: str = ""
    backend: str = ""
    corpus_root: Path = Path()
    _sampler: Any | None = field(default=None, init=False, repr=False)
    _sampler_params: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False,
    )

    @property
    def sampler(self) -> Any | None:
        """Return the currently configured sampler, if any."""
        return self._sampler

    @property
    def sampler_params(self) -> dict[str, Any]:
        """Return the normalized sampler configuration."""
        return dict(self._sampler_params)

    def set_sampler(self, **params: Any) -> Any:
        """Build and store a metadata-index-backed sampler for this corpus."""
        normalized = _normalize_sampler_params(params)
        self._sampler = _build_sampler(self.metadata_index, normalized)
        self._sampler_params = normalized
        return self._sampler

    def dataset(
        self,
        *,
        metadata_columns: Sequence[str] | None = None,
    ) -> RawExpressionBatchDataset | LanceExpressionBatchDataset:
        """Return the expression-only dataset for the corpus loader refactor."""
        columns = _normalize_metadata_columns(
            self.metadata_index, metadata_columns,
        )
        if self.backend == "lance" and not columns:
            size_factor = None
            if "size_factor" in self.metadata_index.df.columns:
                size_factor = self.metadata_index.df["size_factor"].to_numpy()
            ordered_entries = sorted(
                self.dataset_entries, key=lambda entry: entry.global_start,
            )
            return LanceExpressionBatchDataset(
                self.batch_executor.expression_reader,
                dataset_starts=np.array(
                    [entry.global_start for entry in ordered_entries],
                    dtype=np.int64,
                ),
                dataset_stops=np.array(
                    [entry.global_end for entry in ordered_entries],
                    dtype=np.int64,
                ),
                dataset_indices=np.array(
                    [self.dataset_index_by_id[entry.dataset_id] for entry in ordered_entries],
                    dtype=np.int32,
                ),
                size_factor=size_factor,
                topology=self.topology,
                backend=self.backend,
            )
        return RawExpressionBatchDataset(
            self.batch_executor,
            metadata_columns=columns,
            topology=self.topology,
            backend=self.backend,
        )

    def loader(
        self,
        *,
        processing: str = "gpu",
        device: torch.device | str | None = None,
        num_workers: int = 0,
        multiprocessing_context: str | None = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        metadata_columns: Sequence[str] | None = None,
        sampler: str | None = None,
        batch_size: int | None = None,
        drop_last: bool = True,
        seed: int = 0,
        shuffle: bool = True,
        dataset_index: int | None = None,
        context_field: str = "raw_cell_type",
    ) -> Iterator[dict[str, Any]]:
        """Yield raw expression batches after validating the corpus API surface.

        Phase 2 intentionally stops at the shared raw-batch contract. Both
        ``processing="gpu"`` and ``processing="cpu"`` currently return the
        same collated raw batch shape; route-specific sparse processing lands in
        later phases.
        """
        validated = _validate_loader_params(
            processing=processing,
            device=device,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            metadata_columns=metadata_columns,
        )

        if (
            validated["num_workers"] > 0
            and self.backend == "lance"
            and validated["metadata_columns"]
        ):
            raise NotImplementedError(
                "Phase 5 Lance worker loading is expression-only; "
                "metadata_columns with num_workers > 0 are deferred to the "
                "later metadata-attachment phases."
            )

        resolved_sampler = _resolve_loader_sampler(
            self,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
            shuffle=shuffle,
            dataset_index=dataset_index,
            context_field=context_field,
        )

        dataset_obj = self.dataset(
            metadata_columns=validated["metadata_columns"],
        )
        loader_kwargs = _build_dataloader_kwargs(
            num_workers=validated["num_workers"],
            multiprocessing_context=validated["multiprocessing_context"],
            pin_memory=validated["pin_memory"],
            persistent_workers=validated["persistent_workers"],
            prefetch_factor=validated["prefetch_factor"],
            backend=self.backend,
        )
        data_loader = DataLoader(
            dataset_obj,
            batch_sampler=resolved_sampler,
            collate_fn=collate_raw_expression_batch,
            **loader_kwargs,
        )
        return iter(data_loader)


# ---------------------------------------------------------------------------
# Backend name normalisation
# ---------------------------------------------------------------------------

_BACKEND_NORMALIZE: dict[str, str] = {
    "arrow-parquet": "parquet",
    "arrow-ipc": "arrow_ipc",
    "lance": "lance",
    "zarr": "zarr",
    "webdataset": "webdataset",
    "csr-memmap": "csr_memmap",
    # legacy / alternate names
    "arrow_hf": "parquet",
    "arrow_ipc": "arrow_ipc",
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


def _normalize_sampler_params(params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize ``Corpus.set_sampler()`` parameters."""
    kind = str(params.get("sampler", params.get("kind", "corpus_random")))
    kind = kind.strip().lower().replace("-", "_")
    if kind not in {"corpus_random", "dataset", "dataset_context"}:
        raise ValueError(
            "sampler must be one of 'corpus_random', 'dataset', or "
            f"'dataset_context', got {kind!r}"
        )

    batch_size = params.get("batch_size")
    if batch_size is None:
        raise ValueError("batch_size is required to configure a sampler")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    normalized: dict[str, Any] = {
        "sampler": kind,
        "batch_size": batch_size,
        "drop_last": bool(params.get("drop_last", True)),
        "seed": int(params.get("seed", 0)),
    }

    if kind == "corpus_random":
        return normalized

    if "dataset_index" not in params or params["dataset_index"] is None:
        raise ValueError(
            f"dataset_index is required for sampler={kind!r}"
        )
    normalized["dataset_index"] = int(params["dataset_index"])

    if kind == "dataset":
        normalized["shuffle"] = bool(params.get("shuffle", True))
        return normalized

    normalized["shuffle"] = bool(params.get("shuffle", True))
    normalized["context_field"] = str(
        params.get("context_field", "raw_cell_type")
    )
    return normalized


def _build_sampler(
    metadata_index: MetadataIndex,
    params: dict[str, Any],
) -> Any:
    """Instantiate the normalized sampler config against a MetadataIndex."""
    kind = params["sampler"]
    if kind == "corpus_random":
        return CorpusRandomBatchSampler(
            metadata_index=metadata_index,
            batch_size=params["batch_size"],
            drop_last=params["drop_last"],
            seed=params["seed"],
        )
    if kind == "dataset":
        return DatasetBatchSampler(
            metadata_index=metadata_index,
            dataset_index=params["dataset_index"],
            batch_size=params["batch_size"],
            drop_last=params["drop_last"],
            shuffle=params["shuffle"],
            seed=params["seed"],
        )
    if kind == "dataset_context":
        return DatasetContextBatchSampler(
            metadata_index=metadata_index,
            batch_size=params["batch_size"],
            context_field=params["context_field"],
            dataset_index=params["dataset_index"],
            drop_last=params["drop_last"],
            shuffle=params["shuffle"],
            seed=params["seed"],
        )
    raise ValueError(f"Unsupported sampler kind: {kind!r}")


def _resolve_loader_sampler(
    corpus: Corpus,
    *,
    sampler: str | None,
    batch_size: int | None,
    drop_last: bool,
    seed: int,
    shuffle: bool,
    dataset_index: int | None,
    context_field: str,
) -> Any:
    """Resolve the sampler to use for one loader invocation."""
    if sampler is None and batch_size is None:
        if corpus.sampler is None:
            raise ValueError(
                "No sampler is configured. Call corpus.set_sampler(...) or "
                "provide loader sampler defaults such as batch_size=."
            )
        return corpus.sampler

    return _build_sampler(
        corpus.metadata_index,
        _normalize_sampler_params(
            {
                "sampler": sampler or "corpus_random",
                "batch_size": batch_size,
                "drop_last": drop_last,
                "seed": seed,
                "shuffle": shuffle,
                "dataset_index": dataset_index,
                "context_field": context_field,
            }
        ),
    )


def _normalize_metadata_columns(
    metadata_index: MetadataIndex,
    metadata_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate requested optional metadata columns."""
    if metadata_columns is None:
        return ()

    if isinstance(metadata_columns, (str, bytes)):
        raise TypeError("metadata_columns must be a sequence of column names")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in metadata_columns:
        name = str(raw)
        if not name:
            raise ValueError("metadata_columns cannot contain empty names")
        if name in _RAW_BATCH_RESERVED_KEYS:
            raise ValueError(
                f"metadata_columns cannot request reserved raw batch field {name!r}"
            )
        if name not in metadata_index.df.columns:
            raise ValueError(
                f"metadata column {name!r} not found. Available columns: "
                f"{metadata_index.df.columns}"
            )
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return tuple(normalized)


def _validate_loader_params(
    *,
    processing: str,
    device: torch.device | str | None,
    num_workers: int,
    multiprocessing_context: str | None,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    metadata_columns: Sequence[str] | None,
) -> dict[str, Any]:
    """Validate public ``Corpus.loader(...)`` parameters."""
    normalized_processing = str(processing).strip().lower()
    if normalized_processing not in {"gpu", "cpu"}:
        raise ValueError(
            f"processing must be 'gpu' or 'cpu', got {processing!r}"
        )

    if device is not None:
        try:
            torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid device {device!r}") from exc

    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers must be >= 0")

    if not isinstance(pin_memory, bool):
        raise TypeError("pin_memory must be a bool")
    if not isinstance(persistent_workers, bool):
        raise TypeError("persistent_workers must be a bool")

    if workers == 0:
        if multiprocessing_context is not None:
            raise ValueError(
                "multiprocessing_context requires num_workers > 0"
            )
        if persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")
        if prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0")
        normalized_context = None
        normalized_prefetch = None
    else:
        if multiprocessing_context is None:
            normalized_context = None
        else:
            normalized_context = str(multiprocessing_context).strip().lower()
            if normalized_context not in {"fork", "spawn", "forkserver"}:
                raise ValueError(
                    "multiprocessing_context must be one of 'fork', 'spawn', "
                    f"or 'forkserver', got {multiprocessing_context!r}"
                )
        if prefetch_factor is None:
            normalized_prefetch = None
        else:
            normalized_prefetch = int(prefetch_factor)
            if normalized_prefetch <= 0:
                raise ValueError("prefetch_factor must be positive")

    return {
        "processing": normalized_processing,
        "device": device,
        "num_workers": workers,
        "multiprocessing_context": normalized_context,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": normalized_prefetch,
        "metadata_columns": metadata_columns,
    }


def _build_dataloader_kwargs(
    *,
    num_workers: int,
    multiprocessing_context: str | None,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    backend: str,
) -> dict[str, Any]:
    """Build validated DataLoader kwargs from the public loader config."""
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0:
        kwargs["multiprocessing_context"] = (
            multiprocessing_context
            if multiprocessing_context is not None
            else ("spawn" if backend == "lance" else None)
        )
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


# ---------------------------------------------------------------------------
# load_corpus factory
# ---------------------------------------------------------------------------


def load_corpus(
    corpus_root: str | Path,
    *,
    seq_len: int | None = None,
    use_canonical: bool = True,
) -> Corpus:
    """Load a training-ready ``Corpus`` from a corpus directory.

    Reads ``corpus-index.yaml``, locates canonical obs/var parquets via
    ``resolve_corpus_paths()``, and constructs all loader components.

    Parameters
    ----------
    corpus_root : str or Path
        Path to a corpus directory containing ``corpus-index.yaml``.
    seq_len : int, optional
        Target sequence length for downstream tokenisation (reserved for
        future use; not consumed by this factory).
    use_canonical : bool
        Whether to use canonical obs/var parquets.  Default ``True``.
        When ``True``, reads ``canonical-obs.parquet`` and
        ``canonical-var.parquet`` from ``meta/{id}/canonical_meta/``.

    Returns
    -------
    Corpus
        Fully-constructed corpus object ready for ``PerturbBatchDataset``
        and pipeline instantiation.

    Raises
    ------
    FileNotFoundError
        If ``corpus-index.yaml``, any canonical parquet file, or required
        matrix file is missing.
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
    metadata_index = _build_metadata_index(
        datasets_info=global_ranges,
        obs_paths=canonical_obs_paths,
        use_canonical=use_canonical,
    )

    # ------------------------------------------------------------------
    # 4. Build dataset entries and ExpressionReader
    # ------------------------------------------------------------------
    entries: list[DatasetEntry]
    if topology == "aggregate":
        if backend == "lance":
            lance_path = str(root / "matrix" / "aggregated-cells.lance")
            if not Path(lance_path).exists():
                raise FileNotFoundError(
                    f"Aggregate Lance file not found: {lance_path}"
                )
            entries = [
                DatasetEntry(
                    dataset_id=ds_id,
                    global_start=g_start,
                    global_end=g_end,
                )
                for ds_id, _dsi, g_start, g_end in global_ranges
            ]
            expression_reader = build_expression_reader(
                backend, topology, entries, lance_path=lance_path,
            )
        elif backend == "csr_memmap":
            # Read CSR manifest to build shard-level entries
            manifest_path = root / "csr-corpus-manifest.yaml"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"CSR corpus manifest not found: {manifest_path}"
                )
            entries = _build_csr_entries(root, manifest_path)
            expression_reader = build_expression_reader(
                backend, topology, entries,
            )
        else:
            raise ValueError(
                f"Unsupported backend '{backend}' for aggregate topology. "
                f"Only 'lance' is currently supported for aggregate."
            )
    elif topology == "federated":
        if backend == "lance":
            entries = [
                LanceDatasetEntry(
                    dataset_id=ds_id,
                    global_start=g_start,
                    global_end=g_end,
                    lance_path=str(
                        root / ds_id / "matrix" / f"{ds_id}.lance"
                    ),
                )
                for ds_id, _dsi, g_start, g_end in global_ranges
            ]
            # Verify all lance files exist
            for entry in entries:
                assert isinstance(entry, LanceDatasetEntry)
                if not Path(str(entry.lance_path)).exists():
                    raise FileNotFoundError(
                        f"Lance file not found for dataset '{entry.dataset_id}': "
                        f"{entry.lance_path}"
                    )
            expression_reader = build_expression_reader(
                backend, topology, entries,
            )
        else:
            raise ValueError(
                f"Unsupported backend '{backend}' for federated topology. "
                f"Only 'lance' is currently supported for federated."
            )
    else:
        raise ValueError(
            f"Unknown topology '{topology}'. "
            f"Expected 'aggregate' or 'federated'."
        )

    # ------------------------------------------------------------------
    # 5. Build BatchExecutor
    # ------------------------------------------------------------------
    batch_executor = BatchExecutor(
        expression_reader=expression_reader,
        metadata_index=metadata_index,
        use_canonical=use_canonical,
    )

    # ------------------------------------------------------------------
    # 6. Build FeatureRegistry from canonical var parquets
    # ------------------------------------------------------------------
    var_path_map: dict[str, str] = {
        ds_id: str(p) for ds_id, p in canonical_var_paths.items()
    }
    feature_registry = FeatureRegistry.from_canonical_var_parquets(
        var_path_map
    )

    # ------------------------------------------------------------------
    # 7. Return Corpus
    # ------------------------------------------------------------------
    return Corpus(
        batch_executor=batch_executor,
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


def _build_csr_entries(
    corpus_root: Path, manifest_path: Path
) -> list[CsrMemmapShardEntry]:
    """Build ``CsrMemmapShardEntry`` list from a ``csr-corpus-manifest.yaml``.

    Parameters
    ----------
    corpus_root : Path
        Root of the CSR corpus directory (for resolving relative shard paths).
    manifest_path : Path
        Path to ``csr-corpus-manifest.yaml``.

    Returns
    -------
    list of CsrMemmapShardEntry
        One entry per shard, sorted by ``global_start``.

    Raises
    ------
    ValueError
        If the manifest is missing required fields or the shards list is empty.
    FileNotFoundError
        If any required ``.npy`` file is missing.
    """
    doc = _read_yaml(manifest_path)
    if doc.get("kind") != "csr-corpus-manifest":
        raise ValueError(
            f"Expected csr-corpus-manifest, got kind={doc.get('kind')!r}"
        )
    shards = doc.get("shards", [])
    if not shards:
        raise ValueError("CSR corpus manifest contains no shards")

    entries: list[CsrMemmapShardEntry] = []
    for s in shards:
        shard_id = int(s["shard_id"])
        shard_dir_name = s["path"]  # relative: "shard_000000"
        shard_dir = corpus_root / shard_dir_name

        ro_path = shard_dir / "row_offsets.npy"
        gi_path = shard_dir / "gene_indices.npy"
        cnt_path = shard_dir / "counts.npy"
        if not ro_path.is_file():
            raise FileNotFoundError(f"Missing: {ro_path}")
        if not gi_path.is_file():
            raise FileNotFoundError(f"Missing: {gi_path}")
        if not cnt_path.is_file():
            raise FileNotFoundError(f"Missing: {cnt_path}")

        n_cells = int(s["n_cells"])
        g_start = int(s["global_start"])
        g_end = int(s["global_end"])

        entries.append(
            CsrMemmapShardEntry(
                dataset_id=shard_dir_name,  # unique shard identifier
                global_start=g_start,
                global_end=g_end,
                shard_id=shard_id,
                shard_path=shard_dir,
                row_offsets_path=ro_path,
                gene_indices_path=gi_path,
                counts_path=cnt_path,
                n_cells=n_cells,
            )
        )

    # Sort by global_start (defensive — manifest should already be sorted)
    entries.sort(key=lambda e: e.global_start)
    return entries


def _build_metadata_index(
    datasets_info: list[tuple[str, int, int, int]],
    obs_paths: dict[str, Path],
    use_canonical: bool = True,
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

    # Numeric columns that arrive as strings in canonical parquets
    _num_cast: dict[str, pl.DataType] = {
        "global_row_index": pl.Int64,
        "dataset_index": pl.Int32,
        "local_row_index": pl.Int64,
        "size_factor": pl.Float64,
    }

    # Required structural columns for MetadataIndex
    _required = {
        "global_row_index", "cell_id", "dataset_id",
        "dataset_index", "local_row_index", "size_factor",
    }

    for ds_id, ds_index, g_start, g_end in datasets_info:
        obs_path = obs_paths[ds_id]
        df = pl.read_parquet(str(obs_path))
        n_obs = len(df)

        # Cast string-typed numeric columns to proper types
        for col_name, dtype in _num_cast.items():
            if col_name in df.columns and df[col_name].dtype == pl.Utf8:
                try:
                    df = df.with_columns(
                        pl.col(col_name).cast(dtype)
                    )
                except Exception:
                    # If casting fails (e.g., "NA" strings for size_factor),
                    # null-instead mode: replace "NA" with null then cast
                    df = df.with_columns(
                        pl.when(pl.col(col_name) == "NA")
                        .then(None)
                        .otherwise(pl.col(col_name))
                        .cast(pl.Utf8)
                        .cast(dtype)
                        .alias(col_name)
                    )

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
    structural = [
        "global_row_index", "cell_id", "dataset_id",
        "dataset_index", "local_row_index", "size_factor",
    ]
    # Ensure all structural columns exist (fill missing with null)
    for col in structural:
        if col not in combined.columns:
            combined = combined.with_columns(
                pl.lit(None).alias(col)
            )

    # Canonical content columns (may or may not all be present)
    _canonical_content = [
        "perturb_label", "perturb_type", "dose", "dose_unit",
        "timepoint", "timepoint_unit", "cell_context", "cell_line_or_type",
        "species", "tissue", "assay", "condition", "batch_id",
        "donor_id", "sex", "disease_state",
    ]
    content_cols = [c for c in _canonical_content if c in combined.columns]

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
