"""Corpus factory and modern composable corpus runtime API.

``load_corpus()`` reconstructs a training-ready ``Corpus`` from a corpus
directory using canonical metadata as the source of truth. Supported
artifact-backed routes are aggregate Lance, federated Lance, federated Zarr,
federated Arrow IPC, federated HuggingFace Datasets, federated Parquet, and
aggregate CSR memmap. WebDataset remains dormant in the package for future
artifact work, but ``load_corpus(...)`` does not enable it yet.

The preferred runtime flow is::

    corpus = load_corpus(path)
    corpus.set_sampler(batch_size=256, seed=0)
    dataset = corpus.dataset()
    expr = corpus.read_expression([0, 1, 2])
    meta = corpus.take_metadata(
        [0, 1, 2],
        columns=["dataset_id", "local_row_index", "perturb_label"],
    )
    raw = corpus.inspect_batch(
        [0, 1, 2],
        metadata_columns=["dataset_id", "perturb_label"],
    )
    for batch in corpus.loader(
        seq_len=1024,
        processing="gpu",
        metadata_columns=["dataset_id", "perturb_label"],
    ):
        ...

    # Reuse the stored sampler when no loader-local override is supplied.
    loader = corpus.loader(seq_len=1024)

    # Loader-local sampler arguments override the stored sampler for this call
    # and emit a UserWarning.
    override_loader = corpus.loader(seq_len=1024, batch_size=64)

Aggregate and federated corpora share this same public API. Loader routes use
the backend-neutral ``read_expression_flat(global_indices) -> ExpressionBatch``
contract internally while topology-specific routing stays inside ``Corpus``.
``Corpus.loader()`` keeps rich metadata in the main process by default,
treats ``size_factor`` as optional metadata, defaults to ``batch_size=128``
when no sampler has been configured, and defaults Lance workers to ``spawn``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Sequence
import warnings

import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader

from .expression import (
    ArrowIpcDatasetEntry,
    CsrMemmapShardEntry,
    DatasetEntry,
    ExpressionReader,
    HfDatasetsDatasetEntry,
    LanceDatasetEntry,
    ParquetDatasetEntry,
    ZarrDatasetEntry,
    build_expression_reader,
)
from .feature_registry import FeatureRegistry
from .gpu_pipeline import GPUSparsePipeline
from .index import MetadataIndex, _CANONICAL_OBS_TYPED_DTYPES, _normalize_canonical_obs_dtypes
from .loaders import (
    CorpusRandomBatchSampler,
    DatasetBatchSampler,
    DatasetContextBatchSampler,
    DatasetRoutingTable,
    ExpressionBatch,
    ExpressionBatchDataset,
    collate_expression_batch,
    collate_expression_batch_cpu,
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

_LOADER_METADATA_RESERVED_OVERRIDES: frozenset[str] = frozenset(
    {"local_row_index", "size_factor"}
)


# ---------------------------------------------------------------------------
# Corpus dataclass
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    """Ready-to-train corpus object and preferred user-facing runtime handle.

    Use ``set_sampler()`` to choose batch sampling, ``dataset()`` to inspect
    the backend-neutral expression dataset contract, and ``loader()`` to
    iterate over processed training batches. Aggregate and federated corpora
    share the same public API.

    Attributes
    ----------
    expression_reader : ExpressionReader
        Backend-aware flat expression reader.
    feature_registry : FeatureRegistry
        Per-dataset local→global gene ID mapping.
    metadata_index : MetadataIndex
        Polars-backed flat-schema metadata table for all cells.
    routing_table : DatasetRoutingTable
        Compact worker-safe dataset-index routing state.
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
    routing_table: DatasetRoutingTable
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
        """Build and store a metadata-index-backed sampler for ``loader()``.

        A later ``Corpus.loader(...)`` call reuses this sampler when that call
        does not provide loader-local sampler overrides.
        """
        normalized = _normalize_sampler_params(params)
        self._sampler = _build_sampler(self.metadata_index, normalized)
        self._sampler_params = normalized
        return self._sampler

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

    def inspect_batch(
        self,
        indices: torch.Tensor | np.ndarray | Sequence[int],
        *,
        metadata_columns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Read the stable raw inspection batch without touching the executor directly.

        This returns the same expression-first raw batch contract used by the
        corpus dataset/loader path: routing fields stay top-level, and any
        requested rich metadata is attached as columnar ``meta_columns``.
        """
        normalized_indices = _normalize_batch_indices(indices).tolist()
        resolved_columns = _normalize_metadata_columns(
            self.metadata_index,
            metadata_columns,
            allow_reserved=_LOADER_METADATA_RESERVED_OVERRIDES,
        )
        return _read_raw_batch(
            self.expression_reader,
            self.metadata_index,
            normalized_indices,
            metadata_columns=resolved_columns,
        )

    def _expression_dataset(self) -> ExpressionBatchDataset:
        """Build the backend-neutral expression-only dataset route."""
        return ExpressionBatchDataset(
            self.expression_reader,
            routing_table=self.routing_table,
            topology=self.topology,
            backend=self.backend,
        )

    def dataset(
        self,
        *,
        metadata_columns: Sequence[str] | None = None,
    ) -> ExpressionBatchDataset:
        """Return the dataset consumed by ``Corpus.loader()``.

        The public dataset surface is now expression-only. Use
        ``inspect_batch(...)``, ``take_metadata(...)``, or
        ``loader(metadata_columns=...)`` for metadata-rich access. This returns
        the modern ``ExpressionBatchDataset`` for custom ``DataLoader`` users.
        """
        columns = tuple(metadata_columns or ())
        if columns:
            raise ValueError(
                "Corpus.dataset() no longer accepts metadata_columns; use "
                "Corpus.inspect_batch(...), Corpus.take_metadata(...), or "
                "Corpus.loader(metadata_columns=...) instead."
            )
        return self._expression_dataset()

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
        seq_len: int | None = None,
        sampler: str | None = None,
        batch_size: int | None = None,
        drop_last: bool = True,
        seed: int = 0,
        shuffle: bool = True,
        dataset_index: int | None = None,
        context_field: str = "raw_cell_type",
        sampling_mode: str = "uniform",
        expressed_weight: float = 3.0,
        hvg_weight: float = 3.0,
    ) -> Iterator[dict[str, Any]]:
        """Yield processed sparse training batches for the requested route.

        ``processing="gpu"`` keeps DataLoader workers expression-only and runs
        ``GPUSparsePipeline.process_batch(...)`` in the main process.
        ``processing="cpu"`` runs the sparse pipeline inside CPU workers via
        ``collate_expression_batch_cpu``. Requested ``metadata_columns`` are
        attached afterward as columnar ``meta_columns`` in the main process, so
        rich metadata and optional ``size_factor`` do not need to live inside
        worker state. If no sampler has been stored and no loader-local sampler
        arguments are supplied, ``Corpus.loader(...)`` falls back to a default
        random sampler with ``batch_size=128``. If a stored sampler exists,
        loader-local sampler arguments override it for that call and emit a
        ``UserWarning``. Lance-backed loaders default to
        ``multiprocessing_context="spawn"`` unless explicitly overridden.
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
        resolved_metadata_columns = _normalize_metadata_columns(
            self.metadata_index,
            validated["metadata_columns"],
            allow_reserved=_LOADER_METADATA_RESERVED_OVERRIDES,
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
        resolved_seq_len = _resolve_loader_seq_len(seq_len=seq_len)
        _validate_processing_device(
            validated["processing"], validated["device"],
        )

        dataset_obj = self.dataset()
        pipeline = GPUSparsePipeline(
            self.feature_registry,
            seq_len=resolved_seq_len,
        )
        loader_kwargs = _build_dataloader_kwargs(
            num_workers=validated["num_workers"],
            multiprocessing_context=validated["multiprocessing_context"],
            pin_memory=validated["pin_memory"],
            persistent_workers=validated["persistent_workers"],
            prefetch_factor=validated["prefetch_factor"],
            backend=self.backend,
        )
        collate_fn = collate_expression_batch
        if validated["processing"] == "cpu":
            collate_fn = partial(
                collate_expression_batch_cpu,
                pipeline=pipeline,
                sampling_mode=sampling_mode,
                expressed_weight=expressed_weight,
                hvg_weight=hvg_weight,
            )
        data_loader = DataLoader(
            dataset_obj,
            batch_sampler=resolved_sampler,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

        def _attach_requested_metadata(
            batch: dict[str, Any],
            global_row_index: torch.Tensor | np.ndarray | Sequence[int],
        ) -> dict[str, Any]:
            if not resolved_metadata_columns:
                return batch
            batch["meta_columns"] = self.metadata_index.take(
                _normalize_batch_indices(global_row_index),
                resolved_metadata_columns,
            )
            return batch

        if validated["processing"] == "cpu":
            def _cpu_iterator() -> Iterator[dict[str, Any]]:
                for processed_batch in data_loader:
                    yield _attach_requested_metadata(
                        processed_batch,
                        processed_batch["global_row_index"],
                    )

            return _cpu_iterator()

        resolved_device = validated["device"]

        def _gpu_iterator() -> Iterator[dict[str, Any]]:
            for raw_batch in data_loader:
                processed_batch = pipeline.process_batch(
                    raw_batch,
                    device=resolved_device,
                    sampling_mode=sampling_mode,
                    expressed_weight=expressed_weight,
                    hvg_weight=hvg_weight,
                )
                yield _attach_requested_metadata(
                    processed_batch,
                    raw_batch["global_row_index"],
                )

        return _gpu_iterator()


# ---------------------------------------------------------------------------
# Backend name normalisation
# ---------------------------------------------------------------------------

_BACKEND_NORMALIZE: dict[str, str] = {
    "arrow-parquet": "parquet",
    "arrow_parquet": "parquet",
    "arrow-ipc": "arrow_ipc",
    "hf-datasets": "hf_datasets",
    "huggingface-datasets": "hf_datasets",
    "datasets": "hf_datasets",
    "lance": "lance",
    "parquet": "parquet",
    "zarr": "zarr",
    "webdataset": "webdataset",
    "csr-memmap": "csr_memmap",
    # legacy / alternate names
    "arrow_hf": "parquet",
    "arrow_ipc": "arrow_ipc",
    "hf_datasets": "hf_datasets",
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


def _coerce_optional_float32(
    values: np.ndarray | tuple | None,
) -> np.ndarray | None:
    """Convert gathered size factors to float32 or omit when absent."""
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0 or np.isnan(arr).all():
            return None
        return arr

    arr = np.asarray(values, dtype=object)
    if arr.size == 0:
        return None

    result = np.empty(arr.shape, dtype=np.float32)
    saw_value = False
    flat_values = arr.reshape(-1)
    flat_result = result.reshape(-1)
    for i, value in enumerate(flat_values):
        if value is None or str(value) == "NA":
            flat_result[i] = np.nan
            continue
        flat_result[i] = np.float32(value)
        saw_value = True

    return result if saw_value else None


def _read_raw_batch(
    expression_reader: ExpressionReader,
    metadata_index: MetadataIndex,
    global_indices: Sequence[int],
    *,
    metadata_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble the stable raw expression batch directly from owned components."""
    indices = [int(i) for i in global_indices]
    expr = expression_reader.read_expression_flat(indices)

    requested_meta = list(dict.fromkeys(metadata_columns or ()))
    gather_cols = ["dataset_index", "local_row_index"]
    if "size_factor" in metadata_index.df.columns:
        gather_cols.append("size_factor")
    gather_cols.extend(col for col in requested_meta if col not in gather_cols)
    gathered = metadata_index.gather_columns(indices, gather_cols)

    batch: dict[str, Any] = {
        "batch_size": expr.batch_size,
        "global_row_index": expr.global_row_index,
        "dataset_index": np.asarray(
            gathered.get(
                "dataset_index",
                np.zeros(expr.batch_size, dtype=np.int32),
            ),
            dtype=np.int32,
        ),
        "row_offsets": expr.row_offsets,
        "expressed_gene_indices": expr.expressed_gene_indices,
        "expression_counts": expr.expression_counts,
    }
    if "local_row_index" in gathered:
        batch["local_row_index"] = np.asarray(
            gathered["local_row_index"],
            dtype=np.int64,
        )
    size_factor = _coerce_optional_float32(gathered.get("size_factor"))
    if size_factor is not None:
        batch["size_factor"] = size_factor
    meta_columns = {
        col: gathered[col]
        for col in requested_meta
        if col in gathered
    }
    if meta_columns:
        batch["meta_columns"] = meta_columns
    return batch


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
    has_loader_local_sampler = sampler is not None or batch_size is not None
    if not has_loader_local_sampler:
        if corpus.sampler is None:
            return _build_sampler(
                corpus.metadata_index,
                {
                    "sampler": "corpus_random",
                    "batch_size": 128,
                    "drop_last": False,
                    "seed": 0,
                },
            )
        return corpus.sampler

    if corpus.sampler is not None:
        warnings.warn(
            "Corpus.loader(...) received loader-local sampler arguments while "
            "a stored sampler exists; using the loader-local sampler for this "
            "call.",
            UserWarning,
            stacklevel=2,
        )

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
        )
    )


def _normalize_metadata_columns(
    metadata_index: MetadataIndex,
    metadata_columns: Sequence[str] | None,
    *,
    allow_reserved: frozenset[str] = frozenset(),
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
        if name in _RAW_BATCH_RESERVED_KEYS and name not in allow_reserved:
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


def _normalize_take_columns(
    metadata_index: MetadataIndex,
    columns: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate columns for ``Corpus.take_metadata(...)``."""
    if columns is None:
        return tuple(metadata_index.df.columns)
    return _normalize_metadata_columns(
        metadata_index,
        columns,
        allow_reserved=_RAW_BATCH_RESERVED_KEYS,
    )


def _normalize_batch_indices(
    indices: torch.Tensor | np.ndarray | Sequence[int],
) -> np.ndarray:
    """Convert batch row indices from torch/numpy/sequence form to int64."""
    if isinstance(indices, torch.Tensor):
        return indices.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(indices, dtype=np.int64)


def _resolve_loader_seq_len(*, seq_len: int | None) -> int:
    """Resolve the sequence length for ``Corpus.loader(...)``."""
    resolved = seq_len
    if resolved is None:
        raise ValueError(
            "seq_len is required for corpus.loader(...). Pass seq_len to "
            "corpus.loader(seq_len=...)."
        )
    resolved = int(resolved)
    if resolved <= 0:
        raise ValueError("seq_len must be positive")
    return resolved


def _validate_processing_device(
    processing: str,
    device: torch.device | str | None,
) -> None:
    """Validate that the chosen device is compatible with the route."""
    if processing != "cpu" or device is None:
        return
    normalized_device = torch.device(device)
    if normalized_device.type != "cpu":
        raise ValueError(
            "processing='cpu' only supports CPU devices; pass device=None "
            "or device='cpu'."
        )


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
    use_canonical: bool = True,
) -> Corpus:
    """Load a training-ready ``Corpus`` from a corpus directory.

    Reads ``corpus-index.yaml``, locates canonical obs/var parquets via
    ``resolve_corpus_paths()``, and constructs all loader components. The
    resulting ``Corpus`` exposes the preferred corpus-level API for both
    aggregate and federated corpora:

    ``load_corpus(...) -> corpus.set_sampler(...) -> corpus.dataset() -> corpus.loader(...)``.

    Parameters
    ----------
    corpus_root : str or Path
        Path to a corpus directory containing ``corpus-index.yaml``.
    use_canonical : bool
        Whether to use canonical obs/var parquets.  Default ``True``.
        When ``True``, reads ``canonical-obs.parquet`` and
        ``canonical-var.parquet`` from ``meta/{id}/canonical_meta/``.

    Returns
    -------
    Corpus
        Fully-constructed corpus object ready for corpus-level sampling,
        inspection, and loading via ``dataset(...)``, ``loader(...)``,
        ``read_expression(...)``, ``take_metadata(...)``, and
        ``inspect_batch(...)``.

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
        if backend in {"lance", "zarr", "arrow_ipc", "hf_datasets", "parquet"}:
            entries = []
            for ds_id, _dsi, g_start, g_end in global_ranges:
                matrix_root = resolve_corpus_paths(topology, root, ds_id).matrix_root
                if backend == "lance":
                    lance_path = matrix_root / f"{ds_id}.lance"
                    if not lance_path.exists():
                        raise FileNotFoundError(
                            f"Lance file not found for dataset '{ds_id}': "
                            f"{lance_path}"
                        )
                    entries.append(
                        LanceDatasetEntry(
                            dataset_id=ds_id,
                            global_start=g_start,
                            global_end=g_end,
                            lance_path=str(lance_path),
                        )
                    )
                    continue

                if backend == "zarr":
                    row_offsets_path = matrix_root / f"{ds_id}-row-offsets.zarr"
                    indices_path = matrix_root / f"{ds_id}-indices.zarr"
                    counts_path = matrix_root / f"{ds_id}-counts.zarr"
                    if not row_offsets_path.is_dir():
                        raise FileNotFoundError(
                            f"Zarr row-offsets artifact not found for dataset '{ds_id}': "
                            f"{row_offsets_path}"
                        )
                    if not indices_path.is_dir():
                        raise FileNotFoundError(
                            f"Zarr indices artifact not found for dataset '{ds_id}': "
                            f"{indices_path}"
                        )
                    if not counts_path.is_dir():
                        raise FileNotFoundError(
                            f"Zarr counts artifact not found for dataset '{ds_id}': "
                            f"{counts_path}"
                        )
                    entries.append(
                        ZarrDatasetEntry(
                            dataset_id=ds_id,
                            global_start=g_start,
                            global_end=g_end,
                            offsets_path=str(row_offsets_path),
                            indices_path=str(indices_path),
                            counts_path=str(counts_path),
                        )
                    )
                    continue

                if backend == "arrow_ipc":
                    arrow_path = matrix_root / f"{ds_id}-cells.arrow"
                    if not arrow_path.is_file():
                        raise FileNotFoundError(
                            f"Arrow IPC file not found for dataset '{ds_id}': "
                            f"{arrow_path}"
                        )
                    entries.append(
                        ArrowIpcDatasetEntry(
                            dataset_id=ds_id,
                            global_start=g_start,
                            global_end=g_end,
                            arrow_path=str(arrow_path),
                        )
                    )
                    continue

                if backend == "parquet":
                    parquet_path = matrix_root / f"{ds_id}-cells.parquet"
                    if not parquet_path.is_file():
                        raise FileNotFoundError(
                            f"Parquet file not found for dataset '{ds_id}': "
                            f"{parquet_path}"
                        )
                    entries.append(
                        ParquetDatasetEntry(
                            dataset_id=ds_id,
                            global_start=g_start,
                            global_end=g_end,
                            parquet_path=str(parquet_path),
                        )
                    )
                    continue

                hf_dataset_path = matrix_root / f"{ds_id}-hf-dataset"
                if not hf_dataset_path.is_dir():
                    raise FileNotFoundError(
                        f"HF datasets directory not found for dataset '{ds_id}': "
                        f"{hf_dataset_path}"
                    )
                entries.append(
                    HfDatasetsDatasetEntry(
                        dataset_id=ds_id,
                        global_start=g_start,
                        global_end=g_end,
                        dataset_path=str(hf_dataset_path),
                    )
                )
            expression_reader = build_expression_reader(backend, topology, entries)
        else:
            raise ValueError(
                f"Unsupported backend '{backend}' for federated topology. "
                f"Only 'lance', 'zarr', 'arrow_ipc', 'hf_datasets', and 'parquet' are currently supported for federated."
            )
    else:
        raise ValueError(
            f"Unknown topology '{topology}'. "
            f"Expected 'aggregate' or 'federated'."
        )

    routing_table = _build_dataset_routing_table(metadata_index, entries)

    # ------------------------------------------------------------------
    # 5. Build FeatureRegistry from canonical var parquets
    # ------------------------------------------------------------------
    var_path_map: dict[str, str] = {
        ds_id: str(p) for ds_id, p in canonical_var_paths.items()
    }
    feature_registry = FeatureRegistry.from_canonical_var_parquets(
        var_path_map
    )

    # ------------------------------------------------------------------
    # 6. Return Corpus
    # ------------------------------------------------------------------
    return Corpus(
        expression_reader=expression_reader,
        feature_registry=feature_registry,
        metadata_index=metadata_index,
        routing_table=routing_table,
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


def _build_dataset_routing_table(
    metadata_index: MetadataIndex,
    dataset_entries: Sequence[DatasetEntry],
) -> DatasetRoutingTable:
    """Build and validate the compact dataset-index routing table."""
    ordered_entries = sorted(dataset_entries, key=lambda entry: entry.global_start)
    if not ordered_entries:
        raise ValueError("Corpus has no dataset entries for expression loading")

    dataset_starts = np.asarray(
        [entry.global_start for entry in ordered_entries],
        dtype=np.int64,
    )
    dataset_stops = np.asarray(
        [entry.global_end for entry in ordered_entries],
        dtype=np.int64,
    )
    if dataset_stops[-1] != len(metadata_index):
        raise ValueError(
            "ExpressionBatchDataset requires routing entries to cover every metadata row"
        )

    probe_indices = dataset_starts.tolist() + (dataset_stops - 1).tolist()
    probes = metadata_index.take(
        probe_indices,
        ("dataset_id", "dataset_index", "local_row_index"),
    )

    start_dataset_ids = tuple(str(value) for value in probes["dataset_id"][: len(ordered_entries)])
    end_dataset_ids = tuple(str(value) for value in probes["dataset_id"][len(ordered_entries) :])
    dataset_index_probe = np.asarray(probes["dataset_index"], dtype=np.int32)
    local_row_probe = np.asarray(probes["local_row_index"], dtype=np.int64)
    n_entries = len(ordered_entries)
    dataset_indices = dataset_index_probe[:n_entries]
    end_dataset_indices = dataset_index_probe[n_entries:]
    local_row_starts = local_row_probe[:n_entries]
    local_row_ends = local_row_probe[n_entries:]

    if start_dataset_ids != end_dataset_ids or not np.array_equal(dataset_indices, end_dataset_indices):
        raise ValueError(
            "ExpressionBatchDataset requires each routing entry to stay within a single dataset"
        )

    expected_local_row_ends = local_row_starts + (dataset_stops - dataset_starts - 1)
    if not np.array_equal(local_row_ends, expected_local_row_ends):
        raise ValueError(
            "ExpressionBatchDataset requires contiguous local row indices within each routing entry"
        )

    size_factor = metadata_index.get_column("size_factor")
    if size_factor is not None and not isinstance(size_factor, np.ndarray):
        raise TypeError("size_factor column must be numeric when present")

    return DatasetRoutingTable(
        dataset_starts=dataset_starts,
        dataset_stops=dataset_stops,
        dataset_indices=dataset_indices,
        size_factor=size_factor,
        dataset_ids=start_dataset_ids,
    )


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

    for ds_id, ds_index, g_start, g_end in datasets_info:
        obs_path = obs_paths[ds_id]
        df = _normalize_canonical_obs_dtypes(pl.read_parquet(str(obs_path)))
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
    structural = [
        "global_row_index", "cell_id", "dataset_id",
        "dataset_index", "local_row_index", "size_factor",
    ]
    # Ensure all structural columns exist (fill missing with null)
    for col in structural:
        if col not in combined.columns:
            fill_dtype = _CANONICAL_OBS_TYPED_DTYPES.get(col, pl.Utf8)
            combined = combined.with_columns(
                pl.lit(None, dtype=fill_dtype).alias(col)
            )

    combined = _normalize_canonical_obs_dtypes(combined)

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
