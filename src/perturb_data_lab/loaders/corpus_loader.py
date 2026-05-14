"""Corpus factory and modern composable corpus runtime API.

``load_corpus()`` reconstructs a training-ready ``Corpus`` from a corpus
directory using canonical metadata as the source of truth. Slim main supports
only Lance and Zarr corpora in aggregate or federated topology.

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

import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
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
from .gene_tokenizer import GeneTokenizer
from .gpu_pipeline import GPUSparsePipeline
from .index import (
    MetadataIndex,
    _CANONICAL_OBS_CONTENT_COLUMNS,
    _CANONICAL_OBS_STRUCTURAL_COLUMNS,
    _CANONICAL_OBS_TYPED_DTYPES,
    _load_canonical_obs_frame,
    _normalize_canonical_obs_dtypes,
)
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
from ..materializers.models import MaterializationManifest
from ..materializers.paths import resolve_corpus_paths

__all__ = [
    "Corpus",
    "ObsSelection",
    "load_corpus",
    "select_obs_indices",
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

_TO_ANNDATA_REQUIRED_OBS_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "dataset_id",
    "dataset_index",
    "global_row_index",
    "local_row_index",
)

_TO_ANNDATA_REQUIRED_VAR_COLUMNS: tuple[str, ...] = (
    "origin_index",
    "gene_id",
    "canonical_gene_id",
    "global_id",
)

_TO_ANNDATA_READ_CHUNK_SIZE: int = 2048

_OBS_SELECTION_ALLOWED_RESERVED_COLUMNS: frozenset[str] = frozenset(
    {
        "global_row_index",
        "dataset_index",
        "local_row_index",
        "size_factor",
    }
)


# ---------------------------------------------------------------------------
# Corpus dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObsSelection:
    """Structured deterministic observation selection result."""

    dataset_id: str
    dataset_index: int
    strategy: str
    seed: int
    stratify_by: tuple[str, ...]
    max_cells: int | None
    min_per_group: int | None
    max_per_group: int | None
    drop_null_groups: bool
    candidate_count: int
    selected_global_row_indices: tuple[int, ...]
    selected_local_row_indices: tuple[int, ...]
    group_counts: tuple[dict[str, Any], ...]
    dropped_or_underfilled_groups: tuple[dict[str, Any], ...]

    @property
    def row_indices(self) -> tuple[int, ...]:
        """Alias the selected corpus-global row indices."""
        return self.selected_global_row_indices

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the selection."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_index": int(self.dataset_index),
            "strategy": self.strategy,
            "seed": int(self.seed),
            "stratify_by": list(self.stratify_by),
            "max_cells": self.max_cells,
            "min_per_group": self.min_per_group,
            "max_per_group": self.max_per_group,
            "drop_null_groups": bool(self.drop_null_groups),
            "candidate_count": int(self.candidate_count),
            "selected_count": len(self.selected_global_row_indices),
            "selected_global_row_indices": list(self.selected_global_row_indices),
            "selected_local_row_indices": list(self.selected_local_row_indices),
            "group_counts": [
                _json_safe_mapping(record) for record in self.group_counts
            ],
            "dropped_or_underfilled_groups": [
                _json_safe_mapping(record)
                for record in self.dropped_or_underfilled_groups
            ],
        }

    def write_provenance(self, output_dir: str | Path) -> dict[str, Path]:
        """Write selection provenance as JSON summary plus row parquet."""
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        summary_path = root / "selection-summary.json"
        summary_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        rows_path = root / "selected-rows.parquet"
        pl.DataFrame(
            {
                "dataset_id": [self.dataset_id] * len(self.selected_global_row_indices),
                "dataset_index": [int(self.dataset_index)] * len(self.selected_global_row_indices),
                "strategy": [self.strategy] * len(self.selected_global_row_indices),
                "seed": [int(self.seed)] * len(self.selected_global_row_indices),
                "global_row_index": list(self.selected_global_row_indices),
                "local_row_index": list(self.selected_local_row_indices),
            }
        ).write_parquet(rows_path)
        return {
            "summary_path": summary_path,
            "rows_path": rows_path,
        }


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
        does not provide loader-local sampler overrides. Optional
        ``row_indices`` restricts sampling to a validated set of unique
        corpus-global row indices.
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

    def to_anndata(
        self,
        *,
        dataset_id: str,
        row_indices: Sequence[int] | np.ndarray | None = None,
        obs_columns: Sequence[str] | None = None,
        var_columns: Sequence[str] | None = None,
        dry_run: bool = False,
        max_memory_bytes: int | None = None,
        on_exceed: str = "raise",
    ) -> ad.AnnData | dict[str, Any]:
        """Materialize one dataset as a counts-only in-memory ``AnnData``.

        ``row_indices`` uses corpus-global row-index semantics and must stay
        within the requested ``dataset_id``. When ``dry_run=True``, returns a
        preflight estimate without constructing ``adata.X``.
        """
        prepared = _prepare_to_anndata_inputs(
            self,
            dataset_id=dataset_id,
            row_indices=row_indices,
            obs_columns=obs_columns,
            var_columns=var_columns,
            max_memory_bytes=max_memory_bytes,
            on_exceed=on_exceed,
        )
        estimate = prepared["estimate"]
        if dry_run:
            return estimate

        if estimate["memory_limit_exceeded"]:
            message = str(estimate["memory_guard_message"])
            if on_exceed == "raise":
                raise MemoryError(message)
            warnings.warn(message, UserWarning, stacklevel=2)

        matrix = _materialize_csr_matrix(
            self.expression_reader,
            prepared["selected_row_indices"],
            n_vars=prepared["n_vars"],
            chunk_size=_TO_ANNDATA_READ_CHUNK_SIZE,
        )
        obs = _build_pandas_frame(
            prepared["obs_data"],
            index_column="cell_id",
        )
        var = _build_pandas_frame(
            prepared["var_data"],
            index_column=_select_var_index_column(prepared["var_data"]),
        )
        return ad.AnnData(X=matrix, obs=obs, var=var)

    def select_obs_indices(
        self,
        *,
        dataset_id: str,
        strategy: str = "all",
        row_indices: Sequence[int] | np.ndarray | None = None,
        max_cells: int | None = None,
        stratify_by: Sequence[str] | None = None,
        min_per_group: int | None = None,
        max_per_group: int | None = None,
        seed: int = 0,
        drop_null_groups: bool = False,
    ) -> ObsSelection:
        """Select deterministic corpus-global observation indices for one dataset."""
        return _select_obs_indices(
            self,
            dataset_id=dataset_id,
            strategy=strategy,
            row_indices=row_indices,
            max_cells=max_cells,
            stratify_by=stratify_by,
            min_per_group=min_per_group,
            max_per_group=max_per_group,
            seed=seed,
            drop_null_groups=drop_null_groups,
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
        hvg_top_k: int | None = None,
        row_indices: Sequence[int] | np.ndarray | None = None,
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
        ``UserWarning``. Optional ``row_indices`` restricts the sampler to a
        validated subset of corpus-global rows without rebuilding the metadata
        index. Lance-backed loaders default to
        ``multiprocessing_context="spawn"`` unless explicitly overridden.

        ``sampling_mode="hvg"`` optionally accepts ``hvg_top_k`` to apply a
        runtime top-k threshold from canonical per-dataset ``hvg.parquet``
        rankings without rematerializing the corpus.
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
            row_indices=row_indices,
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
                hvg_top_k=hvg_top_k,
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
                    hvg_top_k=hvg_top_k,
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

    row_indices = params.get("row_indices")
    if row_indices is not None:
        row_indices_arr = np.asarray(row_indices)
        if row_indices_arr.ndim != 1:
            raise ValueError(
                "row_indices must be a 1-D sequence of corpus-global row indices"
            )
        normalized["row_indices"] = tuple(row_indices_arr.tolist())

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
            row_indices=params.get("row_indices"),
            drop_last=params["drop_last"],
            seed=params["seed"],
        )
    if kind == "dataset":
        return DatasetBatchSampler(
            metadata_index=metadata_index,
            dataset_index=params["dataset_index"],
            batch_size=params["batch_size"],
            row_indices=params.get("row_indices"),
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
            row_indices=params.get("row_indices"),
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
    row_indices: Sequence[int] | None,
) -> Any:
    """Resolve the sampler to use for one loader invocation."""
    has_loader_local_sampler = (
        sampler is not None or batch_size is not None or row_indices is not None
    )
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

    if (
        row_indices is not None
        and sampler is None
        and batch_size is None
        and corpus.sampler is None
    ):
        return _build_sampler(
            corpus.metadata_index,
            {
                "sampler": "corpus_random",
                "batch_size": 128,
                "drop_last": False,
                "seed": 0,
                "row_indices": row_indices,
            },
        )

    if (
        row_indices is not None
        and sampler is None
        and batch_size is None
        and corpus.sampler is not None
    ):
        warnings.warn(
            "Corpus.loader(...) received loader-local sampler arguments while "
            "a stored sampler exists; using the loader-local sampler for this "
            "call.",
            UserWarning,
            stacklevel=2,
        )
        return _build_sampler(
            corpus.metadata_index,
            {
                **corpus.sampler_params,
                "row_indices": row_indices,
            },
        )

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
                "row_indices": row_indices,
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
    extra_metadata_columns: Sequence[str] | None = None,
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
    extra_metadata_columns : sequence of str, optional
        Additional canonical-obs parquet columns to load into
        ``metadata_index`` beyond the default canonical/core projection.

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
        elif backend == "zarr":
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
            entries = [
                DatasetEntry(
                    dataset_id=ds_id,
                    global_start=g_start,
                    global_end=g_end,
                )
                for ds_id, _dsi, g_start, g_end in global_ranges
            ]
            expression_reader = build_expression_reader(
                backend,
                topology,
                entries,
                offsets_path=str(row_offsets_path),
                indices_path=str(indices_path),
                counts_path=str(counts_path),
            )
        else:
            raise ValueError(
                f"Unsupported backend '{backend}' for aggregate topology. "
                "Only 'lance' and 'zarr' are currently "
                "supported for aggregate."
            )
    elif topology == "federated":
        if backend in {"lance", "zarr"}:
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

            expression_reader = build_expression_reader(backend, topology, entries)
        else:
            raise ValueError(
                f"Unsupported backend '{backend}' for federated topology. "
                f"Only 'lance' and 'zarr' are currently supported for federated."
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
        routing_table=routing_table,
        dataset_entries=list(entries),
        dataset_index_by_id={ds_id: ds_index for ds_id, ds_index, *_ in global_ranges},
        topology=topology,
        backend=backend,
        corpus_root=root,
    )


def select_obs_indices(
    corpus: Corpus,
    **kwargs: Any,
) -> ObsSelection:
    """Public function-style alias around ``Corpus.select_obs_indices(...)``."""
    return corpus.select_obs_indices(**kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return the parsed dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class _SelectionGroup:
    group_key: str
    group_values: dict[str, Any]
    candidate_positions: np.ndarray
    available_count: int
    is_null_group: bool


def _select_obs_indices(
    corpus: Corpus,
    *,
    dataset_id: str,
    strategy: str,
    row_indices: Sequence[int] | np.ndarray | None,
    max_cells: int | None,
    stratify_by: Sequence[str] | None,
    min_per_group: int | None,
    max_per_group: int | None,
    seed: int,
    drop_null_groups: bool,
) -> ObsSelection:
    resolved_strategy = _normalize_obs_selection_strategy(strategy)
    resolved_seed = int(seed)
    resolved_max_cells = _normalize_optional_positive_int(max_cells, label="max_cells")
    resolved_min_per_group = _normalize_optional_positive_int(
        min_per_group,
        label="min_per_group",
    )
    resolved_max_per_group = _normalize_optional_positive_int(
        max_per_group,
        label="max_per_group",
    )
    if (
        resolved_min_per_group is not None
        and resolved_max_per_group is not None
        and resolved_max_per_group < resolved_min_per_group
    ):
        raise ValueError("max_per_group must be >= min_per_group when both are provided")

    group_columns = _normalize_selection_group_columns(
        corpus.metadata_index,
        stratify_by,
    )
    _validate_obs_selection_request(
        strategy=resolved_strategy,
        max_cells=resolved_max_cells,
        group_columns=group_columns,
        min_per_group=resolved_min_per_group,
        max_per_group=resolved_max_per_group,
    )

    dataset_entry = _resolve_dataset_entry(corpus.dataset_entries, dataset_id)
    candidate_row_indices = _normalize_dataset_row_indices(dataset_entry, row_indices)
    candidate_frame = _build_obs_selection_frame(
        corpus,
        row_indices=candidate_row_indices,
        group_columns=group_columns,
    )
    groups = _build_obs_selection_groups(candidate_frame, group_columns)

    active_groups: list[_SelectionGroup] = []
    group_records: list[dict[str, Any]] = []
    flagged_groups: list[dict[str, Any]] = []
    for group in groups:
        if group.is_null_group and drop_null_groups:
            flagged_groups.append(
                _build_group_status_record(
                    group,
                    selected_count=0,
                    reason="null-group-dropped",
                    min_per_group=resolved_min_per_group,
                )
            )
        else:
            active_groups.append(group)

    if not active_groups:
        raise ValueError("selection is empty after dropping null groups")

    target_counts = _resolve_obs_selection_targets(
        groups=active_groups,
        strategy=resolved_strategy,
        max_cells=resolved_max_cells,
        min_per_group=resolved_min_per_group,
        max_per_group=resolved_max_per_group,
    )

    rng = np.random.default_rng(resolved_seed)
    sampled_position_parts: list[np.ndarray] = []
    selected_counts_by_key: dict[str, int] = {}
    for group, target_count in zip(active_groups, target_counts, strict=True):
        sampled_positions = _sample_candidate_positions(
            group.candidate_positions,
            target_count=int(target_count),
            rng=rng,
        )
        sampled_position_parts.append(sampled_positions)
        selected_count = int(sampled_positions.size)
        selected_counts_by_key[group.group_key] = selected_count
        if (
            resolved_min_per_group is not None
            and group.available_count < resolved_min_per_group
        ):
            flagged_groups.append(
                _build_group_status_record(
                    group,
                    selected_count=selected_count,
                    reason="underfilled-min-per-group",
                    min_per_group=resolved_min_per_group,
                )
            )

    selected_positions = (
        np.concatenate(sampled_position_parts)
        if sampled_position_parts
        else np.array([], dtype=np.int64)
    )
    if selected_positions.size == 0:
        raise ValueError("selection produced zero rows; adjust max_cells or grouping parameters")
    selected_positions.sort()

    for group in groups:
        group_records.append(
            {
                "group_key": group.group_key,
                "group_values": dict(group.group_values),
                "available_count": int(group.available_count),
                "selected_count": int(selected_counts_by_key.get(group.group_key, 0)),
            }
        )

    global_row_index = candidate_frame["global_row_index"].to_numpy()
    local_row_index = candidate_frame["local_row_index"].to_numpy()
    dataset_index = int(corpus.dataset_index_by_id[dataset_id])
    return ObsSelection(
        dataset_id=dataset_id,
        dataset_index=dataset_index,
        strategy=resolved_strategy,
        seed=resolved_seed,
        stratify_by=group_columns,
        max_cells=resolved_max_cells,
        min_per_group=resolved_min_per_group,
        max_per_group=resolved_max_per_group,
        drop_null_groups=bool(drop_null_groups),
        candidate_count=int(candidate_row_indices.size),
        selected_global_row_indices=tuple(
            int(value) for value in global_row_index[selected_positions].tolist()
        ),
        selected_local_row_indices=tuple(
            int(value) for value in local_row_index[selected_positions].tolist()
        ),
        group_counts=tuple(group_records),
        dropped_or_underfilled_groups=tuple(flagged_groups),
    )


def _normalize_obs_selection_strategy(strategy: str) -> str:
    normalized = str(strategy).strip().lower().replace("_", "-")
    aliases = {
        "all": "all",
        "pass-through": "all",
        "passthrough": "all",
        "random": "random",
        "stratified": "stratified",
        "balanced": "balanced",
    }
    if normalized not in aliases:
        raise ValueError(
            "strategy must be one of 'all', 'random', 'stratified', or 'balanced'"
        )
    return aliases[normalized]


def _normalize_optional_positive_int(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{label} must be positive when provided")
    return normalized


def _normalize_selection_group_columns(
    metadata_index: MetadataIndex,
    stratify_by: Sequence[str] | None,
) -> tuple[str, ...]:
    if stratify_by is None:
        return ()
    return _normalize_metadata_columns(
        metadata_index,
        stratify_by,
        allow_reserved=_OBS_SELECTION_ALLOWED_RESERVED_COLUMNS,
    )


def _validate_obs_selection_request(
    *,
    strategy: str,
    max_cells: int | None,
    group_columns: tuple[str, ...],
    min_per_group: int | None,
    max_per_group: int | None,
) -> None:
    if strategy in {"all", "random"} and group_columns:
        raise ValueError(
            f"strategy='{strategy}' does not accept stratify_by; use 'stratified' or 'balanced'"
        )
    if strategy in {"stratified", "balanced"} and not group_columns:
        raise ValueError(
            f"strategy='{strategy}' requires at least one stratify_by column"
        )
    if strategy == "all" and max_cells is not None:
        raise ValueError(
            "strategy='all' cannot trim rows; use 'random', 'stratified', or 'balanced'"
        )
    if strategy == "random" and (min_per_group is not None or max_per_group is not None):
        raise ValueError(
            "strategy='random' does not accept min_per_group or max_per_group"
        )
    if strategy == "all" and (min_per_group is not None or max_per_group is not None):
        raise ValueError(
            "strategy='all' does not accept min_per_group or max_per_group"
        )


def _build_obs_selection_frame(
    corpus: Corpus,
    *,
    row_indices: np.ndarray,
    group_columns: tuple[str, ...],
) -> pl.DataFrame:
    columns = [
        "global_row_index",
        "dataset_id",
        "dataset_index",
        "local_row_index",
        *group_columns,
    ]
    data = corpus.metadata_index.take(row_indices, columns=columns)
    frame = pl.DataFrame(
        {
            column_name: list(values) if isinstance(values, tuple) else values
            for column_name, values in data.items()
        }
    )
    return frame.with_columns(
        pl.Series("_candidate_position", np.arange(len(frame), dtype=np.int64))
    )


def _build_obs_selection_groups(
    frame: pl.DataFrame,
    group_columns: tuple[str, ...],
) -> list[_SelectionGroup]:
    if not group_columns:
        positions = frame["_candidate_position"].to_numpy().astype(np.int64, copy=False)
        return [
            _SelectionGroup(
                group_key="__all__",
                group_values={},
                candidate_positions=positions,
                available_count=int(positions.size),
                is_null_group=False,
            )
        ]

    grouped = frame.group_by(list(group_columns), maintain_order=True).agg(
        pl.len().alias("available_count"),
        pl.col("_candidate_position").alias("candidate_positions"),
    )
    groups: list[_SelectionGroup] = []
    for row in grouped.iter_rows(named=True):
        group_values = {
            column_name: _json_safe_value(row[column_name])
            for column_name in group_columns
        }
        groups.append(
            _SelectionGroup(
                group_key=_format_group_key(group_values),
                group_values=group_values,
                candidate_positions=np.asarray(
                    row["candidate_positions"],
                    dtype=np.int64,
                ),
                available_count=int(row["available_count"]),
                is_null_group=any(value is None for value in group_values.values()),
            )
        )
    return groups


def _resolve_obs_selection_targets(
    *,
    groups: Sequence[_SelectionGroup],
    strategy: str,
    max_cells: int | None,
    min_per_group: int | None,
    max_per_group: int | None,
) -> np.ndarray:
    available = np.asarray([group.available_count for group in groups], dtype=np.int64)
    capacity = available.copy()
    if max_per_group is not None:
        capacity = np.minimum(capacity, max_per_group)
    total_capacity = int(capacity.sum())

    if strategy == "all":
        return available.copy()
    if strategy == "random":
        total_available = int(available.sum())
        total_target = total_available if max_cells is None else min(max_cells, total_available)
        return np.asarray([total_target], dtype=np.int64)

    baseline = np.zeros(len(groups), dtype=np.int64)
    if min_per_group is not None:
        baseline = np.minimum(capacity, min_per_group)
    if max_cells is not None and int(baseline.sum()) > max_cells:
        raise ValueError(
            "max_cells is too small to satisfy the requested min_per_group across groups"
        )

    total_target = total_capacity if max_cells is None else min(max_cells, total_capacity)
    if strategy == "balanced":
        return _allocate_balanced_targets(capacity, baseline, total_target)
    return _allocate_stratified_targets(available, capacity, baseline, total_target)


def _allocate_balanced_targets(
    capacity: np.ndarray,
    baseline: np.ndarray,
    total_target: int,
) -> np.ndarray:
    targets = baseline.copy()
    if targets.sum() >= total_target:
        return targets
    if np.array_equal(capacity, baseline):
        return capacity.copy()

    residual = int(total_target - targets.sum())
    active = np.where(targets < capacity)[0]
    while residual > 0 and active.size > 0:
        per_group = max(1, residual // active.size)
        increment = np.minimum(capacity[active] - targets[active], per_group)
        targets[active] += increment
        used = int(increment.sum())
        residual -= used
        if residual <= 0:
            break
        active = np.where(targets < capacity)[0]
        if used == 0 and active.size > 0:
            for idx in active:
                if residual == 0:
                    break
                targets[idx] += 1
                residual -= 1
            break
    return targets


def _allocate_stratified_targets(
    available: np.ndarray,
    capacity: np.ndarray,
    baseline: np.ndarray,
    total_target: int,
) -> np.ndarray:
    targets = baseline.copy()
    if targets.sum() >= total_target:
        return targets
    if np.array_equal(capacity, baseline):
        return capacity.copy()

    residual = int(total_target - targets.sum())
    extra_capacity = capacity - baseline
    weights = np.maximum(available - baseline, 0)
    if int(weights.sum()) == 0:
        weights = extra_capacity.copy()
    proportional = residual * (weights / weights.sum())
    floor_allocation = np.minimum(
        np.floor(proportional).astype(np.int64),
        extra_capacity,
    )
    targets += floor_allocation
    residual -= int(floor_allocation.sum())
    if residual <= 0:
        return targets

    remainders = proportional - np.floor(proportional)
    candidate_order = np.argsort(-remainders, kind="stable")
    for idx in candidate_order:
        if residual == 0:
            break
        if targets[idx] >= capacity[idx]:
            continue
        targets[idx] += 1
        residual -= 1
    if residual <= 0:
        return targets

    for idx in np.where(targets < capacity)[0]:
        if residual == 0:
            break
        room = int(capacity[idx] - targets[idx])
        if room <= 0:
            continue
        step = min(room, residual)
        targets[idx] += step
        residual -= step
    return targets


def _sample_candidate_positions(
    candidate_positions: np.ndarray,
    *,
    target_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_count <= 0:
        return np.array([], dtype=np.int64)
    if target_count >= candidate_positions.size:
        return np.sort(candidate_positions.astype(np.int64, copy=True))
    sampled = rng.choice(candidate_positions, size=target_count, replace=False)
    sampled.sort()
    return sampled.astype(np.int64, copy=False)


def _build_group_status_record(
    group: _SelectionGroup,
    *,
    selected_count: int,
    reason: str,
    min_per_group: int | None,
) -> dict[str, Any]:
    return {
        "group_key": group.group_key,
        "group_values": dict(group.group_values),
        "available_count": int(group.available_count),
        "selected_count": int(selected_count),
        "reason": reason,
        "min_per_group": min_per_group,
    }


def _format_group_key(group_values: dict[str, Any]) -> str:
    if not group_values:
        return "__all__"
    return json.dumps(group_values, sort_keys=True)


def _json_safe_mapping(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe_value(value) for key, value in record.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    return value


def _prepare_to_anndata_inputs(
    corpus: Corpus,
    *,
    dataset_id: str,
    row_indices: Sequence[int] | np.ndarray | None,
    obs_columns: Sequence[str] | None,
    var_columns: Sequence[str] | None,
    max_memory_bytes: int | None,
    on_exceed: str,
) -> dict[str, Any]:
    if on_exceed not in {"raise", "warn"}:
        raise ValueError("on_exceed must be 'raise' or 'warn'")
    if max_memory_bytes is not None and int(max_memory_bytes) <= 0:
        raise ValueError("max_memory_bytes must be positive when provided")

    dataset_entry = _resolve_dataset_entry(corpus.dataset_entries, dataset_id)
    selected_row_indices = _normalize_dataset_row_indices(
        dataset_entry,
        row_indices,
    )
    resolved_obs_columns = _normalize_selected_columns(
        available_columns=corpus.metadata_index.df.columns,
        requested_columns=obs_columns,
        required_columns=_TO_ANNDATA_REQUIRED_OBS_COLUMNS,
        label="obs_columns",
    )
    obs_data = corpus.metadata_index.take(
        selected_row_indices,
        columns=resolved_obs_columns,
    )

    var_frame = _load_dataset_var_frame(corpus, dataset_id)
    resolved_var_columns = _normalize_selected_columns(
        available_columns=var_frame.columns,
        requested_columns=var_columns,
        required_columns=_TO_ANNDATA_REQUIRED_VAR_COLUMNS,
        label="var_columns",
    )
    var_data = _polars_frame_to_columnar(var_frame.select(resolved_var_columns))

    nnz = _count_selected_nnz(
        corpus.expression_reader,
        selected_row_indices,
        chunk_size=_TO_ANNDATA_READ_CHUNK_SIZE,
    )
    csr_memory_bytes = {
        "data": int(nnz * np.dtype(np.int32).itemsize),
        "indices": int(nnz * np.dtype(np.int32).itemsize),
        "indptr": int((selected_row_indices.size + 1) * np.dtype(np.int64).itemsize),
    }
    csr_memory_bytes["total"] = int(
        csr_memory_bytes["data"]
        + csr_memory_bytes["indices"]
        + csr_memory_bytes["indptr"]
    )
    obs_memory_bytes = _estimate_columnar_memory_bytes(obs_data)
    var_memory_bytes = _estimate_columnar_memory_bytes(var_data)
    estimated_total_memory_bytes = int(
        csr_memory_bytes["total"] + obs_memory_bytes + var_memory_bytes
    )

    limit = None if max_memory_bytes is None else int(max_memory_bytes)
    exceeds_limit = limit is not None and estimated_total_memory_bytes > limit
    guard_message = None
    if exceeds_limit:
        guard_message = (
            f"Estimated AnnData materialization for dataset '{dataset_id}' "
            f"requires {estimated_total_memory_bytes} bytes "
            f"(csr={csr_memory_bytes['total']}, obs≈{obs_memory_bytes}, var≈{var_memory_bytes}), "
            f"which exceeds max_memory_bytes={limit}."
        )

    estimate = {
        "dataset_id": dataset_id,
        "n_obs": int(selected_row_indices.size),
        "n_vars": int(var_frame.height),
        "nnz": int(nnz),
        "csr_memory_bytes": csr_memory_bytes,
        "obs_memory_bytes": int(obs_memory_bytes),
        "var_memory_bytes": int(var_memory_bytes),
        "estimated_total_memory_bytes": estimated_total_memory_bytes,
        "selected_row_index_summary": {
            "source": "dataset-full" if row_indices is None else "user-specified",
            "selection_size": int(selected_row_indices.size),
            "min_global_row_index": int(selected_row_indices.min()),
            "max_global_row_index": int(selected_row_indices.max()),
            "is_contiguous": bool(
                np.array_equal(
                    selected_row_indices,
                    np.arange(
                        int(selected_row_indices[0]),
                        int(selected_row_indices[0]) + selected_row_indices.size,
                        dtype=np.int64,
                    ),
                )
            ),
        },
        "obs_columns": tuple(resolved_obs_columns),
        "var_columns": tuple(resolved_var_columns),
        "memory_limit_bytes": limit,
        "memory_limit_exceeded": bool(exceeds_limit),
        "memory_guard_action": on_exceed if exceeds_limit else "none",
        "memory_guard_message": guard_message,
    }
    return {
        "estimate": estimate,
        "selected_row_indices": selected_row_indices,
        "obs_data": obs_data,
        "var_data": var_data,
        "n_vars": int(var_frame.height),
    }


def _resolve_dataset_entry(
    dataset_entries: Sequence[DatasetEntry],
    dataset_id: str,
) -> DatasetEntry:
    for entry in dataset_entries:
        if entry.dataset_id == dataset_id:
            return entry
    available = ", ".join(sorted(entry.dataset_id for entry in dataset_entries))
    raise KeyError(
        f"dataset_id '{dataset_id}' not found; available datasets: {available}"
    )


def _normalize_dataset_row_indices(
    dataset_entry: DatasetEntry,
    row_indices: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    if row_indices is None:
        return np.arange(
            dataset_entry.global_start,
            dataset_entry.global_end,
            dtype=np.int64,
        )

    raw = np.asarray(row_indices)
    if raw.ndim != 1:
        raise ValueError(
            "row_indices must be a 1-D sequence of corpus-global row indices"
        )
    if raw.size == 0:
        raise ValueError("row_indices must contain at least one corpus-global row index")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError(
            "row_indices must contain only integer corpus-global row indices"
        )

    normalized = raw.astype(np.int64, copy=False)
    if np.unique(normalized).size != normalized.size:
        raise ValueError("row_indices must be unique corpus-global row indices")
    if np.any(normalized < dataset_entry.global_start) or np.any(
        normalized >= dataset_entry.global_end
    ):
        raise IndexError(
            f"row_indices must stay within dataset '{dataset_entry.dataset_id}' "
            f"global range [{dataset_entry.global_start}, {dataset_entry.global_end})"
        )
    return normalized.copy()


def _normalize_selected_columns(
    *,
    available_columns: Sequence[str],
    requested_columns: Sequence[str] | None,
    required_columns: Sequence[str],
    label: str,
) -> list[str]:
    available = set(available_columns)
    if requested_columns is None:
        requested = list(available_columns)
    else:
        requested = [str(column) for column in requested_columns]
        missing = sorted({column for column in requested if column not in available})
        if missing:
            raise ValueError(f"{label} contains unknown columns: {missing}")

    resolved: list[str] = []
    for column in (*required_columns, *requested):
        if column in available and column not in resolved:
            resolved.append(column)
    return resolved


def _load_dataset_var_frame(corpus: Corpus, dataset_id: str) -> pl.DataFrame:
    paths = resolve_corpus_paths(corpus.topology, corpus.corpus_root, dataset_id)
    var_path = paths.canonical_meta_root / "canonical-var.parquet"
    if not var_path.exists():
        raise FileNotFoundError(
            f"canonical-var.parquet not found for dataset '{dataset_id}' at {var_path}"
        )
    frame = pl.read_parquet(str(var_path))
    expressions: list[pl.Expr] = []
    if "origin_index" in frame.columns and frame["origin_index"].dtype == pl.Utf8:
        expressions.append(pl.col("origin_index").cast(pl.Int64))
    if "global_id" in frame.columns and frame["global_id"].dtype == pl.Utf8:
        expressions.append(pl.col("global_id").cast(pl.Int64))
    if expressions:
        frame = frame.with_columns(expressions)
    if "origin_index" in frame.columns:
        frame = frame.sort("origin_index")
    return frame


def _polars_frame_to_columnar(frame: pl.DataFrame) -> dict[str, np.ndarray | tuple]:
    result: dict[str, np.ndarray | tuple] = {}
    for column_name in frame.columns:
        dtype = frame[column_name].dtype
        if dtype in MetadataIndex._NUMERIC_DTYPES:
            result[column_name] = frame[column_name].to_numpy()
            continue
        result[column_name] = tuple(frame[column_name].to_list())
    return result


def _count_selected_nnz(
    expression_reader: ExpressionReader,
    row_indices: np.ndarray,
    *,
    chunk_size: int,
) -> int:
    total_nnz = 0
    for chunk in _chunk_global_indices(row_indices, chunk_size=chunk_size):
        batch = expression_reader.read_expression_flat(chunk.tolist())
        total_nnz += int(batch.row_offsets[-1])
    return total_nnz


def _chunk_global_indices(
    row_indices: np.ndarray,
    *,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    for start in range(0, row_indices.size, chunk_size):
        yield row_indices[start : start + chunk_size]


def _materialize_csr_matrix(
    expression_reader: ExpressionReader,
    row_indices: np.ndarray,
    *,
    n_vars: int,
    chunk_size: int,
) -> sparse.csr_matrix:
    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr_parts: list[np.ndarray] = [np.array([0], dtype=np.int64)]
    nnz_offset = 0

    for chunk in _chunk_global_indices(row_indices, chunk_size=chunk_size):
        batch = expression_reader.read_expression_flat(chunk.tolist())
        data = np.asarray(batch.expression_counts, dtype=np.int32)
        indices = np.asarray(batch.expressed_gene_indices, dtype=np.int32)
        if data.size:
            data_parts.append(data)
            index_parts.append(indices)
        indptr_parts.append(np.asarray(batch.row_offsets[1:], dtype=np.int64) + nnz_offset)
        nnz_offset += int(batch.row_offsets[-1])

    matrix_data = (
        np.concatenate(data_parts) if data_parts else np.array([], dtype=np.int32)
    )
    matrix_indices = (
        np.concatenate(index_parts) if index_parts else np.array([], dtype=np.int32)
    )
    matrix_indptr = np.concatenate(indptr_parts)
    return sparse.csr_matrix(
        (matrix_data, matrix_indices, matrix_indptr),
        shape=(int(row_indices.size), int(n_vars)),
        dtype=np.int32,
    )


def _build_pandas_frame(
    data: dict[str, np.ndarray | tuple],
    *,
    index_column: str | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            column_name: list(values) if isinstance(values, tuple) else values
            for column_name, values in data.items()
        }
    )
    if index_column is not None and index_column in frame.columns:
        frame.index = pd.Index(frame[index_column].astype(str), name=index_column)
    return frame


def _select_var_index_column(data: dict[str, np.ndarray | tuple]) -> str | None:
    for candidate in ("gene_id", "canonical_gene_id", "global_id", "origin_index"):
        if candidate in data:
            return candidate
    return None


def _estimate_columnar_memory_bytes(
    data: dict[str, np.ndarray | tuple],
) -> int:
    total = 0
    for values in data.values():
        if isinstance(values, np.ndarray):
            total += int(values.nbytes)
            continue
        total += int(sys.getsizeof(values))
        total += sum(_estimate_python_value_bytes(value) for value in values)
    return total


def _estimate_python_value_bytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return int(sys.getsizeof(value))


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
