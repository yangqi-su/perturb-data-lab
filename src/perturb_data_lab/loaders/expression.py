"""Backend-agnostic expression readers.

Defines the ``ExpressionReader`` protocol, a ``BaseExpressionReader`` abstract
class that handles global→local routing and order-preserving reassembly, and
backend-specific implementations for the remaining slim-main readers.

Supported backends: Lance and Zarr.
Supported topologies: aggregate and federated.

Readers return **only expression data** via
``read_expression_flat(global_indices) -> ExpressionBatch``. Identity and
metadata fields (dataset_id, dataset_index, local_row_index, size_factor,
etc.) belong in ``MetadataIndex``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .loaders import ExpressionBatch

__all__ = [
    "ExpressionReader",
    "BaseExpressionReader",
    "AggregateLanceReader",
    "FederatedLanceReader",
    "AggregateZarrReader",
    "FederatedZarrReader",
    "LanceDatasetEntry",
    "ZarrDatasetEntry",
    "DatasetEntry",
    "build_expression_reader",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TAKE_CHUNK: int = 2048
"""Maximum number of indices per ``lance.Dataset.take()`` call.

Lance has a known offset overflow bug when more than 2048 indices are
passed in a single ``take()`` call.  All Lance chunking logic uses this
constant as the hard upper bound.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetEntry:
    """Base routing entry describing a dataset's global index range.

    Subclasses add backend-specific path attributes.
    """

    dataset_id: str
    global_start: int  # inclusive, first global index belonging to this dataset
    global_end: int  # exclusive, first global index after this dataset


@dataclass(frozen=True)
class LanceDatasetEntry(DatasetEntry):
    """Lance-specific dataset entry with a path to the Lance file."""

    lance_path: str | Path


@dataclass(frozen=True)
class ZarrDatasetEntry(DatasetEntry):
    """Zarr-specific dataset entry with paths to the CSR-format arrays."""

    offsets_path: str | Path
    indices_path: str | Path
    counts_path: str | Path


@dataclass(frozen=True)
class ArrowIpcDatasetEntry(DatasetEntry):
    """Legacy removed-backend entry retained only for direct imports."""

    arrow_path: str | Path


@dataclass(frozen=True)
class HfDatasetsDatasetEntry(DatasetEntry):
    """Legacy removed-backend entry retained only for direct imports."""

    dataset_path: str | Path


@dataclass(frozen=True)
class ParquetDatasetEntry(DatasetEntry):
    """Legacy removed-backend entry retained only for direct imports."""

    parquet_path: str | Path


@dataclass(frozen=True)
class WebDatasetEntry(DatasetEntry):
    """Legacy removed-backend entry retained only for direct imports."""

    tar_path: str | Path


@dataclass(frozen=True)
class CsrMemmapShardEntry(DatasetEntry):
    """Legacy removed-backend entry retained only for direct imports."""

    shard_id: int
    shard_path: Path
    row_offsets_path: Path
    gene_indices_path: Path
    counts_path: Path
    n_cells: int


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ExpressionReader(Protocol):
    """Protocol for topology-aware expression readers.

    Implementations must return expression data **only**. Metadata
    enrichment is handled separately by ``MetadataIndex`` and higher-level
    loader code.
    """

    def read_expression_flat(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data as an order-preserving ``ExpressionBatch``."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_indices(
    indices: Sequence[int], chunk_size: int = _MAX_TAKE_CHUNK
) -> list[list[int]]:
    """Split a sequence of indices into sub-batches of at most *chunk_size*."""
    indices = list(indices)
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]


def _extract_list_columns(table, col_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract a Lance/Arrow list column as (offsets, flat_values) numpy arrays.

    Uses ``combine_chunks()`` for contiguous access, then slices offsets
    and flat values.  This avoids per-row ``.as_py()`` conversions for
    list columns.
    """
    raw = table.column(col_name).combine_chunks()
    offsets = np.asarray(raw.offsets.to_numpy(), dtype=np.int64)
    flat = np.asarray(raw.flatten().to_numpy(), dtype=np.int32)
    return offsets, flat


def _empty_expression_batch(global_indices: Sequence[int] | None = None) -> ExpressionBatch:
    """Return an empty ``ExpressionBatch`` preserving index dtype conventions."""
    if global_indices is None:
        indices = np.array([], dtype=np.int64)
    else:
        indices = np.array([int(idx) for idx in global_indices], dtype=np.int64)
    return ExpressionBatch(
        batch_size=0,
        global_row_index=indices,
        row_offsets=np.array([0], dtype=np.int64),
        expressed_gene_indices=np.array([], dtype=np.int32),
        expression_counts=np.array([], dtype=np.int32),
    )


def _cell_arrays_to_expression_batch(
    global_indices: Sequence[int],
    cells: Sequence[tuple[np.ndarray, np.ndarray]],
) -> ExpressionBatch:
    """Convert ordered per-cell arrays into an ``ExpressionBatch``."""
    indices = np.array([int(idx) for idx in global_indices], dtype=np.int64)
    if len(cells) == 0:
        return _empty_expression_batch(indices)

    row_offsets = np.zeros(len(cells) + 1, dtype=np.int64)
    egi_parts: list[np.ndarray] = []
    ec_parts: list[np.ndarray] = []
    for pos, (gene_indices, counts) in enumerate(cells):
        gene_indices = np.asarray(gene_indices, dtype=np.int32)
        counts = np.asarray(counts, dtype=np.int32)
        egi_parts.append(gene_indices)
        ec_parts.append(counts)
        row_offsets[pos + 1] = row_offsets[pos] + len(gene_indices)

    return ExpressionBatch(
        batch_size=len(cells),
        global_row_index=indices,
        row_offsets=row_offsets,
        expressed_gene_indices=np.concatenate(egi_parts),
        expression_counts=np.concatenate(ec_parts),
    )


def _table_to_cells(table) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert a Lance table into per-cell expression arrays."""
    n = table.num_rows
    egi_offsets, egi_flat = _extract_list_columns(
        table, "expressed_gene_indices"
    )
    ec_offsets, ec_flat = _extract_list_columns(
        table, "expression_counts"
    )

    cells: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        s_egi = slice(egi_offsets[i], egi_offsets[i + 1])
        s_ec = slice(ec_offsets[i], ec_offsets[i + 1])
        cells.append((egi_flat[s_egi].copy(), ec_flat[s_ec].copy()))
    return cells


# ---------------------------------------------------------------------------
# BaseExpressionReader — handles routing, grouping, and reassembly
# ---------------------------------------------------------------------------


class BaseExpressionReader(ABC):
    """Abstract base for expression readers.

    Handles global→local routing, per-dataset grouping, and order-preserving
    reassembly.  Subclasses implement only backend-specific local-row reading.

    Parameters
    ----------
    entries : list of DatasetEntry
        One entry per dataset, sorted by ``global_start``.  For aggregate
        topology this is a single entry covering the full range.
    """

    def __init__(self, entries: list[DatasetEntry]):
        self._entries: list[DatasetEntry] = sorted(
            entries, key=lambda e: e.global_start
        )
        # Precompute stop array for np.searchsorted-based routing
        self._stops: np.ndarray = np.array(
            [e.global_end for e in self._entries], dtype=np.int64
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_expression_flat(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data with shared routing and order reassembly.

        Backends with a more efficient flat representation can override this
        method. Other readers inherit this shared order-preserving
        implementation backed by per-dataset local reads.
        """
        if not global_indices:
            return _empty_expression_batch()

        indices = [int(idx) for idx in global_indices]
        self._validate_all(indices)

        grouped: dict[str, list[tuple[int, int]]] = {}
        for output_pos, global_idx in enumerate(indices):
            entry, local_idx = self._resolve_entry(global_idx)
            grouped.setdefault(entry.dataset_id, []).append((output_pos, local_idx))

        cells_by_pos: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for ds_id, selections in grouped.items():
            entry = self._find_entry_by_id(ds_id)
            local_indices = [local for _, local in selections]
            cells = self._read_local_cells(entry, local_indices)
            if len(cells) != len(selections):
                raise ValueError(
                    f"Reader for dataset_id '{ds_id}' returned {len(cells)} cells "
                    f"for {len(selections)} requested indices"
                )
            for (output_pos, _), cell in zip(selections, cells, strict=True):
                cells_by_pos[output_pos] = cell

        ordered_cells = [cells_by_pos[pos] for pos in range(len(indices))]
        return _cell_arrays_to_expression_batch(indices, ordered_cells)

    # ------------------------------------------------------------------
    # Routing helpers (shared across all backends)
    # ------------------------------------------------------------------

    def _resolve_entry(self, global_index: int) -> tuple[DatasetEntry, int]:
        """Resolve a global index to its owning dataset entry and local index.

        Uses ``np.searchsorted`` on ``global_end`` stops for O(log N) lookup.
        """
        entry_idx = _searchsorted_entry(self._stops, global_index)
        entry = self._entries[entry_idx]
        if not (entry.global_start <= global_index < entry.global_end):
            raise IndexError(
                f"global_index {global_index} does not fall within any "
                f"registered dataset range"
            )
        local_idx = global_index - entry.global_start
        return entry, local_idx

    def _find_entry_by_id(self, dataset_id: str) -> DatasetEntry:
        """Return the entry for *dataset_id*."""
        for entry in self._entries:
            if entry.dataset_id == dataset_id:
                return entry
        raise KeyError(
            f"dataset_id '{dataset_id}' not found; "
            f"available: {[e.dataset_id for e in self._entries]}"
        )

    def _validate_all(self, indices: list[int]) -> None:
        """Validate that all indices are within the total corpus range.

        Subclasses may override to add backend-specific checks (e.g.,
        Lance row count check).  The base implementation checks only
        that indices are non-negative and within the overall start/end
        bounds implied by the entry list.
        """
        total_start = self._entries[0].global_start
        total_end = self._entries[-1].global_end
        for idx in indices:
            if idx < total_start or idx >= total_end:
                raise IndexError(
                    f"global_index {idx} out of range "
                    f"[{total_start}, {total_end})"
                )

    # ------------------------------------------------------------------
    # Backend-specific hook
    # ------------------------------------------------------------------

    @abstractmethod
    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Read per-cell expression arrays within one dataset.

        Returns one ``(expressed_gene_indices, expression_counts)`` tuple per
        requested local index, in the same order as ``local_indices``.
        """
        ...


# ===================================================================
# Lance readers
# ===================================================================


class AggregateLanceReader(BaseExpressionReader):
    """Expression reader for aggregate Lance topology.

    Opens a single ``lance.Dataset`` handle and uses vectorized ``take()``
    chunked to ≤2048 indices per call.

    Parameters
    ----------
    lance_path : str or Path
        Path to the aggregated ``.lance`` directory.
    entries : list of DatasetEntry
        Index ranges for each dataset in the corpus.  For aggregate
        topology this is typically a single entry covering the full
        range (0 to total_rows).
    """

    def __init__(self, lance_path: str | Path, entries: list[DatasetEntry]):
        super().__init__(entries)
        self._lance_path = str(Path(lance_path))
        self._dataset = None
        self._dataset_pid: int | None = None
        self._total_rows: int | None = None

    @property
    def lance_path(self) -> str:
        return self._lance_path

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_dataset"] = None
        state["_dataset_pid"] = None
        return state

    def _open_dataset(self):
        import lance

        current_pid = os.getpid()
        if self._dataset is None or self._dataset_pid != current_pid:
            self._dataset = lance.dataset(self._lance_path)
            self._dataset_pid = current_pid
        return self._dataset

    def _count_rows(self) -> int:
        if self._total_rows is None:
            self._total_rows = int(self._open_dataset().count_rows())
        return self._total_rows

    def _validate_all(self, indices: list[int]) -> None:
        super()._validate_all(indices)
        total_rows = self._count_rows()
        for idx in indices:
            if idx >= total_rows:
                raise IndexError(
                    f"global_index {idx} out of range [0, {total_rows})"
                )

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        dataset = self._open_dataset()
        global_positions = [entry.global_start + li for li in local_indices]
        cells: list[tuple[np.ndarray, np.ndarray]] = []
        for chunk in _chunk_indices(global_positions):
            table = dataset.take(chunk)
            cells.extend(_table_to_cells(table))
        return cells

    # ------------------------------------------------------------------
    # Fast path — direct flat read (Phase 2)
    # ------------------------------------------------------------------

    def read_expression_flat(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data directly as flat arrays (aggregate fast path).

        Bypasses the shared per-dataset regrouping in
        ``BaseExpressionReader.read_expression_flat()``. Performs direct
        chunked ``take()`` calls on the aggregate Lance file and returns
        concatenated flat expression arrays with row offsets.

        Parameters
        ----------
        global_indices : sequence of int
            Corpus-global cell indices into the aggregate Lance file.

        Returns
        -------
        ExpressionBatch
            Flat expression batch with concatenated arrays and row offsets.

        Raises
        ------
        IndexError
            If any index is out of range.
        """
        if not global_indices:
            return _empty_expression_batch()

        indices = [int(idx) for idx in global_indices]

        # Validate all indices against Lance row count
        dataset = self._open_dataset()
        total_rows = self._count_rows()
        for idx in indices:
            if idx < 0 or idx >= total_rows:
                raise IndexError(
                    f"global_index {idx} out of range [0, {total_rows})"
                )

        cells: list[tuple[np.ndarray, np.ndarray]] = []

        for chunk in _chunk_indices(indices):
            cells.extend(_table_to_cells(dataset.take(chunk)))

        return _cell_arrays_to_expression_batch(indices, cells)


class FederatedLanceReader(BaseExpressionReader):
    """Expression reader for federated Lance topology.

    Opens per-dataset ``lance.Dataset`` handles (lazily cached) and
    reads expression data grouped by dataset.

    Parameters
    ----------
    entries : list of LanceDatasetEntry
        One entry per dataset, each with its own Lance file path and
        global index range.
    """

    def __init__(self, entries: list[LanceDatasetEntry]):
        super().__init__(entries)  # type: ignore[arg-type]
        self._datasets: dict[str, "lance.Dataset"] = {}
        self._datasets_pid: int | None = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_datasets"] = {}
        state["_datasets_pid"] = None
        return state

    def _open_dataset(self, entry: LanceDatasetEntry):
        import lance

        current_pid = os.getpid()
        if self._datasets_pid != current_pid:
            self._datasets = {}
            self._datasets_pid = current_pid
        if entry.dataset_id not in self._datasets:
            self._datasets[entry.dataset_id] = lance.dataset(str(entry.lance_path))
        return self._datasets[entry.dataset_id]

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lance_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(lance_entry, LanceDatasetEntry)
        ds = self._open_dataset(lance_entry)

        cells: list[tuple[np.ndarray, np.ndarray]] = []
        for chunk in _chunk_indices(local_indices):
            table = ds.take(chunk)
            cells.extend(_table_to_cells(table))
        return cells

    # ------------------------------------------------------------------
    # Fast path — direct flat read (Phase 3)
    # ------------------------------------------------------------------

    def read_expression_flat(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data directly as flat arrays (federated fast path).

        Groups indices by dataset/file, performs chunked ``take()`` calls
        per dataset, and returns concatenated flat expression arrays with
        row offsets.

        This path retains per-dataset grouping (federated topology cannot
        avoid it) but bypasses the shared reassembly helper for a more direct
        flat implementation.

        Parameters
        ----------
        global_indices : sequence of int
            Corpus-global cell indices.

        Returns
        -------
        ExpressionBatch
            Flat expression batch with concatenated arrays and row offsets.

        Raises
        ------
        IndexError
            If any index is out of range or not registered in any dataset.
        """
        if not global_indices:
            return _empty_expression_batch()

        indices = [int(idx) for idx in global_indices]
        n = len(indices)
        self._validate_all(indices)

        # Group by dataset, preserving the output position of each index.
        # Within each dataset, selections are in the same relative order
        # as they appear in the input.
        # grouped[dataset_id] = list[(output_pos, local_index)]
        grouped: dict[str, list[tuple[int, int]]] = {}
        for output_pos, global_idx in enumerate(indices):
            entry, local_idx = self._resolve_entry(global_idx)
            grouped.setdefault(entry.dataset_id, []).append(
                (output_pos, local_idx)
            )

        cells_by_pos: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for ds_id, selections in grouped.items():
            lance_entry = self._find_entry_by_id(ds_id)
            assert isinstance(lance_entry, LanceDatasetEntry)
            ds = self._open_dataset(lance_entry)

            local_indices = [local for _, local in selections]
            output_positions = [pos for pos, _ in selections]

            idx_pos = 0
            for chunk in _chunk_indices(local_indices):
                for cell in _table_to_cells(ds.take(chunk)):
                    pos = output_positions[idx_pos]
                    cells_by_pos[pos] = cell
                    idx_pos += 1

        ordered_cells = [cells_by_pos[pos] for pos in range(n)]
        return _cell_arrays_to_expression_batch(indices, ordered_cells)


# ===================================================================
# Zarr readers
# ===================================================================


def _zarr_read_cells(
    offsets: np.ndarray,
    indices_flat: np.ndarray,
    counts_flat: np.ndarray,
    local_indices: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read per-cell expression arrays from CSR-format Zarr arrays."""
    cells: list[tuple[np.ndarray, np.ndarray]] = []
    for local_idx in local_indices:
        s = slice(offsets[local_idx], offsets[local_idx + 1])
        cells.append((indices_flat[s].copy(), counts_flat[s].copy()))
    return cells


class AggregateZarrReader(BaseExpressionReader):
    """Expression reader for aggregate Zarr topology (CSR format).

    Parameters
    ----------
    offsets_path : str or Path
        Path to the row-offsets Zarr array (int64, shape [n_rows+1]).
    indices_path : str or Path
        Path to the flat gene-indices Zarr array (int32).
    counts_path : str or Path
        Path to the flat expression-counts Zarr array (int32).
    entries : list of DatasetEntry
        Dataset range entries (single entry for aggregate).
    """

    def __init__(
        self,
        offsets_path: str | Path,
        indices_path: str | Path,
        counts_path: str | Path,
        entries: list[DatasetEntry],
    ):
        super().__init__(entries)
        import zarr

        self._offsets = zarr.open(str(offsets_path), mode="r")["row_offsets"]
        self._indices = zarr.open(str(indices_path), mode="r")["indices"]
        self._counts = zarr.open(str(counts_path), mode="r")["counts"]

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        global_positions = [entry.global_start + li for li in local_indices]
        return _zarr_read_cells(
            self._offsets,
            self._indices,
            self._counts,
            global_positions,
        )


class FederatedZarrReader(BaseExpressionReader):
    """Expression reader for federated Zarr topology (CSR format).

    Parameters
    ----------
    entries : list of ZarrDatasetEntry
        One entry per dataset, each with its own Zarr array paths.
    """

    def __init__(self, entries: list[ZarrDatasetEntry]):
        super().__init__(entries)  # type: ignore[arg-type]
        self._offsets_cache: dict[str, np.ndarray] = {}
        self._indices_cache: dict[str, np.ndarray] = {}
        self._counts_cache: dict[str, np.ndarray] = {}

    def _open_arrays(self, entry: ZarrDatasetEntry):
        import zarr

        ds_id = entry.dataset_id
        if ds_id not in self._offsets_cache:
            self._offsets_cache[ds_id] = zarr.open(
                str(entry.offsets_path), mode="r"
            )["row_offsets"]
            self._indices_cache[ds_id] = zarr.open(
                str(entry.indices_path), mode="r"
            )["indices"]
            self._counts_cache[ds_id] = zarr.open(
                str(entry.counts_path), mode="r"
            )["counts"]
        return (
            self._offsets_cache[ds_id],
            self._indices_cache[ds_id],
            self._counts_cache[ds_id],
        )

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        zarr_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(zarr_entry, ZarrDatasetEntry)
        offsets, indices_arr, counts_arr = self._open_arrays(zarr_entry)
        return _zarr_read_cells(
            offsets, indices_arr, counts_arr, local_indices
        )


# ===================================================================
# Removed backend stubs
# ===================================================================


def _unsupported_backend_error(backend: str) -> ValueError:
    return ValueError(
        f"Backend '{backend}' is not supported in slim main. Only "
        "'lance' and 'zarr' readers remain available."
    )


class AggregateTileDBReader(BaseExpressionReader):
    """Legacy removed-backend stub retained only for direct imports."""

    def __init__(self, *args, **kwargs):
        raise _unsupported_backend_error("tiledb")

    def _read_local_cells(self, entry: DatasetEntry, local_indices: list[int]):
        raise AssertionError("unreachable")


class FederatedArrowIpcReader(BaseExpressionReader):
    """Legacy removed-backend stub retained only for direct imports."""

    def __init__(self, *args, **kwargs):
        raise _unsupported_backend_error("arrow_ipc")

    def _read_local_cells(self, entry: DatasetEntry, local_indices: list[int]):
        raise AssertionError("unreachable")


class FederatedHfDatasetsReader(BaseExpressionReader):
    """Legacy removed-backend stub retained only for direct imports."""

    def __init__(self, *args, **kwargs):
        raise _unsupported_backend_error("hf_datasets")

    def _read_local_cells(self, entry: DatasetEntry, local_indices: list[int]):
        raise AssertionError("unreachable")


class FederatedParquetReader(BaseExpressionReader):
    """Legacy removed-backend stub retained only for direct imports."""

    def __init__(self, *args, **kwargs):
        raise _unsupported_backend_error("parquet")

    def _read_local_cells(self, entry: DatasetEntry, local_indices: list[int]):
        raise AssertionError("unreachable")


class FederatedWebDatasetReader(BaseExpressionReader):
    """Legacy removed-backend stub retained only for direct imports."""

    def __init__(self, *args, **kwargs):
        raise _unsupported_backend_error("webdataset")

    def _read_local_cells(self, entry: DatasetEntry, local_indices: list[int]):
        raise AssertionError("unreachable")


class AggregateCsrMemmapReader(BaseExpressionReader):
    """Legacy removed-backend stub retained only for direct imports."""

    def __init__(self, *args, **kwargs):
        raise _unsupported_backend_error("csr_memmap")

    def _read_local_cells(self, entry: DatasetEntry, local_indices: list[int]):
        raise AssertionError("unreachable")


# ===================================================================
# Factory
# ===================================================================


def build_expression_reader(
    backend: str,
    topology: str,
    entries: list[DatasetEntry],
    **kwargs,
) -> ExpressionReader:
    """Build an expression reader for the given backend and topology.

    Parameters
    ----------
    backend : str
        One of ``"lance"`` or ``"zarr"``.
    topology : str
        Either ``"aggregate"`` or ``"federated"``.
    entries : list of DatasetEntry
        Dataset routing entries.  The concrete entry type must match the
        backend (e.g., ``LanceDatasetEntry`` for Lance).
    **kwargs
        Backend-specific arguments (e.g., ``lance_path`` for aggregate Lance).

    Returns
    -------
    ExpressionReader
        A concrete reader instance.

    Raises
    ------
    ValueError
        If the backend/topology combination is unsupported.
    """
    if topology not in ("aggregate", "federated"):
        raise ValueError(
            f"Unknown topology '{topology}'. Expected 'aggregate' or 'federated'."
        )

    if backend == "lance":
        if topology == "aggregate":
            lance_path = kwargs.pop("lance_path")
            return AggregateLanceReader(lance_path, entries)
        else:
            return FederatedLanceReader(entries)  # type: ignore[arg-type]

    elif backend == "zarr":
        if topology == "aggregate":
            offsets_path = kwargs.pop("offsets_path")
            indices_path = kwargs.pop("indices_path")
            counts_path = kwargs.pop("counts_path")
            return AggregateZarrReader(offsets_path, indices_path, counts_path, entries)
        else:
            return FederatedZarrReader(entries)  # type: ignore[arg-type]

    else:
        if backend in {
            "tiledb",
            "arrow_ipc",
            "arrow-parquet",
            "parquet",
            "hf_datasets",
            "hf-datasets",
            "webdataset",
            "csr_memmap",
            "csr-memmap",
        }:
            raise _unsupported_backend_error(backend)
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Supported: lance, zarr."
        )


# ---------------------------------------------------------------------------
# Internal helper: np.searchsorted-based entry lookup
# ---------------------------------------------------------------------------


def _searchsorted_entry(stops: np.ndarray, global_index: int) -> int:
    """Return the index into *stops* array for *global_index*.

    Uses ``np.searchsorted(stops, global_index, side='right')`` which
    returns the first stop > *global_index*, i.e., the entry whose range
    contains *global_index* (if any).
    """
    return int(np.searchsorted(stops, global_index, side="right"))
