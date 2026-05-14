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

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..materializers.backends.csr_cache import ShardLRUCache
from .loaders import ExpressionBatch

__all__ = [
    "ExpressionReader",
    "BaseExpressionReader",
    "AggregateLanceReader",
    "FederatedLanceReader",
    "AggregateZarrReader",
    "FederatedZarrReader",
    "AggregateTileDBReader",
    "FederatedArrowIpcReader",
    "FederatedHfDatasetsReader",
    "FederatedParquetReader",
    "FederatedWebDatasetReader",
    "AggregateCsrMemmapReader",
    "LanceDatasetEntry",
    "ZarrDatasetEntry",
    "ArrowIpcDatasetEntry",
    "HfDatasetsDatasetEntry",
    "ParquetDatasetEntry",
    "WebDatasetEntry",
    "CsrMemmapShardEntry",
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
    """Arrow IPC (feather) dataset entry with a path to the ``.arrow`` file."""

    arrow_path: str | Path


@dataclass(frozen=True)
class HfDatasetsDatasetEntry(DatasetEntry):
    """HuggingFace datasets entry with a path to the saved dataset directory."""

    dataset_path: str | Path


@dataclass(frozen=True)
class ParquetDatasetEntry(DatasetEntry):
    """Parquet dataset entry with a path to the ``.parquet`` file."""

    parquet_path: str | Path


@dataclass(frozen=True)
class WebDatasetEntry(DatasetEntry):
    """WebDataset (tar shard) dataset entry with a path to the ``.tar`` file."""

    tar_path: str | Path


@dataclass(frozen=True)
class CsrMemmapShardEntry(DatasetEntry):
    """CSR memmap shard entry with paths to the CSR ``.npy`` files.

    Each entry maps a contiguous range of global cell indices to a single
    shard directory containing three ``.npy`` files in CSR format.

    Attributes
    ----------
    shard_id : int
        Zero-based shard index within the corpus.
    shard_path : Path
        Path to the shard directory (e.g. ``<root>/shard_000000/``).
    row_offsets_path : Path
        Path to ``row_offsets.npy`` (int64, shape ``(n_cells+1,)``).
    gene_indices_path : Path
        Path to ``gene_indices.npy`` (int32, flat).
    counts_path : Path
        Path to ``counts.npy`` (int32, flat).
    n_cells : int
        Number of cells stored in this shard.
    """

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


def _list_column_as_py(table, col_name: str, row_idx: int):
    """Extract a single row's list-column value as a Python list."""
    return table.column(col_name)[row_idx].as_py()


def _import_hf_datasets():
    """Import optional HuggingFace datasets helpers with an actionable error."""
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise ImportError(
            "hf_datasets backend requires the optional 'datasets' package; "
            "install perturb-data-lab[hf-datasets] or pip install datasets"
        ) from exc
    return load_from_disk


def _import_tiledb():
    """Import optional TileDB with an actionable error."""
    try:
        import tiledb
    except ImportError as exc:
        raise ImportError(
            "tiledb backend requires the TileDB Python package; "
            "install tiledb in the selected runtime"
        ) from exc
    return tiledb


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
            cells.extend(self._table_to_cells(table))
        return cells

    @staticmethod
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
        n = len(indices)

        # Validate all indices against Lance row count
        dataset = self._open_dataset()
        total_rows = self._count_rows()
        for idx in indices:
            if idx < 0 or idx >= total_rows:
                raise IndexError(
                    f"global_index {idx} out of range [0, {total_rows})"
                )

        # Accumulate flat arrays and row offsets chunk by chunk
        egi_parts: list[np.ndarray] = []
        ec_parts: list[np.ndarray] = []
        row_offsets = np.zeros(n + 1, dtype=np.int64)
        cursor = 0  # tracks position in assembled rows

        for chunk in _chunk_indices(indices):
            table = dataset.take(chunk)
            chunk_n = len(chunk)

            egi_offsets, egi_flat = _extract_list_columns(
                table, "expressed_gene_indices"
            )
            ec_offsets, ec_flat = _extract_list_columns(
                table, "expression_counts"
            )

            for i in range(chunk_n):
                s_egi = slice(egi_offsets[i], egi_offsets[i + 1])
                s_ec = slice(ec_offsets[i], ec_offsets[i + 1])
                egi_parts.append(np.asarray(egi_flat[s_egi]))
                ec_parts.append(np.asarray(ec_flat[s_ec]))
                cursor += 1
                row_offsets[cursor] = (
                    row_offsets[cursor - 1] + len(egi_parts[-1])
                )

        return ExpressionBatch(
            batch_size=n,
            global_row_index=np.array(indices, dtype=np.int64),
            row_offsets=row_offsets,
            expressed_gene_indices=(
                np.concatenate(egi_parts)
                if egi_parts
                else np.array([], dtype=np.int32)
            ),
            expression_counts=(
                np.concatenate(ec_parts)
                if ec_parts
                else np.array([], dtype=np.int32)
            ),
        )


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
            cells.extend(self._table_to_cells(table))
        return cells

    @staticmethod
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

        # Accumulate per-cell flat array parts indexed by output position
        egi_by_pos: dict[int, np.ndarray] = {}
        ec_by_pos: dict[int, np.ndarray] = {}

        for ds_id, selections in grouped.items():
            lance_entry = self._find_entry_by_id(ds_id)
            assert isinstance(lance_entry, LanceDatasetEntry)
            ds = self._open_dataset(lance_entry)

            local_indices = [local for _, local in selections]
            output_positions = [pos for pos, _ in selections]

            idx_pos = 0
            for chunk in _chunk_indices(local_indices):
                table = ds.take(chunk)
                chunk_n = len(chunk)

                egi_offsets, egi_flat = _extract_list_columns(
                    table, "expressed_gene_indices"
                )
                ec_offsets, ec_flat = _extract_list_columns(
                    table, "expression_counts"
                )

                for i in range(chunk_n):
                    s_egi = slice(egi_offsets[i], egi_offsets[i + 1])
                    s_ec = slice(ec_offsets[i], ec_offsets[i + 1])
                    pos = output_positions[idx_pos]
                    egi_by_pos[pos] = np.asarray(egi_flat[s_egi])
                    ec_by_pos[pos] = np.asarray(ec_flat[s_ec])
                    idx_pos += 1

        # Build row offsets and concatenate in output order
        row_offsets = np.zeros(n + 1, dtype=np.int64)
        egi_parts: list[np.ndarray] = []
        ec_parts: list[np.ndarray] = []
        for pos in range(n):
            egi_parts.append(egi_by_pos[pos])
            ec_parts.append(ec_by_pos[pos])
            row_offsets[pos + 1] = (
                row_offsets[pos] + len(egi_by_pos[pos])
            )

        return ExpressionBatch(
            batch_size=n,
            global_row_index=np.array(indices, dtype=np.int64),
            row_offsets=row_offsets,
            expressed_gene_indices=np.concatenate(egi_parts),
            expression_counts=np.concatenate(ec_parts),
        )


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
# TileDB reader
# ===================================================================


class AggregateTileDBReader(BaseExpressionReader):
    """Expression reader for aggregate TileDB topology.

    Opens a single sparse TileDB array lazily per process/worker and
    reconstructs per-row expression arrays from ``(global_row_index,
    local_gene_index) -> count`` coordinates.
    """

    def __init__(
        self,
        tiledb_path: str | Path,
        entries: list[DatasetEntry],
        *,
        tiledb_meta_path: str | Path | None = None,
        max_local_gene_index_exclusive: int | None = None,
    ):
        super().__init__(entries)
        self._tiledb_path = str(Path(tiledb_path))
        self._tiledb_meta_path = (
            str(Path(tiledb_meta_path))
            if tiledb_meta_path is not None
            else None
        )
        self._array = None
        self._array_pid: int | None = None
        self._max_local_gene_index_exclusive = self._resolve_gene_bound(
            max_local_gene_index_exclusive
        )

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_array"] = None
        state["_array_pid"] = None
        return state

    def _resolve_gene_bound(self, explicit: int | None) -> int:
        if explicit is not None:
            bound = int(explicit)
        elif self._tiledb_meta_path is not None:
            meta = json.loads(Path(self._tiledb_meta_path).read_text())
            bound = int(meta.get("max_observed_local_vocabulary_size", 0))
        else:
            raise ValueError(
                "AggregateTileDBReader requires either tiledb_meta_path or "
                "max_local_gene_index_exclusive"
            )
        if bound <= 0:
            raise ValueError(
                "AggregateTileDBReader requires a positive local gene bound; "
                f"got {bound}"
            )
        return bound

    def _open_array(self):
        tiledb = _import_tiledb()
        current_pid = os.getpid()
        if self._array is None or self._array_pid != current_pid:
            self._array = tiledb.open(self._tiledb_path, mode="r")
            self._array_pid = current_pid
        return self._array

    @staticmethod
    def _query_result_to_cell(result) -> tuple[np.ndarray, np.ndarray]:
        genes = np.asarray(result["local_gene_index"], dtype=np.int32)
        counts = np.asarray(result["count"], dtype=np.int32)
        if genes.size == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        order = np.argsort(genes, kind="stable")
        return genes[order].copy(), counts[order].copy()

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        array = self._open_array()
        cells: list[tuple[np.ndarray, np.ndarray]] = []
        gene_stop = self._max_local_gene_index_exclusive
        for local_idx in local_indices:
            global_idx = entry.global_start + local_idx
            result = array.query(attrs=["count"], coords=True)[
                global_idx : global_idx + 1,
                0:gene_stop,
            ]
            cells.append(self._query_result_to_cell(result))
        return cells


# ===================================================================
# Arrow IPC federated reader
# ===================================================================


class FederatedArrowIpcReader(BaseExpressionReader):
    """Expression reader for federated Arrow IPC (feather) topology.

    Parameters
    ----------
    entries : list of ArrowIpcDatasetEntry
        One entry per dataset, each with an ``.arrow`` file path.
    """

    def __init__(self, entries: list[ArrowIpcDatasetEntry]):
        super().__init__(entries)  # type: ignore[arg-type]
        self._readers: dict[str, "pa.ipc.RecordBatchFileReader"] = {}
        self._tables: dict[str, "pa.Table"] = {}

    def _open_table(self, entry: ArrowIpcDatasetEntry):
        import pyarrow as pa

        ds_id = entry.dataset_id
        if ds_id not in self._tables:
            self._tables[ds_id] = pa.ipc.open_file(
                pa.memory_map(str(entry.arrow_path), "r")
            ).read_all()
        return self._tables[ds_id]

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        arrow_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(arrow_entry, ArrowIpcDatasetEntry)
        table = self._open_table(arrow_entry)

        cells: list[tuple[np.ndarray, np.ndarray]] = []
        for local_idx in local_indices:
            egi = _list_column_as_py(table, "expressed_gene_indices", local_idx)
            ec = _list_column_as_py(table, "expression_counts", local_idx)
            cells.append(
                (np.array(egi, dtype=np.int32), np.array(ec, dtype=np.int32))
            )
        return cells


# ===================================================================
# HuggingFace datasets federated reader
# ===================================================================


class FederatedHfDatasetsReader(BaseExpressionReader):
    """Expression reader for federated HuggingFace datasets topology."""

    def __init__(self, entries: list[HfDatasetsDatasetEntry]):
        super().__init__(entries)  # type: ignore[arg-type]
        _import_hf_datasets()
        self._datasets: dict[str, object] = {}
        self._dataset_pids: dict[str, int] = {}

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_datasets"] = {}
        state["_dataset_pids"] = {}
        return state

    def _open_dataset(self, entry: HfDatasetsDatasetEntry):
        load_from_disk = _import_hf_datasets()
        current_pid = os.getpid()
        ds_id = entry.dataset_id
        if ds_id not in self._datasets or self._dataset_pids.get(ds_id) != current_pid:
            self._datasets[ds_id] = load_from_disk(
                str(entry.dataset_path),
                keep_in_memory=False,
            )
            self._dataset_pids[ds_id] = current_pid
        return self._datasets[ds_id]

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        hf_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(hf_entry, HfDatasetsDatasetEntry)
        dataset = self._open_dataset(hf_entry)

        cells: list[tuple[np.ndarray, np.ndarray]] = []
        for local_idx in local_indices:
            row = dataset[int(local_idx)]
            cells.append(
                (
                    np.asarray(row["expressed_gene_indices"], dtype=np.int32),
                    np.asarray(row["expression_counts"], dtype=np.int32),
                )
            )
        return cells


# ===================================================================
# Parquet federated reader
# ===================================================================


class FederatedParquetReader(BaseExpressionReader):
    """Expression reader for federated Parquet topology.

    Parameters
    ----------
    entries : list of ParquetDatasetEntry
        One entry per dataset, each with a ``.parquet`` file path.
    """

    def __init__(self, entries: list[ParquetDatasetEntry]):
        super().__init__(entries)  # type: ignore[arg-type]
        self._tables: dict[str, "pa.Table"] = {}

    def _open_table(self, entry: ParquetDatasetEntry):
        import pyarrow.parquet as pq

        ds_id = entry.dataset_id
        if ds_id not in self._tables:
            self._tables[ds_id] = pq.read_table(str(entry.parquet_path))
        return self._tables[ds_id]

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        pq_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(pq_entry, ParquetDatasetEntry)
        table = self._open_table(pq_entry)

        cells: list[tuple[np.ndarray, np.ndarray]] = []
        for local_idx in local_indices:
            egi = _list_column_as_py(table, "expressed_gene_indices", local_idx)
            ec = _list_column_as_py(table, "expression_counts", local_idx)
            cells.append(
                (np.array(egi, dtype=np.int32), np.array(ec, dtype=np.int32))
            )
        return cells


# ===================================================================
# WebDataset federated reader
# ===================================================================


class FederatedWebDatasetReader(BaseExpressionReader):
    """Expression reader for federated WebDataset (tar shard) topology.

    Reads pickle-serialized cells from tar archives.  Each tar member
    is named ``cell_NNNNNNNN.pkl`` where the number is the local row
    index (zero-padded).  This reader seeks to specific cells rather
    than streaming the entire archive.

    Parameters
    ----------
    entries : list of WebDatasetEntry
        One entry per dataset, each with a ``.tar`` file path.
    """

    def __init__(self, entries: list[WebDatasetEntry]):
        super().__init__(entries)  # type: ignore[arg-type]
        self._tar_files: dict[str, "tarfile.TarFile"] = {}

    def _open_tar(self, entry: WebDatasetEntry):
        import tarfile

        ds_id = entry.dataset_id
        if ds_id not in self._tar_files:
            self._tar_files[ds_id] = tarfile.open(str(entry.tar_path), "r")
        return self._tar_files[ds_id]

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        import pickle

        wds_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(wds_entry, WebDatasetEntry)
        tar = self._open_tar(wds_entry)

        cells: list[tuple[np.ndarray, np.ndarray]] = []
        for local_idx in local_indices:
            member_name = f"cell_{local_idx:08d}.pkl"
            member = tar.getmember(member_name)
            f = tar.extractfile(member)
            if f is None:
                raise FileNotFoundError(
                    f"Member '{member_name}' not found in {wds_entry.tar_path}"
                )
            data = pickle.load(f)
            f.close()
            cells.append(
                (
                    np.asarray(data["expressed_gene_indices"], dtype=np.int32),
                    np.asarray(data["expression_counts"], dtype=np.int32),
                )
            )
        return cells


# ===================================================================
# CSR memmap reader (Phase 3)
# ===================================================================


def _build_cache(cache_config: object | None) -> ShardLRUCache | None:
    """Build a :class:`ShardLRUCache` from a cache configuration dict.

    Returns ``None`` when *cache_config* is ``None`` or has
    ``enabled=False``, meaning all reads go directly to source files.
    """
    if cache_config is None:
        return None

    if not isinstance(cache_config, dict):
        raise TypeError(
            f"cache_config must be a dict, got {type(cache_config).__name__}"
        )

    enabled = cache_config.get("enabled", False)
    if not enabled:
        return None

    cache_root = cache_config.get("cache_root")
    max_bytes = cache_config.get("max_bytes")

    if cache_root is None:
        raise ValueError("cache_config must include 'cache_root' when enabled")
    if max_bytes is None:
        raise ValueError("cache_config must include 'max_bytes' when enabled")

    cache_root_path = Path(cache_root)
    max_bytes_int = int(max_bytes)

    per_worker = bool(cache_config.get("per_worker", False))

    return ShardLRUCache(
        cache_root=cache_root_path,
        max_bytes=max_bytes_int,
        per_worker=per_worker,
    )


class AggregateCsrMemmapReader(BaseExpressionReader):
    """Expression reader for sharded CSR memmap corpora (aggregate topology).

    Routes global cell indices to shards, opens shard ``.npy`` files via
    ``np.load(..., mmap_mode="r")`` for OS-level paging, and returns flat
    ``ExpressionBatch`` data via ``read_expression_flat()``.

    Shards are opened lazily on first access and cached in
    ``self._mmaps`` per reader instance.  When used with
    ``torch.utils.data.DataLoader``, each worker process maintains its
    own set of memmap handles (fork-safe, read-only).

    Optional bounded shard LRU cache (Phase 4):
    When *cache_config* is provided with ``"enabled": True``, shard
    ``.npy`` files are first copied to a local scratch directory
    (e.g. ``/tmp``) before being opened via memmap.  The cache enforces
    a configurable byte limit with LRU eviction and is safe for
    multi-worker use (per-worker namespace or per-shard file locking).

    Parameters
    ----------
    entries : list of CsrMemmapShardEntry
        One entry per shard, sorted by ``global_start``.  Shard ranges
        must be contiguous and non-overlapping.
    cache_config : dict or None
        Optional shard cache configuration.  When ``None`` or
        ``{"enabled": False}``, reads go directly to source ``.npy``
        files via memmap.  When enabled, the config dict must contain:

        - ``"enabled"`` (bool): ``True`` to activate the cache.
        - ``"cache_root"`` (str or Path): local scratch directory.
        - ``"max_bytes"`` (int): cache capacity in bytes (e.g.
          ``20_000_000_000`` for 20 GB).
        - ``"per_worker"`` (bool, optional): if ``True``, each worker
          gets its own pid-based subdirectory.  Default ``False``.
    """

    def __init__(
        self,
        entries: list[CsrMemmapShardEntry],
        *,
        cache_config: object | None = None,
    ):
        super().__init__(list(entries))  # type: ignore[arg-type]
        self._cache_config = cache_config
        # Phase 4: build the optional ShardLRUCache
        self._cache: ShardLRUCache | None = _build_cache(cache_config)
        # Lazily resolved shard paths: shard_id → dict of file paths
        self._mmaps: dict[int, dict[str, Path]] = {}

    # ------------------------------------------------------------------
    # Shard opening (lazy, fork-safe, cache-aware)
    # ------------------------------------------------------------------

    def _open_shard(self, entry: CsrMemmapShardEntry) -> dict[str, Path]:
        """Return resolved shard file paths for *entry*, opening them lazily.

        When the optional shard LRU cache is active, ``.npy`` files are
        first copied to the local cache root.
        Otherwise, files are resolved directly from the source shard
        directory.
        """
        shard_id = entry.shard_id
        if shard_id not in self._mmaps:
            if self._cache is not None:
                # Read from local cache (copied on first access)
                local_dir = self._cache.get_shard_path(
                    shard_id, entry.shard_path
                )
                row_offsets_path = local_dir / "row_offsets.npy"
                gene_indices_path = local_dir / "gene_indices.npy"
                counts_path = local_dir / "counts.npy"
            else:
                # Read directly from the source shard directory
                row_offsets_path = entry.row_offsets_path
                gene_indices_path = entry.gene_indices_path
                counts_path = entry.counts_path

            self._mmaps[shard_id] = {
                "offsets": row_offsets_path,
                "indices": gene_indices_path,
                "counts": counts_path,
            }
        return self._mmaps[shard_id]

    def _load_shard_arrays(
        self, entry: CsrMemmapShardEntry
    ) -> dict[str, np.ndarray]:
        """Open one shard's arrays as temporary read-only memmaps."""
        paths = self._open_shard(entry)
        return {
            "offsets": np.load(str(paths["offsets"]), mmap_mode="r"),
            "indices": np.load(str(paths["indices"]), mmap_mode="r"),
            "counts": np.load(str(paths["counts"]), mmap_mode="r"),
        }

    # ------------------------------------------------------------------
    # Backend-specific hook
    # ------------------------------------------------------------------

    def _read_local_cells(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Read per-cell expression arrays from a single shard."""
        csr_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(csr_entry, CsrMemmapShardEntry)
        arrays = self._load_shard_arrays(csr_entry)
        offsets = arrays["offsets"]
        indices = arrays["indices"]
        counts = arrays["counts"]
        cells: list[tuple[np.ndarray, np.ndarray]] = []
        for li in local_indices:
            s = slice(int(offsets[li]), int(offsets[li + 1]))
            cells.append(
                (
                    np.array(indices[s], dtype=np.int32, copy=True),
                    np.array(counts[s], dtype=np.int32, copy=True),
                )
            )
        return cells

    # ------------------------------------------------------------------
    # Fast path — direct flat read (Phase 3)
    # ------------------------------------------------------------------

    def read_expression_flat(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data directly as flat arrays.

        Groups indices by shard via ``_resolve_entry()``, reads per-shard
        flat arrays from the memmap, and reassembles in the original
        input order.

        This is the aggregate CSR fast path that bypasses the shared base
        implementation for direct shard-oriented assembly.

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
            If any index is out of range.
        """
        if not global_indices:
            return _empty_expression_batch()

        indices = [int(idx) for idx in global_indices]
        n = len(indices)
        self._validate_all(indices)

        # Group by shard, preserving the output position of each index.
        # grouped[shard_dataset_id] = list[(output_pos, local_index)]
        grouped: dict[str, list[tuple[int, int]]] = {}
        for output_pos, global_idx in enumerate(indices):
            entry, local_idx = self._resolve_entry(global_idx)
            grouped.setdefault(entry.dataset_id, []).append(
                (output_pos, local_idx)
            )

        # Accumulate per-cell flat array parts indexed by output position
        egi_by_pos: dict[int, np.ndarray] = {}
        ec_by_pos: dict[int, np.ndarray] = {}

        for ds_id, selections in grouped.items():
            csr_entry = self._find_entry_by_id(ds_id)
            assert isinstance(csr_entry, CsrMemmapShardEntry)
            arrays = self._load_shard_arrays(csr_entry)
            offsets = arrays["offsets"]
            indices_arr = arrays["indices"]
            counts_arr = arrays["counts"]

            for output_pos, local_idx in selections:
                s = slice(int(offsets[local_idx]), int(offsets[local_idx + 1]))
                egi_by_pos[output_pos] = np.array(
                    indices_arr[s], dtype=np.int32, copy=True
                )
                ec_by_pos[output_pos] = np.array(
                    counts_arr[s], dtype=np.int32, copy=True
                )

        # Build row offsets and concatenate in output order
        row_offsets = np.zeros(n + 1, dtype=np.int64)
        egi_parts: list[np.ndarray] = []
        ec_parts: list[np.ndarray] = []
        for pos in range(n):
            egi_parts.append(egi_by_pos[pos])
            ec_parts.append(ec_by_pos[pos])
            row_offsets[pos + 1] = (
                row_offsets[pos] + len(egi_by_pos[pos])
            )

        return ExpressionBatch(
            batch_size=n,
            global_row_index=np.array(indices, dtype=np.int64),
            row_offsets=row_offsets,
            expressed_gene_indices=np.concatenate(egi_parts),
            expression_counts=np.concatenate(ec_parts),
        )


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
            raise ValueError(
                f"Backend '{backend}' is not supported in slim main. Only "
                "'lance' and 'zarr' readers remain available."
            )
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
