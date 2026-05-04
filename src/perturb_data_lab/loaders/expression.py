"""Phase 2: Backend-agnostic expression readers.

Defines the ``ExpressionReader`` protocol, a ``BaseExpressionReader`` abstract
class that handles global→local routing and order-preserving reassembly, and
backend-specific implementations for all supported backends.

Supported backends: Lance, Zarr, Arrow IPC, Parquet, WebDataset.
Supported topologies: aggregate, federated.

The reader returns **only expression data** — no metadata fields.
Identity/metadata fields (dataset_id, dataset_index, local_row_index,
size_factor, etc.) belong in ``MetadataIndex``, not in ``ExpressionRow``.
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Protocol, Sequence

import numpy as np

from ..materializers.backends.csr_cache import ShardLRUCache
from .loaders import ExpressionBatch

__all__ = [
    "ExpressionRow",
    "ExpressionReader",
    "BaseExpressionReader",
    "AggregateLanceReader",
    "FederatedLanceReader",
    "AggregateZarrReader",
    "FederatedZarrReader",
    "FederatedArrowIpcReader",
    "FederatedParquetReader",
    "FederatedWebDatasetReader",
    "AggregateCsrMemmapReader",
    "LanceDatasetEntry",
    "ZarrDatasetEntry",
    "ArrowIpcDatasetEntry",
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
class ExpressionRow:
    """A single row of expression data for one cell.

    Contains only expression data — no metadata.  Identity and metadata
    fields are handled separately by ``MetadataIndex``.

    Fields
    ------
    global_row_index : int
        Corpus-level global cell index.
    expressed_gene_indices : np.ndarray
        1-D int32 array of gene indices (dataset-local feature space).
    expression_counts : np.ndarray
        1-D int32 array of corresponding expression counts.
    """

    global_row_index: int
    expressed_gene_indices: np.ndarray  # dtype int32
    expression_counts: np.ndarray  # dtype int32


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

    Implementations must return expression data **only**.  Metadata
    enrichment is handled by ``MetadataIndex`` + ``BatchExecutor``.
    """

    def read_expression(self, global_indices: Sequence[int]) -> list[ExpressionRow]:
        """Read expression data for the given global cell indices.

        Returns one ``ExpressionRow`` per input index, in the same order.
        """
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

    def read_expression(self, global_indices: Sequence[int]) -> list[ExpressionRow]:
        """Read expression data for the given global cell indices.

        Routes each index to its owning dataset, reads through the
        backend-specific ``_read_local_rows``, and reassembles results
        in the original input order.
        """
        if not global_indices:
            return []

        indices = [int(idx) for idx in global_indices]
        self._validate_all(indices)

        # Group by dataset, preserving the output position of each index
        # grouped[dataset_id] = list[(output_pos, local_index)]
        grouped: dict[str, list[tuple[int, int]]] = {}
        for output_pos, global_idx in enumerate(indices):
            entry, local_idx = self._resolve_entry(global_idx)
            grouped.setdefault(entry.dataset_id, []).append(
                (output_pos, local_idx)
            )

        # Read per-dataset and map back to output positions
        result_map: dict[int, ExpressionRow] = {}
        for ds_id, selections in grouped.items():
            entry = self._find_entry_by_id(ds_id)
            local_indices = [local for _, local in selections]
            rows = self._read_local_rows(entry, local_indices)
            for i, row in enumerate(rows):
                result_map[selections[i][0]] = row

        # Reassemble in original input order
        return [result_map[pos] for pos in range(len(indices))]

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
    def _read_local_rows(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[ExpressionRow]:
        """Read expression rows for *local_indices* within a single dataset.

        For aggregate topology, *local_indices* equal *global_indices*
        (the single entry covers the full range).

        For federated topology, *local_indices* are 0-based offsets
        within the dataset file.

        Subclasses implement backend-specific I/O here.
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
        import lance

        self._dataset = lance.dataset(lance_path)

    def _validate_all(self, indices: list[int]) -> None:
        super()._validate_all(indices)
        total_rows = self._dataset.count_rows()
        for idx in indices:
            if idx >= total_rows:
                raise IndexError(
                    f"global_index {idx} out of range [0, {total_rows})"
                )

    def _read_local_rows(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[ExpressionRow]:
        # Convert local (per-dataset offset) back to global positions
        # for the aggregate Lance file, which stores rows at global positions.
        global_positions = [entry.global_start + li for li in local_indices]
        rows: list[ExpressionRow] = []
        for chunk in _chunk_indices(global_positions):
            table = self._dataset.take(chunk)
            rows.extend(self._table_to_rows(chunk, table))
        return rows

    @staticmethod
    def _table_to_rows(
        requested_indices: list[int], table
    ) -> list[ExpressionRow]:
        """Convert a Lance table into ``ExpressionRow`` objects."""
        n = len(requested_indices)
        egi_offsets, egi_flat = _extract_list_columns(
            table, "expressed_gene_indices"
        )
        ec_offsets, ec_flat = _extract_list_columns(
            table, "expression_counts"
        )

        rows: list[ExpressionRow] = []
        for i in range(n):
            global_idx = requested_indices[i]
            s_egi = slice(egi_offsets[i], egi_offsets[i + 1])
            s_ec = slice(ec_offsets[i], ec_offsets[i + 1])
            rows.append(
                ExpressionRow(
                    global_row_index=global_idx,
                    expressed_gene_indices=egi_flat[s_egi].copy(),
                    expression_counts=ec_flat[s_ec].copy(),
                )
            )
        return rows

    # ------------------------------------------------------------------
    # Fast path — direct flat read (Phase 2)
    # ------------------------------------------------------------------

    def read_expression_flat(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data directly as flat arrays (aggregate fast path).

        Bypasses the per-dataset regrouping in
        ``BaseExpressionReader.read_expression()`` and avoids per-row
        ``ExpressionRow`` object construction.  Performs direct chunked
        ``take()`` calls on the aggregate Lance file and returns
        concatenated flat expression arrays with row offsets.

        This is the aggregate-specific fast path.  The generic
        ``read_expression()`` path (returning ``list[ExpressionRow]``)
        remains available for backward compatibility.

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
            return ExpressionBatch(
                batch_size=0,
                global_row_index=np.array([], dtype=np.int64),
                row_offsets=np.array([0], dtype=np.int64),
                expressed_gene_indices=np.array([], dtype=np.int32),
                expression_counts=np.array([], dtype=np.int32),
            )

        indices = [int(idx) for idx in global_indices]
        n = len(indices)

        # Validate all indices against Lance row count
        total_rows = self._dataset.count_rows()
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
            table = self._dataset.take(chunk)
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

    def _open_dataset(self, entry: LanceDatasetEntry):
        import lance

        if entry.dataset_id not in self._datasets:
            self._datasets[entry.dataset_id] = lance.dataset(entry.lance_path)
        return self._datasets[entry.dataset_id]

    def _read_local_rows(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[ExpressionRow]:
        lance_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(lance_entry, LanceDatasetEntry)
        ds = self._open_dataset(lance_entry)

        rows: list[ExpressionRow] = []
        for chunk in _chunk_indices(local_indices):
            table = ds.take(chunk)
            rows.extend(
                self._table_to_rows(
                    chunk, lance_entry.global_start, table
                )
            )
        return rows

    @staticmethod
    def _table_to_rows(
        requested_local_indices: list[int],
        global_start: int,
        table,
    ) -> list[ExpressionRow]:
        """Convert a Lance table into ``ExpressionRow`` objects.

        Rows are in the same order as ``requested_local_indices``.
        """
        n = len(requested_local_indices)
        egi_offsets, egi_flat = _extract_list_columns(
            table, "expressed_gene_indices"
        )
        ec_offsets, ec_flat = _extract_list_columns(
            table, "expression_counts"
        )

        rows: list[ExpressionRow] = []
        for i in range(n):
            global_idx = global_start + requested_local_indices[i]
            s_egi = slice(egi_offsets[i], egi_offsets[i + 1])
            s_ec = slice(ec_offsets[i], ec_offsets[i + 1])
            rows.append(
                ExpressionRow(
                    global_row_index=global_idx,
                    expressed_gene_indices=egi_flat[s_egi].copy(),
                    expression_counts=ec_flat[s_ec].copy(),
                )
            )
        return rows

    # ------------------------------------------------------------------
    # Fast path — direct flat read (Phase 3)
    # ------------------------------------------------------------------

    def read_expression_flat(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data directly as flat arrays (federated fast path).

        Groups indices by dataset/file, performs chunked ``take()`` calls
        per dataset, and returns concatenated flat expression arrays with
        row offsets.  Avoids per-row ``ExpressionRow`` object construction.

        This path retains per-dataset grouping (federated topology cannot
        avoid it) but eliminates the de-batch/re-batch overhead of
        ``ExpressionRow`` → flat-array reconstruction.

        The generic ``read_expression()`` path (returning
        ``list[ExpressionRow]``) remains available for backward
        compatibility.

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
            return ExpressionBatch(
                batch_size=0,
                global_row_index=np.array([], dtype=np.int64),
                row_offsets=np.array([0], dtype=np.int64),
                expressed_gene_indices=np.array([], dtype=np.int32),
                expression_counts=np.array([], dtype=np.int32),
            )

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


def _zarr_read_rows(
    offsets: np.ndarray,
    indices_flat: np.ndarray,
    counts_flat: np.ndarray,
    global_start: int,
    local_indices: list[int],
) -> list[ExpressionRow]:
    """Read expression rows from CSR-format Zarr arrays."""
    rows: list[ExpressionRow] = []
    for local_idx in local_indices:
        global_idx = global_start + local_idx
        s = slice(offsets[local_idx], offsets[local_idx + 1])
        rows.append(
            ExpressionRow(
                global_row_index=global_idx,
                expressed_gene_indices=indices_flat[s].copy(),
                expression_counts=counts_flat[s].copy(),
            )
        )
    return rows


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

    def _read_local_rows(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[ExpressionRow]:
        return _zarr_read_rows(
            self._offsets,
            self._indices,
            self._counts,
            entry.global_start,
            local_indices,
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

    def _read_local_rows(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[ExpressionRow]:
        zarr_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(zarr_entry, ZarrDatasetEntry)
        offsets, indices_arr, counts_arr = self._open_arrays(zarr_entry)
        return _zarr_read_rows(
            offsets, indices_arr, counts_arr, entry.global_start, local_indices
        )


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

    def _read_local_rows(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[ExpressionRow]:
        arrow_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(arrow_entry, ArrowIpcDatasetEntry)
        table = self._open_table(arrow_entry)

        rows: list[ExpressionRow] = []
        for local_idx in local_indices:
            global_idx = entry.global_start + local_idx
            egi = _list_column_as_py(table, "expressed_gene_indices", local_idx)
            ec = _list_column_as_py(table, "expression_counts", local_idx)
            rows.append(
                ExpressionRow(
                    global_row_index=global_idx,
                    expressed_gene_indices=np.array(egi, dtype=np.int32),
                    expression_counts=np.array(ec, dtype=np.int32),
                )
            )
        return rows


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

    def _read_local_rows(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[ExpressionRow]:
        pq_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(pq_entry, ParquetDatasetEntry)
        table = self._open_table(pq_entry)

        rows: list[ExpressionRow] = []
        for local_idx in local_indices:
            global_idx = entry.global_start + local_idx
            egi = _list_column_as_py(table, "expressed_gene_indices", local_idx)
            ec = _list_column_as_py(table, "expression_counts", local_idx)
            rows.append(
                ExpressionRow(
                    global_row_index=global_idx,
                    expressed_gene_indices=np.array(egi, dtype=np.int32),
                    expression_counts=np.array(ec, dtype=np.int32),
                )
            )
        return rows


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

    def _read_local_rows(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[ExpressionRow]:
        import pickle

        wds_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(wds_entry, WebDatasetEntry)
        tar = self._open_tar(wds_entry)

        rows: list[ExpressionRow] = []
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
            rows.append(
                ExpressionRow(
                    global_row_index=entry.global_start + local_idx,
                    expressed_gene_indices=np.asarray(
                        data["expressed_gene_indices"], dtype=np.int32
                    ),
                    expression_counts=np.asarray(
                        data["expression_counts"], dtype=np.int32
                    ),
                )
            )
        return rows


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
        # Lazily opened memmap handles: shard_id → dict of arrays
        self._mmaps: dict[int, dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Shard opening (lazy, fork-safe, cache-aware)
    # ------------------------------------------------------------------

    def _open_shard(self, entry: CsrMemmapShardEntry) -> dict[str, np.ndarray]:
        """Return memmap-backed arrays for *entry*, opening them lazily.

        When the optional shard LRU cache is active, ``.npy`` files are
        first copied to the local cache root and opened from there.
        Otherwise, files are opened directly from the source shard
        directory via ``np.load(..., mmap_mode="r")``.
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
                "offsets": np.load(str(row_offsets_path), mmap_mode="r"),
                "indices": np.load(str(gene_indices_path), mmap_mode="r"),
                "counts": np.load(str(counts_path), mmap_mode="r"),
            }
        return self._mmaps[shard_id]

    # ------------------------------------------------------------------
    # Backend-specific hook (ExpressionRow path)
    # ------------------------------------------------------------------

    def _read_local_rows(
        self, entry: DatasetEntry, local_indices: list[int]
    ) -> list[ExpressionRow]:
        """Read expression rows from a single shard.

        Opens the shard's memmap arrays and slices per-row CSR data
        using ``row_offsets``.
        """
        csr_entry = self._find_entry_by_id(entry.dataset_id)
        assert isinstance(csr_entry, CsrMemmapShardEntry)
        arrays = self._open_shard(csr_entry)
        offsets = arrays["offsets"]
        indices = arrays["indices"]
        counts = arrays["counts"]
        global_start = csr_entry.global_start

        rows: list[ExpressionRow] = []
        for li in local_indices:
            s = slice(int(offsets[li]), int(offsets[li + 1]))
            rows.append(
                ExpressionRow(
                    global_row_index=global_start + li,
                    expressed_gene_indices=np.asarray(indices[s], dtype=np.int32),
                    expression_counts=np.asarray(counts[s], dtype=np.int32),
                )
            )
        return rows

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

        This is the aggregate CSR fast path that avoids per-row
        ``ExpressionRow`` object construction.  The generic
        ``read_expression()`` path (returning ``list[ExpressionRow]``)
        remains available for backward compatibility.

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
            return ExpressionBatch(
                batch_size=0,
                global_row_index=np.array([], dtype=np.int64),
                row_offsets=np.array([0], dtype=np.int64),
                expressed_gene_indices=np.array([], dtype=np.int32),
                expression_counts=np.array([], dtype=np.int32),
            )

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
            arrays = self._open_shard(csr_entry)
            offsets = arrays["offsets"]
            indices_arr = arrays["indices"]
            counts_arr = arrays["counts"]

            for output_pos, local_idx in selections:
                s = slice(int(offsets[local_idx]), int(offsets[local_idx + 1]))
                egi_by_pos[output_pos] = np.asarray(
                    indices_arr[s], dtype=np.int32
                )
                ec_by_pos[output_pos] = np.asarray(
                    counts_arr[s], dtype=np.int32
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
        One of ``"lance"``, ``"zarr"``, ``"arrow_ipc"``, ``"parquet"``,
        ``"webdataset"``, ``"csr_memmap"``.
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

    elif backend == "arrow_ipc":
        if topology == "aggregate":
            raise ValueError("Arrow IPC aggregate topology is not supported.")
        return FederatedArrowIpcReader(entries)  # type: ignore[arg-type]

    elif backend == "parquet":
        if topology == "aggregate":
            raise ValueError("Parquet aggregate topology is not supported.")
        return FederatedParquetReader(entries)  # type: ignore[arg-type]

    elif backend == "webdataset":
        if topology == "aggregate":
            raise ValueError("WebDataset aggregate topology is not supported.")
        return FederatedWebDatasetReader(entries)  # type: ignore[arg-type]

    elif backend == "csr_memmap":
        if topology == "aggregate":
            cache_config = kwargs.pop("cache_config", None)
            if kwargs:
                raise TypeError(
                    f"Unexpected keyword arguments for csr_memmap backend: "
                    f"{sorted(kwargs.keys())}"
                )
            return AggregateCsrMemmapReader(entries, cache_config=cache_config)  # type: ignore[arg-type]
        else:
            raise ValueError(
                "csr_memmap only supports aggregate topology "
                "(sharding is internal to the backend)."
            )

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Supported: lance, zarr, arrow_ipc, parquet, webdataset, csr_memmap."
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
