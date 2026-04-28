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
    "LanceDatasetEntry",
    "ZarrDatasetEntry",
    "ArrowIpcDatasetEntry",
    "ParquetDatasetEntry",
    "WebDatasetEntry",
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
        ``"webdataset"``.
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

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Supported: lance, zarr, arrow_ipc, parquet, webdataset."
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
