"""Sharded CSR memmap writer and manifest support.

Provides an incremental writer that builds immutable CSR shard directories
from streaming cell data.  Each shard is a self-contained directory with
memmap-compatible ``.npy`` files and a ``shard-manifest.yaml``.

Usage::

    writer = CsrMemmapWriter(output_dir=Path("/out"), shard_n_cells=100_000)
    global_ids = writer.append_cells(
        gene_indices_list=[np.array([0, 1], dtype=np.int32), ...],
        counts_list=[np.array([5, 3], dtype=np.int32), ...],
    )
    manifest_path = writer.finalize()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROW_OFFSETS_DTYPE: np.dtype = np.dtype("int64")
_GENE_INDICES_DTYPE: np.dtype = np.dtype("int32")
_COUNTS_DTYPE: np.dtype = np.dtype("int32")

_CONTRACT_VERSION: str = "0.1.0"

_MANIFEST_FILE: str = "csr-corpus-manifest.yaml"
_SHARD_MANIFEST_FILE: str = "shard-manifest.yaml"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _ShardRecord:
    """Internal record of a finalized shard for manifest generation."""

    shard_id: int
    path: Path
    global_start: int
    global_end: int
    n_cells: int
    total_nnz: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_cell_data(
    gene_indices: np.ndarray, counts: np.ndarray
) -> None:
    """Validate dtype and length consistency of one cell's data.

    Raises
    ------
    TypeError
        If dtypes are not int32.
    ValueError
        If lengths differ.
    """
    if gene_indices.dtype != _GENE_INDICES_DTYPE:
        raise TypeError(
            f"gene_indices must be dtype {_GENE_INDICES_DTYPE}, "
            f"got {gene_indices.dtype}"
        )
    if counts.dtype != _COUNTS_DTYPE:
        raise TypeError(
            f"counts must be dtype {_COUNTS_DTYPE}, "
            f"got {counts.dtype}"
        )
    if len(gene_indices) != len(counts):
        raise ValueError(
            f"Mismatched lengths: gene_indices has {len(gene_indices)} "
            f"elements but counts has {len(counts)}"
        )


def _validate_csr_chunk_data(
    indptr: np.ndarray,
    indices: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Validate a CSR chunk before appending it to the writer."""
    if indptr.dtype != _ROW_OFFSETS_DTYPE:
        raise TypeError(
            f"indptr must be dtype {_ROW_OFFSETS_DTYPE}, got {indptr.dtype}"
        )
    if indices.dtype != _GENE_INDICES_DTYPE:
        raise TypeError(
            f"indices must be dtype {_GENE_INDICES_DTYPE}, got {indices.dtype}"
        )
    if counts.dtype != _COUNTS_DTYPE:
        raise TypeError(
            f"counts must be dtype {_COUNTS_DTYPE}, got {counts.dtype}"
        )
    if indptr.ndim != 1:
        raise ValueError(f"indptr must be 1-D, got shape {indptr.shape}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1-D, got shape {indices.shape}")
    if counts.ndim != 1:
        raise ValueError(f"counts must be 1-D, got shape {counts.shape}")
    if len(indices) != len(counts):
        raise ValueError(
            f"indices and counts must have the same length, got "
            f"{len(indices)} vs {len(counts)}"
        )
    if len(indptr) == 0:
        raise ValueError("indptr must contain at least one element")
    if int(indptr[0]) != 0:
        raise ValueError(f"indptr[0] must be 0, got {int(indptr[0])}")
    if np.any(np.diff(indptr) < 0):
        raise ValueError("indptr must be non-decreasing")
    if int(indptr[-1]) != len(indices):
        raise ValueError(
            f"indptr[-1] ({int(indptr[-1])}) must equal len(indices) "
            f"({len(indices)})"
        )


def _write_shard_manifest(
    shard_dir: Path,
    shard_id: int,
    n_cells: int,
    total_nnz: int,
    row_offsets_shape: tuple[int, ...],
    gene_indices_shape: tuple[int, ...],
    counts_shape: tuple[int, ...],
) -> None:
    """Write a ``shard-manifest.yaml`` into *shard_dir*."""
    manifest_path = shard_dir / _SHARD_MANIFEST_FILE
    data: dict[str, Any] = {
        "kind": "csr-shard-manifest",
        "contract_version": _CONTRACT_VERSION,
        "shard_id": shard_id,
        "n_cells": n_cells,
        "total_nnz": total_nnz,
        "row_offsets_dtype": str(_ROW_OFFSETS_DTYPE),
        "gene_indices_dtype": str(_GENE_INDICES_DTYPE),
        "counts_dtype": str(_COUNTS_DTYPE),
        "row_offsets_shape": list(row_offsets_shape),
        "gene_indices_shape": list(gene_indices_shape),
        "counts_shape": list(counts_shape),
    }
    with open(manifest_path, "w") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


def _write_corpus_manifest(
    manifest_path: Path,
    total_cells: int,
    total_nnz: int,
    shard_n_cells_target: int,
    source_corpus_root: Path | None,
    shard_records: list[_ShardRecord],
) -> None:
    """Write the top-level ``csr-corpus-manifest.yaml``."""
    shards: list[dict[str, Any]] = []
    for rec in shard_records:
        shards.append(
            {
                "shard_id": rec.shard_id,
                "path": rec.path.name,  # relative to corpus root
                "global_start": rec.global_start,
                "global_end": rec.global_end,
                "n_cells": rec.n_cells,
                "total_nnz": rec.total_nnz,
            }
        )

    data: dict[str, Any] = {
        "kind": "csr-corpus-manifest",
        "contract_version": _CONTRACT_VERSION,
        "total_cells": total_cells,
        "total_nnz": total_nnz,
        "shard_n_cells_target": shard_n_cells_target,
        "n_shards": len(shard_records),
        "shards": shards,
    }
    if source_corpus_root is not None:
        data["source_corpus_root"] = str(source_corpus_root)

    with open(manifest_path, "w") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# CsrMemmapWriter
# ---------------------------------------------------------------------------


class CsrMemmapWriter:
    """Incrementally write cells into fixed-capacity CSR shards.

    Cells are assigned contiguous global indices starting from 0.
    When the current shard reaches ``shard_n_cells`` it is finalized as
    an immutable directory containing:

    - ``row_offsets.npy`` (int64, shape ``(n_cells + 1,)``)
    - ``gene_indices.npy`` (int32, flat, shape ``(total_nnz,)``)
    - ``counts.npy`` (int32, flat, shape ``(total_nnz,)``)
    - ``shard-manifest.yaml``

    After all cells have been appended, call ``finalize()`` to flush
    the last partial shard and write the top-level
    ``csr-corpus-manifest.yaml``.

    Parameters
    ----------
    output_dir : Path
        Root directory for the CSR corpus.  Shard directories and the
        corpus manifest will be created inside this directory.
    shard_n_cells : int
        Maximum number of cells per shard.  When a shard reaches this
        count it is finalized and a new shard is started.
    source_corpus_root : Path, optional
        If provided, recorded in the corpus manifest for provenance.
    """

    def __init__(
        self,
        output_dir: Path,
        shard_n_cells: int,
        *,
        source_corpus_root: Path | None = None,
        global_row_start: int = 0,
    ):
        if shard_n_cells <= 0:
            raise ValueError(
                f"shard_n_cells must be positive, got {shard_n_cells}"
            )
        if global_row_start < 0:
            raise ValueError(
                f"global_row_start must be non-negative, got {global_row_start}"
            )

        self._output_dir = Path(output_dir)
        self._shard_n_cells = shard_n_cells
        self._source_corpus_root = source_corpus_root
        self._global_row_start = global_row_start
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Shard state
        self._shard_records: list[_ShardRecord] = []
        self._next_shard_id: int = 0
        self._total_cells_written: int = 0
        self._total_nnz_written: int = 0
        self._finalized: bool = False

        # Current shard accumulation buffers
        self._cur_shard_cell_count: int = 0
        self._cur_shard_nnz: int = 0
        self._cur_shard_row_offset_parts: list[np.ndarray] = []
        self._cur_shard_gene_parts: list[np.ndarray] = []
        self._cur_shard_count_parts: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_cells(
        self,
        gene_indices_list: list[np.ndarray],
        counts_list: list[np.ndarray],
    ) -> list[int]:
        """Write a batch of cells.  Returns the assigned global indices.

        Each cell is described by a pair of equal-length int32 arrays
        (``gene_indices`` and corresponding ``counts``).  Cells are
        assigned contiguous global indices starting from 0 and filling
        the current shard until it reaches ``shard_n_cells``, at which
        point the shard is finalized and a new shard is started.

        Parameters
        ----------
        gene_indices_list : list of np.ndarray
            One int32 array per cell with gene indices for that cell.
        counts_list : list of np.ndarray
            One int32 array per cell with expression counts.
            Must have the same length as the corresponding entry in
            *gene_indices_list*.

        Returns
        -------
        list of int
            Assigned global indices (zero-based, contiguous).

        Raises
        ------
        RuntimeError
            If ``finalize()`` has already been called.
        TypeError
            If any array has an incorrect dtype.
        ValueError
            If the two lists have different lengths, or if any pair of
            arrays has mismatched lengths.
        """
        self._check_not_finalized()

        n_cells = len(gene_indices_list)
        if n_cells != len(counts_list):
            raise ValueError(
                f"gene_indices_list and counts_list must have the same "
                f"length, got {n_cells} vs {len(counts_list)}"
            )

        # Validate all cells before writing any
        for gi, cnt in zip(gene_indices_list, counts_list):
            _validate_cell_data(gi, cnt)

        assigned: list[int] = []

        for gene_indices, counts in zip(gene_indices_list, counts_list):
            # If the current shard is full, finalize it first
            if self._cur_shard_cell_count >= self._shard_n_cells:
                self._flush_shard()

            global_idx = self._global_row_start + self._total_cells_written
            assigned.append(global_idx)

            self._append_chunk_part(
                indptr=np.array([0, len(gene_indices)], dtype=_ROW_OFFSETS_DTYPE),
                indices=np.asarray(gene_indices, dtype=_GENE_INDICES_DTYPE),
                counts=np.asarray(counts, dtype=_COUNTS_DTYPE),
            )

            # Flush if this cell filled the shard
            if self._cur_shard_cell_count >= self._shard_n_cells:
                self._flush_shard()

        return assigned

    def append_csr_chunk(
        self,
        global_row_start: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        counts: np.ndarray,
    ) -> tuple[int, int]:
        """Append a chunk of CSR rows without per-cell Python unpacking.

        Parameters
        ----------
        global_row_start : int
            Global row index of the first row in this chunk. Must match the
            writer's next expected row to preserve manifest continuity.
        indptr : np.ndarray
            Chunk-local CSR row offsets (int64, length ``n_rows + 1``).
        indices : np.ndarray
            Flat int32 gene indices for the chunk.
        counts : np.ndarray
            Flat int32 expression counts for the chunk.

        Returns
        -------
        tuple[int, int]
            Inclusive/exclusive global row range written by this call.
        """
        self._check_not_finalized()
        _validate_csr_chunk_data(indptr, indices, counts)

        expected_start = self._global_row_start + self._total_cells_written
        if global_row_start != expected_start:
            raise ValueError(
                f"global_row_start {global_row_start} does not match next "
                f"expected global row {expected_start}"
            )

        row_count = len(indptr) - 1
        if row_count == 0:
            return global_row_start, global_row_start

        row_cursor = 0
        while row_cursor < row_count:
            if self._cur_shard_cell_count >= self._shard_n_cells:
                self._flush_shard()

            rows_available = self._shard_n_cells - self._cur_shard_cell_count
            rows_to_take = min(rows_available, row_count - row_cursor)

            data_start = int(indptr[row_cursor])
            data_stop = int(indptr[row_cursor + rows_to_take])
            chunk_indptr = (
                np.asarray(
                    indptr[row_cursor : row_cursor + rows_to_take + 1],
                    dtype=_ROW_OFFSETS_DTYPE,
                )
                - data_start
            )
            chunk_indices = np.asarray(
                indices[data_start:data_stop], dtype=_GENE_INDICES_DTYPE
            )
            chunk_counts = np.asarray(
                counts[data_start:data_stop], dtype=_COUNTS_DTYPE
            )

            self._append_chunk_part(
                indptr=chunk_indptr,
                indices=chunk_indices,
                counts=chunk_counts,
            )
            row_cursor += rows_to_take

            if self._cur_shard_cell_count >= self._shard_n_cells:
                self._flush_shard()

        return global_row_start, global_row_start + row_count

    def finalize(self) -> Path:
        """Finalize the last shard (if any partial data) and write the corpus
        manifest.

        After this call, ``append_cells`` will raise ``RuntimeError``.

        Returns
        -------
        Path
            Path to the ``csr-corpus-manifest.yaml`` file.
        """
        if self._finalized:
            return self._output_dir / _MANIFEST_FILE

        # Flush the last partial shard if it has any cells
        if self._cur_shard_cell_count > 0:
            self._flush_shard()

        # Write the corpus-level manifest
        manifest_path = self._output_dir / _MANIFEST_FILE
        _write_corpus_manifest(
            manifest_path=manifest_path,
            total_cells=self._total_cells_written,
            total_nnz=self._total_nnz_written,
            shard_n_cells_target=self._shard_n_cells,
            source_corpus_root=self._source_corpus_root,
            shard_records=self._shard_records,
        )

        self._finalized = True
        return manifest_path

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_cells_written(self) -> int:
        """Total number of cells appended so far."""
        return self._total_cells_written

    @property
    def n_shards(self) -> int:
        """Number of finalized shards (not including the current partial shard)."""
        return len(self._shard_records)

    @property
    def total_nnz_written(self) -> int:
        """Total number of non-zero entries appended so far."""
        return self._total_nnz_written

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_not_finalized(self) -> None:
        if self._finalized:
            raise RuntimeError(
                "CsrMemmapWriter has been finalized; cannot call append_cells()"
            )

    def _flush_shard(self) -> None:
        """Write the current in-memory shard to disk as an immutable directory."""
        if self._cur_shard_cell_count == 0:
            return

        shard_id = self._next_shard_id
        self._next_shard_id += 1

        shard_dir = self._output_dir / f"shard_{shard_id:06d}"
        shard_dir.mkdir(parents=True, exist_ok=False)

        n_cells = self._cur_shard_cell_count

        row_offsets = np.concatenate(
            [np.array([0], dtype=_ROW_OFFSETS_DTYPE)]
            + self._cur_shard_row_offset_parts
        )
        gene_indices_flat = (
            np.concatenate(self._cur_shard_gene_parts)
            if self._cur_shard_gene_parts
            else np.array([], dtype=_GENE_INDICES_DTYPE)
        )
        counts_flat = (
            np.concatenate(self._cur_shard_count_parts)
            if self._cur_shard_count_parts
            else np.array([], dtype=_COUNTS_DTYPE)
        )
        total_nnz = self._cur_shard_nnz

        # Write .npy files
        np.save(str(shard_dir / "row_offsets.npy"), row_offsets)
        np.save(str(shard_dir / "gene_indices.npy"), gene_indices_flat)
        np.save(str(shard_dir / "counts.npy"), counts_flat)

        # Write shard manifest
        _write_shard_manifest(
            shard_dir=shard_dir,
            shard_id=shard_id,
            n_cells=n_cells,
            total_nnz=total_nnz,
            row_offsets_shape=row_offsets.shape,
            gene_indices_shape=gene_indices_flat.shape,
            counts_shape=counts_flat.shape,
        )

        # Record for corpus manifest
        global_start = self._global_row_start + self._total_cells_written - n_cells
        global_end = self._global_row_start + self._total_cells_written
        self._shard_records.append(
            _ShardRecord(
                shard_id=shard_id,
                path=shard_dir,
                global_start=global_start,
                global_end=global_end,
                n_cells=n_cells,
                total_nnz=total_nnz,
            )
        )

        # Reset current shard buffers
        self._cur_shard_cell_count = 0
        self._cur_shard_nnz = 0
        self._cur_shard_row_offset_parts = []
        self._cur_shard_gene_parts = []
        self._cur_shard_count_parts = []

    def _append_chunk_part(
        self,
        *,
        indptr: np.ndarray,
        indices: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        """Append one normalized CSR sub-chunk into the active shard buffers."""
        row_count = len(indptr) - 1
        if row_count < 0:
            raise ValueError("indptr must have length >= 1")

        if row_count:
            self._cur_shard_row_offset_parts.append(
                np.asarray(indptr[1:] + self._cur_shard_nnz, dtype=_ROW_OFFSETS_DTYPE)
            )
        if len(indices):
            self._cur_shard_gene_parts.append(np.asarray(indices, dtype=_GENE_INDICES_DTYPE))
            self._cur_shard_count_parts.append(np.asarray(counts, dtype=_COUNTS_DTYPE))

        self._cur_shard_cell_count += row_count
        self._total_cells_written += row_count
        self._cur_shard_nnz += len(indices)
        self._total_nnz_written += len(indices)
