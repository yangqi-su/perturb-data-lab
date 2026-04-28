"""Phase 3: BatchExecutor — composes MetadataIndex + ExpressionReader.

The ``BatchExecutor`` is the single entry point for reading cell data from a
corpus.  It queries metadata from ``MetadataIndex`` and expression data from
an ``ExpressionReader``, then composes them into ``CellState`` objects.

Key design rules:
- Metadata comes from ``MetadataIndex``, **not** from expression readers.
- Expression readers return only the 3-field ``ExpressionRow``.
- Samplers operate on global indices and receive full ``CellState`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .index import MetadataIndex
from .expression import ExpressionReader
from .loaders import CellIdentity, CellState, SparseBatchCollator, SparseBatchPayload

__all__ = ["BatchExecutor"]


# ---------------------------------------------------------------------------
# Default mappings: which raw_ columns map to canonical_perturbation / context
# ---------------------------------------------------------------------------

_DEFAULT_PERTURBATION_KEYS: tuple[str, ...] = (
    "guide_1", "guide_2", "treatment", "site", "genotype",
)

_DEFAULT_CONTEXT_KEYS: tuple[str, ...] = (
    "cell_type", "cellline", "donor_id", "batch", "passage",
)


# ---------------------------------------------------------------------------
# BatchExecutor
# ---------------------------------------------------------------------------


class BatchExecutor:
    """Corpus-level batch reader composing metadata and expression I/O.

    Parameters
    ----------
    expression_reader : ExpressionReader
        Backend-agnostic expression reader (returns only expression data).
    metadata_index : MetadataIndex
        Polars-backed metadata index with flat columnar schema.
    perturbation_raw_keys : iterable of str, optional
        ``raw_``-column stems that map to ``canonical_perturbation`` dict.
        Default: ``("guide_1","guide_2","treatment","site","genotype")``.
    context_raw_keys : iterable of str, optional
        ``raw_``-column stems that map to ``canonical_context`` dict.
        Default: ``("cell_type","cellline","donor_id","batch","passage")``.
    """

    def __init__(
        self,
        expression_reader: ExpressionReader,
        metadata_index: MetadataIndex,
        *,
        perturbation_raw_keys: Sequence[str] | None = None,
        context_raw_keys: Sequence[str] | None = None,
    ):
        self._reader = expression_reader
        self._meta = metadata_index

        self._perturbation_keys: frozenset[str] = frozenset(
            perturbation_raw_keys if perturbation_raw_keys is not None
            else _DEFAULT_PERTURBATION_KEYS
        )
        self._context_keys: frozenset[str] = frozenset(
            context_raw_keys if context_raw_keys is not None
            else _DEFAULT_CONTEXT_KEYS
        )

        # Pre-warm: identify raw_-prefixed columns and their stems
        self._raw_cols: list[str] = [
            c for c in metadata_index.df.columns if c.startswith("raw_")
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_cells(self, global_indices: Sequence[int]) -> list[CellState]:
        """Read and compose ``CellState`` objects for the given global indices.

        Parameters
        ----------
        global_indices : sequence of int
            Corpus-global cell indices (0 ≤ idx < len(metadata_index)).

        Returns
        -------
        list[CellState]
            One ``CellState`` per input index, preserving input order.
        """
        indices = [int(i) for i in global_indices]
        n = len(indices)
        if n == 0:
            return []

        # 1. Fetch expression data (preserves order)
        expr_rows = self._reader.read_expression(indices)

        # 2. Fetch metadata via positional indexing (preserves order)
        meta_subset = self._meta[indices]

        # 3. Compose CellState by zipping expression + metadata
        cells: list[CellState] = []
        for i in range(n):
            cells.append(self._compose(expr_rows[i], meta_subset, i))
        return cells

    def collate_sparse_batch(
        self, global_indices: Sequence[int]
    ) -> SparseBatchPayload:
        """Read and collate a batch into a flat sparse payload.

        Convenience wrapper around ``read_cells`` + ``SparseBatchCollator``.

        Parameters
        ----------
        global_indices : sequence of int
            Corpus-global cell indices.

        Returns
        -------
        SparseBatchPayload
            Flat sparse batch ready for downstream processing.
        """
        collator = SparseBatchCollator()
        return collator(self.read_cells(global_indices))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metadata_index(self) -> MetadataIndex:
        """The backing metadata index."""
        return self._meta

    @property
    def expression_reader(self) -> ExpressionReader:
        """The backing expression reader."""
        return self._reader

    def __len__(self) -> int:
        return len(self._meta)

    # ------------------------------------------------------------------
    # Internal: composition
    # ------------------------------------------------------------------

    def _compose(
        self,
        expr_row,
        meta_subset: MetadataIndex,
        position: int,
    ) -> CellState:
        """Compose a single ``CellState`` from expression + metadata."""
        # Extract metadata row as a dict
        row = meta_subset.df.row(position, named=True)

        canonical_perturbation = self._extract_canonical(row, self._perturbation_keys)
        canonical_context = self._extract_canonical(row, self._context_keys)
        raw_fields = self._extract_remaining_raw(row)

        return CellState(
            identity=CellIdentity(
                global_row_index=int(row["global_row_index"]),
                dataset_index=int(row["dataset_index"]),
                dataset_id=str(row["dataset_id"]),
                local_row_index=int(row["local_row_index"]),
            ),
            cell_id=str(row["cell_id"]),
            expressed_gene_indices=tuple(
                int(i) for i in expr_row.expressed_gene_indices
            ),
            expression_counts=tuple(
                int(c) for c in expr_row.expression_counts
            ),
            size_factor=float(row["size_factor"]),
            canonical_perturbation=canonical_perturbation,
            canonical_context=canonical_context,
            raw_fields=raw_fields,
        )

    @staticmethod
    def _extract_canonical(row: dict[str, Any], keys: frozenset[str]) -> dict[str, str]:
        """Extract canonical dict from a metadata row.

        Looks for ``raw_<key>`` columns and maps them to ``{key: value}``.
        Skips None values.
        """
        result: dict[str, str] = {}
        for key in keys:
            col = f"raw_{key}"
            if col in row and row[col] is not None:
                result[key] = str(row[col])
        return result

    def _extract_remaining_raw(self, row: dict[str, Any]) -> dict[str, Any]:
        """Collect all raw_ columns not in perturbation or context keys."""
        all_canonical = self._perturbation_keys | self._context_keys
        result: dict[str, Any] = {}
        for col in self._raw_cols:
            stem = col.removeprefix("raw_")
            if stem in all_canonical:
                continue
            if col in row and row[col] is not None:
                result[stem] = row[col]
        return result
