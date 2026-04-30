"""Phase 3: BatchExecutor — composes MetadataIndex + ExpressionReader.

The ``BatchExecutor`` is the single entry point for reading cell data from a
corpus.  It queries metadata from ``MetadataIndex`` and expression data from
an ``ExpressionReader``, returning flat numpy array dicts via ``read_batch()``.

Key design rules:
- Metadata comes from ``MetadataIndex``, **not** from expression readers.
- Expression readers return only the 3-field ``ExpressionRow``.
- ``read_batch()`` returns flat dicts with zero per-cell Python objects.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .index import MetadataIndex
from .expression import ExpressionReader
from .loaders import (
    ExpressionBatch,
)

__all__ = [
    "BatchExecutor",
]


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

        # Note: raw_ columns are extracted on-demand via _extract_canonical

    # ------------------------------------------------------------------
    # Public API — flat-batch, zero per-cell Python objects
    # ------------------------------------------------------------------

    def read_expression_batch(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data as a flat ``ExpressionBatch``.

        Unlike ``read_cells()``, this method produces **zero** per-cell
        Python objects.  Expression data stays as numpy arrays from the
        Lance reader through to the collator.

        Parameters
        ----------
        global_indices : sequence of int
            Corpus-global cell indices.

        Returns
        -------
        ExpressionBatch
            Flat expression batch with row offsets for per-cell slicing.
        """
        indices = [int(i) for i in global_indices]
        n = len(indices)
        if n == 0:
            return ExpressionBatch(
                batch_size=0,
                global_row_index=np.array([], dtype=np.int64),
                row_offsets=np.array([0], dtype=np.int64),
                expressed_gene_indices=np.array([], dtype=np.int32),
                expression_counts=np.array([], dtype=np.int32),
            )

        # Fetch expression rows (preserves order)
        expr_rows = self._reader.read_expression(indices)

        # Compute row offsets and concatenate flat arrays
        row_offsets = np.zeros(n + 1, dtype=np.int64)
        for i, row in enumerate(expr_rows):
            row_offsets[i + 1] = row_offsets[i] + len(row.expressed_gene_indices)

        flat_egi_parts: list[np.ndarray] = []
        flat_ec_parts: list[np.ndarray] = []
        for row in expr_rows:
            flat_egi_parts.append(row.expressed_gene_indices)
            flat_ec_parts.append(row.expression_counts)

        return ExpressionBatch(
            batch_size=n,
            global_row_index=np.array(
                [r.global_row_index for r in expr_rows], dtype=np.int64
            ),
            row_offsets=row_offsets,
            expressed_gene_indices=(
                np.concatenate(flat_egi_parts).astype(np.int32, copy=False)
                if flat_egi_parts
                else np.array([], dtype=np.int32)
            ),
            expression_counts=(
                np.concatenate(flat_ec_parts).astype(np.int32, copy=False)
                if flat_ec_parts
                else np.array([], dtype=np.int32)
            ),
        )

    def read_metadata_batch(
        self, global_indices: Sequence[int]
    ) -> dict[str, Any]:
        """Read metadata as columnar numpy arrays without per-cell objects.

        Each metadata column is extracted once (vectorized), not once per
        cell.  String fields (``dataset_id``, ``cell_id``) are returned as
        tuples.  Canonical perturbation and context dicts are still
        per-cell tuples of dicts (metadata, not GPU data).

        Parameters
        ----------
        global_indices : sequence of int
            Corpus-global cell indices.

        Returns
        -------
        dict
            Keys: ``global_row_index``, ``dataset_index``, ``local_row_index``,
            ``size_factor``, ``dataset_id``, ``cell_id``,
            ``canonical_perturbation``, ``canonical_context``.
        """
        indices = [int(i) for i in global_indices]
        n = len(indices)
        if n == 0:
            return {
                "global_row_index": np.array([], dtype=np.int64),
                "dataset_index": np.array([], dtype=np.int32),
                "local_row_index": np.array([], dtype=np.int64),
                "size_factor": np.array([], dtype=np.float32),
                "dataset_id": (),
                "cell_id": (),
                "canonical_perturbation": (),
                "canonical_context": (),
            }

        # Get metadata subset preserving positional order
        meta_subset = self._meta[indices]
        df = meta_subset.df

        # Extract scalar columns vectorized (one conversion per column)
        global_row_index = df["global_row_index"].to_numpy().astype(np.int64)
        dataset_index = df["dataset_index"].to_numpy().astype(np.int32)
        local_row_index = df["local_row_index"].to_numpy().astype(np.int64)
        size_factor = df["size_factor"].to_numpy().astype(np.float32)
        dataset_id = tuple(df["dataset_id"].to_list())
        cell_id = tuple(df["cell_id"].to_list())

        # Canonical dicts: extracted per-row (metadata, not GPU data)
        canonical_perturbation: list[dict[str, str]] = []
        canonical_context: list[dict[str, str]] = []
        for i in range(n):
            row = df.row(i, named=True)
            canonical_perturbation.append(
                self._extract_canonical(row, self._perturbation_keys)
            )
            canonical_context.append(
                self._extract_canonical(row, self._context_keys)
            )

        return {
            "global_row_index": global_row_index,
            "dataset_index": dataset_index,
            "local_row_index": local_row_index,
            "size_factor": size_factor,
            "dataset_id": dataset_id,
            "cell_id": cell_id,
            "canonical_perturbation": tuple(canonical_perturbation),
            "canonical_context": tuple(canonical_context),
        }

    def read_batch(
        self, global_indices: Sequence[int]
    ) -> dict[str, Any]:
        """Read expression + metadata as a GPU-ready dict.

        Composes ``read_expression_batch()`` and ``read_metadata_batch()``
        into a single dict with all numpy arrays needed for downstream
        collation and GPU transfer.

        This is the primary hot-path entry point.  No ``CellState`` objects
        are created.

        Parameters
        ----------
        global_indices : sequence of int
            Corpus-global cell indices.

        Returns
        -------
        dict
            Keys:
            - ``batch_size`` (int)
            - ``global_row_index`` (int64 array)
            - ``row_offsets`` (int64 array)
            - ``expressed_gene_indices`` (int32 flat array)
            - ``expression_counts`` (int32 flat array)
            - ``dataset_index`` (int32 array)
            - ``local_row_index`` (int64 array)
            - ``size_factor`` (float32 array)
            - ``dataset_id`` (tuple of str)
            - ``cell_id`` (tuple of str)
            - ``canonical_perturbation`` (tuple of dict)
            - ``canonical_context`` (tuple of dict)
        """
        expr = self.read_expression_batch(global_indices)
        meta = self.read_metadata_batch(global_indices)

        return {
            "batch_size": expr.batch_size,
            "global_row_index": expr.global_row_index,
            "row_offsets": expr.row_offsets,
            "expressed_gene_indices": expr.expressed_gene_indices,
            "expression_counts": expr.expression_counts,
            "dataset_index": meta["dataset_index"],
            "local_row_index": meta["local_row_index"],
            "size_factor": meta["size_factor"],
            "dataset_id": meta["dataset_id"],
            "cell_id": meta["cell_id"],
            "canonical_perturbation": meta["canonical_perturbation"],
            "canonical_context": meta["canonical_context"],
        }

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


