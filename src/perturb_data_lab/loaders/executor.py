"""Phase 3: BatchExecutor — composes MetadataIndex + ExpressionReader.

The ``BatchExecutor`` is the single entry point for reading cell data from a
corpus.  It queries metadata from ``MetadataIndex`` and expression data from
an ``ExpressionReader``, returning flat numpy array dicts via ``read_batch()``.

Key design rules:
- Metadata comes from ``MetadataIndex``, **not** from expression readers.
- Expression readers return only the 3-field ``ExpressionRow``.
- ``read_batch()`` returns flat dicts with zero per-cell Python objects.
- When ``use_canonical=True``, canonical columns are read directly from the
  ``MetadataIndex`` DataFrame — no per-row dict extraction.
- Backward-compatible: ``use_canonical=False`` preserves raw-column extraction
  via the ``raw_`` prefix convention.
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
# Canonical field groupings (used when use_canonical=True)
# ---------------------------------------------------------------------------

_CANONICAL_PERTURBATION_FIELDS: tuple[str, ...] = (
    "perturb_label", "perturb_type", "dose", "dose_unit",
    "timepoint", "timepoint_unit",
)

_CANONICAL_CONTEXT_FIELDS: tuple[str, ...] = (
    "cell_context", "cell_line_or_type", "species", "tissue",
    "assay", "condition", "batch_id", "donor_id", "sex",
    "disease_state",
)


# ---------------------------------------------------------------------------
# Default raw mappings (used when use_canonical=False)
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
        When ``use_canonical=False``, ``raw_``-column stems that map to
        ``canonical_perturbation`` dict.
        Default: ``("guide_1","guide_2","treatment","site","genotype")``.
    context_raw_keys : iterable of str, optional
        When ``use_canonical=False``, ``raw_``-column stems that map to
        ``canonical_context`` dict.
        Default: ``("cell_type","cellline","donor_id","batch","passage")``.
    use_canonical : bool, optional
        When ``True``, reads canonical columns directly from the
        ``MetadataIndex`` DataFrame.  No per-row raw-column extraction.
        Default: ``False``.
    """

    def __init__(
        self,
        expression_reader: ExpressionReader,
        metadata_index: MetadataIndex,
        *,
        perturbation_raw_keys: Sequence[str] | None = None,
        context_raw_keys: Sequence[str] | None = None,
        use_canonical: bool = False,
    ):
        self._reader = expression_reader
        self._meta = metadata_index
        self._use_canonical = use_canonical

        self._perturbation_keys: frozenset[str] = frozenset(
            perturbation_raw_keys if perturbation_raw_keys is not None
            else _DEFAULT_PERTURBATION_KEYS
        )
        self._context_keys: frozenset[str] = frozenset(
            context_raw_keys if context_raw_keys is not None
            else _DEFAULT_CONTEXT_KEYS
        )

        # Canonical field sets (precomputed for speed)
        self._canonical_pert_fields: frozenset[str] = frozenset(
            _CANONICAL_PERTURBATION_FIELDS
        )
        self._canonical_ctx_fields: frozenset[str] = frozenset(
            _CANONICAL_CONTEXT_FIELDS
        )

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

        When ``use_canonical=True``, reads canonical columns directly;
        when ``False``, extracts from ``raw_``-prefixed columns.

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

        # Build canonical_perturbation and canonical_context dicts
        if self._use_canonical:
            canonical_perturbation, canonical_context = (
                self._build_canonical_dicts(df, n)
            )
        else:
            # Legacy path: extract from raw_ columns
            canonical_perturbation, canonical_context = (
                self._build_dicts_from_raw(df, n)
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
    # Internal: canonical dict builders
    # ------------------------------------------------------------------

    def _build_canonical_dicts(
        self, df: "pl.DataFrame", n: int
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Build canonical_perturbation and canonical_context from flat columns.

        Reads canonical columns directly from the DataFrame (no ``raw_``
        prefix, no per-row extraction).  Skips ``None`` and ``NA`` values.
        """
        pert_cols = [
            c for c in _CANONICAL_PERTURBATION_FIELDS
            if c in df.columns
        ]
        ctx_cols = [
            c for c in _CANONICAL_CONTEXT_FIELDS
            if c in df.columns
        ]

        pert_list: list[dict[str, str]] = []
        ctx_list: list[dict[str, str]] = []
        for i in range(n):
            row = df.row(i, named=True)
            pert = {}
            for col in pert_cols:
                val = row.get(col)
                if val is not None and str(val) != "NA":
                    pert[col] = str(val)
            ctx = {}
            for col in ctx_cols:
                val = row.get(col)
                if val is not None and str(val) != "NA":
                    ctx[col] = str(val)
            pert_list.append(pert)
            ctx_list.append(ctx)

        return pert_list, ctx_list

    def _build_dicts_from_raw(
        self, df: "pl.DataFrame", n: int
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Build canonical dicts from ``raw_``-prefixed columns (legacy path).

        Looks for ``raw_<key>`` columns and maps them to ``{key: value}``.
        Skips None and NA values.
        """
        pert_list: list[dict[str, str]] = []
        ctx_list: list[dict[str, str]] = []
        for i in range(n):
            row = df.row(i, named=True)
            pert_list.append(
                self._dict_from_raw(row, self._perturbation_keys)
            )
            ctx_list.append(
                self._dict_from_raw(row, self._context_keys)
            )
        return pert_list, ctx_list

    @staticmethod
    def _dict_from_raw(
        row: dict[str, Any], keys: frozenset[str]
    ) -> dict[str, str]:
        """Extract ``{key: value}`` from ``raw_<key>`` columns."""
        result: dict[str, str] = {}
        for key in keys:
            col = f"raw_{key}"
            if col in row and row[col] is not None:
                val = str(row[col])
                if val != "NA":
                    result[key] = val
        return result


