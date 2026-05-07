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
    BatchMetadata,
    ExpressionBatch,
    RawExpressionBatch,
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

    @property
    def expression_reader(self) -> ExpressionReader:
        """Return the underlying expression reader."""
        return self._reader

    # ------------------------------------------------------------------
    # Public API — flat-batch, zero per-cell Python objects
    # ------------------------------------------------------------------

    def read_expression_batch(
        self, global_indices: Sequence[int]
    ) -> ExpressionBatch:
        """Read expression data as a flat ``ExpressionBatch``.

        Uses the reader's native flat read path when available
        (``read_expression_flat()`` for Lance-backed corpora), falling
        back to the legacy ``ExpressionRow`` reconstruction path for
        other backends.

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

        # Fast path: use reader's native flat expression method
        # (Phase 2 aggregate Lance, Phase 3 federated Lance).
        # Avoids ExpressionRow object construction and de-batch/re-batch.
        if hasattr(self._reader, "read_expression_flat"):
            return self._reader.read_expression_flat(indices)

        # Legacy fallback: fetch ExpressionRow objects and reconstruct
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
        self,
        global_indices: Sequence[int],
        *,
        columnar: bool = False,
    ) -> dict[str, Any]:
        """Read metadata as columnar numpy arrays without per-cell objects.

        Each metadata column is extracted once (vectorized) via
        ``MetadataIndex.gather_columns()``.  No per-row
        ``df.row(i, named=True)`` calls are made in the default path.

        String fields (``dataset_id``, ``cell_id``) are returned as
        tuples.  When ``columnar=False`` (default), canonical perturbation
        and context are still returned as per-cell tuples of dicts for
        backward compatibility, but they are built from columnar arrays
        rather than from per-row Polars access.

        When ``columnar=True``, ``meta_columns`` is returned containing
        columnar metadata arrays/tuples, and ``canonical_perturbation`` /
        ``canonical_context`` are omitted.

        Parameters
        ----------
        global_indices : sequence of int
            Corpus-global cell indices.
        columnar : bool, optional
            If ``True``, return ``meta_columns`` dict instead of
            per-cell ``canonical_perturbation`` / ``canonical_context``
            dicts.  Default: ``False``.

        Returns
        -------
        dict
            When ``columnar=False``:
            Keys: ``global_row_index``, ``dataset_index``, ``local_row_index``,
            ``size_factor``, ``dataset_id``, ``cell_id``,
            ``canonical_perturbation``, ``canonical_context``.

            When ``columnar=True``:
            Adds ``meta_columns`` (BatchedMetadata dict).  Omits
            ``canonical_perturbation`` and ``canonical_context``.
        """
        indices = [int(i) for i in global_indices]
        n = len(indices)
        if n == 0:
            result: dict[str, Any] = {
                "global_row_index": np.array([], dtype=np.int64),
                "dataset_index": np.array([], dtype=np.int32),
                "local_row_index": np.array([], dtype=np.int64),
                "size_factor": np.array([], dtype=np.float32),
                "dataset_id": (),
                "cell_id": (),
            }
            if columnar:
                result["meta_columns"] = {}
            else:
                result["canonical_perturbation"] = ()
                result["canonical_context"] = ()
            return result

        # --- Gather all identity columns + canonical columns in one call ---
        identity_cols = [
            "global_row_index", "dataset_index", "local_row_index",
            "size_factor", "dataset_id", "cell_id",
        ]

        if columnar:
            # Gather all canonical fields for meta_columns output
            canonical_cols = []
            if self._use_canonical:
                canonical_cols = sorted(
                    set(_CANONICAL_PERTURBATION_FIELDS)
                    | set(_CANONICAL_CONTEXT_FIELDS)
                )
            else:
                # Legacy raw mode: gather raw_<key> columns for canonical
                canonical_cols = sorted(
                    [f"raw_{k}" for k in self._perturbation_keys]
                    + [f"raw_{k}" for k in self._context_keys]
                )
            all_cols = identity_cols + canonical_cols
        else:
            # Default path: also gather canonical fields for dict building
            if self._use_canonical:
                canonical_cols = sorted(
                    set(_CANONICAL_PERTURBATION_FIELDS)
                    | set(_CANONICAL_CONTEXT_FIELDS)
                )
            else:
                canonical_cols = sorted(
                    [f"raw_{k}" for k in self._perturbation_keys]
                    + [f"raw_{k}" for k in self._context_keys]
                )
            all_cols = identity_cols + canonical_cols

        # Single vectorized gather call — no per-row Polars access
        gathered = self._meta.gather_columns(indices, all_cols)

        # --- Identity columns ---
        global_row_index = gathered.get(
            "global_row_index",
            np.array(indices, dtype=np.int64),
        )
        if global_row_index.dtype != np.int64:
            global_row_index = global_row_index.astype(np.int64, copy=False)

        dataset_index = gathered.get(
            "dataset_index",
            np.zeros(n, dtype=np.int32),
        )
        if dataset_index.dtype != np.int32:
            dataset_index = dataset_index.astype(np.int32, copy=False)

        local_row_index = gathered.get(
            "local_row_index",
            np.array(indices, dtype=np.int64),
        )
        if local_row_index.dtype != np.int64:
            local_row_index = local_row_index.astype(np.int64, copy=False)

        size_factor = gathered.get(
            "size_factor",
            np.ones(n, dtype=np.float32),
        )
        if size_factor.dtype != np.float32:
            size_factor = size_factor.astype(np.float32, copy=False)

        dataset_id = gathered.get("dataset_id", ())
        cell_id = gathered.get("cell_id", ())

        result = {
            "global_row_index": global_row_index,
            "dataset_index": dataset_index,
            "local_row_index": local_row_index,
            "size_factor": size_factor,
            "dataset_id": dataset_id,
            "cell_id": cell_id,
        }

        if columnar:
            # --- Columnar mode: meta_columns dict ---
            meta_cols: dict[str, Any] = {}
            if self._use_canonical:
                for field in sorted(
                    set(_CANONICAL_PERTURBATION_FIELDS)
                    | set(_CANONICAL_CONTEXT_FIELDS)
                ):
                    if field in gathered:
                        meta_cols[field] = gathered[field]
            else:
                for key in sorted(self._perturbation_keys):
                    col = f"raw_{key}"
                    if col in gathered:
                        meta_cols[key] = gathered[col]
                for key in sorted(self._context_keys):
                    col = f"raw_{key}"
                    if col in gathered:
                        meta_cols[key] = gathered[col]
            result["meta_columns"] = meta_cols
        else:
            # --- Backward-compatible: build dict tuples from columnar arrays ---
            if self._use_canonical:
                canonical_perturbation = self._build_canonical_from_columnar(
                    gathered, _CANONICAL_PERTURBATION_FIELDS, n
                )
                canonical_context = self._build_canonical_from_columnar(
                    gathered, _CANONICAL_CONTEXT_FIELDS, n
                )
            else:
                canonical_perturbation = self._build_raw_from_columnar(
                    gathered, self._perturbation_keys, n
                )
                canonical_context = self._build_raw_from_columnar(
                    gathered, self._context_keys, n
                )

            result["canonical_perturbation"] = tuple(canonical_perturbation)
            result["canonical_context"] = tuple(canonical_context)

        return result

    def read_batch(
        self, global_indices: Sequence[int], *, columnar: bool = False
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
        columnar : bool, optional
            Forwarded to ``read_metadata_batch()``.  When ``True``,
            ``meta_columns`` is included and ``canonical_perturbation`` /
            ``canonical_context`` are omitted.  Default: ``False``.

        Returns
        -------
        dict
            Keys when ``columnar=False``:
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

            When ``columnar=True``, ``canonical_perturbation`` and
            ``canonical_context`` are replaced by ``meta_columns``
            (dict of columnar arrays/tuples).
        """
        expr = self.read_expression_batch(global_indices)
        meta = self.read_metadata_batch(global_indices, columnar=columnar)

        result: dict[str, Any] = {
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
        }
        if columnar:
            result["meta_columns"] = meta.get("meta_columns", {})
        else:
            result["canonical_perturbation"] = meta["canonical_perturbation"]
            result["canonical_context"] = meta["canonical_context"]
        return result

    def read_raw_batch(
        self,
        global_indices: Sequence[int],
        *,
        metadata_columns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Read the stable raw expression batch contract.

        This keeps the worker-facing batch minimal: expression arrays plus the
        required ``dataset_index`` routing field, with ``local_row_index`` and
        ``size_factor`` carried only as optional pass-through metadata.
        Additional metadata columns are attached only when explicitly asked for.
        """
        indices = [int(i) for i in global_indices]
        expr = self.read_expression_batch(indices)

        requested_meta = list(dict.fromkeys(metadata_columns or ()))
        gather_cols = ["dataset_index", "local_row_index"]
        if "size_factor" in self._meta.df.columns:
            gather_cols.append("size_factor")
        gather_cols.extend(
            col for col in requested_meta if col not in gather_cols
        )
        gathered = self._meta.gather_columns(indices, gather_cols)

        batch = RawExpressionBatch(
            batch_size=expr.batch_size,
            global_row_index=expr.global_row_index,
            dataset_index=np.asarray(
                gathered.get(
                    "dataset_index",
                    np.zeros(expr.batch_size, dtype=np.int32),
                ),
                dtype=np.int32,
            ),
            row_offsets=expr.row_offsets,
            expressed_gene_indices=expr.expressed_gene_indices,
            expression_counts=expr.expression_counts,
            local_row_index=np.asarray(
                gathered["local_row_index"], dtype=np.int64,
            ) if "local_row_index" in gathered else None,
            size_factor=_coerce_optional_float32(gathered.get("size_factor")),
            meta_columns={
                col: gathered[col]
                for col in requested_meta
                if col in gathered
            },
        )
        return batch.to_dict()

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

    # ------------------------------------------------------------------
    # Internal: columnar dict builders (Phase 4 fast path)
    # ------------------------------------------------------------------
    # These replace ``_build_canonical_dicts()`` and
    # ``_build_dicts_from_raw()`` in the hot path.  Instead of
    # ``df.row(i, named=True)`` per cell, they iterate over pre-gathered
    # columnar numpy arrays / tuples.  The old methods are preserved as
    # backward-compatible fallbacks (not called in the default path).

    @staticmethod
    def _build_canonical_from_columnar(
        gathered: dict[str, np.ndarray | tuple],
        field_names: tuple[str, ...],
        n: int,
    ) -> list[dict[str, str]]:
        """Build per-cell dicts from columnar canonical arrays.

        Iterates row-by-row over numpy arrays and tuples (no Polars
        DataFrame access).  Skips ``None`` and ``"NA"`` values.
        """
        # Pre-fetch column data for fields present in gathered
        col_data: dict[str, np.ndarray | tuple] = {}
        col_names: list[str] = []
        for field in field_names:
            if field in gathered:
                col_data[field] = gathered[field]
                col_names.append(field)

        if not col_names:
            return [{} for _ in range(n)]

        result: list[dict[str, str]] = []
        for i in range(n):
            row_dict: dict[str, str] = {}
            for field in col_names:
                data = col_data[field]
                val = data[i] if isinstance(data, (np.ndarray, list)) else data[i]  # type: ignore[index]
                if val is not None and str(val) != "NA":
                    row_dict[field] = str(val)
            result.append(row_dict)
        return result

    @staticmethod
    def _build_raw_from_columnar(
        gathered: dict[str, np.ndarray | tuple],
        keys: frozenset[str],
        n: int,
    ) -> list[dict[str, str]]:
        """Build per-cell dicts from columnar ``raw_<key>`` arrays.

        Analogous to ``_build_dicts_from_raw()`` but operates on
        columnar data from ``gather_columns()`` instead of
        ``df.row(i, named=True)``.
        """
        col_data: dict[str, np.ndarray | tuple] = {}
        active_keys: list[str] = []
        for key in sorted(keys):
            col = f"raw_{key}"
            if col in gathered:
                col_data[key] = gathered[col]
                active_keys.append(key)

        if not active_keys:
            return [{} for _ in range(n)]

        result: list[dict[str, str]] = []
        for i in range(n):
            row_dict: dict[str, str] = {}
            for key in active_keys:
                data = col_data[key]
                val = data[i]  # type: ignore[index]
                if val is not None and str(val) != "NA":
                    row_dict[key] = str(val)
            result.append(row_dict)
        return result

    # ------------------------------------------------------------------
    # Deprecated: row-by-row Polars dict builders (preserved for compat)
    # ------------------------------------------------------------------
    # DO NOT call these in the default hot path.  They are kept only
    # for backward compatibility and are superseded by the columnar
    # builders above (``_build_canonical_from_columnar``,
    # ``_build_raw_from_columnar``) which avoid ``df.row(i, named=True)``.
