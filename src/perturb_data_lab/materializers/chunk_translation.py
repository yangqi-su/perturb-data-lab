"""Shared sparse chunk translation for Stage 2 backend writers.

The materializer slices the selected count matrix into CSR chunks, then this module
normalizes each chunk into raw sparse buffers. Lance and Zarr writers decide how to
persist those buffers; Zarr writes them directly, while Lance builds Arrow tables in
the Lance backend only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
from scipy.sparse import csr_matrix, issparse


HVG_RANKING_SCHEMA = pa.schema(
    [
        pa.field("origin_index", pa.int32()),
        pa.field("feature_id", pa.string()),
        pa.field("mean_log1p_expr", pa.float64()),
        pa.field("variance_log1p_expr", pa.float64()),
        pa.field("dispersion_log", pa.float64()),
        pa.field("dispersion_norm", pa.float64()),
        pa.field("hvg_rank", pa.int32()),
        pa.field("selected_at_default_n_hvg", pa.bool_()),
    ]
)


# ---------------------------------------------------------------------------
# ChunkBundle dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkBundle:
    """Sparse chunk buffers consumed by backend writers.

    Attributes
    ----------
    global_row_index : np.ndarray
        Corpus-global row indices for this chunk.
    row_sums : np.ndarray
        Raw (un-normalized) per-cell row sums (float64). These are computed inside
        ``_translate_chunk()`` and accumulated by the caller across all chunks. After
        the chunk loop, the caller computes the global median of all accumulated row
        sums and produces globally-normalized size factors (``row_sum / global_median``).
        Size factors are written to a separate Parquet sidecar — no backend writer
        receives or embeds them.
    indptr : np.ndarray
        CSR indptr array for this chunk. Int64, length ``n_rows + 1``. Raw buffer
        available for backends that need flat-buffer access (Zarr).
    indices : np.ndarray
        **Dataset-local** gene indices (int32) for all non-zero entries in this chunk.
        Raw buffer for backends that need it. These indices are in the original
        dataset's feature space — no canonical gene mapping has been applied.
        Canonical mapping is deferred to Stage 3.
    counts : np.ndarray
        Expression counts (int32) for all non-zero entries in this chunk. Raw buffer
        for backends that need it.
    row_count : int
        Number of cell rows in this chunk.
    """

    global_row_index: np.ndarray
    row_sums: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    counts: np.ndarray
    row_count: int


# ---------------------------------------------------------------------------
# Core translation function
# ---------------------------------------------------------------------------


def _translate_chunk(
    *,
    dataset_id: str,
    global_row_start: int,
    matrix_chunk: csr_matrix,
    chunk_start: int,
    needs_recovery: bool = False,
) -> ChunkBundle:
    """Translate a CSR matrix chunk into a ChunkBundle for backend writers.

    This is the shared translation hot path. It converts a sparse CSR batch into a
    raw ``ChunkBundle`` once, then backend writers persist those raw arrays.

    Recovery (when ``needs_recovery=True``) is applied as a vectorized per-row
    operation on the nonzero values — no full-matrix densification.

    ``expressed_gene_indices`` are in **dataset-local feature space** — no canonical
    gene mapping is applied at this stage. Canonical mapping is deferred to Stage 3.

    Parameters
    ----------
    dataset_id : str
        Stable dataset identifier, used in validation errors.
    global_row_start : int
        Starting global row index for this dataset within the corpus.
    matrix_chunk : csr_matrix
        Sparse CSR matrix chunk to translate. Must be a ``csr_matrix`` or convert
        to one without densification.
    chunk_start : int
        Row offset within the dataset at which this chunk begins (0-indexed).
        Used to compute ``global_row_index``.
    needs_recovery : bool, default False
        When False (integer counts path): matrix_chunk data must be integer-valued
        (max deviation < 1e-6 for nonzero entries). Float-dtype integer-like data
        is accepted, verified inline, and cast to int32.
        When True (log1p recovery path): matrix_chunk is treated as log1p-normalized
        counts; vectorized recovery produces integer counts via
        ``expm1(data) / row_min_expm1``.

    Returns
    -------
    ChunkBundle
        Translated chunk with raw row sums, global row indices, and raw CSR
        buffers. ``indices`` in the bundle are dataset-local.

    Raises
    ------
    ValueError
        If ``matrix_chunk`` is not sparse or cannot be converted to CSR.
    ValueError
        If ``needs_recovery=False`` and nonzero values deviate from integer by > 1e-6.
    ValueError
        If ``needs_recovery=True`` and recovered values deviate from integer by > 0.01.
    """
    if not issparse(matrix_chunk):
        raise ValueError(f"{dataset_id}:{chunk_start} chunk densified unexpectedly")
    matrix_chunk = matrix_chunk.tocsr(copy=False)
    if not isinstance(matrix_chunk, csr_matrix):
        matrix_chunk = csr_matrix(matrix_chunk)

    # For recovery, eliminate explicit zeros BEFORE extracting components.
    if needs_recovery:
        matrix_chunk.eliminate_zeros()

    indptr = np.asarray(matrix_chunk.indptr, dtype=np.int64)
    local_indices = np.asarray(matrix_chunk.indices, dtype=np.int32)
    row_count = int(matrix_chunk.shape[0])
    global_row_index = np.arange(
        global_row_start + chunk_start,
        global_row_start + chunk_start + row_count,
        dtype=np.int64,
    )

    if needs_recovery:
        # --- Vectorized log1p recovery ---
        raw_data = matrix_chunk.data  # may be float32 or float64
        expm1_data = np.expm1(raw_data)

        # Per-row sums and minima via reduceat (replaces expensive expm1_matrix
        # construction + per-row Python loop). Both np.add.reduceat and
        # np.minimum.reduceat produce incorrect results when consecutive indptr
        # entries are equal (empty rows), so we use a valid_rows guard.
        row_sums = np.zeros(row_count, dtype=np.float64)
        row_min_expm1 = np.full(row_count, 1.0, dtype=np.float64)
        row_lengths = np.diff(indptr)
        valid_rows = row_lengths > 0
        if valid_rows.any():
            valid_starts = indptr[:-1][valid_rows]
            row_sums[valid_rows] = np.add.reduceat(expm1_data, valid_starts).astype(
                np.float64
            )
            row_min_expm1[valid_rows] = np.minimum.reduceat(expm1_data, valid_starts)
        row_min_expm1[row_min_expm1 <= 0] = 1.0

        # Row sums are normalized by row_min (the scaling factor for recovery).
        row_sums = row_sums / row_min_expm1

        # Broadcast per-row minima to per-nonzero for elementwise division.
        row_starts = indptr[:-1]
        row_stops = indptr[1:]
        row_min_per_nonzero = np.repeat(
            row_min_expm1,
            (row_stops - row_starts).astype(np.intp),
        )
        recovered_data = expm1_data / row_min_per_nonzero

        # Integer verification: max deviation from nearest integer must be < 0.01.
        deviations = np.abs(recovered_data - np.rint(recovered_data))
        if np.any(deviations > 0.01):
            max_dev = float(np.max(deviations))
            raise ValueError(
                f"{dataset_id}:{chunk_start} recovered values are non-integer "
                f"(max_deviation={max_dev:.6f}); recovery validation failed"
            )

        counts = np.rint(recovered_data).astype(np.int32)
    else:
        # --- Integer counts path (no recovery) ---
        raw_data = np.asarray(matrix_chunk.data)
        if raw_data.dtype.kind not in {"i", "u"}:
            # Float-dtype integer-like matrix: verify and accept.
            nonzero_mask = raw_data != 0
            if nonzero_mask.any():
                deviations = np.abs(
                    raw_data[nonzero_mask] - np.rint(raw_data[nonzero_mask])
                )
                if np.any(deviations > 1e-6):
                    max_dev = float(np.max(deviations))
                    raise ValueError(
                        f"{dataset_id}:{chunk_start} float matrix is not integer-like "
                        f"(max_deviation={max_dev:.6f}); materialization requires integer counts"
                    )
            counts = np.rint(raw_data).astype(np.int32)
        else:
            counts = raw_data.astype(np.int32, copy=False)

        # Compute raw row sums from original counts.
        row_sums = np.asarray(matrix_chunk.sum(axis=1)).ravel()

    return ChunkBundle(
        global_row_index=global_row_index,
        row_sums=row_sums,
        indptr=indptr,
        indices=local_indices,
        counts=counts,
        row_count=row_count,
    )


# ---------------------------------------------------------------------------
# Seurat-style HVG selection
# ---------------------------------------------------------------------------


def _compute_hvg_ranking_arrays(
    sum_log1p: np.ndarray,
    sum_log1p_sq: np.ndarray,
    n_cells_total: int,
    n_vars: int,
    n_hvg: int = 2000,
) -> dict[str, np.ndarray]:
    """Compute deterministic HVG ranking arrays from streaming accumulators.

    Parameters
    ----------
    sum_log1p : np.ndarray
        Per-gene sum of log1p(counts) accumulated via np.add.at during the chunk loop.
    sum_log1p_sq : np.ndarray
        Per-gene sum of log1p(counts)^2 accumulated via np.add.at during the chunk loop.
    n_cells_total : int
        Total number of cells across all chunks.
    n_vars : int
        Number of genes (features) in the dataset.
    n_hvg : int, default 2000
        Number of top-dispersion genes to mark as selected by default.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays for the canonical HVG ranking artifact. ``hvg_rank`` is 1-indexed,
        with ties broken by ``origin_index`` ascending after
        ``dispersion_norm`` descending.
    """
    from ..pp.hvg import compute_hvg_ranking_arrays

    if len(sum_log1p) != n_vars or len(sum_log1p_sq) != n_vars:
        raise ValueError("streaming HVG accumulators must match n_vars")
    return compute_hvg_ranking_arrays(
        sum_log1p,
        sum_log1p_sq,
        n_cells_total,
        n_hvg=n_hvg,
    )


def _build_hvg_ranking_table(
    sum_log1p: np.ndarray,
    sum_log1p_sq: np.ndarray,
    n_cells_total: int,
    feature_ids: Any,
    *,
    n_hvg: int = 2000,
) -> pa.Table:
    """Build the canonical per-dataset ``hvg.parquet`` ranking table.

    ``mean_log1p_expr`` and ``variance_log1p_expr`` are computed directly from the
    streaming log1p(count) accumulators. ``dispersion_log`` is the log of
    ``variance_log1p_expr / mean_log1p_expr`` after the historical zero guard.
    ``dispersion_norm`` preserves the existing Seurat-style mean-bin normalization,
    including the legacy ``log1p(mean_log1p_expr)`` binning step used by the
    previous fixed-top-N helper.
    """

    feature_id_list = [str(feature_id) for feature_id in feature_ids]
    arrays = _compute_hvg_ranking_arrays(
        sum_log1p=sum_log1p,
        sum_log1p_sq=sum_log1p_sq,
        n_cells_total=n_cells_total,
        n_vars=len(feature_id_list),
        n_hvg=n_hvg,
    )
    return pa.table(
        {
            "origin_index": pa.array(arrays["origin_index"], type=pa.int32()),
            "feature_id": pa.array(feature_id_list, type=pa.string()),
            "mean_log1p_expr": pa.array(arrays["mean_log1p_expr"], type=pa.float64()),
            "variance_log1p_expr": pa.array(
                arrays["variance_log1p_expr"], type=pa.float64()
            ),
            "dispersion_log": pa.array(arrays["dispersion_log"], type=pa.float64()),
            "dispersion_norm": pa.array(arrays["dispersion_norm"], type=pa.float64()),
            "hvg_rank": pa.array(arrays["hvg_rank"], type=pa.int32()),
            "selected_at_default_n_hvg": pa.array(
                arrays["selected_at_default_n_hvg"], type=pa.bool_()
            ),
        },
        schema=HVG_RANKING_SCHEMA,
    )
