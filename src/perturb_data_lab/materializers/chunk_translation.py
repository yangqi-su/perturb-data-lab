"""Shared chunk translation layer for Stage 2 backend writers.

This module introduces the shared benchmark-derived translation pattern used by all
five backend writers (arrow-parquet, arrow-ipc, webdataset, zarr, lance). It provides
a common ``ChunkBundle`` dataclass, shared PyArrow schemas, and the core
``_translate_chunk()`` function that converts a CSR matrix batch into a form all
backends can consume without backend-specific sparse re-encoding.

Architecture
------------
The translation layer sits between the materialization entry point (Stage2Materializer)
and the backend-specific writer. The caller builds
or receives a CSR matrix chunk, and passes it along with a DatasetSpec to
_translate_chunk(), which returns a ChunkBundle. Each backend writer then converts
the ChunkBundle into its native format.

Design principles from the archived benchmark (materialize_split_brain_backends.py):
  - One shared _translate_chunk() pass produces the ChunkBundle; writers do not
    re-encode sparse rows independently.
  - Heavy-row schema uses flat Arrow list-arrays built via pa.ListArray.from_arrays()
    from CSR indptr/data/indices buffers directly — no per-row Python nested-list assembly.
  - Metadata table is a separate Arrow table with its own schema, allowing backends
    to handle it independently (write to separate Parquet, embed in Lance rows, etc.).
  - Row sums (un-normalized) are computed during translation and accumulated globally
    by the caller for median-based size factor computation after all chunks are written.
  - Recovery (when ``needs_recovery=True``) is applied as a vectorized per-row operation
    inside ``_translate_chunk()`` using a temporary expm1 CSR matrix — no full-matrix
    densification. The caller passes ``needs_recovery`` based on Stage 1's decision.
  - HVG statistics are accumulated by the caller during the chunk loop (via np.add.at)
    and finalized by ``_finalize_hvg()`` after all chunks are written — no separate pass.

Schema summary
--------------
HEAVY_CELL_SCHEMA: global_row_index: int64, expressed_gene_indices: list<int32>,
                   expression_counts: list<int32>

METADATA_SCHEMA:    global_row_index: int64, stable_row_id: string, cell_id: string,
                   dataset_id: string, dataset_index: int32, row_index_in_dataset: int64,
                   cell_context: string, perturbation_label: string, pair_id: string,
                   pair_role: string, paired_stable_row_id: string, paired_row_index: int64,
                   paired_global_row_index: int64, donor_id: string, batch_id: string,
                   replicate_id: string, size_factor: float32

FEATURE_REGISTRY_SCHEMA: dataset_id: string, dataset_index: int32, local_gene_index: int32,
                         global_gene_index: int32, gene_id: string, is_hvg: bool

DATASET_OFFSET_SCHEMA: dataset_id: string, dataset_index: int32, source_h5ad_path: string,
                     rows: int64, pairs: int64, local_vocabulary_size: int32,
                     global_row_start: int64, global_row_stop: int64, nnz_total: int64

WEBDATASET_SHARD_INDEX_SCHEMA: dataset_id: string, dataset_index: int32, global_row_index: int64,
                               shard_id: int32, shard_path: string, member_name: string, nnz: int32

Backends/topology separation
----------------------------
backend names the storage format only. topology names the corpus organization only.
The ``AVAILABLE_WRITERS[backend][topology]`` dispatch table replaces the legacy
fused ``AVAILABLE_BACKENDS`` registry. The shared translation layer is backend-agnostic
and works for both federated (per-dataset files) and aggregate (corpus-scoped) topologies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from scipy.sparse import csr_matrix, issparse


# ---------------------------------------------------------------------------
# Shared schemas
# ---------------------------------------------------------------------------

HEAVY_CELL_SCHEMA = pa.schema(
    [
        pa.field("global_row_index", pa.int64()),
        pa.field("expressed_gene_indices", pa.list_(pa.int32())),
        pa.field("expression_counts", pa.list_(pa.int32())),
    ]
)

METADATA_SCHEMA = pa.schema(
    [
        pa.field("global_row_index", pa.int64()),
        pa.field("stable_row_id", pa.string()),
        pa.field("cell_id", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("dataset_index", pa.int32()),
        pa.field("row_index_in_dataset", pa.int64()),
        pa.field("cell_context", pa.string()),
        pa.field("perturbation_label", pa.string()),
        pa.field("pair_id", pa.string()),
        pa.field("pair_role", pa.string()),
        pa.field("paired_stable_row_id", pa.string()),
        pa.field("paired_row_index", pa.int64()),
        pa.field("paired_global_row_index", pa.int64()),
        pa.field("donor_id", pa.string()),
        pa.field("batch_id", pa.string()),
        pa.field("replicate_id", pa.string()),
        pa.field("size_factor", pa.float32()),
    ]
)

FEATURE_REGISTRY_SCHEMA = pa.schema(
    [
        pa.field("dataset_id", pa.string()),
        pa.field("dataset_index", pa.int32()),
        pa.field("local_gene_index", pa.int32()),
        pa.field("global_gene_index", pa.int32()),
        pa.field("gene_id", pa.string()),
        pa.field("is_hvg", pa.bool_()),
    ]
)

DATASET_OFFSET_SCHEMA = pa.schema(
    [
        pa.field("dataset_id", pa.string()),
        pa.field("dataset_index", pa.int32()),
        pa.field("source_h5ad_path", pa.string()),
        pa.field("rows", pa.int64()),
        pa.field("pairs", pa.int64()),
        pa.field("local_vocabulary_size", pa.int32()),
        pa.field("global_row_start", pa.int64()),
        pa.field("global_row_stop", pa.int64()),
        pa.field("nnz_total", pa.int64()),
    ]
)

WEBDATASET_SHARD_INDEX_SCHEMA = pa.schema(
    [
        pa.field("dataset_id", pa.string()),
        pa.field("dataset_index", pa.int32()),
        pa.field("global_row_index", pa.int64()),
        pa.field("shard_id", pa.int32()),
        pa.field("shard_path", pa.string()),
        pa.field("member_name", pa.string()),
        pa.field("nnz", pa.int32()),
    ]
)


# ---------------------------------------------------------------------------
# DatasetSpec (mirrors archived benchmark)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Specification for a single dataset within a corpus.

    Attributes
    ----------
    dataset_id : str
        Stable dataset identifier.
    dataset_index : int
        Zero-based index of this dataset within the corpus.
    file_path : Path
        Absolute path to the source h5ad file.
    rows : int
        Number of cells (rows) in this dataset.
    pairs : int
        Number of perturbation pairs in this dataset.
    local_vocabulary_size : int
        Number of features (genes) in this dataset's local vocabulary.
    nnz_total : int
        Total number of non-zero entries in the sparse count matrix.
    global_row_start : int
        Starting global row index for this dataset within the corpus.
    global_row_stop : int
        Stop global row index (exclusive) for this dataset within the corpus.
    """

    dataset_id: str
    dataset_index: int
    file_path: Path
    rows: int
    pairs: int
    local_vocabulary_size: int
    nnz_total: int
    global_row_start: int
    global_row_stop: int

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_path: Path | None = None) -> "DatasetSpec":
        """Construct a DatasetSpec from a dictionary (e.g. loaded from JSON manifest)."""
        rows = int(data["rows"])
        global_row_start = int(data.get("global_row_start", 0))
        return cls(
            dataset_id=str(data["dataset_id"]),
            dataset_index=int(data.get("dataset_index", 0)),
            file_path=Path(str(data["file_path"])).resolve() if base_path is None else base_path / data["file_path"],
            rows=rows,
            pairs=int(data["pairs"]),
            local_vocabulary_size=int(data["local_vocabulary_size"]),
            nnz_total=int(data["nnz_total"]),
            global_row_start=global_row_start,
            global_row_stop=global_row_start + rows,
        )


# ---------------------------------------------------------------------------
# ChunkBundle dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkBundle:
    """Shared translation output consumed by all five backend writers.

    Attributes
    ----------
    table : pa.Table
        Heavy-row Arrow table with schema HEAVY_CELL_SCHEMA. Contains one row per
        cell in the chunk, with ``expressed_gene_indices`` and ``expression_counts``
        as Arrow list arrays built directly from CSR buffers via
        ``pa.ListArray.from_arrays()``.
    row_sums : np.ndarray
        Raw (un-normalized) per-cell row sums (float64). These are computed inside
        ``_translate_chunk()`` and accumulated by the caller across all chunks. After
        the chunk loop, the caller computes the global median of all accumulated row
        sums and produces globally-normalized size factors (``row_sum / global_median``).
        Size factors are written to a separate Parquet sidecar — no backend writer
        receives or embeds them.
    indptr : np.ndarray
        CSR indptr array for this chunk. Int64, length ``n_rows + 1``. Raw buffer
        available for backends that need flat-buffer access (Zarr, WebDataset).
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

    Notes
    -----
    The ``metadata_table`` field has been removed. Per-cell metadata is written by
    the caller (``Stage2Materializer``) as a separate Parquet sidecar
    (``{release_id}-raw-obs.parquet``). The ``_build_metadata_table()`` helper remains
    available for callers that need a ``METADATA_SCHEMA`` table from obs data.

    The ``indptr``, ``indices``, and ``counts`` arrays are raw CSR components.
    Backends that only need the Arrow table can ignore these.
    """

    table: pa.Table
    row_sums: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    counts: np.ndarray
    row_count: int


# ---------------------------------------------------------------------------
# List array builder
# ---------------------------------------------------------------------------

def _build_list_array(offsets: np.ndarray, values: np.ndarray) -> pa.Array:
    """Build a pa.ListArray from flat offsets and values arrays.

    This is the core pattern from the archived benchmark: instead of building
    per-row Python lists, we pass the CSR indptr (offsets) and flat value buffer
    directly to Arrow. This eliminates the hottest Python overhead in the write path.

    Parameters
    ----------
    offsets : np.ndarray
        Row-offset array (length n_rows + 1). Must be int32-castable for Arrow
        offset dtype. For CSR matrices, this is ``indptr``.
    values : np.ndarray
        Flat value buffer. For CSR indices/data, this is the flat ``indices`` or
        ``data`` array. Must be int32-castable.

    Returns
    -------
    pa.ListArray
        Arrow list array suitable for direct Parquet / IPC / Zarr write.

    Examples
    --------
    >>> import numpy as np, pyarrow as pa
    >>> from perturb_data_lab.materializers.chunk_translation import _build_list_array
    >>> indptr = np.array([0, 2, 5], dtype=np.int64)  # 2 non-zeros in row 0, 3 in row 1
    >>> indices = np.array([10, 30, 5, 7, 99], dtype=np.int32)
    >>> arr = _build_list_array(indptr, indices)
    >>> assert arr[0].as_py() == [10, 30]
    >>> assert arr[1].as_py() == [5, 7, 99]
    """
    return pa.ListArray.from_arrays(
        pa.array(offsets.astype(np.int32, copy=False), type=pa.int32()),
        pa.array(values, type=pa.int32()),
    )


# ---------------------------------------------------------------------------
# Core translation function
# ---------------------------------------------------------------------------

def _translate_chunk(
    *,
    dataset: DatasetSpec,
    matrix_chunk: csr_matrix,
    chunk_start: int,
    needs_recovery: bool = False,
) -> ChunkBundle:
    """Translate a CSR matrix chunk into a ChunkBundle for backend writers.

    This is the shared translation hot path. It converts a sparse CSR batch into the
    canonical ``ChunkBundle`` form once; all five backend writers then consume this
    same bundle without re-encoding sparse rows independently.

    Recovery (when ``needs_recovery=True``) is applied as a vectorized per-row
    operation using a temporary expm1 CSR matrix — no full-matrix densification.

    ``expressed_gene_indices`` are in **dataset-local feature space** — no canonical
    gene mapping is applied at this stage. Canonical mapping is deferred to Stage 3.

    Parameters
    ----------
    dataset : DatasetSpec
        Dataset specification for the source dataset.
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
        Translated chunk with heavy-row Arrow table, raw row sums, and raw CSR
        buffers. ``indices`` in the bundle are dataset-local (not mapped through
        any gene_lookup). No ``metadata_table`` is returned — callers that need
        a ``METADATA_SCHEMA`` table should use ``_build_metadata_table()``.

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
        raise ValueError(f"{dataset.dataset_id}:{chunk_start} chunk densified unexpectedly")
    matrix_chunk = matrix_chunk.tocsr(copy=False)
    if not isinstance(matrix_chunk, csr_matrix):
        matrix_chunk = csr_matrix(matrix_chunk)

    # For recovery, eliminate explicit zeros BEFORE extracting components.
    if needs_recovery:
        matrix_chunk.eliminate_zeros()

    indptr = np.asarray(matrix_chunk.indptr, dtype=np.int64)
    local_indices = np.asarray(matrix_chunk.indices, dtype=np.int32)
    row_count = int(matrix_chunk.shape[0])
    n_vars = int(matrix_chunk.shape[1])
    global_row_index = np.arange(
        dataset.global_row_start + chunk_start,
        dataset.global_row_start + chunk_start + row_count,
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
            row_sums[valid_rows] = np.add.reduceat(expm1_data, valid_starts).astype(np.float64)
            row_min_expm1[valid_rows] = np.minimum.reduceat(expm1_data, valid_starts)
        row_min_expm1[row_min_expm1 <= 0] = 1.0

        # Row sums are normalized by row_min (the scaling factor for recovery).
        row_sums = row_sums / row_min_expm1

        # Broadcast per-row minima to per-nonzero for elementwise division.
        row_starts = indptr[:-1]
        row_stops = indptr[1:]
        row_min_per_nonzero = np.repeat(row_min_expm1, (row_stops - row_starts).astype(np.intp))
        recovered_data = expm1_data / row_min_per_nonzero

        # Integer verification: max deviation from nearest integer must be < 0.01.
        deviations = np.abs(recovered_data - np.rint(recovered_data))
        if np.any(deviations > 0.01):
            max_dev = float(np.max(deviations))
            raise ValueError(
                f"{dataset.dataset_id}:{chunk_start} recovered values are non-integer "
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
                deviations = np.abs(raw_data[nonzero_mask] - np.rint(raw_data[nonzero_mask]))
                if np.any(deviations > 1e-6):
                    max_dev = float(np.max(deviations))
                    raise ValueError(
                        f"{dataset.dataset_id}:{chunk_start} float matrix is not integer-like "
                        f"(max_deviation={max_dev:.6f}); materialization requires integer counts"
                    )
            counts = np.rint(raw_data).astype(np.int32)
        else:
            counts = raw_data.astype(np.int32, copy=False)

        # Compute raw row sums from original counts.
        row_sums = np.asarray(matrix_chunk.sum(axis=1)).ravel()

    # Single sparse hot path: pa.ListArray.from_arrays() from CSR buffers.
    # expressed_gene_indices uses dataset-local indices (no canonical mapping).
    expressed_gene_indices = pa.ListArray.from_arrays(
        pa.array(indptr.astype(np.int32, copy=False), type=pa.int32()),
        pa.array(local_indices, type=pa.int32()),
    )
    expression_counts = pa.ListArray.from_arrays(
        pa.array(indptr.astype(np.int32, copy=False), type=pa.int32()),
        pa.array(counts.astype(np.int32, copy=False), type=pa.int32()),
    )

    table = pa.table(
        {
            "global_row_index": pa.array(global_row_index, type=pa.int64()),
            "expressed_gene_indices": expressed_gene_indices,
            "expression_counts": expression_counts,
        },
        schema=HEAVY_CELL_SCHEMA,
    )

    return ChunkBundle(
        table=table,
        row_sums=row_sums,
        indptr=indptr,
        indices=local_indices,
        counts=counts,
        row_count=row_count,
    )


# ---------------------------------------------------------------------------
# Seurat-style HVG selection
# ---------------------------------------------------------------------------

def _finalize_hvg(
    sum_log1p: np.ndarray,
    sum_log1p_sq: np.ndarray,
    n_cells_total: int,
    n_vars: int,
    n_hvg: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Seurat-style highly variable gene selection from streaming accumulators.

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
        Number of top-dispersion genes to mark as highly variable.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(hvg_indices, nonhvg_indices)`` — dataset-local gene indices (int32) for
        highly variable and non-highly variable genes respectively.
    """
    import pandas as pd

    mean = sum_log1p / n_cells_total
    var = (sum_log1p_sq - sum_log1p**2 / n_cells_total) / max(n_cells_total - 1, 1)
    mean[mean == 0] = 1e-12
    dispersion = var / mean
    dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    mean = np.log1p(mean)

    df = pd.DataFrame({"means": mean, "dispersions": dispersion})
    # pd.cut(bins=20) can produce empty bins when gene count < 20.
    # Lines below handle the single-gene-bin edge case: when a bin contains
    # exactly one gene, dev=std is NaN; it is replaced with dev=avg so the
    # normalization does not crash (single-gene bins end up with deviation=0).
    df["mean_bin"] = pd.cut(df["means"], bins=20)
    disp_grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    disp_stats = disp_grouped.agg(avg="mean", dev="std")
    one_gene_per_bin = disp_stats["dev"].isnull()
    disp_stats.loc[one_gene_per_bin, "dev"] = disp_stats.loc[one_gene_per_bin, "avg"]
    disp_stats.loc[one_gene_per_bin, "avg"] = 0

    df["dispersions_norm"] = (
        df["dispersions"] - disp_stats.loc[df["mean_bin"], "avg"].values
    ) / disp_stats.loc[df["mean_bin"], "dev"].values

    df = df.sort_values("dispersions_norm", ascending=False)
    df["highly_variable"] = False
    df.iloc[:n_hvg, df.columns.get_loc("highly_variable")] = True
    df = df.sort_index()

    hvg_indices = df.index[df["highly_variable"]].to_numpy().astype(np.int32)
    hvg_indices.sort()
    nonhvg_indices = np.array(
        [j for j in range(n_vars) if j not in set(hvg_indices)], dtype=np.int32
    )
    return hvg_indices, nonhvg_indices


# ---------------------------------------------------------------------------
# Metadata table builder
# ---------------------------------------------------------------------------

def _build_metadata_table(
    dataset: DatasetSpec,
    obs: Any,
    size_factors: np.ndarray,
    *,
    chunk_start: int = 0,
) -> pa.Table:
    """Build a filled METADATA_SCHEMA table from obs columns and size factors.

    Parameters
    ----------
    dataset : DatasetSpec
        Dataset specification.
    obs : pd.DataFrame
        In-memory obs DataFrame for the dataset. Must contain the required columns:
        ``stable_row_id``, ``cell_id``, ``row_index_in_dataset``, ``cell_context``,
        ``perturbation_label``, ``pair_id``, ``pair_role``, ``paired_stable_row_id``,
        ``paired_row_index``, ``donor_id``, ``batch_id``, ``replicate_id``.
    size_factors : np.ndarray
        Per-cell size factors (float64 or float32). Must have length equal to
        ``len(obs)``.
    chunk_start : int, default 0
        Starting row offset for this chunk. Used to compute global_row_index.

    Returns
    -------
    pa.Table
        Metadata Arrow table with schema METADATA_SCHEMA, fully populated.
    """
    # defensive import to avoid pulling pandas into the module signature
    import pandas as pd

    n_rows = len(obs)
    row_index = obs["row_index_in_dataset"].to_numpy(dtype=np.int64, copy=False)
    global_row_index = row_index + np.int64(dataset.global_row_start)

    paired_row_index = obs["paired_row_index"].to_numpy(dtype=np.int64, copy=False)
    paired_global_row_index = paired_row_index + np.int64(dataset.global_row_start)

    def _str_col(col_name: str) -> list[str]:
        vals = obs[col_name].tolist()
        return ["" if pd.isna(v) else str(v) for v in vals]

    return pa.table(
        {
            "global_row_index": pa.array(global_row_index, type=pa.int64()),
            "stable_row_id": pa.array(_str_col("stable_row_id"), type=pa.string()),
            "cell_id": pa.array(_str_col("cell_id"), type=pa.string()),
            "dataset_id": pa.array([dataset.dataset_id] * n_rows, type=pa.string()),
            "dataset_index": pa.array([dataset.dataset_index] * n_rows, type=pa.int32()),
            "row_index_in_dataset": pa.array(row_index, type=pa.int64()),
            "cell_context": pa.array(_str_col("cell_context"), type=pa.string()),
            "perturbation_label": pa.array(_str_col("perturbation_label"), type=pa.string()),
            "pair_id": pa.array(_str_col("pair_id"), type=pa.string()),
            "pair_role": pa.array(_str_col("pair_role"), type=pa.string()),
            "paired_stable_row_id": pa.array(_str_col("paired_stable_row_id"), type=pa.string()),
            "paired_row_index": pa.array(paired_row_index, type=pa.int64()),
            "paired_global_row_index": pa.array(paired_global_row_index, type=pa.int64()),
            "donor_id": pa.array(_str_col("donor_id"), type=pa.string()),
            "batch_id": pa.array(_str_col("batch_id"), type=pa.string()),
            "replicate_id": pa.array(_str_col("replicate_id"), type=pa.string()),
            "size_factor": pa.array(
                size_factors[:n_rows].astype(np.float32, copy=False), type=pa.float32()
            ),
        },
        schema=METADATA_SCHEMA,
    )
