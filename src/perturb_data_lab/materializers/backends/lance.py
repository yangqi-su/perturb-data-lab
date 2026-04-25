"""Backend adapter: Lance federated and aggregate writers.

Phase 3: refactored to consume the shared flat-buffer Arrow list-array
pattern from the translation layer for heavy-row construction.

Phase 4: adds aggregate topology writer that delegates to ``lancedb_aggregated.py``
for corpus-scoped append-safe Lance storage.

Produces per-dataset .lance files using ``lance.write_dataset`` for federated.
For aggregate topology, see ``lancedb_aggregated.py``.

Topology: federated (per-dataset files) and aggregate (corpus-scoped single store).
Backend name in registry: ``lance``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pyarrow as pa
from scipy.sparse import issparse

from ..chunk_translation import DatasetSpec, _build_list_array


def _is_csr_dataset(x: object) -> bool:
    """Check if x is an anndata _CSRDataset (backed sparse)."""
    return x.__class__.__name__ == "_CSRDataset"


def _import_lance():
    try:
        import lance
        import lancedb
    except ImportError as exc:
        raise ImportError(
            "lance backend requires the Lance stack; "
            "install lancedb==0.30.2 in the selected runtime"
        ) from exc
    return lance, lancedb


def write_lance_federated(
    adata: ad.AnnData,
    count_matrix: Any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    dataset_id: str = "",
) -> dict[str, Path]:
    """Write federated sparse per-cell data as a Lance dataset.

    Uses the flat-buffer Arrow list-array pattern to build heavy-row
    Arrow tables, then writes them via ``lance.write_dataset``.

    Returns a dict with keys: ``{"cells": lance_path}``.
    """
    lance, _ = _import_lance()
    matrix_root.mkdir(parents=True, exist_ok=True)

    lance_path = matrix_root / f"{release_id}.lance"
    n_obs = adata.n_obs
    batch_size = 50_000

    writer_initialized = False

    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)
        batch_csr = count_matrix[start:end].tocsr()

        batch_indptr = np.asarray(batch_csr.indptr, dtype=np.int64)
        batch_data = np.asarray(batch_csr.data, dtype=np.int32)
        batch_indices = np.asarray(batch_csr.indices, dtype=np.int32)
        batch_n_rows = end - start

        global_row_indices = np.arange(start, end, dtype=np.int64)

        indices_list_array = _build_list_array(batch_indptr, batch_indices)
        counts_list_array = _build_list_array(batch_indptr, batch_data)

        table = pa.table(
            {
                "global_row_index": pa.array(global_row_indices, type=pa.int64()),
                "expressed_gene_indices": indices_list_array,
                "expression_counts": counts_list_array,
            }
        )

        mode = "append" if writer_initialized else "create"
        lance.write_dataset(
            table,
            str(lance_path),
            mode=mode,
            max_rows_per_group=4096,
        )
        writer_initialized = True

    return {"cells": lance_path}


def read_lance_cell(
    lance_path: Path,
    cell_index: int,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """Read a single cell's sparse data from a Lance dataset.

    Returns ``(expressed_gene_indices, expression_counts, size_factor)``.
    """
    import lance

    ds = lance.dataset(str(lance_path))
    row = ds.take([cell_index]).to_pylist()[0]

    indices = tuple(row["expressed_gene_indices"])
    counts = tuple(row["expression_counts"])

    sf = 1.0
    if size_factor_path is not None:
        import pyarrow.parquet as pq

        sf_table = pq.read_table(str(size_factor_path))
        sf = float(sf_table["size_factor"][cell_index].as_py())

    return (indices, counts, sf)


# ---------------------------------------------------------------------------
# Aggregate writer (Phase 4)
# ---------------------------------------------------------------------------


def write_lance_aggregate(
    datasets: list[DatasetSpec],
    count_matrices: list[Any],
    size_factors_list: list[np.ndarray],
    matrix_root: Path,
    corpus_index_path: Path | None = None,
) -> tuple[dict[str, Path], list[np.ndarray]]:
    """Write aggregate sparse per-cell data as a Lance dataset.

    This is the ``lance × aggregate`` backend writer. It delegates to
    ``write_lancedb_aggregated`` for each dataset, which handles corpus-scoped
    append-safety via the lance append sidecar.

    Parameters
    ----------
    datasets : list[DatasetSpec]
        Dataset specifications in order.
    count_matrices : list[Any]
        Sparse count matrices (CSR or dense), one per dataset.
    size_factors_list : list[np.ndarray]
        Pre-computed size factors for each dataset.
    matrix_root : Path
        Output directory.
    corpus_index_path : Path | None
        Path to the corpus index YAML for corpus registration.

    Returns
    -------
    tuple[dict[str, Path], list[np.ndarray]]
        ``(paths_dict, size_factors_out_list)``.
    """
    # Import the aggregate writer from lancedb_aggregated
    from .lancedb_aggregated import write_lancedb_aggregated

    paths_list: list[dict[str, Path]] = []
    size_factors_out_list: list[np.ndarray] = []

    for ds, count_matrix, size_factors in zip(datasets, count_matrices, size_factors_list):
        # Normalize size factors.
        if size_factors is None:
            raw_sums = np.ones(ds.rows, dtype=np.float64)
        else:
            raw_sums = size_factors.copy()

        row_median = float(np.median(raw_sums))
        if row_median > 0:
            sf_norm = raw_sums / row_median
        else:
            sf_norm = raw_sums.copy()
        sf_norm = np.where(sf_norm <= 0, 1.0, sf_norm)
        sf_norm = np.where(np.isnan(sf_norm), 1.0, sf_norm)

        # Create a minimal AnnData-like object for the writer.
        # The writer needs an adata with n_obs and obs.index.
        class _FakeAdata:
            def __init__(self, n_obs):
                self.n_obs = n_obs
                import pandas as pd
                self.obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])

        fake_adata = _FakeAdata(ds.rows)

        paths = write_lancedb_aggregated(
            adata=fake_adata,
            count_matrix=count_matrix,
            size_factors=sf_norm,
            release_id=ds.dataset_id,
            matrix_root=matrix_root,
            dataset_id=ds.dataset_id,
            corpus_index_path=corpus_index_path,
        )
        paths_list.append(paths)
        size_factors_out_list.append(sf_norm)

    # Return the paths from the last dataset's write (which is the aggregate store)
    return (paths_list[-1], size_factors_out_list)