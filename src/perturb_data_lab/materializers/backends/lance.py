"""Backend adapter: Lance federated and aggregate writers.

Phase 5 (this file): thin serializer refactor — all writers accept ``ChunkBundle``
directly. No per-writer CSR logic, no legacy fallback, no ``_is_csr_dataset()``.
Gene indices in ``ChunkBundle.indices`` are always dataset-local.

Produces per-dataset .lance files using ``lance.write_dataset()`` directly.

Topology: federated (per-dataset files) and aggregate (corpus-scoped single store).
Backend name in registry: ``lance``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from ..chunk_translation import ChunkBundle


def _import_lance():
    try:
        import lance
    except ImportError as exc:
        raise ImportError(
            "lance backend requires the Lance library; "
            "install lance in the selected runtime"
        ) from exc
    return lance


def write_lance_federated(
    bundle: ChunkBundle,
    release_id: str,
    matrix_root: Path,
    dataset_id: str = "",
) -> dict[str, Path]:
    """Write a ``ChunkBundle`` as a Lance dataset.

    This is the ``lance × federated`` thin serializer.
    It accepts a ``ChunkBundle`` from ``_translate_chunk()`` and writes the
    heavy-row ``table`` directly to Lance via ``lance.write_dataset``.

    Parameters
    ----------
    bundle : ChunkBundle
        The translated chunk bundle from ``_translate_chunk()``.
    release_id : str
        Release identifier used for output file naming.
    matrix_root : Path
        Output directory for matrix artifacts.
    dataset_id : str, optional
        Dataset identifier (used for Lance metadata context only).

    Returns a dict with keys: ``{"cells": lance_path}``.
    """
    lance = _import_lance()
    matrix_root.mkdir(parents=True, exist_ok=True)

    lance_path = matrix_root / f"{release_id}.lance"

    lance.write_dataset(
        bundle.table,
        str(lance_path),
        mode="create",
        max_rows_per_group=4096,
    )

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
    bundles: list[ChunkBundle],
    matrix_root: Path,
    corpus_index_path: Path | None = None,
) -> tuple[dict[str, Path], list[np.ndarray]]:
    """Write aggregate sparse per-cell data as a Lance dataset.

    This is the ``lance × aggregate`` thin serializer.
    It consumes an ordered list of ``ChunkBundle`` objects (one per dataset)
    and appends them into a single corpus-scoped .lance file with
    deterministic global_row_index values spanning all datasets.

    Parameters
    ----------
    bundles : list[ChunkBundle]
        Chunk bundles in corpus order (one per dataset).
    matrix_root : Path
        Output directory.
    corpus_index_path : Path | None
        Path to the corpus index YAML for corpus registration (unused, kept for API compat).

    Returns
    -------
    tuple[dict[str, Path], list[np.ndarray]]
        ``(paths_dict, size_factors_out_list)``.
    """
    lance = _import_lance()
    matrix_root.mkdir(parents=True, exist_ok=True)

    lance_path = matrix_root / "aggregated-corpus.lance"
    writer_initialized = False
    size_factors_out_list: list[np.ndarray] = []

    for bundle in bundles:
        mode = "append" if writer_initialized else "create"
        lance.write_dataset(
            bundle.table,
            str(lance_path),
            mode=mode,
            max_rows_per_group=4096,
        )
        writer_initialized = True
        size_factors_out_list.append(bundle.size_factors)

    return ({"cells": lance_path}, size_factors_out_list)