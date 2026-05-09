"""Unit tests for ``load_corpus()`` factory with mock data (no h5ad needed)."""

from __future__ import annotations

from functools import partial
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import torch
from torch.utils.data import DataLoader
import yaml

from perturb_data_lab.loaders import (
    CPUPipeline,
    CorpusRandomBatchSampler,
    DatasetEntry,
    DatasetRoutingTable,
    ExpressionBatch,
    ExpressionBatchDataset,
    GPUSparsePipeline,
    MetadataIndex,
    collate_expression_batch,
    collate_expression_batch_cpu,
)
from perturb_data_lab.loaders.corpus_loader import (
    Corpus,
    _build_dataset_routing_table,
    _read_raw_batch,
    load_corpus,
)
from perturb_data_lab.materializers.backends import build_backend_fn
from perturb_data_lab.materializers.chunk_translation import ChunkBundle


LOADER_SEQ_LEN = 8


def _assert_processed_loader_batch(batch: dict[str, Any], *, batch_size: int) -> None:
    assert batch["batch_size"] == batch_size
    assert batch["seq_len"] == LOADER_SEQ_LEN
    assert isinstance(batch["sampled_gene_ids"], torch.Tensor)
    assert isinstance(batch["sampled_counts"], torch.Tensor)
    assert isinstance(batch["valid_mask"], torch.Tensor)
    assert isinstance(batch["exact_match_mask"], torch.Tensor)
    assert isinstance(batch["dataset_index"], torch.Tensor)
    assert isinstance(batch["global_row_index"], torch.Tensor)
    assert "row_offsets" not in batch
    assert "expressed_gene_indices" not in batch
    assert "expression_counts" not in batch


def _assert_columnar_value_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        return
    assert actual == expected


def _assert_expression_batch_equal(actual: Any, expected: Any) -> None:
    assert actual.batch_size == expected.batch_size
    np.testing.assert_array_equal(actual.global_row_index, expected.global_row_index)
    np.testing.assert_array_equal(actual.row_offsets, expected.row_offsets)
    np.testing.assert_array_equal(
        actual.expressed_gene_indices,
        expected.expressed_gene_indices,
    )
    np.testing.assert_array_equal(actual.expression_counts, expected.expression_counts)


def _assert_raw_batch_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    assert actual["batch_size"] == expected["batch_size"]
    for key in (
        "global_row_index",
        "dataset_index",
        "local_row_index",
        "row_offsets",
        "expressed_gene_indices",
        "expression_counts",
        "size_factor",
    ):
        if key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
    if "meta_columns" in expected:
        assert set(actual["meta_columns"]) == set(expected["meta_columns"])
        for key, value in expected["meta_columns"].items():
            _assert_columnar_value_equal(actual["meta_columns"][key], value)


def _assert_take_metadata_equal(
    actual: dict[str, np.ndarray | tuple],
    expected: dict[str, np.ndarray | tuple],
) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        _assert_columnar_value_equal(actual[key], value)


def _make_routing_metadata_index(
    *,
    dataset_id: Sequence[str],
    dataset_index: Sequence[int],
    local_row_index: Sequence[int],
) -> MetadataIndex:
    n_rows = len(dataset_id)
    return MetadataIndex(
        pl.DataFrame(
            {
                "global_row_index": np.arange(n_rows, dtype=np.int64),
                "cell_id": [f"cell_{idx:03d}" for idx in range(n_rows)],
                "dataset_id": list(dataset_id),
                "dataset_index": np.asarray(dataset_index, dtype=np.int32),
                "local_row_index": np.asarray(local_row_index, dtype=np.int64),
            }
        )
    )


# ---------------------------------------------------------------------------
# Helpers — build mock corpus on disk
# ---------------------------------------------------------------------------

N_GENES = 100  # per dataset


def _make_obs_df(
    dataset_id: str,
    dataset_index: int,
    n_cells: int,
    global_start: int,
    *,
    canonical: bool = True,
    include_size_factor: bool = True,
    typed_structural: bool = False,
    safe_nulls: bool = False,
) -> pl.DataFrame:
    """Create a canonical-obs DataFrame for testing."""
    rows = []
    rng = np.random.RandomState(dataset_index * 10_000 + global_start + n_cells)
    for i in range(n_cells):
        row: dict[str, Any] = {
            "cell_id": f"{dataset_id}_cell_{i:04d}",
            "dataset_id": dataset_id,
            "dataset_index": dataset_index if typed_structural else str(dataset_index),
            "global_row_index": global_start + i if typed_structural else str(global_start + i),
            "local_row_index": i if typed_structural else str(i),
            "perturb_label": "CRISPR_control" if i % 3 == 0 else "CRISPR_geneX",
            "perturb_type": "CRISPR",
            "dose": None if safe_nulls else "1.0",
            "dose_unit": None if safe_nulls else "MOI",
            "timepoint": None if safe_nulls else "7",
            "timepoint_unit": None if safe_nulls else "days",
            "cell_context": "",
            "cell_line_or_type": "K562",
            "species": "Homo sapiens",
            "tissue": "bone marrow",
            "assay": "Perturb-seq",
            "condition": "NA",
            "batch_id": f"batch_{i // 5}",
            "donor_id": "donor_01",
            "sex": "NA",
            "disease_state": "healthy",
        }
        if include_size_factor:
            size_factor = round(float(rng.uniform(0.5, 2.0)), 4)
            row["size_factor"] = size_factor if typed_structural else str(size_factor)
        rows.append(row)

    return pl.DataFrame(rows)


def _make_var_df(dataset_id: str, n_genes: int) -> pl.DataFrame:
    """Create a canonical-var DataFrame for testing."""
    rows = []
    for i in range(n_genes):
        rows.append({
            "origin_index": str(i),
            "gene_id": f"ENSG_{dataset_id}_{i:05d}",
            "canonical_gene_id": f"GENE_{i:05d}",
            "global_id": str(i),
        })
    return pl.DataFrame(rows)


def _make_lance_rows(n_cells: int) -> list[dict[str, Any]]:
    """Create expression rows for Lance files."""
    rows = []
    rng = np.random.RandomState(42)
    for _ in range(n_cells):
        n_nonzero = rng.randint(20, min(90, N_GENES))
        gene_indices = rng.choice(N_GENES, size=n_nonzero, replace=False).astype(np.int32)
        gene_indices.sort()
        counts = rng.poisson(2, size=n_nonzero).astype(np.int32)
        rows.append({
            "expressed_gene_indices": list(gene_indices),
            "expression_counts": list(counts),
        })
    return rows


def _bundle_for_rows(global_start: int, rows: list[dict[str, Any]]) -> ChunkBundle:
    indptr = [0]
    gene_indices: list[int] = []
    counts: list[int] = []
    row_sums: list[float] = []
    expressed_gene_indices = [row["expressed_gene_indices"] for row in rows]
    expression_counts = [row["expression_counts"] for row in rows]

    for genes, values in zip(expressed_gene_indices, expression_counts, strict=True):
        gene_indices.extend(genes)
        counts.extend(values)
        indptr.append(indptr[-1] + len(genes))
        row_sums.append(float(np.sum(values)))

    return ChunkBundle(
        table=pa.table(
            {
                "global_row_index": pa.array(
                    np.arange(global_start, global_start + len(rows), dtype=np.int64),
                    type=pa.int64(),
                ),
                "expressed_gene_indices": pa.array(expressed_gene_indices, type=pa.list_(pa.int32())),
                "expression_counts": pa.array(expression_counts, type=pa.list_(pa.int32())),
            }
        ),
        row_sums=np.asarray(row_sums, dtype=np.float64),
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(gene_indices, dtype=np.int32),
        counts=np.asarray(counts, dtype=np.int32),
        row_count=len(rows),
    )


def _write_backend_rows(
    *,
    backend: str,
    dataset_id: str,
    matrix_root: Path,
    global_start: int,
    rows: list[dict[str, Any]],
) -> None:
    writer = build_backend_fn(backend, "federated")
    writer(
        bundle=_bundle_for_rows(global_start, rows),
        dataset_id=dataset_id,
        matrix_root=matrix_root,
        _writer_state=None,
        _is_last_chunk=True,
    )


def _build_mock_federated_backend_corpus(
    corpus_root: Path,
    *,
    index_backend: str,
    writer_backend: str,
    include_size_factor: bool = True,
    typed_structural: bool = False,
    safe_nulls: bool = False,
) -> None:
    """Build a minimal federated corpus for a non-Lance matrix backend."""
    corpus_root.mkdir(parents=True, exist_ok=True)

    ds_configs = [
        {"dataset_id": "mock_00", "cell_count": 10, "global_start": 0, "global_end": 10, "dataset_index": 0},
        {"dataset_id": "mock_01", "cell_count": 15, "global_start": 10, "global_end": 25, "dataset_index": 1},
    ]

    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {
            "backend": index_backend,
            "topology": "federated",
        },
        "datasets": [
            {
                "dataset_id": ds["dataset_id"],
                "join_mode": "create_new" if i == 0 else "append_routed",
                "dataset_index": ds["dataset_index"],
                "cell_count": ds["cell_count"],
                "global_start": ds["global_start"],
                "global_end": ds["global_end"],
            }
            for i, ds in enumerate(ds_configs)
        ],
    }
    with open(corpus_root / "corpus-index.yaml", "w") as f:
        yaml.safe_dump(index_doc, f)

    for ds in ds_configs:
        ds_id = ds["dataset_id"]
        ds_dir = corpus_root / ds_id
        meta_dir = ds_dir / "meta" / "canonical_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        obs_df = _make_obs_df(
            ds_id,
            ds["dataset_index"],
            ds["cell_count"],
            ds["global_start"],
            include_size_factor=include_size_factor,
            typed_structural=typed_structural,
            safe_nulls=safe_nulls,
        )
        obs_df.write_parquet(str(meta_dir / "canonical-obs.parquet"))

        var_df = _make_var_df(ds_id, N_GENES)
        var_df.write_parquet(str(meta_dir / "canonical-var.parquet"))

        matrix_dir = ds_dir / "matrix"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        _write_backend_rows(
            backend=writer_backend,
            dataset_id=ds_id,
            matrix_root=matrix_dir,
            global_start=ds["global_start"],
            rows=_make_lance_rows(ds["cell_count"]),
        )


def _assert_public_api_matches_federated_lance(
    candidate: Corpus,
    lance: Corpus,
    indices: list[int],
) -> None:
    _assert_expression_batch_equal(
        candidate.read_expression(indices),
        lance.read_expression(indices),
    )
    _assert_raw_batch_equal(
        candidate.dataset().__getitems__(indices)[0],
        lance.dataset().__getitems__(indices)[0],
    )
    _assert_take_metadata_equal(
        candidate.take_metadata(indices, columns=["dataset_id", "local_row_index", "perturb_label"]),
        lance.take_metadata(indices, columns=["dataset_id", "local_row_index", "perturb_label"]),
    )
    _assert_raw_batch_equal(
        candidate.inspect_batch(indices, metadata_columns=["perturb_label"]),
        lance.inspect_batch(indices, metadata_columns=["perturb_label"]),
    )


def _build_mock_aggregate_corpus(
    corpus_root: Path,
    *,
    include_size_factor: bool = True,
    typed_structural: bool = False,
    safe_nulls: bool = False,
) -> None:
    """Build a minimal aggregate Lance corpus (2 datasets)."""
    corpus_root.mkdir(parents=True, exist_ok=True)

    # Dataset configs
    ds_configs = [
        {"dataset_id": "mock_00", "cell_count": 10, "global_start": 0, "global_end": 10, "dataset_index": 0},
        {"dataset_id": "mock_01", "cell_count": 15, "global_start": 10, "global_end": 25, "dataset_index": 1},
    ]

    # Write corpus-index.yaml
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {
            "backend": "lance",
            "topology": "aggregate",
        },
        "datasets": [
            {
                "dataset_id": ds["dataset_id"],
                "join_mode": "create_new" if i == 0 else "append_routed",
                "dataset_index": ds["dataset_index"],
                "cell_count": ds["cell_count"],
                "global_start": ds["global_start"],
                "global_end": ds["global_end"],
            }
            for i, ds in enumerate(ds_configs)
        ],
    }
    with open(corpus_root / "corpus-index.yaml", "w") as f:
        yaml.safe_dump(index_doc, f)

    # Write per-dataset canonical obs/var parquets (under meta/)
    total_cells = 0
    lance_rows: list[dict[str, Any]] = []
    for ds in ds_configs:
        ds_id = ds["dataset_id"]
        meta_dir = corpus_root / "meta" / ds_id / "canonical_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # canonical-obs.parquet
        obs_df = _make_obs_df(
            ds_id,
            ds["dataset_index"],
            ds["cell_count"],
            ds["global_start"],
            include_size_factor=include_size_factor,
            typed_structural=typed_structural,
            safe_nulls=safe_nulls,
        )
        obs_df.write_parquet(str(meta_dir / "canonical-obs.parquet"))

        # canonical-var.parquet
        var_df = _make_var_df(ds_id, N_GENES)
        var_df.write_parquet(str(meta_dir / "canonical-var.parquet"))

        # Lance expression rows
        lance_rows.extend(_make_lance_rows(ds["cell_count"]))
        total_cells += ds["cell_count"]

    # Write aggregate Lance file
    matrix_dir = corpus_root / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("expressed_gene_indices", pa.list_(pa.int32())),
        ("expression_counts", pa.list_(pa.int32())),
    ])
    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [row["expressed_gene_indices"] for row in lance_rows],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [row["expression_counts"] for row in lance_rows],
                type=pa.list_(pa.int32()),
            ),
        },
        schema=schema,
    )
    lance.write_dataset(table, str(matrix_dir / "aggregated-cells.lance"), mode="overwrite")


def _build_mock_aggregate_zarr_corpus(
    corpus_root: Path,
    *,
    include_size_factor: bool = True,
    typed_structural: bool = False,
    safe_nulls: bool = False,
) -> None:
    """Build a minimal aggregate Zarr corpus (2 datasets)."""
    corpus_root.mkdir(parents=True, exist_ok=True)

    ds_configs = [
        {"dataset_id": "mock_00", "cell_count": 10, "global_start": 0, "global_end": 10, "dataset_index": 0},
        {"dataset_id": "mock_01", "cell_count": 15, "global_start": 10, "global_end": 25, "dataset_index": 1},
    ]

    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {
            "backend": "zarr",
            "topology": "aggregate",
        },
        "datasets": [
            {
                "dataset_id": ds["dataset_id"],
                "join_mode": "create_new" if i == 0 else "append_routed",
                "dataset_index": ds["dataset_index"],
                "cell_count": ds["cell_count"],
                "global_start": ds["global_start"],
                "global_end": ds["global_end"],
            }
            for i, ds in enumerate(ds_configs)
        ],
    }
    with open(corpus_root / "corpus-index.yaml", "w") as f:
        yaml.safe_dump(index_doc, f)

    writer = build_backend_fn("zarr", "aggregate")
    writer_state: dict[str, Any] | None = None
    matrix_dir = corpus_root / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    for i, ds in enumerate(ds_configs):
        ds_id = ds["dataset_id"]
        meta_dir = corpus_root / "meta" / ds_id / "canonical_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        obs_df = _make_obs_df(
            ds_id,
            ds["dataset_index"],
            ds["cell_count"],
            ds["global_start"],
            include_size_factor=include_size_factor,
            typed_structural=typed_structural,
            safe_nulls=safe_nulls,
        )
        obs_df.write_parquet(str(meta_dir / "canonical-obs.parquet"))

        var_df = _make_var_df(ds_id, N_GENES)
        var_df.write_parquet(str(meta_dir / "canonical-var.parquet"))

        _, writer_state = writer(
            bundle=_bundle_for_rows(ds["global_start"], _make_lance_rows(ds["cell_count"])),
            dataset_id=ds_id,
            matrix_root=matrix_dir,
            _writer_state=writer_state,
            _is_last_chunk=i == len(ds_configs) - 1,
        )


def _build_mock_aggregate_tiledb_corpus(
    corpus_root: Path,
    *,
    include_size_factor: bool = True,
    typed_structural: bool = False,
    safe_nulls: bool = False,
) -> None:
    """Build a minimal aggregate TileDB corpus (2 datasets)."""
    pytest.importorskip("tiledb")
    corpus_root.mkdir(parents=True, exist_ok=True)

    ds_configs = [
        {"dataset_id": "mock_00", "cell_count": 10, "global_start": 0, "global_end": 10, "dataset_index": 0},
        {"dataset_id": "mock_01", "cell_count": 15, "global_start": 10, "global_end": 25, "dataset_index": 1},
    ]

    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {
            "backend": "tiledb",
            "topology": "aggregate",
        },
        "datasets": [
            {
                "dataset_id": ds["dataset_id"],
                "join_mode": "create_new" if i == 0 else "append_routed",
                "dataset_index": ds["dataset_index"],
                "cell_count": ds["cell_count"],
                "global_start": ds["global_start"],
                "global_end": ds["global_end"],
            }
            for i, ds in enumerate(ds_configs)
        ],
    }
    with open(corpus_root / "corpus-index.yaml", "w") as f:
        yaml.safe_dump(index_doc, f)

    writer = build_backend_fn("tiledb", "aggregate")
    writer_state: dict[str, Any] | None = None
    matrix_dir = corpus_root / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    for i, ds in enumerate(ds_configs):
        ds_id = ds["dataset_id"]
        meta_dir = corpus_root / "meta" / ds_id / "canonical_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        obs_df = _make_obs_df(
            ds_id,
            ds["dataset_index"],
            ds["cell_count"],
            ds["global_start"],
            include_size_factor=include_size_factor,
            typed_structural=typed_structural,
            safe_nulls=safe_nulls,
        )
        obs_df.write_parquet(str(meta_dir / "canonical-obs.parquet"))

        var_df = _make_var_df(ds_id, N_GENES)
        var_df.write_parquet(str(meta_dir / "canonical-var.parquet"))

        _, writer_state = writer(
            bundle=_bundle_for_rows(ds["global_start"], _make_lance_rows(ds["cell_count"])),
            dataset_id=ds_id,
            matrix_root=matrix_dir,
            _writer_state=writer_state,
            _is_last_chunk=i == len(ds_configs) - 1,
            local_vocabulary_size=N_GENES,
        )


def _build_mock_aggregate_backend_corpus_from_configs(
    corpus_root: Path,
    *,
    backend: str,
    ds_configs: Sequence[dict[str, Any]],
    include_size_factor: bool = True,
    typed_structural: bool = False,
    safe_nulls: bool = False,
) -> None:
    if backend == "tiledb":
        pytest.importorskip("tiledb")
    corpus_root.mkdir(parents=True, exist_ok=True)

    normalized_configs: list[dict[str, Any]] = []
    for ds in ds_configs:
        rows = ds["rows"]
        global_start = int(ds["global_start"])
        cell_count = len(rows)
        normalized_configs.append({
            **ds,
            "cell_count": cell_count,
            "global_end": global_start + cell_count,
        })

    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {
            "backend": backend,
            "topology": "aggregate",
        },
        "datasets": [
            {
                "dataset_id": ds["dataset_id"],
                "join_mode": "create_new" if i == 0 else "append_routed",
                "dataset_index": ds["dataset_index"],
                "cell_count": ds["cell_count"],
                "global_start": ds["global_start"],
                "global_end": ds["global_end"],
            }
            for i, ds in enumerate(normalized_configs)
        ],
    }
    with open(corpus_root / "corpus-index.yaml", "w") as f:
        yaml.safe_dump(index_doc, f)

    matrix_dir = corpus_root / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    lance_rows: list[dict[str, Any]] = []
    writer_state: dict[str, Any] | None = None
    writer = build_backend_fn("tiledb", "aggregate") if backend == "tiledb" else None

    for i, ds in enumerate(normalized_configs):
        ds_id = str(ds["dataset_id"])
        global_start = int(ds["global_start"])
        dataset_index = int(ds["dataset_index"])
        rows = list(ds["rows"])
        n_genes = int(ds["n_genes"])

        meta_dir = corpus_root / "meta" / ds_id / "canonical_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        obs_df = _make_obs_df(
            ds_id,
            dataset_index,
            len(rows),
            global_start,
            include_size_factor=include_size_factor,
            typed_structural=typed_structural,
            safe_nulls=safe_nulls,
        )
        obs_df.write_parquet(str(meta_dir / "canonical-obs.parquet"))

        var_df = _make_var_df(ds_id, n_genes)
        var_df.write_parquet(str(meta_dir / "canonical-var.parquet"))

        if backend == "lance":
            lance_rows.extend(rows)
            continue

        assert writer is not None
        _, writer_state = writer(
            bundle=_bundle_for_rows(global_start, rows),
            dataset_id=ds_id,
            matrix_root=matrix_dir,
            _writer_state=writer_state,
            _is_last_chunk=i == len(normalized_configs) - 1,
            local_vocabulary_size=n_genes,
        )

    if backend != "lance":
        return

    schema = pa.schema([
        ("expressed_gene_indices", pa.list_(pa.int32())),
        ("expression_counts", pa.list_(pa.int32())),
    ])
    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [row["expressed_gene_indices"] for row in lance_rows],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [row["expression_counts"] for row in lance_rows],
                type=pa.list_(pa.int32()),
            ),
        },
        schema=schema,
    )
    lance.write_dataset(table, str(matrix_dir / "aggregated-cells.lance"), mode="overwrite")


def _build_mock_federated_corpus(
    corpus_root: Path,
    *,
    include_size_factor: bool = True,
    typed_structural: bool = False,
    safe_nulls: bool = False,
) -> None:
    """Build a minimal federated Lance corpus (2 datasets)."""
    corpus_root.mkdir(parents=True, exist_ok=True)

    # Dataset configs
    ds_configs = [
        {"dataset_id": "mock_00", "cell_count": 10, "global_start": 0, "global_end": 10, "dataset_index": 0},
        {"dataset_id": "mock_01", "cell_count": 15, "global_start": 10, "global_end": 25, "dataset_index": 1},
    ]

    # Write corpus-index.yaml
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {
            "backend": "lance",
            "topology": "federated",
        },
        "datasets": [
            {
                "dataset_id": ds["dataset_id"],
                "join_mode": "create_new" if i == 0 else "append_routed",
                "dataset_index": ds["dataset_index"],
                "cell_count": ds["cell_count"],
                "global_start": ds["global_start"],
                "global_end": ds["global_end"],
            }
            for i, ds in enumerate(ds_configs)
        ],
    }
    with open(corpus_root / "corpus-index.yaml", "w") as f:
        yaml.safe_dump(index_doc, f)

    # Write per-dataset canonical obs/var parquets and Lance files
    for ds in ds_configs:
        ds_id = ds["dataset_id"]
        ds_dir = corpus_root / ds_id
        meta_dir = ds_dir / "meta" / "canonical_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # canonical-obs.parquet
        obs_df = _make_obs_df(
            ds_id,
            ds["dataset_index"],
            ds["cell_count"],
            ds["global_start"],
            include_size_factor=include_size_factor,
            typed_structural=typed_structural,
            safe_nulls=safe_nulls,
        )
        obs_df.write_parquet(str(meta_dir / "canonical-obs.parquet"))

        # canonical-var.parquet
        var_df = _make_var_df(ds_id, N_GENES)
        var_df.write_parquet(str(meta_dir / "canonical-var.parquet"))

        # Per-dataset Lance file
        matrix_dir = ds_dir / "matrix"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        lance_rows = _make_lance_rows(ds["cell_count"])
        schema = pa.schema([
            ("expressed_gene_indices", pa.list_(pa.int32())),
            ("expression_counts", pa.list_(pa.int32())),
        ])
        table = pa.table(
            {
                "expressed_gene_indices": pa.array(
                    [row["expressed_gene_indices"] for row in lance_rows],
                    type=pa.list_(pa.int32()),
                ),
                "expression_counts": pa.array(
                    [row["expression_counts"] for row in lance_rows],
                    type=pa.list_(pa.int32()),
                ),
            },
            schema=schema,
        )
        lance.write_dataset(table, str(matrix_dir / f"{ds_id}.lance"), mode="overwrite")


def _build_mock_federated_arrow_ipc_corpus(
    corpus_root: Path,
    *,
    include_size_factor: bool = True,
    typed_structural: bool = False,
    safe_nulls: bool = False,
) -> None:
    """Build a minimal federated Arrow IPC corpus (2 datasets)."""
    corpus_root.mkdir(parents=True, exist_ok=True)

    ds_configs = [
        {"dataset_id": "mock_00", "cell_count": 10, "global_start": 0, "global_end": 10, "dataset_index": 0},
        {"dataset_id": "mock_01", "cell_count": 15, "global_start": 10, "global_end": 25, "dataset_index": 1},
    ]

    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {
            "backend": "arrow-ipc",
            "topology": "federated",
        },
        "datasets": [
            {
                "dataset_id": ds["dataset_id"],
                "join_mode": "create_new" if i == 0 else "append_routed",
                "dataset_index": ds["dataset_index"],
                "cell_count": ds["cell_count"],
                "global_start": ds["global_start"],
                "global_end": ds["global_end"],
            }
            for i, ds in enumerate(ds_configs)
        ],
    }
    with open(corpus_root / "corpus-index.yaml", "w") as f:
        yaml.safe_dump(index_doc, f)

    schema = pa.schema([
        ("expressed_gene_indices", pa.list_(pa.int32())),
        ("expression_counts", pa.list_(pa.int32())),
    ])
    for ds in ds_configs:
        ds_id = ds["dataset_id"]
        ds_dir = corpus_root / ds_id
        meta_dir = ds_dir / "meta" / "canonical_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        obs_df = _make_obs_df(
            ds_id,
            ds["dataset_index"],
            ds["cell_count"],
            ds["global_start"],
            include_size_factor=include_size_factor,
            typed_structural=typed_structural,
            safe_nulls=safe_nulls,
        )
        obs_df.write_parquet(str(meta_dir / "canonical-obs.parquet"))

        var_df = _make_var_df(ds_id, N_GENES)
        var_df.write_parquet(str(meta_dir / "canonical-var.parquet"))

        matrix_dir = ds_dir / "matrix"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        arrow_rows = _make_lance_rows(ds["cell_count"])
        table = pa.table(
            {
                "expressed_gene_indices": pa.array(
                    [row["expressed_gene_indices"] for row in arrow_rows],
                    type=pa.list_(pa.int32()),
                ),
                "expression_counts": pa.array(
                    [row["expression_counts"] for row in arrow_rows],
                    type=pa.list_(pa.int32()),
                ),
            },
            schema=schema,
        )
        with pa.ipc.new_file(
            str(matrix_dir / f"{ds_id}-cells.arrow"),
            table.schema,
        ) as writer:
            writer.write_table(table)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadCorpusAggregate:
    """Test ``load_corpus()`` with aggregate Lance topology."""

    def test_corpus_structure(self, tmp_path: Path) -> None:
        """``load_corpus()`` returns a fully populated ``Corpus`` object."""
        _build_mock_aggregate_corpus(tmp_path)

        corpus = load_corpus(str(tmp_path))

        assert isinstance(corpus, Corpus)
        assert corpus.topology == "aggregate"
        assert corpus.backend == "lance"
        assert corpus.corpus_root == tmp_path.resolve()
        assert not hasattr(corpus, "batch_executor")
        assert corpus.feature_registry is not None
        assert corpus.metadata_index is not None
        assert len(corpus.dataset_entries) == 2

    def test_feature_registry_properties(self, tmp_path: Path) -> None:
        """Feature registry reflects mock gene vocabulary."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        fr = corpus.feature_registry
        assert fr.n_datasets == 2
        assert fr.global_vocab_size == N_GENES  # same gene pool
        assert fr.max_local_vocab == N_GENES
        assert fr.dataset_ids == ("mock_00", "mock_01")

    def test_metadata_index_properties(self, tmp_path: Path) -> None:
        """Metadata index has correct row count and columns."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        mi = corpus.metadata_index
        assert len(mi) == 25  # 10 + 15 cells
        assert "global_row_index" in mi.df.columns
        assert "cell_id" in mi.df.columns
        assert "dataset_id" in mi.df.columns
        assert "size_factor" in mi.df.columns
        assert "perturb_label" in mi.df.columns

    def test_load_corpus_casts_legacy_string_structural_fields(self, tmp_path: Path) -> None:
        """Legacy all-string canonical obs fixtures still load with typed structure."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        schema = corpus.metadata_index.df.schema
        assert schema["global_row_index"] == pl.Int64
        assert schema["dataset_index"] == pl.Int32
        assert schema["local_row_index"] == pl.Int64
        assert schema["size_factor"] == pl.Float64

    def test_load_corpus_preserves_typed_structural_fields_and_safe_nulls(
        self, tmp_path: Path,
    ) -> None:
        """Typed canonical obs and safe nulls round-trip through load_corpus()."""
        _build_mock_aggregate_corpus(
            tmp_path,
            typed_structural=True,
            safe_nulls=True,
        )
        corpus = load_corpus(str(tmp_path))

        mi = corpus.metadata_index
        schema = mi.df.schema
        assert schema["global_row_index"] == pl.Int64
        assert schema["dataset_index"] == pl.Int32
        assert schema["local_row_index"] == pl.Int64
        assert schema["size_factor"] == pl.Float64
        assert mi.df["dose"].null_count() == len(mi)
        assert mi.df["timepoint"].null_count() == len(mi)

        taken = mi.take([0, 10], ["dataset_index", "size_factor", "dose"])
        assert np.issubdtype(taken["dataset_index"].dtype, np.integer)
        assert np.issubdtype(taken["size_factor"].dtype, np.floating)
        assert taken["dose"] == (None, None)

    def test_global_row_indices_contiguous(self, tmp_path: Path) -> None:
        """Global row indices are 0..N-1 contiguous."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        gr = corpus.metadata_index.df["global_row_index"].to_numpy()
        assert gr[0] == 0
        assert gr[-1] == 24
        assert np.array_equal(gr, np.arange(25))

    def test_inspect_batch_reads_cells(self, tmp_path: Path) -> None:
        """Corpus inspection helper can read expression + metadata."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = corpus.inspect_batch([0, 5, 12, 24], metadata_columns=["perturb_label"])
        assert batch["batch_size"] == 4
        assert len(batch["global_row_index"]) == 4
        assert batch["row_offsets"][0] == 0
        # Every cell should have at least one expressed gene
        assert len(batch["expressed_gene_indices"]) > 0
        assert len(batch["expression_counts"]) > 0
        assert set(batch["meta_columns"]) == {"perturb_label"}

    def test_dataset_entries_cover_full_range(self, tmp_path: Path) -> None:
        """Dataset entries have correct global_start/end ranges."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        entries = sorted(corpus.dataset_entries, key=lambda e: e.global_start)
        assert entries[0].dataset_id == "mock_00"
        assert entries[0].global_start == 0
        assert entries[0].global_end == 10
        assert entries[1].dataset_id == "mock_01"
        assert entries[1].global_start == 10
        assert entries[1].global_end == 25


class TestLoadCorpusAggregateZarrPhase2:
    """Phase 2 aggregate Zarr public API tests."""

    @staticmethod
    def _load_backend_pair(tmp_path: Path) -> tuple[Corpus, Corpus]:
        lance_root = tmp_path / "lance"
        zarr_root = tmp_path / "zarr"
        _build_mock_aggregate_corpus(lance_root)
        _build_mock_aggregate_zarr_corpus(zarr_root)
        return load_corpus(str(lance_root)), load_corpus(str(zarr_root))

    def test_load_corpus_builds_aggregate_zarr_entries(self, tmp_path: Path) -> None:
        _build_mock_aggregate_zarr_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        assert corpus.topology == "aggregate"
        assert corpus.backend == "zarr"
        assert len(corpus.dataset_entries) == 2
        assert [
            (entry.dataset_id, entry.global_start, entry.global_end)
            for entry in corpus.dataset_entries
        ] == [
            ("mock_00", 0, 10),
            ("mock_01", 10, 25),
        ]

    def test_load_corpus_allows_missing_optional_aggregate_zarr_meta(
        self,
        tmp_path: Path,
    ) -> None:
        _build_mock_aggregate_zarr_corpus(tmp_path)
        (tmp_path / "matrix" / "aggregated-meta.json").unlink()

        corpus = load_corpus(str(tmp_path))

        assert corpus.backend == "zarr"

    @pytest.mark.parametrize(
        "indices",
        [
            [0, 1, 2, 3],
            [24, 10, 9, 0],
            [8, 10, 24, 0],
            [],
        ],
    )
    def test_aggregate_zarr_public_reads_match_aggregate_lance(
        self,
        tmp_path: Path,
        indices: list[int],
    ) -> None:
        lance_corpus, zarr_corpus = self._load_backend_pair(tmp_path)

        _assert_public_api_matches_federated_lance(zarr_corpus, lance_corpus, indices)

    def test_aggregate_zarr_cpu_loader_matches_sampler_and_metadata_contract(
        self,
        tmp_path: Path,
    ) -> None:
        lance_corpus, zarr_corpus = self._load_backend_pair(tmp_path)
        lance_corpus.set_sampler(batch_size=4, seed=11)
        zarr_corpus.set_sampler(batch_size=4, seed=11)

        lance_batch = next(
            lance_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )
        zarr_batch = next(
            zarr_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )

        _assert_processed_loader_batch(zarr_batch, batch_size=4)
        np.testing.assert_array_equal(
            zarr_batch["global_row_index"],
            lance_batch["global_row_index"],
        )
        np.testing.assert_array_equal(
            zarr_batch["dataset_index"],
            lance_batch["dataset_index"],
        )
        assert zarr_batch["meta_columns"] == lance_batch["meta_columns"]

    @pytest.mark.parametrize(
        ("artifact_name", "message"),
        [
            ("aggregated-row-offsets.zarr", "Aggregate Zarr row-offsets artifact not found"),
            ("aggregated-indices.zarr", "Aggregate Zarr indices artifact not found"),
            ("aggregated-counts.zarr", "Aggregate Zarr counts artifact not found"),
        ],
    )
    def test_missing_aggregate_zarr_artifact_fails_with_clear_error(
        self,
        tmp_path: Path,
        artifact_name: str,
        message: str,
    ) -> None:
        _build_mock_aggregate_zarr_corpus(tmp_path)
        shutil.rmtree(tmp_path / "matrix" / artifact_name)

        with pytest.raises(FileNotFoundError, match=message):
            load_corpus(str(tmp_path))


class TestLoadCorpusAggregateTileDBPhase3:
    """Phase 3 aggregate TileDB public API tests."""

    @staticmethod
    def _load_backend_pair(tmp_path: Path) -> tuple[Corpus, Corpus]:
        pytest.importorskip("tiledb")
        lance_root = tmp_path / "lance"
        tiledb_root = tmp_path / "tiledb"
        _build_mock_aggregate_corpus(lance_root)
        _build_mock_aggregate_tiledb_corpus(tiledb_root)
        return load_corpus(str(lance_root)), load_corpus(str(tiledb_root))

    def test_load_corpus_builds_aggregate_tiledb_entries(self, tmp_path: Path) -> None:
        pytest.importorskip("tiledb")
        _build_mock_aggregate_tiledb_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        assert corpus.topology == "aggregate"
        assert corpus.backend == "tiledb"
        assert len(corpus.dataset_entries) == 2
        assert [
            (entry.dataset_id, entry.global_start, entry.global_end)
            for entry in corpus.dataset_entries
        ] == [
            ("mock_00", 0, 10),
            ("mock_01", 10, 25),
        ]

    @pytest.mark.parametrize(
        "indices",
        [
            [0, 1, 2, 3],
            [24, 10, 9, 0],
            [8, 10, 24, 0],
            [],
        ],
    )
    def test_aggregate_tiledb_public_reads_match_aggregate_lance(
        self,
        tmp_path: Path,
        indices: list[int],
    ) -> None:
        lance_corpus, tiledb_corpus = self._load_backend_pair(tmp_path)

        _assert_public_api_matches_federated_lance(
            tiledb_corpus,
            lance_corpus,
            indices,
        )

    def test_aggregate_tiledb_cpu_loader_matches_sampler_and_metadata_contract(
        self,
        tmp_path: Path,
    ) -> None:
        lance_corpus, tiledb_corpus = self._load_backend_pair(tmp_path)
        lance_corpus.set_sampler(batch_size=4, seed=11)
        tiledb_corpus.set_sampler(batch_size=4, seed=11)

        lance_batch = next(
            lance_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )
        tiledb_batch = next(
            tiledb_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )

        _assert_processed_loader_batch(tiledb_batch, batch_size=4)
        np.testing.assert_array_equal(
            tiledb_batch["global_row_index"],
            lance_batch["global_row_index"],
        )
        np.testing.assert_array_equal(
            tiledb_batch["dataset_index"],
            lance_batch["dataset_index"],
        )
        assert tiledb_batch["meta_columns"] == lance_batch["meta_columns"]

    @pytest.mark.parametrize(
        ("artifact_path", "message"),
        [
            ("matrix/aggregated-cells.tiledb", "Aggregate TileDB array not found"),
            ("matrix/aggregated-meta.json", "Aggregate TileDB metadata not found"),
        ],
    )
    def test_missing_aggregate_tiledb_artifact_fails_with_clear_error(
        self,
        tmp_path: Path,
        artifact_path: str,
        message: str,
    ) -> None:
        pytest.importorskip("tiledb")
        _build_mock_aggregate_tiledb_corpus(tmp_path)
        path = tmp_path / artifact_path
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

        with pytest.raises(FileNotFoundError, match=message):
            load_corpus(str(tmp_path))


class TestLoadCorpusAggregateTileDBPhase4:
    """Phase 4 aggregate TileDB synthetic correctness tests."""

    @staticmethod
    def _mixed_local_vocab_configs() -> list[dict[str, Any]]:
        return [
            {
                "dataset_id": "mock_00",
                "dataset_index": 0,
                "global_start": 0,
                "n_genes": 5,
                "rows": [
                    {"expressed_gene_indices": [0, 2], "expression_counts": [5, 7]},
                    {"expressed_gene_indices": [], "expression_counts": []},
                    {"expressed_gene_indices": [1], "expression_counts": [3]},
                ],
            },
            {
                "dataset_id": "mock_01",
                "dataset_index": 1,
                "global_start": 3,
                "n_genes": 9,
                "rows": [
                    {"expressed_gene_indices": [0, 4], "expression_counts": [1, 2]},
                    {"expressed_gene_indices": [2, 5, 6], "expression_counts": [4, 5, 6]},
                ],
            },
        ]

    @classmethod
    def _load_backend_pair(cls, tmp_path: Path) -> tuple[Corpus, Corpus]:
        pytest.importorskip("tiledb")
        configs = cls._mixed_local_vocab_configs()
        lance_root = tmp_path / "lance"
        tiledb_root = tmp_path / "tiledb"
        _build_mock_aggregate_backend_corpus_from_configs(
            lance_root,
            backend="lance",
            ds_configs=configs,
        )
        _build_mock_aggregate_backend_corpus_from_configs(
            tiledb_root,
            backend="tiledb",
            ds_configs=configs,
        )
        return load_corpus(str(lance_root)), load_corpus(str(tiledb_root))

    def test_load_corpus_handles_mixed_local_vocab_tiledb(self, tmp_path: Path) -> None:
        pytest.importorskip("tiledb")
        _build_mock_aggregate_backend_corpus_from_configs(
            tmp_path,
            backend="tiledb",
            ds_configs=self._mixed_local_vocab_configs(),
        )

        corpus = load_corpus(str(tmp_path))

        assert corpus.backend == "tiledb"
        assert corpus.topology == "aggregate"
        assert corpus.feature_registry.max_local_vocab == 9
        assert corpus.feature_registry.global_vocab_size == 9

    @pytest.mark.parametrize(
        "indices",
        [
            [4, 1, 0, 3],
            [2, 4],
            [],
        ],
    )
    def test_aggregate_tiledb_mixed_local_vocab_public_reads_match_lance(
        self,
        tmp_path: Path,
        indices: list[int],
    ) -> None:
        lance_corpus, tiledb_corpus = self._load_backend_pair(tmp_path)

        _assert_public_api_matches_federated_lance(
            tiledb_corpus,
            lance_corpus,
            indices,
        )

    def test_aggregate_tiledb_cpu_loader_smoke_with_mixed_local_vocab_spawn(
        self,
        tmp_path: Path,
    ) -> None:
        lance_corpus, tiledb_corpus = self._load_backend_pair(tmp_path)
        lance_corpus.set_sampler(batch_size=4, seed=13)
        tiledb_corpus.set_sampler(batch_size=4, seed=13)

        lance_batch = next(
            lance_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                num_workers=1,
                multiprocessing_context="spawn",
                metadata_columns=["perturb_label"],
            )
        )
        tiledb_batch = next(
            tiledb_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                num_workers=1,
                multiprocessing_context="spawn",
                metadata_columns=["perturb_label"],
            )
        )

        _assert_processed_loader_batch(tiledb_batch, batch_size=4)
        np.testing.assert_array_equal(
            tiledb_batch["global_row_index"],
            lance_batch["global_row_index"],
        )
        np.testing.assert_array_equal(
            tiledb_batch["dataset_index"],
            lance_batch["dataset_index"],
        )
        assert tiledb_batch["meta_columns"] == lance_batch["meta_columns"]


class TestLoadCorpusFederated:
    """Test ``load_corpus()`` with federated Lance topology."""

    def test_corpus_structure(self, tmp_path: Path) -> None:
        """``load_corpus()`` returns a fully populated ``Corpus`` object."""
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        assert isinstance(corpus, Corpus)
        assert corpus.topology == "federated"
        assert corpus.backend == "lance"
        assert corpus.corpus_root == tmp_path.resolve()
        assert not hasattr(corpus, "batch_executor")
        assert corpus.feature_registry is not None
        assert corpus.metadata_index is not None
        assert len(corpus.dataset_entries) == 2

    def test_feature_registry_properties(self, tmp_path: Path) -> None:
        """Feature registry reflects mock gene vocabulary."""
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        fr = corpus.feature_registry
        assert fr.n_datasets == 2
        assert fr.global_vocab_size == N_GENES
        assert fr.max_local_vocab == N_GENES
        assert fr.dataset_ids == ("mock_00", "mock_01")

    def test_metadata_index_properties(self, tmp_path: Path) -> None:
        """Metadata index has correct row count and columns."""
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        mi = corpus.metadata_index
        assert len(mi) == 25
        assert "dataset_id" in mi.df.columns
        # Verify dataset separation
        mock00 = mi.df.filter(pl.col("dataset_id") == "mock_00")
        assert len(mock00) == 10
        mock01 = mi.df.filter(pl.col("dataset_id") == "mock_01")
        assert len(mock01) == 15

    def test_inspect_batch_reads_cells(self, tmp_path: Path) -> None:
        """Corpus inspection helper can read federated expression + metadata."""
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = corpus.inspect_batch([2, 8, 11, 20], metadata_columns=["perturb_label"])
        assert batch["batch_size"] == 4
        assert len(batch["expressed_gene_indices"]) > 0
        assert batch["meta_columns"]["perturb_label"]

    def test_dataset_entries_have_lance_paths(self, tmp_path: Path) -> None:
        """Federated entries are ``LanceDatasetEntry`` with per-file paths."""
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        from perturb_data_lab.loaders.expression import LanceDatasetEntry

        for entry in corpus.dataset_entries:
            assert isinstance(entry, LanceDatasetEntry)
            assert Path(str(entry.lance_path)).exists()
            assert str(entry.lance_path).endswith(f"{entry.dataset_id}.lance")


class TestLoadCorpusFederatedArrowIpcPhase2:
    """Phase 2 federated Arrow IPC public API tests."""

    @staticmethod
    def _load_backend_pair(tmp_path: Path) -> tuple[Corpus, Corpus]:
        lance_root = tmp_path / "lance"
        arrow_root = tmp_path / "arrow_ipc"
        _build_mock_federated_corpus(lance_root)
        _build_mock_federated_arrow_ipc_corpus(arrow_root)
        return load_corpus(str(lance_root)), load_corpus(str(arrow_root))

    def test_load_corpus_builds_arrow_ipc_dataset_entries(self, tmp_path: Path) -> None:
        _build_mock_federated_arrow_ipc_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        from perturb_data_lab.loaders.expression import ArrowIpcDatasetEntry

        assert corpus.topology == "federated"
        assert corpus.backend == "arrow_ipc"
        assert len(corpus.dataset_entries) == 2
        for entry in corpus.dataset_entries:
            assert isinstance(entry, ArrowIpcDatasetEntry)
            assert Path(str(entry.arrow_path)).is_file()
            assert str(entry.arrow_path).endswith(f"{entry.dataset_id}-cells.arrow")

    @pytest.mark.parametrize(
        "indices",
        [
            [0, 1, 2, 3],
            [24, 10, 9, 0],
            [8, 10, 24, 0],
            [],
        ],
    )
    def test_arrow_ipc_public_reads_match_federated_lance(
        self,
        tmp_path: Path,
        indices: list[int],
    ) -> None:
        lance_corpus, arrow_corpus = self._load_backend_pair(tmp_path)

        _assert_public_api_matches_federated_lance(arrow_corpus, lance_corpus, indices)

    def test_arrow_ipc_cpu_loader_matches_sampler_and_metadata_contract(
        self,
        tmp_path: Path,
    ) -> None:
        lance_corpus, arrow_corpus = self._load_backend_pair(tmp_path)
        lance_corpus.set_sampler(batch_size=4, seed=11)
        arrow_corpus.set_sampler(batch_size=4, seed=11)

        lance_batch = next(
            lance_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )
        arrow_batch = next(
            arrow_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )

        _assert_processed_loader_batch(arrow_batch, batch_size=4)
        np.testing.assert_array_equal(
            arrow_batch["global_row_index"],
            lance_batch["global_row_index"],
        )
        np.testing.assert_array_equal(
            arrow_batch["dataset_index"],
            lance_batch["dataset_index"],
        )
        assert arrow_batch["meta_columns"] == lance_batch["meta_columns"]

    def test_missing_arrow_ipc_file_fails_with_clear_error(self, tmp_path: Path) -> None:
        _build_mock_federated_arrow_ipc_corpus(tmp_path)
        arrow_path = tmp_path / "mock_00" / "matrix" / "mock_00-cells.arrow"
        arrow_path.unlink()

        with pytest.raises(FileNotFoundError, match="Arrow IPC file not found"):
            load_corpus(str(tmp_path))

    def test_arrow_ipc_read_expression_rejects_out_of_range_index(self, tmp_path: Path) -> None:
        _build_mock_federated_arrow_ipc_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        with pytest.raises(IndexError, match="out of range"):
            corpus.read_expression([25])


class TestLoadCorpusFederatedZarrPhase2:
    """Phase 2 federated Zarr public API tests."""

    @staticmethod
    def _load_backend_pair(tmp_path: Path) -> tuple[Corpus, Corpus]:
        lance_root = tmp_path / "lance"
        zarr_root = tmp_path / "zarr"
        _build_mock_federated_corpus(lance_root)
        _build_mock_federated_backend_corpus(
            zarr_root,
            index_backend="zarr",
            writer_backend="zarr",
        )
        return load_corpus(str(lance_root)), load_corpus(str(zarr_root))

    def test_load_corpus_builds_zarr_dataset_entries(self, tmp_path: Path) -> None:
        _build_mock_federated_backend_corpus(
            tmp_path,
            index_backend="zarr",
            writer_backend="zarr",
        )
        corpus = load_corpus(str(tmp_path))

        from perturb_data_lab.loaders.expression import ZarrDatasetEntry

        assert corpus.topology == "federated"
        assert corpus.backend == "zarr"
        assert len(corpus.dataset_entries) == 2
        for entry in corpus.dataset_entries:
            assert isinstance(entry, ZarrDatasetEntry)
            assert Path(str(entry.offsets_path)).is_dir()
            assert Path(str(entry.indices_path)).is_dir()
            assert Path(str(entry.counts_path)).is_dir()

    @pytest.mark.parametrize(
        "indices",
        [
            [0, 1, 2, 3],
            [24, 10, 9, 0],
            [8, 10, 24, 0],
            [],
        ],
    )
    def test_zarr_public_reads_match_federated_lance(
        self,
        tmp_path: Path,
        indices: list[int],
    ) -> None:
        lance_corpus, zarr_corpus = self._load_backend_pair(tmp_path)

        _assert_public_api_matches_federated_lance(zarr_corpus, lance_corpus, indices)

    def test_zarr_cpu_loader_matches_sampler_and_metadata_contract(
        self,
        tmp_path: Path,
    ) -> None:
        lance_corpus, zarr_corpus = self._load_backend_pair(tmp_path)
        lance_corpus.set_sampler(batch_size=4, seed=11)
        zarr_corpus.set_sampler(batch_size=4, seed=11)

        lance_batch = next(
            lance_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )
        zarr_batch = next(
            zarr_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )

        _assert_processed_loader_batch(zarr_batch, batch_size=4)
        np.testing.assert_array_equal(
            zarr_batch["global_row_index"],
            lance_batch["global_row_index"],
        )
        np.testing.assert_array_equal(
            zarr_batch["dataset_index"],
            lance_batch["dataset_index"],
        )
        assert zarr_batch["meta_columns"] == lance_batch["meta_columns"]

    @pytest.mark.parametrize(
        ("artifact_name", "message"),
        [
            ("mock_00-row-offsets.zarr", "Zarr row-offsets artifact not found"),
            ("mock_00-indices.zarr", "Zarr indices artifact not found"),
            ("mock_00-counts.zarr", "Zarr counts artifact not found"),
        ],
    )
    def test_missing_zarr_artifact_fails_with_clear_error(
        self,
        tmp_path: Path,
        artifact_name: str,
        message: str,
    ) -> None:
        _build_mock_federated_backend_corpus(
            tmp_path,
            index_backend="zarr",
            writer_backend="zarr",
        )
        artifact_path = tmp_path / "mock_00" / "matrix" / artifact_name
        shutil.rmtree(artifact_path)

        with pytest.raises(FileNotFoundError, match=message):
            load_corpus(str(tmp_path))


class TestLoadCorpusFederatedParquetPhase2:
    """Phase 2 federated Parquet public API tests."""

    @staticmethod
    def _load_backend_pair(tmp_path: Path) -> tuple[Corpus, Corpus]:
        lance_root = tmp_path / "lance"
        parquet_root = tmp_path / "parquet"
        _build_mock_federated_corpus(lance_root)
        _build_mock_federated_backend_corpus(
            parquet_root,
            index_backend="arrow-parquet",
            writer_backend="arrow-parquet",
        )
        return load_corpus(str(lance_root)), load_corpus(str(parquet_root))

    def test_load_corpus_builds_parquet_dataset_entries(self, tmp_path: Path) -> None:
        _build_mock_federated_backend_corpus(
            tmp_path,
            index_backend="arrow-parquet",
            writer_backend="arrow-parquet",
        )
        corpus = load_corpus(str(tmp_path))

        from perturb_data_lab.loaders.expression import ParquetDatasetEntry

        assert corpus.topology == "federated"
        assert corpus.backend == "parquet"
        assert len(corpus.dataset_entries) == 2
        for entry in corpus.dataset_entries:
            assert isinstance(entry, ParquetDatasetEntry)
            assert Path(str(entry.parquet_path)).is_file()
            assert str(entry.parquet_path).endswith(f"{entry.dataset_id}-cells.parquet")

    @pytest.mark.parametrize(
        "indices",
        [
            [0, 1, 2, 3],
            [24, 10, 9, 0],
            [8, 10, 24, 0],
            [],
        ],
    )
    def test_parquet_public_reads_match_federated_lance(
        self,
        tmp_path: Path,
        indices: list[int],
    ) -> None:
        lance_corpus, parquet_corpus = self._load_backend_pair(tmp_path)

        _assert_public_api_matches_federated_lance(parquet_corpus, lance_corpus, indices)

    def test_parquet_cpu_loader_matches_sampler_and_metadata_contract(
        self,
        tmp_path: Path,
    ) -> None:
        lance_corpus, parquet_corpus = self._load_backend_pair(tmp_path)
        lance_corpus.set_sampler(batch_size=4, seed=11)
        parquet_corpus.set_sampler(batch_size=4, seed=11)

        lance_batch = next(
            lance_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )
        parquet_batch = next(
            parquet_corpus.loader(
                processing="cpu",
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )

        _assert_processed_loader_batch(parquet_batch, batch_size=4)
        np.testing.assert_array_equal(
            parquet_batch["global_row_index"],
            lance_batch["global_row_index"],
        )
        np.testing.assert_array_equal(
            parquet_batch["dataset_index"],
            lance_batch["dataset_index"],
        )
        assert parquet_batch["meta_columns"] == lance_batch["meta_columns"]

    def test_missing_parquet_file_fails_with_clear_error(self, tmp_path: Path) -> None:
        _build_mock_federated_backend_corpus(
            tmp_path,
            index_backend="arrow-parquet",
            writer_backend="arrow-parquet",
        )
        parquet_path = tmp_path / "mock_00" / "matrix" / "mock_00-cells.parquet"
        parquet_path.unlink()

        with pytest.raises(FileNotFoundError, match="Parquet file not found"):
            load_corpus(str(tmp_path))


class TestCorpusApiPhase2:
    """Phase 2 corpus-level API scaffolding tests."""

    def test_set_sampler_stores_default_sampler(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        sampler = corpus.set_sampler(batch_size=4, seed=7)

        assert sampler is corpus.sampler
        assert corpus.sampler_params == {
            "sampler": "corpus_random",
            "batch_size": 4,
            "drop_last": True,
            "seed": 7,
        }

    def test_dataset_rejects_metadata_columns(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        with pytest.raises(ValueError, match="no longer accepts metadata_columns"):
            corpus.dataset(metadata_columns=["perturb_label"])

    def test_expression_dataset_uses_explicit_routing_table(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        dataset = corpus.dataset()

        assert type(dataset) is ExpressionBatchDataset
        assert type(dataset.routing_table) is DatasetRoutingTable
        np.testing.assert_array_equal(dataset.routing_table.dataset_starts, [0, 10])
        np.testing.assert_array_equal(dataset.routing_table.dataset_stops, [10, 25])
        np.testing.assert_array_equal(dataset.routing_table.dataset_indices, [0, 1])
        assert dataset.routing_table.dataset_ids == ("mock_00", "mock_01")

        batch = dataset.__getitems__([0, 10, 24])[0]
        assert set(batch) == {
            "batch_size",
            "global_row_index",
            "dataset_index",
            "size_factor",
            "row_offsets",
            "expressed_gene_indices",
            "expression_counts",
        }
        assert "local_row_index" not in batch

    def test_loader_can_attach_local_row_index_as_metadata_column(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(
            corpus.loader(
                processing="gpu",
                batch_size=4,
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["local_row_index"],
            )
        )

        _assert_processed_loader_batch(batch, batch_size=4)
        assert "local_row_index" not in batch
        expected = corpus.take_metadata(
            batch["global_row_index"].cpu().numpy(),
            columns=["local_row_index"],
        )
        np.testing.assert_array_equal(
            batch["meta_columns"]["local_row_index"],
            expected["local_row_index"],
        )

    def test_build_dataset_routing_table_rejects_cross_dataset_entry(self) -> None:
        metadata_index = _make_routing_metadata_index(
            dataset_id=("ds0", "ds1"),
            dataset_index=(0, 1),
            local_row_index=(0, 0),
        )

        with pytest.raises(ValueError, match="stay within a single dataset"):
            _build_dataset_routing_table(
                metadata_index,
                [DatasetEntry(dataset_id="entry_0", global_start=0, global_end=2)],
            )

    def test_build_dataset_routing_table_rejects_noncontiguous_local_rows(self) -> None:
        metadata_index = _make_routing_metadata_index(
            dataset_id=("ds0", "ds0"),
            dataset_index=(0, 0),
            local_row_index=(0, 2),
        )

        with pytest.raises(ValueError, match="contiguous local row indices"):
            _build_dataset_routing_table(
                metadata_index,
                [DatasetEntry(dataset_id="entry_0", global_start=0, global_end=2)],
            )

    def test_loader_defaults_to_batch_size_128(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))
        captured: dict[str, Any] = {}

        class FakeDataLoader:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                captured.update(kwargs)

            def __iter__(self):
                return iter(())

        monkeypatch.setattr(
            "perturb_data_lab.loaders.corpus_loader.DataLoader",
            FakeDataLoader,
        )

        corpus.loader(seq_len=LOADER_SEQ_LEN)

        assert captured["batch_sampler"].batch_size == 128

    def test_loader_accepts_inline_sampler_defaults(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(
            corpus.loader(
                processing="gpu",
                batch_size=4,
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )

        _assert_processed_loader_batch(batch, batch_size=4)
        assert batch["meta_columns"]["perturb_label"]

    def test_loader_metadata_attachment_supports_size_factor(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(
            corpus.loader(
                processing="gpu",
                batch_size=4,
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["size_factor", "perturb_label"],
            )
        )

        _assert_processed_loader_batch(batch, batch_size=4)
        assert set(batch["meta_columns"]) == {"size_factor", "perturb_label"}
        np.testing.assert_allclose(
            batch["meta_columns"]["size_factor"],
            batch["size_factor"].cpu().numpy(),
        )

    def test_loader_uses_stored_sampler(self, tmp_path: Path) -> None:
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))
        corpus.set_sampler(batch_size=5, seed=3)

        batch = next(corpus.loader(processing="cpu", seq_len=LOADER_SEQ_LEN))

        _assert_processed_loader_batch(batch, batch_size=5)

    def test_loader_warns_when_local_sampler_overrides_stored_sampler(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))
        corpus.set_sampler(batch_size=5, seed=3)

        with pytest.warns(UserWarning, match="loader-local sampler"):
            batch = next(
                corpus.loader(
                    processing="cpu",
                    batch_size=4,
                    seq_len=LOADER_SEQ_LEN,
                )
            )

        _assert_processed_loader_batch(batch, batch_size=4)

    def test_loader_validates_metadata_columns_and_worker_params(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        with pytest.raises(ValueError, match="reserved raw batch field"):
            corpus.inspect_batch([0], metadata_columns=["dataset_index"])

        with pytest.raises(ValueError, match="persistent_workers requires num_workers > 0"):
            next(corpus.loader(batch_size=4, persistent_workers=True))

        with pytest.raises(ValueError, match="seq_len is required"):
            next(corpus.loader(batch_size=4))

        with pytest.raises(ValueError, match="processing='cpu' only supports CPU devices"):
            next(corpus.loader(batch_size=4, seq_len=LOADER_SEQ_LEN, processing="cpu", device="cuda"))

    def test_loader_omits_metadata_columns_by_default(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(corpus.loader(processing="gpu", batch_size=4, seq_len=LOADER_SEQ_LEN))

        _assert_processed_loader_batch(batch, batch_size=4)
        assert "meta_columns" not in batch

    def test_loader_allows_multiworker_federated_lance_spawn(self, tmp_path: Path) -> None:
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(
            corpus.loader(
                batch_size=4,
                seq_len=LOADER_SEQ_LEN,
                num_workers=1,
                multiprocessing_context="spawn",
            )
        )

        _assert_processed_loader_batch(batch, batch_size=4)


class TestAggregateLancePhase4:
    """Phase 4 aggregate Lance worker-safe dataset tests."""

    def test_dataset_opens_lance_lazily(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        dataset = corpus.dataset()

        assert type(dataset) is ExpressionBatchDataset
        assert dataset.topology == "aggregate"
        assert dataset.backend == "lance"
        assert dataset._reader._dataset is None

        batch = dataset.__getitems__([0, 10, 24])[0]

        assert dataset._reader._dataset is not None
        np.testing.assert_array_equal(batch["global_row_index"], [0, 10, 24])
        assert batch["dataset_index"].tolist() == [0, 1, 1]
        assert "local_row_index" not in batch

    def test_dataset_pickle_state_drops_open_lance_handle(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        dataset = corpus.dataset()
        dataset.__getitems__([1, 11, 21])
        assert dataset._reader._dataset is not None

        restored = pickle.loads(pickle.dumps(dataset))

        assert restored._reader._dataset is None
        batch = restored.__getitems__([2, 12, 22])[0]
        np.testing.assert_array_equal(batch["global_row_index"], [2, 12, 22])

    def test_loader_allows_multiworker_aggregate_lance_spawn(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(
            corpus.loader(
                processing="gpu",
                batch_size=4,
                seq_len=LOADER_SEQ_LEN,
                num_workers=1,
                multiprocessing_context="spawn",
            )
        )

        _assert_processed_loader_batch(batch, batch_size=4)

    def test_loader_attaches_metadata_after_gpu_processing(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(
            corpus.loader(
                batch_size=4,
                seq_len=LOADER_SEQ_LEN,
                num_workers=1,
                multiprocessing_context="spawn",
                metadata_columns=["perturb_label", "cell_line_or_type"],
            )
        )

        _assert_processed_loader_batch(batch, batch_size=4)
        assert batch["meta_columns"]["perturb_label"]
        assert batch["meta_columns"]["cell_line_or_type"] == (
            "K562",
            "K562",
            "K562",
            "K562",
        )

    def test_loader_dataset_sampler_drives_aggregate_batches(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))
        corpus.set_sampler(
            sampler="dataset",
            dataset_index=1,
            batch_size=4,
            seed=7,
        )

        batch = next(corpus.loader(processing="gpu", seq_len=LOADER_SEQ_LEN))

        _assert_processed_loader_batch(batch, batch_size=4)
        global_indices = batch["global_row_index"].cpu().numpy()
        assert np.all(global_indices >= 10)
        assert np.all(global_indices < 25)

    def test_loader_dataset_context_sampler_preserves_context_group(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(
            corpus.loader(
                processing="gpu",
                sampler="dataset_context",
                dataset_index=1,
                context_field="perturb_label",
                batch_size=3,
                seq_len=LOADER_SEQ_LEN,
                metadata_columns=["perturb_label"],
            )
        )

        _assert_processed_loader_batch(batch, batch_size=3)
        assert len(set(batch["meta_columns"]["perturb_label"])) == 1


class TestFederatedLancePhase5:
    """Phase 5 federated Lance worker-safe dataset tests."""

    def test_dataset_opens_federated_lance_lazily(self, tmp_path: Path) -> None:
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        dataset = corpus.dataset()

        assert type(dataset) is ExpressionBatchDataset
        assert dataset.topology == "federated"
        assert dataset.backend == "lance"
        assert dataset._reader._datasets == {}

        batch = dataset.__getitems__([24, 0, 10, 9])[0]

        assert set(dataset._reader._datasets) == {"mock_00", "mock_01"}
        np.testing.assert_array_equal(batch["global_row_index"], [24, 0, 10, 9])
        assert batch["dataset_index"].tolist() == [1, 0, 1, 0]
        assert "local_row_index" not in batch

    def test_dataset_pickle_state_drops_open_federated_handles(self, tmp_path: Path) -> None:
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        dataset = corpus.dataset()
        dataset.__getitems__([1, 11, 21])
        assert dataset._reader._datasets

        restored = pickle.loads(pickle.dumps(dataset))

        assert restored._reader._datasets == {}
        batch = restored.__getitems__([2, 12, 22])[0]
        np.testing.assert_array_equal(batch["global_row_index"], [2, 12, 22])
        assert set(restored._reader._datasets) == {"mock_00", "mock_01"}

    def test_loader_attaches_metadata_after_cpu_processing(self, tmp_path: Path) -> None:
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(
            corpus.loader(
                processing="cpu",
                batch_size=4,
                seq_len=LOADER_SEQ_LEN,
                num_workers=1,
                multiprocessing_context="spawn",
                metadata_columns=["perturb_label"],
            )
        )

        _assert_processed_loader_batch(batch, batch_size=4)
        assert batch["sampled_gene_ids"].device.type == "cpu"
        assert batch["meta_columns"]["perturb_label"]

    def test_loader_dataset_sampler_drives_federated_batches(self, tmp_path: Path) -> None:
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))
        corpus.set_sampler(
            sampler="dataset",
            dataset_index=0,
            batch_size=4,
            seed=11,
        )

        batch = next(corpus.loader(processing="cpu", seq_len=LOADER_SEQ_LEN))

        _assert_processed_loader_batch(batch, batch_size=4)
        global_indices = batch["global_row_index"].cpu().numpy()
        assert np.all(global_indices >= 0)
        assert np.all(global_indices < 10)


class TestCorpusInspectionHelpersPhase5:
    """Phase 5 corpus-level inspection helper tests."""

    def test_read_expression_matches_expression_reader(self, tmp_path: Path) -> None:
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))
        indices = [24, 0, 10, 9]

        expr = corpus.read_expression(indices)
        expected = corpus.expression_reader.read_expression_flat(indices)

        _assert_expression_batch_equal(expr, expected)

    def test_take_metadata_matches_metadata_index_take(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(
            tmp_path,
            typed_structural=True,
            safe_nulls=True,
        )
        corpus = load_corpus(str(tmp_path))
        indices = [0, 10]

        taken = corpus.take_metadata(
            indices,
            columns=["dataset_id", "cell_id", "dataset_index", "size_factor", "dose", "timepoint", "perturb_label"],
        )
        expected = corpus.metadata_index.take(
            indices,
            ["dataset_id", "cell_id", "dataset_index", "size_factor", "dose", "timepoint", "perturb_label"],
        )

        assert np.issubdtype(taken["dataset_index"].dtype, np.integer)
        assert np.issubdtype(taken["size_factor"].dtype, np.floating)
        assert taken["dose"] == (None, None)
        assert taken["timepoint"] == (None, None)
        _assert_columnar_value_equal(taken["dataset_id"], expected["dataset_id"])
        _assert_columnar_value_equal(taken["cell_id"], expected["cell_id"])
        _assert_columnar_value_equal(taken["dataset_index"], expected["dataset_index"])
        np.testing.assert_allclose(taken["size_factor"], expected["size_factor"])
        _assert_columnar_value_equal(taken["dose"], expected["dose"])
        _assert_columnar_value_equal(taken["timepoint"], expected["timepoint"])
        _assert_columnar_value_equal(taken["perturb_label"], expected["perturb_label"])

    def test_inspect_batch_matches_direct_raw_batch_contract(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path, typed_structural=True, safe_nulls=True)
        corpus = load_corpus(str(tmp_path))
        indices = [0, 5, 12, 24]

        batch = corpus.inspect_batch(
            indices,
            metadata_columns=["perturb_label", "dose", "size_factor"],
        )
        expected = _read_raw_batch(
            corpus.expression_reader,
            corpus.metadata_index,
            indices,
            metadata_columns=["perturb_label", "dose", "size_factor"],
        )

        assert set(batch.keys()) == set(expected.keys())
        for key in (
            "global_row_index",
            "dataset_index",
            "local_row_index",
            "row_offsets",
            "expressed_gene_indices",
            "expression_counts",
            "size_factor",
        ):
            np.testing.assert_array_equal(batch[key], expected[key])
        assert set(batch["meta_columns"]) == set(expected["meta_columns"])
        for key, value in expected["meta_columns"].items():
            _assert_columnar_value_equal(batch["meta_columns"][key], value)


class TestLegacySurfaceRemovalPhase6:
    """Phase 6 import and API removal checks."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "BatchExecutor",
            "RawExpressionBatch",
            "RawExpressionBatchDataset",
            "PerturbBatchDataset",
            "collate_batch_dict",
            "FastTrainingBatch",
            "BatchMetadata",
            "LanceExpressionBatchDataset",
            "AggregateLanceExpressionBatchDataset",
        ],
    )
    def test_removed_public_symbols_are_not_importable(self, symbol: str) -> None:
        with pytest.raises(ImportError):
            exec(f"from perturb_data_lab.loaders import {symbol}", {}, {})


class TestCorpusLoaderPhase6:
    """Phase 6 end-to-end CPU/GPU loader routing tests."""

    def test_aggregate_gpu_loader_route(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(corpus.loader(processing="gpu", batch_size=4, seq_len=LOADER_SEQ_LEN))

        _assert_processed_loader_batch(batch, batch_size=4)
        assert batch["sampled_gene_ids"].shape == (4, LOADER_SEQ_LEN)
        assert "local_row_index" not in batch

    def test_aggregate_cpu_loader_route(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(corpus.loader(processing="cpu", batch_size=4, seq_len=LOADER_SEQ_LEN))

        _assert_processed_loader_batch(batch, batch_size=4)
        assert batch["sampled_gene_ids"].device.type == "cpu"
        assert "local_row_index" not in batch

    def test_federated_gpu_loader_route(self, tmp_path: Path) -> None:
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(corpus.loader(processing="gpu", batch_size=4, seq_len=LOADER_SEQ_LEN))

        _assert_processed_loader_batch(batch, batch_size=4)
        assert batch["sampled_gene_ids"].shape == (4, LOADER_SEQ_LEN)
        assert "local_row_index" not in batch

    def test_federated_cpu_loader_route(self, tmp_path: Path) -> None:
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(corpus.loader(processing="cpu", batch_size=4, seq_len=LOADER_SEQ_LEN))

        _assert_processed_loader_batch(batch, batch_size=4)
        assert batch["sampled_gene_ids"].device.type == "cpu"
        assert "local_row_index" not in batch

    def test_cpu_loader_supports_multiworker_spawn_aggregate(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(
            corpus.loader(
                processing="cpu",
                batch_size=4,
                seq_len=LOADER_SEQ_LEN,
                num_workers=1,
                multiprocessing_context="spawn",
            )
        )

        _assert_processed_loader_batch(batch, batch_size=4)

    def test_cpu_loader_supports_multiworker_spawn_federated(self, tmp_path: Path) -> None:
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = next(
            corpus.loader(
                processing="cpu",
                batch_size=4,
                seq_len=LOADER_SEQ_LEN,
                num_workers=1,
                multiprocessing_context="spawn",
            )
        )

        _assert_processed_loader_batch(batch, batch_size=4)

    def test_lance_workers_default_to_spawn(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))
        captured: dict[str, Any] = {}

        class FakeDataLoader:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                captured.update(kwargs)

            def __iter__(self):
                return iter(())

        monkeypatch.setattr(
            "perturb_data_lab.loaders.corpus_loader.DataLoader",
            FakeDataLoader,
        )

        corpus.loader(processing="gpu", batch_size=4, seq_len=LOADER_SEQ_LEN, num_workers=1)

        assert captured["multiprocessing_context"] == "spawn"


class TestModernCollatesPhase5:
    """Phase 5 custom DataLoader and collate tests."""

    def test_custom_dataloader_supports_collate_expression_batch(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))
        dataset = corpus.dataset()
        sampler = CorpusRandomBatchSampler(
            metadata_index=corpus.metadata_index,
            batch_size=3,
            drop_last=False,
            seed=5,
        )

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_expression_batch,
            num_workers=0,
        )

        batch = next(iter(loader))

        assert batch["batch_size"] == 3
        assert isinstance(batch["global_row_index"], torch.Tensor)
        assert isinstance(batch["dataset_index"], torch.Tensor)
        assert isinstance(batch["row_offsets"], torch.Tensor)
        assert isinstance(batch["expressed_gene_indices"], torch.Tensor)
        assert isinstance(batch["expression_counts"], torch.Tensor)
        assert batch["global_row_index"].dtype == torch.long
        assert batch["size_factor"].dtype == torch.float32
        assert "local_row_index" not in batch

    def test_custom_dataloader_supports_collate_expression_batch_cpu(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))
        dataset = corpus.dataset()
        sampler = CorpusRandomBatchSampler(
            metadata_index=corpus.metadata_index,
            batch_size=3,
            drop_last=False,
            seed=7,
        )
        pipeline = GPUSparsePipeline(corpus.feature_registry, seq_len=LOADER_SEQ_LEN)

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=partial(
                collate_expression_batch_cpu,
                pipeline=pipeline,
                sampled_gene_ids=torch.arange(
                    LOADER_SEQ_LEN,
                    dtype=torch.long,
                ).repeat(3, 1),
            ),
            num_workers=0,
        )

        batch = next(iter(loader))

        _assert_processed_loader_batch(batch, batch_size=3)
        assert batch["sampled_gene_ids"].device.type == "cpu"

    def test_collate_expression_batch_preserves_meta_columns(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = corpus.inspect_batch([0, 10, 24], metadata_columns=["perturb_label"])
        collated = collate_expression_batch([batch])

        assert collated["meta_columns"] == batch["meta_columns"]
        assert collated["size_factor"].dtype == torch.float32

    def test_collate_expression_batch_cpu_preserves_meta_columns(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))
        pipeline = GPUSparsePipeline(corpus.feature_registry, seq_len=LOADER_SEQ_LEN)
        batch = corpus.inspect_batch([1, 11, 21], metadata_columns=["perturb_label"])

        collated = collate_expression_batch_cpu(
            [batch],
            pipeline=pipeline,
            sampled_gene_ids=torch.arange(LOADER_SEQ_LEN, dtype=torch.long).repeat(3, 1),
        )

        assert collated["meta_columns"] == batch["meta_columns"]
        assert collated["size_factor"].dtype == torch.float32


class TestSparsePipelinePhase3:
    """Phase 3 metadata-light sparse pipeline tests."""

    def test_raw_loader_path_allows_missing_size_factor(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path, include_size_factor=False)
        corpus = load_corpus(str(tmp_path))

        raw_batch = corpus.dataset().__getitems__([0, 10, 24])[0]
        assert "size_factor" not in raw_batch

        loader_batch = next(corpus.loader(processing="gpu", batch_size=4, seq_len=LOADER_SEQ_LEN))
        assert "size_factor" not in loader_batch

    def test_gpu_pipeline_output_is_size_factor_independent(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch_with_size_factor = corpus.dataset().__getitems__([0, 10, 24])[0]
        batch_without_size_factor = {
            key: value
            for key, value in batch_with_size_factor.items()
            if key != "size_factor"
        }

        collated_with = collate_expression_batch([batch_with_size_factor])
        collated_without = collate_expression_batch([batch_without_size_factor])
        pipeline = GPUSparsePipeline(corpus.feature_registry, seq_len=8)
        sampled_gene_ids = torch.arange(8, dtype=torch.long).repeat(3, 1)

        result_with = pipeline.process_batch(
            collated_with, device="cpu", sampled_gene_ids=sampled_gene_ids
        )
        result_without = pipeline.process_batch(
            collated_without, device="cpu", sampled_gene_ids=sampled_gene_ids
        )

        assert "size_factor" in result_with
        assert "size_factor" not in result_without
        assert result_with["batch_size"] == result_without["batch_size"] == 3
        assert result_with["seq_len"] == result_without["seq_len"] == 8
        for key in (
            "sampled_gene_ids",
            "sampled_counts",
            "valid_mask",
            "exact_match_mask",
            "dataset_index",
            "global_row_index",
        ):
            assert torch.equal(result_with[key], result_without[key])

    def test_collate_expression_batch_cpu_allows_missing_size_factor(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch_with_size_factor = corpus.dataset().__getitems__([1, 11, 21])[0]
        batch_without_size_factor = {
            key: value
            for key, value in batch_with_size_factor.items()
            if key != "size_factor"
        }

        pipeline = GPUSparsePipeline(corpus.feature_registry, seq_len=8)
        sampled_gene_ids = torch.arange(8, dtype=torch.long).repeat(3, 1)
        collate = partial(
            collate_expression_batch_cpu,
            pipeline=pipeline,
            sampled_gene_ids=sampled_gene_ids,
        )

        result_with = collate([batch_with_size_factor])
        result_without = collate([batch_without_size_factor])

        assert "size_factor" in result_with
        assert "size_factor" not in result_without
        for key in (
            "sampled_gene_ids",
            "sampled_counts",
            "valid_mask",
            "exact_match_mask",
            "dataset_index",
            "global_row_index",
        ):
            assert torch.equal(result_with[key], result_without[key])

    def test_cpu_pipeline_allows_missing_size_factor(self, tmp_path: Path) -> None:
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch_with_size_factor = corpus.dataset().__getitems__([2, 12, 22])[0]
        batch_without_size_factor = {
            key: value
            for key, value in batch_with_size_factor.items()
            if key != "size_factor"
        }

        pipeline = CPUPipeline(corpus.feature_registry, seq_len=8, seed=7)
        sampled_gene_ids = np.arange(8, dtype=np.int32).reshape(1, 8).repeat(3, axis=0)

        result_with = pipeline.process_batch(
            batch_with_size_factor, sampled_gene_ids=sampled_gene_ids
        )
        result_without = pipeline.process_batch(
            batch_without_size_factor, sampled_gene_ids=sampled_gene_ids
        )

        assert "size_factor" in result_with
        assert "size_factor" not in result_without
        assert result_with["batch_size"] == result_without["batch_size"] == 3
        assert result_with["seq_len"] == result_without["seq_len"] == 8
        for key in (
            "sampled_gene_ids",
            "sampled_counts",
            "valid_mask",
            "exact_match_mask",
            "dataset_index",
            "global_row_index",
        ):
            np.testing.assert_array_equal(result_with[key], result_without[key])


class TestLoadCorpusErrors:
    """Error handling tests."""

    def test_missing_corpus_index(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when corpus-index.yaml is missing."""
        with pytest.raises(FileNotFoundError, match="corpus-index.yaml"):
            load_corpus(str(tmp_path))

    def test_missing_canonical_obs(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when canonical-obs.parquet is missing."""
        _build_mock_aggregate_corpus(tmp_path)
        # Remove one obs file
        obs_path = tmp_path / "meta" / "mock_00" / "canonical_meta" / "canonical-obs.parquet"
        obs_path.unlink()

        with pytest.raises(FileNotFoundError, match="canonical-obs.parquet"):
            load_corpus(str(tmp_path))

    def test_missing_canonical_var(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when canonical-var.parquet is missing."""
        _build_mock_aggregate_corpus(tmp_path)
        # Remove one var file
        var_path = tmp_path / "meta" / "mock_00" / "canonical_meta" / "canonical-var.parquet"
        var_path.unlink()

        with pytest.raises(FileNotFoundError, match="canonical-var.parquet"):
            load_corpus(str(tmp_path))

    def test_missing_lance_file_aggregate(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when aggregate Lance file is missing."""
        _build_mock_aggregate_corpus(tmp_path)
        lance_path = tmp_path / "matrix" / "aggregated-cells.lance"
        import shutil
        shutil.rmtree(lance_path)

        with pytest.raises(FileNotFoundError, match="aggregated-cells.lance"):
            load_corpus(str(tmp_path))

    def test_missing_lance_file_federated(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when a federated Lance file is missing."""
        _build_mock_federated_corpus(tmp_path)
        lance_path = tmp_path / "mock_00" / "matrix" / "mock_00.lance"
        import shutil
        shutil.rmtree(lance_path)

        with pytest.raises(FileNotFoundError, match=".lance"):
            load_corpus(str(tmp_path))

    def test_unknown_topology(self, tmp_path: Path) -> None:
        """Raises ValueError for unknown topology."""
        _build_mock_aggregate_corpus(tmp_path)
        # Corrupt the topology in the index
        index_path = tmp_path / "corpus-index.yaml"
        with open(index_path) as f:
            doc = yaml.safe_load(f)
        doc["global_metadata"]["topology"] = "unknown"
        with open(index_path, "w") as f:
            yaml.safe_dump(doc, f)

        with pytest.raises(ValueError, match="topology"):
            load_corpus(str(tmp_path))

    def test_unknown_backend(self, tmp_path: Path) -> None:
        """Raises ValueError for unknown backend."""
        _build_mock_aggregate_corpus(tmp_path)
        # Corrupt the backend
        index_path = tmp_path / "corpus-index.yaml"
        with open(index_path) as f:
            doc = yaml.safe_load(f)
        doc["global_metadata"]["backend"] = "unknown_backend"
        with open(index_path, "w") as f:
            yaml.safe_dump(doc, f)

        with pytest.raises(ValueError, match="backend"):
            load_corpus(str(tmp_path))
