from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csr_matrix

from perturb_data_lab.inspectors.models import (
    DatasetSummaryDocument,
    InspectionBatchConfig,
)
from perturb_data_lab.inspectors.workflow import run_batch
from perturb_data_lab.materializers import DatasetMaterializer
from perturb_data_lab.materializers.models import OutputRoots
from perturb_data_lab.materializers.obs_filter import ObsFilterError, filter_obs_rows


def _read_zarr_cell(
    indices_path: Path,
    counts_path: Path,
    row_index: int,
    *,
    row_offsets_path: Path,
    size_factor_path: Path | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    import zarr

    row_offsets = zarr.open(str(row_offsets_path), mode="r")["row_offsets"][:]
    indices = zarr.open(str(indices_path), mode="r")["indices"]
    counts = zarr.open(str(counts_path), mode="r")["counts"]
    start = int(row_offsets[row_index])
    stop = int(row_offsets[row_index + 1])
    sf = 1.0
    if size_factor_path is not None:
        sf_table = pq.read_table(size_factor_path)
        sf = float(sf_table["size_factor"][row_index].as_py())
    return (
        tuple(indices[start:stop].astype(np.int32).tolist()),
        tuple(counts[start:stop].astype(np.int32).tolist()),
        sf,
    )


def test_filter_obs_rows_supports_membership_nulls_and_parentheses() -> None:
    obs = pd.DataFrame(
        {
            "condition": ["control", "treated", None, "treated"],
            "batch": ["b1", "b2", "b1", "b3"],
            "score": [0.5, 2.0, 3.0, 1.0],
        }
    )

    retained = filter_obs_rows(
        obs,
        "(condition is null or condition in ['treated']) and batch not in ['b3'] and score >= 2",
    )

    np.testing.assert_array_equal(retained, np.array([1, 2], dtype=np.int64))


def test_filter_obs_rows_rejects_unknown_columns() -> None:
    obs = pd.DataFrame({"condition": ["control", "treated"]})

    with pytest.raises(ObsFilterError, match="unknown column"):
        filter_obs_rows(obs, "missing == 'treated'")


def test_dataset_materializer_applies_obs_filter_and_preserves_source_identity(
    tmp_path: Path,
) -> None:
    counts = csr_matrix(
        np.array(
            [
                [1, 0, 2],
                [0, 3, 0],
                [4, 0, 1],
                [0, 2, 5],
            ],
            dtype=np.int32,
        )
    )
    obs = pd.DataFrame(
        {
            "condition": ["control", "treated", "control", "treated"],
            "score": [0.1, 1.7, 2.2, 2.5],
            "batch": ["b1", "b1", "b2", "b2"],
        },
        index=["cell-0", "cell-1", "cell-2", "cell-3"],
    )
    var = pd.DataFrame(index=["gene-a", "gene-b", "gene-c"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.raw = adata.copy()

    source_path = tmp_path / "tiny-filtered.h5ad"
    adata.write_h5ad(source_path)

    summary_path = _build_review_bundle(tmp_path, source_path)
    summary = DatasetSummaryDocument.from_yaml_file(summary_path)
    replace(
        summary,
        obs_filter="condition == 'treated' and score >= 2 and batch in ['b2']",
    ).write_yaml(summary_path)

    meta_root = tmp_path / "materialized" / "metadata"
    matrix_root = tmp_path / "materialized" / "matrix"
    manifest = DatasetMaterializer(
        source_path=str(source_path),
        inspection_summary_path=str(summary_path),
        output_roots=OutputRoots(
            metadata_root=str(meta_root),
            matrix_root=str(matrix_root),
        ),
        dataset_id="tiny_filtered",
        backend="zarr",
        topology="federated",
    ).materialize()

    assert manifest.cell_count == 1

    raw_obs = pq.read_table(meta_root / "raw-obs.parquet").to_pylist()
    assert [row["cell_id"] for row in raw_obs] == ["cell-3"]
    assert [row["source_obs_index"] for row in raw_obs] == ["cell-3"]
    assert [row["source_row_index"] for row in raw_obs] == [3]
    assert json.loads(raw_obs[0]["raw_fields"]) == {
        "condition": "treated",
        "score": 2.5,
        "batch": "b2",
    }

    sf_table = pq.read_table(meta_root / "size-factor.parquet")
    assert sf_table.column("cell_id").to_pylist() == ["cell-3"]
    assert sf_table.num_rows == 1

    indices, values, _ = _read_zarr_cell(
        matrix_root / "tiny_filtered-indices.zarr",
        matrix_root / "tiny_filtered-counts.zarr",
        0,
        row_offsets_path=matrix_root / "tiny_filtered-row-offsets.zarr",
        size_factor_path=meta_root / "size-factor.parquet",
    )
    assert indices == (1, 2)
    assert values == (2, 5)

    assert not (meta_root / "metadata-summary.yaml").exists()
    assert "obs_filter retained 1/4 rows" in manifest.notes


def test_dataset_materializer_rejects_invalid_obs_filter_before_writing_outputs(
    tmp_path: Path,
) -> None:
    counts = csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int32))
    obs = pd.DataFrame({"condition": ["control", "treated"]}, index=["c0", "c1"])
    var = pd.DataFrame(index=["g0", "g1"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    source_path = tmp_path / "invalid-filter.h5ad"
    adata.write_h5ad(source_path)

    summary_path = _build_review_bundle(tmp_path, source_path, dataset_id="invalid_filter")
    summary = DatasetSummaryDocument.from_yaml_file(summary_path)
    replace(summary, obs_filter="missing == 'treated'").write_yaml(summary_path)

    meta_root = tmp_path / "out" / "metadata"
    matrix_root = tmp_path / "out" / "matrix"
    materializer = DatasetMaterializer(
        source_path=str(source_path),
        inspection_summary_path=str(summary_path),
        output_roots=OutputRoots(
            metadata_root=str(meta_root),
            matrix_root=str(matrix_root),
        ),
        dataset_id="invalid_filter",
        backend="zarr",
        topology="federated",
    )

    with pytest.raises(ObsFilterError, match="unknown column"):
        materializer.materialize()

    assert not (meta_root / "raw-obs.parquet").exists()
    assert not (meta_root / "size-factor.parquet").exists()
    assert not (matrix_root / "invalid_filter-row-offsets.zarr").exists()


def _build_review_bundle(
    tmp_path: Path,
    source_path: Path,
    *,
    dataset_id: str = "tiny_filtered",
) -> Path:
    output_root = tmp_path / "inspect"
    config_path = tmp_path / f"{dataset_id}-inspect.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_root: {output_root}",
                "datasets:",
                f"  - dataset_id: {dataset_id}",
                f"    source_path: {source_path}",
                "    source_release: synthetic-v1",
            ]
        ),
        encoding="utf-8",
    )
    run_batch(InspectionBatchConfig.from_yaml_file(config_path), workers=1)
    return output_root / dataset_id / "dataset-summary.yaml"
