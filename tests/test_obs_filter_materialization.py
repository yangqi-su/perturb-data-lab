from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from perturb_data_lab.inspectors.models import (
    DatasetSummaryDocument,
    InspectionBatchConfig,
)
from perturb_data_lab.inspectors.workflow import run_batch
from perturb_data_lab.materializers import DatasetMaterializer
from perturb_data_lab.materializers.models import OutputRoots


def test_dataset_materializer_rejects_obs_filter_before_writing_outputs(
    tmp_path: Path,
) -> None:
    counts = csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int32))
    obs = pd.DataFrame({"condition": ["control", "treated"]}, index=["c0", "c1"])
    var = pd.DataFrame(index=["g0", "g1"])
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    source_path = tmp_path / "needs-prefilter.h5ad"
    adata.write_h5ad(source_path)

    summary_path = _build_inspection_summary(tmp_path, source_path)
    summary = DatasetSummaryDocument.from_yaml_file(summary_path)
    replace(summary, obs_filter="condition == 'treated'").write_yaml(summary_path)

    meta_root = tmp_path / "out" / "metadata"
    matrix_root = tmp_path / "out" / "matrix"
    materializer = DatasetMaterializer(
        source_path=str(source_path),
        inspection_summary_path=str(summary_path),
        output_roots=OutputRoots(
            metadata_root=str(meta_root),
            matrix_root=str(matrix_root),
        ),
        dataset_id="needs_prefilter",
        backend="zarr",
        topology="federated",
    )

    with pytest.raises(ValueError, match="pre-filtered h5ad"):
        materializer.materialize()

    assert not (meta_root / "raw-obs.parquet").exists()
    assert not (meta_root / "size-factor.parquet").exists()
    assert not (matrix_root / "needs_prefilter-row-offsets.zarr").exists()


def _build_inspection_summary(
    tmp_path: Path,
    source_path: Path,
    *,
    dataset_id: str = "needs_prefilter",
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
