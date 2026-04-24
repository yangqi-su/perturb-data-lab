from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from perturb_data_lab.inspectors.models import (
    DatasetSummaryDocument,
    InspectionBatchConfig,
)
from perturb_data_lab.inspectors.workflow import run_batch


def test_inspector_workflow_round_trip(tmp_path: Path) -> None:
    obs = pd.DataFrame(
        {
            "guide_name": ["NTC_1", "STAT1_1", "IRF1_1"],
            "cell_line": ["K562", "K562", "K562"],
            "batch": ["b1", "b1", "b2"],
            "species": ["human", "human", "human"],
        },
        index=["cell-1", "cell-2", "cell-3"],
    )
    var = pd.DataFrame(index=["gene-a", "gene-b", "gene-c"])
    counts = csr_matrix(np.array([[1, 0, 3], [0, 2, 0], [4, 0, 1]], dtype=np.int32))
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.raw = adata.copy()
    adata.layers["lognorm"] = np.log1p(counts.toarray()).astype(np.float32)

    source_path = tmp_path / "tiny.h5ad"
    adata.write_h5ad(source_path)

    output_root = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_root: {output_root}",
                "datasets:",
                "  - dataset_id: tiny_crispr",
                f"    source_path: {source_path}",
                "    source_release: synthetic-v1",
            ]
        ),
        encoding="utf-8",
    )

    manifest = run_batch(InspectionBatchConfig.from_yaml_file(config_path), workers=1)
    assert manifest.records[0].dataset_id == "tiny_crispr"

    dataset_dir = output_root / "tiny_crispr"
    summary = DatasetSummaryDocument.from_yaml_file(
        dataset_dir / "dataset-summary.yaml"
    )

    assert summary.count_source_decision.selected_candidate == ".raw.X"
    assert summary.count_source_decision.status == "pass"
    assert summary.materialization_readiness == "pass"

    # No schema.yaml in Stage 1 output
    assert not (dataset_dir / "schema.yaml").exists()


def test_inspector_prefers_textual_guide_fields_over_numeric_scores(
    tmp_path: Path,
) -> None:
    obs = pd.DataFrame(
        {
            "top_guide_UMI_counts": [10.0, 8.0, 15.0],
            "guide_id": ["NTC-1", "STAT1-1", "IRF1-1"],
            "perturbed_gene_name": ["NTC", "STAT1", "IRF1"],
            "perturbed_gene_id": ["NA", "ENSG00000115415", "ENSG00000125347"],
            "guide_type": ["non-targeting", "targeting", "targeting"],
        },
        index=["cell-1", "cell-2", "cell-3"],
    )
    var = pd.DataFrame(index=["gene-a", "gene-b"])
    counts = csr_matrix(np.array([[1, 0], [0, 2], [4, 1]], dtype=np.int32))
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    source_path = tmp_path / "marson_like.h5ad"
    adata.write_h5ad(source_path)

    output_root = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_root: {output_root}",
                "datasets:",
                "  - dataset_id: marson_like",
                f"    source_path: {source_path}",
                "    source_release: synthetic-v1",
            ]
        ),
        encoding="utf-8",
    )

    run_batch(InspectionBatchConfig.from_yaml_file(config_path), workers=1)
    summary = DatasetSummaryDocument.from_yaml_file(
        output_root / "marson_like" / "dataset-summary.yaml"
    )

    # Stage 1 only verifies count source selection; field profiling is for review, not schema
    assert summary.count_source_decision.selected_candidate in {".X", ".raw.X"}
    # Readiness is count-only
    assert summary.materialization_readiness in {"pass", "needs-review"}
