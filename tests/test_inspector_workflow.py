from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from perturb_data_lab.inspectors.models import (
    DatasetSummaryDocument,
    InspectionBatchConfig,
    SchemaPatchDocument,
    SchemaProposalDocument,
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
    proposal = SchemaProposalDocument.from_yaml_file(
        dataset_dir / "schema-proposal.yaml"
    )
    patch = SchemaPatchDocument.from_yaml_file(dataset_dir / "schema-patch.yaml")

    assert summary.count_source_decision.selected_candidate == ".raw.X"
    assert summary.count_source_decision.status == "pass"
    assert proposal.field_mappings["dataset_id"].literal_value == "tiny_crispr"
    assert proposal.field_mappings["perturbation_type"].literal_value == "CRISPR"
    assert patch.review_status == "pending"


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
    proposal = SchemaProposalDocument.from_yaml_file(
        output_root / "marson_like" / "schema-proposal.yaml"
    )

    assert proposal.field_mappings["perturbation_label"].source_fields == ("guide_id",)
    assert proposal.field_mappings["target_label"].source_fields == (
        "perturbed_gene_name",
    )
    assert proposal.field_mappings["target_id"].source_fields == ("perturbed_gene_id",)
