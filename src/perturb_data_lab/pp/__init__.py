"""Backend-agnostic streamed preprocessing helpers."""

from .artifacts import (
    PpArtifactSpec,
    build_pp_provenance,
    prepare_pp_output,
    write_pp_provenance,
)
from .de import rank_genes_ttest
from .hvg import RecalcHvgDatasetResult, RecalcHvgSummary, calculate_hvgs, recalc_hvg
from .pca import PpPcaResult, run_pca
from .stats import calculate_lognorm_stats
from .streaming import (
    PpBatch,
    PpFeatureContext,
    iter_dataset_batches,
    log1p_size_factor_batch,
)

__all__ = [
    "PpArtifactSpec",
    "PpBatch",
    "PpFeatureContext",
    "PpPcaResult",
    "RecalcHvgDatasetResult",
    "RecalcHvgSummary",
    "build_pp_provenance",
    "calculate_hvgs",
    "calculate_lognorm_stats",
    "iter_dataset_batches",
    "log1p_size_factor_batch",
    "prepare_pp_output",
    "rank_genes_ttest",
    "recalc_hvg",
    "run_pca",
    "write_pp_provenance",
]
