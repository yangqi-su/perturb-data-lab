"""Public slim-main adapter exports.

The pertTF adapter namespace is kept here so the retained training-facing
surface can evolve without modifying the external ``pertTF`` repository.
Legacy mapping helpers remain internal to the implementation module.
"""

from .standard import (
    ExpressionBatchDataset,
    CorpusRandomBatchSampler,
    ContextBatchSampler,
    build_loader,
    collate_expression_batch,
)
from .perttf import (
    PertTFAdapterConfig,
    PertTFPairedBatchLoader,
    PertTFPairedBatchBuilder,
    PerturbationPairBatch,
    PerturbationPairSampler,
    PertTFCorpusAdapter,
)

__all__ = [
    "ExpressionBatchDataset",
    "CorpusRandomBatchSampler",
    "ContextBatchSampler",
    "build_loader",
    "collate_expression_batch",
    "PertTFAdapterConfig",
    "PertTFPairedBatchLoader",
    "PertTFPairedBatchBuilder",
    "PerturbationPairBatch",
    "PerturbationPairSampler",
    "PertTFCorpusAdapter",
]
