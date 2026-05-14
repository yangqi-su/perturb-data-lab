"""Public slim-main model adapter exports.

The pertTF adapter namespace is kept here so the retained training-facing
surface can evolve without modifying the external ``pertTF`` repository.
Legacy mapping helpers remain internal to the implementation module.
"""

from .perttf import (
    PertTFAdapterConfig,
    PertTFPairedBatchLoader,
    PertTFPairedBatchBuilder,
    PerturbationPairBatch,
    PerturbationPairSampler,
    PertTFCorpusAdapter,
)

__all__ = [
    "PertTFAdapterConfig",
    "PertTFPairedBatchLoader",
    "PertTFPairedBatchBuilder",
    "PerturbationPairBatch",
    "PerturbationPairSampler",
    "PertTFCorpusAdapter",
]
