"""Public slim-main corpus loader and pertTF API exports."""

from .expression import (
    AggregateLanceReader,
    AggregateZarrReader,
    BaseExpressionReader,
    DatasetEntry,
    ExpressionReader,
    FederatedLanceReader,
    FederatedZarrReader,
    LanceDatasetEntry,
    ZarrDatasetEntry,
    build_expression_reader,
)
from .index import MetadataIndex
from .adapters import (
    ExpressionBatchDataset,
    CorpusRandomBatchSampler,
    ContextBatchSampler,
    build_loader,
    collate_expression_batch,
)
from .expression import ExpressionBatch
from .feature_registry import (
    FeatureRegistry,
)
from .gene_token_mapper import (
    GeneTokenMapper,
)
from .corpus_loader import (
    Corpus,
    load_corpus,
)
from .sparse_batch import (
    SparseBatchProcessor,
)
from .adapters import (
    PertTFAdapterConfig,
    PertTFPairedBatchLoader,
    PertTFPairedBatchBuilder,
    PerturbationPairBatch,
    PerturbationPairSampler,
    PertTFCorpusAdapter,
)

__all__ = [
    # Phase 1 — MetadataIndex
    "MetadataIndex",
    # Phase 2 — ExpressionReader (backend-agnostic)
    "ExpressionReader",
    "BaseExpressionReader",
    "DatasetEntry",
    "AggregateLanceReader",
    "FederatedLanceReader",
    "AggregateZarrReader",
    "FederatedZarrReader",
    "LanceDatasetEntry",
    "ZarrDatasetEntry",
    "build_expression_reader",
    # Phase 3 — Core types
    "ExpressionBatch",
    # Phase 3 — Samplers (MetadataIndex-backed)
    "CorpusRandomBatchSampler",
    "ContextBatchSampler",
    # Phase 3 — Data loaders
    "ExpressionBatchDataset",
    "build_loader",
    "collate_expression_batch",
    # Phase 2 — Feature Registry
    "FeatureRegistry",
    "GeneTokenMapper",
    # Phase 3 — sparse batch processing
    "SparseBatchProcessor",
    # Phase 4 — pertTF-local adapters
    "PertTFAdapterConfig",
    "PertTFPairedBatchLoader",
    "PertTFPairedBatchBuilder",
    "PerturbationPairBatch",
    "PerturbationPairSampler",
    "PertTFCorpusAdapter",
    # Phase N — Corpus loader factory
    "Corpus",
    "load_corpus",
]
