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
from .loaders import (
    DatasetRoutingTable,
    ExpressionBatchDataset,
    CorpusRandomBatchSampler,
    DatasetBatchSampler,
    DatasetContextBatchSampler,
    ExpressionBatch,
    collate_expression_batch,
    collate_expression_batch_cpu,
)
from .corpus import (
    read_raw_obs_parquet,
    read_raw_var_parquet,
)
from .feature_registry import (
    FeatureRegistry,
    GlobalGeneSampler,
)
from .gene_tokenizer import (
    DatasetTokenSpan,
    GeneTokenizer,
)
from .corpus_loader import (
    Corpus,
    ObsSelection,
    load_corpus,
    select_obs_indices,
)
from .gpu_pipeline import (
    GPUSparsePipeline,
    CPUPipeline,
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
    "DatasetBatchSampler",
    "DatasetContextBatchSampler",
    # Phase 3 — Data loaders
    "DatasetRoutingTable",
    "ExpressionBatchDataset",
    "collate_expression_batch",
    "collate_expression_batch_cpu",
    # Phase 2 — Feature Registry
    "FeatureRegistry",
    "GlobalGeneSampler",
    "DatasetTokenSpan",
    "GeneTokenizer",
    # Phase 3 — GPU Pipeline
    "GPUSparsePipeline",
    "CPUPipeline",
    # Phase 4 — pertTF-local adapters
    "PertTFAdapterConfig",
    "PertTFPairedBatchLoader",
    "PertTFPairedBatchBuilder",
    "PerturbationPairBatch",
    "PerturbationPairSampler",
    "PertTFCorpusAdapter",
    # Phase N — Corpus loader factory
    "Corpus",
    "ObsSelection",
    "load_corpus",
    "select_obs_indices",
    # Utilities
    "read_raw_obs_parquet",
    "read_raw_var_parquet",
]
