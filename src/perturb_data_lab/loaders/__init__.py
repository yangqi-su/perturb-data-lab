"""Public composable corpus-loader API exports."""

from .expression import (
    AggregateLanceReader,
    AggregateTileDBReader,
    AggregateZarrReader,
    AggregateCsrMemmapReader,
    ArrowIpcDatasetEntry,
    BaseExpressionReader,
    CsrMemmapShardEntry,
    DatasetEntry,
    ExpressionReader,
    FederatedArrowIpcReader,
    FederatedHfDatasetsReader,
    FederatedLanceReader,
    FederatedParquetReader,
    FederatedWebDatasetReader,
    FederatedZarrReader,
    HfDatasetsDatasetEntry,
    LanceDatasetEntry,
    ParquetDatasetEntry,
    WebDatasetEntry,
    ZarrDatasetEntry,
    build_expression_reader,
)
from .index import MetadataIndex, MetadataRow
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
    load_corpus,
)
from .gpu_pipeline import (
    GPUSparsePipeline,
    CPUPipeline,
)
from .adapters import (
    CategoricalLabelMap,
    PertTFAdapterConfig,
    PerturbationPairBatch,
    PerturbationPairSampler,
    PertTFCorpusAdapter,
    PertTFLabelAdapter,
    PertTFVocabAdapter,
)

__all__ = [
    # Phase 1 — MetadataIndex
    "MetadataIndex",
    "MetadataRow",
    # Phase 2 — ExpressionReader (backend-agnostic)
    "ExpressionReader",
    "BaseExpressionReader",
    "DatasetEntry",
    "AggregateLanceReader",
    "AggregateTileDBReader",
    "FederatedLanceReader",
    "AggregateZarrReader",
    "FederatedZarrReader",
    "AggregateCsrMemmapReader",
    "FederatedArrowIpcReader",
    "FederatedHfDatasetsReader",
    "FederatedParquetReader",
    "FederatedWebDatasetReader",
    "LanceDatasetEntry",
    "ZarrDatasetEntry",
    "ArrowIpcDatasetEntry",
    "HfDatasetsDatasetEntry",
    "ParquetDatasetEntry",
    "WebDatasetEntry",
    "CsrMemmapShardEntry",
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
    "CategoricalLabelMap",
    "PertTFAdapterConfig",
    "PerturbationPairBatch",
    "PerturbationPairSampler",
    "PertTFCorpusAdapter",
    "PertTFLabelAdapter",
    "PertTFVocabAdapter",
    # Phase N — Corpus loader factory
    "Corpus",
    "load_corpus",
    # Utilities
    "read_raw_obs_parquet",
    "read_raw_var_parquet",
]
