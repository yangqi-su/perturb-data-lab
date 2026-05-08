"""Public composable corpus-loader API exports."""

from .expression import (
    AggregateLanceReader,
    AggregateZarrReader,
    AggregateCsrMemmapReader,
    ArrowIpcDatasetEntry,
    BaseExpressionReader,
    CsrMemmapShardEntry,
    DatasetEntry,
    ExpressionReader,
    FederatedArrowIpcReader,
    FederatedLanceReader,
    FederatedParquetReader,
    FederatedWebDatasetReader,
    FederatedZarrReader,
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
from .corpus_loader import (
    Corpus,
    load_corpus,
)
from .gpu_pipeline import (
    GPUSparsePipeline,
    CPUPipeline,
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
    "FederatedLanceReader",
    "AggregateZarrReader",
    "FederatedZarrReader",
    "AggregateCsrMemmapReader",
    "FederatedArrowIpcReader",
    "FederatedParquetReader",
    "FederatedWebDatasetReader",
    "LanceDatasetEntry",
    "ZarrDatasetEntry",
    "ArrowIpcDatasetEntry",
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
    # Phase 3 — GPU Pipeline
    "GPUSparsePipeline",
    "CPUPipeline",
    # Phase N — Corpus loader factory
    "Corpus",
    "load_corpus",
    # Utilities
    "read_raw_obs_parquet",
    "read_raw_var_parquet",
]
