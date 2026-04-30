"""Phase 3 refactored loaders package."""

from .executor import BatchExecutor
from .expression import (
    AggregateLanceReader,
    AggregateZarrReader,
    ArrowIpcDatasetEntry,
    BaseExpressionReader,
    DatasetEntry,
    ExpressionReader,
    ExpressionRow,
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
    CellIdentity,
    CellState,
    CPUDenseRuntimePath,
    CorpusRandomBatchSampler,
    DatasetBatchSampler,
    DatasetContextBatchSampler,
    DatasetContextKey,
    ExpressedZerosSampler,
    ExpressionBatch,
    GlobalFeatureResolver,
    GPUSparseRuntimePath,
    HVGRandomSampler,
    PerturbBatchDataset,
    PerturbDataLoader,
    PerturbIterableDataset,
    RandomContextSampler,
    ResolvedSparseBatch,
    SamplerState,
    SparseBatchCollator,
    SparseBatchPayload,
    collate_batch_dict,
    cpu_parallel_collate_fn,
)
from .corpus import (
    read_raw_obs_parquet,
    read_raw_var_parquet,
)
from .feature_registry import (
    FeatureRegistry,
    GlobalGeneSampler,
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
    "ExpressionRow",
    "ExpressionReader",
    "BaseExpressionReader",
    "DatasetEntry",
    "AggregateLanceReader",
    "FederatedLanceReader",
    "AggregateZarrReader",
    "FederatedZarrReader",
    "FederatedArrowIpcReader",
    "FederatedParquetReader",
    "FederatedWebDatasetReader",
    "LanceDatasetEntry",
    "ZarrDatasetEntry",
    "ArrowIpcDatasetEntry",
    "ParquetDatasetEntry",
    "WebDatasetEntry",
    "build_expression_reader",
    # Phase 3 — BatchExecutor
    "BatchExecutor",
    # Phase 3 — Core types
    "CellState",
    "CellIdentity",
    "ExpressionBatch",
    "DatasetContextKey",
    # Phase 3 — Collation
    "SparseBatchPayload",
    "ResolvedSparseBatch",
    "SparseBatchCollator",
    "GlobalFeatureResolver",
    # Phase 3 — Samplers (MetadataIndex-backed)
    "CorpusRandomBatchSampler",
    "DatasetBatchSampler",
    "DatasetContextBatchSampler",
    # Phase 3 — Runtime paths (CPUDenseRuntimePath / GPUSparseRuntimePath
    # are deprecated — import from .loaders directly if still needed)
    "SamplerState",
    # Phase 3 — Sampler modes
    "RandomContextSampler",
    "ExpressedZerosSampler",
    "HVGRandomSampler",
    # Phase 3 — Data loaders
    "PerturbDataLoader",
    "PerturbIterableDataset",
    "PerturbBatchDataset",
    "collate_batch_dict",
    "cpu_parallel_collate_fn",
    # Phase 2 — Feature Registry
    "FeatureRegistry",
    "GlobalGeneSampler",
    # Phase 3 — GPU Pipeline
    "GPUSparsePipeline",
    "CPUPipeline",
    # Utilities
    "read_raw_obs_parquet",
    "read_raw_var_parquet",
]
