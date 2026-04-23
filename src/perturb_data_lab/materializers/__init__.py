"""Phase 3 materializer: canonical materialization layer, manifests, join modes.

Tokenization is removed from the materialization flow. The corpus feature set
is maintained separately via a pickle-backed set written by ``canonicalize-meta``.
"""

from .core import (
    CanonicalCellRecord,
    CreateNewRoute,
    AppendRoutedRoute,
    build_materialization_route,
    update_corpus_index,
)
from .models import (
    MaterializationManifest,
    FeatureRegistryManifest,
    SizeFactorManifest,
    QAManifest,
    CorpusIndexDocument,
    GlobalMetadataDocument,
    CellMetadataRecord,
    FeatureRegistryEntry,
    CountSourceSpec,
    OutputRoots,
    ProvenanceSpec,
    SizeFactorEntry,
    # Phase 3 new artifacts
    RawCellMetadataRecord,
    RawFeatureMetadataRecord,
    DatasetMetadataSummary,
    FeatureProvenanceSpec,
)
from .canonicalize_meta import (
    CanonicalizeMetaRoute,
    CorpusCellIndexRange,
    CanonicalizeMetaResult,
    run_canonicalize_meta,
)
from .tokenizer import CorpusTokenizer
from .emission_spec import CorpusEmissionSpec
from .validation import validate_schema_readiness
from .schema_execution import (
    SchemaExecutionResult,
    resolve_field_entry,
    resolve_cell_row,
    resolve_feature_row,
    resolve_all_cell_rows,
    resolve_all_feature_rows,
)

__all__ = [
    # Core classes
    "CanonicalCellRecord",
    "CreateNewRoute",
    "AppendRoutedRoute",
    "build_materialization_route",
    "update_corpus_index",
    # Tokenizer (available but NOT used during materialization)
    "CorpusTokenizer",
    # Emission spec
    "CorpusEmissionSpec",
    # Models
    "MaterializationManifest",
    "FeatureRegistryManifest",
    "SizeFactorManifest",
    "QAManifest",
    "CorpusIndexDocument",
    "GlobalMetadataDocument",
    "CellMetadataRecord",
    "FeatureRegistryEntry",
    "CountSourceSpec",
    "OutputRoots",
    "ProvenanceSpec",
    "SizeFactorEntry",
    # Phase 3 new artifacts
    "RawCellMetadataRecord",
    "RawFeatureMetadataRecord",
    "DatasetMetadataSummary",
    "FeatureProvenanceSpec",
    # Phase 4 canonicalize-meta
    "CanonicalizeMetaRoute",
    "CorpusCellIndexRange",
    "CanonicalizeMetaResult",
    "run_canonicalize_meta",
    # Validation
    "validate_schema_readiness",
    # Schema execution
    "SchemaExecutionResult",
    "resolve_field_entry",
    "resolve_cell_row",
    "resolve_feature_row",
    "resolve_all_cell_rows",
    "resolve_all_feature_rows",
]
