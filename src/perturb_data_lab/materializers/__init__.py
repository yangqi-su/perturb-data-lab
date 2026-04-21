"""Phase 3 materializer: canonical materialization layer, manifests, join modes."""

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
    # Tokenizer
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
