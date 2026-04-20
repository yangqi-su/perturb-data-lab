"""Phase 3 materializer: canonical materialization layer, manifests, join modes."""

from .core import (
    CanonicalCellRecord,
    CreateNewRoute,
    AppendMonolithicRoute,
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

__all__ = [
    # Core classes
    "CanonicalCellRecord",
    "CreateNewRoute",
    "AppendMonolithicRoute",
    "AppendRoutedRoute",
    "build_materialization_route",
    "update_corpus_index",
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
]
