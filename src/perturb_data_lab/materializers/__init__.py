"""Phase 3 materializer: canonical materialization layer, manifests, join modes.

Tokenization is removed from the materialization flow. The corpus feature set
is maintained separately via a pickle-backed set written by ``canonicalize-meta``.

This module exposes the public materializer API lazily so lightweight runtime
imports (for example loader-only smoke validation) do not eagerly import heavy
or environment-sensitive modules such as ``sqlite3`` until they are actually
needed.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MAP = {
    # Core classes
    "CanonicalCellRecord": (".core", "CanonicalCellRecord"),
    "CreateNewRoute": (".core", "CreateNewRoute"),
    "AppendRoutedRoute": (".core", "AppendRoutedRoute"),
    "build_materialization_route": (".core", "build_materialization_route"),
    "update_corpus_index": (".core", "update_corpus_index"),
    # Tokenizer (available but NOT used during materialization)
    "CorpusTokenizer": (".tokenizer", "CorpusTokenizer"),
    # Emission spec
    "CorpusEmissionSpec": (".emission_spec", "CorpusEmissionSpec"),
    # Models
    "MaterializationManifest": (".models", "MaterializationManifest"),
    "FeatureRegistryManifest": (".models", "FeatureRegistryManifest"),
    "SizeFactorManifest": (".models", "SizeFactorManifest"),
    "QAManifest": (".models", "QAManifest"),
    "CorpusIndexDocument": (".models", "CorpusIndexDocument"),
    "GlobalMetadataDocument": (".models", "GlobalMetadataDocument"),
    "CellMetadataRecord": (".models", "CellMetadataRecord"),
    "FeatureRegistryEntry": (".models", "FeatureRegistryEntry"),
    "CountSourceSpec": (".models", "CountSourceSpec"),
    "OutputRoots": (".models", "OutputRoots"),
    "ProvenanceSpec": (".models", "ProvenanceSpec"),
    "SizeFactorEntry": (".models", "SizeFactorEntry"),
    # Phase 3 new artifacts
    "RawCellMetadataRecord": (".models", "RawCellMetadataRecord"),
    "RawFeatureMetadataRecord": (".models", "RawFeatureMetadataRecord"),
    "DatasetMetadataSummary": (".models", "DatasetMetadataSummary"),
    "FeatureProvenanceSpec": (".models", "FeatureProvenanceSpec"),
    # Phase 4 canonicalize-meta
    "CanonicalizeMetaRoute": (".canonicalize_meta", "CanonicalizeMetaRoute"),
    "CorpusCellIndexRange": (".canonicalize_meta", "CorpusCellIndexRange"),
    "CanonicalizeMetaResult": (".canonicalize_meta", "CanonicalizeMetaResult"),
    "run_canonicalize_meta": (".canonicalize_meta", "run_canonicalize_meta"),
    # Validation
    "validate_schema_readiness": (".validation", "validate_schema_readiness"),
    # Schema execution
    "SchemaExecutionResult": (".schema_execution", "SchemaExecutionResult"),
    "resolve_field_entry": (".schema_execution", "resolve_field_entry"),
    "resolve_cell_row": (".schema_execution", "resolve_cell_row"),
    "resolve_feature_row": (".schema_execution", "resolve_feature_row"),
    "resolve_all_cell_rows": (".schema_execution", "resolve_all_cell_rows"),
    "resolve_all_feature_rows": (".schema_execution", "resolve_all_feature_rows"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORT_MAP.keys()))

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
