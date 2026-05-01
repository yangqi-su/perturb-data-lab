"""Stage 2 materializer — schema-independent, Stage-1-gated, count-first.

Phase 3 canonical materializer (expression-first, tokenizer-free) is preserved
as the legacy schema-first path. Stage 2 adds:

- ``Stage2Materializer``: schema-independent materialization entry that accepts
  a Stage 1 ``dataset-summary.yaml`` as the only gating artifact (no schema.yaml)
- Count-first path driven by the Stage 1 approved count source decision
- Parquet raw metadata sidecars (SQLite deprecated for new artifacts)
- Backend/topology separation in all interfaces

This module exposes the public materializer API lazily so lightweight runtime
imports (for example loader-only smoke validation) do not eagerly import heavy
or environment-sensitive modules such as ``sqlite3`` until they are actually
needed.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MAP = {
    # Stage 2 — schema-independent entry point
    "Stage2Materializer": (".core", "Stage2Materializer"),
    # Core classes
    "CanonicalCellRecord": (".core", "CanonicalCellRecord"),
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
    # Phase 3 corpus ledger
    "CorpusLedgerEntry": (".models", "CorpusLedgerEntry"),
    # Phase 4 corpus registration
    "CorpusRegistrationInfo": (".models", "CorpusRegistrationInfo"),
    "register_materialization": (".registration", "register_materialization"),
    "read_corpus_ledger": (".registration", "read_corpus_ledger"),
    "corpus_exists": (".registration", "corpus_exists"),
    "get_corpus_summary": (".registration", "get_corpus_summary"),
    "manifest_to_join_record": (".registration", "manifest_to_join_record"),
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
    # Stage 2 — schema-independent entry point
    "Stage2Materializer",
    # Core classes
    "CanonicalCellRecord",
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
    # Phase 3 corpus ledger
    "CorpusLedgerEntry",
    # Phase 4 corpus registration
    "CorpusRegistrationInfo",
    "register_materialization",
    "read_corpus_ledger",
    "corpus_exists",
    "get_corpus_summary",
    "manifest_to_join_record",
]
