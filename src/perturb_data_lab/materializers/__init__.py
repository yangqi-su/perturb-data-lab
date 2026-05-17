"""Public materialization API.

Exports are loaded lazily so loader-only imports do not eagerly import heavier
materialization dependencies until they are needed.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MAP = {
    "DatasetMaterializer": (".core", "DatasetMaterializer"),
    "update_corpus_index": (".corpus_index", "update_corpus_index"),
    # Models
    "MaterializationManifest": (".models", "MaterializationManifest"),
    "CorpusIndexDocument": (".models", "CorpusIndexDocument"),
    "GlobalMetadataDocument": (".models", "GlobalMetadataDocument"),
    "CountSourceSpec": (".models", "CountSourceSpec"),
    "OutputRoots": (".models", "OutputRoots"),
    "ProvenanceSpec": (".models", "ProvenanceSpec"),
    "CorpusRegistrationInfo": (".models", "CorpusRegistrationInfo"),
    "register_materialization": (".registration", "register_materialization"),
    "corpus_exists": (".registration", "corpus_exists"),
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
    "DatasetMaterializer",
    "update_corpus_index",
    # Models
    "MaterializationManifest",
    "CorpusIndexDocument",
    "GlobalMetadataDocument",
    "CountSourceSpec",
    "OutputRoots",
    "ProvenanceSpec",
    "CorpusRegistrationInfo",
    "register_materialization",
    "corpus_exists",
    "manifest_to_join_record",
]
