"""Canonicalization layer for perturb-data-lab.

Provides:
- ``CanonicalObsSchema`` / ``CanonicalVarSchema`` — immutable contracts
- ``CanonicalizationSchema`` — per-dataset YAML-driven transform config
- ``CanonicalVocab`` — corpus-global vocabulary container
- ``CanonicalizationRunner`` — transforms raw sidecars into canonical parquets
- ``build_canonical_vocab`` — merges per-dataset vocabs into corpus-level vocab
"""

from .contract import (
    CANONICAL_OBS_MUST_HAVE,
    CANONICAL_VAR_MUST_HAVE,
    CanonicalObsSchema,
    CanonicalVarSchema,
    CanonicalVocab,
    CanonicalizationSchema,
    ObsColumnMapping,
    VarColumnMapping,
    ExtensibleColumn,
    TransformRule,
    GeneMappingConfig,
)
from .runner import (
    CanonicalizationResult,
    CanonicalizationRunner,
    build_canonical_vocab,
    run_canonicalization,
)

__all__ = [
    "CANONICAL_OBS_MUST_HAVE",
    "CANONICAL_VAR_MUST_HAVE",
    "CanonicalObsSchema",
    "CanonicalVarSchema",
    "CanonicalVocab",
    "CanonicalizationSchema",
    "CanonicalizationResult",
    "CanonicalizationRunner",
    "ExtensibleColumn",
    "GeneMappingConfig",
    "ObsColumnMapping",
    "TransformRule",
    "VarColumnMapping",
    "build_canonical_vocab",
    "run_canonicalization",
]
