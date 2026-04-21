"""Phase 3 corpus tokenizer: append-safe JSON-backed vocabulary.

This module implements the corpus-level tokenizer contract:
- Stored as JSON with the structure expected by ``SimpleVocab.from_json()``
  (stoi dict: token string → int ID; insertion order = token order).
- Special tokens ``<pad>``, ``<cls>``, ``<unk>``, ``<eos>`` with fixed IDs
  0–3, always appearing first in the stoi dict.
- Regular tokens in sorted ascending order after special tokens.
- Append-only: existing token IDs are never changed.
- Namespace compatibility check at append time.

JSON schema::

    {
        "corpus_id": "<string>",
        "contract_version": "0.2.0",
        "namespace": "<string>",
        "special_tokens": ["<pad>", "<cls>", "<unk>", "<eos>"],
        "stoi": {
            "<pad>": 0, "<cls>": 1, "<unk>": 2, "<eos>": 3,
            "<regular_token_1>": 4,
            ...
        }
    }

Compatible with ``SimpleVocab.from_json()`` from pertTF.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["CorpusTokenizer", "SPECIAL_TOKENS", "SPECIAL_TOKEN_IDS"]


# Fixed special token set and their reserved IDs (per Phase 1 Contract 2)
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<cls>", "<unk>", "<eos>")
SPECIAL_TOKEN_IDS: tuple[int, ...] = (0, 1, 2, 3)

# Current contract version
_TOKENIZER_CONTRACT_VERSION = "0.2.0"


class CorpusTokenizer:
    """Append-safe corpus tokenizer backed by JSON.

    Token IDs are assigned as follows:
    - Special tokens always receive the fixed IDs 0–3 in the order
      ``<pad>``, ``<cls>``, ``<unk>``, ``<eos>``.
    - New regular tokens receive the next available IDs in ascending order,
      appended after all existing tokens.

    The JSON file is compatible with ``SimpleVocab.from_json()`` from pertTF:
    the top-level ``stoi`` dict is written in insertion order (Python 3.7+),
    and ``SimpleVocab.__init__`` iterates over tokens in dict-order.

    Parameters
    ----------
    corpus_id : str
        Corpus identifier.
    namespace : str
        Feature namespace (e.g., ``"gene_symbol"``, ``"ensembl"``).
    stoi : dict[str, int]
        Token string → token ID mapping.  Insertion order must match the
        desired token order: special tokens first (in fixed order), then
        regular tokens in sorted ascending order.
    contract_version : str
        Contract version string.

    Attributes
    ----------
    corpus_id : str
    namespace : str
    stoi : dict[str, int]
        Direct token→ID mapping.
    itos : list[str]
        ID→token list (derived from stoi insertion order).
    max_id : int
        Highest assigned token ID.
    n_tokens : int
        Total number of tokens (special + regular).
    """

    __slots__ = (
        "corpus_id",
        "namespace",
        "contract_version",
        "_stoi",
        "_itos",
        "_max_id",
    )

    def __init__(
        self,
        corpus_id: str,
        namespace: str,
        stoi: dict[str, int],
        contract_version: str = _TOKENIZER_CONTRACT_VERSION,
    ) -> None:
        self.corpus_id = corpus_id
        self.namespace = namespace
        self.contract_version = contract_version
        self._stoi = stoi
        # Build itos from stoi insertion order
        self._itos = ["" for _ in range(len(stoi))]
        for token, tid in stoi.items():
            self._itos[tid] = token
        self._max_id = max(stoi.values()) if stoi else -1

    # -------------------------------------------------------------------------
    # Token queries
    # -------------------------------------------------------------------------

    def to_id(self, token: str) -> int:
        """Return the token ID for ``token``, or -1 if unknown."""
        return self._stoi.get(token, -1)

    def to_token(self, token_id: int) -> str:
        """Return the token string for ``token_id``."""
        if 0 <= token_id < len(self._itos):
            return self._itos[token_id]
        return "<unk>"

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __len__(self) -> int:
        return len(self._stoi)

    @property
    def n_tokens(self) -> int:
        """Total number of tokens (special + regular)."""
        return len(self._stoi)

    @property
    def max_id(self) -> int:
        """Highest assigned token ID."""
        return self._max_id

    @property
    def special_tokens(self) -> tuple[str, ...]:
        """Tuple of special token strings in fixed order."""
        return SPECIAL_TOKENS

    @property
    def regular_tokens(self) -> list[str]:
        """Regular (non-special) tokens in ID order."""
        return [t for t in self._itos if t not in SPECIAL_TOKENS]

    # -------------------------------------------------------------------------
    # JSON serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dict representation."""
        return {
            "corpus_id": self.corpus_id,
            "contract_version": self.contract_version,
            "namespace": self.namespace,
            "special_tokens": list(SPECIAL_TOKENS),
            "stoi": self._stoi,
        }

    def to_json(self, path: Path) -> None:
        """Write the tokenizer to a JSON file.

        Parameters
        ----------
        path : Path
            Output path.  The file is created with ``allow_pickle=False``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=False)
        # Verify round-trip (optional sanity check)
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if loaded["stoi"] != self._stoi:
            raise RuntimeError(
                "tokenizer JSON round-trip failed: stoi mismatch — "
                "ensure Python dict insertion order is preserved"
            )

    @classmethod
    def from_json(cls, path: Path) -> "CorpusTokenizer":
        """Load a tokenizer from a JSON file.

        Parameters
        ----------
        path : Path
            Path to the tokenizer JSON file.

        Returns
        -------
        CorpusTokenizer
            Populated tokenizer instance.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusTokenizer":
        """Reconstruct a tokenizer from a dict (inverse of ``to_dict``)."""
        return cls(
            corpus_id=str(data["corpus_id"]),
            namespace=str(data["namespace"]),
            stoi=dict(data["stoi"]),
            contract_version=str(data.get("contract_version", _TOKENIZER_CONTRACT_VERSION)),
        )

    # -------------------------------------------------------------------------
    # Factory: create new tokenizer (used by create_new route)
    # -------------------------------------------------------------------------

    @classmethod
    def create_new(
        cls,
        corpus_id: str,
        namespace: str,
        regular_tokens: list[str],
    ) -> "CorpusTokenizer":
        """Create a new tokenizer for a fresh corpus.

        Special tokens are assigned the fixed IDs 0–3.
        Regular tokens are sorted ascending and assigned IDs 4, 5, ….

        Parameters
        ----------
        corpus_id : str
            Corpus identifier.
        namespace : str
            Feature namespace.
        regular_tokens : list[str]
            List of regular token strings. Duplicates are removed;
            tokens are sorted before ID assignment.

        Returns
        -------
        CorpusTokenizer
        """
        # Remove duplicates and sort
        unique_tokens = sorted(set(regular_tokens))

        # Build stoi: special tokens first, then regular in sorted order
        stoi: dict[str, int] = {}
        for token, tid in zip(SPECIAL_TOKENS, SPECIAL_TOKEN_IDS):
            stoi[token] = tid

        next_id = len(SPECIAL_TOKENS)
        for token in unique_tokens:
            stoi[token] = next_id
            next_id += 1

        return cls(
            corpus_id=corpus_id,
            namespace=namespace,
            stoi=stoi,
        )

    # -------------------------------------------------------------------------
    # Append logic (used by append_routed route)
    # -------------------------------------------------------------------------

    def append_compatible(
        self,
        new_tokens: list[str],
        append_namespace: str,
    ) -> tuple[bool, str]:
        """Check append-time compatibility without modifying state.

        Two compatibility checks are applied in order:
        1. Namespace match: ``append_namespace`` must equal this tokenizer's
           namespace.
        2. No duplicate tokens: none of ``new_tokens`` may already exist
           in this tokenizer's stoi.

        Parameters
        ----------
        new_tokens : list[str]
            Token strings proposed for append.
        append_namespace : str
            Namespace of the appending dataset.

        Returns
        -------
        tuple[bool, str]
            (True, "") if compatible; (False, reason) if incompatible.
        """
        # Check 1: namespace match
        if append_namespace != self.namespace:
            return (
                False,
                f"namespace mismatch: appending dataset uses namespace "
                f"'{append_namespace}' but corpus tokenizer has namespace "
                f"'{self.namespace}'",
            )

        # Check 2: no duplicate tokens
        duplicate_tokens = [t for t in new_tokens if t in self._stoi]
        if duplicate_tokens:
            return (
                False,
                f"duplicate token labels in appending dataset: {duplicate_tokens[:5]}"
                f"{' ...' if len(duplicate_tokens) > 5 else ''} — "
                f"existing token IDs must be preserved and duplicates are not allowed",
            )

        return True, ""

    def append_tokens(
        self,
        new_tokens: list[str],
        append_namespace: str,
    ) -> "CorpusTokenizer":
        """Return a new tokenizer with ``new_tokens`` appended.

        Existing token IDs are unchanged.  New regular tokens are appended
        in sorted ascending order after the current max ID.

        Parameters
        ----------
        new_tokens : list[str]
            Token strings to append.
        append_namespace : str
            Namespace of the appending dataset (checked for compatibility).

        Returns
        -------
        CorpusTokenizer
            New tokenizer instance with tokens appended.

        Raises
        ------
        ValueError
            If namespace mismatch or duplicate tokens are detected.
        """
        compatible, reason = self.append_compatible(new_tokens, append_namespace)
        if not compatible:
            raise ValueError(reason)

        # Remove duplicates from new_tokens and sort
        unique_new = sorted(set(new_tokens))

        # Build new stoi: copy existing, then add new tokens
        new_stoi = dict(self._stoi)
        next_id = self._max_id + 1
        for token in unique_new:
            if token not in new_stoi:
                new_stoi[token] = next_id
                next_id += 1

        return CorpusTokenizer(
            corpus_id=self.corpus_id,
            namespace=self.namespace,
            stoi=new_stoi,
            contract_version=self.contract_version,
        )

    # -------------------------------------------------------------------------
    # Feature tokenization helpers
    # -------------------------------------------------------------------------

    def tokenize_labels(
        self,
        labels: list[str],
        on_unknown: str = "raise",
    ) -> list[int]:
        """Translate a list of feature label strings to token IDs.

        Parameters
        ----------
        labels : list[str]
            Feature label strings (e.g., gene symbols from schema resolution).
        on_unknown : str
            How to handle unknown tokens: ``"raise"`` raises ValueError,
            ``"skip"`` returns -1 for unknown tokens.

        Returns
        -------
        list[int]
            Token IDs in the same order as ``labels``.
        """
        results = []
        for label in labels:
            tid = self._stoi.get(label, -1)
            if tid == -1:
                if on_unknown == "raise":
                    raise ValueError(f"token '{label}' not found in corpus tokenizer")
                # on_unknown == "skip"
            results.append(tid)
        return results
