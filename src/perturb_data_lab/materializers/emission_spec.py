"""Phase 4 corpus emission spec: defines what fields loaders emit at runtime.

The emission spec is corpus-level (one per corpus) and controls which
canonical perturbation and context fields are included in the CellState
yielded by loaders.  The spec is written during corpus creation and
read at loader initialization time.

Artifact: ``corpus-emission-spec.yaml`` at corpus root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL


@dataclass(frozen=True)
class CorpusEmissionSpec:
    """Corpus-level emission spec controlling which fields loaders emit at runtime.

    The spec is written as ``corpus-emission-spec.yaml`` next to
    ``corpus-index.yaml`` and ``tokenizer.json`` during corpus creation.

    At runtime, a loader reads this spec and includes only the listed
    fields in the ``CellState`` it returns.  Fields not listed in the
    spec remain available in the SQLite backing store but are not
    emitted.

    The ``hvg_sidecar_path`` field points to the directory containing
    per-dataset ``hvg.npy`` and ``nonhvg.npy`` artifacts (written during
    materialization).  Loaders read these files to construct HVG token
    ID sets for ``HVGRandomSampler``.
    """

    kind: str = "corpus-emission-spec"
    contract_version: str = CONTRACT_VERSION
    corpus_id: str = ""
    perturbation_fields: tuple[str, ...] = field(
        default_factory=lambda: (
            "perturbation_label",
            "perturbation_type",
            "target_id",
            "control_flag",
            "dose",
            "combination_key",
        )
    )
    context_fields: tuple[str, ...] = field(
        default_factory=lambda: (
            "dataset_id",
            "cell_context",
            "cell_line_or_type",
            "tissue",
            "assay",
            "condition",
        )
    )
    output_convention: str = "dict"  # only "dict" for now
    hvg_sidecar_path: str | None = None  # relative path from corpus root

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "contract_version": self.contract_version,
            "corpus_id": self.corpus_id,
            "perturbation_fields": list(self.perturbation_fields),
            "context_fields": list(self.context_fields),
            "output_convention": self.output_convention,
            "hvg_sidecar_path": self.hvg_sidecar_path,
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def write_yaml(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "CorpusEmissionSpec":
        return cls(
            kind=str(data.get("kind", "corpus-emission-spec")),
            contract_version=str(data.get("contract_version", CONTRACT_VERSION)),
            corpus_id=str(data.get("corpus_id", "")),
            perturbation_fields=tuple(str(f) for f in data.get("perturbation_fields", [])),
            context_fields=tuple(str(f) for f in data.get("context_fields", [])),
            output_convention=str(data.get("output_convention", "dict")),
            hvg_sidecar_path=data.get("hvg_sidecar_path"),
        )

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "CorpusEmissionSpec":
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected a YAML mapping in {file_path}")
        return cls.from_dict(payload)

    def emitted_perturbation_fields(self) -> tuple[str, ...]:
        """Fields to emit from canonical_perturbation into CellState."""
        return self.perturbation_fields

    def emitted_context_fields(self) -> tuple[str, ...]:
        """Fields to emit from canonical_context into CellState."""
        return self.context_fields