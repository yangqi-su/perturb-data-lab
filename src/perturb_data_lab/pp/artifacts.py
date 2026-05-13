"""Small output/provenance helpers for streamed ``pp`` artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..loaders.corpus_loader import Corpus


@dataclass(frozen=True)
class PpArtifactSpec:
    """Resolved output/provenance paths for one dataset-local pp artifact."""

    output_dir: Path
    dataset_id: str
    artifact_name: str
    artifact_path: Path
    provenance_path: Path


def prepare_pp_output(
    output_dir: str | Path,
    *,
    dataset_id: str,
    artifact_name: str,
    suffix: str,
) -> PpArtifactSpec:
    """Create a dataset-local output location inside a caller-specified root."""
    if not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")
    if not artifact_name:
        raise ValueError("artifact_name must be a non-empty string")
    if not suffix:
        raise ValueError("suffix must be a non-empty string")

    resolved_output_dir = Path(output_dir)
    dataset_dir = resolved_output_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    artifact_path = dataset_dir / f"{artifact_name}{normalized_suffix}"
    provenance_path = dataset_dir / f"{artifact_name}.provenance.json"
    return PpArtifactSpec(
        output_dir=resolved_output_dir,
        dataset_id=str(dataset_id),
        artifact_name=str(artifact_name),
        artifact_path=artifact_path,
        provenance_path=provenance_path,
    )


def build_pp_provenance(
    corpus: Corpus,
    *,
    operation: str,
    dataset_id: str,
    parameters: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a small JSON-serializable provenance payload for pp outputs."""
    if dataset_id not in corpus.dataset_index_by_id:
        raise KeyError(f"Unknown dataset_id for pp provenance: {dataset_id!r}")
    if not operation:
        raise ValueError("operation must be a non-empty string")

    return {
        "kind": "pp-provenance",
        "operation": str(operation),
        "dataset_id": str(dataset_id),
        "dataset_index": int(corpus.dataset_index_by_id[dataset_id]),
        "backend": str(corpus.backend),
        "topology": str(corpus.topology),
        "corpus_root": str(corpus.corpus_root),
        "parameters": _json_safe(parameters or {}),
        "extra": _json_safe(extra or {}),
    }


def write_pp_provenance(
    spec: PpArtifactSpec,
    *,
    corpus: Corpus,
    operation: str,
    parameters: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a provenance sidecar next to a planned pp artifact."""
    payload = build_pp_provenance(
        corpus,
        operation=operation,
        dataset_id=spec.dataset_id,
        parameters=parameters,
        extra={
            "artifact_name": spec.artifact_name,
            "artifact_path": str(spec.artifact_path),
            **dict(extra or {}),
        },
    )
    spec.provenance_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return spec.provenance_path


def _json_safe(value: Any) -> Any:
    """Convert common numpy/path objects into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
