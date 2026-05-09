"""Backend adapter: HuggingFace datasets federated writer.

Produces one saved ``datasets.Dataset`` directory per dataset at:

- ``{dataset_id}-hf-dataset/``

The saved dataset preserves the flat expression contract columns:

- ``global_row_index``
- ``expressed_gene_indices``
- ``expression_counts``
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..chunk_translation import ChunkBundle


def _import_hf_datasets():
    try:
        from datasets import Dataset, Features, Sequence, Value
        from datasets.arrow_writer import ArrowWriter
    except ImportError as exc:
        raise ImportError(
            "hf-datasets backend requires the optional 'datasets' package; "
            "install perturb-data-lab[hf-datasets] or pip install datasets"
        ) from exc
    return Dataset, Features, Sequence, Value, ArrowWriter


def _hf_dataset_features():
    _, Features, Sequence, Value, _ = _import_hf_datasets()
    return Features(
        {
            "global_row_index": Value("int64"),
            "expressed_gene_indices": Sequence(Value("int32")),
            "expression_counts": Sequence(Value("int32")),
        }
    )


def write_hf_datasets_federated(
    bundle: ChunkBundle,
    dataset_id: str,
    matrix_root: Path,
    *,
    _writer_state: dict[str, Any] | None = None,
    _is_last_chunk: bool = False,
) -> tuple[dict[str, Path], dict | None]:
    """Stream ChunkBundle tables into a saved HuggingFace dataset directory."""
    Dataset, _, _, _, ArrowWriter = _import_hf_datasets()

    matrix_root.mkdir(parents=True, exist_ok=True)
    dataset_path = matrix_root / f"{dataset_id}-hf-dataset"
    staging_path = matrix_root / f"{dataset_id}-hf-dataset.arrow"

    if _writer_state is None:
        if staging_path.exists():
            staging_path.unlink()
        writer = ArrowWriter(
            path=str(staging_path),
            features=_hf_dataset_features(),
        )
        _writer_state = {
            "writer": writer,
            "dataset_path": dataset_path,
            "staging_path": staging_path,
        }
    else:
        writer = _writer_state["writer"]

    writer.write_table(bundle.table)

    if not _is_last_chunk:
        return {"cells": dataset_path}, _writer_state

    writer.finalize()
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    dataset = Dataset.from_file(str(staging_path))
    dataset.save_to_disk(str(dataset_path), num_shards=1)
    staging_path.unlink()
    return {"cells": dataset_path}, None
