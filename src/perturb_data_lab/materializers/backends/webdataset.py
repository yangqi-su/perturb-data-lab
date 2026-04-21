"""Backend adapter: WebDataset shards for sequential streaming."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from ..models import OutputRoots


def write_webdataset_shards(
    adata: ad.AnnData,
    count_matrix: any,
    size_factors: np.ndarray,
    release_id: str,
    matrix_root: Path,
    shard_size: int = 10000,
    canonical_perturbation: tuple[dict[str, str], ...] | None = None,
    canonical_context: tuple[dict[str, str], ...] | None = None,
    raw_fields: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Path]:
    """Write sparse per-cell data in WebDataset shard format.

    Each shard is a .tar file containing per-cell .pt (pickle) files:
    - cell_<index>.pt: dict with expressed_gene_indices, expression_counts,
      size_factor, canonical_perturbation, canonical_context, raw_fields

    WebDataset is optimal for sequential streaming in PyTorch training loops.

    Canonical metadata (canonical_perturbation, canonical_context, raw_fields)
    is written to per-cell .pt records so WebDataset reader can return full
    CellState parity with Arrow/HF backend.

    Returns a dict with keys: "shard_paths", "meta".
    """
    matrix_root.mkdir(parents=True, exist_ok=True)

    n_obs = adata.n_obs
    n_shards = (n_obs + shard_size - 1) // shard_size

    import pickle

    shard_paths = []
    meta_rows = []

    # Normalize metadata tuples to-safe form
    pert_tuple = canonical_perturbation or tuple([{}] * n_obs)
    ctx_tuple = canonical_context or tuple([{}] * n_obs)
    raw_tuple = raw_fields or tuple([{}] * n_obs)

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, n_obs)
        shard_path = matrix_root / f"{release_id}-{shard_idx:05d}.tar"
        shard_paths.append(shard_path)

        import tarfile

        with tarfile.open(str(shard_path), "w") as tar:
            for i in range(start, end):
                if issparse(count_matrix):
                    row = count_matrix[i]
                    if hasattr(row, "toarray"):
                        row = row.toarray().ravel()
                    else:
                        row = np.asarray(row).ravel()
                else:
                    row = np.asarray(count_matrix[i]).ravel()

                nonzero_mask = row != 0
                indices = np.where(nonzero_mask)[0].astype(np.int32)
                counts = row[nonzero_mask].astype(np.int32)

                cell_record = {
                    "expressed_gene_indices": indices.tobytes(),
                    "expression_counts": counts.tobytes(),
                    "size_factor": float(size_factors[i]),
                    "cell_id": str(adata.obs.index[i]),
                    "canonical_perturbation": dict(pert_tuple[i]),
                    "canonical_context": dict(ctx_tuple[i]),
                    "raw_fields": dict(raw_tuple[i]),
                }

                import io

                data_bytes = pickle.dumps(cell_record)
                info = tarfile.TarInfo(name=f"cell_{i:08d}.pt")
                info.size = len(data_bytes)
                tar.addfile(info, io.BytesIO(data_bytes))

                meta_rows.append(
                    {
                        "cell_id": str(adata.obs.index[i]),
                        "shard": shard_idx,
                        "size_factor": float(size_factors[i]),
                    }
                )

    # Write meta as simple newline-delimited text
    meta_path = matrix_root / f"{release_id}-meta.txt"
    with open(meta_path, "w") as f:
        for row in meta_rows:
            f.write(f"{row['cell_id']}\t{row['shard']}\t{row['size_factor']}\n")

    return {"shard_paths": shard_paths, "meta": meta_path}
