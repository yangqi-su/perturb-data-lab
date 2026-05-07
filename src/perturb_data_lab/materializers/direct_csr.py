"""Direct h5ad-to-CSR conversion helpers for the aggregate CSR backend."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import yaml

from .backends.csr_memmap import CsrMemmapWriter


_DATASET_MANIFEST_FILE = "direct-csr-dataset-manifest.yaml"
_CORPUS_MANIFEST_FILE = "csr-corpus-manifest.yaml"
_CONTRACT_VERSION = "0.1.0"


@dataclass(frozen=True)
class H5adCsrChunk:
    """One row-contiguous chunk read directly from a CSR-encoded h5ad matrix."""

    row_start: int
    row_stop: int
    indptr: np.ndarray
    indices: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class DirectCsrDatasetManifest:
    """Per-dataset manifest emitted by the direct CSR converter."""

    dataset_id: str
    dataset_index: int
    source_h5ad_path: str
    matrix_path: str
    output_dir: str
    csr_manifest_path: str
    global_start: int
    global_end: int
    n_cells: int
    n_features: int
    total_nnz: int
    shard_n_cells_target: int
    shard_count: int
    source_encoding_type: str
    source_counts_dtype: str
    source_indices_dtype: str
    source_indptr_dtype: str

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "direct-csr-dataset-manifest",
            "contract_version": _CONTRACT_VERSION,
            "dataset_id": self.dataset_id,
            "dataset_index": self.dataset_index,
            "source_h5ad_path": self.source_h5ad_path,
            "matrix_path": self.matrix_path,
            "output_dir": self.output_dir,
            "csr_manifest_path": self.csr_manifest_path,
            "global_start": self.global_start,
            "global_end": self.global_end,
            "n_cells": self.n_cells,
            "n_features": self.n_features,
            "total_nnz": self.total_nnz,
            "shard_n_cells_target": self.shard_n_cells_target,
            "shard_count": self.shard_count,
            "source_encoding_type": self.source_encoding_type,
            "source_counts_dtype": self.source_counts_dtype,
            "source_indices_dtype": self.source_indices_dtype,
            "source_indptr_dtype": self.source_indptr_dtype,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DirectCsrDatasetManifest":
        with open(path, "r") as fh:
            doc = yaml.safe_load(fh) or {}
        if doc.get("kind") != "direct-csr-dataset-manifest":
            raise ValueError(
                f"Expected direct-csr-dataset-manifest, got {doc.get('kind')!r}"
            )
        return cls(
            dataset_id=str(doc["dataset_id"]),
            dataset_index=int(doc.get("dataset_index", 0)),
            source_h5ad_path=str(doc["source_h5ad_path"]),
            matrix_path=str(doc.get("matrix_path", "X")),
            output_dir=str(doc["output_dir"]),
            csr_manifest_path=str(doc["csr_manifest_path"]),
            global_start=int(doc["global_start"]),
            global_end=int(doc["global_end"]),
            n_cells=int(doc["n_cells"]),
            n_features=int(doc["n_features"]),
            total_nnz=int(doc["total_nnz"]),
            shard_n_cells_target=int(doc["shard_n_cells_target"]),
            shard_count=int(doc["shard_count"]),
            source_encoding_type=str(doc["source_encoding_type"]),
            source_counts_dtype=str(doc["source_counts_dtype"]),
            source_indices_dtype=str(doc["source_indices_dtype"]),
            source_indptr_dtype=str(doc["source_indptr_dtype"]),
        )

    def write_yaml(self, path: str | Path) -> Path:
        out = Path(path)
        with open(out, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)
        return out


class DirectH5adCsrReader:
    """Stream row chunks directly from a CSR-encoded h5ad sparse matrix."""

    def __init__(self, source_path: str | Path, matrix_path: str = "X"):
        self.source_path = Path(source_path)
        self.matrix_path = matrix_path

        with h5py.File(self.source_path, "r") as handle:
            group = self._open_csr_group(handle, self.matrix_path)
            self.encoding_type = _decode_h5_attr(group.attrs.get("encoding-type"))
            self.encoding_version = _decode_h5_attr(group.attrs.get("encoding-version"))
            shape = group.attrs.get("shape")
            if shape is None or len(shape) != 2:
                raise ValueError(
                    f"CSR group {self.matrix_path!r} is missing a valid shape attribute"
                )
            self.n_rows = int(shape[0])
            self.n_cols = int(shape[1])
            self.counts_dtype = str(group["data"].dtype)
            self.indices_dtype = str(group["indices"].dtype)
            self.indptr_dtype = str(group["indptr"].dtype)

    def iter_chunks(self, rows_per_chunk: int) -> Iterable[H5adCsrChunk]:
        if rows_per_chunk <= 0:
            raise ValueError(
                f"rows_per_chunk must be positive, got {rows_per_chunk}"
            )

        with h5py.File(self.source_path, "r") as handle:
            group = self._open_csr_group(handle, self.matrix_path)
            data_ds = group["data"]
            indices_ds = group["indices"]
            indptr_ds = group["indptr"]

            for row_start in range(0, self.n_rows, rows_per_chunk):
                row_stop = min(self.n_rows, row_start + rows_per_chunk)
                data_start = int(indptr_ds[row_start])
                data_stop = int(indptr_ds[row_stop])
                indptr = np.asarray(
                    indptr_ds[row_start : row_stop + 1], dtype=np.int64
                )
                indptr = indptr - data_start
                yield H5adCsrChunk(
                    row_start=row_start,
                    row_stop=row_stop,
                    indptr=indptr,
                    indices=np.asarray(indices_ds[data_start:data_stop]),
                    counts=np.asarray(data_ds[data_start:data_stop]),
                )

    @staticmethod
    def _open_csr_group(handle: h5py.File, matrix_path: str) -> h5py.Group:
        if matrix_path not in handle:
            raise KeyError(f"Matrix path {matrix_path!r} not found in {handle.filename}")
        node = handle[matrix_path]
        if not isinstance(node, h5py.Group):
            raise NotImplementedError(
                f"Matrix path {matrix_path!r} is not an HDF5 group-backed sparse matrix"
            )
        encoding_type = _decode_h5_attr(node.attrs.get("encoding-type"))
        if encoding_type != "csr_matrix":
            raise NotImplementedError(
                f"Unsupported h5ad sparse encoding {encoding_type!r}; only csr_matrix is supported"
            )
        for key in ("data", "indices", "indptr"):
            if key not in node:
                raise ValueError(
                    f"CSR group {matrix_path!r} is missing required member {key!r}"
                )
        return node


def convert_h5ad_to_csr_dataset(
    *,
    dataset_id: str,
    source_h5ad_path: str | Path,
    output_dir: str | Path,
    global_row_start: int,
    dataset_index: int = 0,
    shard_n_cells: int = 25_000,
    rows_per_chunk: int = 25_000,
    matrix_path: str = "X",
) -> Path:
    """Convert one CSR-encoded h5ad dataset directly into CSR memmap shards."""
    source_path = Path(source_h5ad_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = DirectH5adCsrReader(source_path, matrix_path=matrix_path)
    writer = CsrMemmapWriter(
        out_dir,
        shard_n_cells=shard_n_cells,
        source_corpus_root=source_path,
        global_row_start=global_row_start,
    )

    for chunk in reader.iter_chunks(rows_per_chunk=rows_per_chunk):
        writer.append_csr_chunk(
            global_row_start=global_row_start + chunk.row_start,
            indptr=np.asarray(chunk.indptr, dtype=np.int64),
            indices=_coerce_indices_int32(chunk.indices),
            counts=_coerce_counts_int32(chunk.counts),
        )

    csr_manifest_path = writer.finalize()
    dataset_manifest = DirectCsrDatasetManifest(
        dataset_id=dataset_id,
        dataset_index=dataset_index,
        source_h5ad_path=str(source_path),
        matrix_path=matrix_path,
        output_dir=str(out_dir),
        csr_manifest_path=str(csr_manifest_path),
        global_start=global_row_start,
        global_end=global_row_start + writer.total_cells_written,
        n_cells=writer.total_cells_written,
        n_features=reader.n_cols,
        total_nnz=writer.total_nnz_written,
        shard_n_cells_target=shard_n_cells,
        shard_count=writer.n_shards,
        source_encoding_type=reader.encoding_type,
        source_counts_dtype=reader.counts_dtype,
        source_indices_dtype=reader.indices_dtype,
        source_indptr_dtype=reader.indptr_dtype,
    )
    return dataset_manifest.write_yaml(out_dir / _DATASET_MANIFEST_FILE)


def merge_csr_dataset_manifests(
    dataset_manifest_paths: Sequence[str | Path],
    output_root: str | Path,
    *,
    shard_subdir: str = "shards",
) -> Path:
    """Merge per-dataset direct CSR manifests into one aggregate CSR corpus manifest."""
    manifests = [DirectCsrDatasetManifest.from_yaml(path) for path in dataset_manifest_paths]
    if not manifests:
        raise ValueError("dataset_manifest_paths must not be empty")

    manifests.sort(key=lambda m: m.global_start)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_target_root = output_dir / shard_subdir
    shard_target_root.mkdir(parents=True, exist_ok=True)

    shard_n_cells_target = manifests[0].shard_n_cells_target
    total_cells = 0
    total_nnz = 0
    prev_global_end: int | None = None
    next_shard_id = 0
    merged_shards: list[dict[str, Any]] = []

    for manifest in manifests:
        if prev_global_end is not None and manifest.global_start != prev_global_end:
            raise ValueError(
                f"Dataset manifests are not contiguous: expected {prev_global_end}, "
                f"got {manifest.global_start} for {manifest.dataset_id}"
            )
        if manifest.shard_n_cells_target != shard_n_cells_target:
            raise ValueError(
                "All dataset manifests must use the same shard_n_cells_target"
            )

        with open(manifest.csr_manifest_path, "r") as fh:
            csr_doc = yaml.safe_load(fh) or {}
        if csr_doc.get("kind") != "csr-corpus-manifest":
            raise ValueError(
                f"Expected csr-corpus-manifest, got {csr_doc.get('kind')!r}"
            )

        _validate_manifest_ranges(csr_doc["shards"], dataset_id=manifest.dataset_id)

        dataset_shard_root = shard_target_root / manifest.dataset_id
        dataset_shard_root.mkdir(parents=True, exist_ok=True)
        for shard in csr_doc["shards"]:
            source_rel = Path(str(shard["path"]))
            source_shard_dir = Path(manifest.output_dir) / source_rel
            target_rel = Path(shard_subdir) / manifest.dataset_id / source_rel.name
            target_shard_dir = output_dir / target_rel
            _symlink_tree(source_shard_dir, target_shard_dir)

            merged_shards.append(
                {
                    "shard_id": next_shard_id,
                    "path": str(target_rel),
                    "global_start": int(shard["global_start"]),
                    "global_end": int(shard["global_end"]),
                    "n_cells": int(shard["n_cells"]),
                    "total_nnz": int(shard["total_nnz"]),
                    "dataset_id": manifest.dataset_id,
                    "source_shard_id": int(shard["shard_id"]),
                }
            )
            next_shard_id += 1

        total_cells += manifest.n_cells
        total_nnz += manifest.total_nnz
        prev_global_end = manifest.global_end

    merged_shards.sort(key=lambda s: int(s["global_start"]))
    _validate_manifest_ranges(merged_shards, dataset_id="merged")

    merged_doc = {
        "kind": "csr-corpus-manifest",
        "contract_version": _CONTRACT_VERSION,
        "total_cells": total_cells,
        "total_nnz": total_nnz,
        "shard_n_cells_target": shard_n_cells_target,
        "n_shards": len(merged_shards),
        "shards": merged_shards,
    }
    manifest_path = output_dir / _CORPUS_MANIFEST_FILE
    with open(manifest_path, "w") as fh:
        yaml.safe_dump(merged_doc, fh, default_flow_style=False, sort_keys=False)
    return manifest_path


def package_csr_corpus(
    *,
    dataset_manifest_paths: Sequence[str | Path],
    template_corpus_root: str | Path,
    output_root: str | Path,
) -> Path:
    """Package a merged CSR corpus using metadata copied from an existing corpus."""
    template_root = Path(template_corpus_root)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_required_metadata_tree(template_root, output_dir)
    _rewrite_backend_metadata(output_dir / "corpus-index.yaml")
    global_metadata_path = output_dir / "global-metadata.yaml"
    if global_metadata_path.exists():
        _rewrite_backend_metadata(global_metadata_path)

    return merge_csr_dataset_manifests(dataset_manifest_paths, output_dir)


def _decode_h5_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_h5_attr(value.item())
    return str(value)


def _coerce_indices_int32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.int32:
        return arr
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"indices must be integer-typed, got {arr.dtype}")
    if arr.size:
        info = np.iinfo(np.int32)
        if int(arr.min()) < info.min or int(arr.max()) > info.max:
            raise OverflowError("indices exceed int32 range")
    return arr.astype(np.int32, copy=False)


def _coerce_counts_int32(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.int32:
        return arr
    info = np.iinfo(np.int32)
    if np.issubdtype(arr.dtype, np.integer):
        if arr.size:
            if int(arr.min()) < info.min or int(arr.max()) > info.max:
                raise OverflowError("counts exceed int32 range")
        return arr.astype(np.int32, copy=False)
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"counts must be integer-like, got {arr.dtype}")
    rounded = np.rint(arr)
    if not np.array_equal(arr, rounded):
        raise ValueError("counts contain non-integer float values")
    if rounded.size:
        if int(rounded.min()) < info.min or int(rounded.max()) > info.max:
            raise OverflowError("counts exceed int32 range")
    return rounded.astype(np.int32, copy=False)


def _copy_required_metadata_tree(template_root: Path, output_dir: Path) -> None:
    corpus_index_path = template_root / "corpus-index.yaml"
    meta_root = template_root / "meta"
    if not corpus_index_path.is_file():
        raise FileNotFoundError(f"Missing template corpus-index.yaml: {corpus_index_path}")
    if not meta_root.is_dir():
        raise FileNotFoundError(f"Missing template meta directory: {meta_root}")

    shutil.copy2(corpus_index_path, output_dir / "corpus-index.yaml")
    for optional_name in (
        "global-metadata.yaml",
        "corpus-ledger.parquet",
        "corpus-emission-spec.yaml",
    ):
        src = template_root / optional_name
        if src.exists():
            shutil.copy2(src, output_dir / optional_name)

    shutil.copytree(meta_root, output_dir / "meta", dirs_exist_ok=True)


def _rewrite_backend_metadata(path: Path) -> None:
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh) or {}
    if path.name == "corpus-index.yaml":
        doc.setdefault("global_metadata", {})["backend"] = "csr-memmap"
        doc["global_metadata"]["topology"] = "aggregate"
    else:
        doc["backend"] = "csr-memmap"
        doc["topology"] = "aggregate"
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, default_flow_style=False, sort_keys=False)


def _validate_manifest_ranges(
    shards: Sequence[dict[str, Any]], *, dataset_id: str
) -> None:
    prev_end: int | None = None
    for shard in sorted(shards, key=lambda s: int(s["global_start"])):
        start = int(shard["global_start"])
        end = int(shard["global_end"])
        n_cells = int(shard["n_cells"])
        if end - start != n_cells:
            raise ValueError(
                f"Shard range mismatch for {dataset_id}: [{start}, {end}) "
                f"does not match n_cells={n_cells}"
            )
        if prev_end is not None and start != prev_end:
            raise ValueError(
                f"Shard ranges are not contiguous for {dataset_id}: expected "
                f"{prev_end}, got {start}"
            )
        prev_end = end


def _symlink_tree(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists() or target_dir.is_symlink():
        raise FileExistsError(f"Target shard path already exists: {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir.symlink_to(source_dir, target_is_directory=True)
