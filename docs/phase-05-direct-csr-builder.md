# Direct CSR builder fast path

This phase adds a direct h5ad-to-CSR path for aggregate corpora.

## Supported source layout

- HDF5-backed sparse group at `X`
- `encoding-type: csr_matrix`
- datasets: `data`, `indices`, `indptr`
- integer-valued counts are cast to `int32`
- `indices` are cast to `int32`

## Fast-path behavior

- reads row chunks directly from the source h5ad CSR buffers
- writes shard chunks through `CsrMemmapWriter.append_csr_chunk(...)`
- avoids Lance readback
- avoids Arrow `to_pylist()`
- avoids per-cell NumPy conversion in the conversion hot path

## Unsupported layouts

- dense `X`
- CSC-backed sparse groups
- alternate matrix paths not explicitly passed to the reader/converter

Unsupported layouts currently raise a clear error so Phase 4 pilot work can stop early instead of silently falling back to a slow path.

## Packaging contract

- each dataset conversion writes `direct-csr-dataset-manifest.yaml`
- aggregate packaging merges those manifests into `csr-corpus-manifest.yaml`
- packaging reuses corpus metadata from an existing aggregate corpus and rewrites top-level backend metadata to `csr-memmap`
