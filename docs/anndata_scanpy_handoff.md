# AnnData extraction, subsampling, Scanpy handoff, and IncrementalPCA

## When to use Scanpy vs streamed `pp`

- Use `corpus.to_anndata(...)` when one dataset or one deterministic subset will fit in memory and you want Scanpy to own the next steps.
- Use streamed `perturb_data_lab.pp` helpers when you want bounded-memory stats/HVG/PCA/DE directly from the on-disk corpus without building an in-memory `AnnData` first.
- `to_anndata(...)` is per-dataset only and eager; it is not a backed or fully on-disk Scanpy workflow.
- Corpus extraction is counts-only: it builds CSR `adata.X` and does **not** add normalized or log-transformed layers.
- Scanpy is user-managed and optional. `perturb-data-lab` does not install it as a core dependency.

## Imports

```python
from perturb_data_lab.loaders import load_corpus, select_obs_indices
from perturb_data_lab.pp import calculate_hvgs, run_pca
```

If you want centered `method="incremental_pca"`, install the optional PCA dependency first:

```bash
pip install ".[pca]"
```

## Dry-run before eager AnnData construction

Load extra canonical metadata columns up front if you want to stratify on them or include them in `adata.obs`:

```python
corpus = load_corpus(
    "/path/to/corpus",
    extra_metadata_columns=["donor_id", "batch_id"],
)

estimate = corpus.to_anndata(
    dataset_id="replogle_k562",
    obs_columns=["perturb_label", "donor_id"],
    var_columns=["gene_id"],
    dry_run=True,
    max_memory_bytes=8_000_000_000,
    on_exceed="warn",
)

print(estimate["n_obs"], estimate["n_vars"], estimate["nnz"])
print(estimate["csr_memory_bytes"])
print(estimate["selected_row_index_summary"])
```

- `dry_run=True` returns shape, `nnz`, CSR memory estimates, metadata footprint estimates, and memory-guard status without materializing `adata.X`.
- Set `on_exceed="raise"` when you want the same request to fail before eager construction if `max_memory_bytes` is exceeded.

## Deterministic observation selection

### Random subset

```python
random_selection = select_obs_indices(
    corpus,
    dataset_id="replogle_k562",
    strategy="random",
    max_cells=20_000,
    seed=17,
)
```

### Stratified subset

```python
stratified_selection = corpus.select_obs_indices(
    dataset_id="replogle_k562",
    strategy="stratified",
    max_cells=20_000,
    stratify_by=["perturb_label"],
    seed=17,
)
```

### Balanced subset with provenance

```python
balanced_selection = corpus.select_obs_indices(
    dataset_id="replogle_k562",
    strategy="balanced",
    max_cells=20_000,
    stratify_by=["perturb_label", "donor_id"],
    max_per_group=500,
    drop_null_groups=True,
    seed=17,
)

balanced_selection.write_provenance("./artifacts/selections/replogle-balanced")
```

- Returned `row_indices` are corpus-global row indices and can be passed directly into `to_anndata(...)` or `run_pca(...)`.
- Provenance outputs should go to repo-local real directories such as `./artifacts/...`, never to `data/`, `pertTF/`, or `perturb/`.

## Scanpy handoff after counts-only extraction

```python
adata = corpus.to_anndata(
    dataset_id="replogle_k562",
    row_indices=balanced_selection.row_indices,
    obs_columns=["perturb_label", "donor_id"],
)

import scanpy as sc

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata)
```

- `adata.obs` always includes stable provenance fields such as `dataset_id`, `dataset_index`, `global_row_index`, and `local_row_index`; requested `obs_columns` are added alongside them.
- `adata.var` includes dataset-local feature identifiers plus global feature IDs.
- Because the corpus runtime never writes normalized/log layers into the exported object, any Scanpy normalization or log transform is explicitly owned by the Scanpy-side workflow.

## Bounded-memory IncrementalPCA on the corpus

```python
hvg_frame = calculate_hvgs(
    corpus,
    dataset_id="replogle_k562",
    batch_size=1024,
    n_hvg=2000,
)

fit_selection = corpus.select_obs_indices(
    dataset_id="replogle_k562",
    strategy="balanced",
    max_cells=20_000,
    stratify_by=["perturb_label"],
    max_per_group=500,
    seed=17,
)

transform_selection = corpus.select_obs_indices(
    dataset_id="replogle_k562",
    strategy="random",
    max_cells=80_000,
    seed=23,
)

result = run_pca(
    corpus,
    dataset_id="replogle_k562",
    method="incremental_pca",
    batch_size=1024,
    n_components=50,
    hvg_frame=hvg_frame,
    fit_row_indices=fit_selection.row_indices,
    transform_row_indices=transform_selection.row_indices,
    max_dense_batch_bytes=2_000_000_000,
    output_dir="./artifacts/pp/replogle-ipca",
    overwrite=True,
)
```

- `fit_row_indices` lets you fit on a bounded deterministic subset while `transform_row_indices` controls which rows receive embeddings.
- Omit `transform_row_indices` to transform all rows in the requested dataset.
- `max_dense_batch_bytes` guards the dense `batch_size × selected_features` working set before outputs are written.
- `method="incremental_pca"` is the supported slim-main streamed PCA route; the legacy `method="truncated_svd"` path remains only on the local experimental snapshot branch `experimental/all-backends-pre-slim-20260514`.
