# perturb-data-lab

Corpus-first preprocessing and runtime-loading system for large-scale perturb-seq training.

## Current scope

- inspect real h5ad metadata and count-source candidates without eager matrix materialization
- materialize new corpora and later append datasets through the public CLI
- keep materialization count-first and schema-independent
- draft and finalize canonical schemas after materialization, then canonicalize corpus metadata
- backfill canonical `hvg.parquet` ranking tables for existing Lance corpora without reopening source h5ad files
- expose a corpus-level `load_corpus()` / `Corpus` runtime API for unified access
- expose a pertTF-local adapter path and Marson/Xorion smoke workflow without modifying `pertTF`
- keep aggregate Lance as the production-default backend while leaving other backends wired for controlled experiments

## Repo layout

```text
perturb-data-lab/
├── docs/
│   ├── git-workflow.md
│   ├── phase-01-contract-blueprint.md
│   └── phase-02-inspector-workflow.md
├── examples/
│   └── contracts/
├── src/
│   └── perturb_data_lab/
│       ├── contracts.py
│       ├── inspectors/
│       ├── materializers/
│       │   ├── tokenizer.py      # corpus-level JSON tokenizer
│       │   └── emission_spec.py # corpus-level emission spec
│       ├── loaders/
│       │   ├── corpus_loader.py  # load_corpus(), Corpus, sampler/loader API
│       │   ├── loaders.py        # datasets, samplers, collate helpers
│       │   └── corpus.py         # legacy raw parquet utilities
│       └── pp/                   # streamed per-dataset stats/HVG/PCA/DE helpers
├── tests/
└── pyproject.toml
```

## Current outputs

- `src/perturb_data_lab/cli.py`: public `inspect`, `materialize`, `draft-schema`, `canonicalize`, `backfill-hvg`, `corpus-validate`, and `corpus-gc` entrypoints
- `src/perturb_data_lab/inspectors/`: inspection models, count-source audits, recovery classification, and review-bundle generation
- `src/perturb_data_lab/materializers/`: create/append corpus writers, aggregate/federated backends, manifests, and emission-spec helpers
- `src/perturb_data_lab/canonical/`: draft/final schema application and canonical obs/var generation
- `src/perturb_data_lab/loaders/corpus_loader.py`: `load_corpus()` and `Corpus` for unified runtime access
- `src/perturb_data_lab/pp/`: backend-agnostic streamed per-dataset stats, HVG, PCA/SVD, and Welch DE helpers
- `scripts/perttf_marson_xorion_smoke.py`: bounded full-corpus public-loader smoke for Marson/Xorion
- `docs/v0-onboarding-workflow.md`: current inspect → materialize → draft-schema → finalize-schema → canonicalize → load workflow
- `docs/canonicalization_handbook.md`: canonical schema review rules, transform behavior, tokenizer notes, and common failure modes
- `docs/perttf_marson_xorion_adapter_smoke.md`: user-facing handoff notes for the pertTF-local adapter smoke path
- `docs/v0-default-backend-decision.md`: current backend policy and default/experimental guidance
- `tests/`: focused regression and runtime smoke coverage

## Preferred onboarding workflow

The public workflow is now:

```text
inspect
→ materialize
→ draft-schema
→ finalize final-schema.yaml
→ canonicalize
→ load_corpus
```

Important constraints:

- Materialization is count-first and does **not** require finalized canonical schema inputs.
- Canonical metadata is required before `load_corpus()` succeeds.
- For large h5ad inputs, run inspection/materialization on Slurm CPU in `torch_flashv3`.
- Treat `data/`, `pertTF/`, and `perturb/` as read-only sources; write outputs only to repo-local real directories.

See `docs/v0-onboarding-workflow.md` for concrete create/append CLI examples and `docs/canonicalization_handbook.md` for draft-to-final schema review guidance.

### Materialization-time obs filtering

Add the dataset-summary `obs_filter:` field when you want to materialize only a
safe subset of cells before expression emission:

```yaml
obs_filter: "cell_line_or_type == 'T_cell' and donor_id in ['D1', 'D2'] and disease_state is not null"
```

- Supported filter language is intentionally small: comparisons, `is null`,
  `is not null`, `in [...]`, `not in [...]`, `and`, `or`, and parentheses.
- Filtering is applied before expression, raw obs, and size-factor emission.
- Filtered raw-obs outputs preserve `source_row_index` and `source_obs_index`
  so retained cells still map back to the original source rows.
- Keep materialized outputs in repo-local real directories only; never write
  generated artifacts into the read-only `data/`, `pertTF/`, or `perturb/`
  symlink roots.

## Preferred corpus-first API

Recommended policy:

- **Default production path:** aggregate Lance
- **Optional node-local staging path:** Zarr when chunked array artifacts are operationally preferable
- **Experimental but wired:** aggregate TileDB, aggregate CSR memmap, federated Lance, federated Zarr, federated Arrow IPC, federated HuggingFace datasets, federated Parquet
- **Not currently enabled by `load_corpus(...)`:** WebDataset

`load_corpus(path)` reconstructs a corpus from canonical metadata plus backend artifacts and exposes one backend-neutral runtime API across those supported routes.

```python
from perturb_data_lab.loaders import load_corpus

corpus = load_corpus("/path/to/corpus")
corpus.set_sampler(batch_size=128, seed=0)

dataset = corpus.dataset()  # ExpressionBatchDataset for custom loaders

for batch in corpus.loader(seq_len=1024, processing="gpu", num_workers=4):
    ...
```

- `load_corpus(...)` no longer accepts `seq_len`; pass it to `corpus.loader(...)`
  or to an explicit sparse pipeline.
- `corpus.set_sampler(...)` stores a sampler that `corpus.loader(...)` reuses when
  that loader call does not provide sampler-local overrides.
- If no sampler is stored and no loader-local sampler config is passed,
  `corpus.loader(...)` uses a default random sampler with `batch_size=128`.

### Canonical metadata loading defaults

```python
from perturb_data_lab.loaders import load_corpus

corpus = load_corpus("/path/to/corpus")

corpus_with_extra = load_corpus(
    "/path/to/corpus",
    extra_metadata_columns=["raw_cell_type", "donor_age"],
)
```

- `load_corpus(...)` now projects only canonical/core columns from
  `canonical-obs.parquet` by default instead of eagerly loading every stored
  column.
- Use `extra_metadata_columns=[...]` when a downstream workflow needs specific
  extra canonical-obs columns; missing requested extras fail early.
- Canonical string columns normalize null-like literals such as `"NA"`,
  `"null"`, `"."`, and empty strings at read time.
- Canonicalization write behavior is unchanged: the on-disk canonical parquet
  files are not rewritten or normalized by `load_corpus(...)`.

### pertTF paired loader

```python
from perturb_data_lab.loaders import (
    PertTFAdapterConfig,
    PertTFPairedBatchLoader,
    load_corpus,
)

corpus = load_corpus("/path/to/corpus")
perttf_loader = PertTFPairedBatchLoader(
    corpus,
    batch_size=8,
    seq_len=1024,
    config=PertTFAdapterConfig(control_labels=("WT",)),
    num_workers=2,
)

for batch in perttf_loader:
    train_step(batch)
```

- `PertTFPairedBatchLoader` yields final pertTF-ready batch dictionaries; normal
  callers do not need to assemble raw pair requests or call
  `build_from_raw_pair_batch(...)` directly.
- `set_epoch(epoch)` is available for deterministic reshuffling between epochs.
- When `num_workers > 0`, the loader keeps pair planning in the main process and
  uses worker processes only for source/target expression reads.

### Metadata and inspection helpers

```python
meta = corpus.take_metadata(
    [0, 10, 24],
    columns=["dataset_id", "local_row_index", "perturb_label"],
)

raw_batch = corpus.inspect_batch(
    [0, 10, 24],
    metadata_columns=["dataset_id", "perturb_label"],
)

for batch in corpus.loader(
    processing="cpu",
    seq_len=1024,
    num_workers=4,
    metadata_columns=["dataset_id", "perturb_label", "local_row_index"],
):
    meta_columns = batch["meta_columns"]
    ...
```

- `corpus.take_metadata(...)` is the easiest way to recover provenance fields such
  as `local_row_index` from global row indices.
- `corpus.inspect_batch(...)` returns the raw flat expression batch plus optional
  metadata columns for spot checks and debugging.
- `corpus.loader(metadata_columns=...)` attaches rich metadata after sparse
  processing, so workers stay expression-only.

### Custom row subsets and pertTF pairing pools

```python
from perturb_data_lab.loaders.adapters.perttf import PerturbationPairSampler

row_subset = [0, 3, 10, 22]

for batch in corpus.loader(
    seq_len=1024,
    row_indices=row_subset,
    metadata_columns=["dataset_id", "perturb_label"],
):
    ...

pair_sampler = PerturbationPairSampler(
    corpus.metadata_index,
    batch_size=64,
    source_indices=row_subset,
    target_candidate_indices=[0, 3, 10, 41, 52],
)
```

- `row_indices` are corpus-global row positions; they restrict loader/sampler
  candidate pools without rebuilding the full metadata index.
- `source_indices` restrict which rows can be sampled as pertTF sources.
- `target_candidate_indices` further restrict valid paired targets and is
  accepted only when `source_indices` is also provided.
- Same-dataset and same-context pairing invariants are still enforced inside
  the configured source/target pools.

### Stored sampler reuse and loader-local override warning

```python
import warnings

corpus.set_sampler(batch_size=256, seed=0)

loader = corpus.loader(seq_len=2048)
first_batch = next(loader)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    override_loader = corpus.loader(seq_len=2048, batch_size=64)
    override_batch = next(override_loader)

warning_messages = [str(item.message) for item in caught]
```

- `loader` reuses the stored sampler because no loader-local sampler arguments were
  supplied.
- `override_loader` uses the loader-local `batch_size=64` sampler for that call and
  emits a `UserWarning` because it overrides the stored sampler.

### Custom DataLoader composition

```python
from functools import partial

from torch.utils.data import DataLoader

from perturb_data_lab.loaders import (
    GPUSparsePipeline,
    collate_expression_batch,
    collate_expression_batch_cpu,
)

dataset = corpus.dataset()  # ExpressionBatchDataset
sampler = corpus.set_sampler(batch_size=256, seed=0)

gpu_loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collate_expression_batch,
)

cpu_loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=4,
    persistent_workers=True,
    collate_fn=partial(
        collate_expression_batch_cpu,
        pipeline=GPUSparsePipeline(corpus.feature_registry, seq_len=2048),
    ),
)

gpu_batch = next(iter(gpu_loader))
cpu_batch = next(iter(cpu_loader))
```

- Use `collate_expression_batch` when you want the raw expression batch tensors and
  will run sparse processing later in the main process.
- Use `collate_expression_batch_cpu` when you want sparse processing to happen in
  DataLoader workers.

### Runtime notes

- Aggregate and federated corpora share the same public API; topology-specific
  routing stays internal.
- `corpus.dataset()` exposes the backend-neutral expression dataset contract used
  by `corpus.loader(...)`.
- Compatible backends read expression through
  `read_expression_flat(global_indices) -> ExpressionBatch`.
- `corpus.loader(processing="gpu")` reads expression in the DataLoader and runs
  sparse processing in the main process.
- `corpus.loader(processing="cpu")` runs sparse processing in CPU workers and
  returns processed batches to the main process.
- `size_factor` is optional metadata/pass-through, not a required sparse-processing
  input.
- Lance-backed loaders default to `multiprocessing_context="spawn"` for worker
  safety.

## Streamed `pp` helpers

```python
from perturb_data_lab.pp import (
    calculate_hvgs,
    calculate_lognorm_stats,
    rank_genes_ttest,
    run_pca,
)

output_dir = "./artifacts/pp"

stats = calculate_lognorm_stats(corpus, batch_size=2048, output_dir=output_dir)
hvg_frame = calculate_hvgs(corpus, batch_size=2048, n_hvg=2000)
pca = run_pca(
    corpus,
    batch_size=2048,
    n_components=32,
    hvg_frame=hvg_frame,
    output_dir=output_dir,
    overwrite=True,
)
de = rank_genes_ttest(
    corpus,
    control_label="CRISPR_control",
    batch_size=2048,
    top_k=50,
    output_dir=output_dir,
    overwrite=True,
)
```

- `calculate_lognorm_stats(...)` streams `log1p(count / size_factor)` summary
  stats and can write per-dataset `lognorm-stats.parquet` outputs plus
  provenance sidecars.
- `calculate_hvgs(...)` returns a ranked per-dataset HVG frame that can be fed
  into `run_pca(..., hvg_frame=...)`.
- `run_pca(...)` currently exposes uncentered sparse `method="truncated_svd"`
  semantics while keeping a PCA-like public API.
- `rank_genes_ttest(...)` runs streamed per-dataset Welch DE against an
  explicit control label and writes top-k `ttest-degs.parquet` artifacts when
  `output_dir` is set.
- Write generated `pp` outputs only to repo-local real directories such as
  `./artifacts/pp`, never into the protected `data/`, `pertTF/`, or
  `perturb/` symlink roots.

## Migration note

- Preferred usage is corpus-centric: `load_corpus(...)`, `corpus.set_sampler(...)`,
  `corpus.dataset()`, `corpus.loader(...)`, `corpus.read_expression(...)`,
  `corpus.take_metadata(...)`, and `corpus.inspect_batch(...)`.
- Legacy executor-centric guidance is intentionally gone; use
  `corpus.inspect_batch(...)` for raw batch inspection and
  `corpus.loader(processing="gpu" | "cpu", ...)` for training-time iteration.
- Readers are flat-only; runtime code should prefer `Corpus.read_expression(...)`
  or backend `read_expression_flat(...)` for direct expression access.

## Running the inspector

Batch mode remains available:

```bash
PYTHONPATH=src python -m perturb_data_lab.inspectors.cli --config /path/to/inspection-config.yaml --workers 3
```

The preferred public entrypoint for a single dataset is:

```bash
PYTHONPATH=src python -m perturb_data_lab.cli inspect \
  --source /path/to/dataset.h5ad \
  --dataset-id my_dataset \
  --output-dir /path/to/review/my_dataset
```

Real large h5ad inspection should run on Slurm CPU in the `torch_flashv3` environment.
