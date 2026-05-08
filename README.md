# perturb-data-lab

Corpus-first preprocessing and runtime-loading system for large-scale perturb-seq training.

## Current scope

- define repo boundaries for the data-lab codebase
- lock the v0 contract catalog and additive metadata rules
- provide typed YAML review artifacts for dataset summaries and schema review
- inspect real h5ad metadata and count-source candidates on Slurm without eager matrix materialization
- materialize per-dataset Arrow/HF outputs with a corpus-level JSON tokenizer
- expose corpus-level perturbation/context emission specs for runtime sample generation
- provide a corpus-level `load_corpus()` / `Corpus` runtime API for unified corpus access

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
│       └── loaders/
│           ├── corpus_loader.py  # load_corpus(), Corpus, sampler/loader API
│           ├── loaders.py        # datasets, samplers, collate helpers
│           └── corpus.py         # legacy raw parquet utilities
├── tests/
└── pyproject.toml
```

## Current outputs

- `src/perturb_data_lab/contracts.py`: typed phase-1 blueprint objects
- `src/perturb_data_lab/inspectors/`: typed inspection models, transform catalog, workflow, and CLI
- `src/perturb_data_lab/materializers/tokenizer.py`: append-safe JSON corpus tokenizer (compatible with pertTF `SimpleVocab`)
- `src/perturb_data_lab/materializers/emission_spec.py`: corpus-level emission spec for runtime field emission
- `src/perturb_data_lab/loaders/corpus_loader.py`: `load_corpus()` and `Corpus` for unified corpus runtime access
- `docs/phase-01-contract-blueprint.md`: repo boundaries, contract catalog, and canonical field sets
- `docs/phase-02-inspector-workflow.md`: inspector design, count audit rules, and YAML workflow
- `docs/git-workflow.md`: local git bootstrap and commit cadence
- `examples/contracts/*.yaml`: human-editable review artifact examples
- `tests/`: synthetic smoke coverage

## Preferred corpus-first API

`load_corpus(path)` currently supports these artifact-backed corpus routes:

- aggregate Lance
- federated Lance
- aggregate CSR memmap

Dormant readers for aggregate/federated Zarr plus federated Arrow IPC, Parquet,
and WebDataset remain in the package for future artifact plans, but
`load_corpus(...)` does not wire those routes yet.

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

Use a YAML config and run the CLI as a module:

```bash
PYTHONPATH=src python -m perturb_data_lab.inspectors.cli --config /path/to/inspection-config.yaml --workers 3
```

Real large h5ad inspection should run on Slurm CPU in the `torch_flashv3` environment.
