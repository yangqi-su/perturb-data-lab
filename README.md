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

```python
from perturb_data_lab.loaders import load_corpus

corpus = load_corpus("/path/to/corpus", seq_len=1024)
corpus.set_sampler(batch_size=128, seed=0)

dataset = corpus.dataset()  # optional: raw expression dataset contract

for batch in corpus.loader(processing="gpu", num_workers=4):
    ...

for batch in corpus.loader(
    processing="cpu",
    num_workers=4,
    metadata_columns=["dataset_id", "cell_id", "size_factor"],
):
    meta = batch["meta_columns"]
    ...
```

- Aggregate and federated corpora share the same public API; topology-specific routing stays internal.
- `corpus.dataset()` exposes the backend-neutral expression dataset contract used by `corpus.loader(...)`.
- Compatible backends read expression through `read_expression_flat(global_indices) -> ExpressionBatch`.
- `corpus.loader(processing="gpu")` reads expression in the DataLoader and runs sparse processing in the main process.
- `corpus.loader(processing="cpu")` runs sparse processing in CPU workers and returns processed batches to the main process.
- Requested `metadata_columns` are attached after processing as columnar `batch["meta_columns"]`, so Lance workers do not need the full `MetadataIndex`.
- `size_factor` is optional metadata/pass-through, not a required sparse-processing input.
- Lance-backed loaders default to `multiprocessing_context="spawn"` for worker safety.

For direct inspection, prefer corpus-level helpers over reaching into `corpus.batch_executor`:

```python
expr = corpus.read_expression([0, 10, 24])
meta = corpus.take_metadata([0, 10, 24], columns=["dataset_id", "perturb_label"])
raw_batch = corpus.inspect_batch([0, 10, 24], metadata_columns=["perturb_label"])
```

## Migration note

- Preferred usage is now corpus-centric: `load_corpus(...)`, `corpus.set_sampler(...)`, `corpus.dataset()`, `corpus.loader(...)`, `corpus.read_expression(...)`, `corpus.take_metadata(...)`, and `corpus.inspect_batch(...)`.
- Migrate `corpus.batch_executor.read_batch(indices, ...)` to `corpus.inspect_batch(indices, metadata_columns=...)`.
- Migrate `corpus.batch_executor.read_expression_batch(indices)` to `corpus.read_expression(indices)`.
- Migrate `PerturbBatchDataset`, `collate_batch_dict`, and manual `DataLoader` + `GPUSparsePipeline` wiring to `corpus.set_sampler(...)` plus `corpus.loader(processing="gpu" | "cpu", ...)`.
- `BatchExecutor` and the legacy runtime dataset/collate surfaces remain available for compatibility-only use during this deprecation cycle.
- The per-row `ExpressionRow` / `read_expression(...)` reader path is also compatibility-only; runtime code should prefer flat `ExpressionBatch` reads via `Corpus.read_expression(...)` or backend `read_expression_flat(...)`.

## Running the inspector

Use a YAML config and run the CLI as a module:

```bash
PYTHONPATH=src python -m perturb_data_lab.inspectors.cli --config /path/to/inspection-config.yaml --workers 3
```

Real large h5ad inspection should run on Slurm CPU in the `torch_flashv3` environment.
