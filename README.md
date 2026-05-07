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

## Preferred corpus API

```python
from perturb_data_lab.loaders import load_corpus

corpus = load_corpus("/path/to/corpus", seq_len=1024)
corpus.set_sampler(batch_size=128, seed=0)

dataset = corpus.dataset()

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

- The same public API works for aggregate Lance and federated Lance corpora.
- `corpus.dataset()` is expression-only for the Lance fast path; rich metadata stays on the `Corpus` side by default.
- `corpus.loader(processing="gpu")` reads expression in the DataLoader and runs sparse processing in the main process.
- `corpus.loader(processing="cpu")` runs sparse processing in CPU workers and returns processed batches to the main process.
- Requested `metadata_columns` are attached after processing as columnar `batch["meta_columns"]`, so Lance workers do not need the full `MetadataIndex`.
- `size_factor` is optional metadata/pass-through, not a required sparse-processing input.
- Lance-backed loaders default to `multiprocessing_context="spawn"` for worker safety.

## Migration note

- Preferred usage is now corpus-centric: `load_corpus(...)`, `corpus.set_sampler(...)`, and `corpus.loader(...)`.
- `BatchExecutor` remains available for direct or legacy batch reads, but new training loops should not need to manually wire `BatchExecutor`, `DataLoader`, and `GPUSparsePipeline` together.

## Running the inspector

Use a YAML config and run the CLI as a module:

```bash
PYTHONPATH=src python -m perturb_data_lab.inspectors.cli --config /path/to/inspection-config.yaml --workers 3
```

Real large h5ad inspection should run on Slurm CPU in the `torch_flashv3` environment.
