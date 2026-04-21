# perturb-data-lab

Corpus-first preprocessing and runtime-loading system for large-scale perturb-seq training.

## Current scope

- define repo boundaries for the data-lab codebase
- lock the v0 contract catalog and additive metadata rules
- provide typed YAML review artifacts for dataset summaries and schema review
- inspect real h5ad metadata and count-source candidates on Slurm without eager matrix materialization
- materialize per-dataset Arrow/HF outputs with a corpus-level JSON tokenizer
- expose corpus-level perturbation/context emission specs for runtime sample generation
- provide a multi-dataset `CorpusLoader` for unified corpus runtime access

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
│           ├── loaders.py        # ArrowHFCellReader, samplers
│           └── corpus.py         # CorpusLoader (multi-dataset runtime)
├── tests/
└── pyproject.toml
```

## Current outputs

- `src/perturb_data_lab/contracts.py`: typed phase-1 blueprint objects
- `src/perturb_data_lab/inspectors/`: typed inspection models, transform catalog, workflow, and CLI
- `src/perturb_data_lab/materializers/tokenizer.py`: append-safe JSON corpus tokenizer (compatible with pertTF `SimpleVocab`)
- `src/perturb_data_lab/materializers/emission_spec.py`: corpus-level emission spec for runtime field emission
- `src/perturb_data_lab/loaders/corpus.py`: `CorpusLoader` for unified multi-dataset runtime access
- `docs/phase-01-contract-blueprint.md`: repo boundaries, contract catalog, and canonical field sets
- `docs/phase-02-inspector-workflow.md`: inspector design, count audit rules, and YAML workflow
- `docs/git-workflow.md`: local git bootstrap and commit cadence
- `examples/contracts/*.yaml`: human-editable review artifact examples
- `tests/`: synthetic smoke coverage

## Running the inspector

Use a YAML config and run the CLI as a module:

```bash
PYTHONPATH=src python -m perturb_data_lab.inspectors.cli --config /path/to/inspection-config.yaml --workers 3
```

Real large h5ad inspection should run on Slurm CPU in the `torch_flashv3` environment.
