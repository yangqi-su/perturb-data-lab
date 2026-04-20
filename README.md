# perturb-data-lab

Phase 2 now includes a lightweight h5ad inspector and YAML schema workflow (single `schema.yaml` replacing the old proposal/patch pair) for perturb-seq onboarding.

## Current scope

- define repo boundaries for the data-lab codebase
- lock the v0 contract catalog and additive metadata rules
- provide typed YAML review artifacts for dataset summaries, schema proposals, and schema patches
- inspect real h5ad metadata and count-source candidates on Slurm without eager matrix materialization
- keep materialization and backend benchmarking out of scope for now

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
├── tests/
└── pyproject.toml

```

## Current outputs

- `src/perturb_data_lab/contracts.py`: typed phase-1 blueprint objects
- `src/perturb_data_lab/inspectors/`: typed inspection models, transform catalog, workflow, and CLI
- `docs/phase-01-contract-blueprint.md`: repo boundaries and contract semantics
- `docs/phase-02-inspector-workflow.md`: inspector design, count audit rules, and YAML workflow
- `docs/git-workflow.md`: local git bootstrap and commit cadence
- `examples/contracts/*.yaml`: human-editable review artifact examples aligned with the runtime models
- `tests/test_inspector_workflow.py`: synthetic smoke coverage for artifact generation and YAML round-trips

YAML remains the human review format; runtime code consumes strict typed models.

## Running the inspector

Use a YAML config and run the CLI as a module:

```bash
PYTHONPATH=src python -m perturb_data_lab.inspectors.cli --config /path/to/inspection-config.yaml --workers 3
```

Real large h5ad inspection should run on Slurm CPU in the `torch_flashv3` environment.
