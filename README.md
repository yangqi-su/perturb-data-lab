# perturb-data-lab

Phase 1 bootstraps the schema-first data project for perturb-seq onboarding.

## Current scope

- define repo boundaries for the data-lab codebase
- lock the v0 contract catalog and additive metadata rules
- provide example YAML review artifacts for later phases
- keep real h5ad inspection and materialization out of this phase

## Repo layout

```text
perturb-data-lab/
├── docs/
│   ├── git-workflow.md
│   └── phase-01-contract-blueprint.md
├── examples/
│   └── contracts/
├── src/
│   └── perturb_data_lab/
└── pyproject.toml
```

## Phase 1 outputs

- `src/perturb_data_lab/contracts.py`: typed phase-1 blueprint objects
- `docs/phase-01-contract-blueprint.md`: repo boundaries and contract semantics
- `docs/git-workflow.md`: local git bootstrap and commit cadence
- `examples/contracts/*.yaml`: human-editable review artifact examples

YAML remains the human review format; runtime code consumes strict typed models.
