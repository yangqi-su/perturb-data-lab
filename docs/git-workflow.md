# Git workflow

## Bootstrap

- initialize `perturb-data-lab` and `perturb-backend-benchmark` as separate local git repositories
- keep each repo self-contained because the parent workspace is not a git repository
- start with a local default branch and do not push unless the user asks

## Commit cadence

- commit after milestone-sized progress, not after every file save
- keep commits scoped to one purpose: contracts, inspector workflow, materializer, loader, or benchmark changes
- prefer informative messages that explain why the milestone matters

## Message style

- `Define v0 contract blueprint for perturb-data-lab`
- `Bootstrap backend benchmark rubric and scenarios`
- `Add schema proposal validation for raw-to-canonical mappings`

## Guardrails

- do not commit generated outputs from protected symlink roots
- keep run artifacts under repo-local writable paths only
- preserve unrelated local changes when making later milestone commits
