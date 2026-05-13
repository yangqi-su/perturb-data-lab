# Marson/Xorion pertTF public-loader smoke

This repository now contains a Phase 7 smoke path for validating the public
`PertTFPairedBatchLoader` on the full Marson/Xorion aggregate corpus without
editing the external `pertTF` repository.

## What the smoke does

- loads the existing aggregate Marson/Xorion Lance corpus with `load_corpus(...)`
- verifies that all 14 datasets expose canonical `hvg.parquet` rankings at runtime
- constructs a public `PertTFPairedBatchLoader`
- filters out rows missing required pertTF label columns before loader sampling
- scans a bounded number of public-loader batches until it finds a small mixed-dataset batch with `sampling_mode="hvg"`
- validates:
  - same-dataset and same-context source/target invariants
  - sampled source/target gene-ID alignment
  - union-vocabulary `full_expr_mask` / `full_expr_next_mask` semantics
  - worker-visible loader dataset state stays expression-only by design
- writes lightweight JSON + Markdown summaries to a caller-provided output directory

The entrypoint is:

```bash
python scripts/perttf_marson_xorion_smoke.py \
  --corpus-root /path/to/aggregate-corpus \
  --output-dir /path/to/phase-07-artifacts
```

## Why the smoke scans bounded public-loader batches

For the full 41.5M-cell corpus, this smoke keeps the validation bounded by using
the real public loader API directly and scanning only a small number of batches
until it finds one that mixes datasets.

That means the smoke exercises the same `for batch in perttf_loader:` path that
normal callers use while still avoiding an unnecessarily broad full-corpus run
just to prove the end-to-end tensor contract. Rows with null `cell_context`,
`perturb_label`, or `batch_id` are excluded from the smoke row pool so the
public loader only samples valid pertTF label rows.

## Typical Slurm invocation

```bash
PYTHONPATH=/path/to/perturb-data-lab/src \
conda run -n torch_flashv3 python scripts/perttf_marson_xorion_smoke.py \
  --corpus-root /autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/copilot/plans/archive/plans-20260503-marson-xorion-lance-aggregate-random-access/output/aggregate-corpus \
  --output-dir /autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/copilot/plans/active/<plan-id>/artifacts/phase-07-smoke \
  --num-workers 1 \
  --multiprocessing-context spawn
```

Recommended runtime knobs for the current smoke:

- `--batch-size 4`
- `--seq-len 64`
- `--hvg-top-k 2000`
- `--max-batches 32`
- small control-label override(s) when the corpus uses non-`WT` control names
- optional `--dataset-id ...` constraints when a particular mixed batch membership is desired

## Expected output files

- `phase-07-smoke-command.json`
- `phase-07-smoke-summary.json`
- `phase-07-smoke-summary.md`

These outputs are intentionally lightweight and safe to keep in a plan artifact
directory.

## Current limitations / handoff notes

- No files inside `pertTF/` are modified.
- The smoke validates public-loader-side `full_expr_mask` emission, but pertTF-side
  loss consumption of those masks remains future work.
- Control-label normalization is still configuration-driven. If a new corpus uses
  different control names, pass explicit `--control-label` overrides.
- The smoke is a contract/invariant check, not a training or throughput benchmark.
