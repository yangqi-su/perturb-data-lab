# Marson/Xorion pertTF adapter smoke

This repository now contains a Phase 8 smoke path for validating the full
`perturb-data-lab` → pertTF-style adapter contract without editing the external
`pertTF` repository.

## What the smoke does

- loads the existing aggregate Marson/Xorion Lance corpus with `load_corpus(...)`
- verifies that all 14 datasets expose canonical `hvg.parquet` rankings at runtime
- constructs `PertTFCorpusAdapter` and `PertTFPairedBatchBuilder`
- builds a small mixed-dataset paired batch with `sampling_mode="hvg"`
- validates:
  - same-dataset and same-context source/target invariants
  - sampled source/target gene-ID alignment
  - union-vocabulary `full_expr_mask` / `full_expr_next_mask` semantics
- writes lightweight JSON + Markdown summaries to a caller-provided output directory

The entrypoint is:

```bash
python scripts/perttf_marson_xorion_smoke.py \
  --corpus-root /path/to/aggregate-corpus \
  --output-dir /path/to/phase-08-artifacts
```

## Why the smoke uses explicit source rows

For the full 41.5M-cell corpus, this smoke keeps the validation bounded: it
selects a small set of non-control source rows across multiple datasets and then
applies the adapter-side `self_to_control_label` semantics that correspond to the
current `next_cell_pred="pert"` path.

That means the smoke exercises the real adapter/builder code on the production
corpus while avoiding an unnecessarily broad full-corpus pairing pass just to
prove the end-to-end tensor contract.

## Typical Slurm invocation

```bash
PYTHONPATH=/path/to/perturb-data-lab/src \
conda run -n torch_flashv3 python scripts/perttf_marson_xorion_smoke.py \
  --corpus-root /autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/copilot/plans/archive/plans-20260503-marson-xorion-lance-aggregate-random-access/output/aggregate-corpus \
  --output-dir /autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/copilot/plans/active/<plan-id>/artifacts/phase-08-smoke
```

Recommended runtime knobs for the current smoke:

- `--batch-size 4`
- `--seq-len 64`
- `--hvg-top-k 2000`
- small control-label override(s) when the corpus uses non-`WT` control names

## Expected output files

- `phase-08-smoke-command.json`
- `phase-08-smoke-summary.json`
- `phase-08-smoke-summary.md`

These outputs are intentionally lightweight and safe to keep in a plan artifact
directory.

## Current limitations / handoff notes

- No files inside `pertTF/` are modified.
- The smoke validates adapter-side `full_expr_mask` emission, but pertTF-side loss
  consumption of those masks remains future work.
- Control-label normalization is still configuration-driven. If a new corpus uses
  different control names, pass explicit `--control-label` overrides.
- The smoke is a contract/invariant check, not a training or throughput benchmark.
