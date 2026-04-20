# Phase 2 inspector and schema proposal workflow

## Goal

Produce dataset-level YAML artifacts from real h5ad inputs before any materialization:

- `dataset-summary.yaml` — dataset statistics, field inventory, count-source audit
- `schema.yaml` — single editable canonical field mapping (replaces old proposal + patch pair)

## Design constraints

- use `anndata.read_h5ad(..., backed="r")` for real inspection
- treat `.obs` and `.var` as lightweight metadata surfaces
- inspect `.X`, `.raw.X`, and named layers through small row samples rather than full matrix loads
- keep YAML human-reviewable while validating through strict typed runtime models
- preserve raw source fields and use literal `NA` for missing canonical values

## Count-source audit

Each candidate matrix source records:

- candidate name and rank
- backed storage kind and dtype
- shape
- sampled row count
- sampled nonzero value count and density
- fraction of non-integer sampled nonzero values
- maximum absolute deviation from the nearest integer
- non-negativity check
- inferred transform family from the source name
- recovery policy: `not-needed`, `allowed-with-explicit-assumption`, or `disallowed`

Preference order is count-like layers, then `.raw.X`, then `.X`, with normalized/log/binned layers demoted unless they are the only evidence available.

## Transform catalog

The runtime transform catalog currently exposes:

- `strip_prefix`
- `strip_suffix`
- `regex_sub`
- `normalize_case`
- `recognize_control`
- `join_with_plus`

These transforms are represented as typed specs in runtime objects and serialized as concise YAML records for review.

## CLI

The Slurm job uses a YAML config with dataset ids, source paths, releases, and an output root.

```bash
PYTHONPATH=src python -m perturb_data_lab.inspectors.cli --config /path/to/inspection-config.yaml --workers 3
```

Each dataset receives its own output directory containing the three review artifacts plus a batch manifest at the output root.
