# Canonicalization Handbook

## Purpose

This handbook explains the current canonicalization path in `perturb-data-lab`:

```text
inspect
→ materialize
→ draft-schema
→ finalize final-schema.yaml
→ canonicalize
→ load_corpus
```

Use it when you need to review `draft-schema.yaml`, write `final-schema.yaml`, understand how canonical obs/var files are produced, or reason about how loader gene IDs become runtime token IDs.

## What canonicalization reads and writes

Canonicalization is dataset-local and runs after materialization.

For each dataset, it reads:

- `meta/<dataset_id>/dataset-summary.yaml`
- `meta/<dataset_id>/raw-obs.parquet`
- `meta/<dataset_id>/raw-var.parquet`
- `meta/<dataset_id>/size-factor.parquet` when present
- `meta/<dataset_id>/final-schema.yaml`

It writes:

- `meta/<dataset_id>/canonical_meta/canonical-obs.parquet`
- `meta/<dataset_id>/canonical_meta/canonical-var.parquet`

Current CLI behavior:

- `draft-schema` writes `draft-schema.yaml`
- `canonicalize` only discovers `final-schema.yaml`
- `canonicalize` no longer writes `corpus-vocab.yaml`

## Data flow at a glance

1. `inspect` profiles metadata, count-source candidates, sampled values, and likely control labels.
2. `materialize` writes raw sidecars under `meta/<dataset_id>/`.
3. `draft-schema` converts those sidecars plus inspection hints into a reviewable `draft-schema.yaml`.
4. A human or agent reviews the draft and saves the approved mapping as `final-schema.yaml`.
5. `canonicalize` applies the approved mappings and ordered transforms to raw obs/var data.
6. `load_corpus()` consumes canonical metadata and reconstructs corpus-level gene IDs from `canonical_gene_id` values, optionally anchored by `gene-tokenizer.json`.

## Raw sidecars versus canonical outputs

Raw sidecars are provenance-preserving inputs. They should answer: “what did the source dataset actually contain?”

- `raw-obs.parquet`: raw cell metadata surface used for canonical obs mapping
- `raw-var.parquet`: raw feature metadata surface used for canonical var mapping
- `size-factor.parquet`: optional numeric sidecar for `size_factor`

Canonical outputs are normalized views used by runtime loaders.

- `canonical-obs.parquet`: required canonical cell metadata columns plus optional extensible columns
- `canonical-var.parquet`: required canonical feature identity columns plus optional extensible columns

Keep high-cardinality or ambiguous fields raw-only unless there is a clear downstream need to canonicalize them.

## `draft-schema.yaml` versus `final-schema.yaml`

`draft-schema.yaml` is heuristic and review-oriented.

- it may suggest transforms from sampled values
- it may suggest `coalesce` for complementary perturbation columns
- it may include conservative notes about uncertainty or review-needed decisions
- it should not be treated as automatically safe for production canonicalization

`final-schema.yaml` is the reviewed contract that `canonicalize` executes.

- use the same schema structure
- keep only mappings you actually approve
- set `status: ready` as the review marker
- prefer the smallest faithful mapping over speculative cleanup

Minimal example:

```yaml
kind: canonicalization-schema
contract_version: 0.3.0
dataset_id: replogle_like
status: ready
gene_mapping:
  enabled: false
  engine: identity
obs_column_mappings:
  - canonical_name: perturb_label
    strategy: coalesce
    source_columns: [perturbation, target_gene]
    transforms:
      - name: strip_whitespace
        args: {}
      - name: strip_guide_suffix
        args: {}
      - name: map_control_labels
        args:
          candidates: [NTC, non-targeting]
          output: ctrl
var_column_mappings:
  - canonical_name: origin_index
    strategy: passthrough
  - canonical_name: gene_id
    strategy: source-field
    source_column: feature_id
    transforms:
      - name: strip_ensembl_version
        args: {}
  - canonical_name: canonical_gene_id
    strategy: gene-mapping
    enabled: false
    engine: identity
  - canonical_name: global_id
    strategy: auto
```

## Required canonical fields

### Required obs fields

- `assay`
- `batch_id`
- `cell_context`
- `cell_id`
- `cell_line_or_type`
- `condition`
- `dataset_id`
- `dataset_index`
- `disease_state`
- `donor_id`
- `dose`
- `dose_unit`
- `global_row_index`
- `local_row_index`
- `perturb_label`
- `perturb_type`
- `sex`
- `size_factor`
- `species`
- `timepoint`
- `timepoint_unit`
- `tissue`

### Required var fields

- `origin_index`
- `gene_id`
- `canonical_gene_id`
- `global_id`

Notes:

- `size_factor` comes from `size-factor.parquet` when present and otherwise defaults to `1.0`.
- `row-index` currently emits zero-based row order for the dataset sidecars.
- `canonical_gene_id` is the harmonized gene identifier that runtime loading actually uses.

## Obs mapping strategies

The current runner supports these obs mapping strategies:

### `source-field`

Read one raw column, then run ordered transforms.

```yaml
- canonical_name: sex
  strategy: source-field
  source_column: donor_sex
  transforms:
    - name: normalize_case
      args: {mode: lower}
```

### `literal`

Fill one constant value for every row.

```yaml
- canonical_name: species
  strategy: literal
  literal_value: human
```

### `null`

Fill the mapping fallback, usually `NA` or another explicit placeholder.

### `row-index`

Emit `0..n-1` in sidecar row order.

### `coalesce`

Take the first non-null-like value from `source_columns`, then run transforms.

### `join`

Join multiple columns with `separator`, optionally skipping nulls.

### `template`

Render a Python format template like `{cellline}:{treatment}:{time_raw}`.

`missing_value_behavior` can be:

- `fallback`
- `empty`
- `literal`

### `conditional`

Evaluate ordered cases, then optional default output.

Predicates currently supported:

- `equals`
- `in`
- `not_null`

## Var mapping strategies

The current runner supports these var strategies:

- `passthrough`
- `source-field`
- `literal`
- `null`
- `gene-mapping`
- `auto`

Typical usage:

- `origin_index`: `passthrough`
- `gene_id`: `source-field`
- `canonical_gene_id`: `gene-mapping`
- `global_id`: `auto`

`global_id` is generated deterministically from canonical gene IDs during canonicalization, but loader tokenization is rebuilt from `canonical_gene_id` values rather than trusting that field as the corpus-global runtime contract.

## Transform execution order

Transforms run after the mapping strategy resolves a value.

- `source-field`: transforms apply to the extracted raw value
- `coalesce`, `join`, `template`, `conditional`: transforms apply to the derived output value
- transforms run in the order listed in YAML
- if a transform name is unknown to the runtime dispatcher, it is warned and skipped
- if a transform raises an exception or ends in a null-like value, the mapping falls back to its `fallback`

Practical rule: put cleanup transforms before harmonization transforms.

Good pattern for perturb labels:

```yaml
transforms:
  - name: strip_whitespace
    args: {}
  - name: strip_guide_suffix
    args: {}
  - name: map_control_labels
    args:
      candidates: [NTC, non-targeting]
      output: ctrl
```

## Runtime-dispatched transforms

These transform names are currently wired through the canonicalization runner:

- `map_control_labels`
- `strip_whitespace`
- `replace_empty_with_null`
- `strip_prefix`
- `strip_suffix`
- `strip_guide_suffix`
- `regex_sub`
- `normalize_case`
- `map_values`
- `split_on_delimiter`
- `dose_parse`
- `dose_unit`
- `normalize_dose_unit`
- `timepoint_parse`
- `timepoint_unit`
- `normalize_time_unit`
- `strip_ensembl_version`
- `normalize_boolean`

Important: some helper or legacy functions exist in `inspectors/transforms.py`, but only the names above are currently dispatched by `get_transform()` for schema execution.

## Control-label review and the default `ctrl`

Inspection now records `control_label_candidates` from likely metadata columns such as perturbation labels or explicit control flags.

Drafting uses those hints conservatively:

- high-confidence labels like `NTC` or `non-targeting` can trigger a suggested `map_control_labels`
- ambiguous labels like `WT` stay review-only and should not be blindly collapsed to controls

Recommended policy:

- standardize to `ctrl` unless you have a corpus-wide reason not to
- keep the configured `output` consistent across datasets in the same corpus
- review candidate values in `dataset-summary.yaml` before approving the transform

## Gene identifiers, `canonical_gene_id`, and `gene-tokenizer.json`

### `gene_id` versus `canonical_gene_id`

- `gene_id`: the raw or lightly cleaned source identifier from `raw-var.parquet`
- `canonical_gene_id`: the harmonized identifier used for corpus-level runtime alignment

The top-level `gene_mapping` block controls how `gene_id` becomes `canonical_gene_id`.

Current engines:

- `identity`
- `mapping_file`
- `gget`

If drafting sees Ensembl-like IDs in samples, it may suggest identity-preserving cleanup such as `strip_ensembl_version` and can infer a non-identity mapping configuration when appropriate.

### What `gene-tokenizer.json` means

`gene-tokenizer.json` is the persisted corpus-level token contract for `canonical_gene_id` values.

- token IDs are append-stable
- dataset order comes from `corpus-index.yaml`, not alphabetical ordering
- unseen genes are appended in each dataset's local `origin_index` order
- the file records `dataset_build_order` and per-dataset token spans for auditability

### How `load_corpus()` uses it

At load time:

1. `load_corpus()` discovers dataset order from `corpus-index.yaml`.
2. If `gene-tokenizer.json` exists, it is loaded and its `dataset_build_order` must match the corpus index order.
3. If the tokenizer file is absent, the loader deterministically rebuilds the same mapping from each dataset's `canonical-var.parquet` in corpus order.
4. `FeatureRegistry` then maps per-dataset local `origin_index` values onto corpus-global token IDs using `canonical_gene_id`.

Current limitation: canonicalization itself does not automatically emit `gene-tokenizer.json`; the runtime can reconstruct it when absent.

## Worked examples

### Replogle-like perturb labels

Use `coalesce` when one column holds control labels while another holds target genes.

```yaml
- canonical_name: perturb_label
  strategy: coalesce
  source_columns: [perturbation, target_gene, guide_id]
  transforms:
    - name: strip_whitespace
      args: {}
    - name: strip_guide_suffix
      args: {}
    - name: map_control_labels
      args:
        candidates: [NTC]
        output: ctrl
```

### Sex or species normalization

```yaml
- canonical_name: species
  strategy: source-field
  source_column: organism
  transforms:
    - name: map_values
      args:
        mapping:
          Homo sapiens: human
          Mus musculus: mouse
```

### Dose and time parsing

```yaml
- canonical_name: dose
  strategy: source-field
  source_column: treatment_dose
  transforms:
    - name: dose_parse
      args: {}

- canonical_name: dose_unit
  strategy: source-field
  source_column: treatment_dose
  transforms:
    - name: dose_unit
      args: {}

- canonical_name: timepoint
  strategy: source-field
  source_column: time_raw
  transforms:
    - name: timepoint_parse
      args: {}

- canonical_name: timepoint_unit
  strategy: source-field
  source_column: time_raw
  transforms:
    - name: timepoint_unit
      args: {}
```

### Ensembl version stripping

```yaml
- canonical_name: gene_id
  strategy: source-field
  source_column: feature_id
  transforms:
    - name: strip_ensembl_version
      args: {}
```

### Combinatorial perturbations

```yaml
- canonical_name: condition
  strategy: join
  source_columns: [guide_1, guide_2]
  separator: "+"
  skip_nulls: true
```

Use `join` when both fields should survive. Use `coalesce` when only the first non-null field should win.

## Common failure modes

### Duplicate canonical names

Schema validation rejects duplicate canonical output names across required and extensible columns.

### Wrong raw field selected

`draft-schema` is heuristic. Always compare the chosen `source_column` against sampled examples from `dataset-summary.yaml`.

### Over-aggressive control mapping

Do not collapse ambiguous labels like `WT` to `ctrl` without dataset-specific evidence.

### High-cardinality metadata

Do not promote large free-text or near-unique columns into canonical vocab-driving fields unless they are truly part of the runtime contract.

### Unstable gene or token IDs

Do not reorder datasets during append and do not replace an existing persisted tokenizer with a differently ordered one.

### Missing source columns

`coalesce`, `join`, `template`, and `conditional` require all referenced source columns to exist. Missing columns raise actionable errors; fix the schema instead of weakening the mapping.

## Safe schema-finalization checklist

- Confirm every required canonical obs/var field is present.
- Review every heuristic note in `draft-schema.yaml` before copying it into `final-schema.yaml`.
- Check control-label candidates against sampled values, not just column names.
- Keep transform order intentional: cleanup first, harmonization second.
- Prefer `coalesce` over manual literals when complementary raw columns exist.
- Use `join` only when the combined string is the desired canonical value.
- Keep Ensembl cleanup and gene mapping decisions explicit.
- Make sure file names in your workflow are `draft-schema.yaml`, `final-schema.yaml`, `canonical-obs.parquet`, and `canonical-var.parquet`.
- Do not expect `canonicalize` to write `corpus-vocab.yaml`.
- If you rely on stable corpus-global gene IDs across appends, preserve or regenerate `gene-tokenizer.json` from canonical vars in corpus order.
