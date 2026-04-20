# Phase 1 contract and repo blueprint

## Repo boundary

`perturb-data-lab` is the source of truth for dataset onboarding. It owns inspection, YAML review artifacts, typed runtime contracts, materialization logic, and training-facing loaders. `perturb-backend-benchmark` is a separate consumer that measures backend behavior against outputs produced by this repo.

## Repo-local layout

```text
perturb-data-lab/
├── docs/
├── examples/contracts/
├── src/perturb_data_lab/
│   ├── contracts/
│   ├── inspectors/
│   ├── materializers/
│   └── loaders/
└── tests/

perturb-backend-benchmark/
├── docs/
├── examples/
├── src/perturb_backend_benchmark/
│   ├── config/
│   ├── runners/
│   └── reports/
└── tests/
```

The benchmark repo must not duplicate inspection or materialization code; it consumes manifests and materialized outputs from `perturb-data-lab`.

## Canonical metadata rules

- canonical metadata is additive rather than lossy
- missing canonical values use the literal string `NA`
- raw source fields remain preserved unchanged alongside canonical columns
- count-bearing expression data must stay integer-only after audit; non-integer sources fail or quarantine later phases
- dataset-level and corpus-level metadata stay separate so immutable releases remain traceable

## Contract catalog

| Artifact | Scope | Review format | Runtime model | Purpose |
| --- | --- | --- | --- | --- |
| canonical perturbation metadata | cell | YAML | `CanonicalPerturbationFields` | stable perturbation fields for training and joins |
| canonical context metadata | cell | YAML | `CanonicalContextFields` | stable biological and technical context fields |
| feature registry | corpus | YAML | `FeatureRegistryManifest` | append-only feature ids and namespace provenance |
| dataset summary | dataset | YAML | `DatasetSummaryDocument` | lightweight inspection evidence |
| schema proposal | dataset | YAML | `SchemaProposalDocument` | proposed raw-to-canonical mappings |
| schema patch | dataset | YAML | `SchemaPatchDocument` | human-reviewed corrections or overrides |
| materialization manifest | release | YAML | `MaterializationManifest` | route, inputs, outputs, and provenance |
| corpus index | corpus | YAML | `CorpusIndexDocument` | immutable dataset release membership |
| global metadata | global | YAML | `GlobalMetadataDocument` | corpus defaults and contract pointers |

YAML is the human-editable review layer. Runtime code validates the same structures through strict typed objects before execution.

## Canonical field sets

### Perturbation metadata

- `perturbation_label`
- `perturbation_type`
- `target_id`
- `target_label`
- `control_flag`
- `dose`
- `dose_unit`
- `timepoint`
- `timepoint_unit`
- `combination_key`

### Context metadata

- `dataset_id`
- `dataset_release`
- `cell_context`
- `cell_line_or_type`
- `species`
- `tissue`
- `assay`
- `condition`
- `batch_id`
- `donor_id`
- `sex`
- `disease_state`

## Feature registry expectations

- corpus-scoped and append-only
- records namespace provenance for every feature id source
- preserves stable token ids across later dataset joins
- stores release and registry version pointers in global metadata

## Backend comparison rubric

The benchmark repo ranks candidate backends on:

- build cost
- read throughput
- worker scaling at 1, 4, and 8 workers
- random access behavior
- sequential streaming behavior
- storage footprint
- join complexity for `create_new`, `append_monolithic`, and `append_routed`

The default v0 backend should be the option that most reliably feeds downstream training without forcing the data contract to become backend-specific.
