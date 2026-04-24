# Phase 2 Aggregated Backend Mapping

## Status

- This document locks the Phase 2 backend-mapping design for adapting benchmark aggregated `lancedb` and aggregated `zarr` ideas into `perturb-data-lab`.
- Phase 1 contract remains unchanged and authoritative.
- This phase defines migration semantics only; it does not implement later runtime refactors or canonicalization outputs.

## Phase 2 Decisions

- Keep current per-dataset backends in parallel:
  - `arrow-hf`
  - `zarr-ts`
  - `webdataset`
- Introduce new corpus-level aggregated backend names rather than overloading existing names:
  - `lancedb-aggregated`
  - `zarr-aggregated`
- Treat benchmark aggregated writers as structural starting points only.
- Do not transplant benchmark `metadata.parquet`, `feature-registry.parquet`, mask matrices, or `corpus-routing.json` into materialization-time `perturb-data-lab`.
- Keep `corpus-index.yaml` as the authoritative append ledger; aggregated-backend sidecars are subordinate caches and integrity records.

## Backend-by-Backend Migration Map

| Surface | Benchmark aggregated Zarr | Benchmark aggregated LanceDB | Current `perturb-data-lab` `zarr-ts` | Current `perturb-data-lab` `arrow-hf` | Phase 2 adaptation decision |
| --- | --- | --- | --- | --- | --- |
| Physical scope | One corpus-global hierarchy | One corpus-global Lance dataset | One dataset per `release_id` | One dataset per `release_id` | Aggregated backends stay corpus-global; current backends stay dataset-local |
| Heavy-row key | `global_row_index` | `global_row_index` | implicit per-dataset cell index inside chunked stores | implicit row order inside parquet | Aggregated rows must carry `global_row_index`, `dataset_index`, and `local_row_index` explicitly |
| Feature space at write time | canonical global indices | canonical global indices | dataset-local indices | dataset-local indices | Keep dataset-local indices for both aggregated backends per Phase 1 |
| Heavy schema | `global_row_index`, indices, counts | `global_row_index`, indices, counts | release-named indices/counts/sf stores plus `meta.json` | per-cell parquet plus canonical cell SQLite | Aggregated schema extends benchmark writers with `dataset_index`, `local_row_index`, optional `size_factor`; no canonical metadata in heavy rows |
| Metadata dependency | requires centralized `metadata.parquet` | requires centralized `metadata.parquet` | embeds per-cell canonical metadata in `meta.json` | writes per-dataset canonical metadata SQLite | Do not require corpus-wide canonical metadata at materialization time; keep metadata dataset-local until canonicalization/runtime prep |
| Feature registry dependency | requires `feature-registry.parquet` at ETL time | requires `feature-registry.parquet` at ETL time | optional dataset-local feature registry | dataset-local feature metadata / provenance | Do not require corpus-global feature registry before canonicalization |
| Append model | single writer appends flat arrays and row offsets | versioned dataset append | overwrite-style per-dataset write | overwrite-style per-dataset write | Add append-safe corpus reservation + commit semantics for one dataset at a time |
| Dataset routing sidecar | `dataset-offsets.parquet` + `corpus-routing.json` | `dataset-offsets.parquet` + `corpus-routing.json` | none beyond manifest/corpus index | corpus index + manifest | Move routing authority to `corpus-index.yaml`; sidecars only record backend-local commit evidence |
| Release naming | dataset ids in sidecars, not heavy rows | dataset ids in sidecars, not heavy rows | `release_id` embedded in file names and metadata | `release_id` embedded in file names and metadata | Aggregated heavy rows must not depend on `release_id`; provenance stays in manifests and corpus index |
| Reuse level | layout pattern reusable | append/write pattern reusable | sparse dataset-local extraction reusable | sparse row extraction + metadata separation reusable | Reuse sparse row extraction and aggregated storage shapes, but not benchmark split-brain canonical assumptions |

## Locked Aggregated Backend Names

### Keep current names for current semantics

- `arrow-hf` remains the per-dataset parquet + SQLite backend.
- `zarr-ts` remains the per-dataset Zarr/TensorStore backend with release-scoped stores.
- `webdataset` remains unchanged for this phase.

### Introduce new names for new semantics

- `lancedb-aggregated` means one corpus-scoped Lance dataset storing mixed-dataset heavy rows.
- `zarr-aggregated` means one corpus-scoped Zarr hierarchy storing mixed-dataset heavy rows.

### Why new names are required

- Current `zarr-ts` and proposed aggregated Zarr have different write scope, metadata placement, and loader assumptions.
- `arrow-hf` is a per-dataset federated backend, not a Lance-backed aggregated backend.
- Current manifest validation and loader dispatch assume one backend name corresponds to one stable storage contract.
- Reusing `zarr-ts` for aggregated storage would make existing corpora and readers ambiguous.

## Aggregated Heavy-Row Schema Mapping

Both aggregated backends should implement the same logical row fields:

- `global_row_index: int64`
- `dataset_index: int32`
- `local_row_index: int64`
- `expressed_gene_indices: list<int32>` or flat `int32` payload addressed by offsets
- `expression_counts: list<int32>` or flat `int32` payload addressed by offsets
- optional `size_factor: float32/float64`

Explicitly not written into aggregated heavy rows during materialization:

- `release_id`
- `dataset_id`
- canonical perturbation/context metadata
- canonical feature ids
- token ids
- corpus-wide feature registry references

## Common Append Semantics

Each one-dataset-at-a-time append should follow the same logical flow for both aggregated backends:

1. Read current `corpus-index.yaml` and derive the next immutable `dataset_index`, `global_row_start`, and `global_row_end` in memory.
2. Materialize the dataset-local sparse payload using dataset-local feature indices only.
3. Write the dataset rows into the aggregated heavy backend using the reserved `dataset_index` and global range.
4. Validate that backend row counts and range endpoints match the planned append.
5. Write or update the dataset-local `materialization-manifest.yaml`.
6. Append the new dataset entry to `corpus-index.yaml`.
7. Mark the backend sidecar append record as committed.

If the heavy-store append fails, `corpus-index.yaml` must not be updated.

If `corpus-index.yaml` update fails after the heavy-store append succeeds, the backend sidecar must preserve enough commit evidence to repair or reconcile the corpus index later.

## Aggregated LanceDB Append Design

### What is reused from the benchmark writer

- Arrow-table batch construction
- `lance.write_dataset(..., mode="append")` append pattern
- dataset-version inspection for post-write validation

### What changes for `perturb-data-lab`

- add `dataset_index`, `local_row_index`, and optional `size_factor` to the heavy schema
- keep `expressed_gene_indices` in dataset-local feature space
- remove dependence on benchmark `metadata.parquet`
- remove dependence on benchmark `feature-registry.parquet`
- stop assuming a pre-canonicalized global gene vocabulary

### Append rule

- One dataset append becomes one Lance append transaction over Arrow batches.
- The aggregated Lance path should be stable for the corpus, e.g. one shared `aggregated-corpus.lance` under the corpus matrix root.
- Post-append validation should confirm:
  - row-count delta equals `cell_count`
  - first and last appended `global_row_index` values match the reserved range
  - appended rows all carry the reserved `dataset_index`
- The append sidecar should record at least the committed Lance version number for that dataset append.

### Minimal Lance sidecar requirement

- one append log entry per dataset append with:
  - `dataset_index`
  - `dataset_id`
  - `release_id`
  - `global_row_start`
  - `global_row_end`
  - `cell_count`
  - `lance_version`
  - `status: pending|committed`

## Aggregated Zarr Append Design

### What is reused from the benchmark writer

- one corpus-global hierarchy
- flat sparse payload arrays plus `row_offsets`
- append-by-extending arrays rather than rewriting prior datasets

### What changes for `perturb-data-lab`

- add `dataset_index`, `local_row_index`, and optional `size_factor` arrays
- keep `expressed_gene_indices` in dataset-local feature space
- remove benchmark canonical-gene-count assumptions and `global_gene_index` translation
- do not write centralized metadata tables at materialization time

### Required aggregated Zarr arrays

- `row_offsets`
- `global_row_index`
- `dataset_index`
- `local_row_index`
- `expressed_gene_indices`
- `expression_counts`
- optional `size_factor`

### Append rule

- The shared Zarr root is corpus-scoped and append-only.
- Each dataset append extends all row-level arrays in lockstep.
- `row_offsets` remains length `total_rows + 1` and is the primary sparse payload boundary record.
- Existing rows must never be rewritten or renumbered during normal append.

### Why Zarr needs a stronger sidecar than LanceDB

- LanceDB appends are versioned transactions.
- Zarr append touches multiple arrays and is not inherently transactional across them.

### Minimal Zarr sidecar requirement

- one append log entry per dataset append with:
  - `dataset_index`
  - `dataset_id`
  - `release_id`
  - `global_row_start`
  - `global_row_end`
  - `cell_count`
  - `row_offsets_stop`
  - `flat_nnz_stop`
  - `status: pending|committed`

This sidecar is the crash-recovery aid for multi-array append integrity. It remains subordinate to `corpus-index.yaml` for dataset ownership and routing.

## `corpus-index.yaml` and Aggregated Sidecar Cooperation

## Authoritative in `corpus-index.yaml`

- `dataset_index`
- `dataset_id`
- `release_id`
- `manifest_path`
- `cell_count`
- `global_row_start`
- `global_row_end`
- declared corpus backend

## Cached or integrity-only in aggregated sidecars

- shared backend root path
- backend-specific append evidence (`lance_version`, `row_offsets_stop`, `flat_nnz_stop`)
- append status markers (`pending`, `committed`)
- optional chunking / writer settings used to create the shared store

## Cooperation rule

- Runtime and canonicalization must derive dataset ownership and row routing from `corpus-index.yaml`, not from backend-local append logs.
- Aggregated sidecars may be consulted to locate the shared heavy store quickly and to detect incomplete appends.
- If sidecar contents disagree with `corpus-index.yaml`, the corpus index is the source of truth for ownership and the sidecar is treated as stale or in need of repair.

## Minimal Pre-Canonicalization Sidecar Set

Before canonicalization, aggregated backends require only:

1. existing per-dataset `materialization-manifest.yaml`
2. existing raw metadata and feature-provenance artifacts already produced by current materialization
3. `corpus-index.yaml` with explicit `dataset_index` and global row range fields
4. one aggregated-backend append sidecar for backend-local commit evidence

Not required before canonicalization:

- corpus-wide canonical metadata
- corpus-wide `metadata.parquet`
- corpus-wide feature registry mapping local indices to global ids
- probability masks or HVG mask matrices
- runtime batch-grouping tables

## Concrete Mapping From Current `perturb-data-lab` Patterns

### From `arrow-hf`

- Keep the good separation between heavy expression payload and metadata artifacts.
- Reuse sparse row extraction utilities and dataset-local manifest/provenance discipline.
- Do not copy the current Arrow/HF per-dataset canonical cell SQLite into aggregated backends.

### From `zarr-ts`

- Keep sparse integer payload storage and optional `size_factor` persistence.
- Drop release-scoped file naming and per-cell `meta.json` storage for aggregated mode.
- Replace dataset-local chunk addressing with corpus-level append addressing.

### From the benchmark aggregated writers

- Reuse the idea that LanceDB and Zarr can act as heavy-store-only aggregated payload backends.
- Do not reuse the assumption that heavy rows already live in a canonical feature space.
- Do not reuse the split-brain requirement for ETL-time `metadata.parquet` or `feature-registry.parquet`.

## Implementation Boundary Locked By Phase 2

Later phases may implement:

- manifest model expansion for new backend names
- `DatasetJoinRecord` expansion so `dataset_index` is explicit in `corpus-index.yaml`
- new aggregated backend writers and readers
- runtime metadata and feature-resolution sidecars after canonicalization

Later phases must not re-decide:

- whether aggregated backends keep dataset-local indices at write time
- whether aggregated backends use new backend names
- whether `corpus-index.yaml` remains the routing authority
- whether aggregated sidecars stay minimal and subordinate before canonicalization
