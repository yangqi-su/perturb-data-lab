# Phase 1 Aggregated Materialization Contract

## Status

- This document locks the Phase 1 contract for planned aggregated `lancedb` and aggregated `zarr` materialization in `perturb-data-lab`.
- It is a design contract only for this phase; later phases implement the backend and runtime changes.
- Current per-dataset backends and current runtime payloads are not rewritten in this phase.

## Locked Architecture Rules

- Keep `materialize first, canonicalize after`.
- Treat aggregated heavy backends as storage-only during materialization.
- Preserve dataset-local feature indices in every aggregated heavy row.
- Store `dataset_index` explicitly in every aggregated heavy row.
- Remove `release_id` from future runtime payloads and loader routing dependence.
- Preserve provenance/version outside the heavy runtime payload.
- Resolve global feature identity only after canonicalization.

## Backend-Agnostic Heavy Row Contract

The aggregated LanceDB and aggregated Zarr backends share one logical heavy-row contract.

| Field | Required | Type | Locked meaning |
| --- | --- | --- | --- |
| `global_row_index` | yes | integer | Corpus-stable append-only row id. Unique within one corpus path. |
| `dataset_index` | yes | integer | Corpus-stable integer id for the owning dataset append entry. |
| `local_row_index` | yes | integer | Original row position inside the dataset-local materialization order. |
| `expressed_gene_indices` | yes | `int32[]` | Dataset-local feature indices only. Same value across datasets has no cross-dataset meaning. |
| `expression_counts` | yes | `int32[]` | Integer counts aligned 1:1 with `expressed_gene_indices`. |
| `size_factor` | optional | float | Optional cached row scalar; not an identity field. |

### Required invariants

- `len(expressed_gene_indices) == len(expression_counts)` for every row.
- `expressed_gene_indices` stay in dataset-local feature space until canonicalization has produced a resolver.
- Heavy rows do **not** store `release_id`, global feature ids, token ids, feature labels, or canonical metadata.
- Heavy rows may co-store multiple datasets physically, but they remain logically blind to cross-dataset feature equivalence.

## Identity and Provenance Contract

### Runtime hot-path identity

- Runtime routing and heavy-row interpretation use `global_row_index`, `dataset_index`, and `local_row_index`.
- `dataset_index` is the only dataset identity field required inside heavy payloads.
- `release_id` must not be required to read, route, or decode aggregated heavy rows.

### Where provenance/version remains

`release_id` and related provenance/version semantics remain in non-hot-path artifacts such as:

- `materialization-manifest.yaml`
- `corpus-index.yaml` dataset membership / append ledger entries
- accepted schema copies
- raw cell / raw feature metadata artifacts
- feature provenance artifacts used by canonicalization
- later canonicalization-produced provenance tables, if needed

If runtime needs human-readable provenance, it should load that from metadata sidecars or ledgers rather than from each heavy row.

## Append-Safe Corpus Bookkeeping

`corpus-index.yaml` remains the authoritative append ledger. Aggregated backend sidecars may cache routing information later, but they are subordinate to the corpus index.

Each appended dataset must receive an immutable append entry containing at least:

- `dataset_index`
- `dataset_id`
- `release_id`
- `manifest_path`
- `cell_count`
- `global_row_start`
- `global_row_end`
- `local_feature_count`

Locked bookkeeping rules:

- `dataset_index` is assigned once per dataset append and never recycled within the corpus path.
- `global_row_index` values are append-only, contiguous, and never renumbered in place.
- Each dataset owns one contiguous range `[global_row_start, global_row_end)`.
- For aggregated backends, `global_row_index = global_row_start + local_row_index`.
- Backend-local rewrite, compaction, or re-chunking must not change prior `dataset_index` or `global_row_index` semantics.

## Boundary Between Materialization, Canonicalization, and Runtime Prep

### Materialization creates

- aggregated heavy rows with the locked fields above
- append-ledger updates in `corpus-index.yaml`
- dataset-local raw metadata artifacts
- accepted schema copies
- dataset-local feature-order / provenance artifacts
- optional row-local scalars such as `size_factor`

### Materialization must not create

- canonical global feature ids inside heavy rows
- token ids inside heavy rows
- cross-dataset feature equivalence assumptions
- runtime metadata tables that require canonicalization to be correct

### Canonicalization creates

- corpus-level canonical cell metadata
- corpus-level canonical feature metadata
- a resolver from `(dataset_index, local_feature_index)` to global feature identity
- any dataset-aware validity / translation artifacts derived from that resolver

### Canonicalization must not do

- rewrite historical heavy rows into canonical feature space
- require re-materializing already written heavy payloads just to express global identity

### Post-canonicalization runtime prep creates

- runtime metadata tables keyed by `global_row_index`
- runtime translation tables derived from the canonical resolver
- batch-aware routing / grouping sidecars for CPU-dense and GPU-sparse loaders

### Post-canonicalization runtime prep must not do

- restore `release_id` as a required runtime join key
- mutate the meaning of dataset-local heavy-row indices

## Locked Runtime-Facing Removals For Later Phases

The following current runtime-facing dependencies are locked for removal or demotion from routing responsibility in later phases:

- per-row aggregated payload dependence on `release_id`
- loader routing keyed by `release_id`
- runtime cell payload fields whose only role is hot-path provenance transport

In current code terms, later phases should move runtime identity away from `release_id`-centric surfaces such as:

- `CellState.dataset_release`
- `DatasetReaderEntry.release_id`
- `BackendCellReader.release_id`

These fields may remain as compatibility or provenance fields outside the hot path until the later refactor lands, but they are not part of the locked aggregated runtime contract.

## Explicit Divergence From The Archived Benchmark Suite

- No corpus-global canonical feature ids are required at write time.
- No centralized `metadata.parquet` is required before materialization can proceed.
- Aggregated heavy rows keep dataset-local sparse indices rather than canonical global indices.
- The global resolver is a post-canonicalization artifact, not a prerequisite for materialization.
- Runtime metadata tables and translation tables are downstream preparation artifacts, not ETL-time requirements.

## Phase 1 Completion Lock

Phase 1 is complete when later phases no longer need to re-decide:

- the mandatory aggregated heavy-row fields
- whether heavy payload indices remain dataset-local during materialization
- whether `dataset_index` is explicit in heavy payloads
- whether `release_id` is excluded from runtime payloads and loader dependence
- where provenance/version remains after that removal
- which responsibilities belong to materialization vs canonicalization vs post-canonicalization runtime prep
