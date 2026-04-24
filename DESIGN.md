# perturb-data-lab Design Narrative

## Status

This document is the architecture narrative for the next cleanup and rebuild of `perturb-data-lab`.

It is intentionally not a concrete implementation plan. Instead, it defines:

- what the system is supposed to do
- where the current codebase is structurally confused
- what the target data flow should be
- how backend format and corpus topology should be separated
- how the cleanup should be understood through the 3(4)-step sequence

The 3(4)-step sequence is:

1. inspection and materialization approval
2. materialization and corpus registration
3.1 simple dataloading
3.2 full dataloading methods

The goal is to make later detailed plans easy to derive from one stable design story.

## Problem Statement

The current repository has useful pieces, but the end-to-end system is difficult to reason about because several architectural generations are visible at once.

The main symptoms are:

- inspection is doing both lightweight evidence gathering and heavy schema drafting
- materialization is partly tokenizer-free and partly still surrounded by tokenizer-era artifacts
- corpus runtime loading uses some materialized outputs directly and ignores some corpus-level canonicalization outputs
- feature identity, token identity, and dataset-local feature order are not cleanly separated
- backend storage format and corpus topology are mixed together in backend names and reader/writer logic
- some code remains only for tests, backward compatibility, or unfinished migration paths

The result is not just bugs. The result is that the intended data flow is hard to describe in one sentence.

This document fixes that by defining one clean narrative.

## Core Principles

The redesigned system should follow these principles.

### Materialize first, canonicalize later

Materialization should not depend on a finalized cross-dataset metadata contract or a finalized shared feature vocabulary.

Materialization should preserve dataset-local truth and produce durable artifacts that can later support canonicalization.

### Counts first

The first hard requirement is always the expression matrix:

- find a valid integer count source
- preserve sparse structure
- write counts efficiently
- keep feature indices dataset-local at write time

Everything else is secondary to getting counts into a correct durable form.

### Preserve raw dataset truth

The system should preserve:

- raw `obs`
- raw `var`
- source path and source release
- chosen count source
- dataset-local feature ordering

This preserved raw truth is what later canonicalization and debugging should use.

### Separate storage format from corpus topology

Storage backend and corpus organization are different concerns.

Storage backend answers: how are heavy rows written and read?

Corpus topology answers: is each dataset stored independently or are all datasets appended into one shared heavy object?

These must not be fused into one conceptual axis.

### Keep runtime identity simple

The stable runtime identity should be:

- `global_row_index`
- `dataset_index`
- `dataset_id`
- `local_row_index`

`release_id` is provenance, not a hot-path join key.

### Canonicalization is additive, not destructive

Canonicalization should add:

- canonical cell metadata
- canonical feature metadata
- optional dataset-local to global feature mappings

It should not rewrite the meaning of already-materialized heavy rows.

## Target End-to-End Story

The system should be easy to explain in one end-to-end story.

### Step 1: inspect a dataset

A user points the system at one `.h5ad` file.

The inspection layer reads the file in a lightweight mode and produces a compact review artifact that answers:

- what are the dataset dimensions
- what does `obs` look like
- what does `var` look like
- which matrix candidates exist among `.X`, `.raw.X`, and `.layers[...]`
- whether a direct integer count source exists
- whether a recoverable count source exists
- whether the file is approved for materialization

This stage is evidence gathering and approval gating only.

### Step 2: materialize the dataset

Once approved, materialization writes the dataset into the chosen backend and registers it in a corpus.

Materialization should produce:

- heavy count storage in the chosen backend
- raw dataset metadata sidecars
- raw feature metadata sidecars
- dataset-local HVG artifacts
- a proposed canonicalization schema for later review and refinement
- a per-dataset materialization manifest
- corpus ledger updates

This is the moment the dataset becomes durable and loadable.

### Step 3.1: simple dataloading

After materialization, the system should support the easiest useful loading modes across all supported backends.

This stage should provide:

- random row sampling
- dataset-restricted row sampling
- metadata/context-grouped row sampling
- sparse batch collation

This stage should not require global feature identity to be settled.

### Step 3.2: full dataloading methods

After the simple loading path is stable, the system should support all intended gene-level and feature-level sampling modes.

This includes:

- random feature-context sampling
- expressed-plus-zero sampling
- HVG-based feature sampling
- any canonicalized/global-feature-space loading path that is needed for tokenization or shared-vocabulary training

This is the stage where feature-space semantics become explicit and enforced.

## The 3(4)-Step Architecture

The rest of this document expands each of those stages as architecture, not as a task list.

## Stage 1: Inspection And Materialization Approval

### Purpose

Stage 1 exists to answer a narrow question:

"Can this dataset be materialized safely, and from which count source?"

It is not responsible for final canonical metadata design.

### Inputs

- one raw `.h5ad` path
- a dataset identifier
- optional source release string

Large files should be inspected on Slurm CPU in the `torch_flashv3` environment, following repository execution rules.

### Outputs

The stage should emit a small dataset review bundle containing:

- dimensions summary
- `obs` inventory and summary
- `var` inventory and summary
- count-source candidate audit
- selected count source decision
- approval state

The key point is that the primary output is an approval artifact, not a full canonical contract.

### What this stage should decide

This stage should decide:

- whether `.X`, `.raw.X`, or a layer is the selected source
- whether the selected source is direct integer
- whether recovery is needed and acceptable
- whether the dataset is ready for materialization

### What this stage should not decide

This stage should not decide:

- final corpus-global feature identity
- final token IDs
- final cross-dataset metadata harmonization
- final training emission contract

### Why this stage must remain small

The repository currently contains too much coupling between inspection and downstream schema machinery.

The cleanup direction should make Stage 1 easier to trust by keeping it small, evidence-driven, and approval-oriented.

### Reference datasets for this stage

The first validation targets for this architecture are:

- `dummy_data/*`
- `perturb/marson2025_data/D1_Rest.assigned_guide.h5ad`

The dummy files prove the path on controlled examples.

The Marson file proves the path on a real large dataset with realistic metadata complexity.

## Stage 2: Materialization And Corpus Registration

### Purpose

Stage 2 is the durable ingestion step.

After Stage 2, a dataset should exist as a stored object with enough artifacts to support:

- immediate basic loading
- later canonicalization
- later debugging and provenance review

### Inputs

Stage 2 takes:

- the raw `.h5ad`
- an approved count-source decision from Stage 1
- an output location
- a chosen backend
- a chosen corpus topology
- either a new corpus request or an append-to-existing-corpus request

### Core responsibilities

Stage 2 must do five things.

#### 1. Write counts correctly

This is the primary job.

Counts must be:

- integer
- sparse when possible
- written in dataset-local feature space
- recoverable by row without metadata re-execution

Heavy rows should carry only runtime-critical count-side information.

For aggregate topologies, the logical heavy-row contract is:

- `global_row_index`
- `dataset_index`
- `local_row_index`
- `expressed_gene_indices`
- `expression_counts`
- optional `size_factor`

For federated topologies, the per-dataset object can omit corpus-global fields internally, but the loader-facing interface should still be able to produce the same runtime identity once routed through the corpus ledger.

#### 2. Preserve raw metadata

Stage 2 must preserve enough raw information to rebuild or revise metadata decisions later.

At minimum this means preserving:

- raw `obs`
- raw `var`
- dataset-local feature order
- chosen count source
- source provenance

The current repository already points in this direction, and that direction should be retained.

#### 3. Save feature-side dataset artifacts

Feature-side artifacts should be simple and durable.

The preferred design direction is:

- save dataset-local feature metadata directly from `adata.var`
- preserve the dataset-local feature index order explicitly
- compute HVGs after counts are settled
- save HVGs in dataset-local feature space

This is intentionally simpler than the current mixture of feature registry, token sidecars, and canonicalization-era feature outputs.

#### 4. DO NOT Produce a proposed canonicalization schema

No lightweight canonicalization schema

#### 5. Update the corpus tracking object

The corpus needs a single authoritative tracking object.

This object should act as a ledger recording:

- corpus identity
- datasets present in the corpus
- dataset append order
- dataset index assignment
- backend choice
- topology choice
- manifest locations
- cell counts
- global row ranges when relevant

The corpus tracking object should not become an over-smart mutable metadata rewrite surface.

It should remain the routing ledger.

### Backend and topology must be independent

The current combined backend names should be conceptually replaced by a matrix.

### Storage backend axis

The intended storage backends are:

- `arrow-parquet`
- `arrow-ipc`
- `webdataset`
- `zarr`
- `lance`

This axis answers:

- how rows are physically written
- how rows are read
- how append works
- what access patterns are efficient

### Corpus topology axis

The intended corpus topologies are:

- `federated`
- `aggregate`

This axis answers:

- does each dataset have its own heavy object and routing entry
- or do multiple datasets share one heavy object with append-safe row ranges

### Why this split matters

This split removes a major source of current bloat.

The code should not need one conceptual backend name for each pair such as:

- `lancedb-aggregated`
- `zarr-aggregated`
- `arrow-hf`

Instead, the system should support combinations where they make sense.

Examples:

- `arrow-parquet` x `federated`
- `arrow-ipc` x `federated`
- `webdataset` x `federated`
- `zarr` x `federated`
- `lance` x `aggregate`

Not every backend must support every topology immediately.

The design only requires that the axes be separated.

### What Stage 2 should not do

Stage 2 should not require:

- a finalized cross-dataset feature vocabulary
- finalized token IDs
- finalized corpus-global feature equivalence
- a canonicalized metadata contract to be fully settled

This is the most important simplification in the new architecture.

## Stage 3.1: Simple Dataloading

### Purpose

Stage 3.1 exists to make all backends and supported topologies immediately usable after materialization.

This is the easiest loader milestone and should be achieved before any feature-space-heavy work.

### Required capabilities

Stage 3.1 should provide:

- corpus-global row routing
- single-row random access where the backend allows it
- dataset iteration
- random row sampling
- dataset-restricted row sampling
- metadata/context-grouped row sampling
- sparse batch collation

### Loader-facing data model

The loader layer should consume:

- the corpus tracking object
- per-dataset materialization manifests
- backend-specific heavy objects
- metadata sidecars that were written during materialization

The runtime-facing row identity should remain:

- `global_row_index`
- `dataset_index`
- `dataset_id`
- `local_row_index`

### Metadata loading model

The simplest working runtime should use:

- lazy heavy-row reads from the backend
- a lightweight RAM-resident metadata table when that improves grouping and batch routing

This keeps counts I/O and metadata I/O clearly separated.

### What Stage 3.1 should not require

Stage 3.1 should not require:

- global feature IDs
- token IDs
- canonicalized corpus-wide feature resolution
- advanced gene sampling semantics

If the system cannot do simple row sampling on every backend after materialization, then more advanced loader work should not begin yet.

### Efficiency direction

The phrase "as efficient as possible" in this stage means:

- exploit backend-native grouped reads where possible
- keep metadata reads off the heavy I/O hot path when possible
- batch by owning dataset before issuing heavy reads
- collate sparse rows without densifying

The design target is not maximum theoretical optimization yet. The design target is backend-wide correctness with obviously sensible access patterns.

## Stage 3.2: Full Dataloading Methods

### Purpose

Stage 3.2 settles the full training-facing loader semantics.

This is where the system must support all intended sampling methods correctly and unambiguously.

### Required capabilities

Stage 3.2 should provide every supported sampling method, including:

- random feature-context sampling
- expressed-plus-zero sampling
- HVG-based feature sampling
- any other training-time feature subsampling modes the project needs

### The crucial design question

Stage 3.2 must explicitly define feature space.

Every advanced sampling method must say whether it operates in:

- dataset-local feature space
- canonicalized/global feature space

This choice cannot remain implicit.

### Local-space loading

Local-space loading is:

- simpler
- available immediately after materialization
- useful for dataset-specific training or debugging

Its limitation is that feature identity is not shared across datasets.

### Global-space loading

Global-space loading is:

- needed for shared-vocabulary or tokenization-like training
- dependent on canonicalization or an equivalent global mapping step
- more complex but semantically stronger across datasets

### HVGs must follow feature space explicitly

HVG artifacts should be written in dataset-local feature space during materialization.

If a global/canonicalized loader is later needed, then HVGs must be transformed into that feature space using an explicit mapping artifact.

The system must not silently mix local feature indices with global feature indices.

### Canonicalization's real role in Stage 3.2

Canonicalization becomes important here, not earlier.

If the project wants multi-dataset shared feature identity, canonicalization should produce artifacts that make this explicit, such as:

- canonical cell metadata
- canonical feature metadata
- dataset-local to global feature mapping tables

Those artifacts should feed the advanced loaders cleanly.

They should not be loosely present while the loader continues to depend on older sidecars.

## Role Of Canonicalization In The Overall System

Canonicalization is important, but it should not be allowed to distort the main data flow.

Its role is:

- harmonize metadata across datasets
- harmonize feature identity across datasets when needed
- generate optional shared-feature loading artifacts

Its role is not:

- to be required before any useful loading exists
- to rewrite historical heavy-row semantics
- to become the place that silently changes corpus routing rules

The target architecture therefore treats canonicalization as a downstream enrichment layer.

Basic materialization and basic loading must still stand on their own.

## Corpus Tracking Object

The design assumes one authoritative corpus tracking object.

Today this is naturally closest to `corpus-index.yaml`, but the exact file format can evolve.

What matters is the role.

The corpus tracking object should record:

- corpus ID
- backend
- topology
- ordered dataset membership
- dataset index
- materialization manifest path
- cell count
- global row range when relevant

It may also point to optional corpus-level artifacts, but it should remain first and foremost the routing and membership ledger.

The system should avoid allowing later metadata passes to accidentally erase or redefine routing-critical fields.

## What Should Be Simplified Or Removed Over Time

This document is not a deletion list, but the architecture does imply a cleanup direction.

The likely simplification targets are:

- tokenizer-era scaffolding that is no longer part of the chosen design
- feature registry logic if raw `var` plus later canonicalization fully replaces it
- duplicated contract models that split one concept across inspection and materialization packages
- backend names that combine storage format and topology into one label
- loader fallback paths that exist only because the system is supporting incompatible generations of artifacts at the same time

The cleanup rule is simple:

keep code that participates in the target data flow, and remove code that only preserves abandoned architecture branches.

## Architecture Summary

The target architecture can be summarized as follows.

### Stage 1

Inspect one `.h5ad`, summarize dimensions and metadata, audit count sources, and produce a materialization approval artifact.

### Stage 2

Materialize integer counts into a chosen backend and topology, preserve raw metadata and feature metadata, compute HVGs, emit a proposed canonicalization scaffold, and append the dataset to the corpus ledger.

### Stage 3.1

Make all backends loadable for basic row-level and batch-level access with simple sampling and sparse batch collation.

### Stage 3.2

Settle feature-space semantics and make all advanced dataloading methods work correctly, including HVG-aware and global-feature-aware paths where needed.

## Final Design Intent

`perturb-data-lab` should become easy to explain again.

The intended sentence is:

"Inspect a dataset to approve its count source, materialize it into a backend while preserving raw metadata and feature truth, register it in a corpus ledger, and then load it through simple or advanced dataloaders depending on whether only local feature space or full canonicalized global feature space is required."

If later implementation plans do not reinforce that sentence, they are pushing the repository away from this design.
