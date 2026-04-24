# Phase 4 Runtime Path Adaptation

## Goal

- adapt the benchmark-suite CPU-dense and GPU-sparse runtime ideas onto the Phase 3 loader boundary split in `perturb-data-lab`
- keep runtime identity keyed by `global_row_index`, `dataset_index`, and `dataset_id`
- keep `release_id` out of runtime-hot-path identity and feature resolution

## Adopted Directly from the Benchmark Suite

- RAM-resident metadata table keyed by contiguous `global_row_index`
- dataset-aware grouped batch sampling using metadata-derived grouping keys rather than per-record routing in the hot path
- flat sparse batch payloads represented as:
  - concatenated `expressed_gene_indices`
  - concatenated `expression_counts`
  - `row_offsets` delimiting each row
- split runtime paths:
  - CPU baseline that may densify after feature resolution
  - sparse-first path that keeps flat sparse payloads plus offsets through the hot path

## Rewritten for `perturb-data-lab`

- feature resolution is post-canonicalization and dataset-aware; it is not assumed at materialization time
- the resolver contract is `dataset_index + local_feature_index -> global feature id`
- `CorpusLoader` remains the orchestrator for grouped row reads and metadata joins
- dataset/context grouping comes from `MetadataTable.canonical_context` instead of a centralized benchmark metadata parquet contract
- CPU-dense and GPU-sparse helpers operate on `CellState` / `SparseBatchPayload` abstractions rather than benchmark-suite `SparseCellRecord`

## Runtime Contracts

### Metadata Table

- `MetadataTable` remains the RAM-resident source of per-row metadata keyed by `global_row_index`
- it now also exposes:
  - per-dataset row index lists
  - dataset+context grouping keys for batch samplers

### Global Feature Resolver

- `GlobalFeatureResolver` is the post-canonicalization integration point
- it holds per-dataset mapping arrays from dataset-local feature indices to global feature ids
- current implementation can build the resolver from preloaded token-mapping sidecars when they exist
- later optimization can swap in cached canonicalization outputs without changing runtime-path callers

### CPU-Dense Path

- `CPUDenseRuntimePath` resolves sparse rows to global feature ids first
- it densifies only after resolution
- this is the correctness-first baseline and intentionally not the hot-path optimization target

### GPU-Sparse Path

- `GPUSparseRuntimePath` resolves rows into flat global sparse payloads plus offsets
- it avoids constructing full dense vectors in the hot path
- sampled-count lookup uses per-row sorted sparse ids with `searchsorted`-style matching semantics
- this keeps the path compatible with later probability-mask-driven feature sampling without changing the batch payload contract

## Known Limitations

- `CorpusLoader.build_global_feature_resolver()` currently requires per-dataset token sidecars or another explicit mapping source; corpora without those mappings must provide a resolver explicitly
- current GPU-sparse helpers are numpy-based semantic runtime adapters, not CUDA kernels
- WebDataset still lacks the richer RAM-metadata-table parity needed for full dataset-context grouped semantics in all corpus constructions

## Follow-up Direction (Later Phases, Not Implemented Here)

- persist canonicalization-derived dataset-local-to-global mapping tables as dedicated runtime artifacts
- move probability-mask-driven sampling onto the sparse runtime path without reintroducing densification
- add bounded equivalence smoke tests between CPU-dense and GPU-sparse paths on representative multi-dataset batches
