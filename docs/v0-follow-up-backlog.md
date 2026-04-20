# V0 Follow-Up Backlog

**Date**: 2026-04-20  
**Plan ID**: `copilot-plans-20260419-perturb-data-lab-v0`  
**Scope**: Prioritized list of follow-up work items that are out of scope for v0 but should be tracked for future phases.

---

## Priority 1 — Confirmatory Real-Data Benchmark (Medium Priority)

**What**: Run a confirmatory benchmark against real Phase 3 materialized outputs (brain + replogle Arrow/HF corpus) instead of synthetic random data.

**Why**: Current Phase 5 benchmark used synthetic random sparsity. Real h5ad data has structured sparsity, non-uniform gene distributions, and batch effects that could shift relative backend rankings.

**Acceptance**: Real-corpus benchmark confirms Arrow/HF remains the v0 default; results are recorded in the benchmark repo.

**Effort**: One Slurm GPU job reusing Phase 3 outputs and Phase 5 runner infrastructure.

---

## Priority 2 — WebDataset Production Hardening (Medium Priority)

**What**: Reduce WebDataset's 10–13× storage overhead relative to Arrow/HF through smarter shard packing (grouping cells with similar sparsity profiles) and evaluating compression options.

**Why**: WebDataset is the named alternate backend and acceptable when sequential streaming dominates. Reducing its storage footprint would make it more attractive for large corpora.

**Acceptance**: WebDataset storage footprint reduced to within 3× of Arrow/HF without degrading throughput.

---

## Priority 3 — Richer Ontology Handling (Medium Priority)

**What**: Beyond safe canonicalization and raw-field preservation (v0), implement ontology-aware field harmonization for `cell_type`, `perturbation_target`, and `treatment` fields across datasets from different labs/provenances.

**Why**: Current v0 canonical fields use literal string matching. Different labs use different naming conventions for the same biological entities (e.g., "K562" vs "K-562", "BRD4" vs "BRD4 (human)").

**Acceptance**: A second corpus join does not require manual renaming of every conflicting ontology term; instead, a mapping table drives the reconciliation.

---

## Priority 4 — Zarr/TensorStore Revisit (Lower Priority)

**What**: Re-evaluate Zarr/TS with a cell-group chunking strategy that reads batches of cells rather than individual cells, reducing per-cell random-access overhead.

**Why**: Zarr/TS's v0 exclusion was driven by severe wall-time scaling failure (8–9× slower than Arrow/HF). The chunking strategy in the current Phase 3 Zarr adapter was minimal. A properly chunked Zarr/TS could be competitive for semi-random access patterns.

**Acceptance**: A new Zarr/TS benchmark at matched chunking strategy shows wall times within 2× of Arrow/HF.

---

## Priority 5 — Distributed Training Optimization (Lower Priority)

**What**: Evaluate multi-node dataloader performance for Arrow/HF and WebDataset on datasets that exceed single-host memory.

**Why**: v0 is scoped to single-node training. Larger perturb-seq datasets (e.g., genome-scale CRISPR screens) may require multi-node distribution.

**Acceptance**: A multi-node benchmark confirms Arrow/HF Parquet sharding strategy works across 2–4 nodes without data shuffling bottleneck.

---

## Priority 6 — Additional Pilot Dataset Onboarding (Lower Priority)

**What**: Onboard additional datasets from `data/other_datasets/` to expand corpus coverage and test the onboarding workflow on more varied h5ad schemas.

**Why**: v0 validation used three pilots (Marson, Replogle, brain). Additional datasets would stress-test the inspector's heuristic field mapping and reveal edge cases.

**Acceptance**: Three additional datasets onboarded through the full Step 1–4 workflow with `materialization_readiness: pass`.

---

## Priority 7 — Feature Registry Namespace Versioning (Lower Priority)

**What**: Implement explicit namespace provenance in the feature registry and a versioning scheme for registry updates across corpus versions.

**Why**: Current v0 feature registry uses `"unknown"` namespace for new entries. A proper namespace + version scheme would make corpus provenance explicit and support reproducible tokenization across releases.

**Acceptance**: Feature registry entries include explicit `namespace`, `version`, and `provenance` fields; corpus index references a specific registry version.

---

## Deferred / Out of Scope for V0

These items are intentionally **not in v0** per the plan's scope definition and are not tracked in this backlog unless a future phase explicitly adopts them:

- Non-h5ad format support
- Full ontology harmonization beyond safe canonicalization
- Production multi-node distributed training
- Rebuilding external collators
- Final long-term removal of losing backends beyond v0 recommendation

---

## Backlog Summary Table

| Priority | Item | Rationale | Estimated Effort |
|----------|------|-----------|-----------------|
| 1 | Confirmatory real-data benchmark | Synthetic → real gap | 1 GPU job |
| 2 | WebDataset storage reduction | Alternate backend hardening | Medium |
| 3 | Ontology harmonization | Cross-lab interoperability | Medium |
| 4 | Zarr/TS revisit with chunking | Potential future backend | Medium |
| 5 | Distributed training eval | Scale-out readiness | High |
| 6 | Additional pilots onboarding | Workflow stress test | Medium |
| 7 | Feature registry versioning | Provenance reproducibility | Low |
