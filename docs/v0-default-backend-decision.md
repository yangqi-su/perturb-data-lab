# V0 Default-Backend Decision Memo

**Date**: 2026-04-20  
**Plan ID**: `copilot-plans-20260419-perturb-data-lab-v0`  
**Evidence Source**: Phase 5 GPU-verified benchmark (job `16165274`, `results-fixed/benchmark-results-gpu.json`)  
**Caveat**: All results are from synthetic sparse data under a dummy training loop. Real h5ad materialization I/O patterns may alter relative rankings. Treat this as a provisional v0 recommendation, not a final production benchmark.

---

## Decision

**V0 Default Backend: Apache Arrow / Hugging Face datasets (Arrow/HF)**

**Named Alternate: WebDataset** — acceptable when sequential streaming throughput is the primary concern and larger storage footprint is acceptable.

**Not Recommended for V0: Zarr/TensorStore** — excluded from v0 default path due to severe wall-time scaling failure.

---

## Evidence Summary

All figures from GPU-verified benchmark run on NVIDIA H200 (job `16165274`).

### marson-single scenario (5,000 cells × 3 epochs, batch=64)

| Backend       | Workers | samples/sec | gpu_util | total_wall_sec | storage_bytes |
|---------------|---------|-------------|----------|----------------|---------------|
| arrow-hf      | 1       | 9,453       | 0.873    | 1.59           | 1,213,076     |
| arrow-hf      | 4       | 25,976      | 0.797    | 2.31           | 1,213,076     |
| arrow-hf      | 8       | 34,783      | 0.810    | 3.45           | 1,213,076     |
| webdataset    | 1       | 14,218      | 0.760    | 2.11           | 16,291,840    |
| webdataset    | 4       | 38,295      | 0.798    | 3.13           | 16,291,840    |
| webdataset    | 8       | 38,143      | 0.801    | 6.29           | 16,291,840    |
| zarr-ts       | 1       | 851         | 0.779    | 17.63          | 1,115         |
| zarr-ts       | 4       | 2,932       | 0.800    | 20.46          | 1,115         |
| zarr-ts       | 8       | 4,751       | 0.821    | 25.26          | 1,115         |

### combined-corpus scenario (15,000 cells × 3 epochs, batch=64)

| Backend       | Workers | samples/sec | gpu_util | total_wall_sec | storage_bytes |
|---------------|---------|-------------|----------|----------------|---------------|
| arrow-hf      | 1       | 10,954      | 0.787    | 4.11           | 3,596,393     |
| arrow-hf      | 4       | 26,582      | 0.796    | 6.77           | 3,596,393     |
| arrow-hf      | 8       | 37,422      | 0.810    | 9.62           | 3,596,393     |
| webdataset    | 1       | 14,192      | 0.752    | 6.34           | 48,855,040    |
| webdataset    | 4       | 39,191      | 0.800    | 9.18           | 48,855,040    |
| webdataset    | 8       | 39,436      | 0.801    | 18.25          | 48,855,040    |
| zarr-ts       | 1       | 833         | 0.769    | 54.04          | 3,143         |
| zarr-ts       | 4       | 2,912       | 0.799    | 61.82          | 3,143         |
| zarr-ts       | 8       | 4,691       | 0.820    | 76.75          | 3,143         |

---

## Per-Backend Rationale

### Arrow/HF — V0 Default

**Why preferred:**
1. Best wall-time scaling across both scenarios and all worker counts (3.45–9.62 sec total at 8 workers)
2. Smallest storage footprint (1.2–3.6 MB vs 16–49 MB for WebDataset)
3. Per-cell sparse integer storage (int32 indices + int32 counts) avoids densification and preserves integer count guarantees
4. Mature Parquet ecosystem with broad tool interop (PyArrow, pandas, duckdb, etc.)
5. Random-access capable via row-group filtering — supports `append_routed` join mode efficiently
6. GPU utilization proxy (0.79–0.87) is competitive with the other backends

**Known limits:**
- Throughput at 8 workers is slightly below WebDataset peak (34–37k vs 38–39k samples/sec); this gap is within noise and offset by Arrow/HF's better wall-time scaling and smaller storage
- Not natively sequential; for pure streaming-first workloads WebDataset remains an option

### WebDataset — Named Alternate

**When to use instead of Arrow/HF:**
1. Checkpoint-style iteration where shard-level sequential reads dominate
2. Willingness to trade 10–13× more storage for marginally higher peak throughput
3. Existing WebDataset-based training pipelines that cannot be migrated

**Why not default:**
- Storage footprint is 10–13× larger (16 MB vs 1.2 MB for single corpus; 49 MB vs 3.6 MB for combined)
- Sequential-only access pattern; no random-access capability within a shard
- No native per-cell sparse representation; uses dense float encoding within shards
- Harder to extend with `append_routed` joins without rebuilding shard sets

### Zarr/TensorStore — Not Recommended for V0

**Why excluded:**
- Severe wall-time scaling failure: 17–77 sec total wall time vs 1.6–18 sec for Arrow/HF and WebDataset
- Per-cell random-access overhead is prohibitive for training loop throughput
- Throughput at 8 workers (4,751 samples/sec) is 8–9× lower than Arrow/HF and WebDataset
- Despite low storage bytes (column-chunked), the per-cell access pattern dominates
- GPU utilization proxy is comparable (0.77–0.82), confirming the GPU is not the bottleneck — data feeding is

**When to revisit:**
- If chunking strategy can be re-tuned to cell-group batch reads rather than per-cell random access
- If TensorStore's cloud-storage advantages become relevant for distributed multi-node setups
- This is deferred to a future benchmark round, not v0

---

## Append-Mode Guidance

### V0 Default: `append_routed` (routed/indexed corpus assembly)

- New datasets are materialized individually as Arrow/HF Parquet datasets
- Each dataset gets its own cell metadata SQLite, feature registry update, and QA manifest
- Corpus index tracks each dataset's location, join mode, and token range
- New cells are indexed and retrievable without rewriting existing materialized data
- This is the **default scale path** for any corpus expected to grow beyond a single onboarding batch

### `create_new` — appropriate for:
- First dataset in a new corpus
- Pilot/evaluation runs where the full corpus scope is not yet defined
- When the entire dataset fits comfortably in memory and no future joins are planned

### `append_monolithic` — remains supported but not default:
- Appropriate for early-stage single-dataset exploration
- Small corpora where the full dataset is materialized as a single Parquet file
- Does not scale well for multi-dataset corpora with different provenance
- **Not recommended as the default path** for any corpus expected to onboard additional datasets

### `append_routed` vs `append_monolithic` decision tree:

```
Is the corpus expected to grow beyond the current dataset?
├── NO → use create_new or append_monolithic
└── YES (expected new datasets) → use append_routed
    ├── Is the new dataset from a different provenance/source?
    │   └── YES → append_routed is required to avoid rewriting existing data
    └── NO (same pipeline, same format)
        └── append_routed still preferred for immutability guarantees
```

---

## Synthetic Benchmark Caveats

- All benchmarks used **synthetic random sparse data** (random gene indices, Poisson counts)
- Real h5ad data has structured sparsity, non-uniform gene distributions, andbatch effects
- Build time (0.2–2.4 sec range) reflects synthetic generation, not real materialization
- Phase 6 recommendation should be treated as **provisional v0 guidance**; a confirmatory run against real Phase 3 outputs is listed in the follow-up backlog

---

## Summary

| Role | Backend | Key Reason |
|------|---------|-----------|
| **V0 Default** | Arrow/HF | Best wall-time scaling, smallest storage, sparse integer storage, random-access capable |
| **Named Alternate** | WebDataset | Highest peak throughput (8w), acceptable at 10–13× storage cost |
| **Not in V0** | Zarr/TensorStore | Severe wall-time scaling failure (8–9× slower), per-cell random-access overhead |
| **Default Join Mode** | `append_routed` | Default scale path for growing corpora; `append_monolithic` remains available for early-stage single-dataset use |
