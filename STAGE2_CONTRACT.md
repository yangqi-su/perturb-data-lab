# Stage 2 Contract — Schema-Independent Materialization And Corpus Registration

- `Contract ID`: `stage2-contract-v0.3.0`
- `Created`: `2026-04-24`
- `Status`: `frozen`
- `Owner`: `perturb-data-lab`

## Purpose

This document is the authoritative contract for Stage 2 execution. It defines the exact inputs, outputs, storage formats, gating behavior, and compatibility policy for the schema-independent materialization path. When this contract is frozen, implementation can proceed without ambiguity about any of these dimensions.

---

## 1. Stage 1 Gating Behavior

### 1.1 The Single Source of Truth

Stage 1 produces a **`DatasetSummaryDocument`** (written at `{dataset_id}/dataset-summary.yaml` during inspection). The `count_source_decision` field inside is the **only** authoritative count-source decision for a dataset. Stage 2 MUST NOT select a count source independently.

### 1.2 Reuse vs. Rerun Precedence Rule

Stage 2 accepts two gating modes, in this precedence order:

1. **Reuse mode (default)**: Stage 2 consumes an existing `dataset-summary.yaml` from a prior Stage 1 run. The path to this file is passed as `review_bundle_path`. Stage 2 reads `count_source_decision` directly from this artifact.

2. **Rerun mode (opt-in)**: Stage 2 re-executes Stage 1 inspection as a preflight step before materialization. Enabled by passing `rerun_stage1=True`. The resulting `dataset-summary.yaml` is used as the gate.

   - Rerun mode is useful when the source h5ad has changed since the last Stage 1 run, or when no prior review bundle exists.
   - Rerun mode does NOT produce or consume a `schema.yaml` — Stage 1 inspection explicitly does not write `schema.yaml` under the current contract.

### 1.3 Hard Gate Condition

Stage 2 proceeds only when `materialization_readiness` in the `DatasetSummaryDocument` is `"pass"`. If it is `"needs-review"` or `"fail"`, Stage 2 raises `ValueError` with a descriptive message and does not write any artifacts.

---

## 2. Approved Count-Source Handoff

### 2.1 What Is Passed In

Stage 2 receives from the Stage 1 decision:

| Field | Type | Description |
|-------|------|-------------|
| `selected` | `str` | Candidate name: `.X`, `.raw.X`, or `.layers[{name}]` |
| `uses_recovery` | `bool` | Whether reverse-normalization is required |
| `recovery_policy` | `str` | e.g. `expm1_over_size_factor`, `disallowed`, `not-needed` |

### 2.2 Count Matrix Selection Logic

```
if selected == ".raw.X":
    count_matrix = adata.raw.X
elif selected.startswith(".layers["):
    layer_name = selected[len(".layers[") : -1]
    count_matrix = adata.layers[layer_name]
else:
    count_matrix = adata.X
```

### 2.3 Recovery Application

When `uses_recovery == True`, Stage 2 applies vectorized reverse-normalization inside `_translate_chunk()` using a **temporary expm1 CSR matrix** — no full-matrix densification. The caller passes `needs_recovery=True` based on Stage 1's `uses_recovery` decision.

The approved path is:

```
expm1_data = expm1(source_nonzeros)
recovered = rint(expm1_data / row_min_expm1)   # row_min_expm1 per-row over actual nonzero entries
```

The **recovered integer count matrix** is what is written to storage. Recovery metadata (scale factors) is NOT persisted — only the final integer counts.

### 2.4 Integer Verification

After any recovery step, Stage 2 verifies that the count matrix is integer-like (max deviation from nearest integer < 1e-6). If not, materialization fails with `ValueError`. This check is applied to the **final written counts**, not just the source matrix.

---

## 3. Heavy-Row Contract

### 3.1 Storage Principle

Counts are written in **dataset-local feature space**. Feature identity is NOT globalized at materialization time. Heavy rows carry only:

- `global_row_index` — assigned at corpus-append time, not at materialization time
- `dataset_index` — assigned at corpus-append time
- `local_row_index` — row offset within the dataset
- `expressed_gene_indices` — column indices in the dataset-local feature space
- `expression_counts` — integer counts, one per expressed gene
- optional `size_factor` — computed as `row_sum / median(row_sum)` post-recovery

### 3.2 Sparse Contract

All heavy-row storage uses **sparse representation** (CSR/CSC or equivalent). Dense conversion is prohibited for normal operation. The sparse contract requires:

- `expressed_gene_indices` and `expression_counts` must be the same length
- All counts must be non-negative integers
- Zero entries are excluded from storage

### 3.3 Heavy Row Fields Per Topology

**Federated topology** (each dataset stored independently):
- Heavy rows are stored per-dataset with local feature indices
- `global_row_index` is always present in heavy rows (assigned at materialization time from `global_row_start + local_offset`)
- `global_row_index` is also recorded in the corpus ledger for routing purposes

**Aggregate topology** (multiple datasets in one shared object):
- Heavy rows include `dataset_index` and `local_row_index`
- `global_row_index` is computed as the cumulative offset at write time (`global_row_start + local_row_index`)
- Row ranges are deterministic and non-overlapping across datasets
- `global_row_index` is always present in heavy rows for both federated and aggregate topologies

---

## 4. Raw Metadata Sidecar Contract

### 4.1 Raw `obs` Sidecar (Parquet, not SQLite)

- **Format**: Parquet (machine-readable, columnar, Polars-compatible)
- **Path**: `{metadata_root}/{release_id}-raw-obs.parquet`
- **Schema**:
  - `cell_id`: `string` — the obs index value as a string
  - `dataset_id`: `string`
  - `dataset_release`: `string`
  - `raw_obs`: `string` — JSON-serialized dict of all obs columns (preserving nulls as `null`)
- **No canonical mapping applied** — this is the raw obs as it appeared in the source h5ad
- **SQLite is deprecated** for this purpose — existing SQLite-based cell metadata sidecars are legacy and will not be produced by the new path

### 4.2 Raw `var` Sidecar (Parquet)

- **Format**: Parquet
- **Path**: `{metadata_root}/{release_id}-raw-var.parquet`
- **Schema**:
  - `origin_index`: `int32` — the position in the original dataset var order (0-indexed)
  - `feature_id`: `string` — the var index value as a string
  - `raw_var`: `string` — JSON-serialized dict of all var columns (preserving nulls as `null`)
- **No canonical mapping applied** — this is the raw var as it appeared in the source h5ad

### 4.3 Size-Factor Sidecar (Parquet, not inline)

- **Format**: Parquet
- **Path**: `{metadata_root}/{release_id}-size-factor.parquet`
- **Schema**:
  - `size_factor`: `float64` — one row per cell, ordered to match the heavy-row ordering
- **Content**: Per-cell size factor computed as `row_sum / global_median(row_sum)` after all chunks are written — `ChunkBundle.row_sums` (raw float64 per-cell sums) are accumulated chunk-wise during the chunk loop, a global median is computed across all cells, and the normalized size factors are written in a single pass after the loop
- **Rationale**: Size factors are stored in a separate Parquet to keep the cells heavy-row parquet free of non-count data; runtime loaders read this sidecar to provide per-row normalization context
- **No `size_factor` column in the cells parquet itself** — the cells parquet schema is strictly `['global_row_index', 'expressed_gene_indices', 'expression_counts']` following `HEAVY_CELL_SCHEMA`

### 4.4 Provenance Sidecar

- **Format**: YAML (human-readable) + Parquet (machine-readable)
- **Path**: `{metadata_root}/{release_id}-feature-provenance.parquet`
- **Schema**:
  - `origin_index`: `int32`
  - `feature_id`: `string`
  - `count_source`: `string` — the selected count source candidate name
  - `source_path`: `string` — absolute path to the source h5ad

---

## 5. Feature-Order And HVG Artifact Contract

### 5.1 Feature-Order Preservation

The **dataset-local feature order is the authoritative feature order** for this release. No canonical feature mapping is applied at this stage. The `origin_index` field in the raw `var` sidecar and provenance parquet files establishes the feature order.

### 5.2 HVG Arrays

- **Format**: NumPy `.npy` files
- **Path**: `{metadata_root}/hvg_sidecar/{release_id}-hvg.npy` and `{release_id}-nonhvg.npy`
- **Content**:
  - `hvg.npy`: `int32` array of dataset-local feature indices selected as HVGs
  - `nonhvg.npy`: `int32` array of the complement (all other feature indices)
- **Selection method**: Top-N dispersion (`variance / mean` of log1p(counts), computed via exact all-cells streaming accumulation using `np.add.at` during the chunk loop — no sampling)
- **N default**: 2000 (configurable via `n_hvg` parameter)
- **HVG indices are in dataset-local feature space** — they are NOT token IDs or global feature indices

---

## 6. Per-Dataset Manifest Contract

### 6.1 Manifest Document

- **Format**: YAML
- **Path**: `{metadata_root}/{release_id}-materialization-manifest.yaml`
- **Kind**: `materialization-manifest`

### 6.2 Required Fields

```yaml
kind: materialization-manifest
contract_version: "0.3.0"   # Stage 2 contract version
dataset_id: str
release_id: str
route: str                   # create_new | append_routed
backend: str                 # storage backend name
topology: str                # federated | aggregate
count_source:
  selected: str              # .X | .raw.X | .layers[{name}]
  integer_only: bool
  uses_recovery: bool
outputs:
  metadata_root: str
  matrix_root: str
provenance:
  source_path: str
  review_bundle: str          # path to the Stage 1 dataset-summary.yaml used as gate
raw_cell_meta_path: str      # path to raw-obs.parquet
raw_feature_meta_path: str   # path to raw-var.parquet
provenance_spec_path: str     # path to feature-provenance.parquet
hvg_sidecar_path: str        # path to hvg_sidecar directory
size_factor_parquet_path: str # path to size-factor.parquet (separate, not inline)
qa_manifest_path: str
integer_verified: bool
cell_count: int
feature_count: int
```

### 6.3 `backend` and `topology` Are Separate

`backend` names the storage format only:
- `arrow-parquet` — Arrow IPC serialized to Parquet via `pa.ipc.new_file`
- `arrow-ipc` — Arrow IPC serialized via `pa.ipc.new_file`
- `webdataset` — WebDataset tar format
- `zarr` — Zarr 1D flat-buffer layout
- `lance` — Direct Lance format (no lancedb wrapper)

`topology` names the corpus organization only:
- `federated` — each dataset has its own heavy object
- `aggregate` — datasets are appended into a shared heavy object

The **combination** of `backend` + `topology` is what was previously fused into single route names like `lancedb-aggregated`.

---

## 7. Corpus Ledger Contract

### 7.1 Ledger Document

- **Format**: Parquet (primary, machine-readable) + optional CSV export
- **Path**: `{corpus_root}/corpus-ledger.parquet`
- **Kind**: `corpus-ledger`

### 7.2 Schema

| Column | Type | Description |
|--------|------|-------------|
| `corpus_id` | `string` | Corpus identifier |
| `dataset_id` | `string` | Dataset identifier |
| `release_id` | `string` | Immutable release identifier |
| `dataset_index` | `int32` | Assigned index in corpus |
| `join_mode` | `string` | `create_new` or `append_routed` |
| `manifest_path` | `string` | Relative path from corpus root to manifest |
| `backend` | `string` | Storage backend name |
| `topology` | `string` | `federated` or `aggregate` |
| `cell_count` | `int64` | Number of cells in this dataset |
| `feature_count` | `int64` | Number of features (dataset-local) |
| `global_start` | `int64` | Inclusive start of global row range (for aggregate) |
| `global_end` | `int64` | Exclusive end of global row range (for aggregate) |
| `created_at` | `string` | ISO timestamp of materialization |

### 7.3 Create vs. Append Flow

**Create corpus** (`create_new`):
- Write `corpus-ledger.parquet` with the first dataset entry
- Create `global-metadata.yaml` alongside the ledger

**Append to corpus** (`append_routed`):
- Load existing `corpus-ledger.parquet`
- Append new dataset row with monotonically increasing `dataset_index`
- For aggregate topology, compute `global_start` = sum of all prior `cell_count`s
- Overwrite the ledger parquet in-place (append-safe via Parquet rewrite)

### 7.4 SQLite Is Deprecated

The previous `corpus-index.yaml` + SQLite-backed corpus tracking is **deprecated for Stage 2**. Existing corpora using YAML/SQLite are still readable during a transition period, but new Stage 2 runs will produce Parquet-ledger corpora only.

---

## 8. Storage-Format Policy

### 8.1 Format Preference

| Artifact | Preferred Format | Deprecated |
|----------|-----------------|------------|
| Raw cell metadata | Parquet | SQLite |
| Raw feature metadata | Parquet | — |
| Feature provenance | Parquet + YAML | — |
| Per-dataset manifest | YAML | — |
| Corpus ledger | Parquet | YAML/SQLite |
| Heavy count storage | Backend-specific (Arrow/Zarr/WebDataset/Lance) | — |
| HVG arrays | NumPy .npy | — |
| Size factors | Parquet (separate sidecar) | — |

### 8.2 Parquet Reading Preference

Parquet sidecars are intended to be read with **Polars** for speed. The reading contract does not require Polars (pandas is acceptable), but Polars is the preferred implementation for Stage 3.1+ consumers.

### 8.3 Deprecation Notice

**SQLite for cell metadata sidecars is deprecated as of this contract.** The new materialization path will not produce `.sqlite` cell metadata files. Any code consuming Stage 2 artifacts should migrate to Parquet readers. SQLite-based consumers will receive a deprecation warning but will continue to function for existing legacy corpora during a transition window.

---

## 9. Compatibility Policy

### 9.1 Entry Point

`Stage2Materializer` is the only active Stage 2 entry point. It accepts:

```
Stage2Materializer(
    source_path: str,
    review_bundle_path: str,      # Stage 1 dataset-summary.yaml
    output_roots: OutputRoots,
    release_id: str,
    dataset_id: str,
    backend: str,                 # arrow-parquet (default), arrow-ipc, webdataset, zarr, lance
    topology: str,               # federated (default) or aggregate
    rerun_stage1: bool = False,
)
```

It does **not** accept `schema_path`. Canonical metadata mapping is deferred to Stage 3. `MaterializationRoute`, `CreateNewRoute`, `AppendRoutedRoute`, and `build_materialization_route()` have been removed — `Stage2Materializer` replaces all of them.

### 9.2 Legacy Path Removed

`MaterializationRoute`, `CreateNewRoute`, `AppendRoutedRoute`, and `build_materialization_route()` have been removed. `Stage2Materializer` is the only active materialization entry point. The `AVAILABLE_BACKENDS` registry is removed. Code that depended on these artifacts has been migrated or removed.

---

## 10. Backend-Topology Matrix

### 10.1 Supported Matrix (v0.4.0 — This Contract)

All 10 backend×topology combinations are implemented. Validation scope:

- **Non-Lance (8 combinations)**: Validated end-to-end on `dummy_data` via Slurm in Phase 5. Result: 6/8 combinations fully pass; 2 failures (zarr × federated, zarr × aggregate) are validation-script path issues, not implementation bugs.
- **Lance (2 combinations)**: Validated end-to-end on `dummy_data` via Slurm in Phase 6 (Lance 4.0.0). Result: lance × federated PASS, lance × aggregate PASS.

| Backend | Federated | Aggregate |
|---------|-----------|-----------|
| `arrow-parquet` | ✅ implemented + validated | ✅ implemented + validated |
| `arrow-ipc` | ✅ implemented + validated | ✅ implemented + validated |
| `webdataset` | ✅ implemented + validated | ✅ implemented + validated |
| `zarr` | ✅ implemented (validation-script issue) | ✅ implemented (validation-script issue) |
| `lance` | ✅ implemented + validated | ✅ implemented + validated |

### 10.2 Canonical Backend Names

The following legacy backend names have been removed and are not produced by any current Stage 2 code:

| Legacy Name | New Canonical | Notes |
|-------------|---------------|-------|
| `arrow-hf` | `arrow-parquet` | `arrow-parquet` is the default backend |
| `zarr-ts` | `zarr` | `zarr` is canonical |
| `zarr-aggregated` | `zarr` | `zarr` handles both topologies |
| `lancedb-aggregated` | `lance` | Direct `lance` backend replaces lancedb wrapper |
| `lance-dataset` | `lance` | `lance` is canonical |

The `AVAILABLE_BACKENDS` registry is removed. New code uses explicit `backend` + `topology` dispatch via `AVAILABLE_WRITERS[backend][topology]`. `AVAILABLE_READERS` uses canonical names only. No `arrow-hf`, `zarr-ts`, `lancedb-aggregated`, or `zarr-aggregated` artifacts are produced by the current implementation.

### 10.3 Shared Chunk Translation

All 5 backends consume a shared translation layer (`perturb_data_lab/materializers/chunk_translation.py`) that translates CSR batches into a canonical `ChunkBundle` once, then each backend writer produces its own format from that shared payload. This eliminates per-backend sparse re-encoding on the hot path.

The shared `ChunkBundle` carries exactly 6 fields:
- `table`: `pa.Table` with heavy-row list-arrays (`global_row_index`, `expressed_gene_indices`, `expression_counts`) following `HEAVY_CELL_SCHEMA`
- `row_sums`: `np.ndarray` (float64) of raw (un-normalized) per-cell row sums — the caller accumulates these across chunks, computes the global median, and writes globally-normalized size factors to a separate Parquet sidecar after the loop
- `indptr`: raw CSR indptr array (int64) for backends that need flat-buffer access (Zarr, WebDataset)
- `indices`: raw **dataset-local** gene indices array (int32) — no canonical gene mapping is applied at this stage; canonical mapping is deferred to Stage 3
- `counts`: raw expression counts array (int32)
- `row_count`: total rows in this chunk

**`_translate_chunk()` contract**: accepts `dataset: DatasetSpec`, `matrix_chunk: csr_matrix`, `chunk_start: int` — no `gene_lookup` parameter. `expressed_gene_indices` in the output `ChunkBundle` are always in dataset-local feature space.

**Thin serializer contract**: each backend writer accepts a `ChunkBundle` and writes it to its native format. Writers do not call `_translate_chunk()`, do not implement CSR logic, and have no legacy fallback paths. The `Stage2Materializer` owns the chunk loop and calls `_translate_chunk()` once per chunk.

The `metadata_table` field has been removed. Per-cell metadata is written by the caller as a separate Parquet sidecar (`{release_id}-raw-obs.parquet`). The `_build_metadata_table()` helper remains available for callers that need a `METADATA_SCHEMA` table from obs data.

### 10.4 Implementation Notes

- The **default route** is **`arrow-parquet` × `federated`** — this is the default and must always work.
- `backend` and `topology` are separate manifest fields, not fused into a single route name.
- Size factors are stored in a separate Parquet sidecar (`{release_id}-size-factor.parquet`), not inline in the heavy-row parquet. `_translate_chunk()` computes raw `row_sums` (float64) per chunk; the caller accumulates them into a global array, computes the global median, and produces `size_factor = row_sum / global_median` after all chunks are written.
- No new SQLite artifacts are emitted on the active Stage 2 path.
- `lance` uses direct `lance.write_dataset()` — no `lancedb` dependency.

---

## 11. Execution Contract

### 11.1 Slurm Requirement

Per repository execution rules, **dataset-facing runs must use Slurm** when the dataset is a real `.h5ad` file. Dummy data runs for validation may use local execution. The Slurm environment is `torch_flashv3` with the `ihc` partition.

### 11.2 Large File Handling

Large `.h5ad` files (e.g., `perturb/marson2025_data/D1_Rest.assigned_guide.h5ad`) must be processed with backed/partial reads where possible. Stage 1 inspection already uses backed reads. Stage 2 materialization must also avoid eager-loading the full matrix into memory — sparse row-wise writes are the expected pattern.

---

## 12. Summary Paragraph (Contract Frozen State)

**Stage 2 materializes integer counts from a Stage 1-approved count source into a chosen backend and topology, preserving raw `obs`, raw `var`, and dataset-local feature order as Parquet sidecars without applying canonical metadata mapping. `Stage2Materializer` is the only active entry point; `MaterializationRoute`, `CreateNewRoute`, `AppendRoutedRoute`, and `build_materialization_route()` have been removed. Stage 2 is gated by a Stage 1 `dataset-summary.yaml` whose `materialization_readiness` must be `pass`; it supports reusing an existing review bundle or rerunning Stage 1 as preflight. Heavy rows are written in sparse dataset-local feature space with per-row size factors and are verifiable as integer at write time. HVG arrays are computed post-count-settlement and stored in dataset-local feature indices. A Parquet corpus ledger tracks dataset membership, append order, backend, topology, and row ranges. No `arrow-hf`, `zarr-ts`, `lancedb-aggregated`, or `zarr-aggregated` artifacts are produced. Backend and topology are separate manifest fields. `_translate_chunk()` accepts only `dataset`, `matrix_chunk`, and `chunk_start` — no `gene_lookup` parameter. Canonical gene mapping is deferred to Stage 3.**
