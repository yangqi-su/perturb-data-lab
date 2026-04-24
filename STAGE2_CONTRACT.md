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

When `uses_recovery == True`, Stage 2 applies the approved reverse-normalization path **during materialization only**. The approved path is:

```
recovered_row = expm1(source_row) / min_nonzero_expm1(source_row)
```

This is applied per-row on the selected matrix. The **recovered integer count matrix** is what is written to storage. Recovery metadata (scale factors) is NOT persisted — only the final integer counts.

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
- `global_row_index` is assigned at corpus-append time and stored in the corpus ledger
- No per-row global index is written inside the per-dataset heavy object

**Aggregate topology** (multiple datasets in one shared object):
- Heavy rows include `dataset_index` and `local_row_index`
- `global_row_index` is computed as the cumulative offset at write time
- Row ranges are deterministic and non-overlapping across datasets

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

### 4.3 Provenance Sidecar

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
- **Selection method**: Top-N dispersion (`variance / mean` of log1p(counts), computed on a sampled cell subset)
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
size_factor_manifest_path: str
qa_manifest_path: str
integer_verified: bool
cell_count: int
feature_count: int
```

### 6.3 `backend` and `topology` Are Separate

`backend` names the storage format only:
- `arrow-hf` — Arrow HDF5 Feathers
- `webdataset` — WebDataset tar format
- `zarr` — Zarr format
- `lance` — LanceDB format

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
| Size factors | YAML | — |

### 8.2 Parquet Reading Preference

Parquet sidecars are intended to be read with **Polars** for speed. The reading contract does not require Polars (pandas is acceptable), but Polars is the preferred implementation for Stage 3.1+ consumers.

### 8.3 Deprecation Notice

**SQLite for cell metadata sidecars is deprecated as of this contract.** The new materialization path will not produce `.sqlite` cell metadata files. Any code consuming Stage 2 artifacts should migrate to Parquet readers. SQLite-based consumers will receive a deprecation warning but will continue to function for existing legacy corpora during a transition window.

---

## 9. Compatibility Policy

### 9.1 Schema-Dependent Paths Are Legacy

The current `MaterializationRoute.materialize(source_path, schema_path)` entry point that requires `schema.yaml` is **legacy**. It remains functional for existing workflows but is explicitly out of the new Stage 2 contract. The new entry point:

```
Stage2Materializer.materialize(
    source_path: str,
    review_bundle_path: str,      # Stage 1 dataset-summary.yaml
    output_roots: OutputRoots,
    release_id: str,
    dataset_id: str,
    backend: str,
    topology: str,
    rerun_stage1: bool = False,
)
```

does NOT accept `schema_path`.

### 9.2 Existing Schema-Required Code

Code that currently calls the legacy entry point (e.g., CLI commands, test fixtures) should be updated to use the new entry point or should be marked as requiring a schema update. The compatibility boundary is:

- **Acceptable**: Legacy code that reads existing schema-produced artifacts (backward compatibility)
- **Prohibited**: Legacy code that **produces** new artifacts via the schema-dependent path for datasets not yet migrated

### 9.3 Accepted Schema Copy

When a dataset has been inspected and a `schema.yaml` exists from prior work, Stage 2 may **optionally** copy the accepted schema to `{metadata_root}/{release_id}-accepted-schema.yaml` for audit trail purposes. This copy is for provenance only — it is NOT read back during materialization or loading.

---

## 10. Backend-Topology Matrix

### 10.1 Supported Subset (This Contract)

| Backend | Federated | Aggregate |
|---------|-----------|-----------|
| `arrow-hf` | ✅ supported | ❌ deferred |
| `webdataset` | ✅ supported | ❌ deferred |
| `zarr` | ✅ supported | ❌ deferred |
| `lance` | ❌ deferred | ✅ supported |

### 10.2 Implementation Notes

- The minimum supported subset is **`arrow-hf` × `federated`** — this is the default and must always work.
- Other combinations are implemented where already present but are not required to be fully validated until later phases.
- Backend-topology separation is enforced in the manifest schema — `backend` and `topology` are separate string fields, not fused into a single route name.

---

## 11. Execution Contract

### 11.1 Slurm Requirement

Per repository execution rules, **dataset-facing runs must use Slurm** when the dataset is a real `.h5ad` file. Dummy data runs for validation may use local execution. The Slurm environment is `torch_flashv3` with the `ihc` partition.

### 11.2 Large File Handling

Large `.h5ad` files (e.g., `perturb/marson2025_data/D1_Rest.assigned_guide.h5ad`) must be processed with backed/partial reads where possible. Stage 1 inspection already uses backed reads. Stage 2 materialization must also avoid eager-loading the full matrix into memory — sparse row-wise writes are the expected pattern.

---

## 12. Summary Paragraph (Contract Frozen State)

**Stage 2 materializes integer counts from a Stage 1-approved count source into a chosen backend and topology, preserving raw `obs`, raw `var`, and dataset-local feature order as Parquet sidecars without applying canonical metadata mapping. Stage 2 is gated by a Stage 1 `dataset-summary.yaml` whose `materialization_readiness` must be `pass`; it supports reusing an existing review bundle or rerunning Stage 1 as preflight. Heavy rows are written in sparse dataset-local feature space with per-row size factors and are verifiable as integer at write time. HVG arrays are computed post-count-settlement and stored in dataset-local feature indices. A Parquet corpus ledger tracks dataset membership, append order, backend, topology, and row ranges. SQLite is deprecated for all new Stage 2 artifacts. Backend and topology are separate manifest fields. The new entry point does not accept `schema_path`.**
