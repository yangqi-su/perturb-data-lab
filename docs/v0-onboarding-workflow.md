# V0 Onboarding Workflow

**Date**: 2026-04-20  
**Plan ID**: `copilot-plans-20260419-perturb-data-lab-v0`  
**Scope**: How to onboard a new h5ad dataset into the perturb-data-lab v0 stack, from raw h5ad inspection through to a training-ready loader.

---

## Prerequisites

- A raw `.h5ad` file accessible via a read-only path
- `perturb-data-lab` installed or available on a machine with Slurm access (`torch_flashv3` environment for large files)
- At least one existing pilot dataset already materialized (Phase 3 outputs) to understand the feature registry baseline
- No data is ever written to protected symlink roots (`data/`, `pertTF/`, `perturb/`); all outputs go to repo-local real directories

---

## Step 1 — Lightweight h5ad Inspection (Slurm CPU)

**Tool**: `perturb_data_lab.inspect` CLI or `InspectionWorkflow`

**Goal**: Produce a `dataset-summary.yaml` and `schema-proposal.yaml` without loading the full matrix into memory.

**What to run**:
```bash
cd /path/to/perturb-data-lab
PYTHONPATH=src python -m perturb_data_lab.inspect \
  --h5ad /path/to/your/data.h5ad \
  --output-dir /path/to/plan/phase-02-inspector-outputs/your_dataset \
  --config /path/to/plan/phase-02-inspection-config.yaml
```

**What it does**:
- Opens the h5ad with `backed='r'` (no full load)
- Extracts `.obs` column names, dtypes, nunique, sample values
- Extracts `.var` gene metadata and genome annotation
- Checks `.raw` and named `.layers` for count-source candidates
- Audits integer-likeness of selected count source (sample-based, not full scan)
- Runs heuristic field mapping for canonical perturbation and context fields
- Outputs `dataset-summary.yaml` and `schema-proposal.yaml`

**What to inspect in the output**:
- `count_source`: which matrix was selected (`.X`, `.raw.X`, or a named layer)
- `integer_verified`: boolean; if `false`, this dataset may need explicit count-recovery review
- `materialization_readiness`: `pass`, `needs-review`, or `fail`
- `unresolved_fields`: list of canonical fields not auto-mapped

**Slurm resource request** (for large h5ad files, >1M cells):
```bash
#SBATCH --partition=ihc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --account=ihc
```

---

## Step 2 — Human Schema Review

**Actors**: Dataset owner or curator

**Goal**: Review `schema-proposal.yaml`, fill in `schema-patch.yaml` for any unresolved fields, confirm count-source choice.

**Input artifacts**:
- `dataset-summary.yaml` — dataset statistics and field inventory
- `schema-proposal.yaml` — proposed raw-to-canonical field mappings
- `schema-patch.yaml` (to be created) — human corrections and unresolved-field resolutions

**What to decide**:
1. **Count source confirmation**: Is the auto-detected count source correct? If the data has been normalized, can you confirm integer recovery is trustworthy?
2. **Perturbation fields**: Is `perturbation_type`, `perturbation_target`, `guide_id`, etc. mapped correctly?
3. **Context fields**: Is `cell_type`, `treatment`, `dose`, `time_point`, etc. correct?
4. **Control identification**: Are control/vehicle cells labeled correctly?
5. **Any `NA` literals**: Missing values should use literal `NA` with documented rationale

**Output**: `schema-patch.yaml` with `review_status: accepted`

**Decision tree for count source**:
```
Is the selected count source (.X, .raw.X, or named layer) composed of integers?
├── YES, and no normalization transform is evident → proceed
├── YES, but transform is evident (e.g., log1p) and invertible
│   └── Can you document the inversion? → proceed with explicit transform note
└── NO (floats, non-integer values) → FAIL
    → Do not materialize until integer count source is confirmed
```

---

## Step 3 — Materialization

**Tool**: `perturb_data_lab.MaterializationWorkflow`

**Route selection**:
```
Is this the first dataset in a new corpus?
├── YES → use create_new
└── NO
    ├── Is the corpus expected to grow with more datasets?
    │   ├── YES → use append_routed (default scale path)
    │   └── NO → append_monolithic is acceptable
    └── Are you joining to an existing single-dataset corpus?
        └── YES → append_monolithic (acceptable for fixed small corpora)
```

**Command for create_new**:
```bash
PYTHONPATH=src python -m perturb_data_lab.materialize \
  --route create_new \
  --h5ad /path/to/your/data.h5ad \
  --schema-patch /path/to/phase-02-inspector-outputs/your_dataset/schema-patch.yaml \
  --output-dir /path/to/plan/phase-03-materialization-outputs/ \
  --corpus-name your_corpus_v0
```

**Command for append_routed**:
```bash
PYTHONPATH=src python -m perturb_data_lab.materialize \
  --route append_routed \
  --h5ad /path/to/your/data.h5ad \
  --schema-patch /path/to/phase-02-inspector-outputs/your_dataset/schema-patch.yaml \
  --output-dir /path/to/plan/phase-03-materialization-outputs/ \
  --corpus-name existing_corpus_v0 \
  --corpus-index /path/to/plan/phase-03-materialization-outputs/metadata/corpus-index.yaml
```

**What gets written**:
- `your-dataset-v0-cells.parquet` — sparse per-cell expression (int32 indices + int32 counts)
- `your-dataset-v0-meta.parquet` — dense metadata columns
- `your-dataset-v0-cell-meta.sqlite` — SQLite-backed obs lookup
- `feature-registry.yaml` — append-only gene vocabulary (token IDs preserved)
- `size-factor-manifest.yaml` — per-cell size factors
- `qa-manifest.yaml` — QA checks including integer verification
- `materialization-manifest.yaml` — full provenance and config snapshot
- `corpus-index.yaml` (for routed) — updated corpus membership

**Validation**:
- `integer_verified: true` in `materialization-manifest.yaml`
- `qa_status: pass` in `qa-manifest.yaml`
- Feature registry token IDs are preserved (no ID reuse collisions)

---

## Step 4 — Training Loader Integration

**Import**: `from perturb_data_lab.loaders import ArrowHFCellReader, PerturbIterableDataset`

**Minimal example**:
```python
from perturb_data_lab.loaders import ArrowHFCellReader, PerturbIterableDataset
from perturb_data_lab.loaders.samplers import RandomContextSampler

reader = ArrowHFCellReader(
    cells_parquet="/path/to/your-dataset-v0-cells.parquet",
    meta_parquet="/path/to/your-dataset-v0-meta.parquet",
    cell_meta_sqlite="/path/to/your-dataset-v0-cell-meta.sqlite",
    feature_registry_path="/path/to/feature-registry.yaml",
)
sampler = RandomContextSampler(reader, context_size=128)
dataset = PerturbIterableDataset(reader, sampler, batch_size=64)

for batch in dataset:
    # batch["gene_indices"]: (batch, context_size) int32
    # batch["expression_counts"]: (batch, context_size) int32
    # batch["size_factor"]: (batch,) float32
    # batch["cell_id"]: (batch,) str
    pass
```

**Sampler options** (all backend-agnostic):
- `RandomContextSampler` — uniform random context from expressed genes
- `ExpressedZerosSampler` — half expressed genes, half zero genes
- `HVGRandomSampler` — HVGs + equal random non-HVG genes

**Backend-agnostic guarantee**: All three backends (Arrow/HF, WebDataset, Zarr/TS) implement `BackendCellReader`; sampler behavior is identical across backends.

---

## Step 5 — Coroutine with Existing Corpus

If joining an existing corpus:
1. Confirm `corpus-index.yaml` is updated with the new dataset entry
2. Confirm feature registry token IDs are preserved (no remapping of existing genes)
3. Confirm `append_routed` was used if the corpus is expected to grow further
4. For training with a multi-dataset corpus, use `CorpusIndexDocument` to enumerate all member datasets and loaders

---

## Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|-------------|------------|
| `integer_verified: false` in QA manifest | Count source contains non-integer values | Review count-source decision in Step 2; do not proceed |
| `materialization_readiness: needs-review` | Unresolved canonical fields | Complete `schema-patch.yaml` before materializing |
| Worker stall in training loop | Backend not keeping GPU fed | Check worker count; Arrow/HF at 8 workers is generally sufficient |
| Feature registry token ID collision | Two datasets used same gene identifiers with different gene symbols | Audit namespace provenance in `feature-registry.yaml` |
| Zarr/TS slow wall time | Expected — Zarr/TS not recommended in v0 | Switch to Arrow/HF; Zarr/TS deferred to future benchmark round |

---

## End-to-End Flow Summary

```
h5ad file
   │
   ▼
[Step 1] Slurm-backed backed h5ad inspection
   dataset-summary.yaml + schema-proposal.yaml
   │
   ▼
[Step 2] Human schema review
   schema-patch.yaml (review_status: accepted)
   │
   ▼
[Step 3] Materialization (create_new / append_routed / append_monolithic)
   Arrow/HF Parquet outputs + manifests
   │
   ▼
[Step 4] Training loader (ArrowHFCellReader + sampler)
   PyTorch batch ready for transformer forward pass
```

**Onboarding a second dataset into an existing corpus**:
- Start from Step 1 for the new h5ad
- In Step 3, use `append_routed` with existing `--corpus-index` and `--corpus-name`
- Step 4 uses the same reader + sampler API; corpus-level iteration is handled by enumerating datasets via corpus index
