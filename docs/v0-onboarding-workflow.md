# V0 Onboarding Workflow

**Date**: 2026-04-20
**Plan ID**: `copilot-plans-20260420-perturb-data-lab-schema-realignment`
**Scope**: How to onboard a new h5ad dataset into the perturb-data-lab v0 stack, from raw h5ad inspection through to a training-ready loader.

---

## Prerequisites

- A raw `.h5ad` file accessible via a read-only path
- `perturb-data-lab` installed or available on a machine with Slurm access (`torch_flashv3` environment for large files)
- At least one existing pilot dataset already materialized to understand the feature registry baseline
- No data is ever written to protected symlink roots (`data/`, `pertTF/`, `perturb/`); all outputs go to repo-local real directories

---

## Route Model

The materialization layer exposes two public routes:

| Route | When to use |
|---|---|
| `create_new` | First dataset of a new corpus, or a standalone dataset not joined to any corpus |
| `append_routed` | Adding a new dataset to an existing corpus that may continue growing |

`append_monolithic` is removed — its behaviour was identical to `append_routed` and its presence created a redundant public branch.

---

## Step 1 — Batch Inspection Config (Slurm CPU for large files)

**Tool**: `perturb_data_lab.inspect` CLI via `InspectionBatchConfig`

**Goal**: Produce a `dataset-summary.yaml` and `schema.yaml` draft without loading the full matrix into memory.

**Inspection config** (`inspection-config.yaml`):
```yaml
output_root: /path/to/plan/phase-02-inspector-outputs/
datasets:
  - dataset_id: your_dataset_v0
    source_path: /path/to/your/data.h5ad
    source_release: "2026-01"
```

**What to run**:
```bash
cd /path/to/perturb-data-lab
PYTHONPATH=src python -m perturb_data_lab.inspect \
  --config /path/to/plan/phase-02-inspection-config.yaml
```

**What it does**:
- Opens each h5ad with `backed='r'` (no full load)
- Extracts `.obs` column names, dtypes, nunique, sample values
- Extracts `.var` gene metadata and genome annotation
- Checks `.raw` and named `.layers` for count-source candidates
- Audits integer-likeness of selected count source (sample-based, not full scan)
- Runs heuristic field mapping for canonical cell perturbation, context, and feature fields
- Writes `dataset-summary.yaml` and `schema.yaml` (draft) per dataset

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

**Goal**: Review the generated `schema.yaml`, fill in any unresolved fields, confirm the count-source and feature tokenization choices, and set `status: ready`.

**Input artifacts**:
- `dataset-summary.yaml` — dataset statistics and field inventory
- `schema.yaml` (draft) — proposed raw-to-canonical field mappings for all cell and feature fields

**What to review**:

1. **Count source**: Which matrix was selected (`.X`, `.raw.X`, or a named layer). Is the auto-detected choice correct?
2. **Perturbation fields**: `perturbation_label`, `perturbation_type`, `target_id`, `target_label`, `control_flag`, `dose`, `dose_unit`, `timepoint`, `timepoint_unit`, `combination_key`
3. **Context fields**: `cell_context`, `cell_line_or_type`, `species`, `tissue`, `assay`, `condition`, `batch_id`, `donor_id`, `sex`, `disease_state`
4. **Feature fields**: `feature_id` (required), `feature_label` (optional), `feature_namespace` (optional)
5. **Unresolved fields**: Fields with `strategy: null` will materialize to `NA`. If a field is genuinely not applicable, leave it null.
6. **Control identification**: Are control/vehicle cells labelled correctly via the `control_flag` derived field?

**Required action before materialization**: Set `status: ready` at the top of `schema.yaml`. Materialization will refuse to run on schemas that are not `ready`.

**Output**: Reviewed and edited `schema.yaml` with `status: ready`.

---

## Step 3 — Materialization

**Tool**: Python API via `perturb_data_lab.materializers`

**Route selection**:
```
Is this the first dataset in a new corpus?
├── YES → create_new
└── NO  → append_routed (default scale path for growing corpora)
```

**Python API for create_new**:
```python
from perturb_data_lab.materializers import (
    build_materialization_route,
    update_corpus_index,
    OutputRoots,
)
from perturb_data_lab.materializers.models import DatasetJoinRecord, CountSourceSpec
from perturb_data_lab.inspectors.models import SchemaDocument

schema = SchemaDocument.from_yaml_file("/path/to/reviewed-schema.yaml")
count_source = CountSourceSpec(
    selected=schema.count_source.selected,
    integer_only=schema.count_source.integer_only,
)

roots = OutputRoots(
    metadata_root="/path/to/outputs/metadata",
    matrix_root="/path/to/outputs/matrix",
)
route = build_materialization_route(
    route="create_new",
    output_roots=roots,
    release_id="your-dataset-v0",
    dataset_id="your_dataset_v0",
    count_source=count_source,
)
manifest = route.materialize(
    source_path="/path/to/your/data.h5ad",
    schema_path="/path/to/reviewed-schema.yaml",
)
```

**Python API for append_routed** (on top of an existing corpus):
```python
# Same as above, but use route="append_routed"
route = build_materialization_route(
    route="append_routed",
    output_roots=roots,
    release_id="your-dataset-v0",
    dataset_id="your_dataset_v0",
    count_source=count_source,
)
manifest = route.materialize(
    source_path="/path/to/your/data.h5ad",
    schema_path="/path/to/reviewed-schema.yaml",
)

# Update the corpus index to record the new dataset
corpus_record = DatasetJoinRecord(
    dataset_id="your_dataset_v0",
    release_id="your-dataset-v0",
    join_mode="append_routed",
    manifest_path=str(Path(roots.metadata_root) / "materialization-manifest.yaml"),
)
from perturb_data_lab.materializers import update_corpus_index
update_corpus_index(Path("/path/to/outputs/metadata/corpus-index.yaml"), corpus_record)
```

**What gets written**:
- `{release_id}-cells/` — Arrow/HF sparse per-cell expression directory (int32 indices + int32 counts)
- `{release_id}-cell-meta.sqlite` — SQLite-backed per-cell canonical metadata (canonical_perturbation, canonical_context, raw_obs as JSON columns)
- `{release_id}-features-origin.parquet` — canonical feature metadata in original dataset feature order
- `{release_id}-features-token.parquet` — tokenized companion mapping original indices to token IDs
- `feature-registry.yaml` — append-only feature vocabulary (token IDs preserved across datasets)
- `size-factor-manifest.yaml` — per-cell size factors
- `qa-manifest.yaml` — QA checks including integer verification
- `materialization-manifest.yaml` — full provenance and configuration snapshot
- `corpus-index.yaml` (for `append_routed`) — updated corpus membership

**Validation checks**:
- `integer_verified: true` in `materialization-manifest.yaml`
- `all_passed: true` in `qa-manifest.yaml`
- Feature registry token IDs are preserved across append operations (no ID reuse collisions)

---

## Step 4 — Training Loader Integration

**Import**: `from perturb_data_lab.loaders import ArrowHFCellReader, PerturbIterableDataset, PerturbDataLoader`

**Minimal example** (standalone dataset):
```python
from perturb_data_lab.loaders import ArrowHFCellReader, PerturbIterableDataset
from perturb_data_lab.loaders.samplers import RandomContextSampler

reader = ArrowHFCellReader(
    cells_parquet="/path/to/outputs/matrix/{release_id}-cells/",
    cell_meta_sqlite="/path/to/outputs/metadata/{release_id}-cell-meta.sqlite",
    feature_meta_paths={  # optional: preload feature objects for token translation
        "your_dataset_v0": "/path/to/outputs/metadata/{release_id}-features-token.parquet",
    },
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

**Token-space translation** (optional):
```python
# Add translate_indices=True to get token-space gene indices instead of
# original dataset-order indices
dataset = PerturbIterableDataset(reader, sampler, batch_size=64, translate_indices=True)
for batch in dataset:
    # batch["token_gene_indices"]: (batch, context_size) int32
    pass
```

**Multi-dataset corpus loading**: Use `CorpusIndexDocument` to enumerate all member datasets, construct one `ArrowHFCellReader` per dataset, and iterate over them programmatically.

**Sampler options** (all backend-agnostic):
- `RandomContextSampler` — uniform random context from expressed genes
- `ExpressedZerosSampler` — half expressed genes, half zero genes
- `HVGRandomSampler` — HVGs + equal random non-HVG genes

---

## End-to-End Flow Summary

```
h5ad file
   │
   ▼
[Step 1] Batch inspection (backed h5ad, no full load)
   dataset-summary.yaml + schema.yaml (draft)
   │
   ▼
[Step 2] Human schema review
   schema.yaml (status: ready)
   │
   ▼
[Step 3] Materialization (create_new or append_routed)
   SQLite + Arrow/HF + feature parquets + manifests
   │
   ▼
[Step 4] Training loader (ArrowHFCellReader + sampler)
   PyTorch batch ready for transformer forward pass
```

**Onboarding a second dataset into an existing corpus**:
- Start from Step 1 for the new h5ad
- In Step 3, use `append_routed` (not `create_new`)
- Call `update_corpus_index()` to record the new dataset in the corpus index
- Step 4 uses the same reader + sampler API; corpus-level iteration is handled by enumerating datasets via corpus index
