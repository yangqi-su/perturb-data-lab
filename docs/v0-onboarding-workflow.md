# V0 Onboarding Workflow

**Date**: 2026-05-11  
**Plan ID**: `plans-20260511-lance-create-append-canonicalization`  
**Scope**: Current public onboarding workflow for turning raw `.h5ad` inputs into a canonicalized corpus that can be loaded through `load_corpus()`.

---

## Current public workflow

```text
inspect
→ materialize
→ draft-schema
→ finalize final-schema.yaml
→ canonicalize
→ load_corpus
```

This order is intentional:

- **Materialization is count-first and schema-independent.** It needs an inspection review bundle, not a finalized canonical schema.
- **Canonical metadata is required before runtime loading.** `load_corpus()` expects `canonical-obs.parquet` and `canonical-var.parquet`.
- **Append is a first-class public route.** Create the corpus with the first dataset, then append later datasets into the existing corpus.

---

## Prerequisites and safety rules

- Source `.h5ad` files must be treated as read-only inputs.
- Large h5ad inspection/materialization should run on Slurm CPU in the `torch_flashv3` environment.
- Do not write into protected symlink roots: `data/`, `pertTF/`, or `perturb/`.
- Write all review bundles, corpus outputs, schemas, and validation artifacts to repo-local real directories.
- Use stable dataset identifiers because they become corpus paths under `meta/<dataset_id>/`.

Baseline Slurm request for large inspection/materialization work:

```bash
#SBATCH --partition=ihc
#SBATCH --nodelist=ihc-grid-1-1-1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --account=ihc
```

---

## Step 1 — Inspect the raw h5ad

**Goal**: Produce a review bundle that records metadata inventory, matrix-source candidates, and materialization readiness without loading the whole matrix eagerly.

Preferred single-dataset CLI:

```bash
PYTHONPATH=src python -m perturb_data_lab.cli inspect \
  --source /path/to/dataset.h5ad \
  --dataset-id my_dataset \
  --output-dir /path/to/review/my_dataset
```

Batch mode is also available:

```bash
PYTHONPATH=src python -m perturb_data_lab.cli inspect \
  --config /path/to/inspection-batch.yaml \
  --workers 1
```

Primary output:

- `dataset-summary.yaml`

What inspection decides:

- Which expression source is selected (`.X`, `.raw.X`, or a named layer)
- Whether sampled values are directly integer-like
- Whether recovery is required
- Whether the dataset is `pass`, `needs-review`, or blocked for materialization

### Interpreting count-source outcomes

| Inspection outcome | Meaning | Action |
|---|---|---|
| direct | Selected values already behave like counts | Materialize with the reviewed bundle |
| recovered | Selected values need an approved recovery policy before integer verification | Materialize only through the reviewed/approved path |
| needs-review | Recovery is plausible but should not pass silently | Resolve explicitly before materialization |
| fail | No acceptable count source | Stop and diagnose before continuing |

Recovery-specific note:

- Float storage alone is not a blocker.
- A float matrix can still pass directly if sampled nonzero values are exactly integer-valued.
- A log-like or binned source may be usable only through recovery, and Stage 2 must still verify integer counts before writing the corpus.

---

## Step 2 — Materialize counts into a corpus

**Goal**: Write backend artifacts, raw metadata sidecars, manifests, and corpus registration entries.

### 2A. Create a new corpus from the first dataset

```bash
PYTHONPATH=src python -m perturb_data_lab.cli materialize \
  --mode create \
  --source /path/to/first_dataset.h5ad \
  --dataset-id first_dataset \
  --review-bundle /path/to/review/first_dataset/dataset-summary.yaml \
  --output-corpus /path/to/corpus \
  --backend lance \
  --topology aggregate \
  --corpus-id my_corpus
```

### 2B. Append a later dataset into that existing corpus

```bash
PYTHONPATH=src python -m perturb_data_lab.cli materialize \
  --mode append \
  --source /path/to/later_dataset.h5ad \
  --dataset-id later_dataset \
  --review-bundle /path/to/review/later_dataset/dataset-summary.yaml \
  --output-corpus /path/to/corpus
```

Append notes:

- `--backend` and `--topology` can usually be omitted for append; the existing corpus metadata is authoritative.
- Aggregate append must preserve contiguous global row ranges across datasets.
- Materialization copies the reviewed `dataset-summary.yaml` into `meta/<dataset_id>/` for downstream schema drafting.

Key outputs written during materialization:

- `corpus-index.yaml`
- `corpus-ledger.parquet`
- `global-metadata.yaml`
- backend matrix artifacts (for example `matrix/aggregated-cells.lance`)
- `meta/<dataset_id>/raw-obs.parquet`
- `meta/<dataset_id>/raw-var.parquet`
- `meta/<dataset_id>/metadata-summary.yaml`
- `meta/<dataset_id>/materialization-manifest.yaml`
- `meta/<dataset_id>/qa-manifest.yaml`
- `meta/<dataset_id>/size-factor.parquet`

Materialization checks to review before moving on:

- `integer_verified: true` in `materialization-manifest.yaml`
- `all_passed: true` in `qa-manifest.yaml`
- expected `global_start` / `global_end` registration in `corpus-index.yaml`

---

## Step 3 — Draft the canonical schema after materialization

**Goal**: Generate a starting `draft-schema.yaml` from the corpus-local raw metadata artifacts.

```bash
PYTHONPATH=src python -m perturb_data_lab.cli draft-schema \
  --corpus /path/to/corpus
```

What it reads:

- `corpus-index.yaml`
- `meta/<dataset_id>/dataset-summary.yaml`
- corpus-local raw metadata sidecars already written by materialization

What it writes:

- `meta/<dataset_id>/draft-schema.yaml`

Important: schema drafting happens **after** materialization in the current public API. Older documentation that implies pre-materialization schema approval is obsolete.

Known caveat:

- If the public `draft-schema` path fails because raw fields collide onto the same canonical name, inspect the raw metadata sidecars directly and write the reviewed schema manually as a temporary workaround.

---

## Step 4 — Finalize `final-schema.yaml`

**Goal**: Turn the draft into the reviewed schema used for canonicalization.

Use `docs/canonicalization_handbook.md` as the current reference for mapping strategies, transform order, control-label review, gene-tokenizer behavior, and common schema failure modes.

Review these inputs together:

- `meta/<dataset_id>/draft-schema.yaml`
- `meta/<dataset_id>/dataset-summary.yaml`
- `meta/<dataset_id>/metadata-summary.yaml`
- sampled values from `raw-obs.parquet` and `raw-var.parquet`

What to decide explicitly:

- dataset-level literals such as `dataset_id`, assay, species, or cell context when they are constant
- canonical cell metadata mappings (`perturb_label`, `cell_line_or_type`, `timepoint`, etc.)
- canonical feature identity fields (`feature_id`, `feature_label`, namespace)
- whether high-cardinality raw fields should stay raw-only rather than become vocabulary-driving canonical fields

Output:

- `meta/<dataset_id>/final-schema.yaml`

Practical rule: prefer the smallest faithful mapping that matches the dataset's actual raw fields; do not invent ontology that the source metadata does not support.

---

## Step 5 — Canonicalize the corpus

**Goal**: Build canonical obs/var parquets from finalized schemas.

All finalized datasets:

```bash
PYTHONPATH=src python -m perturb_data_lab.cli canonicalize \
  --corpus /path/to/corpus
```

Single dataset only:

```bash
PYTHONPATH=src python -m perturb_data_lab.cli canonicalize \
  --corpus /path/to/corpus \
  --dataset-id my_dataset
```

Dry-run:

```bash
PYTHONPATH=src python -m perturb_data_lab.cli canonicalize \
  --corpus /path/to/corpus \
  --dry-run
```

Primary outputs:

- `meta/<dataset_id>/canonical_meta/canonical-obs.parquet`
- `meta/<dataset_id>/canonical_meta/canonical-var.parquet`

Important:

- `canonicalize` no longer writes `corpus-vocab.yaml`.
- `load_corpus()` uses `canonical_gene_id` as the runtime identity surface and can rebuild a deterministic corpus-global tokenizer when `gene-tokenizer.json` is absent.

---

## Step 6 — Validate with `load_corpus()`

**Goal**: Confirm the canonicalized corpus is actually loadable through the public runtime API.

```python
from perturb_data_lab.loaders import load_corpus

corpus = load_corpus("/path/to/corpus")

expr = corpus.read_expression([0, 1, 2])
meta = corpus.take_metadata([0, 1, 2], columns=["dataset_id", "local_row_index"])
raw = corpus.inspect_batch([0, 1, 2], metadata_columns=["dataset_id", "perturb_label"])

corpus.set_sampler(batch_size=128, seed=0)
batch = next(iter(corpus.loader(seq_len=1024, processing="gpu", num_workers=0)))
```

Recommended validation sequence:

1. `load_corpus()` succeeds
2. `read_expression(...)` succeeds on representative rows
3. `take_metadata(...)` returns expected canonical columns
4. `inspect_batch(...)` succeeds on spot-check rows
5. `corpus.loader(...)` returns at least one valid batch

For multi-dataset corpora, include boundary checks across the dataset transition.

---

## Backend policy summary

- **Default production backend:** aggregate Lance
- **Optional node-local staging backend:** Zarr
- **Also supported on slim main:** federated Lance and federated Zarr
- **Experimental snapshot only:** TileDB, CSR memmap/direct CSR, Arrow IPC, HF datasets, Parquet, WebDataset, and truncated-SVD PCA live only on local branch `experimental/all-backends-pre-slim-20260514`

Use aggregate Lance unless there is a clear operational reason to stage a chunked array representation locally.

---

## End-to-end summary

```text
h5ad file
   │
   ▼
[1] inspect
   dataset-summary.yaml
   │
   ▼
[2] materialize
   raw metadata + matrix artifacts + manifests
   │
   ▼
[3] draft-schema
   draft-schema.yaml
   │
   ▼
[4] finalize-schema
   final-schema.yaml
   │
   ▼
[5] canonicalize
   canonical-obs.parquet + canonical-var.parquet + gene-tokenizer.json (when emitted)
   │
   ▼
[6] load_corpus
   runtime smoke / loader validation
```
