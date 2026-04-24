# Phase 3 Runtime Identity and Loader Boundaries

## Locked Runtime Identity Contract

- runtime row identity is keyed by `global_row_index`
- dataset ownership is expressed by `dataset_index` and `dataset_id`
- heavy-row readers may still open release-scoped files internally, but runtime-hot-path routing must not depend on `release_id`
- `release_id` remains provenance-only in manifests, canonicalization inputs, and archival/debug artifacts

## Loader Boundary Split

- heavy row reading returns only sparse counts payload plus runtime row identity:
  - `global_row_index`
  - `dataset_index`
  - `dataset_id`
  - `local_row_index`
  - `expressed_gene_indices`
  - `expression_counts`
  - `size_factor`
- metadata resolution is a separate RAM-resident `MetadataTable` keyed by contiguous `global_row_index`
- `CorpusLoader` joins heavy rows with the metadata table and emits `CellState`
- feature identity resolution remains separate from both heavy-row reads and metadata resolution via existing preloaded feature translation hooks

## DatasetJoinRecord Direction

- `DatasetJoinRecord` now carries explicit `dataset_index`
- older corpus index files are upgraded on read by inferring `dataset_index` from dataset order
- `global_start` and `global_end` remain the authoritative global row ranges for routing

## Batch-Aware Read Direction

- `BackendCellReader` now exposes:
  - `read_row(local_row_index)` for heavy payload only
  - `read_rows(local_row_indices)` for grouped local batch reads
  - `read_cell(local_row_index)` remains as the compatibility API that can compose with metadata
- `CorpusLoader.read_cells()` groups requested `global_row_index` values by owning dataset before issuing heavy reads
- Arrow/HF uses table `take()` for grouped row fetches; other readers keep compatible grouped hooks even if their current implementation remains per-row internally

## Metadata Table Plan

- Arrow/HF metadata is loaded once from canonical SQLite into `MetadataTable`
- Zarr metadata is loaded once from `meta.json` plus size-factor array into `MetadataTable`
- WebDataset is not migrated to a RAM metadata table in this phase because its current metadata sidecar is too thin for parity; keep current per-record fallback and treat richer metadata sidecars as later work

## Explicit Non-Goals for Phase 3

- no aggregated LanceDB or aggregated Zarr backend implementation yet
- no post-canonicalization global feature resolver file format yet
- no GPU runtime path changes beyond interface preparation
