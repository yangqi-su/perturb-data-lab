# V0 Default-Backend Decision Memo

**Date**: 2026-05-11  
**Plan ID**: `plans-20260511-lance-create-append-canonicalization`  
**Evidence Source**: create/append/canonicalize/load validation on real corpora plus full-corpus loader smoke on `marson_xorion`  
**Scope**: Current backend policy for the public corpus workflow, not a broad benchmark ranking of every wired backend.

---

## Decision

**Production default: aggregate Lance**

**Named operational alternate: Zarr for node-local staging**

**Other backends: experimental but wired**

The current public workflow and docs should assume:

- `inspect → materialize → draft-schema → finalize-schema → canonicalize → load_corpus`
- corpus creation starts with one dataset
- later datasets are appended into the existing corpus
- runtime loading should succeed through the same `load_corpus()` API regardless of supported backend/topology route

---

## Why aggregate Lance is the default

Aggregate Lance is the current recommended backend because it is the path validated most directly against real corpus work:

1. **Create-then-append workflow is now exercised on real datasets.** A new corpus was created from one dataset and a second dataset was appended later through the public CLI.
2. **Canonicalized runtime loading is validated.** `load_corpus()`, `read_expression(...)`, `take_metadata(...)`, `inspect_batch(...)`, and loader smoke succeeded on the canonicalized corpus.
3. **Full-corpus loader smoke exists on the existing large production-style corpus.** The `marson_xorion` aggregate Lance corpus passed public runtime smoke and workered loader checks.
4. **Aggregate topology matches the intended operational model.** One shared matrix root plus per-dataset metadata roots is the current primary corpus pattern.
5. **Append safety matters more than dormant feature breadth.** The current policy prioritizes the backend with the clearest validated create/append/runtime path.

---

## Zarr policy

Zarr remains supported as an **optional node-local staging backend**, not the default policy path.

Use Zarr when:

- you need chunked array artifacts for operational reasons
- node-local staging is the main concern
- you are intentionally evaluating a non-default backend

Do not describe Zarr as the default production choice in handbook or onboarding docs.

---

## Experimental but wired backends

The package still contains additional backend routes. They remain useful for targeted experiments, compatibility work, or follow-up evaluation, but they are not the documented default path.

| Backend / route | Policy status | Notes |
|---|---|---|
| aggregate Lance | default | primary create/append/load path |
| federated Lance | experimental | supported by loader, not the default operational recommendation |
| aggregate Zarr | optional alternate | use for node-local staging when justified |
| federated Zarr | experimental | wired, not default |
| aggregate TileDB | experimental | keep wired, but do not present as preferred path |
| aggregate CSR memmap | experimental | useful for targeted investigations, not default policy |
| federated Arrow IPC | experimental | wired reader path |
| federated HuggingFace datasets | experimental | wired reader path |
| federated Parquet | experimental | wired reader path |
| WebDataset | dormant | not currently enabled by `load_corpus()` |

---

## Documentation rules that follow from this policy

Docs should now state all of the following explicitly:

- Lance is the default production backend.
- Zarr is optional for node-local staging, not the default.
- Other backends remain wired but experimental.
- Canonical schema review happens **after** materialization, not before it.
- `load_corpus()` is the public runtime entrypoint after canonicalization.

---

## Caveats and follow-up notes

- This policy does **not** mean every experimental backend is incorrect; it means the project should not steer new users to them first.
- Some public source-level rough edges remain, including a known `draft-schema` duplicate-canonical-name bug on at least one real dataset.
- The policy is based on validated workflow fit and runtime readiness, not on a fresh full benchmark sweep across all backends.

---

## Summary

| Role | Backend | Current guidance |
|---|---|---|
| default production path | aggregate Lance | use for create, append, canonicalize, and runtime loading |
| operational alternate | Zarr | use only when node-local staging or chunked-array handling is the main need |
| experimental options | TileDB, CSR memmap, federated Lance/Zarr/Arrow IPC/HF datasets/Parquet | keep wired, but do not lead handbook examples with them |
| dormant route | WebDataset | do not document as part of the current `load_corpus()` path |
