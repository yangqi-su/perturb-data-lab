#!/usr/bin/env python3
"""Run a bounded pertTF-adapter smoke on the Marson/Xorion aggregate corpus.

This script stays entirely on the ``perturb-data-lab`` side: it loads an
existing corpus, verifies per-dataset ``hvg.parquet`` discovery, builds the
local pertTF adapter objects, constructs a small mixed-dataset paired batch,
and writes lightweight smoke summaries without touching the external ``pertTF``
repository.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from perturb_data_lab.loaders import (
    PertTFAdapterConfig,
    PertTFPairedBatchBuilder,
    PertTFCorpusAdapter,
    load_corpus,
)
from perturb_data_lab.loaders.adapters.perttf import PerturbationPairBatch


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(message: str) -> None:
    print(f"[{_utc_now()}] {message}", flush=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected mapping YAML at {path}")
    return loaded


def _max_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


@dataclass(frozen=True)
class DatasetEntry:
    dataset_id: str
    dataset_index: int
    global_start: int
    global_end: int
    manifest_path: Path


def _load_dataset_entries(corpus_root: Path) -> list[DatasetEntry]:
    index_doc = _read_yaml(corpus_root / "corpus-index.yaml")
    entries: list[DatasetEntry] = []
    for item in index_doc.get("datasets", []):
        dataset_id = str(item["dataset_id"])
        manifest_value = item.get(
            "manifest_path",
            f"meta/{dataset_id}/materialization-manifest.yaml",
        )
        manifest_path = Path(str(manifest_value))
        if not manifest_path.is_absolute():
            manifest_path = corpus_root / manifest_path
        entries.append(
            DatasetEntry(
                dataset_id=dataset_id,
                dataset_index=int(item["dataset_index"]),
                global_start=int(item["global_start"]),
                global_end=int(item["global_end"]),
                manifest_path=manifest_path,
            )
        )
    if not entries:
        raise ValueError(f"no datasets found in {corpus_root / 'corpus-index.yaml'}")
    return entries


def _choose_mixed_dataset_ids(
    entries: list[DatasetEntry],
    *,
    requested: tuple[str, ...],
    batch_size: int,
) -> list[str]:
    if requested:
        available = {entry.dataset_id for entry in entries}
        missing = [dataset_id for dataset_id in requested if dataset_id not in available]
        if missing:
            raise ValueError(f"requested dataset ids not found in corpus: {missing}")
        return list(requested[:batch_size])

    selected: list[str] = []
    by_prefix = {
        "marson": [entry.dataset_id for entry in entries if entry.dataset_id.startswith("marson")],
        "xorion": [entry.dataset_id for entry in entries if entry.dataset_id.startswith("xorion")],
    }
    for prefix in ("marson", "xorion"):
        if by_prefix[prefix]:
            selected.append(by_prefix[prefix][0])
    for entry in entries:
        if entry.dataset_id in selected:
            continue
        selected.append(entry.dataset_id)
        if len(selected) >= batch_size:
            break
    return selected[:batch_size]


def _iter_probe_windows(entry: DatasetEntry, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(entry.global_start, entry.global_end, chunk_size):
        end = min(start + chunk_size, entry.global_end)
        yield start, end


def _find_first_perturbed_index(
    corpus,
    entry: DatasetEntry,
    *,
    perturbation_column: str,
    control_labels: frozenset[str],
    chunk_size: int = 4096,
) -> tuple[int, str]:
    for start, end in _iter_probe_windows(entry, chunk_size):
        indices = list(range(start, end))
        metadata = corpus.take_metadata(indices, columns=[perturbation_column])
        labels = metadata[perturbation_column]
        for offset, label in enumerate(labels):
            label_str = str(label)
            if label_str not in control_labels:
                return start + offset, label_str
    raise RuntimeError(
        f"failed to find a non-control source row in dataset {entry.dataset_id!r}"
    )


def _build_self_to_control_pair_batch(
    corpus,
    adapter: PertTFCorpusAdapter,
    *,
    source_indices: list[int],
    control_label: str,
) -> PerturbationPairBatch:
    config = adapter.config
    metadata = corpus.take_metadata(
        source_indices,
        columns=[
            "dataset_index",
            config.cell_context_column,
            config.perturbation_column,
            config.batch_column,
        ],
    )
    dataset_indices = np.asarray(metadata["dataset_index"], dtype=np.int32)
    source_context_labels = tuple(str(value) for value in metadata[config.cell_context_column])
    source_perturbation_labels = tuple(
        str(value) for value in metadata[config.perturbation_column]
    )
    source_batch_labels = tuple(str(value) for value in metadata[config.batch_column])
    target_perturbation_labels = tuple(control_label for _ in source_indices)

    return PerturbationPairBatch(
        source_indices=np.asarray(source_indices, dtype=np.int64),
        target_indices=np.asarray(source_indices, dtype=np.int64),
        source_dataset_indices=dataset_indices.copy(),
        target_dataset_indices=dataset_indices.copy(),
        source_cell_context_ids=adapter.labels.cell_context.encode_many(source_context_labels),
        target_cell_context_ids=adapter.labels.cell_context.encode_many(source_context_labels),
        source_perturbation_ids=adapter.labels.perturbation.encode_many(source_perturbation_labels),
        target_perturbation_ids=adapter.labels.perturbation.encode_many(target_perturbation_labels),
        source_batch_ids=adapter.labels.batch.encode_many(source_batch_labels),
        target_batch_ids=adapter.labels.batch.encode_many(source_batch_labels),
        source_cell_context_labels=source_context_labels,
        target_cell_context_labels=source_context_labels,
        source_perturbation_labels=source_perturbation_labels,
        target_perturbation_labels=target_perturbation_labels,
        source_batch_labels=source_batch_labels,
        target_batch_labels=source_batch_labels,
    )


def _row_global_counts(raw_batch: dict[str, Any], row_pos: int, local_to_global: np.ndarray) -> dict[int, float]:
    row_offsets = np.asarray(raw_batch["row_offsets"], dtype=np.int64)
    start = int(row_offsets[row_pos])
    end = int(row_offsets[row_pos + 1])
    local_ids = np.asarray(raw_batch["expressed_gene_indices"][start:end], dtype=np.int64)
    counts = np.asarray(raw_batch["expression_counts"][start:end], dtype=np.float32)
    mapped = local_to_global[local_ids]
    return {
        int(global_id): float(count)
        for global_id, count in zip(mapped.tolist(), counts.tolist())
        if int(global_id) >= 0
    }


def _verify_sampled_alignment(
    corpus,
    adapter: PertTFCorpusAdapter,
    pair_batch: PerturbationPairBatch,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    source_raw = corpus.inspect_batch(pair_batch.source_indices.tolist())
    target_raw = corpus.inspect_batch(pair_batch.target_indices.tolist())
    pad_token_id = adapter.vocab.to_simple_vocab_stoi()[adapter.config.pad_token]
    cls_offset = 1 if adapter.config.append_cls else 0
    token_offset = adapter.vocab.special_token_offset
    local_to_global = corpus.feature_registry.local_to_global_map
    checked_positions = 0
    mismatches: list[str] = []

    for row_pos in range(len(pair_batch.source_indices)):
        source_counts = _row_global_counts(
            source_raw,
            row_pos,
            local_to_global[int(pair_batch.source_dataset_indices[row_pos])],
        )
        target_counts = _row_global_counts(
            target_raw,
            row_pos,
            local_to_global[int(pair_batch.target_dataset_indices[row_pos])],
        )
        for col in range(cls_offset, batch["gene_ids"].shape[1]):
            token_id = int(batch["gene_ids"][row_pos, col])
            if token_id == pad_token_id:
                continue
            global_id = token_id - token_offset
            expected_source = source_counts.get(global_id, 0.0)
            expected_target = target_counts.get(global_id, 0.0)
            actual_source = float(batch["target_values"][row_pos, col].item())
            actual_target = float(batch["target_values_next"][row_pos, col].item())
            checked_positions += 1
            if not np.isclose(actual_source, expected_source):
                mismatches.append(
                    f"source row={row_pos} col={col} global_id={global_id} expected={expected_source} actual={actual_source}"
                )
            if not np.isclose(actual_target, expected_target):
                mismatches.append(
                    f"target row={row_pos} col={col} global_id={global_id} expected={expected_target} actual={actual_target}"
                )

    return {
        "gene_ids_match": bool(torch.equal(batch["gene_ids"], batch["next_gene_ids"])),
        "checked_positions": checked_positions,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
        "passed": bool(torch.equal(batch["gene_ids"], batch["next_gene_ids"]) and not mismatches),
    }


def _verify_full_expression_masks(
    corpus,
    adapter: PertTFCorpusAdapter,
    pair_batch: PerturbationPairBatch,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    cls_offset = 1 if adapter.config.append_cls else 0
    source_expected = torch.as_tensor(
        corpus.feature_registry.dataset_has_gene[pair_batch.source_dataset_indices],
        dtype=torch.bool,
    )
    target_expected = torch.as_tensor(
        corpus.feature_registry.dataset_has_gene[pair_batch.target_dataset_indices],
        dtype=torch.bool,
    )
    source_mask = batch["full_expr_mask"][:, cls_offset:].cpu()
    target_mask = batch["full_expr_next_mask"][:, cls_offset:].cpu()
    source_expr = batch["full_expr"][:, cls_offset:].cpu()
    target_expr = batch["full_expr_next"][:, cls_offset:].cpu()
    source_present_zero = ((source_mask) & (source_expr == 0)).sum(dim=1).tolist()
    source_absent_zero = ((~source_mask) & (source_expr == 0)).sum(dim=1).tolist()
    target_present_zero = ((target_mask) & (target_expr == 0)).sum(dim=1).tolist()
    target_absent_zero = ((~target_mask) & (target_expr == 0)).sum(dim=1).tolist()

    return {
        "full_gene_ids_shape": list(batch["full_gene_ids"].shape),
        "full_expr_shape": list(batch["full_expr"].shape),
        "full_expr_next_shape": list(batch["full_expr_next"].shape),
        "source_mask_matches_dataset_presence": bool(torch.equal(source_mask, source_expected)),
        "target_mask_matches_dataset_presence": bool(torch.equal(target_mask, target_expected)),
        "source_present_zero_counts": [int(value) for value in source_present_zero],
        "source_absent_zero_counts": [int(value) for value in source_absent_zero],
        "target_present_zero_counts": [int(value) for value in target_present_zero],
        "target_absent_zero_counts": [int(value) for value in target_absent_zero],
        "passed": bool(
            torch.equal(source_mask, source_expected)
            and torch.equal(target_mask, target_expected)
            and any(int(value) > 0 for value in source_present_zero)
            and any(int(value) > 0 for value in source_absent_zero)
            and any(int(value) > 0 for value in target_present_zero)
            and any(int(value) > 0 for value in target_absent_zero)
        ),
    }


def _dataset_hvg_summary(corpus, entries: list[DatasetEntry]) -> list[dict[str, Any]]:
    feature_registry = corpus.feature_registry
    rank_matrix = feature_registry.hvg_rank_matrix
    has_gene = feature_registry.dataset_has_gene
    summary: list[dict[str, Any]] = []
    for entry in entries:
        manifest_doc = _read_yaml(entry.manifest_path)
        hvg_path = Path(str(manifest_doc.get("hvg_ranking_path", "")))
        summary.append(
            {
                "dataset_id": entry.dataset_id,
                "dataset_index": entry.dataset_index,
                "manifest_path": str(entry.manifest_path),
                "hvg_ranking_path": str(hvg_path),
                "hvg_ranking_exists": hvg_path.exists(),
                "default_n_hvg": int(manifest_doc.get("default_n_hvg", 0) or 0),
                "feature_count": int(has_gene[entry.dataset_index].sum()),
                "ranked_feature_count": int((rank_matrix[entry.dataset_index] > 0).sum()),
            }
        )
    return summary


def _write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    forced = summary["batch"].get("forced_mixed_dataset_batch", False)
    dataset_lines = "\n".join(
        f"- `{item['dataset_id']}`: ranked `{item['ranked_feature_count']}` / feature `{item['feature_count']}`, default_n_hvg `{item['default_n_hvg']}`, manifest path exists `{item['hvg_ranking_exists']}`"
        for item in summary["hvg_discovery"]
    )
    selected_lines = "\n".join(
        f"- `{item['dataset_id']}` row `{item['source_index']}` perturb `{item['source_perturbation_label']}`"
        for item in summary["batch"]["selected_sources"]
    )
    path.write_text(
        "\n".join(
            [
                "# Marson/Xorion pertTF Adapter Smoke",
                "",
                f"- `Timestamp`: `{summary['timestamp']}`",
                f"- `Corpus Root`: `{summary['corpus_root']}`",
                f"- `Datasets Covered`: `{summary['dataset_count']}`",
                f"- `Global Vocab Size`: `{summary['global_vocab_size']}`",
                f"- `Batch Size`: `{summary['batch']['paired_rows_sampled']}`",
                f"- `HVG Top K`: `{summary['batch']['hvg_top_k']}`",
                f"- `Forced Mixed Batch`: `{str(forced).lower()}`",
                "",
                "## HVG Discovery",
                "",
                dataset_lines,
                "",
                "## Selected Source Rows",
                "",
                selected_lines,
                "",
                "## Checks",
                "",
                f"- Same-dataset invariant: `{summary['batch']['same_dataset_check']}`",
                f"- Same-context invariant: `{summary['batch']['same_context_check']}`",
                f"- Sampled source/target gene alignment: `{summary['alignment']['passed']}` ({summary['alignment']['checked_positions']} checked positions)",
                f"- Union full-expression masks: `{summary['full_expression_masks']['passed']}`",
                "",
                "## Runtime Notes",
                "",
                f"- `load_corpus_seconds`: `{summary['timing_seconds']['load_corpus']:.2f}`",
                f"- `adapter_seconds`: `{summary['timing_seconds']['build_adapter']:.2f}`",
                f"- `batch_build_seconds`: `{summary['timing_seconds']['build_batch']:.2f}`",
                f"- `max_rss_mib`: `{summary['runtime_notes']['max_rss_mib']:.1f}`",
                "",
                "## Limitations",
                "",
                "- This smoke keeps all work inside `perturb-data-lab`; it does not modify or import from the external `pertTF` repository.",
                "- The paired batch uses the adapter-side `self_to_control_label` semantics for perturbed sources, so it validates the current source→target tensor contract without full pertTF training.",
                "- `full_expr_mask` / `full_expr_next_mask` are emitted and validated here, but pertTF-side loss consumption remains future work.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 8 Marson/Xorion pertTF adapter smoke.",
    )
    parser.add_argument("--corpus-root", required=True, help="Aggregate corpus root.")
    parser.add_argument("--output-dir", required=True, help="Directory for smoke outputs.")
    parser.add_argument("--seq-len", type=int, default=64, help="Sampled sequence length.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of paired rows to validate.")
    parser.add_argument("--hvg-top-k", type=int, default=2000, help="Runtime HVG top-k threshold.")
    parser.add_argument("--hvg-weight", type=float, default=4.0, help="Runtime HVG bonus weight.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic RNG seed.")
    parser.add_argument(
        "--control-label",
        action="append",
        default=["WT"],
        help="Control perturbation label to reserve in the adapter (repeatable).",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Optional dataset IDs to force into the mixed batch (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.monotonic()
    corpus_root = Path(args.corpus_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command_snapshot = {
        "timestamp": _utc_now(),
        "corpus_root": str(corpus_root),
        "output_dir": str(output_dir),
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "hvg_top_k": int(args.hvg_top_k),
        "hvg_weight": float(args.hvg_weight),
        "seed": int(args.seed),
        "control_labels": list(dict.fromkeys(str(value) for value in args.control_label)),
        "requested_dataset_ids": list(dict.fromkeys(str(value) for value in args.dataset_id)),
    }
    (output_dir / "phase-08-smoke-command.json").write_text(
        json.dumps(command_snapshot, indent=2),
        encoding="utf-8",
    )

    _log(f"Loading corpus from {corpus_root}")
    load_start = time.monotonic()
    corpus = load_corpus(corpus_root)
    load_seconds = time.monotonic() - load_start
    _log(
        "Loaded corpus with "
        f"{len(corpus.feature_registry.dataset_ids)} datasets, "
        f"global vocab {corpus.feature_registry.global_vocab_size}, "
        f"max RSS {_max_rss_mib():.1f} MiB"
    )

    entries = _load_dataset_entries(corpus_root)
    hvg_summary = _dataset_hvg_summary(corpus, entries)
    missing_hvg = [item["dataset_id"] for item in hvg_summary if not item["hvg_ranking_exists"]]
    incomplete_hvg = [
        item["dataset_id"]
        for item in hvg_summary
        if item["ranked_feature_count"] != item["feature_count"]
    ]
    if missing_hvg:
        raise RuntimeError(f"missing hvg.parquet outputs for datasets: {missing_hvg}")
    if incomplete_hvg:
        raise RuntimeError(
            "HVG rank discovery did not cover every feature for datasets: "
            f"{incomplete_hvg}"
        )
    _log("Verified hvg.parquet discovery for all datasets")

    adapter_start = time.monotonic()
    config = PertTFAdapterConfig(
        control_labels=tuple(dict.fromkeys(str(value) for value in args.control_label)),
        include_full_expr=True,
        mask_ratio=0.0,
    )
    adapter = PertTFCorpusAdapter.from_corpus(corpus, config)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=int(args.seq_len), config=config, adapter=adapter)
    adapter_seconds = time.monotonic() - adapter_start
    _log(f"Built pertTF adapter objects in {adapter_seconds:.2f}s")

    selected_dataset_ids = _choose_mixed_dataset_ids(
        entries,
        requested=tuple(dict.fromkeys(str(value) for value in args.dataset_id)),
        batch_size=int(args.batch_size),
    )
    entry_by_id = {entry.dataset_id: entry for entry in entries}
    control_label_set = frozenset(config.control_labels)
    source_indices: list[int] = []
    selected_sources: list[dict[str, Any]] = []
    for dataset_id in selected_dataset_ids:
        source_index, perturbation_label = _find_first_perturbed_index(
            corpus,
            entry_by_id[dataset_id],
            perturbation_column=config.perturbation_column,
            control_labels=control_label_set,
        )
        source_indices.append(source_index)
        selected_sources.append(
            {
                "dataset_id": dataset_id,
                "source_index": int(source_index),
                "source_perturbation_label": perturbation_label,
            }
        )
    _log(
        "Selected mixed-dataset source rows: "
        + ", ".join(f"{item['dataset_id']}@{item['source_index']}" for item in selected_sources)
    )

    pair_batch = _build_self_to_control_pair_batch(
        corpus,
        adapter,
        source_indices=source_indices,
        control_label=config.control_labels[0],
    )

    batch_start = time.monotonic()
    batch = builder.build_paired_batch(
        pair_batch,
        seed=int(args.seed),
        sampling_mode="hvg",
        hvg_weight=float(args.hvg_weight),
        hvg_top_k=int(args.hvg_top_k),
    )
    batch_seconds = time.monotonic() - batch_start
    _log(f"Built paired batch in {batch_seconds:.2f}s")

    same_dataset = bool(
        np.array_equal(pair_batch.source_dataset_indices, pair_batch.target_dataset_indices)
    )
    same_context = bool(
        np.array_equal(pair_batch.source_cell_context_ids, pair_batch.target_cell_context_ids)
    )
    mixed_dataset = bool(len(np.unique(pair_batch.source_dataset_indices)) > 1)
    alignment = _verify_sampled_alignment(corpus, adapter, pair_batch, batch)
    full_expr_masks = _verify_full_expression_masks(corpus, adapter, pair_batch, batch)

    summary = {
        "timestamp": _utc_now(),
        "corpus_root": str(corpus_root),
        "dataset_count": len(entries),
        "global_vocab_size": int(corpus.feature_registry.global_vocab_size),
        "hvg_discovery": hvg_summary,
        "batch": {
            "hvg_top_k": int(args.hvg_top_k),
            "hvg_weight": float(args.hvg_weight),
            "seq_len": int(args.seq_len),
            "paired_rows_sampled": int(len(pair_batch.source_indices)),
            "mixed_dataset_batch": mixed_dataset,
            "forced_mixed_dataset_batch": True,
            "same_dataset_check": same_dataset,
            "same_context_check": same_context,
            "selected_sources": selected_sources,
            "source_indices": [int(value) for value in pair_batch.source_indices.tolist()],
            "target_indices": [int(value) for value in pair_batch.target_indices.tolist()],
            "source_dataset_ids": selected_dataset_ids,
            "source_cell_context_labels": list(pair_batch.source_cell_context_labels),
            "source_perturbation_labels": list(pair_batch.source_perturbation_labels),
            "target_perturbation_labels": list(pair_batch.target_perturbation_labels),
            "batch_field_shapes": {
                key: list(value.shape)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            },
        },
        "alignment": alignment,
        "full_expression_masks": full_expr_masks,
        "runtime_notes": {
            "max_rss_mib": _max_rss_mib(),
            "device": str(builder.device),
            "control_labels": list(config.control_labels),
        },
        "timing_seconds": {
            "load_corpus": load_seconds,
            "build_adapter": adapter_seconds,
            "build_batch": batch_seconds,
            "total": time.monotonic() - start_time,
        },
    }

    summary_path = output_dir / "phase-08-smoke-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_summary(output_dir / "phase-08-smoke-summary.md", summary)
    _log(f"Wrote smoke summaries to {summary_path}")

    if not same_dataset:
        raise RuntimeError("same-dataset pairing invariant failed")
    if not same_context:
        raise RuntimeError("same-context pairing invariant failed")
    if not mixed_dataset:
        raise RuntimeError("mixed-dataset smoke batch was not achieved")
    if not alignment["passed"]:
        raise RuntimeError("sampled source/target gene alignment validation failed")
    if not full_expr_masks["passed"]:
        raise RuntimeError("full-expression mask validation failed")

    _log("Smoke completed successfully")


if __name__ == "__main__":
    main()
