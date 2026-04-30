"""Phase 4 smoke test: aggregate Lance corpus via PerturbBatchDataset + GPU pipeline.

Exercises the full hot path:
  BatchExecutor.read_batch() → PerturbBatchDataset → collate_batch_dict/cpu_parallel_collate_fn → GPUSparsePipeline

Runs 10 batches per sampling mode and validates:
- Batch sizes and shapes
- All 12 flat-batch dict keys present with correct dtypes
- collate_batch_dict produces CPU tensors compatible with GPUSparsePipeline.process_batch()
- GPUSparsePipeline outputs correct shape (bsz, seq_len) on CPU device
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

from perturb_data_lab.loaders import (
    AggregateLanceReader,
    BatchExecutor,
    DatasetEntry,
    FeatureRegistry,
    GPUSparsePipeline,
    MetadataIndex,
    PerturbBatchDataset,
    collate_batch_dict,
    cpu_parallel_collate_fn,
)

# ===================================================================
# Constants
# ===================================================================

_ARCHIVED_ROOT = (
    Path("/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2")
    / "copilot/plans/archive/plans-20260427-backend-topology-validation/outputs"
)

AGGREGATE_LANCE = str(
    _ARCHIVED_ROOT / "lance-aggregate/matrix/aggregated-cells.lance"
)

_VAR_00_PATH = str(
    _ARCHIVED_ROOT / "lance-federated/dummy_00/metadata/dummy_00-release-raw-var.parquet"
)
_VAR_01_PATH = str(
    _ARCHIVED_ROOT / "lance-federated/dummy_01/metadata/dummy_01-release-raw-var.parquet"
)

SEQ_LEN = 20
BATCH_SIZE = 32
N_BATCHES = 10
RNG_SEED = 42

# ===================================================================
# Module-scoped fixtures (loaded once)
# ===================================================================


@pytest.fixture(scope="module")
def meta() -> MetadataIndex:
    return MetadataIndex.from_dummy_data()


@pytest.fixture(scope="module")
def agg_reader() -> AggregateLanceReader:
    return AggregateLanceReader(
        AGGREGATE_LANCE, [DatasetEntry("aggregated", 0, 50_000 + 75_000)]
    )


@pytest.fixture(scope="module")
def executor(agg_reader: AggregateLanceReader, meta: MetadataIndex) -> BatchExecutor:
    return BatchExecutor(agg_reader, meta)


@pytest.fixture(scope="module")
def registry() -> FeatureRegistry:
    var00 = pl.read_parquet(_VAR_00_PATH)
    var01 = pl.read_parquet(_VAR_01_PATH)
    return FeatureRegistry(named_var_dfs={"dummy_00": var00, "dummy_01": var01})


@pytest.fixture(scope="module")
def pipeline(registry: FeatureRegistry) -> GPUSparsePipeline:
    return GPUSparsePipeline(registry, seq_len=SEQ_LEN)


# ===================================================================
# Shared validation helpers
# ===================================================================


def _validate_flat_batch(batch: dict[str, Any], expected_batch_size: int) -> None:
    """Validate that a read_batch() dict has all expected keys and correct dtypes."""
    assert batch["batch_size"] == expected_batch_size
    assert batch["global_row_index"].dtype == np.int64
    assert batch["dataset_index"].dtype == np.int32
    assert batch["local_row_index"].dtype == np.int64
    assert batch["size_factor"].dtype == np.float32
    assert batch["expressed_gene_indices"].dtype == np.int32
    assert batch["expression_counts"].dtype == np.int32
    assert batch["row_offsets"].dtype == np.int64
    assert batch["row_offsets"][0] == 0
    assert batch["row_offsets"][-1] == len(batch["expressed_gene_indices"])
    assert len(batch["dataset_id"]) == expected_batch_size
    assert len(batch["cell_id"]) == expected_batch_size
    assert len(batch["canonical_perturbation"]) == expected_batch_size
    assert len(batch["canonical_context"]) == expected_batch_size
    assert "guide_1" in batch["canonical_perturbation"][0]


def _validate_collated_batch(item: dict[str, Any], expected_batch_size: int) -> None:
    """Validate collate_batch_dict output has CPU tensors."""
    assert item["batch_size"] == expected_batch_size
    assert item["global_row_index"].device.type == "cpu"
    assert item["row_offsets"].device.type == "cpu"
    assert item["expressed_gene_indices"].device.type == "cpu"
    assert item["expression_counts"].device.type == "cpu"
    assert item["dataset_index"].device.type == "cpu"
    assert item["size_factor"].device.type == "cpu"


def _validate_pipeline_output(
    result: dict[str, Any], expected_batch_size: int, expected_seq_len: int
) -> None:
    """Validate GPUSparsePipeline.process_batch() output."""
    assert result["batch_size"] == expected_batch_size
    expected_shape = (expected_batch_size, expected_seq_len)
    for key in ["sampled_gene_ids", "sampled_counts", "valid_mask", "exact_match_mask"]:
        assert key in result, f"Missing key: {key}"
        tensor = result[key]
        assert tensor.shape == expected_shape, (
            f"{key} shape {tuple(tensor.shape)} != {expected_shape}"
        )
        assert tensor.device.type == "cpu", f"{key} device: {tensor.device}"
    # dataset_index and global_row_index should also be present
    assert result["dataset_index"].device.type == "cpu"
    assert result["size_factor"].device.type == "cpu"


# ===================================================================
# Tests — PerturbBatchDataset + collate_batch_dict
# ===================================================================


class TestAggregateLanceSmoke:
    """Smoke test: aggregate Lance corpus via PerturbBatchDataset + collation."""

    def test_read_batch_via_dataset(self, executor: BatchExecutor):
        """PerturbBatchDataset.__getitems__ produces valid flat batch dict."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        assert len(ds) == 125_000

        rng = np.random.default_rng(RNG_SEED)
        indices = rng.integers(0, len(executor), size=BATCH_SIZE).tolist()
        items = ds.__getitems__(indices)
        assert isinstance(items, list)
        assert len(items) == 1
        batch = items[0]
        _validate_flat_batch(batch, BATCH_SIZE)

    def test_read_batch_empty_via_dataset(self, executor: BatchExecutor):
        """Empty indices via PerturbBatchDataset produces valid empty batch."""
        ds = PerturbBatchDataset(executor)
        items = ds.__getitems__([])
        assert len(items) == 1
        batch = items[0]
        assert batch["batch_size"] == 0

    def test_collate_batch_dict(self, executor: BatchExecutor):
        """collate_batch_dict converts numpy arrays to CPU tensors."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        rng = np.random.default_rng(RNG_SEED)
        indices = rng.integers(0, len(executor), size=BATCH_SIZE).tolist()
        items = ds.__getitems__(indices)
        collated = collate_batch_dict(items)
        _validate_collated_batch(collated, BATCH_SIZE)

    def test_collate_batch_dict_10_batches(self, executor: BatchExecutor):
        """10 random batches via PerturbBatchDataset + collate_batch_dict."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(N_BATCHES):
            indices = rng.integers(0, len(executor), size=BATCH_SIZE).tolist()
            items = ds.__getitems__(indices)
            collated = collate_batch_dict(items)
            _validate_collated_batch(collated, BATCH_SIZE)

    def test_cross_dataset_batch(self, executor: BatchExecutor):
        """Batch spanning dummy_00 and dummy_01 boundary via new API."""
        ds = PerturbBatchDataset(executor)
        indices = [49998, 49999, 50000, 50001]
        items = ds.__getitems__(indices)
        batch = items[0]
        assert batch["batch_size"] == 4
        assert batch["dataset_id"][0] == "dummy_00"
        assert batch["dataset_id"][2] == "dummy_01"


# ===================================================================
# Tests — GPUSparsePipeline integration
# ===================================================================


class TestGPUPipelineIntegration:
    """Smoke test: PerturbBatchDataset → collate_batch_dict → GPUSparsePipeline."""

    def test_pipeline_basic_cpu(self, executor: BatchExecutor, pipeline: GPUSparsePipeline):
        """GPUSparsePipeline.process_batch() on CPU produces correct output shapes."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        rng = np.random.default_rng(RNG_SEED)
        indices = rng.integers(0, len(executor), size=BATCH_SIZE).tolist()
        items = ds.__getitems__(indices)
        collated = collate_batch_dict(items)

        result = pipeline.process_batch(collated, device="cpu", sampling_mode="uniform")
        _validate_pipeline_output(result, BATCH_SIZE, SEQ_LEN)

    @pytest.mark.parametrize("sampling_mode", ["uniform", "expressed", "hvg"])
    def test_pipeline_all_sampling_modes(
        self,
        executor: BatchExecutor,
        pipeline: GPUSparsePipeline,
        sampling_mode: str,
    ):
        """All three sampling modes produce valid output on CPU."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        rng = np.random.default_rng(RNG_SEED)
        indices = rng.integers(0, len(executor), size=BATCH_SIZE).tolist()
        items = ds.__getitems__(indices)
        collated = collate_batch_dict(items)

        result = pipeline.process_batch(
            collated, device="cpu", sampling_mode=sampling_mode
        )
        _validate_pipeline_output(result, BATCH_SIZE, SEQ_LEN)

    def test_pipeline_10_batches(self, executor: BatchExecutor, pipeline: GPUSparsePipeline):
        """10 batches through the full pipeline on CPU."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(N_BATCHES):
            indices = rng.integers(0, len(executor), size=BATCH_SIZE).tolist()
            items = ds.__getitems__(indices)
            collated = collate_batch_dict(items)
            result = pipeline.process_batch(
                collated, device="cpu", sampling_mode="uniform"
            )
            _validate_pipeline_output(result, BATCH_SIZE, SEQ_LEN)


# ===================================================================
# Tests — cpu_parallel_collate_fn (collation + pipeline in one step)
# ===================================================================


class TestCPUParallelCollate:
    """Smoke test: cpu_parallel_collate_fn with DataLoader."""

    def test_cpu_parallel_collate_single_process(self, executor: BatchExecutor, pipeline: GPUSparsePipeline):
        """cpu_parallel_collate_fn produces valid output with num_workers=0."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)

        sampler = BatchSampler(
            SequentialSampler(ds),
            batch_size=BATCH_SIZE,
            drop_last=True,
        )

        collate_fn = partial(
            cpu_parallel_collate_fn,
            pipeline=pipeline,
            sampling_mode="uniform",
        )

        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 3:
                break
            assert batch["sampled_gene_ids"].device.type == "cpu"
            assert batch["sampled_counts"].device.type == "cpu"
            assert batch["sampled_gene_ids"].shape == (BATCH_SIZE, SEQ_LEN)

    def test_cpu_parallel_collate_single_process_all_modes(
        self, executor: BatchExecutor, pipeline: GPUSparsePipeline
    ):
        """cpu_parallel_collate_fn with all 3 sampling modes."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        sampler = BatchSampler(SequentialSampler(ds), batch_size=BATCH_SIZE, drop_last=True)

        for mode in ["uniform", "expressed", "hvg"]:
            collate_fn = partial(cpu_parallel_collate_fn, pipeline=pipeline, sampling_mode=mode)
            loader = DataLoader(
                ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0,
            )
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= 1:
                    break
                assert batch["sampled_gene_ids"].device.type == "cpu"
                assert batch["sampled_gene_ids"].shape == (BATCH_SIZE, SEQ_LEN)


# ===================================================================
# Tests — edge cases
# ===================================================================


class TestEdgeCases:
    """Smoke test: edge cases for the new hot-path architecture."""

    def test_empty_batch(self, executor: BatchExecutor):
        """Empty index list returns empty batch dict."""
        batch = executor.read_batch([])
        assert batch["batch_size"] == 0

    def test_single_cell_batch(self, executor: BatchExecutor):
        """Single cell batch produces valid read_batch output."""
        batch = executor.read_batch([0])
        assert batch["batch_size"] == 1
        assert batch["global_row_index"][0] == 0

    def test_dataset_boundary_indices(self, executor: BatchExecutor):
        """First and last cell of each dataset."""
        batch = executor.read_batch([0, 49999, 50000, 124999])
        assert batch["batch_size"] == 4
        np.testing.assert_array_equal(
            batch["global_row_index"],
            np.array([0, 49999, 50000, 124999], dtype=np.int64),
        )


# ===================================================================
# Tests — canonical metadata
# ===================================================================


class TestCanonicalMetadata:
    """Validate canonical_perturbation and canonical_context in read_batch output."""

    def test_perturbation_fields(self, executor: BatchExecutor):
        """canonical_perturbation contains expected keys from dummy data."""
        batch = executor.read_batch([0, 10, 50000, 75000])
        for pert in batch["canonical_perturbation"]:
            assert "guide_1" in pert
            assert isinstance(pert["guide_1"], str)

    def test_context_fields(self, executor: BatchExecutor):
        """canonical_context contains expected keys from dummy data."""
        batch = executor.read_batch([0, 10, 50000, 75000])
        for ctx in batch["canonical_context"]:
            assert "cell_type" in ctx
            assert isinstance(ctx["cell_type"], str)
