"""Phase 4 smoke test: federated Lance corpus via PerturbBatchDataset + GPU pipeline.

Exercises the full hot path with federated reader:
  BatchExecutor.read_batch() → PerturbBatchDataset → collate_batch_dict → GPUSparsePipeline

Runs 10 batches per sampling mode and validates:
- Batch sizes and shapes
- Interleaved dataset batches are handled correctly
- Both aggregate-style and federated-specific edge cases
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

from perturb_data_lab.loaders import (
    BatchExecutor,
    FederatedLanceReader,
    FeatureRegistry,
    GPUSparsePipeline,
    LanceDatasetEntry,
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

FEDERATED_BASE = _ARCHIVED_ROOT / "lance-federated"

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


def _fed_entries() -> list[LanceDatasetEntry]:
    return [
        LanceDatasetEntry(
            "dummy_00", 0, 50_000,
            FEDERATED_BASE / "dummy_00/matrix/dummy_00-release.lance",
        ),
        LanceDatasetEntry(
            "dummy_01", 50_000, 125_000,
            FEDERATED_BASE / "dummy_01/matrix/dummy_01-release.lance",
        ),
    ]


# ===================================================================
# Module-scoped fixtures
# ===================================================================


@pytest.fixture(scope="module")
def meta() -> MetadataIndex:
    return MetadataIndex.from_dummy_data()


@pytest.fixture(scope="module")
def fed_reader() -> FederatedLanceReader:
    return FederatedLanceReader(_fed_entries())


@pytest.fixture(scope="module")
def executor(fed_reader: FederatedLanceReader, meta: MetadataIndex) -> BatchExecutor:
    return BatchExecutor(fed_reader, meta)


@pytest.fixture(scope="module")
def registry() -> FeatureRegistry:
    var00 = pl.read_parquet(_VAR_00_PATH)
    var01 = pl.read_parquet(_VAR_01_PATH)
    return FeatureRegistry(named_var_dfs={"dummy_00": var00, "dummy_01": var01})


@pytest.fixture(scope="module")
def pipeline(registry: FeatureRegistry) -> GPUSparsePipeline:
    return GPUSparsePipeline(registry, seq_len=SEQ_LEN)


# ===================================================================
# Tests — PerturbBatchDataset with federated reader
# ===================================================================


class TestFederatedLanceSmoke:
    """Smoke test: federated Lance corpus via PerturbBatchDataset + collation."""

    def test_read_batch_via_dataset(self, executor: BatchExecutor):
        """PerturbBatchDataset.__getitems__ produces valid flat batch with federated reader."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        assert len(ds) == 125_000

        rng = np.random.default_rng(RNG_SEED)
        indices = rng.integers(0, len(executor), size=BATCH_SIZE).tolist()
        items = ds.__getitems__(indices)
        batch = items[0]
        assert batch["batch_size"] == BATCH_SIZE
        assert batch["global_row_index"].dtype == np.int64
        assert batch["dataset_index"].dtype == np.int32
        assert batch["size_factor"].dtype == np.float32
        assert len(batch["cell_id"]) == BATCH_SIZE

    def test_collate_batch_dict_10_batches(self, executor: BatchExecutor):
        """10 random batches via PerturbBatchDataset + collate_batch_dict with federated reader."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(N_BATCHES):
            indices = rng.integers(0, len(executor), size=BATCH_SIZE).tolist()
            items = ds.__getitems__(indices)
            collated = collate_batch_dict(items)
            assert collated["batch_size"] == BATCH_SIZE
            assert collated["global_row_index"].device.type == "cpu"
            assert isinstance(collated["dataset_id"], tuple)

    def test_pipeline_all_sampling_modes(
        self,
        executor: BatchExecutor,
        pipeline: GPUSparsePipeline,
    ):
        """All three sampling modes produce valid output via federated reader."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        rng = np.random.default_rng(RNG_SEED)
        indices = rng.integers(0, len(executor), size=BATCH_SIZE).tolist()
        items = ds.__getitems__(indices)
        collated = collate_batch_dict(items)

        for mode in ["uniform", "expressed", "hvg"]:
            result = pipeline.process_batch(collated, device="cpu", sampling_mode=mode)
            assert result["sampled_gene_ids"].shape == (BATCH_SIZE, SEQ_LEN)
            assert result["sampled_gene_ids"].device.type == "cpu"

    def test_pipeline_10_batches(self, executor: BatchExecutor, pipeline: GPUSparsePipeline):
        """10 batches through full pipeline with federated reader."""
        ds = PerturbBatchDataset(executor, seq_len=SEQ_LEN)
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(N_BATCHES):
            indices = rng.integers(0, len(executor), size=BATCH_SIZE).tolist()
            items = ds.__getitems__(indices)
            collated = collate_batch_dict(items)
            result = pipeline.process_batch(collated, device="cpu", sampling_mode="uniform")
            assert result["sampled_gene_ids"].shape == (BATCH_SIZE, SEQ_LEN)


# ===================================================================
# Tests — interleaved dataset batches (federated-specific)
# ===================================================================


class TestFederatedInterleavedBatches:
    """Federated-specific: interleaved batches spanning multiple datasets."""

    def test_interleaved_order_preserved(self, executor: BatchExecutor):
        """Output order matches input order for interleaved dataset indices."""
        indices = [0, 50000, 1, 50001, 2, 50002]
        batch = executor.read_batch(indices)
        assert batch["batch_size"] == 6
        np.testing.assert_array_equal(
            batch["global_row_index"],
            np.array(indices, dtype=np.int64),
        )

    def test_interleaved_dataset_ids(self, executor: BatchExecutor):
        """Dataset IDs are correct for interleaved indices."""
        indices = [0, 50000, 1, 50001]
        batch = executor.read_batch(indices)
        assert batch["dataset_id"][0] == "dummy_00"
        assert batch["dataset_id"][1] == "dummy_01"
        assert batch["dataset_id"][2] == "dummy_00"
        assert batch["dataset_id"][3] == "dummy_01"

    def test_interleaved_expression_content(self, executor: BatchExecutor):
        """Each cell in an interleaved batch has valid expression data."""
        indices = [0, 50000, 10, 50010, 100, 50100]
        batch = executor.read_batch(indices)
        # row_offsets should be valid
        assert batch["row_offsets"][0] == 0
        assert batch["row_offsets"][-1] == len(batch["expressed_gene_indices"])
        assert len(batch["expressed_gene_indices"]) == len(batch["expression_counts"])
        assert len(batch["expressed_gene_indices"]) > 0

    def test_batch_across_interleaved_boundary(self, executor: BatchExecutor):
        """Large interleaved batch near the dataset boundary."""
        rng = np.random.default_rng(777)
        indices_d00 = rng.integers(49_000, 50_000, size=32).tolist()
        indices_d01 = rng.integers(50_000, 51_000, size=32).tolist()
        all_indices = indices_d00 + indices_d01
        rng.shuffle(all_indices)

        batch = executor.read_batch(all_indices)
        assert batch["batch_size"] == 64
        assert len(batch["dataset_id"]) == 64


# ===================================================================
# Tests — edge cases
# ===================================================================


class TestFederatedEdgeCases:
    """Smoke test: edge cases for federated topology."""

    def test_empty_batch(self, executor: BatchExecutor):
        """Empty index list returns empty batch dict."""
        batch = executor.read_batch([])
        assert batch["batch_size"] == 0

    def test_single_dataset_only(self, executor: BatchExecutor):
        """Batch with only dummy_00 cells."""
        batch = executor.read_batch(list(range(10)))
        assert batch["batch_size"] == 10
        for ds_id in batch["dataset_id"]:
            assert ds_id == "dummy_00"
        for ds_idx in batch["dataset_index"]:
            assert ds_idx == 0

    def test_second_dataset_only(self, executor: BatchExecutor):
        """Batch with only dummy_01 cells."""
        batch = executor.read_batch(list(range(50000, 50010)))
        assert batch["batch_size"] == 10
        for ds_id in batch["dataset_id"]:
            assert ds_id == "dummy_01"

    def test_dataset_boundary_indices(self, executor: BatchExecutor):
        """First and last cell of each dataset via federated reader."""
        batch = executor.read_batch([0, 49999, 50000, 124999])
        np.testing.assert_array_equal(
            batch["global_row_index"],
            np.array([0, 49999, 50000, 124999], dtype=np.int64),
        )

    def test_all_single_dataset(self, executor: BatchExecutor):
        """Multiple cells all from dummy_01 via federated reader."""
        rng = np.random.default_rng(42)
        indices = rng.integers(50000, 125000, size=50).tolist()
        batch = executor.read_batch(indices)
        assert batch["batch_size"] == 50
        for ds_id in batch["dataset_id"]:
            assert ds_id == "dummy_01"

    def test_read_batch_deterministic(self, executor: BatchExecutor):
        """Same indices produce identical read_batch output via federated reader."""
        indices = [0, 50000, 1, 50001, 99999, 124999]
        batch1 = executor.read_batch(indices)
        batch2 = executor.read_batch(indices)
        np.testing.assert_array_equal(batch1["global_row_index"], batch2["global_row_index"])
        np.testing.assert_array_equal(batch1["expressed_gene_indices"], batch2["expressed_gene_indices"])
        np.testing.assert_array_equal(batch1["expression_counts"], batch2["expression_counts"])
        assert batch1["dataset_id"] == batch2["dataset_id"]

    def test_no_crash_large_interleaved(self, executor: BatchExecutor):
        """Large interleaved batch does not crash."""
        rng = np.random.default_rng(42)
        d00 = rng.integers(0, 50_000, size=100).tolist()
        d01 = rng.integers(50_000, 125_000, size=100).tolist()
        all_indices = d00 + d01
        rng.shuffle(all_indices)
        batch = executor.read_batch(all_indices)
        assert batch["batch_size"] == 200


# ===================================================================
# Tests — canonical metadata (federated)
# ===================================================================


class TestFederatedCanonicalMetadata:
    """Validate canonical fields via federated reader."""

    def test_perturbation_fields(self, executor: BatchExecutor):
        """canonical_perturbation contains expected keys."""
        batch = executor.read_batch([0, 50000])
        for pert in batch["canonical_perturbation"]:
            assert "guide_1" in pert

    def test_context_fields(self, executor: BatchExecutor):
        """canonical_context contains expected keys."""
        batch = executor.read_batch([0, 50000])
        for ctx in batch["canonical_context"]:
            assert "cell_type" in ctx


# ===================================================================
# Tests — cpu_parallel_collate_fn with federated reader
# ===================================================================


class TestFederatedCPUParallelCollate:
    """cpu_parallel_collate_fn with federated reader."""

    def test_single_process_all_modes(self, executor: BatchExecutor, pipeline: GPUSparsePipeline):
        """cpu_parallel_collate_fn produces valid output with all 3 sampling modes."""
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
