import numpy as np
import polars as pl
import pytest

from perturb_data_lab.loaders import (
    MetadataIndex,
    PertTFAdapterConfig,
    PerturbationPairSampler,
)


def _build_metadata_index() -> MetadataIndex:
    return MetadataIndex(
        pl.DataFrame(
            {
                "global_row_index": np.arange(8, dtype=np.int64),
                "cell_id": [f"cell_{idx}" for idx in range(8)],
                "dataset_id": ["ds0", "ds0", "ds0", "ds0", "ds0", "ds1", "ds1", "ds1"],
                "dataset_index": np.asarray([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32),
                "local_row_index": np.arange(8, dtype=np.int64),
                "size_factor": np.asarray([1.0, 1.1, 1.2, 0.9, 1.0, 1.3, 1.4, 1.5], dtype=np.float32),
                "cell_context": ["T_cell", "T_cell", "T_cell", "B_cell", "B_cell", "T_cell", "T_cell", "B_cell"],
                "perturb_label": ["WT", "KO_A", "KO_B", "WT", "KO_C", "WT", "KO_A", "KO_D"],
                "batch_id": ["b0", "b1", "b1", "b2", "b2", "b3", "b4", "b5"],
            }
        )
    )


def test_control_sources_pair_to_treated_targets_in_same_dataset_and_context() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=4,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        seed=7,
    )

    batch = sampler.pair_source_indices([0, 3], seed=11)

    np.testing.assert_array_equal(batch.source_indices, np.asarray([0, 3], dtype=np.int64))
    assert batch.target_indices[0] in {1, 2}
    assert batch.target_indices[1] == 4
    assert batch.source_dataset_indices.tolist() == batch.target_dataset_indices.tolist()
    assert batch.source_cell_context_labels == batch.target_cell_context_labels
    assert batch.target_perturbation_labels[0] in {"KO_A", "KO_B"}
    assert batch.target_perturbation_labels[1] == "KO_C"


def test_perturbed_sources_default_to_self_with_control_label_override() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=4,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        seed=3,
    )

    batch = sampler.pair_source_indices([1, 6])

    np.testing.assert_array_equal(batch.target_indices, np.asarray([1, 6], dtype=np.int64))
    assert batch.target_perturbation_labels == ("WT", "WT")
    np.testing.assert_array_equal(
        batch.target_perturbation_ids,
        np.asarray([0, 0], dtype=np.int64),
    )


def test_matched_control_policy_samples_same_dataset_and_context_controls() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=4,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        perturbed_target_policy="matched_control_cell",
        seed=13,
    )

    batch = sampler.pair_source_indices([1, 2, 6], seed=19)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([0, 0, 5], dtype=np.int64))
    assert batch.target_perturbation_labels == ("WT", "WT", "WT")
    assert batch.source_dataset_indices.tolist() == batch.target_dataset_indices.tolist()
    assert batch.source_cell_context_labels == batch.target_cell_context_labels


def test_missing_target_pool_raises_explicit_error() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=4,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        perturbed_target_policy="matched_control_cell",
    )

    with pytest.raises(RuntimeError, match="no matched control pool exists"):
        sampler.pair_source_indices([7])


def test_warn_skip_policy_drops_unpairable_sources_with_warning() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=4,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        perturbed_target_policy="matched_control_cell",
        missing_target_policy="warn_skip",
    )

    with pytest.warns(RuntimeWarning, match="unable to pair source row 7"):
        batch = sampler.pair_source_indices([7, 6], seed=17)

    np.testing.assert_array_equal(batch.source_indices, np.asarray([6], dtype=np.int64))
    np.testing.assert_array_equal(batch.target_indices, np.asarray([5], dtype=np.int64))


def test_sampler_iteration_is_seed_deterministic_and_preserves_pairing_invariants() -> None:
    config = PertTFAdapterConfig(control_labels=("WT",))
    sampler_a = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=3,
        config=config,
        seed=29,
        drop_last=False,
        perturbed_target_policy="matched_control_cell",
        missing_target_policy="warn_skip",
    )
    sampler_b = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=3,
        config=config,
        seed=29,
        drop_last=False,
        perturbed_target_policy="matched_control_cell",
        missing_target_policy="warn_skip",
    )

    with pytest.warns(RuntimeWarning, match="unable to pair source row 7"):
        batches_a = list(sampler_a)
    with pytest.warns(RuntimeWarning, match="unable to pair source row 7"):
        batches_b = list(sampler_b)

    assert len(batches_a) == len(batches_b)
    for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
        np.testing.assert_array_equal(batch_a.source_indices, batch_b.source_indices)
        np.testing.assert_array_equal(batch_a.target_indices, batch_b.target_indices)
        np.testing.assert_array_equal(
            batch_a.source_dataset_indices,
            batch_a.target_dataset_indices,
        )
        assert batch_a.source_cell_context_labels == batch_a.target_cell_context_labels
