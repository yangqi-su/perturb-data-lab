import numpy as np
import polars as pl
import pytest

from perturb_data_lab.loaders import (
    MetadataIndex,
    PertTFAdapterConfig,
    PerturbationPairSampler,
)
from perturb_data_lab.loaders.adapters.perttf import _resolve_perttf_row_selection


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


def _build_two_row_metadata_index() -> MetadataIndex:
    return MetadataIndex(
        pl.DataFrame(
            {
                "global_row_index": np.arange(2, dtype=np.int64),
                "cell_id": ["cell_0", "cell_1"],
                "dataset_id": ["ds0", "ds0"],
                "dataset_index": np.asarray([0, 0], dtype=np.int32),
                "local_row_index": np.arange(2, dtype=np.int64),
                "size_factor": np.asarray([1.0, 1.1], dtype=np.float32),
                "cell_context": ["T_cell", "T_cell"],
                "perturb_label": ["WT", "KO_A"],
                "batch_id": ["b0", "b1"],
            }
        )
    )


def _build_null_context_metadata_index() -> MetadataIndex:
    return MetadataIndex(
        pl.DataFrame(
            {
                "global_row_index": np.arange(2, dtype=np.int64),
                "cell_id": ["cell_0", "cell_1"],
                "dataset_id": ["ds0", "ds0"],
                "dataset_index": np.asarray([0, 0], dtype=np.int32),
                "local_row_index": np.arange(2, dtype=np.int64),
                "size_factor": np.asarray([1.0, 1.1], dtype=np.float32),
                "cell_context": [None, None],
                "perturb_label": ["WT", "KO_A"],
                "batch_id": ["b0", "b1"],
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


def test_row_selection_intersects_all_requested_pools_and_preserves_order() -> None:
    selection = _resolve_perttf_row_selection(
        _build_metadata_index(),
        config=PertTFAdapterConfig(control_labels=("WT",)),
        row_indices=[3, 0, 1],
        source_indices=[1, 6, 0],
        target_candidate_indices=[0, 2, 3],
    )

    np.testing.assert_array_equal(selection.base_indices, np.asarray([3, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(selection.source_indices, np.asarray([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(selection.target_candidate_indices, np.asarray([0, 3], dtype=np.int64))


def test_source_only_subset_uses_filtered_base_pool_for_targets() -> None:
    sampler = PerturbationPairSampler(
        _build_two_row_metadata_index(),
        batch_size=1,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        source_indices=[0],
        seed=11,
        drop_last=False,
    )

    batch = sampler.pair_source_indices([0], seed=13)

    np.testing.assert_array_equal(sampler.effective_source_indices, np.asarray([0], dtype=np.int64))
    np.testing.assert_array_equal(
        sampler.effective_target_candidate_indices,
        np.asarray([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(batch.target_indices, np.asarray([1], dtype=np.int64))


def test_target_candidate_only_subset_is_allowed() -> None:
    sampler = PerturbationPairSampler(
        _build_two_row_metadata_index(),
        batch_size=2,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        target_candidate_indices=[1],
        drop_last=False,
    )

    batch = sampler.pair_source_indices([0, 1], seed=17)

    np.testing.assert_array_equal(sampler.effective_source_indices, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        sampler.effective_target_candidate_indices,
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(batch.target_indices, np.asarray([1, 1], dtype=np.int64))


def test_encode_null_labels_keeps_rows_and_uses_null_token() -> None:
    sampler = PerturbationPairSampler(
        _build_null_context_metadata_index(),
        batch_size=1,
        config=PertTFAdapterConfig(
            control_labels=("WT",),
            encode_null_labels=("celltype",),
        ),
        drop_last=False,
    )

    batch = sampler.pair_source_indices([0], seed=19)

    np.testing.assert_array_equal(sampler.effective_source_indices, np.asarray([0, 1], dtype=np.int64))
    assert batch.source_cell_context_labels == ("<null>",)
    assert batch.target_cell_context_labels == ("<null>",)


def test_error_null_labels_fail_on_selected_base_pool() -> None:
    with pytest.raises(ValueError, match="configured error-null field"):
        PerturbationPairSampler(
            _build_null_context_metadata_index(),
            batch_size=1,
            config=PertTFAdapterConfig(
                control_labels=("WT",),
                error_null_labels=("celltype",),
            ),
        )


def test_pair_source_indices_rejects_rows_outside_configured_source_pool() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=2,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        source_indices=[0, 1, 5, 6],
    )

    with pytest.raises(
        ValueError,
        match="configured source_indices pool",
    ):
        sampler.pair_source_indices([2])


def test_asymmetric_target_candidate_pool_preserves_pairing_invariants() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=3,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        source_indices=[0, 3, 6],
        target_candidate_indices=[2, 4, 5],
        perturbed_target_policy="matched_control_cell",
        seed=13,
    )

    batch = sampler.pair_source_indices([0, 3, 6], seed=19)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([2, 4, 5], dtype=np.int64))
    assert set(batch.target_indices.tolist()).issubset({2, 4, 5})
    assert batch.source_dataset_indices.tolist() == batch.target_dataset_indices.tolist()
    assert batch.source_cell_context_labels == batch.target_cell_context_labels


def test_restricted_target_pool_can_make_source_unpairable() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=1,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        source_indices=[6],
        target_candidate_indices=[5],
    )

    with pytest.raises(
        RuntimeError,
        match="configured target pool",
    ):
        sampler.pair_source_indices([6])


def test_sampler_iteration_is_seed_deterministic_and_preserves_pairing_invariants() -> None:
    config = PertTFAdapterConfig(control_labels=("WT",))
    sampler_a = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=3,
        config=config,
        seed=29,
        drop_last=False,
        perturbed_target_policy="matched_control_cell",
        source_indices=[0, 1, 2, 3, 4, 5, 6],
    )
    sampler_b = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=3,
        config=config,
        seed=29,
        drop_last=False,
        perturbed_target_policy="matched_control_cell",
        source_indices=[0, 1, 2, 3, 4, 5, 6],
    )

    batches_a = list(sampler_a)
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


def test_sampler_set_epoch_is_repeatable_and_changes_pair_sequence() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=1,
        config=PertTFAdapterConfig(control_labels=("WT",)),
        source_indices=[0, 1, 5, 6],
        seed=17,
        drop_last=True,
    )

    sampler.set_epoch(0)
    epoch0_first = [
        (tuple(batch.source_indices.tolist()), tuple(batch.target_indices.tolist()))
        for batch in sampler
    ]
    sampler.set_epoch(0)
    epoch0_second = [
        (tuple(batch.source_indices.tolist()), tuple(batch.target_indices.tolist()))
        for batch in sampler
    ]
    sampler.set_epoch(1)
    epoch1 = [
        (tuple(batch.source_indices.tolist()), tuple(batch.target_indices.tolist()))
        for batch in sampler
    ]

    assert epoch0_first == epoch0_second
    assert epoch1 != epoch0_first
