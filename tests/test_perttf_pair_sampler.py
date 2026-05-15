import numpy as np
import polars as pl
import pytest

from perturb_data_lab.loaders import (
    MetadataIndex,
    PertTFAdapterConfig,
    PerturbationPairSampler,
)
from perturb_data_lab.loaders.adapters.perttf import (
    _positions_for_global_rows,
    _prepare_perttf_metadata,
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


def _default_config(**kwargs) -> PertTFAdapterConfig:
    return PertTFAdapterConfig(control_labels=("WT",), **kwargs)


def _dataset_group_config(*group_labels: str, **kwargs) -> PertTFAdapterConfig:
    return PertTFAdapterConfig(
        control_labels=("WT",),
        label_fields={
            "perturb_label": "perturbation",
            "cell_context": "celltype",
            "batch_id": "batch",
            "dataset_id": "dataset",
        },
        pairing_group_labels=group_labels,
        **kwargs,
    )


def test_default_pairing_can_cross_dataset_and_context_when_unconstrained() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=1,
        config=_default_config(),
        source_indices=[0],
        target_candidate_indices=[7],
        seed=7,
        drop_last=False,
    )

    batch = sampler.pair_source_indices([0], seed=11)

    np.testing.assert_array_equal(batch.source_indices, np.asarray([0], dtype=np.int64))
    np.testing.assert_array_equal(batch.target_indices, np.asarray([7], dtype=np.int64))
    assert batch.source_dataset_indices.tolist() == [0]
    assert batch.target_dataset_indices.tolist() == [1]
    assert batch.source_labels_by_name["celltype"] == ("T_cell",)
    assert batch.target_labels_by_name["celltype"] == ("B_cell",)
    assert batch.target_labels_by_name["perturbation"] == ("KO_D",)


def test_explicit_dataset_pairing_uses_configured_dataset_label() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=1,
        config=_dataset_group_config("dataset"),
        source_indices=[0],
        target_candidate_indices=[1, 6],
        seed=7,
        drop_last=False,
    )

    batch = sampler.pair_source_indices([0], seed=11)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([1], dtype=np.int64))
    assert batch.source_labels_by_name["dataset"] == batch.target_labels_by_name["dataset"]


def test_explicit_dataset_and_celltype_pairing_uses_both_group_labels() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=1,
        config=_dataset_group_config("dataset", "celltype"),
        source_indices=[3],
        target_candidate_indices=[4, 7],
        seed=7,
        drop_last=False,
    )

    batch = sampler.pair_source_indices([3], seed=11)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([4], dtype=np.int64))
    assert batch.source_labels_by_name["dataset"] == batch.target_labels_by_name["dataset"]
    assert batch.source_labels_by_name["celltype"] == batch.target_labels_by_name["celltype"]


def test_perturbed_sources_default_to_self_with_control_label_override() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=4,
        config=_default_config(),
        seed=3,
    )

    batch = sampler.pair_source_indices([1, 6])

    np.testing.assert_array_equal(batch.target_indices, np.asarray([1, 6], dtype=np.int64))
    assert batch.target_labels_by_name["perturbation"] == ("WT", "WT")
    np.testing.assert_array_equal(
        batch.target_label_ids_by_name["perturbation"],
        np.asarray([0, 0], dtype=np.int64),
    )


def test_matched_control_policy_respects_explicit_pairing_groups() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=4,
        config=_dataset_group_config("dataset", "celltype"),
        perturbed_target_policy="matched_control_cell",
        seed=13,
    )

    batch = sampler.pair_source_indices([1, 2, 6], seed=19)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([0, 0, 5], dtype=np.int64))
    assert batch.target_labels_by_name["perturbation"] == ("WT", "WT", "WT")
    assert batch.source_labels_by_name["dataset"] == batch.target_labels_by_name["dataset"]
    assert batch.source_labels_by_name["celltype"] == batch.target_labels_by_name["celltype"]


def test_missing_target_pool_raises_explicit_error() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=4,
        config=_dataset_group_config("dataset", "celltype"),
        perturbed_target_policy="matched_control_cell",
    )

    with pytest.raises(RuntimeError, match="no matched control pool exists"):
        sampler.pair_source_indices([7])


def test_row_selection_intersects_all_requested_pools_and_preserves_order() -> None:
    selection = _prepare_perttf_metadata(
        _build_metadata_index(),
        config=_default_config(),
        row_indices=[3, 0, 1],
        source_indices=[1, 6, 0],
        target_candidate_indices=[0, 2, 3],
    ).row_selection

    np.testing.assert_array_equal(selection.base_indices, np.asarray([3, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(selection.source_indices, np.asarray([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(selection.target_candidate_indices, np.asarray([0, 3], dtype=np.int64))


def test_prepared_metadata_compacts_rows_and_keeps_polars_frame() -> None:
    prepared = _prepare_perttf_metadata(
        _build_metadata_index(),
        config=_default_config(),
        row_indices=np.asarray([4, 0, 1], dtype=np.int64),
        source_indices=None,
        target_candidate_indices=None,
    )

    np.testing.assert_array_equal(prepared.row_selection.base_indices, np.asarray([4, 0, 1], dtype=np.int64))
    assert prepared.global_to_local == {4: 0, 0: 1, 1: 2}
    assert prepared.frame.columns[:4] == ["global_row_index", "dataset_index", "perturbation", "perturbation_id"]
    np.testing.assert_array_equal(
        _positions_for_global_rows(prepared, np.asarray([1, 4], dtype=np.int64)),
        np.asarray([2, 0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        prepared.label_ids_by_name["celltype"][_positions_for_global_rows(prepared, np.asarray([1, 4], dtype=np.int64))],
        np.asarray([1, 0], dtype=np.int64),
    )
    assert tuple(prepared.frame["perturbation"].to_list()) == (
        "KO_C",
        "WT",
        "KO_A",
    )
    np.testing.assert_allclose(
        prepared.size_factor[_positions_for_global_rows(prepared, np.asarray([1, 4], dtype=np.int64))],
        np.asarray([1.1, 1.0], dtype=np.float32),
    )


def test_prepared_metadata_uses_filtered_base_pool_for_compaction() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=1,
        config=_default_config(),
        row_indices=[3, 0, 1],
        source_indices=[1],
        target_candidate_indices=[0],
        drop_last=False,
    )

    np.testing.assert_array_equal(
        sampler.effective_label_row_indices,
        np.asarray([3, 0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        sampler._prepared_metadata.row_selection.base_indices,
        np.asarray([3, 0, 1], dtype=np.int64),
    )


def test_source_only_subset_uses_filtered_base_pool_for_targets() -> None:
    sampler = PerturbationPairSampler(
        _build_two_row_metadata_index(),
        batch_size=1,
        config=_default_config(),
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
        config=_default_config(),
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


def test_default_non_perturbation_null_labels_are_encoded() -> None:
    sampler = PerturbationPairSampler(
        _build_null_context_metadata_index(),
        batch_size=1,
        config=_default_config(),
        drop_last=False,
    )

    batch = sampler.pair_source_indices([0], seed=19)

    np.testing.assert_array_equal(sampler.effective_source_indices, np.asarray([0, 1], dtype=np.int64))
    assert batch.source_labels_by_name["celltype"] == ("<null>",)
    assert batch.target_labels_by_name["celltype"] == ("<null>",)


def test_sampler_supports_perturbation_as_only_required_semantic_label() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=1,
        config=_default_config(label_fields={"perturb_label": "perturbation"}),
        source_indices=[0],
        target_candidate_indices=[6],
        seed=23,
        drop_last=False,
    )

    batch = sampler.pair_source_indices([0], seed=29)

    assert tuple(batch.source_label_ids_by_name) == ("perturbation",)
    assert tuple(batch.target_label_ids_by_name) == ("perturbation",)
    np.testing.assert_array_equal(batch.target_indices, np.asarray([6], dtype=np.int64))
    assert batch.target_labels_by_name["perturbation"] == ("KO_A",)


def test_custom_drop_null_labels_removes_selected_base_pool() -> None:
    with pytest.raises(ValueError, match="metadata row pool is empty"):
        PerturbationPairSampler(
            _build_null_context_metadata_index(),
            batch_size=1,
            config=_default_config(drop_null_labels=("celltype",)),
            drop_last=False,
        )


def test_pair_source_indices_rejects_rows_outside_configured_source_pool() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=2,
        config=_default_config(),
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
        config=_dataset_group_config("dataset", "celltype"),
        source_indices=[0, 3, 6],
        target_candidate_indices=[2, 4, 5],
        perturbed_target_policy="matched_control_cell",
        seed=13,
    )

    batch = sampler.pair_source_indices([0, 3, 6], seed=19)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([2, 4, 5], dtype=np.int64))
    assert set(batch.target_indices.tolist()).issubset({2, 4, 5})
    assert batch.source_labels_by_name["dataset"] == batch.target_labels_by_name["dataset"]
    assert batch.source_labels_by_name["celltype"] == batch.target_labels_by_name["celltype"]


def test_restricted_target_pool_can_make_source_unpairable() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=1,
        config=_default_config(),
        source_indices=[6],
        target_candidate_indices=[5],
    )

    with pytest.raises(
        RuntimeError,
        match="configured target pool",
    ):
        sampler.pair_source_indices([6])


def test_sampler_iteration_is_seed_deterministic_and_preserves_pairing_invariants() -> None:
    config = _dataset_group_config("dataset", "celltype")
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
        assert batch_a.source_labels_by_name["dataset"] == batch_a.target_labels_by_name["dataset"]
        assert batch_a.source_labels_by_name["celltype"] == batch_a.target_labels_by_name["celltype"]


def test_sampler_set_epoch_is_repeatable_and_changes_pair_sequence() -> None:
    sampler = PerturbationPairSampler(
        _build_metadata_index(),
        batch_size=1,
        config=_default_config(),
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
