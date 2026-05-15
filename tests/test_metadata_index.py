"""Tests for the slim Polars-backed MetadataIndex."""

import numpy as np
import polars as pl
import pytest

from perturb_data_lab.loaders.index import MetadataIndex


def _meta() -> MetadataIndex:
    return MetadataIndex(
        pl.DataFrame(
            {
                "global_row_index": np.arange(5, dtype=np.int64),
                "cell_id": [f"cell_{i}" for i in range(5)],
                "dataset_id": ["ds0", "ds0", "ds1", "ds1", "ds1"],
                "dataset_index": np.asarray([0, 0, 1, 1, 1], dtype=np.int32),
                "local_row_index": np.asarray([0, 1, 0, 1, 2], dtype=np.int64),
                "size_factor": np.asarray([1.0, 1.1, 0.9, 1.2, 1.3], dtype=np.float64),
                "perturb_label": ["ctrl", "gene_a", "ctrl", "gene_b", None],
            }
        )
    )


def test_len_and_repr() -> None:
    meta = _meta()
    assert len(meta) == 5
    assert repr(meta) == "MetadataIndex(5 rows, 7 columns)"


def test_rejects_nested_columns() -> None:
    with pytest.raises(ValueError, match="non-flat dtype"):
        MetadataIndex(pl.DataFrame({"global_row_index": [0], "nested": [[1, 2]]}))


def test_get_column_returns_numeric_array() -> None:
    meta = _meta()
    values = meta.get_column("dataset_index")
    assert isinstance(values, np.ndarray)
    np.testing.assert_array_equal(values, np.asarray([0, 0, 1, 1, 1], dtype=np.int32))


def test_get_column_returns_string_tuple() -> None:
    assert _meta().get_column("dataset_id") == ("ds0", "ds0", "ds1", "ds1", "ds1")


def test_get_column_missing_returns_none() -> None:
    assert _meta().get_column("missing") is None


def test_take_preserves_order_and_types() -> None:
    meta = _meta()
    taken = meta.take([3, 1, 4], columns=("global_row_index", "dataset_id", "size_factor"))

    np.testing.assert_array_equal(taken["global_row_index"], np.asarray([3, 1, 4]))
    assert taken["dataset_id"] == ("ds1", "ds0", "ds1")
    np.testing.assert_allclose(taken["size_factor"], np.asarray([1.2, 1.1, 1.3]))


def test_gather_columns_defaults_to_all_columns() -> None:
    meta = _meta()
    gathered = meta.gather_columns([0, 2])
    assert tuple(gathered) == tuple(meta.df.columns)
    np.testing.assert_array_equal(gathered["global_row_index"], np.asarray([0, 2]))
    assert gathered["cell_id"] == ("cell_0", "cell_2")
