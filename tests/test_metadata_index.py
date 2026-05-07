"""Tests for Phase 1 MetadataIndex — polars-backed queryable metadata index."""

import time

import numpy as np
import polars as pl

from perturb_data_lab.loaders.index import MetadataIndex, MetadataRow


# ===================================================================
# Fixtures
# ===================================================================

def _meta() -> MetadataIndex:
    """Module-level singleton to avoid reloading in every test."""
    if not hasattr(_meta, "_cache"):
        _meta._cache = MetadataIndex.from_dummy_data()
    return _meta._cache


# ===================================================================
# Tests
# ===================================================================

def test_loads_dummy_data():
    """MetadataIndex loads dummy_00 + dummy_01 and reports 125K rows."""
    meta = _meta()
    assert len(meta) == 125_000, f"Expected 125K rows, got {len(meta)}"


def test_filter_by_dataset():
    """Filtering by dataset_id returns correct row counts."""
    meta = _meta()
    dummy_00 = meta.filter(pl.col("dataset_id") == "dummy_00")
    assert len(dummy_00) == 50_000, f"Expected 50K dummy_00, got {len(dummy_00)}"

    dummy_01 = meta.filter(pl.col("dataset_id") == "dummy_01")
    assert len(dummy_01) == 75_000, f"Expected 75K dummy_01, got {len(dummy_01)}"


def test_sample():
    """Random sample returns the correct number of unique indices."""
    meta = _meta()
    sampled = meta.sample(128, seed=42)
    assert len(sampled) == 128, f"Expected 128 sampled rows, got {len(sampled)}"
    assert sampled["global_row_index"].n_unique() == 128


def test_sample_by():
    """Stratified sampling returns rows from both datasets."""
    meta = _meta()
    sampled = meta.sample_by("dataset_id", n_per_group=10, seed=42)
    assert len(sampled) >= 20, f"Expected >=20 rows, got {len(sampled)}"
    counts = sampled["dataset_id"].value_counts()
    # Both datasets should be represented
    assert len(counts) == 2, f"Expected both datasets, got {counts}"


def test_get_indices():
    """Filtered indices are contiguous within each dataset."""
    meta = _meta()
    dummy_00 = meta.filter(pl.col("dataset_id") == "dummy_00")
    indices = meta.get_indices(dummy_00.df)
    assert len(indices) == 50_000
    assert min(indices) == 0
    assert max(indices) == 49_999


def test_getitem():
    """__getitem__ with a list of indices preserves order."""
    meta = _meta()
    indices = [100, 5, 99999]
    subset = meta[indices]
    assert len(subset) == 3
    assert subset.df["global_row_index"].to_list() == indices


def test_flat_schema():
    """No column uses Struct, Object, or List dtype."""
    meta = _meta()
    for col_name in meta.df.columns:
        dtype = meta.df[col_name].dtype
        assert dtype not in (
            pl.Struct, pl.Object, pl.List
        ), f"Column '{col_name}' has non-flat dtype {dtype}"


def test_metadata_row():
    """MetadataRow objects carry correct flat fields."""
    meta = _meta()
    rows = meta.rows([0, 1, 50000])
    assert len(rows) == 3
    assert rows[0].global_row_index == 0
    assert rows[0].dataset_id == "dummy_00"
    assert rows[1].global_row_index == 1
    assert rows[2].global_row_index == 50000
    assert rows[2].dataset_id == "dummy_01"
    assert rows[0].raw_fields["guide_1"] is not None


def test_metadata_row_excludes_raw_dataset_id():
    """raw_fields dict should not include duplicate primary keys."""
    meta = _meta()
    row = meta.rows([0])[0]
    # dataset_id is a primary column, not in raw_fields
    assert "dataset_id" not in row.raw_fields, (
        "raw_fields should not contain dataset_id"
    )


# ===================================================================
# Performance tests
# ===================================================================

def test_load_time():
    """Full load completes in <2 seconds."""
    t0 = time.perf_counter()
    _ = MetadataIndex.from_dummy_data()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    assert elapsed < 2.0, f"Load took {elapsed:.2f}s, expected <2s"


def test_filter_latency():
    """Filter on dataset_id takes <10ms."""
    meta = _meta()
    t0 = time.perf_counter()
    _ = meta.filter(pl.col("dataset_id") == "dummy_00")
    t1 = time.perf_counter()
    elapsed = t1 - t0
    assert elapsed < 0.01, f"Filter took {elapsed:.3f}s, expected <10ms"


def test_sample_latency():
    """Sample 128 rows takes <1ms."""
    meta = _meta()
    t0 = time.perf_counter()
    _ = meta.sample(128, seed=42)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    assert elapsed < 0.001, f"Sample took {elapsed:.3f}s, expected <1ms"


def test_getitem_latency():
    """Positional lookup via __getitem__ takes <1ms for 128 indices."""
    meta = _meta()
    indices = list(range(0, 125000, 1000))  # 125 indices
    t0 = time.perf_counter()
    _ = meta[indices]
    t1 = time.perf_counter()
    elapsed = t1 - t0
    assert elapsed < 0.001, f"__getitem__ took {elapsed:.3f}s, expected <1ms"


# ===================================================================
# Phase 4 — Hot-column caching and columnar gathering
# ===================================================================


def test_cache_hot_columns_all():
    """Caching all columns stores correct shapes and types."""
    meta = _meta()
    meta.cache_hot_columns()  # cache all
    assert len(meta._cache) > 10, f"Expected many cached columns, got {len(meta._cache)}"

    # Numeric identity columns should be numpy arrays
    for col in ["global_row_index", "dataset_index", "size_factor"]:
        data = meta._cache.get(col)
        assert isinstance(data, np.ndarray), f"{col} should be ndarray, got {type(data)}"
        assert len(data) == len(meta), f"{col} len {len(data)} != {len(meta)}"

    # String columns should be tuples
    for col in ["dataset_id", "cell_id"]:
        data = meta._cache.get(col)
        assert isinstance(data, tuple), f"{col} should be tuple, got {type(data)}"
        assert len(data) == len(meta), f"{col} len {len(data)} != {len(meta)}"


def test_cache_hot_columns_subset():
    """Caching only numeric columns works."""
    meta = MetadataIndex.from_dummy_data()  # fresh instance to avoid cache pollution
    meta.cache_hot_columns(["global_row_index", "dataset_index", "size_factor"])
    assert "global_row_index" in meta._cache
    assert "dataset_index" in meta._cache
    assert "size_factor" in meta._cache
    # String columns not cached since they weren't requested
    assert "cell_id" not in meta._cache
    # Verify cached data is correct
    assert np.issubdtype(meta._cache["global_row_index"].dtype, np.integer)
    assert np.issubdtype(meta._cache["dataset_index"].dtype, np.integer)
    assert np.issubdtype(meta._cache["size_factor"].dtype, np.floating)


def test_gather_columns_numeric():
    """gather_columns returns correct numeric arrays for specific indices."""
    meta = _meta()
    meta.cache_hot_columns(["global_row_index", "dataset_index", "size_factor"])

    indices = [0, 1, 50000, 99999]
    gathered = meta.gather_columns(indices, ["global_row_index", "dataset_index"])

    np.testing.assert_array_equal(
        gathered["global_row_index"],
        np.array(indices, dtype=np.int64),
    )
    # dataset_index: 0 for dummy_00 (indices 0-49999), 1 for dummy_01
    assert gathered["dataset_index"][0] == 0
    assert gathered["dataset_index"][1] == 0
    assert gathered["dataset_index"][2] == 1
    assert gathered["dataset_index"][3] == 1


def test_gather_columns_string():
    """gather_columns returns correct string tuples."""
    meta = _meta()
    meta.cache_hot_columns(["dataset_id", "cell_id"])

    indices = [0, 1, 50000, 50001]
    gathered = meta.gather_columns(indices, ["dataset_id"])

    ds_ids = gathered["dataset_id"]
    assert isinstance(ds_ids, tuple)
    assert len(ds_ids) == 4
    assert ds_ids[0] == "dummy_00"
    assert ds_ids[1] == "dummy_00"
    assert ds_ids[2] == "dummy_01"
    assert ds_ids[3] == "dummy_01"


def test_gather_columns_empty():
    """gather_columns with empty indices returns empty results."""
    meta = _meta()
    meta.cache_hot_columns(["global_row_index", "dataset_id"])
    gathered = meta.gather_columns([], ["global_row_index", "dataset_id"])
    assert len(gathered["global_row_index"]) == 0
    assert len(gathered["dataset_id"]) == 0


def test_gather_columns_order():
    """gather_columns preserves non-monotonic index order."""
    meta = _meta()
    meta.cache_hot_columns(["global_row_index", "dataset_id"])

    indices = [50001, 1, 99999, 0]
    gathered = meta.gather_columns(indices, ["global_row_index", "dataset_id"])

    # global_row_index should match input order
    np.testing.assert_array_equal(
        gathered["global_row_index"],
        np.array(indices, dtype=np.int64),
    )
    # dataset_id should match input order
    assert gathered["dataset_id"][0] == "dummy_01"
    assert gathered["dataset_id"][1] == "dummy_00"
    assert gathered["dataset_id"][2] == "dummy_01"
    assert gathered["dataset_id"][3] == "dummy_00"


def test_gather_columns_uncached_fallback():
    """gather_columns falls back to Polars for uncached columns."""
    meta = MetadataIndex.from_dummy_data()
    # Don't cache anything
    indices = [0, 1, 2]
    gathered = meta.gather_columns(indices, ["global_row_index", "cell_id"])

    np.testing.assert_array_equal(
        gathered["global_row_index"],
        np.array(indices, dtype=np.int64),
    )
    assert isinstance(gathered["cell_id"], tuple)
    assert len(gathered["cell_id"]) == 3
    assert "global_row_index" in meta._cache
    assert "cell_id" not in meta._cache


def test_gather_columns_all_columns():
    """gather_columns with columns=None returns all cached columns."""
    meta = _meta()
    meta.cache_hot_columns()  # cache all
    indices = [0, 5, 10]
    gathered = meta.gather_columns(indices)

    # Should include all cached columns
    assert len(gathered) == len(meta._cache)
    for col, data in gathered.items():
        if isinstance(data, np.ndarray):
            assert len(data) == 3, f"{col} should have 3 entries"
        else:
            assert len(data) == 3, f"{col} should have 3 entries"


def test_gather_columns_large_batch():
    """gather_columns handles a 128-cell batch correctly."""
    meta = _meta()
    meta.cache_hot_columns(["global_row_index", "size_factor", "cell_id"])
    rng = np.random.default_rng(42)
    indices = sorted(rng.choice(len(meta), size=128, replace=False).tolist())
    gathered = meta.gather_columns(indices)

    assert len(gathered["global_row_index"]) == 128
    assert len(gathered["size_factor"]) == 128
    assert len(gathered["cell_id"]) == 128
    np.testing.assert_array_equal(
        gathered["global_row_index"],
        np.array(indices, dtype=np.int64),
    )


def test_take_alias_matches_gather_columns():
    """take() reuses gather_columns semantics and preserves order."""
    meta = _meta()
    meta.cache_hot_columns(["global_row_index", "dataset_id"])

    indices = [50001, 1, 99999, 0]
    gathered = meta.gather_columns(indices, ["global_row_index", "dataset_id"])
    taken = meta.take(indices, ["global_row_index", "dataset_id"])

    np.testing.assert_array_equal(
        taken["global_row_index"],
        gathered["global_row_index"],
    )
    assert taken["dataset_id"] == gathered["dataset_id"]


def test_get_column():
    """get_column returns cached or on-demand column data."""
    meta = _meta()
    # Access before caching — triggers lazy caching
    gr = meta.get_column("global_row_index")
    assert isinstance(gr, np.ndarray)
    assert len(gr) == len(meta)
    assert np.issubdtype(gr.dtype, np.integer)

    ds_id = meta.get_column("dataset_id")
    assert isinstance(ds_id, tuple)
    assert len(ds_id) == len(meta)

    # Nonexistent column
    assert meta.get_column("nonexistent_column") is None


def test_cache_hot_columns_latency():
    """Caching all columns completes in <200ms for 125K rows."""
    meta = _meta()
    t0 = time.perf_counter()
    meta.cache_hot_columns()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    assert elapsed < 0.2, f"Cache took {elapsed:.3f}s, expected <200ms"


def test_gather_columns_latency():
    """gather_columns for 128 indices with cached columns is <0.5ms."""
    meta = _meta()
    meta.cache_hot_columns()
    indices = list(range(0, 128))
    t0 = time.perf_counter()
    _ = meta.gather_columns(indices)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    assert elapsed < 0.001, f"Gather took {elapsed:.4f}s, expected <1ms"
