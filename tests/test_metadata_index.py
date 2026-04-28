"""Tests for Phase 1 MetadataIndex — polars-backed queryable metadata index."""

import time

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
