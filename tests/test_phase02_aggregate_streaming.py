"""Phase 2: Validate streaming aggregate writers for lance, zarr, webdataset.

Each aggregate writer is tested with 2 dummy datasets (dummy_00, dummy_01)
materialized sequentially in 2 chunks each, using the streaming
``_writer_state`` / ``_is_last_chunk`` contract.

Verification:
- Total rows = sum of both datasets' n_obs
- global_row_index values are contiguous across dataset boundary
- _writer_state is None after _is_last_chunk=True
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

DUMMY_00_PATH = Path(
    "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/dummy_data/dummy_00_counts.h5ad"
)
DUMMY_01_PATH = Path(
    "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/dummy_data/dummy_01_counts.h5ad"
)


def _load_h5ad_backed(file_path: Path) -> Any:
    import anndata as ad
    return ad.read_h5ad(str(file_path), backed="r")


def _materialize_csr(adata: Any) -> Any:
    """Materialize a backed AnnData's X as an in-memory CSR matrix."""
    return adata.X[:].tocsr()


def _make_dataset_spec(dataset_id: str, dataset_index: int, adata: Any,
                       global_row_start: int) -> Any:
    from perturb_data_lab.materializers.chunk_translation import DatasetSpec
    nnz = int(adata.X.nnz) if hasattr(adata.X, "nnz") else 0
    rows = adata.n_obs
    return DatasetSpec(
        dataset_id=dataset_id,
        dataset_index=dataset_index,
        file_path=Path(adata.filename),
        rows=rows,
        pairs=0,
        local_vocabulary_size=adata.n_vars,
        nnz_total=nnz,
        global_row_start=global_row_start,
        global_row_stop=global_row_start + rows,
    )


def _chunk_row_indices(n_obs: int, n_chunks: int):
    """Yield (chunk_start, chunk_end) for n_chunks equal-ish chunks."""
    chunk_size = (n_obs + n_chunks - 1) // n_chunks
    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        yield start, end


# ---------------------------------------------------------------------------
# Lance streaming aggregate
# ---------------------------------------------------------------------------


class TestLanceAggregateStreaming:
    """Test write_lance_aggregate with 2 datasets × 2 chunks each."""

    @pytest.fixture(scope="class")
    def lance_env(self, tmp_path_factory: Any) -> dict[str, Any]:
        pytest.importorskip("lance")

        tmp_root = tmp_path_factory.mktemp("lance_agg_test")
        matrix_root = tmp_root / "matrix"
        matrix_root.mkdir(parents=True, exist_ok=True)

        # Load data
        adata0 = _load_h5ad_backed(DUMMY_00_PATH)
        adata1 = _load_h5ad_backed(DUMMY_01_PATH)
        csr0 = _materialize_csr(adata0)
        csr1 = _materialize_csr(adata1)

        specs = [
            _make_dataset_spec("dummy_00", 0, adata0, 0),
            _make_dataset_spec("dummy_01", 1, adata1, adata0.n_obs),
        ]

        # Close backed annData since we now have in-memory CSR
        adata0.file.close()
        adata1.file.close()

        return {
            "matrix_root": matrix_root,
            "csr0": csr0,
            "csr1": csr1,
            "specs": specs,
            "n_obs_0": csr0.shape[0],
            "n_obs_1": csr1.shape[0],
        }

    def test_lance_aggregate_2_datasets(self, lance_env: dict[str, Any]):
        from perturb_data_lab.materializers.backends.lance import (
            write_lance_aggregate,
        )
        from perturb_data_lab.materializers.chunk_translation import (
            _translate_chunk,
        )

        matrix_root = lance_env["matrix_root"]
        csr0 = lance_env["csr0"]
        csr1 = lance_env["csr1"]
        specs = lance_env["specs"]
        n_obs_0 = lance_env["n_obs_0"]
        n_obs_1 = lance_env["n_obs_1"]
        expected_total = n_obs_0 + n_obs_1

        writer_state: dict | None = None
        paths = None

        # Process dataset 0 in 2 chunks
        dataset0_spec = specs[0]
        for chunk_start, chunk_end in _chunk_row_indices(n_obs_0, 2):
            is_last = (chunk_end == n_obs_0) and False  # more datasets follow
            matrix_chunk = csr0[chunk_start:chunk_end].tocsr()
            bundle = _translate_chunk(
                dataset=dataset0_spec,
                matrix_chunk=matrix_chunk,
                chunk_start=chunk_start,
            )
            paths, writer_state = write_lance_aggregate(
                bundle=bundle,
                release_id="test_agg",
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=False,
            )
            assert writer_state is not None, (
                f"writer_state should not be None after non-last chunk "
                f"(dataset 0, chunk_start={chunk_start})"
            )

        # Process dataset 1 in 2 chunks, last chunk is truly last
        dataset1_spec = specs[1]
        chunk_iter = list(_chunk_row_indices(n_obs_1, 2))
        for i, (chunk_start, chunk_end) in enumerate(chunk_iter):
            is_last = (i == len(chunk_iter) - 1)
            matrix_chunk = csr1[chunk_start:chunk_end].tocsr()
            bundle = _translate_chunk(
                dataset=dataset1_spec,
                matrix_chunk=matrix_chunk,
                chunk_start=chunk_start,
            )
            paths, writer_state = write_lance_aggregate(
                bundle=bundle,
                release_id="test_agg",
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=is_last,
            )

        # After last chunk, writer_state should be None
        assert writer_state is None, (
            f"writer_state should be None after _is_last_chunk=True, "
            f"got {writer_state}"
        )

        # Verify paths
        assert paths is not None
        assert "cells" in paths
        lance_path = paths["cells"]
        assert lance_path.exists()
        assert lance_path.name == "aggregated-cells.lance"

        # Verify total row count
        import lance
        ds = lance.dataset(str(lance_path))
        total_rows = ds.count_rows()
        assert total_rows == expected_total, (
            f"aggregate row count {total_rows} != expected {expected_total}"
        )

        # Verify global_row_index is contiguous across dataset boundary
        table = ds.to_table()
        global_row_index_col = table.column("global_row_index").to_numpy()
        assert global_row_index_col[0] == 0, (
            f"first global_row_index should be 0, got {global_row_index_col[0]}"
        )
        assert global_row_index_col[-1] == expected_total - 1, (
            f"last global_row_index should be {expected_total - 1}, "
            f"got {global_row_index_col[-1]}"
        )

        # Verify no gaps in global_row_index
        expected_range = np.arange(expected_total, dtype=np.int64)
        np.testing.assert_array_equal(
            global_row_index_col, expected_range,
            err_msg="global_row_index is not contiguous"
        )


# ---------------------------------------------------------------------------
# Zarr streaming aggregate
# ---------------------------------------------------------------------------


class TestZarrAggregateStreaming:
    """Test write_zarr_aggregate with 2 datasets × 2 chunks each."""

    @pytest.fixture(scope="class")
    def zarr_env(self, tmp_path_factory: Any) -> dict[str, Any]:
        import zarr

        tmp_root = tmp_path_factory.mktemp("zarr_agg_test")
        matrix_root = tmp_root / "matrix"
        matrix_root.mkdir(parents=True, exist_ok=True)

        adata0 = _load_h5ad_backed(DUMMY_00_PATH)
        adata1 = _load_h5ad_backed(DUMMY_01_PATH)
        csr0 = _materialize_csr(adata0)
        csr1 = _materialize_csr(adata1)

        specs = [
            _make_dataset_spec("dummy_00", 0, adata0, 0),
            _make_dataset_spec("dummy_01", 1, adata1, adata0.n_obs),
        ]

        adata0.file.close()
        adata1.file.close()

        return {
            "matrix_root": matrix_root,
            "csr0": csr0,
            "csr1": csr1,
            "specs": specs,
            "n_obs_0": csr0.shape[0],
            "n_obs_1": csr1.shape[0],
        }

    def test_zarr_aggregate_2_datasets(self, zarr_env: dict[str, Any]):
        from perturb_data_lab.materializers.backends.zarr import (
            write_zarr_aggregate,
        )
        from perturb_data_lab.materializers.chunk_translation import (
            _translate_chunk,
        )

        matrix_root = zarr_env["matrix_root"]
        csr0 = zarr_env["csr0"]
        csr1 = zarr_env["csr1"]
        specs = zarr_env["specs"]
        n_obs_0 = zarr_env["n_obs_0"]
        n_obs_1 = zarr_env["n_obs_1"]
        expected_total = n_obs_0 + n_obs_1

        writer_state: dict | None = None
        paths = None

        # Process dataset 0 in 2 chunks
        dataset0_spec = specs[0]
        for chunk_start, chunk_end in _chunk_row_indices(n_obs_0, 2):
            matrix_chunk = csr0[chunk_start:chunk_end].tocsr()
            bundle = _translate_chunk(
                dataset=dataset0_spec,
                matrix_chunk=matrix_chunk,
                chunk_start=chunk_start,
            )
            paths, writer_state = write_zarr_aggregate(
                bundle=bundle,
                release_id="test_agg",
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=False,
            )
            assert writer_state is not None, (
                f"writer_state should not be None after non-last chunk "
                f"(dataset 0, chunk_start={chunk_start})"
            )

        # Process dataset 1 in 2 chunks
        dataset1_spec = specs[1]
        chunk_iter = list(_chunk_row_indices(n_obs_1, 2))
        for i, (chunk_start, chunk_end) in enumerate(chunk_iter):
            is_last = (i == len(chunk_iter) - 1)
            matrix_chunk = csr1[chunk_start:chunk_end].tocsr()
            bundle = _translate_chunk(
                dataset=dataset1_spec,
                matrix_chunk=matrix_chunk,
                chunk_start=chunk_start,
            )
            paths, writer_state = write_zarr_aggregate(
                bundle=bundle,
                release_id="test_agg",
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=is_last,
            )

        # After last chunk, writer_state should be None
        assert writer_state is None, (
            f"writer_state should be None after _is_last_chunk=True, "
            f"got {writer_state}"
        )

        # Verify all paths exist
        assert paths is not None
        assert paths["indices"].exists()
        assert paths["counts"].exists()
        assert paths["row_offsets"].exists()
        assert paths["meta"].exists()

        # Verify total rows via row_offsets length
        import zarr
        row_offsets_arr = zarr.open(str(paths["row_offsets"]), mode="r")["row_offsets"][:]
        assert len(row_offsets_arr) == expected_total + 1, (
            f"row_offsets length {len(row_offsets_arr)} != expected {expected_total + 1}"
        )

        # Verify last row_offset equals total_nnz
        indices_arr = zarr.open(str(paths["indices"]), mode="r")["indices"][:]
        assert int(row_offsets_arr[-1]) == len(indices_arr), (
            f"last row_offset {row_offsets_arr[-1]} != total nnz {len(indices_arr)}"
        )

        # Verify meta.json content
        import json
        with open(paths["meta"]) as f:
            meta = json.load(f)
        assert meta["total_rows"] == expected_total, (
            f"meta total_rows {meta['total_rows']} != expected {expected_total}"
        )
        assert meta["total_nnz"] == len(indices_arr)


# ---------------------------------------------------------------------------
# WebDataset streaming aggregate
# ---------------------------------------------------------------------------


class TestWebDatasetAggregateStreaming:
    """Test write_webdataset_aggregate with 2 datasets × 2 chunks each."""

    @pytest.fixture(scope="class")
    def wds_env(self, tmp_path_factory: Any) -> dict[str, Any]:
        tmp_root = tmp_path_factory.mktemp("wds_agg_test")
        matrix_root = tmp_root / "matrix"
        matrix_root.mkdir(parents=True, exist_ok=True)

        adata0 = _load_h5ad_backed(DUMMY_00_PATH)
        adata1 = _load_h5ad_backed(DUMMY_01_PATH)
        csr0 = _materialize_csr(adata0)
        csr1 = _materialize_csr(adata1)

        specs = [
            _make_dataset_spec("dummy_00", 0, adata0, 0),
            _make_dataset_spec("dummy_01", 1, adata1, adata0.n_obs),
        ]

        adata0.file.close()
        adata1.file.close()

        return {
            "matrix_root": matrix_root,
            "csr0": csr0,
            "csr1": csr1,
            "specs": specs,
            "n_obs_0": csr0.shape[0],
            "n_obs_1": csr1.shape[0],
        }

    def test_webdataset_aggregate_2_datasets(self, wds_env: dict[str, Any]):
        from perturb_data_lab.materializers.backends.webdataset import (
            write_webdataset_aggregate,
        )
        from perturb_data_lab.materializers.chunk_translation import (
            _translate_chunk,
        )

        matrix_root = wds_env["matrix_root"]
        csr0 = wds_env["csr0"]
        csr1 = wds_env["csr1"]
        specs = wds_env["specs"]
        n_obs_0 = wds_env["n_obs_0"]
        n_obs_1 = wds_env["n_obs_1"]
        expected_total = n_obs_0 + n_obs_1

        writer_state: dict | None = None
        paths = None

        # Process dataset 0 in 2 chunks
        dataset0_spec = specs[0]
        for chunk_start, chunk_end in _chunk_row_indices(n_obs_0, 2):
            matrix_chunk = csr0[chunk_start:chunk_end].tocsr()
            bundle = _translate_chunk(
                dataset=dataset0_spec,
                matrix_chunk=matrix_chunk,
                chunk_start=chunk_start,
            )
            paths, writer_state = write_webdataset_aggregate(
                bundle=bundle,
                release_id="test_agg",
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=False,
            )
            assert writer_state is not None, (
                f"writer_state should not be None after non-last chunk "
                f"(dataset 0, chunk_start={chunk_start})"
            )

        # Process dataset 1 in 2 chunks
        dataset1_spec = specs[1]
        chunk_iter = list(_chunk_row_indices(n_obs_1, 2))
        for i, (chunk_start, chunk_end) in enumerate(chunk_iter):
            is_last = (i == len(chunk_iter) - 1)
            matrix_chunk = csr1[chunk_start:chunk_end].tocsr()
            bundle = _translate_chunk(
                dataset=dataset1_spec,
                matrix_chunk=matrix_chunk,
                chunk_start=chunk_start,
            )
            paths, writer_state = write_webdataset_aggregate(
                bundle=bundle,
                release_id="test_agg",
                matrix_root=matrix_root,
                _writer_state=writer_state,
                _is_last_chunk=is_last,
            )

        # After last chunk, writer_state should be None
        assert writer_state is None, (
            f"writer_state should be None after _is_last_chunk=True, "
            f"got {writer_state}"
        )

        # Verify paths
        assert paths is not None
        assert "shard_paths" in paths
        assert "shard_index" in paths
        assert len(paths["shard_paths"]) > 0, "no shard files were created"

        # Verify all shard files exist
        for shard_path in paths["shard_paths"]:
            assert shard_path.exists(), f"shard file not found: {shard_path}"

        # Verify shard-index Parquet exists and has correct total rows
        import pyarrow.parquet as pq
        shard_index_path = paths["shard_index"]
        assert shard_index_path.exists()
        index_table = pq.read_table(str(shard_index_path))
        assert index_table.num_rows == expected_total, (
            f"shard-index row count {index_table.num_rows} != expected {expected_total}"
        )

        # Verify global_row_index values are contiguous
        global_idx_col = index_table.column("global_row_index").to_numpy()
        expected_global = np.arange(expected_total, dtype=np.int64)
        np.testing.assert_array_equal(
            global_idx_col, expected_global,
            err_msg="global_row_index not contiguous in shard-index"
        )

        # Verify first global_row_index starts at 0
        assert index_table.column("global_row_index")[0].as_py() == 0, (
            "first global_row_index should be 0"
        )

        # Verify last global_row_index is expected_total - 1
        assert index_table.column("global_row_index")[-1].as_py() == expected_total - 1, (
            f"last global_row_index should be {expected_total - 1}"
        )

        # Spot check: read a cell from a shard and verify it contains valid data
        import tarfile, pickle
        first_shard = paths["shard_paths"][0]
        with tarfile.open(str(first_shard), "r") as tar:
            members = tar.getmembers()
            assert len(members) > 0, "first shard is empty"
            first_cell = members[0]
            extracted = tar.extractfile(first_cell)
            assert extracted is not None
            record = pickle.loads(extracted.read())
            assert "global_row_index" in record
            assert "expressed_gene_indices" in record
            assert "expression_counts" in record
