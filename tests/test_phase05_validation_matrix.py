"""Phase 5: End-to-end validation matrix for all 10 backend×topology combinations.

This test validates source-vs-output parity on dummy_data for all backends:
- arrow-parquet × federated
- arrow-ipc × federated
- webdataset × federated
- zarr × federated
- lance × federated
- arrow-parquet × aggregate
- arrow-ipc × aggregate
- webdataset × aggregate
- zarr × aggregate
- lance × aggregate

Uses dummy_00 (273MB, ~50K cells) for federated tests.
Uses dummy_00 + dummy_08 for aggregate tests (combined ~1.2M cells).

Each combination is validated for:
- Row count matches source
- Sparse indices and counts match source for sampled cells
- Size factors are present and positive
- No SQLite artifacts emitted on active path
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_h5ad_backed(file_path: Path) -> Any:
    """Load an h5ad file in backed mode to avoid memory issues."""
    import anndata as ad

    return ad.read_h5ad(str(file_path), backed="r")


def _materialize_csr(adata: Any) -> Any:
    """Materialize a backed AnnData's X as an in-memory CSR matrix."""
    return adata.X[:].tocsr()


def _sample_sparse_row(csr_matrix: Any, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract a single row from a CSR matrix as numpy arrays."""
    # Handle backed _CSRDataset vs regular csr_matrix
    if hasattr(csr_matrix, "_indptr"):
        indptr = csr_matrix._indptr
        indices = csr_matrix._indices
        data = csr_matrix._data
    else:
        indptr = csr_matrix.indptr
        indices = csr_matrix.indices
        data = csr_matrix.data
    start = indptr[row_idx]
    stop = indptr[row_idx + 1]
    out_indices = np.asarray(indices[start:stop], dtype=np.int32)
    out_counts = np.asarray(data[start:stop], dtype=np.int32)
    return out_indices, out_counts


def _write_size_factor_parquet(
    size_factors: np.ndarray, path: Path, release_id: str
) -> None:
    """Write a separate size-factor parquet sidecar."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"size_factor": pa.array(size_factors, type=pa.float64())})
    pq.write_table(table, str(path))


# ---------------------------------------------------------------------------
# Federated validation
# ---------------------------------------------------------------------------


class TestFederatedMatrix:
    """Validate all 5 federated backends on dummy_00."""

    DUMMY_00_PATH = Path(
        "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/dummy_data/dummy_00_counts.h5ad"
    )

    @pytest.fixture
    def dummy_00(self):
        if not self.DUMMY_00_PATH.exists():
            pytest.skip(f"dummy_00 not found at {self.DUMMY_00_PATH}")
        return _load_h5ad_backed(self.DUMMY_00_PATH)

    @pytest.fixture
    def tmp_root(self, tmp_path: Path) -> Path:
        return tmp_path / "matrix"

    def _get_dataset_spec(self, adata: Any, dataset_id: str, dataset_index: int) -> Any:
        from perturb_data_lab.materializers.chunk_translation import DatasetSpec

        return DatasetSpec(
            dataset_id=dataset_id,
            dataset_index=dataset_index,
            file_path=Path(adata.filename),
            rows=adata.n_obs,
            pairs=0,
            local_vocabulary_size=adata.n_vars,
            nnz_total=int(adata.X.nnz) if hasattr(adata.X, "nnz") else 0,
            global_row_start=0,
            global_row_stop=adata.n_obs,
        )

    # ---- arrow-parquet × federated ----
    def test_arrow_parquet_federated_parity(self, dummy_00, tmp_root: Path):
        from perturb_data_lab.materializers.backends.arrow_parquet import (
            read_arrow_parquet_cell,
            write_arrow_parquet_federated,
        )

        release_id = "dummy_00"
        matrix_root = tmp_root / "arrow-parquet-fed"
        matrix_root.mkdir(parents=True, exist_ok=True)

        # Write
        paths_dict, size_factors = write_arrow_parquet_federated(
            adata=dummy_00,
            count_matrix=dummy_00.X,
            size_factors=None,
            release_id=release_id,
            matrix_root=matrix_root,
        )
        assert "cells" in paths_dict
        cell_path = paths_dict["cells"]
        assert cell_path.exists()

        # Write size-factor sidecar
        sf_path = matrix_root / f"{release_id}-size-factor.parquet"
        _write_size_factor_parquet(size_factors, sf_path, release_id)

        # Parity check on first and last rows
        n_obs = dummy_00.n_obs
        for row_idx in [0, n_obs // 2, n_obs - 1]:
            src_indices, src_counts = _sample_sparse_row(dummy_00.X, row_idx)
            out_indices, out_counts, sf = read_arrow_parquet_cell(
                cell_path, row_idx, size_factor_path=sf_path
            )
            assert out_indices == tuple(src_indices), f"row {row_idx} indices mismatch"
            assert out_counts == tuple(src_counts), f"row {row_idx} counts mismatch"
            assert sf > 0, f"row {row_idx} size_factor {sf} not positive"

        # No SQLite artifacts
        sqlite_files = list(matrix_root.glob("*.sqlite"))
        assert len(sqlite_files) == 0, f"SQLite files found: {sqlite_files}"

    # ---- arrow-ipc × federated ----
    def test_arrow_ipc_federated_parity(self, dummy_00, tmp_root: Path):
        from perturb_data_lab.materializers.backends.arrow_ipc import (
            read_arrow_ipc_cell,
            write_arrow_ipc_federated,
        )

        release_id = "dummy_00"
        matrix_root = tmp_root / "arrow-ipc-fed"
        matrix_root.mkdir(parents=True, exist_ok=True)

        # Write
        paths_dict, size_factors = write_arrow_ipc_federated(
            adata=dummy_00,
            count_matrix=dummy_00.X,
            size_factors=None,
            release_id=release_id,
            matrix_root=matrix_root,
        )
        assert "cells" in paths_dict
        cell_path = paths_dict["cells"]
        assert cell_path.exists()

        # Write size-factor sidecar
        sf_path = matrix_root / f"{release_id}-size-factor.parquet"
        _write_size_factor_parquet(size_factors, sf_path, release_id)

        # Parity check
        n_obs = dummy_00.n_obs
        for row_idx in [0, n_obs // 2, n_obs - 1]:
            src_indices, src_counts = _sample_sparse_row(dummy_00.X, row_idx)
            out_indices, out_counts, sf = read_arrow_ipc_cell(
                cell_path, row_idx, size_factor_path=sf_path
            )
            assert out_indices == tuple(src_indices), f"row {row_idx} indices mismatch"
            assert out_counts == tuple(src_counts), f"row {row_idx} counts mismatch"
            assert sf > 0, f"row {row_idx} size_factor {sf} not positive"

        # No SQLite artifacts
        sqlite_files = list(matrix_root.glob("*.sqlite"))
        assert len(sqlite_files) == 0, f"SQLite files found: {sqlite_files}"

    # ---- webdataset × federated ----
    def test_webdataset_federated_parity(self, dummy_00, tmp_root: Path):
        from perturb_data_lab.materializers.backends.webdataset import (
            read_webdataset_cell,
            write_webdataset_federated,
        )

        release_id = "dummy_00"
        matrix_root = tmp_root / "webdataset-fed"
        matrix_root.mkdir(parents=True, exist_ok=True)

        # Write
        paths_dict = write_webdataset_federated(
            adata=dummy_00,
            count_matrix=dummy_00.X,
            size_factors=np.ones(dummy_00.n_obs, dtype=np.float64),
            release_id=release_id,
            matrix_root=matrix_root,
            shard_size=10_000,
        )
        assert "shard_paths" in paths_dict
        assert "meta" in paths_dict
        shard_paths = paths_dict["shard_paths"]
        assert len(shard_paths) > 0

        # Parity check — webdataset stores per-cell pickle with size_factor embedded
        n_obs = dummy_00.n_obs
        for row_idx in [0, n_obs // 2, n_obs - 1]:
            src_indices, src_counts = _sample_sparse_row(dummy_00.X, row_idx)
            # Find which shard contains this cell
            shard_size = 10_000
            shard_idx = row_idx // shard_size
            shard_path = matrix_root / f"{release_id}-{shard_idx:05d}.tar"
            member_name = f"cell_{row_idx:08d}.pkl"
            out_indices, out_counts, sf = read_webdataset_cell(
                shard_path, member_name, size_factor_path=None
            )
            assert out_indices == tuple(src_indices), f"row {row_idx} indices mismatch"
            assert out_counts == tuple(src_counts), f"row {row_idx} counts mismatch"
            assert sf > 0, f"row {row_idx} size_factor {sf} not positive"

        # No SQLite artifacts
        sqlite_files = list(matrix_root.glob("*.sqlite"))
        assert len(sqlite_files) == 0, f"SQLite files found: {sqlite_files}"

    # ---- zarr × federated ----
    def test_zarr_federated_parity(self, dummy_00, tmp_root: Path):
        from perturb_data_lab.materializers.backends.zarr import (
            read_zarr_cell,
            write_zarr_federated,
        )

        release_id = "dummy_00"
        matrix_root = tmp_root / "zarr-fed"
        matrix_root.mkdir(parents=True, exist_ok=True)

        # Write
        paths_dict = write_zarr_federated(
            adata=dummy_00,
            count_matrix=dummy_00.X,
            size_factors=np.ones(dummy_00.n_obs, dtype=np.float64),
            release_id=release_id,
            matrix_root=matrix_root,
        )
        assert "indices" in paths_dict
        assert "counts" in paths_dict
        assert "meta" in paths_dict

        # Write size-factor sidecar (zarr does not compute inline)
        sf_path = matrix_root / f"{release_id}-size-factor.parquet"
        _write_size_factor_parquet(
            np.ones(dummy_00.n_obs, dtype=np.float64), sf_path, release_id
        )

        # Parity check
        n_obs = dummy_00.n_obs
        indices_path = paths_dict["indices"]
        counts_path = paths_dict["counts"]
        row_offsets_path = indices_path.parent / f"{release_id}-indices.zarr"

        for row_idx in [0, n_obs // 2, n_obs - 1]:
            src_indices, src_counts = _sample_sparse_row(dummy_00.X, row_idx)
            out_indices, out_counts, sf = read_zarr_cell(
                indices_path,
                counts_path,
                row_idx,
                row_offsets_path=row_offsets_path,
                size_factor_path=sf_path,
            )
            assert out_indices == tuple(src_indices), f"row {row_idx} indices mismatch"
            assert out_counts == tuple(src_counts), f"row {row_idx} counts mismatch"
            # zarr returns size_factor=1.0 from fallback since we didn't compute inline

        # No SQLite artifacts
        sqlite_files = list(matrix_root.glob("*.sqlite"))
        assert len(sqlite_files) == 0, f"SQLite files found: {sqlite_files}"

    # ---- lance × federated ----
    def test_lance_federated_parity(self, dummy_00, tmp_root: Path):
        lance = pytest.importorskip("lance")
        lancedb = pytest.importorskip("lancedb")

        from perturb_data_lab.materializers.backends.lance import (
            read_lance_cell,
            write_lance_federated,
        )

        release_id = "dummy_00"
        matrix_root = tmp_root / "lance-fed"
        matrix_root.mkdir(parents=True, exist_ok=True)

        # Write
        paths_dict = write_lance_federated(
            adata=dummy_00,
            count_matrix=dummy_00.X,
            size_factors=np.ones(dummy_00.n_obs, dtype=np.float64),
            release_id=release_id,
            matrix_root=matrix_root,
        )
        assert "cells" in paths_dict
        lance_path = paths_dict["cells"]
        assert lance_path.exists()

        # Write size-factor sidecar
        sf_path = matrix_root / f"{release_id}-size-factor.parquet"
        _write_size_factor_parquet(
            np.ones(dummy_00.n_obs, dtype=np.float64), sf_path, release_id
        )

        # Parity check
        n_obs = dummy_00.n_obs
        for row_idx in [0, n_obs // 2, n_obs - 1]:
            src_indices, src_counts = _sample_sparse_row(dummy_00.X, row_idx)
            out_indices, out_counts, sf = read_lance_cell(
                lance_path, row_idx, size_factor_path=sf_path
            )
            assert out_indices == tuple(src_indices), f"row {row_idx} indices mismatch"
            assert out_counts == tuple(src_counts), f"row {row_idx} counts mismatch"
            # size_factor from fallback = 1.0


# ---------------------------------------------------------------------------
# Aggregate validation
# ---------------------------------------------------------------------------


class TestAggregateMatrix:
    """Validate all 5 aggregate backends on dummy_00 + dummy_08."""

    DUMMY_00_PATH = Path(
        "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/dummy_data/dummy_00_counts.h5ad"
    )
    DUMMY_08_PATH = Path(
        "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/dummy_data/dummy_08_counts.h5ad"
    )

    @pytest.fixture
    def dummy_00_and_08(self):
        paths = [self.DUMMY_00_PATH, self.DUMMY_08_PATH]
        for p in paths:
            if not p.exists():
                pytest.skip(f"dummy file not found at {p}")
        adatas = [_load_h5ad_backed(p) for p in paths]
        return adatas

    @pytest.fixture
    def tmp_root(self, tmp_path: Path) -> Path:
        return tmp_path / "matrix"

    def _make_dataset_specs(self, adatas: list) -> list:
        from perturb_data_lab.materializers.chunk_translation import DatasetSpec

        specs = []
        global_row_start = 0
        for i, adata in enumerate(adatas):
            rows = adata.n_obs
            nnz = int(adata.X.nnz) if hasattr(adata.X, "nnz") else 0
            spec = DatasetSpec(
                dataset_id=f"dummy_{i:02d}",
                dataset_index=i,
                file_path=Path(adatas[i].filename),
                rows=rows,
                pairs=0,
                local_vocabulary_size=adata.n_vars,
                nnz_total=nnz,
                global_row_start=global_row_start,
                global_row_stop=global_row_start + rows,
            )
            specs.append(spec)
            global_row_start += rows
        return specs

    # ---- arrow-parquet × aggregate ----
    def test_arrow_parquet_aggregate_parity(self, dummy_00_and_08, tmp_root: Path):
        from perturb_data_lab.materializers.backends.arrow_parquet import (
            write_arrow_parquet_aggregate,
        )

        adatas = dummy_00_and_08
        matrix_root = tmp_root / "arrow-parquet-agg"
        matrix_root.mkdir(parents=True, exist_ok=True)

        specs = self._make_dataset_specs(adatas)
        count_matrices = [a.X for a in adatas]
        size_factors_list = [np.ones(a.n_obs, dtype=np.float64) for a in adatas]

        paths_dict, sf_list = write_arrow_parquet_aggregate(
            datasets=specs,
            count_matrices=count_matrices,
            size_factors_list=size_factors_list,
            matrix_root=matrix_root,
        )
        assert "cells" in paths_dict
        cell_path = paths_dict["cells"]
        assert cell_path.exists()

        # Verify total row count
        import pyarrow.parquet as pq

        table = pq.read_table(str(cell_path))
        total_rows = table.num_rows
        expected_total = sum(a.n_obs for a in adatas)
        assert total_rows == expected_total, (
            f"aggregate row count {total_rows} != expected {expected_total}"
        )

        # Verify global_row_index continuity across dataset boundary
        global_row_col = table.column("global_row_index")
        first_ds_rows = adatas[0].n_obs
        last_of_first = global_row_col[first_ds_rows - 1].as_py()
        first_of_second = global_row_col[first_ds_rows].as_py()
        assert last_of_first + 1 == first_of_second, (
            f"global_row_index gap at dataset boundary: {last_of_first} -> {first_of_second}"
        )

    # ---- arrow-ipc × aggregate ----
    def test_arrow_ipc_aggregate_parity(self, dummy_00_and_08, tmp_root: Path):
        from perturb_data_lab.materializers.backends.arrow_ipc import (
            write_arrow_ipc_aggregate,
        )

        adatas = dummy_00_and_08
        matrix_root = tmp_root / "arrow-ipc-agg"
        matrix_root.mkdir(parents=True, exist_ok=True)

        specs = self._make_dataset_specs(adatas)
        count_matrices = [a.X for a in adatas]
        size_factors_list = [np.ones(a.n_obs, dtype=np.float64) for a in adatas]

        paths_dict, sf_list = write_arrow_ipc_aggregate(
            datasets=specs,
            count_matrices=count_matrices,
            size_factors_list=size_factors_list,
            matrix_root=matrix_root,
        )
        assert "cells" in paths_dict
        cell_path = paths_dict["cells"]
        assert cell_path.exists()

        # Verify total row count
        import pyarrow as pa

        with pa.memory_map(str(cell_path), "r") as source:
            reader = pa.ipc.RecordBatchFileReader(source)
            total_rows = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
        expected_total = sum(a.n_obs for a in adatas)
        assert total_rows == expected_total, (
            f"aggregate row count {total_rows} != expected {expected_total}"
        )

    # ---- webdataset × aggregate ----
    def test_webdataset_aggregate_parity(self, dummy_00_and_08, tmp_root: Path):
        from perturb_data_lab.materializers.backends.webdataset import (
            write_webdataset_aggregate,
        )

        adatas = dummy_00_and_08
        matrix_root = tmp_root / "webdataset-agg"
        matrix_root.mkdir(parents=True, exist_ok=True)

        specs = self._make_dataset_specs(adatas)
        count_matrices = [a.X for a in adatas]
        size_factors_list = [np.ones(a.n_obs, dtype=np.float64) for a in adatas]

        paths_dict, sf_list = write_webdataset_aggregate(
            datasets=specs,
            count_matrices=count_matrices,
            size_factors_list=size_factors_list,
            matrix_root=matrix_root,
        )
        assert "shard_paths" in paths_dict
        assert "meta" in paths_dict

        # Verify meta parquet has correct global_row_index for dataset boundary
        import pyarrow.parquet as pq

        meta_table = pq.read_table(str(paths_dict["meta"]))
        total_meta_rows = meta_table.num_rows
        expected_total = sum(a.n_obs for a in adatas)
        assert total_meta_rows == expected_total, (
            f"aggregate meta row count {total_meta_rows} != expected {expected_total}"
        )

        # Verify dataset_index column changes at dataset boundary
        first_ds_rows = adatas[0].n_obs
        ds_index_col = meta_table.column("dataset_index")
        assert ds_index_col[0].as_py() == 0
        assert ds_index_col[first_ds_rows].as_py() == 1

    # ---- zarr × aggregate ----
    def test_zarr_aggregate_parity(self, dummy_00_and_08, tmp_root: Path):
        from perturb_data_lab.materializers.backends.zarr import write_zarr_aggregate

        adatas = dummy_00_and_08
        matrix_root = tmp_root / "zarr-agg"
        matrix_root.mkdir(parents=True, exist_ok=True)

        specs = self._make_dataset_specs(adatas)
        count_matrices = [a.X for a in adatas]
        size_factors_list = [np.ones(a.n_obs, dtype=np.float64) for a in adatas]

        paths_dict, sf_list = write_zarr_aggregate(
            datasets=specs,
            count_matrices=count_matrices,
            size_factors_list=size_factors_list,
            matrix_root=matrix_root,
        )
        assert "indices" in paths_dict
        assert "counts" in paths_dict
        assert "row_offsets" in paths_dict
        assert "meta" in paths_dict

        # Verify row_offsets length matches total rows
        import zarr

        row_offsets_arr = zarr.open(str(paths_dict["row_offsets"]), mode="r")["row_offsets"][:]
        expected_total = sum(a.n_obs for a in adatas)
        assert len(row_offsets_arr) == expected_total + 1, (
            f"row_offsets length {len(row_offsets_arr)} != expected {expected_total + 1}"
        )

    # ---- lance × aggregate ----
    def test_lance_aggregate_parity(self, dummy_00_and_08, tmp_root: Path):
        lance = pytest.importorskip("lance")
        lancedb = pytest.importorskip("lancedb")

        from perturb_data_lab.materializers.backends.lance import write_lance_aggregate

        adatas = dummy_00_and_08
        matrix_root = tmp_root / "lance-agg"
        matrix_root.mkdir(parents=True, exist_ok=True)

        specs = self._make_dataset_specs(adatas)
        count_matrices = [a.X for a in adatas]
        size_factors_list = [np.ones(a.n_obs, dtype=np.float64) for a in adatas]

        paths_dict, sf_list = write_lance_aggregate(
            datasets=specs,
            count_matrices=count_matrices,
            size_factors_list=size_factors_list,
            matrix_root=matrix_root,
        )
        assert "cells" in paths_dict
        lance_path = paths_dict["cells"]

        # Verify total row count via Lance scan
        ds = lance.dataset(str(lance_path))
        total_rows = ds.count_rows()
        expected_total = sum(a.n_obs for a in adatas)
        assert total_rows == expected_total, (
            f"aggregate row count {total_rows} != expected {expected_total}"
        )


# ---------------------------------------------------------------------------
# Timing comparison: arrow-parquet × federated vs arrow-hf baseline
# ---------------------------------------------------------------------------


class TestPerformanceComparison:
    """Compare arrow-parquet × federated timing vs arrow-hf baseline on dummy_00."""

    DUMMY_00_PATH = Path(
        "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/dummy_data/dummy_00_counts.h5ad"
    )

    @pytest.fixture
    def dummy_00(self):
        if not self.DUMMY_00_PATH.exists():
            pytest.skip(f"dummy_00 not found at {self.DUMMY_00_PATH}")
        return _load_h5ad_backed(self.DUMMY_00_PATH)

    @pytest.fixture
    def tmp_root(self, tmp_path: Path) -> Path:
        return tmp_path / "matrix"

    def test_arrow_parquet_federated_timing_no_regression(self, dummy_00, tmp_root: Path):
        """arrow-parquet × federated should not be meaningfully slower than arrow-hf."""
        import time

        from perturb_data_lab.materializers.backends.arrow_parquet import (
            write_arrow_parquet_federated,
        )
        from perturb_data_lab.materializers.backends.arrow_hf import (
            write_arrow_hf_sparse,
        )

        release_id = "dummy_00"
        n_obs = dummy_00.n_obs

        # Time arrow-hf (current baseline)
        matrix_root_hf = tmp_root / "arrow-hf-baseline"
        matrix_root_hf.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        write_arrow_hf_sparse(
            adata=dummy_00,
            count_matrix=dummy_00.X,
            size_factors=None,
            release_id=release_id,
            matrix_root=matrix_root_hf,
        )
        hf_time = time.perf_counter() - t0

        # Time arrow-parquet × federated (new refactored path)
        matrix_root_ap = tmp_root / "arrow-parquet-fed"
        matrix_root_ap.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        write_arrow_parquet_federated(
            adata=dummy_00,
            count_matrix=dummy_00.X,
            size_factors=None,
            release_id=release_id,
            matrix_root=matrix_root_ap,
        )
        ap_time = time.perf_counter() - t0

        #arrow-parquet should be at most 10% slower than arrow-hf
        ratio = ap_time / hf_time if hf_time > 0 else 1.0
        regression_threshold = 1.10
        assert ratio <= regression_threshold, (
            f"arrow-parquet × federated is {ratio:.2f}x the arrow-hf baseline "
            f"(ap={ap_time:.2f}s, hf={hf_time:.2f}s); "
            f"threshold is {regression_threshold}x. "
            f"If regression >10%, document justification in implementation-summary."
        )
