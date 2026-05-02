"""Phase 5: validation matrix smoke for backend×topology writer APIs.

This module validates that backend writer entrypoints accept translated
``ChunkBundle`` inputs, write expected output artifacts, and preserve basic
shape/content properties for both federated and aggregate topologies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _load_h5ad_backed(file_path: Path) -> Any:
    import anndata as ad

    return ad.read_h5ad(str(file_path), backed="r")


def _materialize_csr(adata: Any) -> Any:
    return adata.X[:].tocsr()


def _dataset_spec(dataset_id: str, dataset_index: int, adata: Any, start: int) -> Any:
    from perturb_data_lab.materializers.chunk_translation import DatasetSpec

    return DatasetSpec(
        dataset_id=dataset_id,
        dataset_index=dataset_index,
        file_path=Path(adata.filename),
        rows=adata.n_obs,
        pairs=0,
        local_vocabulary_size=adata.n_vars,
        nnz_total=int(adata.X.nnz) if hasattr(adata.X, "nnz") else 0,
        global_row_start=start,
        global_row_stop=start + adata.n_obs,
    )


def _translate_full_chunk(adata: Any, dataset_id: str, dataset_index: int, start: int):
    from perturb_data_lab.materializers.chunk_translation import _translate_chunk

    spec = _dataset_spec(dataset_id, dataset_index, adata, start)
    csr = _materialize_csr(adata)
    return _translate_chunk(dataset=spec, matrix_chunk=csr, chunk_start=0)


def _write_size_factor_parquet(path: Path, n_rows: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"size_factor": pa.array(np.ones(n_rows, dtype=np.float64))})
    pq.write_table(table, str(path))


class TestFederatedWriters:
    DUMMY_00_PATH = Path(
        "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/dummy_data/dummy_00_counts.h5ad"
    )

    @pytest.fixture
    def dummy_00(self):
        if not self.DUMMY_00_PATH.exists():
            pytest.skip(f"dummy_00 not found at {self.DUMMY_00_PATH}")
        return _load_h5ad_backed(self.DUMMY_00_PATH)

    def test_arrow_parquet_federated_writer(self, dummy_00, tmp_path: Path):
        from perturb_data_lab.materializers.backends.arrow_parquet import (
            read_arrow_parquet_cell,
            write_arrow_parquet_federated,
        )

        matrix_root = tmp_path / "arrow-parquet-fed"
        bundle = _translate_full_chunk(dummy_00, "dummy_00", 0, 0)
        paths, state = write_arrow_parquet_federated(
            bundle=bundle,
            dataset_id="dummy_00",
            matrix_root=matrix_root,
            _writer_state=None,
            _is_last_chunk=True,
        )
        assert state is None
        assert paths["cells"].exists()
        sf_path = matrix_root / "dummy_00-size-factor.parquet"
        _write_size_factor_parquet(sf_path, dummy_00.n_obs)
        idx, counts, sf = read_arrow_parquet_cell(
            paths["cells"],
            0,
            size_factor_path=sf_path,
        )
        assert len(idx) > 0
        assert len(idx) == len(counts)
        assert sf > 0

    def test_arrow_ipc_federated_writer(self, dummy_00, tmp_path: Path):
        from perturb_data_lab.materializers.backends.arrow_ipc import (
            read_arrow_ipc_cell,
            write_arrow_ipc_federated,
        )

        matrix_root = tmp_path / "arrow-ipc-fed"
        bundle = _translate_full_chunk(dummy_00, "dummy_00", 0, 0)
        paths, state = write_arrow_ipc_federated(
            bundle=bundle,
            dataset_id="dummy_00",
            matrix_root=matrix_root,
            _writer_state=None,
            _is_last_chunk=True,
        )
        assert state is None
        assert paths["cells"].exists()
        idx, counts, sf = read_arrow_ipc_cell(paths["cells"], 0)
        assert len(idx) > 0
        assert len(idx) == len(counts)
        assert sf > 0

    def test_webdataset_federated_writer(self, dummy_00, tmp_path: Path):
        from perturb_data_lab.materializers.backends.webdataset import (
            read_webdataset_cell,
            write_webdataset_federated,
        )

        matrix_root = tmp_path / "webdataset-fed"
        bundle = _translate_full_chunk(dummy_00, "dummy_00", 0, 0)
        paths, state = write_webdataset_federated(
            bundle=bundle,
            dataset_id="dummy_00",
            matrix_root=matrix_root,
            _writer_state=None,
            _is_last_chunk=True,
        )
        assert state is None
        shard_path = paths["shard_path"]
        assert shard_path.exists()
        idx, counts, sf = read_webdataset_cell(shard_path, "cell_00000000.pkl")
        assert len(idx) > 0
        assert len(idx) == len(counts)
        assert sf > 0

    def test_zarr_federated_writer(self, dummy_00, tmp_path: Path):
        from perturb_data_lab.materializers.backends.zarr import (
            read_zarr_cell,
            write_zarr_federated,
        )

        matrix_root = tmp_path / "zarr-fed"
        bundle = _translate_full_chunk(dummy_00, "dummy_00", 0, 0)
        paths, state = write_zarr_federated(
            bundle=bundle,
            dataset_id="dummy_00",
            matrix_root=matrix_root,
            _writer_state=None,
            _is_last_chunk=True,
        )
        assert state is None
        assert paths["indices"].exists()
        assert paths["counts"].exists()
        assert paths["row_offsets"].exists()
        idx, counts, sf = read_zarr_cell(
            paths["indices"],
            paths["counts"],
            0,
            row_offsets_path=paths["row_offsets"],
        )
        assert len(idx) > 0
        assert len(idx) == len(counts)
        assert sf > 0


class TestAggregateWriters:
    DUMMY_00_PATH = Path(
        "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/dummy_data/dummy_00_counts.h5ad"
    )
    DUMMY_01_PATH = Path(
        "/autofs/projects-t3/lilab/yangqisu/repos/data_perturb_v2/dummy_data/dummy_01_counts.h5ad"
    )

    @pytest.fixture
    def bundles(self):
        if not self.DUMMY_00_PATH.exists() or not self.DUMMY_01_PATH.exists():
            pytest.skip("dummy datasets not found")
        adata0 = _load_h5ad_backed(self.DUMMY_00_PATH)
        adata1 = _load_h5ad_backed(self.DUMMY_01_PATH)
        b0 = _translate_full_chunk(adata0, "dummy_00", 0, 0)
        b1 = _translate_full_chunk(adata1, "dummy_01", 1, adata0.n_obs)
        return b0, b1, adata0.n_obs + adata1.n_obs

    def test_webdataset_aggregate_writer(self, bundles, tmp_path: Path):
        import pyarrow.parquet as pq
        from perturb_data_lab.materializers.backends.webdataset import write_webdataset_aggregate

        b0, b1, total_rows = bundles
        matrix_root = tmp_path / "webdataset-agg"

        paths, state = write_webdataset_aggregate(
            bundle=b0,
            dataset_id="dummy_00",
            matrix_root=matrix_root,
            _writer_state=None,
            _is_last_chunk=False,
        )
        assert state is not None
        paths, state = write_webdataset_aggregate(
            bundle=b1,
            dataset_id="dummy_01",
            matrix_root=matrix_root,
            _writer_state=state,
            _is_last_chunk=True,
        )
        assert state is None
        assert paths["shard_index"].exists()
        index_table = pq.read_table(str(paths["shard_index"]))
        assert index_table.num_rows == total_rows

    def test_zarr_aggregate_writer(self, bundles, tmp_path: Path):
        import zarr
        from perturb_data_lab.materializers.backends.zarr import write_zarr_aggregate

        b0, b1, total_rows = bundles
        matrix_root = tmp_path / "zarr-agg"

        paths, state = write_zarr_aggregate(
            bundle=b0,
            dataset_id="dummy_00",
            matrix_root=matrix_root,
            _writer_state=None,
            _is_last_chunk=False,
        )
        assert state is not None
        paths, state = write_zarr_aggregate(
            bundle=b1,
            dataset_id="dummy_01",
            matrix_root=matrix_root,
            _writer_state=state,
            _is_last_chunk=True,
        )
        assert state is None
        row_offsets = zarr.open(str(paths["row_offsets"]), mode="r")["row_offsets"][:]
        assert len(row_offsets) == total_rows + 1

    def test_lance_aggregate_writer(self, bundles, tmp_path: Path):
        lance = pytest.importorskip("lance")
        from perturb_data_lab.materializers.backends.lance import write_lance_aggregate

        b0, b1, total_rows = bundles
        matrix_root = tmp_path / "lance-agg"

        paths, state = write_lance_aggregate(
            bundle=b0,
            dataset_id="dummy_00",
            matrix_root=matrix_root,
            _writer_state=None,
            _is_last_chunk=False,
        )
        assert state is not None
        paths, state = write_lance_aggregate(
            bundle=b1,
            dataset_id="dummy_01",
            matrix_root=matrix_root,
            _writer_state=state,
            _is_last_chunk=True,
        )
        assert state is None
        ds = lance.dataset(str(paths["cells"]))
        assert ds.count_rows() == total_rows
