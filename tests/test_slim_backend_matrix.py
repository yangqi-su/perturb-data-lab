from __future__ import annotations

import pytest

from perturb_data_lab.cli import BACKEND_CHOICES
from perturb_data_lab.loaders.corpus_loader import _normalize_backend
from perturb_data_lab.loaders.expression import DatasetEntry, build_expression_reader
from perturb_data_lab.materializers.backends import AVAILABLE_WRITERS, build_backend_fn


def test_cli_backend_choices_are_lance_and_zarr_only() -> None:
    assert BACKEND_CHOICES == ["lance", "zarr"]


@pytest.mark.parametrize("backend", ["lance", "zarr"])
@pytest.mark.parametrize("topology", ["aggregate", "federated"])
def test_materializer_backend_matrix_is_lance_and_zarr_only(
    backend: str,
    topology: str,
) -> None:
    assert sorted(AVAILABLE_WRITERS) == ["lance", "zarr"]
    assert callable(build_backend_fn(backend, topology))


@pytest.mark.parametrize(
    "backend",
    ["arrow-parquet", "arrow-ipc", "hf-datasets", "webdataset", "tiledb", "csr-memmap"],
)
def test_removed_materializer_backends_raise_clear_error(backend: str) -> None:
    with pytest.raises(AssertionError, match="unknown backend"):
        build_backend_fn(backend, "federated")


@pytest.mark.parametrize(
    "backend",
    ["arrow-parquet", "arrow_ipc", "hf-datasets", "parquet", "tiledb", "csr-memmap"],
)
def test_removed_corpus_backends_fail_clearly(backend: str) -> None:
    with pytest.raises(ValueError, match="Unsupported corpus backend"):
        _normalize_backend(backend)


@pytest.mark.parametrize("backend", ["arrow_ipc", "hf_datasets", "parquet", "tiledb", "webdataset", "csr_memmap"])
def test_removed_expression_readers_raise_clear_error(backend: str) -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        build_expression_reader(backend, "federated", [DatasetEntry("ds", 0, 1)])


@pytest.mark.parametrize(
    "symbol",
    [
        "AggregateTileDBReader",
        "AggregateCsrMemmapReader",
        "FederatedArrowIpcReader",
        "FederatedHfDatasetsReader",
        "FederatedParquetReader",
        "FederatedWebDatasetReader",
        "ArrowIpcDatasetEntry",
        "HfDatasetsDatasetEntry",
        "ParquetDatasetEntry",
        "WebDatasetEntry",
        "CsrMemmapShardEntry",
    ],
)
def test_removed_expression_symbols_are_not_exported_from_loaders(symbol: str) -> None:
    with pytest.raises(ImportError):
        exec(f"from perturb_data_lab.loaders import {symbol}", {}, {})
