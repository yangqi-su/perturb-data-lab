"""Phase 4 smoke tests for loaders and samplers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perturb_data_lab.loaders import (
    ArrowHFCellReader,
    CellState,
    ExpressedZerosSampler,
    HVGRandomSampler,
    PerturbDataLoader,
    PerturbIterableDataset,
    RandomContextSampler,
    SamplerState,
    WebDatasetCellReader,
    ZarrCellReader,
    build_cell_reader,
)


class TestCellState:
    def test_cell_state_fields(self):
        cs = CellState(
            cell_id="cell_1",
            dataset_id="ds_1",
            dataset_release="v0",
            expressed_gene_indices=(0, 2, 5),
            expression_counts=(3, 1, 2),
            size_factor=1.0,
            canonical_perturbation={"perturbation_label": "gene_a"},
            canonical_context={"cell_context": "treat"},
            raw_fields={"orig_label": "cell_1"},
        )
        assert cs.cell_id == "cell_1"
        assert cs.expressed_gene_indices == (0, 2, 5)
        assert cs.expression_counts == (3, 1, 2)
        assert cs.size_factor == 1.0
        assert cs.canonical_perturbation["perturbation_label"] == "gene_a"

    def test_is_integer_sparse(self):
        cs = CellState(
            cell_id="c1",
            dataset_id="ds",
            dataset_release="v0",
            expressed_gene_indices=(1, 3),
            expression_counts=(2, 4),
            size_factor=1.0,
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )
        assert cs.expressed_gene_indices == (1, 3)


class TestSamplerState:
    def test_valid_modes(self):
        for mode in ["random_context", "expressed_zeros", "hvg_random"]:
            state = SamplerState(mode=mode, total_cells=100, n_genes=200)
            assert state.mode == mode

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="unknown sampler mode"):
            SamplerState(mode="invalid", total_cells=100, n_genes=200)


class TestRandomContextSampler:
    def test_sample_indices(self):
        state = SamplerState(mode="random_context", total_cells=100, n_genes=50)
        rng = np.random.default_rng(42)
        sampler = RandomContextSampler(state, rng)
        cell = CellState(
            cell_id="c1",
            dataset_id="ds",
            dataset_release="v0",
            expressed_gene_indices=(0, 10, 20),
            expression_counts=(1, 2, 3),
            size_factor=1.0,
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )
        context = sampler.sample_indices(cell, context_size=10)
        assert len(context) == 10
        assert context.max() < 50
        assert len(set(context)) == 10  # no duplicates

    def test_sample_indices_caps_at_n_genes(self):
        state = SamplerState(mode="random_context", total_cells=100, n_genes=50)
        rng = np.random.default_rng(42)
        sampler = RandomContextSampler(state, rng)
        cell = CellState(
            cell_id="c1",
            dataset_id="ds",
            dataset_release="v0",
            expressed_gene_indices=(0, 10, 20),
            expression_counts=(1, 2, 3),
            size_factor=1.0,
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )
        # context_size > n_genes should cap
        context = sampler.sample_indices(cell, context_size=200)
        assert len(context) == 50


class TestExpressedZerosSampler:
    def test_sample_indices(self):
        state = SamplerState(mode="expressed_zeros", total_cells=100, n_genes=50)
        rng = np.random.default_rng(42)
        sampler = ExpressedZerosSampler(state, rng)
        cell = CellState(
            cell_id="c1",
            dataset_id="ds",
            dataset_release="v0",
            expressed_gene_indices=(5, 10, 15),
            expression_counts=(3, 2, 1),
            size_factor=1.0,
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )
        context = sampler.sample_indices(cell)
        # Should include expressed genes
        assert 5 in context
        assert 10 in context
        assert 15 in context
        # Context should be unique
        assert len(set(context)) == len(context)

    def test_sample_batch(self, tmp_path: Path):
        cells_path, meta_path, sqlite_path = _make_synthetic_arrow_parquet(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"
        feature_reg = tmp_path / "feature-registry.yaml"

        from perturb_data_lab.materializers.models import (
            FeatureRegistryEntry,
            FeatureRegistryManifest,
        )
        from perturb_data_lab.contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL

        reg = FeatureRegistryManifest(
            kind="feature-registry",
            contract_version=CONTRACT_VERSION,
            registry_id="syn-reg",
            append_only=True,
            namespace="test",
            feature_id_field="gene_id",
            feature_label_field="gene_symbol",
            default_missing_value=MISSING_VALUE_LITERAL,
            entries=tuple(
                FeatureRegistryEntry(
                    token_id=j,
                    feature_id=f"gene_{j}",
                    feature_label=f"Gene {j}",
                    namespace="test",
                )
                for j in range(50)
            ),
        )
        reg.write_yaml(feature_reg)

        reader = ArrowHFCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            cells_parquet_path=cells_path,
            meta_parquet_path=meta_path,
            cell_meta_sqlite_path=sqlite_path,
            feature_registry_path=feature_reg,
        )

        state = SamplerState(mode="expressed_zeros", total_cells=100, n_genes=50)
        rng = np.random.default_rng(42)
        sampler = ExpressedZerosSampler(state, rng)
        result = sampler.sample_batch([0], reader, max_context=None)
        assert len(result) == 1
        assert isinstance(result[0][0], CellState)


class TestHVGRandomSampler:
    def test_sample_indices(self):
        hvg_set = (0, 5, 10, 15, 20)
        state = SamplerState(
            mode="hvg_random", total_cells=100, n_genes=50, hvg_set=hvg_set
        )
        rng = np.random.default_rng(42)
        sampler = HVGRandomSampler(state, rng)
        cell = CellState(
            cell_id="c1",
            dataset_id="ds",
            dataset_release="v0",
            expressed_gene_indices=(0, 5, 30),
            expression_counts=(3, 2, 1),
            size_factor=1.0,
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )
        context = sampler.sample_indices(cell)
        # Should include HVG genes present in expressed
        assert 0 in context
        assert 5 in context
        # 20 is HVG but not expressed in this cell
        # Context should be unique
        assert len(set(context)) == len(context)


# ---------------------------------------------------------------------------
# Synthetic Arrow/HF reader test fixture
# ---------------------------------------------------------------------------


def _make_synthetic_arrow_parquet(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create synthetic Arrow parquet files for smoke testing."""
    cells_path = tmp_path / "cells.parquet"
    meta_path = tmp_path / "meta.parquet"
    sqlite_path = tmp_path / "cell_meta.sqlite"

    import sqlite3

    n_cells = 20
    n_genes = 50

    # Synthetic sparse cells
    indices_list = []
    counts_list = []
    sf_list = []
    for i in range(n_cells):
        # 5 random expressed genes per cell
        gene_indices = sorted(
            np.random.default_rng(i).choice(n_genes, size=5, replace=False).tolist()
        )
        gene_counts = np.random.default_rng(i + 1000).integers(1, 10, size=5).tolist()
        indices_list.append(gene_indices)
        counts_list.append(gene_counts)
        sf_list.append(float(np.random.default_rng(i).uniform(0.5, 2.0)))

    table = pa.table(
        {
            "expressed_gene_indices": pa.array(indices_list, type=pa.list_(pa.int32())),
            "expression_counts": pa.array(counts_list, type=pa.list_(pa.int32())),
            "size_factor": pa.array(sf_list, type=pa.float64()),
        }
    )
    pq.write_table(table, cells_path)

    # Synthetic meta
    cell_ids = [f"syn_cell_{i}" for i in range(n_cells)]
    sf_vals = sf_list
    meta_table = pa.table(
        {
            "cell_id": pa.array(cell_ids, type=pa.string()),
            "size_factor": pa.array(sf_vals, type=pa.float64()),
            "raw_obs": pa.array([""] * n_cells, type=pa.string()),
        }
    )
    pq.write_table(meta_table, meta_path)

    # Synthetic SQLite — Phase 6+ schema with canonical_perturbation/context columns
    import json
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cell_meta "
        "(cell_id TEXT, dataset_id TEXT, dataset_release TEXT, "
        "size_factor REAL, canonical_perturbation TEXT, "
        "canonical_context TEXT, raw_obs TEXT)"
    )
    for i in range(n_cells):
        conn.execute(
            "INSERT INTO cell_meta VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"syn_cell_{i}",
                "synthetic_ds",
                "v0",
                sf_list[i],
                json.dumps({}),
                json.dumps({}),
                json.dumps({}),
            ),
        )
    conn.commit()
    conn.close()

    return cells_path, meta_path, sqlite_path


class TestArrowHFCellReader:
    def test_read_cell(self, tmp_path: Path):
        cells_path, meta_path, sqlite_path = _make_synthetic_arrow_parquet(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"
        feature_reg = tmp_path / "feature-registry.yaml"

        # Write minimal feature registry
        from perturb_data_lab.materializers.models import (
            FeatureRegistryEntry,
            FeatureRegistryManifest,
        )
        from perturb_data_lab.contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL

        reg = FeatureRegistryManifest(
            kind="feature-registry",
            contract_version=CONTRACT_VERSION,
            registry_id="syn-reg",
            append_only=True,
            namespace="test",
            feature_id_field="gene_id",
            feature_label_field="gene_symbol",
            default_missing_value=MISSING_VALUE_LITERAL,
            entries=tuple(
                FeatureRegistryEntry(
                    token_id=j,
                    feature_id=f"gene_{j}",
                    feature_label=f"Gene {j}",
                    namespace="test",
                )
                for j in range(50)
            ),
        )
        reg.write_yaml(feature_reg)

        reader = ArrowHFCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            cells_parquet_path=cells_path,
            meta_parquet_path=meta_path,
            cell_meta_sqlite_path=sqlite_path,
            feature_registry_path=feature_reg,
        )

        assert len(reader) == 20
        assert reader.total_genes == 50

        cell = reader.read_cell(0)
        assert isinstance(cell, CellState)
        assert cell.cell_id == "syn_cell_0"
        assert cell.dataset_id == "synthetic_ds"
        assert isinstance(cell.expressed_gene_indices, tuple)
        assert isinstance(cell.expression_counts, tuple)
        assert cell.size_factor > 0

    def test_read_cells_batch(self, tmp_path: Path):
        cells_path, meta_path, sqlite_path = _make_synthetic_arrow_parquet(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"
        feature_reg = tmp_path / "feature-registry.yaml"

        from perturb_data_lab.materializers.models import (
            FeatureRegistryEntry,
            FeatureRegistryManifest,
        )
        from perturb_data_lab.contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL

        reg = FeatureRegistryManifest(
            kind="feature-registry",
            contract_version=CONTRACT_VERSION,
            registry_id="syn-reg",
            append_only=True,
            namespace="test",
            feature_id_field="gene_id",
            feature_label_field="gene_symbol",
            default_missing_value=MISSING_VALUE_LITERAL,
            entries=tuple(
                FeatureRegistryEntry(
                    token_id=j,
                    feature_id=f"gene_{j}",
                    feature_label=f"Gene {j}",
                    namespace="test",
                )
                for j in range(50)
            ),
        )
        reg.write_yaml(feature_reg)

        reader = ArrowHFCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            cells_parquet_path=cells_path,
            meta_parquet_path=meta_path,
            cell_meta_sqlite_path=sqlite_path,
            feature_registry_path=feature_reg,
        )

        cells = reader.read_cells([0, 1, 2])
        assert len(cells) == 3
        assert all(isinstance(c, CellState) for c in cells)


class TestPerturbDataLoader:
    def test_map_style_dataset_getitem(self, tmp_path: Path):
        cells_path, meta_path, sqlite_path = _make_synthetic_arrow_parquet(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"
        feature_reg = tmp_path / "feature-registry.yaml"

        from perturb_data_lab.materializers.models import (
            FeatureRegistryEntry,
            FeatureRegistryManifest,
        )
        from perturb_data_lab.contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL

        reg = FeatureRegistryManifest(
            kind="feature-registry",
            contract_version=CONTRACT_VERSION,
            registry_id="syn-reg",
            append_only=True,
            namespace="test",
            feature_id_field="gene_id",
            feature_label_field="gene_symbol",
            default_missing_value=MISSING_VALUE_LITERAL,
            entries=tuple(
                FeatureRegistryEntry(
                    token_id=j,
                    feature_id=f"gene_{j}",
                    feature_label=f"Gene {j}",
                    namespace="test",
                )
                for j in range(50)
            ),
        )
        reg.write_yaml(feature_reg)

        reader = ArrowHFCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            cells_parquet_path=cells_path,
            meta_parquet_path=meta_path,
            cell_meta_sqlite_path=sqlite_path,
            feature_registry_path=feature_reg,
        )

        loader = PerturbDataLoader(
            reader,
            sampler_mode="random_context",
            shuffle=False,
            seed=42,
            context_size=10,
        )

        assert len(loader) == 20
        item = loader[0]
        assert "cell_id" in item
        assert "expressed_gene_indices" in item
        assert "expression_counts" in item
        assert "context_indices" in item
        assert len(item["context_indices"]) == 10


class TestPerturbIterableDataset:
    def test_iter(self, tmp_path: Path):
        cells_path, meta_path, sqlite_path = _make_synthetic_arrow_parquet(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"
        feature_reg = tmp_path / "feature-registry.yaml"

        from perturb_data_lab.materializers.models import (
            FeatureRegistryEntry,
            FeatureRegistryManifest,
        )
        from perturb_data_lab.contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL

        reg = FeatureRegistryManifest(
            kind="feature-registry",
            contract_version=CONTRACT_VERSION,
            registry_id="syn-reg",
            append_only=True,
            namespace="test",
            feature_id_field="gene_id",
            feature_label_field="gene_symbol",
            default_missing_value=MISSING_VALUE_LITERAL,
            entries=tuple(
                FeatureRegistryEntry(
                    token_id=j,
                    feature_id=f"gene_{j}",
                    feature_label=f"Gene {j}",
                    namespace="test",
                )
                for j in range(50)
            ),
        )
        reg.write_yaml(feature_reg)

        reader = ArrowHFCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            cells_parquet_path=cells_path,
            meta_parquet_path=meta_path,
            cell_meta_sqlite_path=sqlite_path,
            feature_registry_path=feature_reg,
        )

        dataset = PerturbIterableDataset(
            reader,
            sampler_mode="random_context",
            shuffle=False,
            seed=42,
            context_size=10,
        )

        items = list(dataset)
        assert len(items) == 20
        item = items[0]
        assert "cell_id" in item
        assert "context_indices" in item
        assert len(item["context_indices"]) == 10


class TestBuildCellReader:
    def test_unknown_backend_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="unknown backend reader"):
            build_cell_reader(
                backend="unknown",
                release_id="v0",
                corpus_index_path=tmp_path / "corpus-index.yaml",
            )


# ---------------------------------------------------------------------------
# Synthetic WebDataset tar-shard fixture
# ---------------------------------------------------------------------------


def _make_synthetic_webdataset_tar(tmp_path: Path) -> tuple[Path, Path]:
    """Create synthetic WebDataset tar shards for smoke testing."""
    import pickle
    import tarfile

    n_cells = 20
    n_genes = 50
    shard_size = 10
    n_shards = (n_cells + shard_size - 1) // shard_size

    shard_paths = []
    for shard_idx in range(n_shards):
        shard_path = tmp_path / f"syn-shard-{shard_idx:05d}.tar"
        shard_paths.append(shard_path)
        start = shard_idx * shard_size
        end = min(start + shard_size, n_cells)
        with tarfile.open(str(shard_path), "w") as tar:
            for i in range(start, end):
                rng = np.random.default_rng(i)
                n_expressed = rng.integers(3, 8)
                gene_indices = sorted(
                    rng.choice(n_genes, size=n_expressed, replace=False).tolist()
                )
                gene_counts = rng.integers(1, 10, size=n_expressed).tolist()
                record = {
                    "expressed_gene_indices": np.array(
                        gene_indices, dtype=np.int32
                    ).tobytes(),
                    "expression_counts": np.array(
                        gene_counts, dtype=np.int32
                    ).tobytes(),
                    "size_factor": float(rng.uniform(0.5, 2.0)),
                    "cell_id": f"syn_cell_{i}",
                }
                data_bytes = pickle.dumps(record)
                import io

                info = tarfile.TarInfo(name=f"cell_{i:08d}.pt")
                info.size = len(data_bytes)
                tar.addfile(info, io.BytesIO(data_bytes))

    meta_path = tmp_path / "syn-webdataset-meta.txt"
    with open(meta_path, "w") as f:
        for i in range(n_cells):
            f.write(f"syn_cell_{i}\t{i // shard_size}\t1.0\n")

    return shard_paths[0].parent, meta_path


class TestWebDatasetCellReader:
    def test_read_cell(self, tmp_path: Path):
        shard_dir, meta_path = _make_synthetic_webdataset_tar(tmp_path)
        shard_paths = sorted(tmp_path.glob("syn-shard-*.tar"))
        corpus_index = tmp_path / "corpus-index.yaml"

        reader = WebDatasetCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            shard_paths=shard_paths,
            meta_path=meta_path,
        )

        assert len(reader) == 20
        cell = reader.read_cell(0)
        assert isinstance(cell, CellState)
        assert cell.cell_id == "syn_cell_0"
        assert cell.dataset_release == "syn-v0"
        assert isinstance(cell.expressed_gene_indices, tuple)
        assert isinstance(cell.expression_counts, tuple)
        assert cell.size_factor > 0

    def test_read_cells_batch(self, tmp_path: Path):
        shard_dir, meta_path = _make_synthetic_webdataset_tar(tmp_path)
        shard_paths = sorted(tmp_path.glob("syn-shard-*.tar"))
        corpus_index = tmp_path / "corpus-index.yaml"

        reader = WebDatasetCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            shard_paths=shard_paths,
            meta_path=meta_path,
        )

        cells = reader.read_cells([0, 1, 2])
        assert len(cells) == 3
        assert all(isinstance(c, CellState) for c in cells)

    def test_total_genes_defaults_to_fallback(self, tmp_path: Path):
        shard_dir, meta_path = _make_synthetic_webdataset_tar(tmp_path)
        shard_paths = sorted(tmp_path.glob("syn-shard-*.tar"))
        corpus_index = tmp_path / "corpus-index.yaml"

        reader = WebDatasetCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            shard_paths=shard_paths,
            meta_path=meta_path,
        )
        # Without feature_registry_path, falls back to max index + 1
        assert reader.total_genes >= 1


# ---------------------------------------------------------------------------
# Synthetic Zarr fixture
# ---------------------------------------------------------------------------


def _make_synthetic_zarr(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create synthetic Zarr cell-chunked sparse storage for smoke testing."""
    zarr = pytest.importorskip("zarr")
    import json

    n_cells = 20
    n_genes = 50
    chunk_cells = 10

    indices_path = tmp_path / "syn-indices.zarr"
    counts_path = tmp_path / "syn-counts.zarr"
    sf_path = tmp_path / "syn-sf.zarr"
    meta_path = tmp_path / "syn-meta.json"

    n_chunks = (n_cells + chunk_cells - 1) // chunk_cells
    max_nonzero = n_genes

    indices_store = zarr.open(str(indices_path), mode="w")
    counts_store = zarr.open(str(counts_path), mode="w")
    sf_store = zarr.open(str(sf_path), mode="w")

    sf_store.create_dataset("sf", data=np.ones(n_cells), shape=(n_cells,), dtype="f8")

    indices_arr = np.full((n_chunks, chunk_cells, max_nonzero), -1, dtype=np.int32)
    counts_arr = np.zeros((n_chunks, chunk_cells, max_nonzero), dtype=np.int32)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_cells
        end = min(start + chunk_cells, n_cells)
        for local_i, global_i in enumerate(range(start, end)):
            rng = np.random.default_rng(global_i)
            n_expressed = rng.integers(3, 8)
            gene_indices = sorted(rng.choice(n_genes, size=n_expressed, replace=False))
            gene_counts = rng.integers(1, 10, size=n_expressed)
            indices_arr[chunk_idx, local_i, :n_expressed] = gene_indices
            counts_arr[chunk_idx, local_i, :n_expressed] = gene_counts

        indices_store.create_dataset(
            f"chunk_{chunk_idx}",
            data=indices_arr[chunk_idx, : end - start],
            shape=(end - start, max_nonzero),
            dtype="i4",
        )
        counts_store.create_dataset(
            f"chunk_{chunk_idx}",
            data=counts_arr[chunk_idx, : end - start],
            shape=(end - start, max_nonzero),
            dtype="i4",
        )

    with open(meta_path, "w") as f:
        json.dump(
            {
                "release_id": "syn-v0",
                "n_obs": n_cells,
                "n_vars": n_genes,
                "chunk_cells": chunk_cells,
                "n_chunks": n_chunks,
            },
            f,
        )

    return indices_path, counts_path, sf_path, meta_path


class TestZarrCellReader:
    def test_read_cell(self, tmp_path: Path):
        indices_path, counts_path, sf_path, meta_path = _make_synthetic_zarr(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"

        reader = ZarrCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            indices_zarr_path=indices_path,
            counts_zarr_path=counts_path,
            sf_zarr_path=sf_path,
            meta_path=meta_path,
            chunk_cells=10,
        )

        assert len(reader) == 20
        cell = reader.read_cell(0)
        assert isinstance(cell, CellState)
        assert cell.dataset_release == "syn-v0"
        assert isinstance(cell.expressed_gene_indices, tuple)
        assert isinstance(cell.expression_counts, tuple)
        assert cell.size_factor > 0

    def test_read_cells_batch(self, tmp_path: Path):
        indices_path, counts_path, sf_path, meta_path = _make_synthetic_zarr(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"

        reader = ZarrCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            indices_zarr_path=indices_path,
            counts_zarr_path=counts_path,
            sf_zarr_path=sf_path,
            meta_path=meta_path,
            chunk_cells=10,
        )

        cells = reader.read_cells([0, 1, 2])
        assert len(cells) == 3
        assert all(isinstance(c, CellState) for c in cells)

    def test_total_genes_from_meta(self, tmp_path: Path):
        indices_path, counts_path, sf_path, meta_path = _make_synthetic_zarr(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"

        reader = ZarrCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            indices_zarr_path=indices_path,
            counts_zarr_path=counts_path,
            sf_zarr_path=sf_path,
            meta_path=meta_path,
            chunk_cells=10,
        )
        # synthetic fixture uses n_genes=50
        assert reader.total_genes == 50


# ---------------------------------------------------------------------------
# Sampler parity across all three backends
# ---------------------------------------------------------------------------


def _make_synthetic_feature_registry(tmp_path: Path) -> Path:
    """Create a shared feature registry for parity tests."""
    from perturb_data_lab.materializers.models import (
        FeatureRegistryEntry,
        FeatureRegistryManifest,
    )
    from perturb_data_lab.contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL

    reg = FeatureRegistryManifest(
        kind="feature-registry",
        contract_version=CONTRACT_VERSION,
        registry_id="syn-reg",
        append_only=True,
        namespace="test",
        feature_id_field="gene_id",
        feature_label_field="gene_symbol",
        default_missing_value=MISSING_VALUE_LITERAL,
        entries=tuple(
            FeatureRegistryEntry(
                token_id=j,
                feature_id=f"gene_{j}",
                feature_label=f"Gene {j}",
                namespace="test",
            )
            for j in range(50)
        ),
    )
    reg_path = tmp_path / "feature-registry.yaml"
    reg.write_yaml(reg_path)
    return reg_path


class TestSamplerParityAcrossBackends:
    """Verify all three backends yield the same sampler behavior."""

    @pytest.fixture
    def arrow_reader(self, tmp_path: Path) -> ArrowHFCellReader:
        cells_path, meta_path, sqlite_path = _make_synthetic_arrow_parquet(tmp_path)
        feature_reg = _make_synthetic_feature_registry(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"

        return ArrowHFCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            cells_parquet_path=cells_path,
            meta_parquet_path=meta_path,
            cell_meta_sqlite_path=sqlite_path,
            feature_registry_path=feature_reg,
        )

    @pytest.fixture
    def webdataset_reader(self, tmp_path: Path) -> "WebDatasetCellReader":
        shard_dir, meta_path = _make_synthetic_webdataset_tar(tmp_path)
        shard_paths = sorted(tmp_path.glob("syn-shard-*.tar"))
        feature_reg = _make_synthetic_feature_registry(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"

        return WebDatasetCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            shard_paths=shard_paths,
            meta_path=meta_path,
            feature_registry_path=feature_reg,
        )

    @pytest.fixture
    def zarr_reader(self, tmp_path: Path) -> "ZarrCellReader":
        indices_path, counts_path, sf_path, meta_path = _make_synthetic_zarr(tmp_path)
        feature_reg = _make_synthetic_feature_registry(tmp_path)
        corpus_index = tmp_path / "corpus-index.yaml"

        return ZarrCellReader(
            release_id="syn-v0",
            corpus_index_path=corpus_index,
            indices_zarr_path=indices_path,
            counts_zarr_path=counts_path,
            sf_zarr_path=sf_path,
            meta_path=meta_path,
            chunk_cells=10,
            feature_registry_path=feature_reg,
        )

    def test_total_genes_parity(
        self,
        arrow_reader,
        webdataset_reader,
        zarr_reader,
    ):
        """All backends report the same total_genes via shared feature registry."""
        assert arrow_reader.total_genes == webdataset_reader.total_genes
        assert webdataset_reader.total_genes == zarr_reader.total_genes
        assert arrow_reader.total_genes == 50

    def test_random_context_sampler_parity(
        self,
        arrow_reader,
        webdataset_reader,
        zarr_reader,
    ):
        """RandomContextSampler produces same-size context across all backends."""
        for reader in [arrow_reader, webdataset_reader, zarr_reader]:
            state = SamplerState(
                mode="random_context",
                total_cells=len(reader),
                n_genes=reader.total_genes,
            )
            rng = np.random.default_rng(42)
            sampler = RandomContextSampler(state, rng)
            cells = reader.read_cells([0, 1, 2])
            for cell in cells:
                context = sampler.sample_indices(cell, context_size=10)
                assert len(context) == 10
                assert context.max() < reader.total_genes

    def test_expressed_zeros_sampler_parity(
        self,
        arrow_reader,
        webdataset_reader,
        zarr_reader,
    ):
        """ExpressedZerosSampler includes expressed genes across all backends."""
        for reader in [arrow_reader, webdataset_reader, zarr_reader]:
            state = SamplerState(
                mode="expressed_zeros",
                total_cells=len(reader),
                n_genes=reader.total_genes,
            )
            rng = np.random.default_rng(42)
            sampler = ExpressedZerosSampler(state, rng)
            cells = reader.read_cells([0, 1, 2])
            for cell in cells:
                context = sampler.sample_indices(cell)
                expressed = set(cell.expressed_gene_indices)
                # All expressed genes must be in context
                assert expressed.issubset(set(context))
                # Context must be unique
                assert len(set(context)) == len(context)

    def test_iterable_dataset_parity(
        self,
        arrow_reader,
        webdataset_reader,
        zarr_reader,
    ):
        """PerturbIterableDataset yields the same keys across all backends."""
        for reader in [arrow_reader, webdataset_reader, zarr_reader]:
            dataset = PerturbIterableDataset(
                reader,
                sampler_mode="random_context",
                shuffle=False,
                seed=42,
                context_size=10,
            )
            items = list(dataset)
            assert len(items) == len(reader)
            item = items[0]
            assert "cell_id" in item
            assert "expressed_gene_indices" in item
            assert "expression_counts" in item
            assert "context_indices" in item
            assert "size_factor" in item
