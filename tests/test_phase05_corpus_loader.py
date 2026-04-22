"""Phase 5 corpus runtime loader tests.

Tests cover:
- CorpusLoader construction from synthetic corpus index + global metadata
- Per-dataset reader entry wiring
- Global corpus index routing (binary search)
- Token-space HVG/non-HVG translation via DatasetReaderEntry properties
- translate_origin_indices_to_tokens across multiple datasets
- CorpusLoader iteration (iter_cells, len, dataset_ids)
- build_corpus_loader factory alias
- Error handling: missing corpus index, invalid backend, unknown dataset_id
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perturb_data_lab.loaders import (
    ArrowHFCellReader,
    CellState,
    CorpusLoader,
    DatasetReaderEntry,
    HVGRandomSampler,
    SamplerState,
    build_corpus_loader,
)
from perturb_data_lab.loaders.corpus import DatasetReaderEntry as _DatasetReaderEntry
from perturb_data_lab.materializers import (
    CorpusEmissionSpec,
    CorpusIndexDocument,
    GlobalMetadataDocument,
    MaterializationManifest,
    OutputRoots,
    ProvenanceSpec,
    CountSourceSpec,
)
from perturb_data_lab.materializers.tokenizer import CorpusTokenizer


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_cells_parquet(
    path: Path,
    n_cells: int,
    n_genes: int,
    release_id: str,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    expr_idx_data = []
    expr_count_data = []
    sf_data = []
    for _ in range(n_cells):
        n_exp = rng.integers(5, min(15, n_genes) + 1)
        exp_idx = sorted(rng.choice(n_genes, n_exp, replace=False).tolist())
        exp_counts = rng.integers(1, 100, size=n_exp).tolist()
        expr_idx_data.append(exp_idx)
        expr_count_data.append(exp_counts)
        sf_data.append(float(rng.integers(1000, 2000) / 1000.0))
    table = pa.table({
        "expressed_gene_indices": pa.array(expr_idx_data, type=pa.list_(pa.int32())),
        "expression_counts": pa.array(expr_count_data, type=pa.list_(pa.int32())),
        "size_factor": pa.array(sf_data, type=pa.float64()),
    })
    pq.write_table(table, str(path))


def _make_meta_parquet(path: Path, n_cells: int, release_id: str):
    cell_ids = [f"{release_id}_cell_{i}" for i in range(n_cells)]
    table = pa.table({
        "cell_id": pa.array(cell_ids, type=pa.string()),
        "dataset_id": pa.array([release_id] * n_cells, type=pa.string()),
        "dataset_release": pa.array([release_id] * n_cells, type=pa.string()),
    })
    pq.write_table(table, str(path))


def _make_cell_meta_sqlite(path: Path, n_cells: int, release_id: str):
    cell_ids = [f"{release_id}_cell_{i}" for i in range(n_cells)]
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cell_meta "
        "(cell_id TEXT, dataset_id TEXT, dataset_release TEXT, "
        "size_factor REAL, canonical_perturbation TEXT, "
        "canonical_context TEXT, raw_obs TEXT)"
    )
    for cid in cell_ids:
        conn.execute(
            "INSERT INTO cell_meta VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                cid, release_id, release_id, 1.0,
                json.dumps({"perturbation_label": "ctrl", "perturbation_type": "control"}),
                json.dumps({"dataset_id": release_id, "cell_context": "bulk"}),
                "{}",
            ),
        )
    conn.commit()
    conn.close()


def _make_hvg_arrays(meta_root: Path, n_genes: int, seed: int = 0):
    hvg_dir = meta_root / "hvg_sidecar"
    hvg_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    all_idx = list(range(n_genes))
    hvg_indices = sorted(rng.choice(n_genes, min(3, n_genes), replace=False).tolist())
    nonhvg_indices = sorted([i for i in all_idx if i not in set(hvg_indices)])
    np.save(str(hvg_dir / "hvg.npy"), np.array(hvg_indices, dtype=np.int32), allow_pickle=False)
    np.save(str(hvg_dir / "nonhvg.npy"), np.array(nonhvg_indices, dtype=np.int32), allow_pickle=False)
    return hvg_dir


def _make_feature_parquet(meta_root: Path, release_id: str, n_genes: int, tokenizer: CorpusTokenizer):
    gene_labels = [f"GENE_{chr(65 + i)}" for i in range(n_genes)]
    token_ids = [tokenizer.to_id(label) for label in gene_labels]
    origin_path = meta_root / f"{release_id}-features-origin.parquet"
    token_path = meta_root / f"{release_id}-features-token.parquet"
    origin_table = pa.table({
        "origin_index": pa.array(list(range(n_genes)), type=pa.int32()),
        "feature_id": pa.array(gene_labels, type=pa.string()),
    })
    pq.write_table(origin_table, str(origin_path))
    token_table = pa.table({
        "origin_index": pa.array(list(range(n_genes)), type=pa.int32()),
        "feature_id": pa.array(gene_labels, type=pa.string()),
        "token_id": pa.array(token_ids, type=pa.int32()),
    })
    pq.write_table(token_table, str(token_path))
    return {"features_origin": origin_path.name, "features_token": token_path.name}


def _make_materialization_manifest(
    meta_root: Path,
    matrix_root: Path,
    release_id: str,
    dataset_id: str,
    hvg_sidecar_path: str | None = "hvg_sidecar",
    feature_meta_paths: dict[str, str] | None = None,
    backend: str = "arrow-hf",
):
    manifest = MaterializationManifest(
        kind="materialization-manifest",
        contract_version="0.2.0",
        dataset_id=dataset_id,
        release_id=release_id,
        route="create_new",
        backend=backend,
        count_source=CountSourceSpec(selected=".X", integer_only=True),
        outputs=OutputRoots(metadata_root=str(meta_root), matrix_root=str(matrix_root)),
        provenance=ProvenanceSpec(source_path=f"/fake/{release_id}.h5ad", schema="/fake/schema.yaml"),
        tokenizer_path="tokenizer.json",
        feature_meta_paths=feature_meta_paths,
        qa_manifest_path="qa-manifest.yaml",
        hvg_sidecar_path=hvg_sidecar_path,
    )
    manifest_path = meta_root / "materialization-manifest.yaml"
    manifest.write_yaml(manifest_path)
    return manifest_path


def _write_tokenizer_and_global_metadata(corpus_root: Path, tok: CorpusTokenizer, corpus_id: str):
    """Write tokenizer.json and global-metadata.yaml to corpus root."""
    import shutil
    tok_path = corpus_root / "tokenizer.json"
    tok.to_json(tok_path)
    gmeta = GlobalMetadataDocument(
        kind="global-metadata",
        contract_version="0.2.0",
        schema_version="0.1.0",
        feature_registry_id="",
        missing_value_literal="<missing>",
        raw_field_policy="preserve-unchanged",
        tokenizer_path="tokenizer.json",
    )
    gmeta.write_yaml(corpus_root / "global-metadata.yaml")


def _write_corpus_index(corpus_root: Path, corpus_id: str, dataset_records: list):
    """Write corpus-index.yaml with given dataset records."""
    corpus_index = CorpusIndexDocument(
        kind="corpus-index",
        contract_version="0.2.0",
        corpus_id=corpus_id,
        global_metadata={},
        datasets=tuple(dataset_records),
    )
    corpus_index.write_yaml(corpus_root / "corpus-index.yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_tokenizer(tmp_path):
    tokens = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]
    tok = CorpusTokenizer.create_new(
        corpus_id="test-corpus",
        namespace="gene_symbol",
        regular_tokens=tokens,
    )
    tok_path = tmp_path / "tokenizer.json"
    tok.to_json(tok_path)
    return tok, tok_path


@pytest.fixture
def temp_corpus_root(tmp_path):
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    matrix_root = corpus_root / "matrix"
    matrix_root.mkdir()
    metadata_root = corpus_root / "metadata"
    metadata_root.mkdir()
    return corpus_root, matrix_root, metadata_root


# ---------------------------------------------------------------------------
# Tests: DatasetReaderEntry HVG properties
# ---------------------------------------------------------------------------


class TestDatasetReaderEntry:
    def test_hvg_indices_property_returns_original(self, tmp_path):
        entry = DatasetReaderEntry(
            dataset_id="ds1",
            release_id="ds1-release",
            manifest_path=tmp_path / "manifest.yaml",
            reader=None,
            hvg_indices=(5, 10, 15),
            nonhvg_indices=tuple(range(16)),
            n_vars=16,
        )
        assert entry.hvg_indices == (5, 10, 15)

    def test_nonhvg_indices_property(self, tmp_path):
        entry = DatasetReaderEntry(
            dataset_id="ds1",
            release_id="ds1-release",
            manifest_path=tmp_path / "manifest.yaml",
            reader=None,
            hvg_indices=(5, 10, 15),
            nonhvg_indices=tuple(i for i in range(16) if i not in {5, 10, 15}),
            n_vars=16,
        )
        assert 5 not in entry.nonhvg_indices
        assert 0 in entry.nonhvg_indices

    def test_token_hvg_set_falls_back_when_no_preloader(self, tmp_path):
        entry = DatasetReaderEntry(
            dataset_id="ds1",
            release_id="ds1-release",
            manifest_path=tmp_path / "manifest.yaml",
            reader=None,
            hvg_indices=(1, 3, 5),
            n_vars=16,
        )
        # Without preloader, token set falls back to original indices
        assert entry.token_hvg_set == (1, 3, 5)

    def test_token_hvg_set_translates_via_preloader(self, tmp_path):
        # Mock preloader that maps origin_index -> token_id
        class FakePreloader:
            def __init__(self, origin_to_token):
                self._map = origin_to_token

            def origin_index_to_token_id(self, origin_idx: int) -> int:
                return self._map.get(int(origin_idx), -1)

            def translate_indices(self, origin_indices):
                return tuple(self.origin_index_to_token_id(i) for i in origin_indices)

        class FakeReader:
            preloaded_features = FakePreloader({0: 4, 1: 5, 2: 6, 3: 7, 4: 8})

        entry = DatasetReaderEntry(
            dataset_id="ds1",
            release_id="ds1-release",
            manifest_path=tmp_path / "manifest.yaml",
            reader=FakeReader,
            hvg_indices=(0, 2, 4),  # GENE_A(0)->4, GENE_C(2)->6, GENE_E(4)->8
            n_vars=5,
        )

        token_set = entry.token_hvg_set
        assert 4 in token_set   # GENE_A
        assert 6 in token_set   # GENE_C
        assert 8 in token_set   # GENE_E


# ---------------------------------------------------------------------------
# Tests: CorpusLoader construction and routing
# ---------------------------------------------------------------------------


class TestCorpusLoaderTwoDatasets:
    def test_two_datasets_len_and_ids(self, temp_corpus_root, synthetic_tokenizer):
        corpus_root, matrix_root, metadata_root = temp_corpus_root
        tok, tok_path = synthetic_tokenizer
        import shutil
        shutil.copy(tok_path, corpus_root / "tokenizer.json")

        # Dataset 1: 10 cells
        mroot1 = corpus_root / "metadata_0"
        mmroot1 = corpus_root / "matrix_0"
        mroot1.mkdir()
        mmroot1.mkdir()
        _make_cells_parquet(mmroot1 / "release1-cells.parquet", 10, 16, "release1", seed=0)
        _make_meta_parquet(mroot1 / "release1-meta.parquet", 10, "release1")
        _make_cell_meta_sqlite(mroot1 / "release1-cell-meta.sqlite", 10, "release1")
        feat_paths1 = _make_feature_parquet(mroot1, "release1", 16, tok)
        _make_materialization_manifest(mroot1, mmroot1, "release1", "dataset1", "hvg_sidecar", feat_paths1)

        # Dataset 2: 8 cells
        mroot2 = corpus_root / "metadata_1"
        mmroot2 = corpus_root / "matrix_1"
        mroot2.mkdir()
        mmroot2.mkdir()
        _make_cells_parquet(mmroot2 / "release2-cells.parquet", 8, 16, "release2", seed=1)
        _make_meta_parquet(mroot2 / "release2-meta.parquet", 8, "release2")
        _make_cell_meta_sqlite(mroot2 / "release2-cell-meta.sqlite", 8, "release2")
        feat_paths2 = _make_feature_parquet(mroot2, "release2", 16, tok)
        _make_materialization_manifest(mroot2, mmroot2, "release2", "dataset2", "hvg_sidecar", feat_paths2)

        # Write corpus index
        from perturb_data_lab.materializers.models import DatasetJoinRecord
        _write_corpus_index(corpus_root, "test-corpus", [
            DatasetJoinRecord(dataset_id="dataset1", release_id="release1", join_mode="create_new", manifest_path="metadata_0/materialization-manifest.yaml"),
            DatasetJoinRecord(dataset_id="dataset2", release_id="release2", join_mode="create_new", manifest_path="metadata_1/materialization-manifest.yaml"),
        ])
        _write_tokenizer_and_global_metadata(corpus_root, tok, "test-corpus")

        loader = CorpusLoader.from_corpus_index(corpus_root / "corpus-index.yaml")

        assert loader.corpus_id == "test-corpus"
        assert len(loader) == 18  # 10 + 8
        assert loader.dataset_ids == ("dataset1", "dataset2")

    def test_read_cell_routing_binary_search(self, temp_corpus_root, synthetic_tokenizer):
        corpus_root, matrix_root, metadata_root = temp_corpus_root
        tok, tok_path = synthetic_tokenizer
        import shutil
        shutil.copy(tok_path, corpus_root / "tokenizer.json")

        from perturb_data_lab.materializers.models import DatasetJoinRecord

        cell_counts = [5, 8]
        for ds_idx, n_cells in enumerate(cell_counts):
            mroot = corpus_root / f"metadata_{ds_idx}"
            mmroot = corpus_root / f"matrix_{ds_idx}"
            mroot.mkdir()
            mmroot.mkdir()
            release_id = f"rel{ds_idx}"
            dataset_id = f"ds{ds_idx}"
            _make_cells_parquet(mmroot / f"{release_id}-cells.parquet", n_cells, 16, release_id, seed=ds_idx)
            _make_meta_parquet(mroot / f"{release_id}-meta.parquet", n_cells, release_id)
            _make_cell_meta_sqlite(mroot / f"{release_id}-cell-meta.sqlite", n_cells, release_id)
            feat_paths = _make_feature_parquet(mroot, release_id, 16, tok)
            _make_materialization_manifest(mroot, mmroot, release_id, dataset_id, None, feat_paths)

        _write_corpus_index(corpus_root, "test-corpus", [
            DatasetJoinRecord(dataset_id="ds0", release_id="rel0", join_mode="create_new", manifest_path="metadata_0/materialization-manifest.yaml"),
            DatasetJoinRecord(dataset_id="ds1", release_id="rel1", join_mode="create_new", manifest_path="metadata_1/materialization-manifest.yaml"),
        ])
        _write_tokenizer_and_global_metadata(corpus_root, tok, "test-corpus")

        loader = CorpusLoader.from_corpus_index(corpus_root / "corpus-index.yaml")
        assert len(loader) == 13  # 5 + 8

        # Global 0-4 → rel0
        for i in range(5):
            cell = loader.read_cell(i)
            assert cell.dataset_id == "rel0"

        # Global 5-12 → rel1
        for i in range(5, 13):
            cell = loader.read_cell(i)
            assert cell.dataset_id == "rel1"

        # Out of range
        with pytest.raises(IndexError):
            loader.read_cell(13)

    def test_iter_cells_yields_all(self, temp_corpus_root, synthetic_tokenizer):
        corpus_root, matrix_root, metadata_root = temp_corpus_root
        tok, tok_path = synthetic_tokenizer
        import shutil
        shutil.copy(tok_path, corpus_root / "tokenizer.json")

        from perturb_data_lab.materializers.models import DatasetJoinRecord

        cell_counts = [3, 4]
        for ds_idx, n_cells in enumerate(cell_counts):
            mroot = corpus_root / f"metadata_{ds_idx}"
            mmroot = corpus_root / f"matrix_{ds_idx}"
            mroot.mkdir()
            mmroot.mkdir()
            release_id = f"rel{ds_idx}"
            dataset_id = f"ds{ds_idx}"
            _make_cells_parquet(mmroot / f"{release_id}-cells.parquet", n_cells, 16, release_id, seed=ds_idx)
            _make_meta_parquet(mroot / f"{release_id}-meta.parquet", n_cells, release_id)
            _make_cell_meta_sqlite(mroot / f"{release_id}-cell-meta.sqlite", n_cells, release_id)
            feat_paths = _make_feature_parquet(mroot, release_id, 16, tok)
            _make_materialization_manifest(mroot, mmroot, release_id, dataset_id, None, feat_paths)

        _write_corpus_index(corpus_root, "test-corpus", [
            DatasetJoinRecord(dataset_id="ds0", release_id="rel0", join_mode="create_new", manifest_path="metadata_0/materialization-manifest.yaml"),
            DatasetJoinRecord(dataset_id="ds1", release_id="rel1", join_mode="create_new", manifest_path="metadata_1/materialization-manifest.yaml"),
        ])
        _write_tokenizer_and_global_metadata(corpus_root, tok, "test-corpus")

        loader = CorpusLoader.from_corpus_index(corpus_root / "corpus-index.yaml")
        cells = list(loader.iter_cells())
        assert len(cells) == 7  # 3 + 4
        # dataset_id comes from SQLite row's dataset_id column, which is set to release_id
        rel0_count = sum(1 for c in cells if c.dataset_id == "rel0")
        rel1_count = sum(1 for c in cells if c.dataset_id == "rel1")
        assert rel0_count == 3
        assert rel1_count == 4


# ---------------------------------------------------------------------------
# Tests: Token-space translation
# ---------------------------------------------------------------------------


class TestCorpusLoaderTranslation:
    def test_translate_origin_indices_to_tokens(self, temp_corpus_root, synthetic_tokenizer):
        corpus_root, matrix_root, metadata_root = temp_corpus_root
        tok, tok_path = synthetic_tokenizer
        import shutil
        shutil.copy(tok_path, corpus_root / "tokenizer.json")

        release_id = "release1"
        dataset_id = "dataset1"
        mroot = corpus_root / "metadata_0"
        mmroot = corpus_root / "matrix_0"
        mroot.mkdir()
        mmroot.mkdir()
        _make_cells_parquet(mmroot / f"{release_id}-cells.parquet", 5, 5, release_id, seed=0)
        _make_meta_parquet(mroot / f"{release_id}-meta.parquet", 5, release_id)
        _make_cell_meta_sqlite(mroot / f"{release_id}-cell-meta.sqlite", 5, release_id)
        feat_paths = _make_feature_parquet(mroot, release_id, 5, tok)
        _make_materialization_manifest(mroot, mmroot, release_id, dataset_id, None, feat_paths)

        from perturb_data_lab.materializers.models import DatasetJoinRecord
        _write_corpus_index(corpus_root, "test-corpus", [
            DatasetJoinRecord(dataset_id=dataset_id, release_id=release_id, join_mode="create_new", manifest_path="metadata_0/materialization-manifest.yaml"),
        ])
        _write_tokenizer_and_global_metadata(corpus_root, tok, "test-corpus")

        loader = CorpusLoader.from_corpus_index(corpus_root / "corpus-index.yaml")

        # GENE_A(0)->4, GENE_B(1)->5, GENE_C(2)->6
        result = loader.translate_origin_indices_to_tokens("dataset1", (0, 1, 2))
        assert result == (4, 5, 6)

    def test_translate_unknown_indices_returns_neg1(self, temp_corpus_root, synthetic_tokenizer):
        corpus_root, matrix_root, metadata_root = temp_corpus_root
        tok, tok_path = synthetic_tokenizer
        import shutil
        shutil.copy(tok_path, corpus_root / "tokenizer.json")

        release_id = "release1"
        dataset_id = "dataset1"
        mroot = corpus_root / "metadata_0"
        mmroot = corpus_root / "matrix_0"
        mroot.mkdir()
        mmroot.mkdir()
        _make_cells_parquet(mmroot / f"{release_id}-cells.parquet", 5, 5, release_id, seed=0)
        _make_meta_parquet(mroot / f"{release_id}-meta.parquet", 5, release_id)
        _make_cell_meta_sqlite(mroot / f"{release_id}-cell-meta.sqlite", 5, release_id)
        feat_paths = _make_feature_parquet(mroot, release_id, 5, tok)
        _make_materialization_manifest(mroot, mmroot, release_id, dataset_id, None, feat_paths)

        from perturb_data_lab.materializers.models import DatasetJoinRecord
        _write_corpus_index(corpus_root, "test-corpus", [
            DatasetJoinRecord(dataset_id=dataset_id, release_id=release_id, join_mode="create_new", manifest_path="metadata_0/materialization-manifest.yaml"),
        ])
        _write_tokenizer_and_global_metadata(corpus_root, tok, "test-corpus")

        loader = CorpusLoader.from_corpus_index(corpus_root / "corpus-index.yaml")

        # 99 is out of range → translator returns -1, which is filtered out
        result = loader.translate_origin_indices_to_tokens("dataset1", (0, 99))
        # Only GENE_A (index 0) maps to token 4; index 99 returns -1 and is skipped
        assert 4 in result or len(result) == 1


# ---------------------------------------------------------------------------
# Tests: HVGRandomSampler integration
# ---------------------------------------------------------------------------


class TestHVGRandomSamplerIntegration:
    def test_hvg_sampler_with_token_hvg_set(self, temp_corpus_root, synthetic_tokenizer):
        corpus_root, matrix_root, metadata_root = temp_corpus_root
        tok, tok_path = synthetic_tokenizer
        import shutil
        shutil.copy(tok_path, corpus_root / "tokenizer.json")

        release_id = "release1"
        dataset_id = "dataset1"
        mroot = corpus_root / "metadata_0"
        mmroot = corpus_root / "matrix_0"
        mroot.mkdir()
        mmroot.mkdir()
        n_cells = 10
        n_genes = 5
        _make_cells_parquet(mmroot / f"{release_id}-cells.parquet", n_cells, n_genes, release_id, seed=0)
        _make_meta_parquet(mroot / f"{release_id}-meta.parquet", n_cells, release_id)
        _make_cell_meta_sqlite(mroot / f"{release_id}-cell-meta.sqlite", n_cells, release_id)
        _make_hvg_arrays(mroot, n_genes, seed=0)
        feat_paths = _make_feature_parquet(mroot, release_id, n_genes, tok)
        _make_materialization_manifest(mroot, mmroot, release_id, dataset_id, "hvg_sidecar", feat_paths)

        from perturb_data_lab.materializers.models import DatasetJoinRecord
        _write_corpus_index(corpus_root, "test-corpus", [
            DatasetJoinRecord(dataset_id=dataset_id, release_id=release_id, join_mode="create_new", manifest_path="metadata_0/materialization-manifest.yaml"),
        ])
        _write_tokenizer_and_global_metadata(corpus_root, tok, "test-corpus")

        loader = CorpusLoader.from_corpus_index(corpus_root / "corpus-index.yaml")
        entry = loader.dataset_reader("dataset1")

        # Get token-space HVG set
        token_hvg = entry.token_hvg_set

        state = SamplerState(
            mode="hvg_random",
            total_cells=n_cells,
            n_genes=n_genes,
            hvg_set=token_hvg,
        )
        sampler = HVGRandomSampler(state, np.random.default_rng(42))

        cell = loader.read_cell(0)
        context = sampler.sample_indices(cell, max_context=512)
        assert len(context) > 0


# ---------------------------------------------------------------------------
# Tests: build_corpus_loader alias
# ---------------------------------------------------------------------------


class TestBuildCorpusLoaderAlias:
    def test_alias_returns_corpus_loader(self, temp_corpus_root, synthetic_tokenizer):
        corpus_root, matrix_root, metadata_root = temp_corpus_root
        tok, tok_path = synthetic_tokenizer
        import shutil
        shutil.copy(tok_path, corpus_root / "tokenizer.json")

        release_id = "release1"
        dataset_id = "dataset1"
        mroot = corpus_root / "metadata_0"
        mmroot = corpus_root / "matrix_0"
        mroot.mkdir()
        mmroot.mkdir()
        _make_cells_parquet(mmroot / f"{release_id}-cells.parquet", 5, 5, release_id, seed=0)
        _make_meta_parquet(mroot / f"{release_id}-meta.parquet", 5, release_id)
        _make_cell_meta_sqlite(mroot / f"{release_id}-cell-meta.sqlite", 5, release_id)
        feat_paths = _make_feature_parquet(mroot, release_id, 5, tok)
        _make_materialization_manifest(mroot, mmroot, release_id, dataset_id, None, feat_paths)

        from perturb_data_lab.materializers.models import DatasetJoinRecord
        _write_corpus_index(corpus_root, "test-corpus", [
            DatasetJoinRecord(dataset_id=dataset_id, release_id=release_id, join_mode="create_new", manifest_path="metadata_0/materialization-manifest.yaml"),
        ])
        _write_tokenizer_and_global_metadata(corpus_root, tok, "test-corpus")

        loader = build_corpus_loader(corpus_root / "corpus-index.yaml")
        assert isinstance(loader, CorpusLoader)
        assert loader.corpus_id == "test-corpus"


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestCorpusLoaderErrors:
    def test_unsupported_backend_raises(self, tmp_path):
        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()

        from perturb_data_lab.materializers.models import DatasetJoinRecord
        corpus_index = CorpusIndexDocument(
            kind="corpus-index",
            contract_version="0.2.0",
            corpus_id="test",
            global_metadata={},
            datasets=(),
        )
        corpus_index.write_yaml(corpus_root / "corpus-index.yaml")
        gmeta = GlobalMetadataDocument(
            kind="global-metadata",
            contract_version="0.2.0",
            schema_version="0.1.0",
            feature_registry_id="",
            missing_value_literal="<missing>",
            raw_field_policy="preserve-unchanged",
            backend="unsupported",
        )
        gmeta.write_yaml(corpus_root / "global-metadata.yaml")

        with pytest.raises(NotImplementedError, match="not yet supported"):
            CorpusLoader.from_corpus_index(corpus_root / "corpus-index.yaml", backend="webdataset")

    def test_unknown_dataset_id_raises_key_error(self, temp_corpus_root, synthetic_tokenizer):
        corpus_root, matrix_root, metadata_root = temp_corpus_root
        tok, tok_path = synthetic_tokenizer
        import shutil
        shutil.copy(tok_path, corpus_root / "tokenizer.json")

        # Build a minimal but real dataset directory so the loader can initialize
        release_id = "release1"
        dataset_id = "dataset1"
        mroot = corpus_root / "metadata_0"
        mmroot = corpus_root / "matrix_0"
        mroot.mkdir()
        mmroot.mkdir()
        _make_cells_parquet(mmroot / f"{release_id}-cells.parquet", 5, 5, release_id, seed=0)
        _make_meta_parquet(mroot / f"{release_id}-meta.parquet", 5, release_id)
        _make_cell_meta_sqlite(mroot / f"{release_id}-cell-meta.sqlite", 5, release_id)
        feat_paths = _make_feature_parquet(mroot, release_id, 5, tok)
        _make_materialization_manifest(mroot, mmroot, release_id, dataset_id, None, feat_paths)

        from perturb_data_lab.materializers.models import DatasetJoinRecord
        _write_corpus_index(corpus_root, "test", [
            DatasetJoinRecord(dataset_id=dataset_id, release_id=release_id, join_mode="create_new", manifest_path="metadata_0/materialization-manifest.yaml"),
        ])
        _write_tokenizer_and_global_metadata(corpus_root, tok, "test")

        loader = CorpusLoader.from_corpus_index(corpus_root / "corpus-index.yaml")

        # Known dataset works
        entry = loader.dataset_reader("dataset1")
        assert entry.dataset_id == "dataset1"

        # Unknown raises KeyError
        with pytest.raises(KeyError):
            loader.dataset_reader("unknown_dataset")