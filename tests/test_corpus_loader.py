"""Unit tests for ``load_corpus()`` factory with mock data (no h5ad needed)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import yaml

from perturb_data_lab.loaders.corpus_loader import Corpus, load_corpus


# ---------------------------------------------------------------------------
# Helpers — build mock corpus on disk
# ---------------------------------------------------------------------------

N_GENES = 100  # per dataset


def _make_obs_df(
    dataset_id: str,
    dataset_index: int,
    n_cells: int,
    global_start: int,
    *,
    canonical: bool = True,
) -> pl.DataFrame:
    """Create a canonical-obs DataFrame for testing."""
    rows = []
    for i in range(n_cells):
        row: dict[str, Any] = {
            "cell_id": f"{dataset_id}_cell_{i:04d}",
            "dataset_id": dataset_id,
            "dataset_index": str(dataset_index),
            "global_row_index": str(global_start + i),
            "local_row_index": str(i),
            "size_factor": str(round(np.random.uniform(0.5, 2.0), 4)),
            "perturb_label": "CRISPR_control" if i % 3 == 0 else "CRISPR_geneX",
            "perturb_type": "CRISPR",
            "dose": "1.0",
            "dose_unit": "MOI",
            "timepoint": "7",
            "timepoint_unit": "days",
            "cell_context": "",
            "cell_line_or_type": "K562",
            "species": "Homo sapiens",
            "tissue": "bone marrow",
            "assay": "Perturb-seq",
            "condition": "NA",
            "batch_id": f"batch_{i // 5}",
            "donor_id": "donor_01",
            "sex": "NA",
            "disease_state": "healthy",
        }
        rows.append(row)

    return pl.DataFrame(rows)


def _make_var_df(dataset_id: str, n_genes: int) -> pl.DataFrame:
    """Create a canonical-var DataFrame for testing."""
    rows = []
    for i in range(n_genes):
        rows.append({
            "origin_index": str(i),
            "gene_id": f"ENSG_{dataset_id}_{i:05d}",
            "canonical_gene_id": f"GENE_{i:05d}",
            "global_id": str(i),
        })
    return pl.DataFrame(rows)


def _make_lance_rows(n_cells: int) -> list[dict[str, Any]]:
    """Create expression rows for Lance files."""
    rows = []
    rng = np.random.RandomState(42)
    for _ in range(n_cells):
        n_nonzero = rng.randint(20, min(90, N_GENES))
        gene_indices = rng.choice(N_GENES, size=n_nonzero, replace=False).astype(np.int32)
        gene_indices.sort()
        counts = rng.poisson(2, size=n_nonzero).astype(np.int32)
        rows.append({
            "expressed_gene_indices": list(gene_indices),
            "expression_counts": list(counts),
        })
    return rows


def _build_mock_aggregate_corpus(corpus_root: Path) -> None:
    """Build a minimal aggregate Lance corpus (2 datasets)."""
    corpus_root.mkdir(parents=True, exist_ok=True)

    # Dataset configs
    ds_configs = [
        {"dataset_id": "mock_00", "cell_count": 10, "global_start": 0, "global_end": 10, "dataset_index": 0},
        {"dataset_id": "mock_01", "cell_count": 15, "global_start": 10, "global_end": 25, "dataset_index": 1},
    ]

    # Write corpus-index.yaml
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {
            "backend": "lance",
            "topology": "aggregate",
        },
        "datasets": [
            {
                "dataset_id": ds["dataset_id"],
                "join_mode": "create_new" if i == 0 else "append_routed",
                "dataset_index": ds["dataset_index"],
                "cell_count": ds["cell_count"],
                "global_start": ds["global_start"],
                "global_end": ds["global_end"],
            }
            for i, ds in enumerate(ds_configs)
        ],
    }
    with open(corpus_root / "corpus-index.yaml", "w") as f:
        yaml.safe_dump(index_doc, f)

    # Write per-dataset canonical obs/var parquets (under meta/)
    total_cells = 0
    lance_rows: list[dict[str, Any]] = []
    for ds in ds_configs:
        ds_id = ds["dataset_id"]
        meta_dir = corpus_root / "meta" / ds_id / "canonical_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # canonical-obs.parquet
        obs_df = _make_obs_df(ds_id, ds["dataset_index"], ds["cell_count"], ds["global_start"])
        obs_df.write_parquet(str(meta_dir / "canonical-obs.parquet"))

        # canonical-var.parquet
        var_df = _make_var_df(ds_id, N_GENES)
        var_df.write_parquet(str(meta_dir / "canonical-var.parquet"))

        # Lance expression rows
        lance_rows.extend(_make_lance_rows(ds["cell_count"]))
        total_cells += ds["cell_count"]

    # Write aggregate Lance file
    matrix_dir = corpus_root / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("expressed_gene_indices", pa.list_(pa.int32())),
        ("expression_counts", pa.list_(pa.int32())),
    ])
    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [row["expressed_gene_indices"] for row in lance_rows],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [row["expression_counts"] for row in lance_rows],
                type=pa.list_(pa.int32()),
            ),
        },
        schema=schema,
    )
    lance.write_dataset(table, str(matrix_dir / "aggregated-cells.lance"), mode="overwrite")


def _build_mock_federated_corpus(corpus_root: Path) -> None:
    """Build a minimal federated Lance corpus (2 datasets)."""
    corpus_root.mkdir(parents=True, exist_ok=True)

    # Dataset configs
    ds_configs = [
        {"dataset_id": "mock_00", "cell_count": 10, "global_start": 0, "global_end": 10, "dataset_index": 0},
        {"dataset_id": "mock_01", "cell_count": 15, "global_start": 10, "global_end": 25, "dataset_index": 1},
    ]

    # Write corpus-index.yaml
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "global_metadata": {
            "backend": "lance",
            "topology": "federated",
        },
        "datasets": [
            {
                "dataset_id": ds["dataset_id"],
                "join_mode": "create_new" if i == 0 else "append_routed",
                "dataset_index": ds["dataset_index"],
                "cell_count": ds["cell_count"],
                "global_start": ds["global_start"],
                "global_end": ds["global_end"],
            }
            for i, ds in enumerate(ds_configs)
        ],
    }
    with open(corpus_root / "corpus-index.yaml", "w") as f:
        yaml.safe_dump(index_doc, f)

    # Write per-dataset canonical obs/var parquets and Lance files
    for ds in ds_configs:
        ds_id = ds["dataset_id"]
        ds_dir = corpus_root / ds_id
        meta_dir = ds_dir / "meta" / "canonical_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # canonical-obs.parquet
        obs_df = _make_obs_df(ds_id, ds["dataset_index"], ds["cell_count"], ds["global_start"])
        obs_df.write_parquet(str(meta_dir / "canonical-obs.parquet"))

        # canonical-var.parquet
        var_df = _make_var_df(ds_id, N_GENES)
        var_df.write_parquet(str(meta_dir / "canonical-var.parquet"))

        # Per-dataset Lance file
        matrix_dir = ds_dir / "matrix"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        lance_rows = _make_lance_rows(ds["cell_count"])
        schema = pa.schema([
            ("expressed_gene_indices", pa.list_(pa.int32())),
            ("expression_counts", pa.list_(pa.int32())),
        ])
        table = pa.table(
            {
                "expressed_gene_indices": pa.array(
                    [row["expressed_gene_indices"] for row in lance_rows],
                    type=pa.list_(pa.int32()),
                ),
                "expression_counts": pa.array(
                    [row["expression_counts"] for row in lance_rows],
                    type=pa.list_(pa.int32()),
                ),
            },
            schema=schema,
        )
        lance.write_dataset(table, str(matrix_dir / f"{ds_id}.lance"), mode="overwrite")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadCorpusAggregate:
    """Test ``load_corpus()`` with aggregate Lance topology."""

    def test_corpus_structure(self, tmp_path: Path) -> None:
        """``load_corpus()`` returns a fully populated ``Corpus`` object."""
        _build_mock_aggregate_corpus(tmp_path)

        corpus = load_corpus(str(tmp_path))

        assert isinstance(corpus, Corpus)
        assert corpus.topology == "aggregate"
        assert corpus.backend == "lance"
        assert corpus.corpus_root == tmp_path.resolve()
        assert corpus.batch_executor is not None
        assert corpus.feature_registry is not None
        assert corpus.metadata_index is not None
        assert len(corpus.dataset_entries) == 2

    def test_feature_registry_properties(self, tmp_path: Path) -> None:
        """Feature registry reflects mock gene vocabulary."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        fr = corpus.feature_registry
        assert fr.n_datasets == 2
        assert fr.global_vocab_size == N_GENES  # same gene pool
        assert fr.max_local_vocab == N_GENES
        assert fr.dataset_ids == ("mock_00", "mock_01")

    def test_metadata_index_properties(self, tmp_path: Path) -> None:
        """Metadata index has correct row count and columns."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        mi = corpus.metadata_index
        assert len(mi) == 25  # 10 + 15 cells
        assert "global_row_index" in mi.df.columns
        assert "cell_id" in mi.df.columns
        assert "dataset_id" in mi.df.columns
        assert "size_factor" in mi.df.columns
        assert "perturb_label" in mi.df.columns

    def test_global_row_indices_contiguous(self, tmp_path: Path) -> None:
        """Global row indices are 0..N-1 contiguous."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        gr = corpus.metadata_index.df["global_row_index"].to_numpy()
        assert gr[0] == 0
        assert gr[-1] == 24
        assert np.array_equal(gr, np.arange(25))

    def test_batch_executor_reads_cells(self, tmp_path: Path) -> None:
        """Batch executor can read expression + metadata."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = corpus.batch_executor.read_batch([0, 5, 12, 24])
        assert batch["batch_size"] == 4
        assert len(batch["global_row_index"]) == 4
        assert batch["row_offsets"][0] == 0
        # Every cell should have at least one expressed gene
        assert len(batch["expressed_gene_indices"]) > 0
        assert len(batch["expression_counts"]) > 0
        assert len(batch["dataset_id"]) == 4
        assert len(batch["canonical_perturbation"]) == 4

    def test_dataset_entries_cover_full_range(self, tmp_path: Path) -> None:
        """Dataset entries have correct global_start/end ranges."""
        _build_mock_aggregate_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        entries = sorted(corpus.dataset_entries, key=lambda e: e.global_start)
        assert entries[0].dataset_id == "mock_00"
        assert entries[0].global_start == 0
        assert entries[0].global_end == 10
        assert entries[1].dataset_id == "mock_01"
        assert entries[1].global_start == 10
        assert entries[1].global_end == 25


class TestLoadCorpusFederated:
    """Test ``load_corpus()`` with federated Lance topology."""

    def test_corpus_structure(self, tmp_path: Path) -> None:
        """``load_corpus()`` returns a fully populated ``Corpus`` object."""
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        assert isinstance(corpus, Corpus)
        assert corpus.topology == "federated"
        assert corpus.backend == "lance"
        assert corpus.corpus_root == tmp_path.resolve()
        assert corpus.batch_executor is not None
        assert corpus.feature_registry is not None
        assert corpus.metadata_index is not None
        assert len(corpus.dataset_entries) == 2

    def test_feature_registry_properties(self, tmp_path: Path) -> None:
        """Feature registry reflects mock gene vocabulary."""
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        fr = corpus.feature_registry
        assert fr.n_datasets == 2
        assert fr.global_vocab_size == N_GENES
        assert fr.max_local_vocab == N_GENES
        assert fr.dataset_ids == ("mock_00", "mock_01")

    def test_metadata_index_properties(self, tmp_path: Path) -> None:
        """Metadata index has correct row count and columns."""
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        mi = corpus.metadata_index
        assert len(mi) == 25
        assert "dataset_id" in mi.df.columns
        # Verify dataset separation
        mock00 = mi.df.filter(pl.col("dataset_id") == "mock_00")
        assert len(mock00) == 10
        mock01 = mi.df.filter(pl.col("dataset_id") == "mock_01")
        assert len(mock01) == 15

    def test_batch_executor_reads_cells(self, tmp_path: Path) -> None:
        """Batch executor can read expression + metadata."""
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        batch = corpus.batch_executor.read_batch([2, 8, 11, 20])
        assert batch["batch_size"] == 4
        assert len(batch["expressed_gene_indices"]) > 0
        assert len(batch["canonical_perturbation"]) == 4

    def test_dataset_entries_have_lance_paths(self, tmp_path: Path) -> None:
        """Federated entries are ``LanceDatasetEntry`` with per-file paths."""
        _build_mock_federated_corpus(tmp_path)
        corpus = load_corpus(str(tmp_path))

        from perturb_data_lab.loaders.expression import LanceDatasetEntry

        for entry in corpus.dataset_entries:
            assert isinstance(entry, LanceDatasetEntry)
            assert Path(str(entry.lance_path)).exists()
            assert str(entry.lance_path).endswith(f"{entry.dataset_id}.lance")


class TestLoadCorpusErrors:
    """Error handling tests."""

    def test_missing_corpus_index(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when corpus-index.yaml is missing."""
        with pytest.raises(FileNotFoundError, match="corpus-index.yaml"):
            load_corpus(str(tmp_path))

    def test_missing_canonical_obs(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when canonical-obs.parquet is missing."""
        _build_mock_aggregate_corpus(tmp_path)
        # Remove one obs file
        obs_path = tmp_path / "meta" / "mock_00" / "canonical_meta" / "canonical-obs.parquet"
        obs_path.unlink()

        with pytest.raises(FileNotFoundError, match="canonical-obs.parquet"):
            load_corpus(str(tmp_path))

    def test_missing_canonical_var(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when canonical-var.parquet is missing."""
        _build_mock_aggregate_corpus(tmp_path)
        # Remove one var file
        var_path = tmp_path / "meta" / "mock_00" / "canonical_meta" / "canonical-var.parquet"
        var_path.unlink()

        with pytest.raises(FileNotFoundError, match="canonical-var.parquet"):
            load_corpus(str(tmp_path))

    def test_missing_lance_file_aggregate(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when aggregate Lance file is missing."""
        _build_mock_aggregate_corpus(tmp_path)
        lance_path = tmp_path / "matrix" / "aggregated-cells.lance"
        import shutil
        shutil.rmtree(lance_path)

        with pytest.raises(FileNotFoundError, match="aggregated-cells.lance"):
            load_corpus(str(tmp_path))

    def test_missing_lance_file_federated(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when a federated Lance file is missing."""
        _build_mock_federated_corpus(tmp_path)
        lance_path = tmp_path / "mock_00" / "matrix" / "mock_00.lance"
        import shutil
        shutil.rmtree(lance_path)

        with pytest.raises(FileNotFoundError, match=".lance"):
            load_corpus(str(tmp_path))

    def test_unknown_topology(self, tmp_path: Path) -> None:
        """Raises ValueError for unknown topology."""
        _build_mock_aggregate_corpus(tmp_path)
        # Corrupt the topology in the index
        index_path = tmp_path / "corpus-index.yaml"
        with open(index_path) as f:
            doc = yaml.safe_load(f)
        doc["global_metadata"]["topology"] = "unknown"
        with open(index_path, "w") as f:
            yaml.safe_dump(doc, f)

        with pytest.raises(ValueError, match="topology"):
            load_corpus(str(tmp_path))

    def test_unknown_backend(self, tmp_path: Path) -> None:
        """Raises ValueError for unknown backend."""
        _build_mock_aggregate_corpus(tmp_path)
        # Corrupt the backend
        index_path = tmp_path / "corpus-index.yaml"
        with open(index_path) as f:
            doc = yaml.safe_load(f)
        doc["global_metadata"]["backend"] = "unknown_backend"
        with open(index_path, "w") as f:
            yaml.safe_dump(doc, f)

        with pytest.raises(ValueError, match="backend"):
            load_corpus(str(tmp_path))
