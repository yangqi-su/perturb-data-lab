"""Tests for the direct h5ad-to-CSR builder and packager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import polars as pl
import pytest
import yaml

from perturb_data_lab.loaders.corpus_loader import load_corpus
from perturb_data_lab.loaders.expression import AggregateCsrMemmapReader, CsrMemmapShardEntry
from perturb_data_lab.loaders.loaders import ExpressionBatchDataset, RawExpressionBatchDataset
from perturb_data_lab.materializers.direct_csr import (
    DirectH5adCsrReader,
    convert_h5ad_to_csr_dataset,
    package_csr_corpus,
)
from perturb_data_lab.materializers.backends.csr_memmap import CsrMemmapWriter


def _write_mock_h5ad(
    path: Path,
    *,
    shape: tuple[int, int],
    indptr: list[int],
    indices: list[int],
    data: list[float],
    encoding_type: str = "csr_matrix",
) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("X")
        group.attrs["encoding-type"] = encoding_type
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["shape"] = shape
        group.create_dataset("indptr", data=np.array(indptr, dtype=np.int64))
        group.create_dataset("indices", data=np.array(indices, dtype=np.int64))
        group.create_dataset("data", data=np.array(data, dtype=np.float32))


def _read_entries_from_manifest(corpus_root: Path) -> list[CsrMemmapShardEntry]:
    with open(corpus_root / "csr-corpus-manifest.yaml", "r") as fh:
        doc = yaml.safe_load(fh)
    entries: list[CsrMemmapShardEntry] = []
    for shard in doc["shards"]:
        shard_path = corpus_root / shard["path"]
        entries.append(
            CsrMemmapShardEntry(
                dataset_id=str(shard.get("dataset_id", shard["path"])),
                global_start=int(shard["global_start"]),
                global_end=int(shard["global_end"]),
                shard_id=int(shard["shard_id"]),
                shard_path=shard_path,
                row_offsets_path=shard_path / "row_offsets.npy",
                gene_indices_path=shard_path / "gene_indices.npy",
                counts_path=shard_path / "counts.npy",
                n_cells=int(shard["n_cells"]),
            )
        )
    entries.sort(key=lambda e: e.global_start)
    return entries


def _write_canonical_meta(
    root: Path,
    dataset_id: str,
    dataset_index: int,
    global_start: int,
    n_cells: int,
    n_genes: int,
) -> None:
    canonical_root = root / "meta" / dataset_id / "canonical_meta"
    canonical_root.mkdir(parents=True, exist_ok=True)

    obs_rows: list[dict[str, Any]] = []
    for local_idx in range(n_cells):
        obs_rows.append(
            {
                "cell_id": f"{dataset_id}_cell_{local_idx}",
                "dataset_id": dataset_id,
                "dataset_index": str(dataset_index),
                "global_row_index": str(global_start + local_idx),
                "local_row_index": str(local_idx),
                "size_factor": "1.0",
                "perturb_label": "control",
                "cell_context": "",
            }
        )
    var_rows = [
        {
            "origin_index": str(i),
            "gene_id": f"GENE_{dataset_id}_{i}",
            "canonical_gene_id": f"GENE_{i}",
            "global_id": str(i),
        }
        for i in range(n_genes)
    ]
    pl.DataFrame(obs_rows).write_parquet(canonical_root / "canonical-obs.parquet")
    pl.DataFrame(var_rows).write_parquet(canonical_root / "canonical-var.parquet")


def _write_template_corpus(root: Path) -> None:
    datasets = [
        {
            "dataset_id": "ds0",
            "dataset_index": 0,
            "cell_count": 3,
            "global_start": 0,
            "global_end": 3,
            "n_genes": 5,
        },
        {
            "dataset_id": "ds1",
            "dataset_index": 1,
            "cell_count": 3,
            "global_start": 3,
            "global_end": 6,
            "n_genes": 5,
        },
    ]
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "corpus-index.yaml", "w") as fh:
        yaml.safe_dump(
            {
                "kind": "corpus-index",
                "contract_version": "0.3.0",
                "global_metadata": {
                    "backend": "lance",
                    "topology": "aggregate",
                },
                "datasets": [
                    {
                        "dataset_id": ds["dataset_id"],
                        "join_mode": "create_new" if ds["dataset_index"] == 0 else "append_routed",
                        "dataset_index": ds["dataset_index"],
                        "cell_count": ds["cell_count"],
                        "global_start": ds["global_start"],
                        "global_end": ds["global_end"],
                        "manifest_path": f"meta/{ds['dataset_id']}/materialization-manifest.yaml",
                    }
                    for ds in datasets
                ],
            },
            fh,
            default_flow_style=False,
            sort_keys=False,
        )
    with open(root / "global-metadata.yaml", "w") as fh:
        yaml.safe_dump(
            {
                "kind": "global-metadata",
                "backend": "lance",
                "topology": "aggregate",
            },
            fh,
            default_flow_style=False,
            sort_keys=False,
        )
    (root / "corpus-ledger.parquet").write_bytes(b"ledger")

    for ds in datasets:
        _write_canonical_meta(
            root,
            dataset_id=ds["dataset_id"],
            dataset_index=ds["dataset_index"],
            global_start=ds["global_start"],
            n_cells=ds["cell_count"],
            n_genes=ds["n_genes"],
        )
        manifest_dir = root / "meta" / ds["dataset_id"]
        with open(manifest_dir / "materialization-manifest.yaml", "w") as fh:
            yaml.safe_dump(
                {
                    "kind": "materialization-manifest",
                    "dataset_id": ds["dataset_id"],
                    "backend": "lance",
                    "topology": "aggregate",
                },
                fh,
                default_flow_style=False,
                sort_keys=False,
            )


def _build_packaged_csr_corpus(tmp_path: Path) -> Path:
    template_root = tmp_path / "template"
    _write_template_corpus(template_root)

    ds0_source = tmp_path / "ds0.h5ad"
    _write_mock_h5ad(
        ds0_source,
        shape=(3, 5),
        indptr=[0, 2, 2, 4],
        indices=[0, 1, 2, 4],
        data=[1.0, 2.0, 3.0, 4.0],
    )
    ds1_source = tmp_path / "ds1.h5ad"
    _write_mock_h5ad(
        ds1_source,
        shape=(3, 5),
        indptr=[0, 1, 3, 3],
        indices=[1, 0, 4],
        data=[5.0, 6.0, 7.0],
    )

    ds0_manifest = convert_h5ad_to_csr_dataset(
        dataset_id="ds0",
        source_h5ad_path=ds0_source,
        output_dir=tmp_path / "ds0-out",
        global_row_start=0,
        dataset_index=0,
        shard_n_cells=2,
        rows_per_chunk=2,
    )
    ds1_manifest = convert_h5ad_to_csr_dataset(
        dataset_id="ds1",
        source_h5ad_path=ds1_source,
        output_dir=tmp_path / "ds1-out",
        global_row_start=3,
        dataset_index=1,
        shard_n_cells=2,
        rows_per_chunk=2,
    )

    packaged_root = tmp_path / "packaged"
    package_csr_corpus(
        dataset_manifest_paths=[ds0_manifest, ds1_manifest],
        template_corpus_root=template_root,
        output_root=packaged_root,
    )
    return packaged_root


class TestCsrChunkWriter:
    def test_append_csr_chunk_across_shards_and_duplicate_reads(self, tmp_path: Path) -> None:
        out = tmp_path / "csr"
        writer = CsrMemmapWriter(out, shard_n_cells=2, global_row_start=10)

        indptr = np.array([0, 2, 2, 5, 6], dtype=np.int64)
        indices = np.array([0, 1, 2, 3, 4, 1], dtype=np.int32)
        counts = np.array([5, 6, 7, 8, 9, 10], dtype=np.int32)
        writer.append_csr_chunk(10, indptr, indices, counts)
        writer.finalize()

        with open(out / "csr-corpus-manifest.yaml", "r") as fh:
            manifest = yaml.safe_load(fh)
        assert [
            (shard["global_start"], shard["global_end"], shard["n_cells"])
            for shard in manifest["shards"]
        ] == [(10, 12, 2), (12, 14, 2)]

        reader = AggregateCsrMemmapReader(_read_entries_from_manifest(out))
        rows = reader.read_expression([13, 10, 12, 10])
        assert [row.global_row_index for row in rows] == [13, 10, 12, 10]
        np.testing.assert_array_equal(rows[0].expressed_gene_indices, [1])
        np.testing.assert_array_equal(rows[1].expression_counts, [5, 6])
        np.testing.assert_array_equal(rows[2].expressed_gene_indices, [2, 3, 4])

    def test_append_csr_chunk_rejects_bad_dtypes(self, tmp_path: Path) -> None:
        writer = CsrMemmapWriter(tmp_path / "csr", shard_n_cells=2)
        with pytest.raises(TypeError, match="indices"):
            writer.append_csr_chunk(
                0,
                np.array([0, 1], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int32),
            )


class TestDirectH5adReaderAndConverter:
    def test_reader_streams_chunks_and_rejects_unsupported_layout(self, tmp_path: Path) -> None:
        source = tmp_path / "toy.h5ad"
        _write_mock_h5ad(
            source,
            shape=(4, 5),
            indptr=[0, 2, 2, 5, 6],
            indices=[0, 1, 2, 3, 4, 1],
            data=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        reader = DirectH5adCsrReader(source)
        chunks = list(reader.iter_chunks(rows_per_chunk=2))
        assert [(chunk.row_start, chunk.row_stop) for chunk in chunks] == [(0, 2), (2, 4)]
        np.testing.assert_array_equal(chunks[0].indptr, [0, 2, 2])
        np.testing.assert_array_equal(chunks[1].indices, [2, 3, 4, 1])

        bad_source = tmp_path / "bad.h5ad"
        _write_mock_h5ad(
            bad_source,
            shape=(1, 1),
            indptr=[0, 1],
            indices=[0],
            data=[1.0],
            encoding_type="csc_matrix",
        )
        with pytest.raises(NotImplementedError, match="csr_matrix"):
            DirectH5adCsrReader(bad_source)

    def test_convert_h5ad_to_csr_dataset_writes_manifest_and_round_trips(self, tmp_path: Path) -> None:
        source = tmp_path / "toy.h5ad"
        _write_mock_h5ad(
            source,
            shape=(4, 5),
            indptr=[0, 2, 2, 5, 6],
            indices=[0, 1, 2, 3, 4, 1],
            data=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        out = tmp_path / "dataset-out"
        manifest_path = convert_h5ad_to_csr_dataset(
            dataset_id="toy_ds",
            source_h5ad_path=source,
            output_dir=out,
            global_row_start=7,
            dataset_index=2,
            shard_n_cells=2,
            rows_per_chunk=3,
        )

        with open(manifest_path, "r") as fh:
            manifest = yaml.safe_load(fh)
        assert manifest["dataset_id"] == "toy_ds"
        assert manifest["global_start"] == 7
        assert manifest["global_end"] == 11
        assert manifest["shard_count"] == 2

        reader = AggregateCsrMemmapReader(_read_entries_from_manifest(out))
        batch = reader.read_expression_flat([10, 7, 9])
        np.testing.assert_array_equal(batch.global_row_index, [10, 7, 9])
        np.testing.assert_array_equal(batch.row_offsets, [0, 1, 3, 6])
        np.testing.assert_array_equal(batch.expressed_gene_indices, [1, 0, 1, 2, 3, 4])
        np.testing.assert_array_equal(batch.expression_counts, [10, 5, 6, 7, 8, 9])


class TestCsrPackager:
    def test_package_csr_corpus_builds_loadable_overlay(self, tmp_path: Path) -> None:
        packaged_root = _build_packaged_csr_corpus(tmp_path)

        with open(packaged_root / "csr-corpus-manifest.yaml", "r") as fh:
            merged = yaml.safe_load(fh)
        assert merged["total_cells"] == 6
        assert merged["n_shards"] == 4
        assert [
            (shard["global_start"], shard["global_end"])
            for shard in merged["shards"]
        ] == [(0, 2), (2, 3), (3, 5), (5, 6)]

        with open(packaged_root / "corpus-index.yaml", "r") as fh:
            corpus_index = yaml.safe_load(fh)
        assert corpus_index["global_metadata"]["backend"] == "csr-memmap"

        corpus = load_corpus(packaged_root)
        assert corpus.backend == "csr_memmap"
        batch = corpus.batch_executor.read_batch([0, 2, 3, 5])
        np.testing.assert_array_equal(batch["global_row_index"], [0, 2, 3, 5])
        np.testing.assert_array_equal(batch["row_offsets"], [0, 2, 4, 5, 5])
        np.testing.assert_array_equal(batch["expressed_gene_indices"], [0, 1, 2, 4, 1])
        np.testing.assert_array_equal(batch["expression_counts"], [1, 2, 3, 4, 5])

    def test_packaged_csr_corpus_uses_expression_dataset_hot_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        packaged_root = _build_packaged_csr_corpus(tmp_path)
        corpus = load_corpus(packaged_root, seq_len=3)

        dataset = corpus.dataset()
        assert type(dataset) is ExpressionBatchDataset
        assert dataset.backend == "csr_memmap"
        assert dataset.topology == "aggregate"

        batch = dataset.__getitems__([0, 2, 3, 5])[0]
        np.testing.assert_array_equal(batch["global_row_index"], [0, 2, 3, 5])
        np.testing.assert_array_equal(batch["dataset_index"], [0, 0, 1, 1])
        np.testing.assert_array_equal(batch["local_row_index"], [0, 2, 0, 2])

        with pytest.warns(DeprecationWarning, match="RawExpressionBatchDataset"):
            legacy_dataset = corpus.dataset(metadata_columns=["perturb_label"])
        assert type(legacy_dataset) is RawExpressionBatchDataset
        legacy_batch = legacy_dataset.__getitems__([0, 3])[0]
        assert legacy_batch["meta_columns"]["perturb_label"] == ("control", "control")

        def _fail_read_raw_batch(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("loader hot path should not call BatchExecutor.read_raw_batch")

        monkeypatch.setattr(corpus.batch_executor, "read_raw_batch", _fail_read_raw_batch)

        processed = next(
            corpus.loader(
                processing="cpu",
                batch_size=4,
                seq_len=3,
                shuffle=False,
                metadata_columns=["perturb_label"],
            )
        )

        assert processed["batch_size"] == 4
        assert processed["sampled_gene_ids"].shape == (4, 3)
        assert tuple(processed["meta_columns"]["perturb_label"]) == (
            "control",
            "control",
            "control",
            "control",
        )
