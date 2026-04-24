"""Phase 5 corpus runtime loader: unified multi-dataset access layer.

This module provides:
- ``CorpusLoader``: reads corpus index/metadata, loads tokenizer once,
  opens per-dataset readers, and exposes unified access.
- ``CorpusEmissionSpec`` integration for runtime field emission.
- ``HVGRandomSampler``-compatible per-dataset HVG/non-HVG token-ID arrays.
- Token-space translation using the corpus tokenizer.

All corpus runtime artifacts are corpus-level and read-only at load time.
Dataset-level per-cell data is read lazily through the per-dataset reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..materializers.models import (
    CorpusIndexDocument,
    GlobalMetadataDocument,
    DatasetJoinRecord,
)
from ..materializers.tokenizer import CorpusTokenizer
from ..materializers.emission_spec import CorpusEmissionSpec
from .loaders import (
    BackendCellReader,
    CellIdentity,
    CellState,
    CPUDenseRuntimePath,
    CorpusRandomBatchSampler,
    DatasetBatchSampler,
    DatasetContextBatchSampler,
    GlobalFeatureResolver,
    GPUSparseRuntimePath,
    HVGRandomSampler,
    MetadataTable,
    SamplerState,
    SparseBatchCollator,
    SparseBatchPayload,
    AVAILABLE_READERS,
    build_cell_reader,
)

__all__ = [
    "CorpusLoader",
    "DatasetReaderEntry",
    "build_corpus_loader",
]


@dataclass(frozen=True)
class DatasetReaderEntry:
    """Holds the per-dataset reader and its associated metadata.

    HVG arrays are stored in dataset-original feature index space (int32).
    Use ``token_hvg_set`` / ``token_nonhvg_set`` to get token-ID versions
    compatible with ``HVGRandomSampler``.

    ``global_start`` and ``global_end`` are the authoritative global cell range
    for this dataset, as written by ``update_corpus_index`` into the corpus index.
    The ``CorpusLoader`` uses these values for global-index routing, making the
    corpus index the single source of truth for cell ownership rather than
    recomputing ranges from reader lengths.
    """

    dataset_id: str
    dataset_index: int
    release_id: str
    manifest_path: Path
    reader: BackendCellReader  # any backend-specific reader

    # Authoritative global cell range for this dataset (from corpus index)
    global_start: int = 0  # inclusive start of this dataset's cells in the corpus
    global_end: int = 0    # exclusive end of this dataset's cells in the corpus

    # HVG arrays in dataset-original feature index space (int32)
    hvg_indices: tuple[int, ...] = ()
    nonhvg_indices: tuple[int, ...] = ()

    # Total number of features in this dataset's original var space
    n_vars: int = 0

    # Cached token-space HVG/non-HVG sets (translated lazily via reader's
    # preloaded feature objects).  Lazily populated on first access.
    _token_hvg_set: tuple[int, ...] | None = field(default=None, repr=False)
    _token_nonhvg_set: tuple[int, ...] | None = field(default=None, repr=False)

    @property
    def hvg_set(self) -> tuple[int, ...]:
        """HVG indices in dataset-original feature space."""
        return self.hvg_indices

    @property
    def nonhvg_set(self) -> tuple[int, ...]:
        """Non-HVG indices in dataset-original feature space."""
        return self.nonhvg_indices

    @property
    def token_hvg_set(self) -> tuple[int, ...]:
        """HVG indices translated to corpus token IDs.

        Uses the reader's preloaded feature objects (written during
        materialization) to translate each dataset-original index to its
        global token ID.  Returns the dataset-original indices unchanged
        if no preloaded feature objects are available.
        """
        if self._token_hvg_set is None:
            self._translate_hvg_arrays()
        # type-ignore: set during _translate_hvg_arrays
        return self._token_hvg_set  # type: ignore[return-value]

    @property
    def token_nonhvg_set(self) -> tuple[int, ...]:
        """Non-HVG indices translated to corpus token IDs."""
        if self._token_nonhvg_set is None:
            self._translate_hvg_arrays()
        return self._token_nonhvg_set  # type: ignore[return-value]

    def _translate_hvg_arrays(self) -> None:
        """Translate HVG/non-HVG arrays from dataset-original indices to token IDs."""
        if self.reader is None or self.reader.preloaded_features is None:
            # No preloaded features — fall back to identity (original indices as tokens)
            object.__setattr__(self, "_token_hvg_set", self.hvg_indices)
            object.__setattr__(self, "_token_nonhvg_set", self.nonhvg_indices)
            return

        translator = self.reader.preloaded_features

        def translate_tuple(indices: tuple[int, ...]) -> tuple[int, ...]:
            result = []
            for idx in indices:
                tok = translator.origin_index_to_token_id(int(idx))
                if tok < 0:
                    continue  # skip unknown indices
                result.append(tok)
            return tuple(result)

        object.__setattr__(self, "_token_hvg_set", translate_tuple(self.hvg_indices))
        object.__setattr__(
            self, "_token_nonhvg_set", translate_tuple(self.nonhvg_indices)
        )


class CorpusLoader:
    """Unified multi-dataset access layer for a corpus.

    Provides corpus-level iteration, indexed random access, and per-dataset
    HVG-aware sampling while preserving dataset-local sparse access internally.

    Usage::

        loader = CorpusLoader.from_corpus_index(
            corpus_index_path=Path("corpus/corpus-index.yaml"),
            backend="arrow-hf",
        )

        # Corpus-level iteration
        for cell_state in loader.iter_cells():
            print(cell_state.cell_id, cell_state.dataset_id)

        # Direct index access (global corpus index)
        cell = loader.read_cell(corpus_global_index=42)

        # Per-dataset sampling with HVG awareness
        entry = loader.dataset_reader("replogle_k562")
        state = SamplerState(
            mode="hvg_random",
            total_cells=len(entry.reader),
            n_genes=entry.n_vars,
            hvg_set=entry.token_hvg_set,  # token-ID space for HVGRandomSampler
        )
        sampler = HVGRandomSampler(state, np.random.default_rng(42))
        context = sampler.sample_indices(cell, max_context=512)
    """

    def __init__(
        self,
        corpus_id: str,
        corpus_root: Path,
        tokenizer: "CorpusTokenizer | None",
        emission_spec: CorpusEmissionSpec,
        dataset_entries: tuple[DatasetReaderEntry, ...],
        metadata_table: MetadataTable | None = None,
    ):
        """
        Parameters
        ----------
        corpus_id : str
            Corpus identifier.
        corpus_root : Path
            Root directory of the corpus (where corpus-index.yaml lives).
        tokenizer : CorpusTokenizer | None
            The corpus-level tokenizer (loaded once from corpus root).
            Optional since Phase 3 (tokenizer-free architecture); may be None.
        emission_spec : CorpusEmissionSpec
            The corpus-level emission spec controlling which fields loaders emit.
        dataset_entries : tuple[DatasetReaderEntry, ...]
            Per-dataset reader entries in the same order as the corpus index.
        """
        self.corpus_id = corpus_id
        self.corpus_root = corpus_root
        self.tokenizer = tokenizer
        self.emission_spec = emission_spec
        self._dataset_entries = dataset_entries
        self._metadata_table = metadata_table
        self._sparse_batch_collator = SparseBatchCollator()

        # Build dataset_id → entry index map for O(1) lookup
        self._dataset_id_to_idx: dict[str, int] = {
            entry.dataset_id: idx for idx, entry in enumerate(dataset_entries)
        }

        # Build per-dataset cumulative cell offsets for global index routing.
        # Phase 4: total_cells and offsets are derived from the authoritative
        # global_start/global_end fields in the corpus index, not from reader lengths.
        self._total_cells = max((entry.global_end for entry in dataset_entries), default=0)

    # -------------------------------------------------------------------------
    # Corpus-level access
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of cells across all datasets in the corpus."""
        return self._total_cells

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """Ordered tuple of dataset IDs in this corpus."""
        return tuple(e.dataset_id for e in self._dataset_entries)

    def dataset_reader(self, dataset_id: str) -> DatasetReaderEntry:
        """Return the reader entry for a given dataset_id."""
        idx = self._dataset_id_to_idx[dataset_id]
        return self._dataset_entries[idx]

    def iter_datasets(self) -> tuple[DatasetReaderEntry, ...]:
        """Return all per-dataset reader entries."""
        return self._dataset_entries

    def read_cell(self, corpus_global_index: int) -> CellState:
        """Read a single cell by its corpus-global index.

        Routing is O(log n_datasets) via binary search on authoritative global_end values
        from the corpus index. The returned CellState carries dataset-order indices (not token IDs).
        """
        if corpus_global_index < 0 or corpus_global_index >= self._total_cells:
            raise IndexError(
                f"corpus_global_index {corpus_global_index} out of range [0, {self._total_cells})"
            )

        # Binary search for the dataset that owns this global index
        dataset_idx = self._find_dataset_idx(corpus_global_index)
        entry = self._dataset_entries[dataset_idx]
        local_index = corpus_global_index - entry.global_start
        if self._metadata_table is None:
            return entry.reader.read_cell(local_index)
        heavy_row = entry.reader.read_row(local_index)
        metadata = self._metadata_table.row(corpus_global_index)
        return self._compose_cell_state(metadata, heavy_row)

    def read_cells(self, corpus_global_indices: list[int]) -> list[CellState]:
        """Read multiple corpus-global rows with dataset-grouped heavy reads."""
        if not corpus_global_indices:
            return []

        normalized = [int(idx) for idx in corpus_global_indices]
        for idx in normalized:
            if idx < 0 or idx >= self._total_cells:
                raise IndexError(
                    f"corpus_global_index {idx} out of range [0, {self._total_cells})"
                )

        if self._metadata_table is None:
            return [self.read_cell(idx) for idx in normalized]

        grouped: dict[int, list[tuple[int, int, int]]] = {}
        for output_position, global_index in enumerate(normalized):
            dataset_idx = self._find_dataset_idx(global_index)
            entry = self._dataset_entries[dataset_idx]
            local_index = global_index - entry.global_start
            grouped.setdefault(dataset_idx, []).append(
                (output_position, global_index, local_index)
            )

        result: list[CellState | None] = [None] * len(normalized)
        for dataset_idx, selections in grouped.items():
            entry = self._dataset_entries[dataset_idx]
            heavy_rows = entry.reader.read_rows([local for _, _, local in selections])
            metadata_rows = self._metadata_table.rows(
                [global_index for _, global_index, _ in selections]
            )
            for (output_position, _global_index, _local_index), metadata, heavy_row in zip(
                selections, metadata_rows, heavy_rows
            ):
                result[output_position] = self._compose_cell_state(metadata, heavy_row)

        return [cell for cell in result if cell is not None]

    def collate_sparse_batch(self, corpus_global_indices: list[int]) -> SparseBatchPayload:
        """Read and collate a dataset-aware sparse batch keyed by global rows."""
        return self._sparse_batch_collator(self.read_cells(corpus_global_indices))

    def dataset_batch_sampler(
        self,
        *,
        dataset_id: str,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ) -> DatasetBatchSampler:
        if self._metadata_table is None:
            raise ValueError("dataset batch sampling requires a RAM-resident metadata table")
        entry = self.dataset_reader(dataset_id)
        return DatasetBatchSampler(
            metadata_table=self._metadata_table,
            dataset_index=entry.dataset_index,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
        )

    def dataset_context_batch_sampler(
        self,
        *,
        batch_size: int,
        context_field: str = "cell_context",
        dataset_id: str | None = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ) -> DatasetContextBatchSampler:
        if self._metadata_table is None:
            raise ValueError("dataset-context batch sampling requires a RAM-resident metadata table")
        dataset_index = None
        if dataset_id is not None:
            dataset_index = self.dataset_reader(dataset_id).dataset_index
        return DatasetContextBatchSampler(
            metadata_table=self._metadata_table,
            batch_size=batch_size,
            context_field=context_field,
            dataset_index=dataset_index,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
        )

    def corpus_random_batch_sampler(
        self,
        *,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
    ) -> CorpusRandomBatchSampler:
        return CorpusRandomBatchSampler(
            total_rows=len(self),
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
        )

    def iter_cells(self):
        """Yield all cells across all datasets in corpus order.

        This is a convenience generator; for random access use ``read_cell``.
        """
        for entry in self._dataset_entries:
            reader = entry.reader
            for local_idx in range(len(reader)):
                yield self.read_cell(entry.global_start + local_idx)

    @property
    def metadata_table(self) -> MetadataTable | None:
        """Shared RAM-resident metadata table when available."""
        return self._metadata_table

    def _find_dataset_idx(self, global_index: int) -> int:
        """Binary search for the dataset index owning global_index.

        Uses the authoritative global_end field from each entry (written by
        update_corpus_index) rather than recomputing from reader lengths.
        """
        lo, hi = 0, len(self._dataset_entries) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._dataset_entries[mid].global_end <= global_index:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _compose_cell_state(
        self,
        metadata: dict[str, Any],
        heavy_row: dict[str, Any],
    ) -> CellState:
        if int(metadata["global_row_index"]) != int(heavy_row["global_row_index"]):
            raise ValueError(
                "metadata/heavy row mismatch for global_row_index: "
                f"{metadata['global_row_index']} != {heavy_row['global_row_index']}"
            )
        return CellState(
            identity=CellIdentity(
                global_row_index=int(metadata["global_row_index"]),
                dataset_index=int(metadata["dataset_index"]),
                dataset_id=str(metadata["dataset_id"]),
                local_row_index=int(metadata["local_row_index"]),
            ),
            cell_id=str(metadata["cell_id"]),
            expressed_gene_indices=tuple(int(i) for i in heavy_row["expressed_gene_indices"]),
            expression_counts=tuple(int(c) for c in heavy_row["expression_counts"]),
            size_factor=float(metadata["size_factor"]),
            canonical_perturbation=dict(metadata["canonical_perturbation"]),
            canonical_context=dict(metadata["canonical_context"]),
            raw_fields=dict(metadata["raw_fields"]),
        )

    # -------------------------------------------------------------------------
    # Token-space translation
    # -------------------------------------------------------------------------

    def translate_origin_indices_to_tokens(
        self,
        dataset_id: str,
        origin_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Translate dataset-order feature indices to global token IDs.

        Uses the per-dataset preloaded feature objects (written during
        materialization) for O(1) per-index translation.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        origin_indices : tuple[int, ...]
            Feature indices in dataset-original feature space.

        Returns
        -------
        tuple[int, ...]
            Corresponding token IDs from the corpus tokenizer.
        """
        entry = self.dataset_reader(dataset_id)
        reader = entry.reader
        if reader.preloaded_features is not None:
            return reader.translate_to_token_ids(origin_indices)
        return origin_indices

    def build_global_feature_resolver(self) -> GlobalFeatureResolver:
        """Build a post-canonicalization dataset-local → global feature resolver."""
        dataset_mappings: dict[int, np.ndarray] = {}
        total_features = 0
        for entry in self._dataset_entries:
            if entry.reader.preloaded_features is None:
                raise ValueError(
                    "post-canonicalization global feature resolution is unavailable for "
                    f"dataset_id={entry.dataset_id}; provide an explicit resolver table "
                    "or materialized token mapping before using CPU-dense/GPU-sparse runtime paths"
                )
            if entry.n_vars <= 0:
                raise ValueError(
                    f"dataset_id={entry.dataset_id} does not expose n_vars needed to build a resolver"
                )
            mapping = np.asarray(
                entry.reader.translate_to_token_ids(tuple(range(entry.n_vars))),
                dtype=np.int32,
            )
            dataset_mappings[int(entry.dataset_index)] = mapping
            valid = mapping[mapping >= 0]
            if valid.size:
                total_features = max(total_features, int(valid.max()) + 1)
        return GlobalFeatureResolver.from_dataset_mappings(
            dataset_mappings,
            total_features=total_features,
        )

    def cpu_dense_runtime_path(
        self,
        resolver: GlobalFeatureResolver | None = None,
    ) -> CPUDenseRuntimePath:
        if resolver is None:
            resolver = self.build_global_feature_resolver()
        return CPUDenseRuntimePath(resolver)

    def gpu_sparse_runtime_path(
        self,
        resolver: GlobalFeatureResolver | None = None,
    ) -> GPUSparseRuntimePath:
        if resolver is None:
            resolver = self.build_global_feature_resolver()
        return GPUSparseRuntimePath(resolver)

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------

    @staticmethod
    def _reader_kwargs(
        manifest: "MaterializationManifest",
        meta_root: Path,
        backend: str,
        corpus_root: Path | None = None,
        manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        """Build backend-specific kwargs for build_cell_reader.

        Each backend reader has a different constructor signature. This method
        bridges the manifest's declared paths to the reader's expected kwargs.

        Parameters
        ----------
        manifest : MaterializationManifest
            The dataset's materialization manifest.
        meta_root : Path
            Directory containing the manifest. For Arrow/HF, used only for
            HVG sidecar resolution (they are in meta/, not matrix/).
        backend : str
            Storage backend identifier.
        corpus_root : Path | None
            The corpus root directory (where corpus-index.yaml lives).
            Used for resolving Arrow/HF matrix paths when manifest_path is relative.
        manifest_path : Path | None
            The join record's manifest_path (relative to corpus_root or absolute).
            Used to determine whether Arrow/HF artifacts are inside or outside corpus_root.
        """
        if backend in {"arrow-hf", "lancedb-aggregated"}:
            if backend == "lancedb-aggregated":
                arrow_meta_root = meta_root
                if corpus_root is None:
                    raise ValueError("lancedb-aggregated reader resolution requires corpus_root")
                arrow_matrix_root = corpus_root / "matrix"
                cells_artifact = arrow_matrix_root / "aggregated-corpus.lance"
            else:
                # Arrow/HF runtime artifacts (cells parquet, canonical SQLite, meta parquet)
                # are written into matrix_root, not meta_root.  For relative manifest_path
                # (stored relative to corpus_root), compute matrix_root from corpus_root.
                # For absolute manifest_path (outside corpus_root), use manifest.outputs.matrix_root.
                if manifest_path is not None and corpus_root is not None:
                    # Try to resolve relative to corpus_root first
                    try:
                        manifest_rel = Path(manifest_path).relative_to(corpus_root)
                        # manifest is inside corpus_root
                        arrow_meta_root = corpus_root / Path(manifest_path).parent
                        arrow_matrix_root = corpus_root / "matrix"
                    except ValueError:
                        # manifest is outside corpus_root — use manifest.outputs.matrix_root
                        arrow_meta_root = Path(manifest_path).parent
                        arrow_matrix_root = Path(manifest.outputs.matrix_root)
                else:
                    # Fallback: derive from meta_root (original behavior for backward compat)
                    arrow_meta_root = meta_root
                    arrow_matrix_root = meta_root.parent / "matrix"
                cells_artifact = arrow_matrix_root / f"{manifest.release_id}-cells.parquet"

            cell_meta_sqlite = Path(manifest.outputs.matrix_root) / f"{manifest.release_id}-cell-meta.sqlite"
            meta_parquet = Path(manifest.outputs.matrix_root) / f"{manifest.release_id}-meta.parquet"
            feature_meta_paths: dict[str, Path] | None = None
            if manifest.feature_meta_paths:
                feature_meta_paths = {
                    k: Path(manifest.outputs.matrix_root) / v
                    for k, v in manifest.feature_meta_paths.items()
                }
            if backend == "lancedb-aggregated":
                return {
                    "lance_dataset_path": cells_artifact,
                    "meta_parquet_path": meta_parquet,
                    "cell_meta_sqlite_path": cell_meta_sqlite,
                    "feature_registry_path": arrow_meta_root / "feature-registry.yaml",
                    "feature_meta_paths": feature_meta_paths,
                }
            return {
                "cells_parquet_path": cells_artifact,
                "meta_parquet_path": meta_parquet,
                "cell_meta_sqlite_path": cell_meta_sqlite,
                "feature_registry_path": arrow_meta_root / "feature-registry.yaml",  # legacy fallback
                "feature_meta_paths": feature_meta_paths,
            }
        elif backend == "webdataset":
            # WebDataset uses shard paths and tab-delimited meta.txt
            shard_dir = Path(manifest.outputs.matrix_root)
            shard_paths = sorted(shard_dir.glob(f"{manifest.release_id}-*.tar"))
            meta_path = meta_root / f"{manifest.release_id}-meta.txt"
            feature_registry_path = meta_root / "feature-registry.yaml"
            return {
                "shard_paths": shard_paths,
                "meta_path": meta_path,
                "feature_registry_path": feature_registry_path,
            }
        elif backend in {"zarr-ts", "zarr-aggregated"}:
            indices_zarr = Path(manifest.outputs.matrix_root) / f"{manifest.release_id}-indices.zarr"
            counts_zarr = Path(manifest.outputs.matrix_root) / f"{manifest.release_id}-counts.zarr"
            sf_zarr = Path(manifest.outputs.matrix_root) / f"{manifest.release_id}-sf.zarr"
            meta_path = meta_root / f"{manifest.release_id}-meta.json"
            feature_registry_path = meta_root / "feature-registry.yaml"
            feature_meta_paths: dict[str, Path] | None = None
            if manifest.feature_meta_paths:
                feature_meta_paths = {
                    k: meta_root / v
                    for k, v in manifest.feature_meta_paths.items()
                }
            return {
                "indices_zarr_path": indices_zarr,
                "counts_zarr_path": counts_zarr,
                "sf_zarr_path": sf_zarr,
                "meta_path": meta_path,
                "feature_registry_path": feature_registry_path,
                "feature_meta_paths": feature_meta_paths,
            }
        else:
            raise ValueError(f"unknown backend: {backend}")

    @staticmethod
    def _build_metadata_table(
        corpus_backend: str | None,
        corpus_root: Path,
        corpus_index: CorpusIndexDocument,
    ) -> MetadataTable | None:
        if corpus_backend not in {"arrow-hf", "zarr-ts", "lancedb-aggregated", "zarr-aggregated"}:
            return None

        from ..materializers.models import MaterializationManifest

        rows: list[dict[str, Any]] = []
        for ds_record in corpus_index.datasets:
            manifest_path = corpus_root / ds_record.manifest_path
            manifest = MaterializationManifest.from_yaml_file(manifest_path)
            meta_root = manifest_path.parent
            if corpus_backend in {"arrow-hf", "lancedb-aggregated"}:
                rows.extend(
                    CorpusLoader._metadata_rows_from_arrow_sqlite(
                        manifest=manifest,
                        meta_root=meta_root,
                        backend=corpus_backend,
                        corpus_root=corpus_root,
                        manifest_path=ds_record.manifest_path,
                        ds_record=ds_record,
                    )
                )
            elif corpus_backend in {"zarr-ts", "zarr-aggregated"}:
                rows.extend(
                    CorpusLoader._metadata_rows_from_zarr_meta(
                        manifest=manifest,
                        meta_root=meta_root,
                        ds_record=ds_record,
                    )
                )

        rows.sort(key=lambda item: int(item["global_row_index"]))
        if any(int(row["global_row_index"]) != idx for idx, row in enumerate(rows)):
            raise ValueError("metadata table rows must form a contiguous global_row_index range")

        return MetadataTable(
            global_row_index=tuple(int(row["global_row_index"]) for row in rows),
            cell_id=tuple(str(row["cell_id"]) for row in rows),
            dataset_id=tuple(str(row["dataset_id"]) for row in rows),
            dataset_index=tuple(int(row["dataset_index"]) for row in rows),
            local_row_index=tuple(int(row["local_row_index"]) for row in rows),
            canonical_perturbation=tuple(dict(row["canonical_perturbation"]) for row in rows),
            canonical_context=tuple(dict(row["canonical_context"]) for row in rows),
            raw_fields=tuple(dict(row["raw_fields"]) for row in rows),
            size_factor=tuple(float(row["size_factor"]) for row in rows),
        )

    @staticmethod
    def _metadata_rows_from_arrow_sqlite(
        manifest: "MaterializationManifest",
        meta_root: Path,
        backend: str,
        corpus_root: Path,
        manifest_path: str,
        ds_record: DatasetJoinRecord,
    ) -> list[dict[str, Any]]:
        import sqlite3

        reader_kwargs = CorpusLoader._reader_kwargs(
            manifest,
            meta_root,
            backend,
            corpus_root,
            manifest_path,
        )
        sqlite_path = reader_kwargs["cell_meta_sqlite_path"]
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT rowid, cell_id, dataset_id, size_factor, canonical_perturbation, canonical_context, raw_obs "
            "FROM cell_meta ORDER BY rowid"
        ).fetchall()
        conn.close()

        result: list[dict[str, Any]] = []
        for row in rows:
            local_row_index = int(row["rowid"]) - 1
            result.append(
                {
                    "global_row_index": ds_record.global_start + local_row_index,
                    "cell_id": str(row["cell_id"]),
                    "dataset_id": str(row["dataset_id"]),
                    "dataset_index": ds_record.dataset_index,
                    "local_row_index": local_row_index,
                    "canonical_perturbation": json.loads(row["canonical_perturbation"]),
                    "canonical_context": json.loads(row["canonical_context"]),
                    "raw_fields": json.loads(row["raw_obs"]),
                    "size_factor": float(row["size_factor"]),
                }
            )
        return result

    @staticmethod
    def _metadata_rows_from_zarr_meta(
        manifest: "MaterializationManifest",
        meta_root: Path,
        ds_record: DatasetJoinRecord,
    ) -> list[dict[str, Any]]:
        import zarr

        meta_path = meta_root / f"{manifest.release_id}-meta.json"
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        sf_path = Path(manifest.outputs.matrix_root) / f"{manifest.release_id}-sf.zarr"
        sf_store = zarr.open(str(sf_path), mode="r")
        size_factors = sf_store["sf"]
        result: list[dict[str, Any]] = []
        for local_row_index, row in enumerate(payload["cells"]):
            result.append(
                {
                    "global_row_index": ds_record.global_start + local_row_index,
                    "cell_id": str(row["cell_id"]),
                    "dataset_id": str(row.get("dataset_id", ds_record.dataset_id)),
                    "dataset_index": ds_record.dataset_index,
                    "local_row_index": local_row_index,
                    "canonical_perturbation": dict(row.get("canonical_perturbation", {})),
                    "canonical_context": dict(row.get("canonical_context", {})),
                    "raw_fields": dict(row.get("raw_fields", {})),
                    "size_factor": float(size_factors[local_row_index]),
                }
            )
        return result

    @classmethod
    def from_corpus_index(
        cls,
        corpus_index_path: Path,
        backend: str = "arrow-hf",
        backend_kwargs: dict[str, Any] | None = None,
    ) -> "CorpusLoader":
        """Build a CorpusLoader from a corpus index YAML file.

        This is the primary entry point. It:
        1. Reads ``corpus-index.yaml`` and ``global-metadata.yaml``
        2. Loads the corpus backend from global-metadata (the authoritative source)
        3. Validates that all manifests declare the same backend as global-metadata
        4. Loads the corpus tokenizer from the path stored in global-metadata
        5. Loads the corpus emission spec from the path stored in global-metadata
        6. Opens per-dataset readers for each dataset in the index
        7. Loads per-dataset HVG arrays from the hvg_sidecar_path stored in each manifest
        8. Preloads per-dataset feature objects for efficient token-space translation

        Parameters
        ----------
        corpus_index_path : Path
            Path to the corpus index YAML file.
        backend : str
            Deprecated; ignored. The corpus backend is read from global-metadata.yaml.
            Kept for backwards API compatibility only.

        Returns
        -------
        CorpusLoader
            Fully initialized corpus loader.

        Raises
        ------
        ValueError
            If global-metadata backend does not match a manifest's backend.
        NotImplementedError
            If the declared corpus backend is not yet implemented.
        """
        from ..materializers.models import MaterializationManifest

        backend_kwargs = backend_kwargs or {}

        # Load corpus index
        corpus_index = CorpusIndexDocument.from_yaml_file(corpus_index_path)
        corpus_root = corpus_index_path.parent
        corpus_id = corpus_index.corpus_id

        # Load global metadata — backend field is the authoritative corpus declaration.
        # If backend is None (unset in older files), fall back to inferring from the
        # first manifest for backward compat with older corpora.
        gmeta_path = corpus_root / "global-metadata.yaml"
        if gmeta_path.exists():
            gmeta = GlobalMetadataDocument.from_yaml_file(gmeta_path)
            corpus_backend = gmeta.backend if gmeta.backend is not None else None
        else:
            gmeta = None
            corpus_backend = None

        # Infer backend from first manifest if not set in global-metadata
        if corpus_backend is None:
            # Legacy corpus without global-metadata or with unset backend:
            # infer from the first manifest's backend field if available.
            if corpus_index.datasets:
                first_manifest_path = corpus_root / corpus_index.datasets[0].manifest_path
                if first_manifest_path.exists():
                    first_manifest = MaterializationManifest.from_yaml_file(first_manifest_path)
                    corpus_backend = first_manifest.backend
                else:
                    raise FileNotFoundError(
                        f"corpus index references {first_manifest_path} which does not exist; "
                        "cannot determine corpus backend"
                    )
            # If corpus_backend is still None (empty corpus with no datasets),
            # leave it as None — the unsupported-backend check below will raise
            # NotImplementedError if an unsupported backend is requested.

        # Validate: all manifests must declare the same backend as the corpus
        if corpus_backend is not None:
            for ds_record in corpus_index.datasets:
                manifest_path = corpus_root / ds_record.manifest_path
                if manifest_path.exists():
                    manifest = MaterializationManifest.from_yaml_file(manifest_path)
                    if manifest.backend is not None and manifest.backend != corpus_backend:
                        raise ValueError(
                            f"backend mismatch: manifest {manifest_path} declares backend "
                            f"'{manifest.backend}' but corpus declares backend '{corpus_backend}' — "
                            "all datasets in a corpus must use the same backend"
                        )

        # Dispatch reader by corpus backend (not by the deprecated caller-provided backend arg)
        if corpus_backend is not None and corpus_backend not in AVAILABLE_READERS:
            raise NotImplementedError(
                f"backend '{corpus_backend}' not yet supported; "
                f"use one of {list(AVAILABLE_READERS)}"
            )

        metadata_table = cls._build_metadata_table(corpus_backend, corpus_root, corpus_index)

        # Load tokenizer if present in corpus root.
        # The tokenizer is optional since Phase 3 (tokenizer-free architecture).
        # If not present, set tokenizer to None; the loader operates without it.
        tokenizer: "CorpusTokenizer | None" = None
        tokenizer_path_str = (
            gmeta.tokenizer_path if gmeta and gmeta.tokenizer_path else None
        )
        if tokenizer_path_str:
            tokenizer_path = corpus_root / tokenizer_path_str
            if tokenizer_path.exists():
                tokenizer = CorpusTokenizer.from_json(tokenizer_path)

        # Load emission spec
        emission_spec_path_str = (
            gmeta.emission_spec_path
            if gmeta and gmeta.emission_spec_path
            else "corpus-emission-spec.yaml"
        )
        emission_spec_path = corpus_root / emission_spec_path_str
        if emission_spec_path.exists():
            emission_spec = CorpusEmissionSpec.from_yaml_file(emission_spec_path)
        else:
            emission_spec = CorpusEmissionSpec(corpus_id=corpus_id)

        # Open per-dataset readers
        dataset_entries: list[DatasetReaderEntry] = []
        for ds_record in corpus_index.datasets:
            manifest_path = corpus_root / ds_record.manifest_path
            manifest = MaterializationManifest.from_yaml_file(manifest_path)

            # Compute metadata root from manifest path
            meta_root = manifest_path.parent

            # Load HVG arrays if hvg_sidecar_path is set
            hvg_indices: tuple[int, ...] = ()
            nonhvg_indices: tuple[int, ...] = ()
            hvg_sidecar = manifest.hvg_sidecar_path
            if hvg_sidecar:
                hvg_dir = meta_root / hvg_sidecar
                hvg_path = hvg_dir / "hvg.npy"
                nonhvg_path = hvg_dir / "nonhvg.npy"
                if hvg_path.exists():
                    hvg_arr = np.load(str(hvg_path), allow_pickle=False)
                    hvg_indices = tuple(int(x) for x in hvg_arr)
                if nonhvg_path.exists():
                    nonhvg_arr = np.load(str(nonhvg_path), allow_pickle=False)
                    nonhvg_indices = tuple(int(x) for x in nonhvg_arr)

            # Build per-dataset reader via the backend factory
            # Each backend has its own reader class and constructor signature
            reader = build_cell_reader(
                backend=corpus_backend,
                dataset_id=manifest.dataset_id,
                dataset_index=ds_record.dataset_index,
                corpus_index_path=corpus_index_path,
                metadata_table=metadata_table,
                global_row_offset=ds_record.global_start,
                **cls._reader_kwargs(manifest, meta_root, corpus_backend, corpus_root, ds_record.manifest_path),
            )

            # Determine n_vars from features_origin parquet.
            # Use matrix_root for Arrow/HF since the writer stores feature parquet
            # alongside expression data in matrix_root (not meta_root).
            entry_n_vars = 0
            if manifest.feature_meta_paths and "features_origin" in manifest.feature_meta_paths:
                import pyarrow.parquet as pq

                if corpus_backend == "arrow-hf":
                    # Resolve from matrix_root (same convention as _reader_kwargs)
                    if manifest_path is not None and corpus_root is not None:
                        try:
                            _ = Path(manifest_path).relative_to(corpus_root)
                            arrow_matrix_root = corpus_root / "matrix"
                        except ValueError:
                            arrow_matrix_root = Path(manifest.outputs.matrix_root)
                    else:
                        arrow_matrix_root = meta_root.parent / "matrix"
                    origin_path = arrow_matrix_root / manifest.feature_meta_paths["features_origin"]
                elif corpus_backend == "lancedb-aggregated":
                    origin_path = Path(manifest.outputs.matrix_root) / manifest.feature_meta_paths["features_origin"]
                else:
                    origin_path = meta_root / manifest.feature_meta_paths["features_origin"]

                if origin_path.exists():
                    tbl = pq.read_table(str(origin_path))
                    entry_n_vars = tbl.num_rows

            entry = DatasetReaderEntry(
                dataset_id=manifest.dataset_id,
                dataset_index=ds_record.dataset_index,
                release_id=manifest.release_id,
                manifest_path=manifest_path,
                reader=reader,
                global_start=ds_record.global_start,
                global_end=ds_record.global_end,
                hvg_indices=hvg_indices,
                nonhvg_indices=nonhvg_indices,
                n_vars=entry_n_vars,
            )

            dataset_entries.append(entry)

        return cls(
            corpus_id=corpus_id,
            corpus_root=corpus_root,
            tokenizer=tokenizer,
            emission_spec=emission_spec,
            dataset_entries=tuple(dataset_entries),
            metadata_table=metadata_table,
        )


def build_corpus_loader(
    corpus_index_path: Path,
    backend: str = "arrow-hf",
    backend_kwargs: dict[str, Any] | None = None,
) -> "CorpusLoader":
    """Alias for ``CorpusLoader.from_corpus_index`` for discoverability."""
    return CorpusLoader.from_corpus_index(
        corpus_index_path=corpus_index_path,
        backend=backend,
        backend_kwargs=backend_kwargs,
    )
