"""Phase 4/8 training-facing dataset and sampling layer.

This module implements:
- Common dataset interface exposing sparse indices, counts, metadata, size factors
- Backend-specific readers for Arrow/HF, WebDataset, and Zarr/TensorStore
- Shared sampler implementations: random context, expressed+zeros, HVGs+random
- Default streaming IterableDataset path and optional map-style path
- Minimal integration examples for external collators
- Feature preloading for efficient dataset-order → token-space index translation

All write operations go to repo-local real directories only.
Never write to protected symlink roots (data/, pertTF/, perturb/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "CellState",
    "CellIdentity",
    "DatasetContextKey",
    "MetadataTable",
    "PreloadedFeatureObjects",
    "BackendCellReader",
    "ArrowHFCellReader",
    "WebDatasetCellReader",
    "ZarrCellReader",
    "build_cell_reader",
    "AVAILABLE_READERS",
    "SparseBatchPayload",
    "ResolvedSparseBatch",
    "GlobalFeatureResolver",
    "SparseBatchCollator",
    "CorpusRandomBatchSampler",
    "DatasetBatchSampler",
    "DatasetContextBatchSampler",
    "CPUDenseRuntimePath",
    "GPUSparseRuntimePath",
    "RandomContextSampler",
    "ExpressedZerosSampler",
    "HVGRandomSampler",
    "PerturbDataLoader",
    "PerturbIterableDataset",
]


# ---------------------------------------------------------------------------
# Feature preloading — per-dataset feature objects kept in memory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreloadedFeatureObjects:
    """Preloaded per-dataset feature objects for a single dataset.

    These objects are small (one row per feature, not per cell) and are shared
    across all cells of the dataset.  Preloading them once at loader startup
    enables efficient dataset-order → token-space index translation during
    sample decoding.

    Two variants are stored:

    - origin:  the canonical feature metadata in original dataset var order.
               The `origin_index` column aligns with the sparse per-cell
               `expressed_gene_indices` values stored during materialization.
    - token:   a mapping from `origin_index` → `token_id` (int), enabling
               O(1) translation of dataset-order indices to global token IDs.

    Both variants are MaterializationRoute outputs written once per dataset
    during the materialization phase (Phase 7).
    """

    # Dataset this object belongs to
    dataset_id: str

    # Parquet path for the origin feature object (canonical metadata in var order)
    features_origin_path: Path

    # Parquet path for the token companion (origin_index → token_id mapping)
    features_token_path: Path

    # Lazily-loaded PyArrow tables (loaded once on first access)
    _origin_table: Any = field(default=None, repr=False)
    _token_table: Any = field(default=None, repr=False)

    def origin_table(self) -> Any:
        """Load and cache the origin feature parquet table."""
        if self._origin_table is None:
            import pyarrow.parquet as pq

            object.__setattr__(
                self, "_origin_table", pq.read_table(str(self.features_origin_path))
            )
        return self._origin_table

    def token_table(self) -> Any:
        """Load and cache the token companion parquet table."""
        if self._token_table is None:
            import pyarrow.parquet as pq

            object.__setattr__(
                self, "_token_table", pq.read_table(str(self.features_token_path))
            )
        return self._token_table

    def origin_index_to_token_id(self, origin_index: int) -> int:
        """Translate a dataset-order feature index to a global token ID.

        Uses the preloaded token parquet to look up the token_id for the
        given origin_index.  Returns -1 if the origin_index is out of range
        or not registered.
        """
        token_table = self.token_table()
        origin_col = token_table["origin_index"]
        # Binary search for the origin_index
        try:
            idx = origin_col.to_pylist().index(origin_index)
        except ValueError:
            return -1
        if idx < 0:
            return -1
        return token_table["token_id"][idx].as_py()

    def translate_indices(self, origin_indices: tuple[int, ...]) -> tuple[int, ...]:
        """Translate a tuple of dataset-order indices to token IDs.

        Convenience method for batch translation during sample decoding.
        Returns a tuple of the same length with each index translated.
        """
        return tuple(self.origin_index_to_token_id(int(i)) for i in origin_indices)


@dataclass(frozen=True)
class CellIdentity:
    """Runtime-facing row identity detached from release_id hot-path routing."""

    global_row_index: int
    dataset_index: int
    dataset_id: str
    local_row_index: int


@dataclass(frozen=True)
class MetadataTable:
    """RAM-resident metadata table keyed by global_row_index."""

    global_row_index: tuple[int, ...]
    cell_id: tuple[str, ...]
    dataset_id: tuple[str, ...]
    dataset_index: tuple[int, ...]
    local_row_index: tuple[int, ...]
    canonical_perturbation: tuple[dict[str, str], ...]
    canonical_context: tuple[dict[str, str], ...]
    raw_fields: tuple[dict[str, Any], ...]
    size_factor: tuple[float, ...]
    _dataset_row_index_cache: dict[int, tuple[int, ...]] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _dataset_context_row_index_cache: dict[
        str,
        dict["DatasetContextKey", tuple[int, ...]],
    ] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __len__(self) -> int:
        return len(self.global_row_index)

    def row(self, global_row_index: int) -> dict[str, Any]:
        if global_row_index < 0 or global_row_index >= len(self):
            raise IndexError(f"global_row_index {global_row_index} out of range")
        return {
            "global_row_index": int(self.global_row_index[global_row_index]),
            "cell_id": self.cell_id[global_row_index],
            "dataset_id": self.dataset_id[global_row_index],
            "dataset_index": int(self.dataset_index[global_row_index]),
            "local_row_index": int(self.local_row_index[global_row_index]),
            "canonical_perturbation": dict(self.canonical_perturbation[global_row_index]),
            "canonical_context": dict(self.canonical_context[global_row_index]),
            "raw_fields": dict(self.raw_fields[global_row_index]),
            "size_factor": float(self.size_factor[global_row_index]),
        }

    def rows(self, global_row_indices: list[int]) -> list[dict[str, Any]]:
        return [self.row(int(idx)) for idx in global_row_indices]

    def dataset_row_indices(self, dataset_index: int) -> tuple[int, ...]:
        cache = self._dataset_row_index_cache
        if cache is None:
            grouped: dict[int, list[int]] = {}
            for global_row_index, row_dataset_index in enumerate(self.dataset_index):
                grouped.setdefault(int(row_dataset_index), []).append(global_row_index)
            cache = {
                int(row_dataset_index): tuple(indices)
                for row_dataset_index, indices in grouped.items()
            }
            object.__setattr__(self, "_dataset_row_index_cache", cache)
        return cache.get(int(dataset_index), ())

    def dataset_context_keys(
        self,
        *,
        context_field: str = "cell_context",
        min_batch_size: int | None = None,
    ) -> list["DatasetContextKey"]:
        cache = self._dataset_context_rows(context_field)
        keys = sorted(
            cache,
            key=lambda item: (item.dataset_index, item.context_value, item.dataset_id),
        )
        if min_batch_size is None:
            return keys
        return [key for key in keys if len(cache[key]) >= int(min_batch_size)]

    def dataset_context_row_indices(
        self,
        key: "DatasetContextKey",
        *,
        context_field: str = "cell_context",
    ) -> tuple[int, ...]:
        return self._dataset_context_rows(context_field).get(key, ())

    def _dataset_context_rows(
        self,
        context_field: str,
    ) -> dict["DatasetContextKey", tuple[int, ...]]:
        caches = self._dataset_context_row_index_cache
        if caches is None:
            caches = {}
            object.__setattr__(self, "_dataset_context_row_index_cache", caches)
        if context_field in caches:
            return caches[context_field]

        grouped: dict[DatasetContextKey, list[int]] = {}
        for global_row_index, (dataset_index, dataset_id, canonical_context) in enumerate(
            zip(self.dataset_index, self.dataset_id, self.canonical_context)
        ):
            key = DatasetContextKey(
                dataset_index=int(dataset_index),
                dataset_id=str(dataset_id),
                context_value=str(canonical_context.get(context_field, "")),
            )
            grouped.setdefault(key, []).append(global_row_index)

        finalized = {key: tuple(indices) for key, indices in grouped.items()}
        caches[context_field] = finalized
        return finalized


@dataclass(frozen=True)
class DatasetContextKey:
    """Dataset-aware batch grouping key built from RAM metadata."""

    dataset_index: int
    dataset_id: str
    context_value: str


# ---------------------------------------------------------------------------
# Common cell state — what a sampler sees from any backend
# ---------------------------------------------------------------------------


@dataclass
class CellState:
    """The minimal per-cell state a sampler operates on.

    Backend-agnostic; returned by every reader regardless of storage format.
    """

    identity: CellIdentity
    cell_id: str
    expressed_gene_indices: tuple[int, ...]  # dataset-order indices
    expression_counts: tuple[int, ...]
    size_factor: float
    canonical_perturbation: dict[str, str]
    canonical_context: dict[str, str]
    raw_fields: dict[str, Any]

    @property
    def global_row_index(self) -> int:
        return self.identity.global_row_index

    @property
    def dataset_id(self) -> str:
        return self.identity.dataset_id

    @property
    def dataset_index(self) -> int:
        return self.identity.dataset_index

    @property
    def local_row_index(self) -> int:
        return self.identity.local_row_index


# ---------------------------------------------------------------------------
# Backend-agnostic cell reader interface
# ---------------------------------------------------------------------------


class BackendCellReader:
    """Abstract backend-agnostic cell reader.

    All concrete reader implementations must produce CellState records so
    sampler logic stays backend-agnostic.

    Concrete implementations must also implement ``preloaded_features``
    (returns PreloadedFeatureObjects | None) and ``translate_to_token_ids``
    (identity by default) so token-space translation is available uniformly.
    """

    def __init__(self, dataset_id: str, dataset_index: int, corpus_index_path: Path):
        self.dataset_id = dataset_id
        self.dataset_index = dataset_index
        self.corpus_index_path = corpus_index_path

    def read_cell(self, cell_index: int) -> CellState:
        """Read a single cell by index position."""
        raise NotImplementedError

    def read_cells(self, cell_indices: list[int]) -> list[CellState]:
        """Read multiple cells by index position."""
        return [self.read_cell(i) for i in cell_indices]

    def read_rows(self, row_indices: list[int]) -> list[dict[str, Any]]:
        """Read heavy payloads only for multiple local rows."""
        return [self.read_row(i) for i in row_indices]

    def read_row(self, row_index: int) -> dict[str, Any]:
        """Read heavy payload only for a single local row."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def total_genes(self) -> int:
        """Total number of features in the global vocab."""
        raise NotImplementedError

    @property
    def preloaded_features(self) -> "PreloadedFeatureObjects | None":
        """Preloaded feature objects for origin→token translation, or None if unavailable."""
        return None

    def translate_to_token_ids(
        self, origin_indices: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Translate dataset-order indices to corpus token IDs.

        Default implementation returns indices unchanged (identity translation).
        ArrowHFCellReader overrides this when preloaded_features are available.
        """
        return origin_indices


# ---------------------------------------------------------------------------
# Arrow/HF reader
# ---------------------------------------------------------------------------


class ArrowHFCellReader(BackendCellReader):
    """Reader for Arrow/HF Parquet storage (primary backend from Phase 3).

    Redesigned in Phase 8 to:
    - Preload per-dataset feature objects into memory for efficient
      dataset-order → token-space index translation during sample decoding
    - Load canonical cell metadata from the materialized SQLite file
      rather than re-executing schema mappings from the schema YAML
    """

    def __init__(
        self,
        dataset_id: str,
        dataset_index: int,
        corpus_index_path: Path,
        cells_parquet_path: Path,
        meta_parquet_path: Path,
        cell_meta_sqlite_path: Path,
        feature_registry_path: Path,
        metadata_table: MetadataTable | None = None,
        global_row_offset: int = 0,
        size_factor_manifest_path: Path | None = None,
        feature_meta_paths: dict[str, Path] | None = None,
    ):
        """
        Parameters
        ----------
        dataset_id : str
            Stable dataset identifier for runtime routing.
        dataset_index : int
            Corpus-level dataset index for runtime routing.
        corpus_index_path : Path
            Path to the corpus index YAML file.
        cells_parquet_path : Path
            Path to the cells parquet file.
        meta_parquet_path : Path
            Path to the metadata parquet file.
        cell_meta_sqlite_path : Path
            Path to the SQLite file containing full canonical cell metadata
            (written by Phase 6 `_write_cell_metadata`).
        feature_registry_path : Path
            Path to the feature registry YAML.
        size_factor_manifest_path : Path | None
            Deprecated; size factors are read from the cells parquet.
            No longer required or used. Kept for backwards compatibility.
        feature_meta_paths : dict[str, Path] | None
            Optional dict with keys ``features_origin`` and optionally ``features_token``
            pointing to the per-dataset feature parquet files written by
            Phase 7 `_write_feature_metadata`.  When ``features_origin`` is provided,
            feature objects are preloaded at reader construction for origin-index
            translation.  ``features_token`` is optional — if absent, the reader
            falls back to identity translation (original dataset-order indices as tokens).
            This allows Phase 3 (tokenizer-deferred) materializations to load without
            requiring a token-sidecar that has not yet been generated.
        """
        super().__init__(dataset_id, dataset_index, corpus_index_path)
        self.cells_parquet_path = cells_parquet_path
        self.meta_parquet_path = meta_parquet_path
        self.cell_meta_sqlite_path = cell_meta_sqlite_path
        self.feature_registry_path = feature_registry_path
        self.metadata_table = metadata_table
        self.global_row_offset = global_row_offset
        self.size_factor_manifest_path = size_factor_manifest_path
        self._feature_meta_paths = feature_meta_paths
        self.__cells_table = None
        self.__meta_table = None
        # Preload feature objects only when both features_origin AND features_token
        # are available.  When features_token is absent (Phase 3 tokenizer-deferred),
        # token-space translation is not yet possible — translate_to_token_ids
        # falls back to identity translation.
        _has_token = (
            feature_meta_paths is not None
            and "features_origin" in feature_meta_paths
            and "features_token" in feature_meta_paths
        )
        self.__preloaded_features: PreloadedFeatureObjects | None = (
            PreloadedFeatureObjects(
                dataset_id=dataset_id,
                features_origin_path=feature_meta_paths["features_origin"],
                features_token_path=feature_meta_paths["features_token"],
            )
            if _has_token
            else None
        )

    def _cells_table(self):
        if self.__cells_table is None:
            import pyarrow.parquet as pq

            self.__cells_table = pq.read_table(self.cells_parquet_path)
        return self.__cells_table

    def _meta_table(self):
        if self.__meta_table is None:
            import pyarrow.parquet as pq

            self.__meta_table = pq.read_table(self.meta_parquet_path)
        return self.__meta_table

    @property
    def preloaded_features(self) -> PreloadedFeatureObjects | None:
        """Return the preloaded feature objects for this dataset, if any."""
        return self.__preloaded_features

    def translate_to_token_ids(
        self, origin_indices: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Translate dataset-order indices to global token IDs using preloaded objects.

        If feature_meta_paths were provided at construction, this method uses
        the preloaded token parquet to translate each dataset-order index to its
        global token ID.  If no preloaded objects are available, the original
        indices are returned unchanged (identity translation).
        """
        if self.__preloaded_features is None:
            return origin_indices
        return self.__preloaded_features.translate_indices(origin_indices)

    def __len__(self) -> int:
        return self._cells_table().num_rows

    def read_row(self, row_index: int) -> dict[str, Any]:
        table = self._cells_table()
        indices = table["expressed_gene_indices"][row_index].as_py()
        counts = table["expression_counts"][row_index].as_py()
        sf = table["size_factor"][row_index].as_py()
        return {
            "global_row_index": self.global_row_offset + row_index,
            "dataset_index": self.dataset_index,
            "dataset_id": self.dataset_id,
            "local_row_index": row_index,
            "expressed_gene_indices": tuple(int(i) for i in indices),
            "expression_counts": tuple(int(c) for c in counts),
            "size_factor": float(sf),
        }

    def read_rows(self, row_indices: list[int]) -> list[dict[str, Any]]:
        if not row_indices:
            return []
        import pyarrow as pa

        table = self._cells_table().take(pa.array(row_indices, type=pa.int64()))
        result: list[dict[str, Any]] = []
        for local_position, row_index in enumerate(row_indices):
            result.append(
                {
                    "global_row_index": self.global_row_offset + row_index,
                    "dataset_index": self.dataset_index,
                    "dataset_id": self.dataset_id,
                    "local_row_index": row_index,
                    "expressed_gene_indices": tuple(
                        int(i) for i in table["expressed_gene_indices"][local_position].as_py()
                    ),
                    "expression_counts": tuple(
                        int(c) for c in table["expression_counts"][local_position].as_py()
                    ),
                    "size_factor": float(table["size_factor"][local_position].as_py()),
                }
            )
        return result

    @property
    def total_genes(self) -> int:
        from ..materializers.models import FeatureRegistryManifest

        reg = FeatureRegistryManifest.from_yaml_file(self.feature_registry_path)
        return len(reg.entries)

    def read_cell(self, cell_index: int) -> CellState:
        import json
        import sqlite3
        heavy_row = self.read_row(cell_index)
        global_row_index = heavy_row["global_row_index"]

        if self.metadata_table is not None:
            metadata = self.metadata_table.row(global_row_index)
            return CellState(
                identity=CellIdentity(
                    global_row_index=global_row_index,
                    dataset_index=int(metadata["dataset_index"]),
                    dataset_id=str(metadata["dataset_id"]),
                    local_row_index=int(metadata["local_row_index"]),
                ),
                cell_id=str(metadata["cell_id"]),
                expressed_gene_indices=tuple(heavy_row["expressed_gene_indices"]),
                expression_counts=tuple(heavy_row["expression_counts"]),
                size_factor=float(metadata["size_factor"]),
                canonical_perturbation=dict(metadata["canonical_perturbation"]),
                canonical_context=dict(metadata["canonical_context"]),
                raw_fields=dict(metadata["raw_fields"]),
            )

        # Load full canonical cell metadata from the materialized SQLite file.
        # This was written by _write_cell_metadata() during Phase 6 materialization
        # and contains all canonical perturbation and context fields resolved
        # row-wise from the schema at materialization time.  No schema re-execution
        # is needed at load time.
        conn = sqlite3.connect(str(self.cell_meta_sqlite_path))
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM cell_meta WHERE rowid = ?", (cell_index + 1,))
        metadata_row = cur.fetchone()
        conn.close()

        if metadata_row is None:
            raise IndexError(f"cell_index {cell_index} out of range")

        canonical_perturbation = json.loads(metadata_row["canonical_perturbation"])
        canonical_context = json.loads(metadata_row["canonical_context"])
        raw_fields = json.loads(metadata_row["raw_obs"])

        return CellState(
            identity=CellIdentity(
                global_row_index=global_row_index,
                dataset_index=self.dataset_index,
                dataset_id=str(metadata_row["dataset_id"]),
                local_row_index=cell_index,
            ),
            cell_id=str(metadata_row["cell_id"]),
            expressed_gene_indices=tuple(heavy_row["expressed_gene_indices"]),
            expression_counts=tuple(heavy_row["expression_counts"]),
            size_factor=float(heavy_row["size_factor"]),
            canonical_perturbation=canonical_perturbation,
            canonical_context=canonical_context,
            raw_fields=raw_fields,
        )


class LanceDBAggregatedCellReader(ArrowHFCellReader):
    """Temporary Phase 5 aggregated-Lance reader using Arrow/HF storage semantics."""


# ---------------------------------------------------------------------------
# WebDataset reader
# ---------------------------------------------------------------------------


class WebDatasetCellReader(BackendCellReader):
    """Reader for WebDataset tar-shard storage."""

    def __init__(
        self,
        dataset_id: str,
        dataset_index: int,
        corpus_index_path: Path,
        shard_paths: list[Path],
        meta_path: Path,
        feature_registry_path: Path | None = None,
        metadata_table: MetadataTable | None = None,
        global_row_offset: int = 0,
    ):
        super().__init__(dataset_id, dataset_index, corpus_index_path)
        self.shard_paths = shard_paths
        self.meta_path = meta_path
        self.feature_registry_path = feature_registry_path
        self.metadata_table = metadata_table
        self.global_row_offset = global_row_offset
        self._cell_index_to_shard: dict[int, tuple[int, int]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Build a flat index mapping global cell index to (shard_idx, local_idx)."""
        import pickle
        import tarfile

        cell_idx = 0
        for shard_idx, shard_path in enumerate(self.shard_paths):
            with tarfile.open(str(shard_path), "r") as tar:
                for member in tar.getmembers():
                    if member.name.startswith("cell_") and member.name.endswith(".pt"):
                        self._cell_index_to_shard[cell_idx] = (
                            shard_idx,
                            int(member.name.split("_")[1].split(".")[0]),
                        )
                        cell_idx += 1
        self._n_cells = cell_idx

    def __len__(self) -> int:
        return self._n_cells

    @property
    def total_genes(self) -> int:
        """Total genes from feature registry when available, else from metadata."""
        if self.feature_registry_path is not None:
            from ..materializers.models import FeatureRegistryManifest

            reg = FeatureRegistryManifest.from_yaml_file(self.feature_registry_path)
            return len(reg.entries)
        # Fallback: infer from max index seen during _build_index
        max_idx = 0
        for cell_idx in range(self._n_cells):
            _, local_idx = self._cell_index_to_shard[cell_idx]
            if local_idx > max_idx:
                max_idx = local_idx
        return max_idx + 1

    def read_cell(self, cell_index: int) -> CellState:
        row = self.read_row(cell_index)
        global_row_index = row["global_row_index"]
        if self.metadata_table is not None:
            metadata = self.metadata_table.row(global_row_index)
            return CellState(
                identity=CellIdentity(
                    global_row_index=global_row_index,
                    dataset_index=int(metadata["dataset_index"]),
                    dataset_id=str(metadata["dataset_id"]),
                    local_row_index=int(metadata["local_row_index"]),
                ),
                cell_id=str(metadata["cell_id"]),
                expressed_gene_indices=tuple(row["expressed_gene_indices"]),
                expression_counts=tuple(row["expression_counts"]),
                size_factor=float(metadata["size_factor"]),
                canonical_perturbation=dict(metadata["canonical_perturbation"]),
                canonical_context=dict(metadata["canonical_context"]),
                raw_fields=dict(metadata["raw_fields"]),
            )

        return CellState(
            identity=CellIdentity(
                global_row_index=global_row_index,
                dataset_index=self.dataset_index,
                dataset_id=self.dataset_id,
                local_row_index=cell_index,
            ),
            cell_id=str(row["cell_id"]),
            expressed_gene_indices=tuple(row["expressed_gene_indices"]),
            expression_counts=tuple(row["expression_counts"]),
            size_factor=float(row["size_factor"]),
            canonical_perturbation=dict(row["canonical_perturbation"]),
            canonical_context=dict(row["canonical_context"]),
            raw_fields=dict(row["raw_fields"]),
        )

    def read_row(self, cell_index: int) -> dict[str, Any]:
        import pickle
        import tarfile

        shard_idx, local_idx = self._cell_index_to_shard[cell_index]
        shard_path = self.shard_paths[shard_idx]

        with tarfile.open(str(shard_path), "r") as tar:
            member = tar.getmember(f"cell_{local_idx:08d}.pt")
            f = tar.extractfile(member)
            if f is None:
                raise RuntimeError(f"Failed to extract {member.name}")
            record = pickle.loads(f.read())

        return {
            "global_row_index": self.global_row_offset + cell_index,
            "dataset_index": self.dataset_index,
            "dataset_id": self.dataset_id,
            "local_row_index": cell_index,
            "cell_id": record["cell_id"],
            "expressed_gene_indices": tuple(
                np.frombuffer(record["expressed_gene_indices"], dtype=np.int32)
            ),
            "expression_counts": tuple(
                np.frombuffer(record["expression_counts"], dtype=np.int32)
            ),
            "size_factor": float(record["size_factor"]),
            "canonical_perturbation": dict(record.get("canonical_perturbation", {})),
            "canonical_context": dict(record.get("canonical_context", {})),
            "raw_fields": dict(record.get("raw_fields", {})),
        }


# ---------------------------------------------------------------------------
# Zarr/TensorStore reader
# ---------------------------------------------------------------------------


class ZarrCellReader(BackendCellReader):
    """Reader for Zarr/TensorStore cell-chunked sparse storage."""

    def __init__(
        self,
        dataset_id: str,
        dataset_index: int,
        corpus_index_path: Path,
        indices_zarr_path: Path,
        counts_zarr_path: Path,
        sf_zarr_path: Path,
        meta_path: Path,
        chunk_cells: int = 1024,
        feature_registry_path: Path | None = None,
        feature_meta_paths: dict[str, Path] | None = None,
        metadata_table: MetadataTable | None = None,
        global_row_offset: int = 0,
    ):
        super().__init__(dataset_id, dataset_index, corpus_index_path)
        self.indices_zarr_path = indices_zarr_path
        self.counts_zarr_path = counts_zarr_path
        self.sf_zarr_path = sf_zarr_path
        self.meta_path = meta_path
        self.chunk_cells = chunk_cells
        self.feature_registry_path = feature_registry_path
        self._feature_meta_paths = feature_meta_paths
        self.metadata_table = metadata_table
        self.global_row_offset = global_row_offset
        self._n_cells: int | None = None
        self._n_vars: int | None = None
        self._meta_cache: list[dict[str, Any]] | None = None  # per-cell meta list
        self.__preloaded_features: PreloadedFeatureObjects | None = None
        if (
            feature_meta_paths is not None
            and "features_origin" in feature_meta_paths
            and "features_token" in feature_meta_paths
        ):
            self.__preloaded_features = PreloadedFeatureObjects(
                dataset_id=dataset_id,
                features_origin_path=feature_meta_paths["features_origin"],
                features_token_path=feature_meta_paths["features_token"],
            )

    def _load_meta(self) -> list[dict[str, Any]]:
        """Load and cache the per-cell meta list from meta.json once."""
        if self._meta_cache is None:
            import json
            with open(self.meta_path) as f:
                meta = json.load(f)
            self._meta_cache = meta["cells"]
            self._n_cells = meta["n_obs"]
            self._n_vars = meta["n_vars"]
        return self._meta_cache

    def __len__(self) -> int:
        meta = self._load_meta()
        return len(meta)

    @property
    def total_genes(self) -> int:
        """Total genes from feature registry when available, else from metadata."""
        if self.feature_registry_path is not None:
            from ..materializers.models import FeatureRegistryManifest

            reg = FeatureRegistryManifest.from_yaml_file(self.feature_registry_path)
            return len(reg.entries)
        # Fallback: use cached n_vars from meta.json
        if self._n_vars is None:
            self._load_meta()
        return self._n_vars or 0

    @property
    def preloaded_features(self) -> PreloadedFeatureObjects | None:
        return self.__preloaded_features

    def translate_to_token_ids(
        self, origin_indices: tuple[int, ...]
    ) -> tuple[int, ...]:
        if self.__preloaded_features is None:
            return origin_indices
        return self.__preloaded_features.translate_indices(origin_indices)

    def read_cell(self, cell_index: int) -> CellState:
        row = self.read_row(cell_index)
        global_row_index = row["global_row_index"]
        if self.metadata_table is not None:
            metadata = self.metadata_table.row(global_row_index)
            return CellState(
                identity=CellIdentity(
                    global_row_index=global_row_index,
                    dataset_index=int(metadata["dataset_index"]),
                    dataset_id=str(metadata["dataset_id"]),
                    local_row_index=int(metadata["local_row_index"]),
                ),
                cell_id=str(metadata["cell_id"]),
                expressed_gene_indices=tuple(row["expressed_gene_indices"]),
                expression_counts=tuple(row["expression_counts"]),
                size_factor=float(metadata["size_factor"]),
                canonical_perturbation=dict(metadata["canonical_perturbation"]),
                canonical_context=dict(metadata["canonical_context"]),
                raw_fields=dict(metadata["raw_fields"]),
            )

        return CellState(
            identity=CellIdentity(
                global_row_index=global_row_index,
                dataset_index=self.dataset_index,
                dataset_id=self.dataset_id,
                local_row_index=cell_index,
            ),
            cell_id=str(row["cell_id"]),
            expressed_gene_indices=tuple(row["expressed_gene_indices"]),
            expression_counts=tuple(row["expression_counts"]),
            size_factor=float(row["size_factor"]),
            canonical_perturbation=dict(row["canonical_perturbation"]),
            canonical_context=dict(row["canonical_context"]),
            raw_fields=dict(row["raw_fields"]),
        )

    def read_row(self, cell_index: int) -> dict[str, Any]:
        import zarr

        meta_list = self._load_meta()
        cell_meta = meta_list[cell_index]

        chunk_idx = cell_index // self.chunk_cells
        local_i = cell_index % self.chunk_cells

        indices_store = zarr.open(str(self.indices_zarr_path), mode="r")
        counts_store = zarr.open(str(self.counts_zarr_path), mode="r")
        sf_store = zarr.open(str(self.sf_zarr_path), mode="r")

        chunk_indices = indices_store[f"chunk_{chunk_idx}"][local_i]
        chunk_counts = counts_store[f"chunk_{chunk_idx}"][local_i]
        sf = sf_store["sf"][cell_index]

        nonzero_mask = chunk_indices != -1
        expressed_indices = tuple(chunk_indices[nonzero_mask].tolist())
        expressed_counts = tuple(chunk_counts[nonzero_mask].tolist())

        return {
            "global_row_index": self.global_row_offset + cell_index,
            "dataset_index": self.dataset_index,
            "dataset_id": self.dataset_id,
            "local_row_index": cell_index,
            "cell_id": cell_meta["cell_id"],
            "expressed_gene_indices": expressed_indices,
            "expression_counts": expressed_counts,
            "size_factor": float(sf),
            "canonical_perturbation": dict(cell_meta.get("canonical_perturbation", {})),
            "canonical_context": dict(cell_meta.get("canonical_context", {})),
            "raw_fields": dict(cell_meta.get("raw_fields", {})),
        }


# ---------------------------------------------------------------------------
# Reader factory
# ---------------------------------------------------------------------------


AVAILABLE_READERS = {
    "arrow-hf": ArrowHFCellReader,
    "webdataset": WebDatasetCellReader,
    "zarr-ts": ZarrCellReader,
    "lancedb-aggregated": LanceDBAggregatedCellReader,
    "zarr-aggregated": ZarrCellReader,
}


def build_cell_reader(
    backend: str,
    dataset_id: str,
    dataset_index: int,
    corpus_index_path: Path,
    **backend_kwargs,
) -> BackendCellReader:
    """Factory to build the correct backend cell reader by name."""
    if backend not in AVAILABLE_READERS:
        raise ValueError(
            f"unknown backend reader: {backend}; available: {list(AVAILABLE_READERS)}"
        )
    return AVAILABLE_READERS[backend](dataset_id, dataset_index, corpus_index_path, **backend_kwargs)


# ---------------------------------------------------------------------------
# Dataset-aware batch helpers and runtime-path payloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseBatchPayload:
    """Flat sparse payload plus offsets for runtime-hot-path batch handling."""

    batch_size: int
    global_row_index: np.ndarray
    dataset_index: np.ndarray
    local_row_index: np.ndarray
    row_offsets: np.ndarray
    expressed_gene_indices: np.ndarray
    expression_counts: np.ndarray
    size_factor: np.ndarray
    dataset_id: tuple[str, ...]
    cell_id: tuple[str, ...]
    canonical_perturbation: tuple[dict[str, str], ...]
    canonical_context: tuple[dict[str, str], ...]

    def row_slice(self, row_position: int) -> slice:
        start = int(self.row_offsets[row_position])
        stop = int(self.row_offsets[row_position + 1])
        return slice(start, stop)

    def row_gene_indices(self, row_position: int) -> np.ndarray:
        return self.expressed_gene_indices[self.row_slice(row_position)]

    def row_counts(self, row_position: int) -> np.ndarray:
        return self.expression_counts[self.row_slice(row_position)]


@dataclass(frozen=True)
class ResolvedSparseBatch:
    """Sparse batch resolved into global feature identity without densification."""

    batch_size: int
    global_row_index: np.ndarray
    dataset_index: np.ndarray
    local_row_index: np.ndarray
    row_offsets: np.ndarray
    global_feature_ids: np.ndarray
    expression_counts: np.ndarray
    size_factor: np.ndarray
    dataset_id: tuple[str, ...]
    cell_id: tuple[str, ...]
    canonical_perturbation: tuple[dict[str, str], ...]
    canonical_context: tuple[dict[str, str], ...]
    unresolved_local_features: int = 0

    def row_slice(self, row_position: int) -> slice:
        start = int(self.row_offsets[row_position])
        stop = int(self.row_offsets[row_position + 1])
        return slice(start, stop)

    def row_feature_ids(self, row_position: int) -> np.ndarray:
        return self.global_feature_ids[self.row_slice(row_position)]

    def row_counts(self, row_position: int) -> np.ndarray:
        return self.expression_counts[self.row_slice(row_position)]


class SparseBatchCollator:
    """Collate `CellState` rows into flat sparse payloads plus offsets."""

    def __call__(self, cells: Sequence[CellState]) -> SparseBatchPayload:
        row_offsets = [0]
        flat_gene_indices: list[np.ndarray] = []
        flat_counts: list[np.ndarray] = []
        for cell in cells:
            gene_indices = np.asarray(cell.expressed_gene_indices, dtype=np.int32)
            counts = np.asarray(cell.expression_counts, dtype=np.int32)
            if gene_indices.shape != counts.shape:
                raise ValueError("cell sparse payload has mismatched gene/count lengths")
            flat_gene_indices.append(gene_indices)
            flat_counts.append(counts)
            row_offsets.append(row_offsets[-1] + int(gene_indices.size))

        return SparseBatchPayload(
            batch_size=len(cells),
            global_row_index=np.asarray([cell.global_row_index for cell in cells], dtype=np.int64),
            dataset_index=np.asarray([cell.dataset_index for cell in cells], dtype=np.int32),
            local_row_index=np.asarray([cell.local_row_index for cell in cells], dtype=np.int64),
            row_offsets=np.asarray(row_offsets, dtype=np.int64),
            expressed_gene_indices=(
                np.concatenate(flat_gene_indices).astype(np.int32, copy=False)
                if flat_gene_indices
                else np.asarray([], dtype=np.int32)
            ),
            expression_counts=(
                np.concatenate(flat_counts).astype(np.int32, copy=False)
                if flat_counts
                else np.asarray([], dtype=np.int32)
            ),
            size_factor=np.asarray([cell.size_factor for cell in cells], dtype=np.float32),
            dataset_id=tuple(cell.dataset_id for cell in cells),
            cell_id=tuple(cell.cell_id for cell in cells),
            canonical_perturbation=tuple(dict(cell.canonical_perturbation) for cell in cells),
            canonical_context=tuple(dict(cell.canonical_context) for cell in cells),
        )


@dataclass(frozen=True)
class GlobalFeatureResolver:
    """Post-canonicalization resolver from dataset-local indices to global ids."""

    dataset_feature_mappings: dict[int, np.ndarray]
    total_features: int
    unknown_feature_id: int = -1

    @classmethod
    def from_dataset_mappings(
        cls,
        dataset_feature_mappings: dict[int, Sequence[int] | np.ndarray],
        *,
        total_features: int | None = None,
        unknown_feature_id: int = -1,
    ) -> "GlobalFeatureResolver":
        normalized: dict[int, np.ndarray] = {}
        inferred_max = -1
        for dataset_index, mapping in dataset_feature_mappings.items():
            array = np.asarray(mapping, dtype=np.int32)
            normalized[int(dataset_index)] = array
            valid = array[array >= 0]
            if valid.size:
                inferred_max = max(inferred_max, int(valid.max()))
        if total_features is None:
            total_features = inferred_max + 1 if inferred_max >= 0 else 0
        return cls(
            dataset_feature_mappings=normalized,
            total_features=int(total_features),
            unknown_feature_id=int(unknown_feature_id),
        )

    def resolve_local_indices(
        self,
        dataset_index: int,
        local_feature_indices: Sequence[int] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if int(dataset_index) not in self.dataset_feature_mappings:
            raise KeyError(f"no global feature mapping registered for dataset_index {dataset_index}")

        mapping = self.dataset_feature_mappings[int(dataset_index)]
        local_indices = np.asarray(local_feature_indices, dtype=np.int64)
        resolved = np.full(local_indices.shape, self.unknown_feature_id, dtype=np.int32)
        within_bounds = (local_indices >= 0) & (local_indices < mapping.shape[0])
        if np.any(within_bounds):
            resolved[within_bounds] = mapping[local_indices[within_bounds]]
        valid = within_bounds & (resolved != self.unknown_feature_id)
        return resolved, valid

    def resolve_batch(
        self,
        payload: SparseBatchPayload,
        *,
        sort_by_global_feature: bool = True,
        drop_unresolved: bool = True,
    ) -> ResolvedSparseBatch:
        row_offsets = [0]
        resolved_gene_ids: list[np.ndarray] = []
        resolved_counts: list[np.ndarray] = []
        unresolved_total = 0

        for row_position in range(payload.batch_size):
            dataset_index = int(payload.dataset_index[row_position])
            local_indices = payload.row_gene_indices(row_position)
            counts = payload.row_counts(row_position)
            resolved_ids, valid_mask = self.resolve_local_indices(dataset_index, local_indices)
            unresolved_total += int(valid_mask.size - valid_mask.sum())

            if drop_unresolved:
                resolved_ids = resolved_ids[valid_mask]
                counts = counts[valid_mask]

            if sort_by_global_feature and resolved_ids.size:
                order = np.argsort(resolved_ids, kind="stable")
                resolved_ids = resolved_ids[order]
                counts = counts[order]

            resolved_gene_ids.append(resolved_ids.astype(np.int32, copy=False))
            resolved_counts.append(counts.astype(np.int32, copy=False))
            row_offsets.append(row_offsets[-1] + int(resolved_ids.size))

        return ResolvedSparseBatch(
            batch_size=payload.batch_size,
            global_row_index=payload.global_row_index.copy(),
            dataset_index=payload.dataset_index.copy(),
            local_row_index=payload.local_row_index.copy(),
            row_offsets=np.asarray(row_offsets, dtype=np.int64),
            global_feature_ids=(
                np.concatenate(resolved_gene_ids).astype(np.int32, copy=False)
                if resolved_gene_ids
                else np.asarray([], dtype=np.int32)
            ),
            expression_counts=(
                np.concatenate(resolved_counts).astype(np.int32, copy=False)
                if resolved_counts
                else np.asarray([], dtype=np.int32)
            ),
            size_factor=payload.size_factor.copy(),
            dataset_id=tuple(payload.dataset_id),
            cell_id=tuple(payload.cell_id),
            canonical_perturbation=tuple(dict(item) for item in payload.canonical_perturbation),
            canonical_context=tuple(dict(item) for item in payload.canonical_context),
            unresolved_local_features=unresolved_total,
        )


class CorpusRandomBatchSampler:
    """Yield shuffled corpus-global batches keyed by `global_row_index`."""

    def __init__(
        self,
        *,
        total_rows: int,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if total_rows <= 0:
            raise ValueError("total_rows must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.total_rows = int(total_rows)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_rows // self.batch_size
        return (self.total_rows + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.permutation(self.total_rows)
        for start in range(0, self.total_rows, self.batch_size):
            batch = indices[start : start + self.batch_size].tolist()
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield [int(idx) for idx in batch]


class DatasetBatchSampler:
    """Yield corpus-global batches restricted to a single dataset."""

    def __init__(
        self,
        *,
        metadata_table: MetadataTable,
        dataset_index: int,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.metadata_table = metadata_table
        self.dataset_index = int(dataset_index)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._row_indices = np.asarray(
            metadata_table.dataset_row_indices(self.dataset_index),
            dtype=np.int64,
        )
        if self._row_indices.size == 0:
            raise ValueError(f"dataset_index {self.dataset_index} has no rows")

    def __len__(self) -> int:
        if self.drop_last:
            return int(self._row_indices.size // self.batch_size)
        return int((self._row_indices.size + self.batch_size - 1) // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        row_indices = self._row_indices.copy()
        if self.shuffle:
            row_indices = row_indices[rng.permutation(row_indices.size)]
        for start in range(0, row_indices.size, self.batch_size):
            batch = row_indices[start : start + self.batch_size]
            if batch.size < self.batch_size and self.drop_last:
                continue
            yield batch.astype(np.int64, copy=False).tolist()


class DatasetContextBatchSampler:
    """Yield dataset-aware grouped batches from the RAM metadata table."""

    def __init__(
        self,
        *,
        metadata_table: MetadataTable,
        batch_size: int,
        context_field: str = "cell_context",
        dataset_index: int | None = None,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.metadata_table = metadata_table
        self.batch_size = int(batch_size)
        self.context_field = context_field
        self.dataset_index = int(dataset_index) if dataset_index is not None else None
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        keys = metadata_table.dataset_context_keys(
            context_field=context_field,
            min_batch_size=batch_size if drop_last else None,
        )
        if self.dataset_index is not None:
            keys = [key for key in keys if key.dataset_index == self.dataset_index]
        self._eligible_keys = tuple(keys)
        if not self._eligible_keys:
            raise ValueError("no dataset/context groups can support the requested batch size")

    def __len__(self) -> int:
        total_batches = 0
        for key in self._eligible_keys:
            row_count = len(
                self.metadata_table.dataset_context_row_indices(
                    key,
                    context_field=self.context_field,
                )
            )
            if self.drop_last:
                total_batches += row_count // self.batch_size
            else:
                total_batches += (row_count + self.batch_size - 1) // self.batch_size
        return total_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        keys = list(self._eligible_keys)
        if self.shuffle:
            rng.shuffle(keys)
        for key in keys:
            row_indices = np.asarray(
                self.metadata_table.dataset_context_row_indices(
                    key,
                    context_field=self.context_field,
                ),
                dtype=np.int64,
            )
            if self.shuffle and row_indices.size:
                row_indices = row_indices[rng.permutation(row_indices.size)]
            for start in range(0, row_indices.size, self.batch_size):
                batch = row_indices[start : start + self.batch_size]
                if batch.size < self.batch_size and self.drop_last:
                    continue
                batches.append(batch.astype(np.int64, copy=False).tolist())
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


class CPUDenseRuntimePath:
    """Baseline runtime path that resolves sparse rows, then densifies on CPU."""

    def __init__(self, resolver: GlobalFeatureResolver, *, total_features: int | None = None):
        self.resolver = resolver
        self.total_features = int(
            resolver.total_features if total_features is None else total_features
        )
        if self.total_features <= 0:
            raise ValueError("CPUDenseRuntimePath requires a positive total_features")

    def resolve_batch(self, payload: SparseBatchPayload) -> ResolvedSparseBatch:
        return self.resolver.resolve_batch(payload)

    def densify(self, batch: SparseBatchPayload | ResolvedSparseBatch) -> dict[str, Any]:
        resolved = batch if isinstance(batch, ResolvedSparseBatch) else self.resolve_batch(batch)
        dense_counts = np.zeros((resolved.batch_size, self.total_features), dtype=np.float32)
        for row_position in range(resolved.batch_size):
            feature_ids = resolved.row_feature_ids(row_position)
            if feature_ids.size == 0:
                continue
            counts = resolved.row_counts(row_position).astype(np.float32, copy=False)
            np.add.at(dense_counts[row_position], feature_ids, counts)
        return {
            "batch_size": resolved.batch_size,
            "global_row_index": resolved.global_row_index.copy(),
            "dataset_index": resolved.dataset_index.copy(),
            "local_row_index": resolved.local_row_index.copy(),
            "size_factor": resolved.size_factor.copy(),
            "dense_counts": dense_counts,
            "dataset_id": tuple(resolved.dataset_id),
            "cell_id": tuple(resolved.cell_id),
            "canonical_perturbation": tuple(dict(item) for item in resolved.canonical_perturbation),
            "canonical_context": tuple(dict(item) for item in resolved.canonical_context),
            "unresolved_local_features": resolved.unresolved_local_features,
        }

    def gather_sampled_counts(
        self,
        batch: SparseBatchPayload | ResolvedSparseBatch,
        sampled_feature_ids: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        dense_batch = self.densify(batch)
        dense_counts = dense_batch["dense_counts"]
        sampled = np.asarray(sampled_feature_ids, dtype=np.int64)
        if sampled.ndim == 1:
            sampled = np.broadcast_to(sampled, (dense_counts.shape[0], sampled.shape[0]))
        if sampled.shape[0] != dense_counts.shape[0]:
            raise ValueError("sampled_feature_ids batch dimension does not match dense batch")
        return dense_counts[np.arange(dense_counts.shape[0])[:, None], sampled]


class GPUSparseRuntimePath:
    """Sparse runtime path that keeps flat payloads plus offsets in the hot path."""

    def __init__(self, resolver: GlobalFeatureResolver):
        self.resolver = resolver

    def resolve_batch(self, payload: SparseBatchPayload) -> ResolvedSparseBatch:
        return self.resolver.resolve_batch(payload)

    def gather_sampled_counts(
        self,
        batch: SparseBatchPayload | ResolvedSparseBatch,
        sampled_feature_ids: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        resolved = batch if isinstance(batch, ResolvedSparseBatch) else self.resolve_batch(batch)
        sampled = np.asarray(sampled_feature_ids, dtype=np.int64)
        if sampled.ndim == 1:
            sampled = np.broadcast_to(sampled, (resolved.batch_size, sampled.shape[0]))
        if sampled.shape[0] != resolved.batch_size:
            raise ValueError("sampled_feature_ids batch dimension does not match sparse batch")

        gathered = np.zeros(sampled.shape, dtype=np.float32)
        for row_position in range(resolved.batch_size):
            row_feature_ids = resolved.row_feature_ids(row_position)
            if row_feature_ids.size == 0:
                continue
            row_counts = resolved.row_counts(row_position).astype(np.float32, copy=False)
            row_targets = sampled[row_position]
            positions = np.searchsorted(row_feature_ids, row_targets, side="left")
            in_bounds = positions < row_feature_ids.size
            if not np.any(in_bounds):
                continue
            clamped = np.clip(positions, 0, row_feature_ids.size - 1)
            exact = in_bounds & (row_feature_ids[clamped] == row_targets)
            if np.any(exact):
                gathered[row_position, exact] = row_counts[clamped[exact]]
        return gathered


# ---------------------------------------------------------------------------
# Sampler state — shared across all sampler modes
# ---------------------------------------------------------------------------


@dataclass
class SamplerState:
    """Per-sampler mutable state for tracking sampling decisions."""

    mode: str  # random_context | expressed_zeros | hvg_random
    total_cells: int
    n_genes: int
    expressed_threshold: int = 1  # minimum count to be considered "expressed"
    hvg_set: tuple[int, ...] = field(default_factory=tuple)  # HVG token IDs

    def __post_init__(self) -> None:
        if self.mode not in {"random_context", "expressed_zeros", "hvg_random"}:
            raise ValueError(f"unknown sampler mode: {self.mode}")


# ---------------------------------------------------------------------------
# Random Context Sampler
# ---------------------------------------------------------------------------


class RandomContextSampler:
    """Sampler: selects a random gene context of fixed size per cell.

    Produces a fixed-size gene subset uniformly at random from the full vocab,
    regardless of whether those genes are expressed. Used for baseline context
    training where the model learns to predict random missing genes.
    """

    def __init__(self, state: SamplerState, rng: np.random.Generator):
        self.state = state
        self.rng = rng

    def sample_indices(self, cell: CellState, context_size: int) -> np.ndarray:
        """Return a random context of context_size gene indices."""
        if context_size > self.state.n_genes:
            context_size = self.state.n_genes
        return self.rng.choice(
            self.state.n_genes, size=context_size, replace=False
        ).astype(np.int32)

    def sample_batch(
        self, cell_indices: list[int], reader: BackendCellReader, context_size: int
    ) -> list[tuple[CellState, np.ndarray]]:
        """Sample a batch of (cell, random_context) pairs."""
        cells = reader.read_cells(cell_indices)
        return [(cell, self.sample_indices(cell, context_size)) for cell in cells]


# ---------------------------------------------------------------------------
# Expressed + Zeros Sampler
# ---------------------------------------------------------------------------


class ExpressedZerosSampler:
    """Sampler: selects expressed genes + an equal number of zero genes.

    Produces a mixed context with all expressed genes plus an equal count of
    randomly sampled unexpressed genes. Used for training on expressed+
    zero context so the model learns both signal and silence.
    """

    def __init__(self, state: SamplerState, rng: np.random.Generator):
        self.state = state
        self.rng = rng

    def sample_indices(
        self, cell: CellState, max_context: int | None = None
    ) -> np.ndarray:
        """Return expressed + equal zeros, capped at max_context."""
        expressed = set(cell.expressed_gene_indices)
        n_expressed = len(expressed)
        max_zeros = (
            (max_context - n_expressed) // 2
            if max_context
            else self.state.n_genes - n_expressed
        )
        n_zeros = min(max_zeros, self.state.n_genes - n_expressed)
        zero_candidates = list(set(range(self.state.n_genes)) - expressed)
        zero_indices = self.rng.choice(zero_candidates, size=n_zeros, replace=False)
        context = np.array(list(expressed) + list(zero_indices), dtype=np.int32)
        return context

    def sample_batch(
        self,
        cell_indices: list[int],
        reader: BackendCellReader,
        max_context: int | None = None,
    ) -> list[tuple[CellState, np.ndarray]]:
        """Sample a batch of (cell, expressed+zeros context) pairs."""
        cells = reader.read_cells(cell_indices)
        return [(cell, self.sample_indices(cell, max_context)) for cell in cells]


# ---------------------------------------------------------------------------
# HVGs + Random Sampler
# ---------------------------------------------------------------------------


class HVGRandomSampler:
    """Sampler: selects HVG genes + an equal number of random non-HVG genes.

    Produces a mixed context with highly variable genes plus randomly sampled
    non-HVG genes. Used for focusing training on variable genes while keeping
    a baseline representation of other genes.
    """

    def __init__(self, state: SamplerState, rng: np.random.Generator):
        self.state = state
        self.rng = rng

    def sample_indices(
        self, cell: CellState, max_context: int | None = None
    ) -> np.ndarray:
        """Return HVG + equal random non-HVG, capped at max_context."""
        hvg_set = set(self.state.hvg_set)
        expressed = set(cell.expressed_gene_indices)
        hvg_expressed = hvg_set & expressed
        n_hvg = len(hvg_expressed)
        max_nonhvg = (
            (max_context - n_hvg) // 2 if max_context else self.state.n_genes - n_hvg
        )
        nonhvg_candidates = list(set(range(self.state.n_genes)) - hvg_set)
        n_nonhvg = min(max_nonhvg, len(nonhvg_candidates))
        nonhvg_indices = self.rng.choice(
            nonhvg_candidates, size=n_nonhvg, replace=False
        )
        context = np.array(list(hvg_expressed) + list(nonhvg_indices), dtype=np.int32)
        return context

    def sample_batch(
        self,
        cell_indices: list[int],
        reader: BackendCellReader,
        max_context: int | None = None,
    ) -> list[tuple[CellState, np.ndarray]]:
        """Sample a batch of (cell, HVG+random context) pairs."""
        cells = reader.read_cells(cell_indices)
        return [(cell, self.sample_indices(cell, max_context)) for cell in cells]


# ---------------------------------------------------------------------------
# Streaming IterableDataset
# ---------------------------------------------------------------------------


class PerturbIterableDataset:
    """PyTorch-friendly IterableDataset wrapping any BackendCellReader.

    This is the default streaming path. It reads cells from a backend reader
    and yields (sparse_indices, sparse_counts, metadata_dict) tuples.

    Collators are expected to handle padding and masking externally.

    Phase 8 adds optional token-space index translation: if the reader has
    preloaded feature objects (via feature_meta_paths), the yielded dict can
    include ``token_gene_indices`` — global token IDs translated from the
    stored dataset-order indices using the preloaded token parquet.
    """

    def __init__(
        self,
        reader: BackendCellReader,
        sampler_mode: str = "random_context",
        shuffle: bool = True,
        seed: int = 42,
        context_size: int | None = None,
        max_context: int | None = None,
        hvg_set: tuple[int, ...] = (),
        translate_indices: bool = False,
    ):
        """
        Parameters
        ----------
        translate_indices : bool
            When True, translate stored dataset-order indices to global token
            IDs using the reader's preloaded feature objects.  The yielded
            dict will include ``token_gene_indices`` alongside the original
            ``expressed_gene_indices``.  Default False to preserve the
            original behaviour for existing callers.
        """
        self.reader = reader
        self.shuffle = shuffle
        self.seed = seed
        self.context_size = context_size
        self.max_context = max_context
        self.translate_indices = translate_indices
        self.rng = np.random.default_rng(seed)

        state = SamplerState(
            mode=sampler_mode,
            total_cells=len(reader),
            n_genes=reader.total_genes,
            hvg_set=hvg_set,
        )

        if sampler_mode == "random_context":
            self.sampler = RandomContextSampler(state, self.rng)
        elif sampler_mode == "expressed_zeros":
            self.sampler = ExpressedZerosSampler(state, self.rng)
        elif sampler_mode == "hvg_random":
            self.sampler = HVGRandomSampler(state, self.rng)
        else:
            raise ValueError(f"unknown sampler mode: {sampler_mode}")

        self._indices: list[int] | None = None

    def _ensure_indices(self) -> None:
        if self._indices is None:
            self._indices = list(range(len(self.reader)))
            if self.shuffle:
                self.rng.shuffle(self._indices)

    def __iter__(self):
        self._ensure_indices()
        for cell_idx in self._indices:
            cell = self.reader.read_cell(cell_idx)
            if self.context_size is not None:
                context = self.sampler.sample_indices(cell, self.context_size)
            elif self.max_context is not None:
                context = self.sampler.sample_indices(cell, self.max_context)
            else:
                context = np.array(cell.expressed_gene_indices, dtype=np.int32)

            result = {
                "cell_id": cell.cell_id,
                "dataset_id": cell.dataset_id,
                "dataset_index": cell.dataset_index,
                "global_row_index": cell.global_row_index,
                "local_row_index": cell.local_row_index,
                "expressed_gene_indices": np.array(
                    cell.expressed_gene_indices, dtype=np.int32
                ),
                "expression_counts": np.array(cell.expression_counts, dtype=np.int32),
                "context_indices": context,
                "size_factor": cell.size_factor,
                "canonical_perturbation": cell.canonical_perturbation,
                "canonical_context": cell.canonical_context,
            }
            if self.translate_indices and hasattr(self.reader, "translate_to_token_ids"):
                result["token_gene_indices"] = np.array(
                    self.reader.translate_to_token_ids(cell.expressed_gene_indices),
                    dtype=np.int32,
                )
            yield result

    def __len__(self) -> int:
        return len(self.reader)


# ---------------------------------------------------------------------------
# Optional map-style dataset
# ---------------------------------------------------------------------------


class PerturbDataLoader:
    """Map-style dataset wrapper for backends that support indexed random access.

    Use this when the backend supports efficient random reads (e.g., Arrow/HF).
    WebDataset (sequential tar shards) and Zarr/TensorStore (chunked sparse) are
    better suited for the streaming IterableDataset path.

    Phase 8 adds optional token-space index translation: if the reader has
    preloaded feature objects, ``__getitem__`` can include ``token_gene_indices``
    translated from stored dataset-order indices.
    """

    def __init__(
        self,
        reader: BackendCellReader,
        sampler_mode: str = "random_context",
        shuffle: bool = False,
        seed: int = 42,
        context_size: int | None = None,
        max_context: int | None = None,
        hvg_set: tuple[int, ...] = (),
        translate_indices: bool = False,
    ):
        """
        Parameters
        ----------
        translate_indices : bool
            When True, translate stored dataset-order indices to global token
            IDs using the reader's preloaded feature objects.  The returned
            dict will include ``token_gene_indices`` alongside the original
            ``expressed_gene_indices``.  Default False to preserve the
            original behaviour for existing callers.
        """
        self.reader = reader
        self.shuffle = shuffle
        self.seed = seed
        self.context_size = context_size
        self.max_context = max_context
        self.translate_indices = translate_indices
        self.rng = np.random.default_rng(seed)

        state = SamplerState(
            mode=sampler_mode,
            total_cells=len(reader),
            n_genes=reader.total_genes,
            hvg_set=hvg_set,
        )

        if sampler_mode == "random_context":
            self.sampler = RandomContextSampler(state, self.rng)
        elif sampler_mode == "expressed_zeros":
            self.sampler = ExpressedZerosSampler(state, self.rng)
        elif sampler_mode == "hvg_random":
            self.sampler = HVGRandomSampler(state, self.rng)
        else:
            raise ValueError(f"unknown sampler mode: {sampler_mode}")

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cell = self.reader.read_cell(idx)
        if self.context_size is not None:
            context = self.sampler.sample_indices(cell, self.context_size)
        elif self.max_context is not None:
            context = self.sampler.sample_indices(cell, self.max_context)
        else:
            context = np.array(cell.expressed_gene_indices, dtype=np.int32)

        result = {
            "cell_id": cell.cell_id,
            "dataset_id": cell.dataset_id,
            "dataset_index": cell.dataset_index,
            "global_row_index": cell.global_row_index,
            "local_row_index": cell.local_row_index,
            "expressed_gene_indices": np.array(
                cell.expressed_gene_indices, dtype=np.int32
            ),
            "expression_counts": np.array(cell.expression_counts, dtype=np.int32),
            "context_indices": context,
            "size_factor": cell.size_factor,
            "canonical_perturbation": cell.canonical_perturbation,
            "canonical_context": cell.canonical_context,
        }
        if self.translate_indices and hasattr(self.reader, "translate_to_token_ids"):
            result["token_gene_indices"] = np.array(
                self.reader.translate_to_token_ids(cell.expressed_gene_indices),
                dtype=np.int32,
            )
        return result
