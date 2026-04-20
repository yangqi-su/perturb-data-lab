"""Phase 4 training-facing dataset and sampling layer.

This module implements:
- Common dataset interface exposing sparse indices, counts, metadata, size factors
- Backend-specific readers for Arrow/HF, WebDataset, and Zarr/TensorStore
- Shared sampler implementations: random context, expressed+zeros, HVGs+random
- Default streaming IterableDataset path and optional map-style path
- Minimal integration examples for external collators

All write operations go to repo-local real directories only.
Never write to protected symlink roots (data/, pertTF/, perturb/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CellState",
    "BackendCellReader",
    "ArrowHFCellReader",
    "WebDatasetCellReader",
    "ZarrCellReader",
    "build_cell_reader",
    "AVAILABLE_READERS",
    "RandomContextSampler",
    "ExpressedZerosSampler",
    "HVGRandomSampler",
    "PerturbDataLoader",
    "PerturbIterableDataset",
]


# ---------------------------------------------------------------------------
# Common cell state — what a sampler sees from any backend
# ---------------------------------------------------------------------------


@dataclass
class CellState:
    """The minimal per-cell state a sampler operates on.

    Backend-agnostic; returned by every reader regardless of storage format.
    """

    cell_id: str
    dataset_id: str
    dataset_release: str
    expressed_gene_indices: tuple[int, ...]
    expression_counts: tuple[int, ...]
    size_factor: float
    canonical_perturbation: dict[str, str]
    canonical_context: dict[str, str]
    raw_fields: dict[str, Any]


# ---------------------------------------------------------------------------
# Backend-agnostic cell reader interface
# ---------------------------------------------------------------------------


class BackendCellReader:
    """Abstract backend-agnostic cell reader.

    All concrete reader implementations must produce CellState records so
    sampler logic stays backend-agnostic.
    """

    def __init__(self, release_id: str, corpus_index_path: Path):
        self.release_id = release_id
        self.corpus_index_path = corpus_index_path

    def read_cell(self, cell_index: int) -> CellState:
        """Read a single cell by index position."""
        raise NotImplementedError

    def read_cells(self, cell_indices: list[int]) -> list[CellState]:
        """Read multiple cells by index position."""
        return [self.read_cell(i) for i in cell_indices]

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def total_genes(self) -> int:
        """Total number of features in the global vocab."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Arrow/HF reader
# ---------------------------------------------------------------------------


class ArrowHFCellReader(BackendCellReader):
    """Reader for Arrow/HF Parquet storage (primary backend from Phase 3)."""

    def __init__(
        self,
        release_id: str,
        corpus_index_path: Path,
        cells_parquet_path: Path,
        meta_parquet_path: Path,
        cell_meta_sqlite_path: Path,
        feature_registry_path: Path,
        size_factor_manifest_path: Path,
    ):
        super().__init__(release_id, corpus_index_path)
        self.cells_parquet_path = cells_parquet_path
        self.meta_parquet_path = meta_parquet_path
        self.cell_meta_sqlite_path = cell_meta_sqlite_path
        self.feature_registry_path = feature_registry_path
        self.size_factor_manifest_path = size_factor_manifest_path
        self.__cells_table = None
        self.__meta_table = None

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

    def __len__(self) -> int:
        return self._cells_table().num_rows

    @property
    def total_genes(self) -> int:
        from ..materializers.models import FeatureRegistryManifest

        reg = FeatureRegistryManifest.from_yaml_file(self.feature_registry_path)
        return len(reg.entries)

    def read_cell(self, cell_index: int) -> CellState:
        import pyarrow.parquet as pq

        import sqlite3

        table = self._cells_table()
        indices = table["expressed_gene_indices"][cell_index].as_py()
        counts = table["expression_counts"][cell_index].as_py()
        sf = table["size_factor"][cell_index].as_py()

        # Load cell metadata from SQLite
        conn = sqlite3.connect(str(self.cell_meta_sqlite_path))
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM cell_meta WHERE rowid = ?", (cell_index + 1,))
        row = cur.fetchone()
        conn.close()

        if row is None:
            raise IndexError(f"cell_index {cell_index} out of range")

        # Load canonical perturbation/context from manifest if available
        from ..materializers.models import (
            MaterializationManifest,
        )

        manifest_path = (
            Path(self.corpus_index_path).parent
            / "metadata"
            / self.release_id
            / "materialization-manifest.yaml"
        )
        canonical_perturbation = {}
        canonical_context = {}
        raw_fields = {}

        if manifest_path.exists():
            manifest = MaterializationManifest.from_yaml_file(manifest_path)
            # Load schema patch for canonical field values
            if manifest.provenance.schema_patch:
                from ..inspectors.models import SchemaPatchDocument

                patch = SchemaPatchDocument.from_yaml_file(
                    Path(manifest.provenance.schema_patch)
                )
                for p in patch.patches:
                    if p.field in (
                        "perturbation_label",
                        "perturbation_type",
                        "target_id",
                        "target_label",
                        "control_flag",
                        "dose",
                        "dose_unit",
                        "timepoint",
                        "timepoint_unit",
                        "combination_key",
                    ):
                        canonical_perturbation[p.field] = p.value or "NA"
                    elif p.field in (
                        "dataset_id",
                        "dataset_release",
                        "cell_context",
                        "cell_line_or_type",
                        "species",
                        "tissue",
                        "assay",
                        "condition",
                        "batch_id",
                        "donor_id",
                        "sex",
                        "disease_state",
                    ):
                        canonical_context[p.field] = p.value or "NA"

        return CellState(
            cell_id=str(row["cell_id"]),
            dataset_id=str(row["dataset_id"]),
            dataset_release=str(row["dataset_release"]),
            expressed_gene_indices=tuple(indices),
            expression_counts=tuple(counts),
            size_factor=float(sf),
            canonical_perturbation=canonical_perturbation,
            canonical_context=canonical_context,
            raw_fields=raw_fields,
        )


# ---------------------------------------------------------------------------
# WebDataset reader
# ---------------------------------------------------------------------------


class WebDatasetCellReader(BackendCellReader):
    """Reader for WebDataset tar-shard storage."""

    def __init__(
        self,
        release_id: str,
        corpus_index_path: Path,
        shard_paths: list[Path],
        meta_path: Path,
        feature_registry_path: Path | None = None,
    ):
        super().__init__(release_id, corpus_index_path)
        self.shard_paths = shard_paths
        self.meta_path = meta_path
        self.feature_registry_path = feature_registry_path
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

        return CellState(
            cell_id=record["cell_id"],
            dataset_id=self.release_id,  # WebDataset doesn't track dataset_id separately
            dataset_release=self.release_id,
            expressed_gene_indices=tuple(
                np.frombuffer(record["expressed_gene_indices"], dtype=np.int32)
            ),
            expression_counts=tuple(
                np.frombuffer(record["expression_counts"], dtype=np.int32)
            ),
            size_factor=record["size_factor"],
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )


# ---------------------------------------------------------------------------
# Zarr/TensorStore reader
# ---------------------------------------------------------------------------


class ZarrCellReader(BackendCellReader):
    """Reader for Zarr/TensorStore cell-chunked sparse storage."""

    def __init__(
        self,
        release_id: str,
        corpus_index_path: Path,
        indices_zarr_path: Path,
        counts_zarr_path: Path,
        sf_zarr_path: Path,
        meta_path: Path,
        chunk_cells: int = 1024,
        feature_registry_path: Path | None = None,
    ):
        super().__init__(release_id, corpus_index_path)
        self.indices_zarr_path = indices_zarr_path
        self.counts_zarr_path = counts_zarr_path
        self.sf_zarr_path = sf_zarr_path
        self.meta_path = meta_path
        self.chunk_cells = chunk_cells
        self.feature_registry_path = feature_registry_path
        self._n_cells: int | None = None
        self._n_vars: int | None = None

    def __len__(self) -> int:
        if self._n_cells is None:
            import json

            with open(self.meta_path) as f:
                meta = json.load(f)
            self._n_cells = meta["n_obs"]
            self._n_vars = meta["n_vars"]
        return self._n_cells

    @property
    def total_genes(self) -> int:
        """Total genes from feature registry when available, else from metadata."""
        if self.feature_registry_path is not None:
            from ..materializers.models import FeatureRegistryManifest

            reg = FeatureRegistryManifest.from_yaml_file(self.feature_registry_path)
            return len(reg.entries)
        # Fallback: use n_vars from Zarr metadata JSON
        if self._n_vars is None:
            import json

            with open(self.meta_path) as f:
                meta = json.load(f)
            self._n_vars = meta["n_vars"]
        return self._n_vars

    def read_cell(self, cell_index: int) -> CellState:
        import json

        import zarr

        with open(self.meta_path) as f:
            meta = json.load(f)

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

        return CellState(
            cell_id=f"{self.release_id}_cell_{cell_index}",
            dataset_id=self.release_id,
            dataset_release=self.release_id,
            expressed_gene_indices=expressed_indices,
            expression_counts=expressed_counts,
            size_factor=float(sf),
            canonical_perturbation={},
            canonical_context={},
            raw_fields={},
        )


# ---------------------------------------------------------------------------
# Reader factory
# ---------------------------------------------------------------------------


AVAILABLE_READERS = {
    "arrow-hf": ArrowHFCellReader,
    "webdataset": WebDatasetCellReader,
    "zarr-ts": ZarrCellReader,
}


def build_cell_reader(
    backend: str,
    release_id: str,
    corpus_index_path: Path,
    **backend_kwargs,
) -> BackendCellReader:
    """Factory to build the correct backend cell reader by name."""
    if backend not in AVAILABLE_READERS:
        raise ValueError(
            f"unknown backend reader: {backend}; available: {list(AVAILABLE_READERS)}"
        )
    return AVAILABLE_READERS[backend](release_id, corpus_index_path, **backend_kwargs)


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
    ):
        self.reader = reader
        self.shuffle = shuffle
        self.seed = seed
        self.context_size = context_size
        self.max_context = max_context
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

            yield {
                "cell_id": cell.cell_id,
                "dataset_id": cell.dataset_id,
                "dataset_release": cell.dataset_release,
                "expressed_gene_indices": np.array(
                    cell.expressed_gene_indices, dtype=np.int32
                ),
                "expression_counts": np.array(cell.expression_counts, dtype=np.int32),
                "context_indices": context,
                "size_factor": cell.size_factor,
                "canonical_perturbation": cell.canonical_perturbation,
                "canonical_context": cell.canonical_context,
            }

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
    ):
        self.reader = reader
        self.shuffle = shuffle
        self.seed = seed
        self.context_size = context_size
        self.max_context = max_context
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

        return {
            "cell_id": cell.cell_id,
            "dataset_id": cell.dataset_id,
            "dataset_release": cell.dataset_release,
            "expressed_gene_indices": np.array(
                cell.expressed_gene_indices, dtype=np.int32
            ),
            "expression_counts": np.array(cell.expression_counts, dtype=np.int32),
            "context_indices": context,
            "size_factor": cell.size_factor,
            "canonical_perturbation": cell.canonical_perturbation,
            "canonical_context": cell.canonical_context,
        }
