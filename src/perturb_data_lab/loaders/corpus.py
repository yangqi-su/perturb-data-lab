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
    ArrowHFCellReader,
    BackendCellReader,
    CellState,
    HVGRandomSampler,
    SamplerState,
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
    """

    dataset_id: str
    release_id: str
    manifest_path: Path
    reader: ArrowHFCellReader

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
        tokenizer: CorpusTokenizer,
        emission_spec: CorpusEmissionSpec,
        dataset_entries: tuple[DatasetReaderEntry, ...],
    ):
        """
        Parameters
        ----------
        corpus_id : str
            Corpus identifier.
        corpus_root : Path
            Root directory of the corpus (where corpus-index.yaml lives).
        tokenizer : CorpusTokenizer
            The corpus-level tokenizer (loaded once from corpus root).
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

        # Build dataset_id → entry index map for O(1) lookup
        self._dataset_id_to_idx: dict[str, int] = {
            entry.dataset_id: idx for idx, entry in enumerate(dataset_entries)
        }

        # Build per-dataset cumulative cell offsets for global index routing
        self._cumulative_offsets: list[int] = []
        offset = 0
        for entry in dataset_entries:
            self._cumulative_offsets.append(offset)
            offset += len(entry.reader)
        self._total_cells = offset

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

        Routing is O(log n_datasets) via binary search on cumulative offsets.
        The returned CellState carries dataset-order indices (not token IDs).
        """
        if corpus_global_index < 0 or corpus_global_index >= self._total_cells:
            raise IndexError(
                f"corpus_global_index {corpus_global_index} out of range [0, {self._total_cells})"
            )

        # Binary search for the dataset that owns this global index
        dataset_idx = self._find_dataset_idx(corpus_global_index)
        local_index = corpus_global_index - self._cumulative_offsets[dataset_idx]
        entry = self._dataset_entries[dataset_idx]
        return entry.reader.read_cell(local_index)

    def iter_cells(self):
        """Yield all cells across all datasets in corpus order.

        This is a convenience generator; for random access use ``read_cell``.
        """
        for entry in self._dataset_entries:
            reader = entry.reader
            for local_idx in range(len(reader)):
                yield reader.read_cell(local_idx)

    def _find_dataset_idx(self, global_index: int) -> int:
        """Binary search for the dataset index owning global_index."""
        lo, hi = 0, len(self._dataset_entries) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumulative_offsets[mid + 1] <= global_index:
                lo = mid + 1
            else:
                hi = mid
        return lo

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

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------

    @staticmethod
    def _reader_kwargs(
        manifest: "MaterializationManifest",
        meta_root: Path,
        backend: str,
    ) -> dict[str, Any]:
        """Build backend-specific kwargs for build_cell_reader.

        Each backend reader has a different constructor signature. This method
        bridges the manifest's declared paths to the reader's expected kwargs.
        """
        if backend == "arrow-hf":
            cell_meta_sqlite = meta_root / f"{manifest.release_id}-cell-meta.sqlite"
            cells_parquet = Path(manifest.outputs.matrix_root) / f"{manifest.release_id}-cells.parquet"
            meta_parquet = meta_root / f"{manifest.release_id}-meta.parquet"
            size_factor_manifest = meta_root / "size-factor-manifest.yaml"
            feature_meta_paths: dict[str, Path] | None = None
            if manifest.feature_meta_paths:
                feature_meta_paths = {
                    k: meta_root / v for k, v in manifest.feature_meta_paths.items()
                }
            return {
                "cells_parquet_path": cells_parquet,
                "meta_parquet_path": meta_parquet,
                "cell_meta_sqlite_path": cell_meta_sqlite,
                "feature_registry_path": meta_root / "feature-registry.yaml",  # legacy fallback
                "size_factor_manifest_path": size_factor_manifest,
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
                "dataset_id": manifest.dataset_id,
            }
        elif backend == "zarr-ts":
            indices_zarr = Path(manifest.outputs.matrix_root) / f"{manifest.release_id}-indices.zarr"
            counts_zarr = Path(manifest.outputs.matrix_root) / f"{manifest.release_id}-counts.zarr"
            sf_zarr = Path(manifest.outputs.matrix_root) / f"{manifest.release_id}-sf.zarr"
            meta_path = meta_root / f"{manifest.release_id}-meta.json"
            feature_registry_path = meta_root / "feature-registry.yaml"
            return {
                "indices_zarr_path": indices_zarr,
                "counts_zarr_path": counts_zarr,
                "sf_zarr_path": sf_zarr,
                "meta_path": meta_path,
                "feature_registry_path": feature_registry_path,
                "dataset_id": manifest.dataset_id,
            }
        else:
            raise ValueError(f"unknown backend: {backend}")

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

        # Load global metadata — backend field is the authoritative corpus declaration
        gmeta_path = corpus_root / "global-metadata.yaml"
        if gmeta_path.exists():
            gmeta = GlobalMetadataDocument.from_yaml_file(gmeta_path)
            corpus_backend = gmeta.backend
        else:
            # Legacy corpus without global-metadata: infer from first manifest
            # (backwards compat path; new corpora must have global-metadata with backend)
            if not corpus_index.datasets:
                raise ValueError("corpus index has no datasets")
            first_manifest_path = corpus_root / corpus_index.datasets[0].manifest_path
            if first_manifest_path.exists():
                first_manifest = MaterializationManifest.from_yaml_file(first_manifest_path)
                corpus_backend = first_manifest.backend
            else:
                raise FileNotFoundError(
                    f"corpus index references {first_manifest_path} which does not exist; "
                    "cannot determine corpus backend"
                )

        # Validate: all manifests must declare the same backend as global-metadata
        for ds_record in corpus_index.datasets:
            manifest_path = corpus_root / ds_record.manifest_path
            if manifest_path.exists():
                manifest = MaterializationManifest.from_yaml_file(manifest_path)
                if manifest.backend != corpus_backend:
                    raise ValueError(
                        f"backend mismatch: manifest {manifest_path} declares backend "
                        f"'{manifest.backend}' but corpus declares backend '{corpus_backend}' — "
                        "all datasets in a corpus must use the same backend"
                    )

        # Dispatch reader by corpus backend (not by the deprecated caller-provided backend arg)
        if corpus_backend not in AVAILABLE_READERS:
            raise NotImplementedError(
                f"backend '{corpus_backend}' not yet supported; "
                f"use one of {list(AVAILABLE_READERS)}"
            )

        # Load tokenizer (required for all backends)
        tokenizer_path_str = (
            gmeta.tokenizer_path if gmeta and gmeta.tokenizer_path else "tokenizer.json"
        )
        tokenizer_path = corpus_root / tokenizer_path_str
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
                release_id=manifest.release_id,
                corpus_index_path=corpus_index_path,
                **cls._reader_kwargs(manifest, meta_root, corpus_backend),
            )

            # Determine n_vars from features_origin parquet
            entry_n_vars = 0
            if manifest.feature_meta_paths and "features_origin" in manifest.feature_meta_paths:
                import pyarrow.parquet as pq
                origin_path = meta_root / manifest.feature_meta_paths["features_origin"]
                if origin_path.exists():
                    tbl = pq.read_table(str(origin_path))
                    entry_n_vars = tbl.num_rows

            entry = DatasetReaderEntry(
                dataset_id=manifest.dataset_id,
                release_id=manifest.release_id,
                manifest_path=manifest_path,
                reader=reader,
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