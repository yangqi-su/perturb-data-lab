"""Bounded LRU shard cache for CSR memmap backend.

Phase 4 — optional job-local shard caching suitable for ``/tmp`` or node-local
scratch with bounded capacity and LRU eviction.

Usage::

    from perturb_data_lab.materializers.backends.csr_cache import ShardLRUCache

    cache = ShardLRUCache(
        cache_root=Path("/tmp/csr_cache"),
        max_bytes=20_000_000_000,   # 20 GB
    )

    # Get the local cached path for a shard (copied on first access):
    local_dir = cache.get_shard_path(shard_id=3, source_shard_path=source)

    # Use files from local_dir:
    np.load(str(local_dir / "row_offsets.npy"), mmap_mode="r")
    np.load(str(local_dir / "gene_indices.npy"), mmap_mode="r")
    np.load(str(local_dir / "counts.npy"), mmap_mode="r")
"""

from __future__ import annotations

import fcntl
import os
import shutil
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Files expected in each shard directory — only these are copied.
_SHARD_FILES: tuple[str, ...] = (
    "row_offsets.npy",
    "gene_indices.npy",
    "counts.npy",
    "shard-manifest.yaml",
)

# Protected symlink directories at the repo root — resolver must not write
# into or under these paths.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_PROTECTED_ROOTS: frozenset[Path] = frozenset(
    p.resolve()
    for p in [
        _REPO_ROOT / "data",
        _REPO_ROOT / "pertTF",
        _REPO_ROOT / "perturb",
    ]
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ShardFileError(OSError):
    """Raised when a shard source directory is missing or unreadable."""


class CacheCapacityError(ValueError):
    """Raised when a single shard exceeds the cache capacity."""


# ---------------------------------------------------------------------------
# ShardLRUCache
# ---------------------------------------------------------------------------


class ShardLRUCache:
    """Bounded LRU cache for CSR memmap shard files.

    Copies shard ``.npy`` files from a source directory to a local cache root
    with bounded overall capacity, evicting least-recently-used shards when
    the byte limit is exceeded.

    Read access is thread-safe.  Copy operations use per-shard ``fcntl.flock``
    to coordinate across multiple processes sharing the same cache root.

    Parameters
    ----------
    cache_root : Path
        Local scratch directory for cached shard copies.  Must not be inside
        any protected repository symlink directory (``data/``, ``pertTF/``,
        ``perturb/``).  Typical value: ``/tmp/csr_cache_<jobid>``.
    max_bytes : int
        Maximum total bytes of cached shard data across all cached shards.
        When a new copy would exceed this limit, least-recently-used shards
        are evicted before the copy proceeds.
    per_worker : bool
        If ``True``, each worker process (identified by ``os.getpid()``)
        gets its own subdirectory under *cache_root*.  This avoids
        cross-process copy contention without file locking.  Default
        ``False`` — lock-based coordination is used.

    Examples
    --------

    >>> cache = ShardLRUCache(Path("/tmp/test"), max_bytes=20_000)
    >>> local = cache.get_shard_path(0, Path("/data/shard_000000"))
    >>> assert local.exists()
    >>> assert cache.current_bytes > 0
    """

    def __init__(
        self,
        cache_root: Path,
        max_bytes: int,
        *,
        per_worker: bool = False,
    ):
        if max_bytes <= 0:
            raise CacheCapacityError(
                f"max_bytes must be positive, got {max_bytes}"
            )

        self._cache_root = Path(cache_root)
        self._max_bytes = max_bytes
        self._per_worker = per_worker

        # Resolve the effective cache root (per-worker if enabled)
        self._resolved_root = self._resolve_root()
        self._validate_root()

        # Per-shard tracking:  shard_id -> (shard_dir_path, size_bytes)
        self._entries: dict[int, tuple[Path, int]] = {}
        # LRU order: most-recently-used at end (OrderedDict insertion order)
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._current_bytes: int = 0

        # Phase 7: benchmark-visible hit/miss/eviction counters
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._eviction_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cache_root(self) -> Path:
        """The resolved cache root (may include per-worker subdirectory)."""
        return self._resolved_root

    @property
    def max_bytes(self) -> int:
        """Configured capacity in bytes."""
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        """Total bytes currently cached across all shards."""
        return self._current_bytes

    @property
    def cached_shard_ids(self) -> frozenset[int]:
        """Set of shard IDs currently held in the cache."""
        return frozenset(self._entries.keys())

    @property
    def hit_count(self) -> int:
        """Number of cache hits since construction."""
        return self._hit_count

    @property
    def miss_count(self) -> int:
        """Number of cache misses since construction."""
        return self._miss_count

    @property
    def eviction_count(self) -> int:
        """Number of shards evicted since construction."""
        return self._eviction_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_shard_path(
        self,
        shard_id: int,
        source_shard_path: Path,
    ) -> Path:
        """Return a local directory path containing a copy of the shard.

        If *shard_id* is already cached, its LRU position is refreshed
        and the existing path is returned immediately (cache hit).

        Otherwise the shard files are copied from *source_shard_path*
        into the cache root, evicting older shards if necessary to
        stay within *max_bytes* (cache miss).

        Parameters
        ----------
        shard_id : int
            Unique shard identifier within the corpus.
        source_shard_path : Path
            Source shard directory containing the ``.npy`` files and
            ``shard-manifest.yaml``.

        Returns
        -------
        Path
            Directory inside the cache root that now holds a copy of
            the shard files.

        Raises
        ------
        ShardFileError
            If *source_shard_path* is missing or contains no shard files.
        CacheCapacityError
            If the shard is larger than *max_bytes*.
        """
        if shard_id in self._entries:
            # -------------------------------------------------------
            # Cache hit — refresh LRU position and return cached path
            # -------------------------------------------------------
            self._hit_count += 1
            self._lru.move_to_end(shard_id)
            return self._entries[shard_id][0]

        # -----------------------------------------------------------
        # Cache miss — copy shard, possibly evicting first
        # -----------------------------------------------------------
        self._miss_count += 1
        return self._insert_shard(shard_id, source_shard_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_root(self) -> Path:
        """If per_worker is enabled, append pid-based subdirectory."""
        if self._per_worker:
            return self._cache_root / f"worker_{os.getpid()}"
        return self._cache_root

    def _validate_root(self) -> None:
        """Raise if the cache root is inside a protected symlink directory."""
        try:
            resolved = self._cache_root.resolve()
        except Exception:
            return  # unresolvable — skip check, user's responsibility

        for protected in _PROTECTED_ROOTS:
            try:
                resolved.relative_to(protected)
            except ValueError:
                continue  # not inside this root — OK
            raise CacheCapacityError(
                f"Cache root {str(self._cache_root)!r} resolves inside "
                f"protected directory {str(protected)!r}.  Cache output "
                f"must go to a repo-local non-symlink directory."
            )

    def _shard_size_bytes(self, source: Path) -> int:
        """Return the total size of shard files in *source*."""
        total = 0
        found_any = False
        for fname in _SHARD_FILES:
            fp = source / fname
            if fp.is_file():
                found_any = True
                total += fp.stat().st_size
        if not found_any:
            raise ShardFileError(f"No shard files found in {source}")
        return total

    def _evict_to_fit(self, needed_bytes: int) -> None:
        """Evict least-recently-used shards until *needed_bytes* can fit."""
        while (
            self._lru
            and self._current_bytes + needed_bytes > self._max_bytes
        ):
            lru_id, _ = self._lru.popitem(last=False)  # LRU = first item
            shard_dir, evict_bytes = self._entries.pop(lru_id)
            self._current_bytes -= evict_bytes
            self._eviction_count += 1

            if shard_dir.exists():
                shutil.rmtree(shard_dir)

    def _insert_shard(self, shard_id: int, source: Path) -> Path:
        """Copy shard files into the cache and record the new entry."""
        source = Path(source)
        if not source.is_dir():
            raise ShardFileError(
                f"Shard source directory does not exist: {source}"
            )

        shard_size = self._shard_size_bytes(source)

        if shard_size > self._max_bytes:
            raise CacheCapacityError(
                f"Shard {shard_id} size ({shard_size} bytes) exceeds "
                f"cache capacity ({self._max_bytes} bytes)"
            )

        # Evict until there's room
        self._evict_to_fit(shard_size)

        # Create target directory
        dest_dir = self._resolved_root / f"shard_{shard_id:06d}"
        self._resolved_root.mkdir(parents=True, exist_ok=True)

        if dest_dir.exists():
            # Stale directory from a previous run — replace atomically
            shutil.rmtree(dest_dir)
        dest_dir.mkdir()

        # Copy under per-shard file lock for cross-process safety
        lock_path = self._resolved_root / f".lock_shard_{shard_id:06d}"
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                for fname in _SHARD_FILES:
                    src_file = source / fname
                    if src_file.is_file():
                        shutil.copy2(src_file, dest_dir / fname)
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

        # Record the new entry
        self._entries[shard_id] = (dest_dir, shard_size)
        self._lru[shard_id] = None
        self._current_bytes += shard_size

        return dest_dir
