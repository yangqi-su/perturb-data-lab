"""Streaming expression statistics collected during materialization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chunk_translation import ChunkBundle


@dataclass
class DatasetStreamingStats:
    """Accumulate per-cell size-factor inputs and per-gene HVG inputs."""

    n_vars: int
    row_sums: list[np.ndarray] = field(default_factory=list)
    n_cells_total: int = 0

    def __post_init__(self) -> None:
        self.sum_log1p = np.zeros(self.n_vars, dtype=np.float64)
        self.sum_log1p_sq = np.zeros(self.n_vars, dtype=np.float64)

    def update(self, bundle: ChunkBundle) -> None:
        self.row_sums.append(bundle.row_sums)
        log1p_counts = np.log1p(bundle.counts.astype(np.float64))
        np.add.at(self.sum_log1p, bundle.indices, log1p_counts)
        np.add.at(self.sum_log1p_sq, bundle.indices, log1p_counts ** 2)
        self.n_cells_total += bundle.row_count

    def size_factors(self) -> np.ndarray:
        row_sums = np.concatenate(self.row_sums)
        median = float(np.median(row_sums))
        if median > 0:
            size_factors = row_sums / median
        else:
            size_factors = np.ones_like(row_sums)
        size_factors = np.where(size_factors <= 0, 1.0, size_factors)
        size_factors = np.where(np.isnan(size_factors), 1.0, size_factors)
        return size_factors.astype(np.float32)
