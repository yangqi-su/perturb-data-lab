"""Phase 2 tests: count recovery and feature identity logic.

Tests cover:
- Reverse-normalization recovery via expm1/size_factor path
- CountSourceDecision.uses_recovery flag
- Bin-named sources are included in both direct and recovery checks
- Largest-passing-source selection with direct-vs-recovered tie-breaking
- Source-order tie-breaking when size and pass mode are equal

New contract (Phase 2):
- Every source receives both direct integer check and reverse-normalized check
- Bin-named sources are NOT excluded from recovery attempts
- Among passing sources, largest feature dimension wins
- Direct integer beats recovered when otherwise tied
- Existing source order resolves remaining ties
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from perturb_data_lab.inspectors.models import (
    CountSourceDecision,
    CountSourceSpec,
)
from perturb_data_lab.inspectors.workflow import (
    _attempt_reverse_normalization,
    _audit_matrix_candidate,
    _choose_count_source,
    _rank_candidates,
    inspect_target,
    InspectionBatchConfig,
    InspectionTarget,
)


# ---------------------------------------------------------------------------
# Reverse-normalization recovery tests
# ---------------------------------------------------------------------------


def test_reverse_normalization_recovers_counts_from_plain_lognorm_layer():
    """Integer counts can be recovered from a plain log1p layer (global_sf = 1).

    When the forward transform is simply log1p(counts) with no size factor
    normalization (global_sf = 1), recovery via expm1(expr) * sf correctly
    recovers the original counts since sf_computed = 1.0 in this case.
    """
    obs = pd.DataFrame(index=["cell-1", "cell-2", "cell-3"])
    var = pd.DataFrame(index=["gene-a", "gene-b", "gene-c"])
    raw_counts = np.array([[1, 0, 3], [0, 2, 0], [4, 0, 1]], dtype=np.int32)

    # Plain log1p: global_sf = 1 acts as no-op in the division
    log_layer = np.log1p(raw_counts).astype(np.float32)
    adata = ad.AnnData(X=csr_matrix(raw_counts), obs=obs, var=var)
    adata.layers["log_norm"] = log_layer

    candidate = _audit_matrix_candidate(
        ".layers[log_norm]",
        adata.layers["log_norm"],
        row_count=3,
        column_count=3,
        sample_rows=3,
    )

    recovered = _attempt_reverse_normalization(candidate, adata)

    assert recovered is not None
    assert recovered.recovery_policy == "expm1_over_size_factor"
    assert recovered.inferred_transform == "recovered-count"
    assert recovered.max_abs_integer_deviation < 0.01


def test_reverse_normalization_not_attempted_on_direct_integer():
    """Reverse normalization is not attempted when the best candidate is already integer."""
    obs = pd.DataFrame(index=["cell-1", "cell-2"])
    var = pd.DataFrame(index=["gene-a", "gene-b"])
    counts = csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int32))
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    candidate = _audit_matrix_candidate(".X", adata.X, row_count=2, column_count=2, sample_rows=2)

    assert candidate.status == "pass"
    recovered = _attempt_reverse_normalization(candidate, adata)
    # No recovery needed — returns None (caller checks candidate.status first)
    assert recovered is None


def test_binned_source_included_in_recovery_attempts():
    """Bin-named sources are included in both direct and recovery checks.

    Under the new contract, bin-named sources are NOT excluded from recovery.
    Recovery is attempted on all non-pass candidates including bin-named ones.
    The values [0.7, 0.35] per row in a 2-column sparse matrix fail recovery
    because expm1 ratios produce max_abs_integer_deviation ~0.42 > 0.01.
    The single-nonzero-per-row pattern used previously accidentally passes
    recovery (each single nonzero becomes 1.0 by definition); the multi-nonzero
    pattern is genuinely non-recoverable.
    """
    obs = pd.DataFrame(index=["cell-1", "cell-2"])
    var = pd.DataFrame(index=["gene-a", "gene-b"])
    # Row 0: [0.7, 0.35], Row 1: [0.3, 0.85] — both rows have two nonzeros
    # with expm1 ratios that fail the integer threshold
    binned = csr_matrix(np.array([[0.7, 0.35], [0.3, 0.85]], dtype=np.float32))
    adata = ad.AnnData(X=np.array([[1, 0], [0, 1]], dtype=np.int32), obs=obs, var=var)
    adata.layers["binned"] = binned

    # Candidate name contains "bin" so _infer_transform_family returns "binned"
    candidate = _audit_matrix_candidate(
        ".layers[binned]",
        adata.layers["binned"],
        row_count=2,
        column_count=2,
        sample_rows=2,
    )

    # Direct check: binned fails (non-integer)
    assert candidate.status == "fail"
    assert candidate.inferred_transform == "binned"
    assert candidate.recovery_policy == "disallowed"

    # Recovery is attempted but fails for non-log-norm data
    recovered = _attempt_reverse_normalization(candidate, adata)
    assert recovered is None


def test_choose_count_source_prefers_largest_passing_source():
    """Among passing sources, the one with the largest feature dimension wins.

    When .raw.X has more features than .X and both pass (or recover), .raw.X wins.
    """
    obs = pd.DataFrame(index=["cell-1", "cell-2", "cell-3"])
    # .X: 3 features
    raw_counts = np.array([[1, 0, 3], [0, 2, 0], [4, 0, 1]], dtype=np.int32)
    adata = ad.AnnData(X=csr_matrix(raw_counts), obs=obs, var=pd.DataFrame(index=["g-a", "g-b", "g-c"]))
    adata.layers["log_norm"] = np.log1p(raw_counts).astype(np.float32)

    # Set up .raw.X with 4 features (larger)
    adata.raw = ad.AnnData(
        X=csr_matrix(np.array([
            [1, 0, 3, 2],
            [0, 2, 0, 1],
            [4, 0, 1, 0],
        ], dtype=np.int32)),
        var=pd.DataFrame(index=["g-a", "g-b", "g-c", "g-d"]),
    )

    candidates = _rank_candidates([
        _audit_matrix_candidate(".X", adata.X, 3, 3, 3),
        _audit_matrix_candidate(".raw.X", adata.raw.X, 3, 4, 3),
        _audit_matrix_candidate(".layers[log_norm]", adata.layers["log_norm"], 3, 3, 3),
    ])

    decision = _choose_count_source(candidates, adata)

    # .raw.X has 4 features (largest) and is integer direct → wins
    assert decision.selected_candidate == ".raw.X"
    assert decision.uses_recovery is False


def test_choose_count_source_direct_integer_beats_recovered():
    """When two sources are equally sized, direct integer beats recovered.

    An integer .X should be selected over a log-norm layer that requires recovery,
    even when both have the same feature count.
    """
    obs = pd.DataFrame(index=["cell-1", "cell-2", "cell-3"])
    var = pd.DataFrame(index=["gene-a", "gene-b", "gene-c"])
    raw_counts = np.array([[1, 0, 3], [0, 2, 0], [4, 0, 1]], dtype=np.int32)
    adata = ad.AnnData(X=csr_matrix(raw_counts), obs=obs, var=var)
    adata.layers["log_norm"] = np.log1p(raw_counts).astype(np.float32)

    candidates = _rank_candidates([
        _audit_matrix_candidate(".layers[log_norm]", adata.layers["log_norm"], 3, 3, 3),
        _audit_matrix_candidate(".X", adata.X, 3, 3, 3),
    ])

    decision = _choose_count_source(candidates, adata)

    # .X is integer direct, log_norm requires recovery → .X wins (same size, direct beats recovered)
    assert decision.selected_candidate == ".X"
    assert decision.uses_recovery is False
    assert decision.status == "pass"


def test_choose_count_source_source_order_resolves_tie():
    """When size and pass mode are equal, existing source order breaks the tie.

    The source_order map is built from the ranked candidate list, with lower indices
    assigned to higher-ranked candidates. When two candidates have equal key tuples
    after (-features, pass_mode), the one with the lower source_order index wins.

    This test uses two layers with identical naming patterns so they receive the
    same _source_priority score; their relative order in the ranked list therefore
    reflects the source-order tie-break. The layer that appears first in the ranked
    list (same score) wins because it has the lower source_order index.
    """
    obs = pd.DataFrame(index=["cell-1", "cell-2", "cell-3"])
    var = pd.DataFrame(index=["gene-a", "gene-b", "gene-c"])
    raw_counts = np.array([[1, 0, 3], [0, 2, 0], [4, 0, 1]], dtype=np.int32)
    adata = ad.AnnData(X=np.zeros((3, 3), dtype=np.int32), obs=obs, var=var)
    # Both layers have identical naming pattern → same priority score
    adata.layers["norm_a"] = np.log1p(raw_counts).astype(np.float32)
    adata.layers["norm_b"] = np.log1p(raw_counts).astype(np.float32)

    # Both are non-integer so both need recovery; since they have identical
    # scores they get the same rank. The one appearing first in the ranked list
    # (norm_a) will have the lower source_order index and should win.
    candidates = [
        _audit_matrix_candidate(".layers[norm_a]", adata.layers["norm_a"], 3, 3, 3),
        _audit_matrix_candidate(".layers[norm_b]", adata.layers["norm_b"], 3, 3, 3),
    ]
    ranked = _rank_candidates(candidates)
    decision = _choose_count_source(ranked, adata)

    # Both have same size (3 features), both would be recovered (or fail),
    # but the tie-break via source_order should prefer the first in ranked list.
    # Since both have identical priority scores and same status, source_order
    # (built from ranked list order) determines the winner.
    assert decision.selected_candidate == ".layers[norm_a]"


def test_choose_count_source_log_norm_recovers():
    """When only log-normalized source exists and it can be recovered, uses_recovery=True.

    Under the new contract, every source gets both checks. If .X is log-norm and
    passes reverse-normalized recovery, uses_recovery is True.
    """
    obs = pd.DataFrame(index=["cell-1", "cell-2", "cell-3"])
    var = pd.DataFrame(index=["gene-a", "gene-b", "gene-c"])
    # Log-normalized float in .X with no integer source anywhere
    log_norm = np.log1p(np.array([[1, 0, 3], [0, 2, 0], [4, 0, 1]], dtype=np.float32))
    adata = ad.AnnData(X=log_norm, obs=obs, var=var)

    candidates = _rank_candidates([
        _audit_matrix_candidate(".X", adata.X, 3, 3, 3)
    ])

    decision = _choose_count_source(candidates, adata)

    # No integer source found; recovery succeeds on log-norm data
    assert decision.uses_recovery is True
    assert decision.status == "pass"


def test_choose_count_source_genuinely_non_recoverable_fails():
    """When every source fails both checks, decision status is 'fail'.

    Truly non-recoverable data (e.g., arbitrary floats that are not log-normalized
    counts) should fail recovery. Since single-nonzero-per-row matrices can
    accidentally pass recovery (each row's single nonzero becomes 1.0), we use
    rows with multiple nonzero values whose expm1 ratios are not near-integer,
    which fails the integer-deviation threshold.
    """
    obs = pd.DataFrame(index=["cell-1", "cell-2"])
    var = pd.DataFrame(index=["gene-a", "gene-b", "gene-c"])
    # Row 0: [0.1, 0.35] expm1 ratios not near-integer → fails recovery
    # Row 1: [0.1, 0.3] expm1 ratios not near-integer → fails recovery
    adata = ad.AnnData(
        X=np.array([[0.1, 0.35, 0.0], [0.0, 0.1, 0.3]], dtype=np.float32),
        obs=obs,
        var=var,
    )

    candidates = _rank_candidates([
        _audit_matrix_candidate(".X", adata.X, 2, 3, 2),
    ])

    decision = _choose_count_source(candidates, adata)

    # No integer source found; recovery attempted but fails (non-integer deviations)
    assert decision.status == "fail"


# ---------------------------------------------------------------------------
# CountSourceDecision.uses_recovery field tests
# ---------------------------------------------------------------------------


def test_count_source_decision_from_dict_with_uses_recovery():
    """CountSourceDecision.from_dict accepts the optional uses_recovery field."""
    data = {
        "selected_candidate": ".X",
        "status": "pass",
        "confidence": "high",
        "recovery_policy": "not-needed",
        "rationale": "direct integer source",
        "uses_recovery": True,
    }
    decision = CountSourceDecision.from_dict(data)
    assert decision.uses_recovery is True


def test_count_source_decision_from_dict_without_uses_recovery():
    """CountSourceDecision.from_dict defaults uses_recovery to False."""
    data = {
        "selected_candidate": ".X",
        "status": "pass",
        "confidence": "high",
        "recovery_policy": "not-needed",
        "rationale": "direct integer source",
    }
    decision = CountSourceDecision.from_dict(data)
    assert decision.uses_recovery is False


# ---------------------------------------------------------------------------
# Binned matrix recovery disallowed test
# ---------------------------------------------------------------------------


def test_binned_candidate_recovery_policy_is_disallowed():
    """Binned candidate has recovery_policy='disallowed' under the new contract.

    Bin-named sources are included in recovery attempts but the binned transform
    family means recovery is not applicable (not log-normalized data).
    """
    obs = pd.DataFrame(index=["cell-1"])
    var = pd.DataFrame(index=["gene-a"])
    binned = csr_matrix(np.array([[0.1]], dtype=np.float32))
    adata = ad.AnnData(X=binned, obs=obs, var=var)
    adata.layers["binned"] = binned

    candidate = _audit_matrix_candidate(
        ".layers[binned]",
        adata.layers["binned"],
        row_count=1,
        column_count=1,
        sample_rows=1,
    )

    assert candidate.inferred_transform == "binned"
    assert candidate.recovery_policy == "disallowed"


# ---------------------------------------------------------------------------
# Integration-style test: inspect_target round-trip with recovery
# ---------------------------------------------------------------------------


def test_inspect_target_recovery_flag_when_raw_counts_available(tmp_path: Path):
    """inspect_target populates uses_recovery when raw integer .X enables recovery.

    When .X contains integer counts AND a log-norm layer exists, the inspector
    selects .X directly (uses_recovery=False) because it is already integer.
    This test verifies that the uses_recovery flag is correctly False in this case.
    """
    obs = pd.DataFrame(
        {
            "guide": ["ctrl", "gene-a", "gene-b"],
            "cell_line": ["K562", "K562", "K562"],
        },
        index=["c1", "c2", "c3"],
    )
    var = pd.DataFrame(index=["gene-x", "gene-y"])
    int_counts = np.array([[1, 2], [3, 0], [0, 4]], dtype=np.int32)
    adata = ad.AnnData(X=csr_matrix(int_counts), obs=obs, var=var)
    # Non-integer layer (would need recovery to be useful)
    adata.layers["log_norm"] = np.log1p(int_counts).astype(np.float32)

    source_path = tmp_path / "recovery_test.h5ad"
    adata.write_h5ad(source_path)

    output_root = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"output_root: {output_root}\n"
        "datasets:\n"
        "  - dataset_id: recovery_check\n"
        f"    source_path: {source_path}\n"
        "    source_release: synthetic-v1\n",
        encoding="utf-8",
    )

    inspect_target(
        InspectionTarget(
            dataset_id="recovery_check",
            source_path=str(source_path),
            source_release="synthetic-v1",
        ),
        output_root=Path(config_path).parent / "outputs",
    )

    from perturb_data_lab.inspectors.models import DatasetSummaryDocument

    summary_path = (
        Path(config_path).parent / "outputs" / "recovery_check" / "dataset-summary.yaml"
    )
    summary = DatasetSummaryDocument.from_yaml_file(summary_path)

    # .X is integer — no recovery needed, uses_recovery is False
    assert summary.count_source_decision.uses_recovery is False
    assert summary.count_source_decision.status == "pass"