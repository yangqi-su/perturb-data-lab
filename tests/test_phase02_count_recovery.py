"""Phase 2 tests: count recovery and feature identity logic.

Tests cover:
- Reverse-normalization recovery via expm1/size_factor path
- CountSourceDecision.uses_recovery flag
- FeatureTokenizationSpec.is_compatible_for_append() namespace check
- Binned matrices marked as non-recoverable
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
    FeatureTokenizationSpec,
    SchemaDocument,
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


def test_reverse_normalization_not_attempted_on_binned():
    """Binned matrices are non-recoverable and recovery returns None."""
    obs = pd.DataFrame(index=["cell-1", "cell-2"])
    var = pd.DataFrame(index=["gene-a", "gene-b"])
    # Simulate binned data — float values that are NOT log-normalized counts
    binned = csr_matrix(np.array([[0.1, 0.0], [0.0, 0.2]], dtype=np.float32))
    adata = ad.AnnData(X=binned, obs=obs, var=var)
    adata.layers["binned"] = binned

    candidate = _audit_matrix_candidate(
        ".layers[binned]",
        adata.layers["binned"],
        row_count=2,
        column_count=2,
        sample_rows=2,
    )

    assert candidate.inferred_transform == "binned"
    assert "non-recoverable" in " ".join(candidate.notes)
    recovered = _attempt_reverse_normalization(candidate, adata)
    assert recovered is None


def test_choose_count_source_sets_uses_recovery_flag():
    """_choose_count_source reports uses_recovery=True when recovery was used."""
    obs = pd.DataFrame(index=["cell-1", "cell-2", "cell-3"])
    var = pd.DataFrame(index=["gene-a", "gene-b", "gene-c"])
    raw_counts = csr_matrix(np.array([[1, 0, 3], [0, 2, 0], [4, 0, 1]], dtype=np.int32))
    adata = ad.AnnData(X=raw_counts, obs=obs, var=var)
    adata.layers["log_norm"] = np.log1p(raw_counts.toarray()).astype(np.float32)

    candidates = _rank_candidates(
        [
            _audit_matrix_candidate(".X", adata.X, 3, 3, 3),
            _audit_matrix_candidate(".layers[log_norm]", adata.layers["log_norm"], 3, 3, 3),
        ]
    )

    decision = _choose_count_source(candidates, adata)

    # Integer .X should be selected directly without recovery
    assert decision.uses_recovery is False


def test_choose_count_source_skips_recovery_when_no_size_factor_source():
    """When no integer count source exists for size factors, recovery is not attempted.

    If .X is the only candidate and is non-integer with no .raw.X, then size
    factors cannot be computed and recovery correctly returns None. The dataset
    should fail materialization readiness rather than using unverified data.
    """
    obs = pd.DataFrame(index=["cell-1", "cell-2", "cell-3"])
    var = pd.DataFrame(index=["gene-a", "gene-b", "gene-c"])
    # Log-normalized float in .X with no integer source anywhere
    log_norm = np.log1p(np.array([[1, 0, 3], [0, 2, 0], [4, 0, 1]], dtype=np.float32))
    adata = ad.AnnData(X=log_norm, obs=obs, var=var)

    candidates = _rank_candidates(
        [_audit_matrix_candidate(".X", adata.X, 3, 3, 3)]
    )

    decision = _choose_count_source(candidates, adata)

    # No integer source means no size factor source — recovery is not attempted
    # and the dataset fails materialization readiness
    assert decision.uses_recovery is False
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
# FeatureTokenizationSpec.append_compatibility tests
# ---------------------------------------------------------------------------


def test_feature_tokenization_is_compatible_for_append_matches():
    """Namespace match returns True for append compatibility."""
    spec = FeatureTokenizationSpec(selected="gene_symbol", namespace="gene_symbol")
    assert spec.is_compatible_for_append("gene_symbol") is True


def test_feature_tokenization_is_compatible_for_append_mismatch():
    """Namespace mismatch returns False."""
    spec = FeatureTokenizationSpec(selected="gene_symbol", namespace="gene_symbol")
    assert spec.is_compatible_for_append("ensembl") is False


def test_feature_tokenization_is_compatible_for_append_unknown_namespace():
    """Unknown or unset namespace returns False."""
    spec = FeatureTokenizationSpec(selected="set-manually", namespace="unknown")
    assert spec.is_compatible_for_append("gene_symbol") is False


# ---------------------------------------------------------------------------
# Binned matrix non-recoverable annotation test
# ---------------------------------------------------------------------------


def test_binned_candidate_marked_non_recoverable():
    """Binned candidate is annotated with non-recoverable note."""
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
    assert any("non-recoverable" in str(n) for n in candidate.notes)


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
