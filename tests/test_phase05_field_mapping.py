"""Phase 5 tests for field mapping and schema utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from perturb_data_lab.inspectors.models import (
    CountSourceSpec,
    FeatureTokenizationSpec,
    SchemaDocument,
    SchemaFieldEntry,
)
from perturb_data_lab.inspectors.schema_utils import (
    SchemaExplanation,
    check_namespace_compatibility,
    explain_schema,
    preview_cell_row,
    preview_feature_row,
    preview_field_resolution,
)
from perturb_data_lab.inspectors.transforms import (
    TRANSFORM_CATALOG,
    build_transform,
    coalesce_values,
    dose_parse,
    dose_unit,
    map_values,
    split_on_delimiter,
    timepoint_parse,
    timepoint_unit,
)
from perturb_data_lab.materializers.schema_execution import (
    SchemaExecutionResult,
    resolve_field_entry,
)
from perturb_data_lab.materializers.validation import (
    ReadinessViolation,
    SchemaReadinessResult,
    validate_schema_readiness,
)


# ---------------------------------------------------------------------------
# Transform function tests
# ---------------------------------------------------------------------------


class TestNewTransforms:
    """Unit tests for the Phase 5 transform functions."""

    def test_coalesce_values_first_non_null(self):
        assert coalesce_values(("NTC", "CTRL", "ctl")) == "NTC"

    def test_coalesce_values_skips_null(self):
        assert coalesce_values(("", "NA", "NTC")) == "NTC"

    def test_coalesce_values_all_null_returns_na(self):
        from perturb_data_lab.contracts import MISSING_VALUE_LITERAL
        assert coalesce_values(("", "NA", "n/a")) == MISSING_VALUE_LITERAL

    def test_coalesce_values_single_value(self):
        assert coalesce_values(("value",)) == "value"

    def test_coalesce_values_empty_tuple(self):
        from perturb_data_lab.contracts import MISSING_VALUE_LITERAL
        assert coalesce_values(()) == MISSING_VALUE_LITERAL

    def test_split_on_delimiter_basic(self):
        assert split_on_delimiter("TP53+MDM2", delimiter="+", part=0) == "TP53"
        assert split_on_delimiter("TP53+MDM2", delimiter="+", part=1) == "MDM2"

    def test_split_on_delimiter_out_of_range(self):
        from perturb_data_lab.contracts import MISSING_VALUE_LITERAL
        assert split_on_delimiter("TP53+MDM2", delimiter="+", part=5) == MISSING_VALUE_LITERAL

    def test_split_on_delimiter_comma(self):
        assert split_on_delimiter("geneA,geneB", delimiter=",", part=0) == "geneA"
        assert split_on_delimiter("geneA,geneB", delimiter=",", part=1) == "geneB"

    def test_split_on_delimiter_default_comma(self):
        assert split_on_delimiter("a,b,c", part=2) == "c"

    def test_split_on_delimiter_strips_whitespace(self):
        assert split_on_delimiter("TP53 , MDM2", delimiter=",", part=1) == "MDM2"

    def test_split_on_delimiter_empty_part(self):
        from perturb_data_lab.contracts import MISSING_VALUE_LITERAL
        assert split_on_delimiter("a,,c", delimiter=",", part=1) == MISSING_VALUE_LITERAL

    def test_map_values_basic(self):
        # mapping: canonical name → preferred short form
        mapping = {"Homo sapiens": "human", "Mus musculus": "mouse"}
        assert map_values("Homo sapiens", mapping) == "human"
        assert map_values("Mus musculus", mapping) == "mouse"
        # unmapped values pass through unchanged
        assert map_values("Rattus norvegicus", mapping) == "Rattus norvegicus"

    def test_map_values_empty_mapping(self):
        assert map_values("anything", {}) == "anything"

    def test_dose_parse_nanomolar(self):
        assert dose_parse("100nM") == "100"
        assert dose_parse("50 nM") == "50"
        assert dose_parse("1.5uM") == "1.5"

    def test_dose_parse_micromolar(self):
        assert dose_parse("1.5uM") == "1.5"
        assert dose_parse("10 μM") == "10"

    def test_dose_parse_mass_per_weight(self):
        assert dose_parse("10mg/kg") == "10"
        assert dose_parse("5 μg/kg") == "5"

    def test_dose_parse_bare_number(self):
        assert dose_parse("42") == "42"
        assert dose_parse("3.14") == "3.14"

    def test_dose_parse_unknown(self):
        from perturb_data_lab.contracts import MISSING_VALUE_LITERAL
        assert dose_parse("unknown") == MISSING_VALUE_LITERAL
        assert dose_parse("") == MISSING_VALUE_LITERAL

    def test_dose_unit_nanomolar(self):
        assert dose_unit("100nM") == "nm"
        assert dose_unit("50 nM") == "nm"

    def test_dose_unit_micromolar(self):
        assert dose_unit("1.5uM") == "um"
        assert dose_unit("10 μM") == "um"

    def test_dose_unit_mass_per_weight(self):
        assert dose_unit("10mg/kg") == "mg/kg"

    def test_dose_unit_unknown(self):
        from perturb_data_lab.contracts import MISSING_VALUE_LITERAL
        assert dose_unit("unknown") == MISSING_VALUE_LITERAL

    def test_timepoint_parse_hours(self):
        assert timepoint_parse("24h") == "24"
        assert timepoint_parse("48 hr") == "48"
        assert timepoint_parse("6hrs") == "6"

    def test_timepoint_parse_days(self):
        assert timepoint_parse("7d") == "7"
        assert timepoint_parse("3 days") == "3"
        assert timepoint_parse("1day") == "1"

    def test_timepoint_parse_minutes(self):
        assert timepoint_parse("30m") == "30"
        assert timepoint_parse("15 min") == "15"

    def test_timepoint_parse_bare_number(self):
        assert timepoint_parse("72") == "72"

    def test_timepoint_parse_unknown(self):
        from perturb_data_lab.contracts import MISSING_VALUE_LITERAL
        assert timepoint_parse("unknown") == MISSING_VALUE_LITERAL
        assert timepoint_parse("") == MISSING_VALUE_LITERAL

    def test_timepoint_unit_hours(self):
        assert timepoint_unit("24h") == "h"
        assert timepoint_unit("48 hr") == "h"

    def test_timepoint_unit_days(self):
        assert timepoint_unit("7d") == "d"
        assert timepoint_unit("3 days") == "d"

    def test_timepoint_unit_minutes(self):
        assert timepoint_unit("30m") == "m"
        assert timepoint_unit("15 min") == "m"

    def test_timepoint_unit_unknown(self):
        from perturb_data_lab.contracts import MISSING_VALUE_LITERAL
        assert timepoint_unit("unknown") == MISSING_VALUE_LITERAL


class TestBuildTransform:
    """Test the build_transform factory."""

    def test_build_transform_simple(self):
        t = build_transform("strip_prefix", prefix="sg")
        assert t.name == "strip_prefix"
        assert t.args["prefix"] == "sg"

    def test_build_transform_coalesce(self):
        t = build_transform("coalesce_values")
        assert t.name == "coalesce_values"
        assert t.args == {}

    def test_build_transform_split(self):
        t = build_transform("split_on_delimiter", delimiter="+", part=0)
        assert t.name == "split_on_delimiter"
        assert t.args["delimiter"] == "+"
        assert t.args["part"] == 0

    def test_build_transform_map_values(self):
        mapping = {"a": "b"}
        t = build_transform("map_values", mapping=mapping)
        assert t.name == "map_values"
        assert t.args["mapping"] == {"a": "b"}


# ---------------------------------------------------------------------------
# Schema execution tests for new transforms
# ---------------------------------------------------------------------------


class TestSchemaExecutionNewTransforms:
    """Test resolve_field_entry with Phase 5 transform patterns."""

    def test_coalesce_derived_strategy(self):
        entry = SchemaFieldEntry(
            source_fields=("guide_a", "guide_b", "guide_c"),
            strategy="derived",
            transforms=(build_transform("coalesce_values"),),
            confidence="high",
            required=True,
        )
        row = {"guide_a": "", "guide_b": "NTC", "guide_c": "CTRL"}
        result = resolve_field_entry(entry, row)
        assert result.value == "NTC"
        assert result.was_resolved

    def test_coalesce_all_null(self):
        from perturb_data_lab.contracts import MISSING_VALUE_LITERAL
        entry = SchemaFieldEntry(
            source_fields=("g1", "g2"),
            strategy="derived",
            transforms=(build_transform("coalesce_values"),),
            confidence="high",
            required=True,
        )
        row = {"g1": "", "g2": "NA"}
        result = resolve_field_entry(entry, row)
        assert result.value == MISSING_VALUE_LITERAL

    def test_split_on_delimiter_derived(self):
        entry = SchemaFieldEntry(
            source_fields=("combo",),
            strategy="derived",
            transforms=(build_transform("split_on_delimiter", delimiter="+", part=0),),
            confidence="high",
            required=False,
        )
        row = {"combo": "TP53+MDM2"}
        result = resolve_field_entry(entry, row)
        assert result.value == "TP53"

    def test_split_second_component(self):
        entry = SchemaFieldEntry(
            source_fields=("combo",),
            strategy="derived",
            transforms=(build_transform("split_on_delimiter", delimiter="+", part=1),),
            confidence="high",
            required=False,
        )
        row = {"combo": "TP53+MDM2"}
        result = resolve_field_entry(entry, row)
        assert result.value == "MDM2"

    def test_dose_parse_transform(self):
        entry = SchemaFieldEntry(
            source_fields=("dose_column",),
            strategy="source-field",
            transforms=(build_transform("dose_parse"),),
            confidence="high",
            required=False,
        )
        row = {"dose_column": "100nM"}
        result = resolve_field_entry(entry, row)
        assert result.value == "100"

    def test_timepoint_parse_transform(self):
        entry = SchemaFieldEntry(
            source_fields=("time_column",),
            strategy="source-field",
            transforms=(build_transform("timepoint_parse"),),
            confidence="high",
            required=False,
        )
        row = {"time_column": "24h"}
        result = resolve_field_entry(entry, row)
        assert result.value == "24"

    def test_map_values_transform(self):
        entry = SchemaFieldEntry(
            source_fields=("species_raw",),
            strategy="source-field",
            transforms=(build_transform("map_values", mapping={"Homo sapiens": "human", "Mus musculus": "mouse"}),),
            confidence="high",
            required=False,
        )
        row = {"species_raw": "Homo sapiens"}
        result = resolve_field_entry(entry, row)
        assert result.value == "human"

    def test_unmapped_passes_through(self):
        entry = SchemaFieldEntry(
            source_fields=("species_raw",),
            strategy="source-field",
            transforms=(build_transform("map_values", mapping={"Homo sapiens": "human"}),),
            confidence="high",
            required=False,
        )
        row = {"species_raw": "dog"}
        result = resolve_field_entry(entry, row)
        assert result.value == "dog"


# ---------------------------------------------------------------------------
# Schema explanation tests
# ---------------------------------------------------------------------------


class TestExplainSchema:
    """Test explain_schema and related utilities."""

    def _make_draft_schema(self) -> SchemaDocument:
        return SchemaDocument.new_draft(
            dataset_id="test_ds",
            source_path="/fake/path.h5ad",
            dataset_metadata={
                "dataset_id": SchemaFieldEntry(
                    source_fields=(),
                    strategy="literal",
                    transforms=(),
                    confidence="high",
                    required=True,
                    literal_value="test_ds",
                ),
            },
            perturbation_fields={
                "perturbation_label": SchemaFieldEntry(
                    source_fields=("guide_id",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
                "perturbation_type": SchemaFieldEntry(
                    source_fields=(),
                    strategy="null",
                    transforms=(),
                    confidence="low",
                    required=True,
                ),
                "control_flag": SchemaFieldEntry(
                    source_fields=("guide_id",),
                    strategy="derived",
                    transforms=(build_transform("recognize_control", patterns=["^ntc", "control"]),),
                    confidence="medium",
                    required=True,
                ),
            },
            context_fields={
                "cell_context": SchemaFieldEntry(
                    source_fields=(),
                    strategy="null",
                    transforms=(),
                    confidence="low",
                    required=True,
                ),
            },
            feature_fields={
                "feature_id": SchemaFieldEntry(
                    source_fields=("gene_id",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
            },
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            feature_tokenization=FeatureTokenizationSpec(
                selected="gene_id",
                namespace="unknown",
            ),
            transform_catalog=TRANSFORM_CATALOG,
        )

    def test_explain_schema_counts(self):
        schema = self._make_draft_schema()
        exp = explain_schema(schema)
        assert exp.dataset_id == "test_ds"
        assert exp.status == "draft"
        # We have: dataset_id + perturbation_label + perturbation_type + control_flag + cell_context + feature_id = 6 fields
        assert exp.field_count == 6
        # perturbation_label, control_flag, feature_id, dataset_id are resolved = 4
        assert exp.resolved_count == 4
        # perturbation_type (required null), cell_context (required null) = 2 unresolved
        assert exp.unresolved_count == 2
        # perturbation_type and cell_context are both required with null strategy = 2
        assert exp.null_required_count == 2

    def test_explain_schema_namespace_unset(self):
        schema = self._make_draft_schema()
        exp = explain_schema(schema)
        assert exp.namespace == "unknown"
        assert exp.namespace_status == "unset"
        assert exp.namespace_issue is not None
        assert "unknown" in exp.namespace_issue

    def test_explain_schema_namespace_valid(self):
        schema = self._make_draft_schema()
        # Replace with a valid namespace
        schema = SchemaDocument.new_draft(
            dataset_id="test_ds",
            source_path="/fake/path.h5ad",
            dataset_metadata={},
            perturbation_fields={},
            context_fields={},
            feature_fields={},
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            feature_tokenization=FeatureTokenizationSpec(
                selected="gene_id",
                namespace="ensembl",
            ),
            transform_catalog=TRANSFORM_CATALOG,
        )
        exp = explain_schema(schema)
        assert exp.namespace == "ensembl"
        assert exp.namespace_status == "valid"
        assert exp.namespace_issue is None

    def test_explain_schema_readiness_blockers(self):
        schema = self._make_draft_schema()
        exp = explain_schema(schema)
        assert len(exp.readiness_blockers) > 0
        assert any("status" in b and "draft" in b for b in exp.readiness_blockers)
        assert any("required field" in b for b in exp.readiness_blockers)

    def test_explain_field_entry_null_required(self):
        schema = self._make_draft_schema()
        exp = explain_schema(schema)
        null_entries = [fe for fe in exp.field_explanations if fe.status == "null_strategy"]
        assert len(null_entries) == 2
        for fe in null_entries:
            assert fe.required
            assert fe.resolution_hint is not None


# ---------------------------------------------------------------------------
# Preview utilities tests
# ---------------------------------------------------------------------------


class TestPreviewFieldResolution:
    """Test preview_field_resolution on sample values."""

    def _make_schema(self) -> SchemaDocument:
        return SchemaDocument.new_draft(
            dataset_id="preview_test",
            source_path="/fake.h5ad",
            dataset_metadata={},
            perturbation_fields={
                "perturbation_label": SchemaFieldEntry(
                    source_fields=("guide_id",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
                "control_flag": SchemaFieldEntry(
                    source_fields=("guide_id",),
                    strategy="derived",
                    transforms=(build_transform("recognize_control", patterns=["^ntc", "control"]),),
                    confidence="medium",
                    required=True,
                ),
                "dose": SchemaFieldEntry(
                    source_fields=("dose_raw",),
                    strategy="source-field",
                    transforms=(build_transform("dose_parse"),),
                    confidence="high",
                    required=False,
                ),
                "timepoint": SchemaFieldEntry(
                    source_fields=("time_raw",),
                    strategy="source-field",
                    transforms=(build_transform("timepoint_parse"),),
                    confidence="high",
                    required=False,
                ),
            },
            context_fields={},
            feature_fields={},
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            feature_tokenization=FeatureTokenizationSpec(selected="gene_id", namespace="ensembl"),
            transform_catalog=TRANSFORM_CATALOG,
        )

    def test_preview_source_field(self):
        schema = self._make_schema()
        entry = schema.perturbation_fields["perturbation_label"]
        result = preview_field_resolution(entry, {"guide_id": "TP53_1"})
        assert result == "TP53_1"

    def test_preview_recognize_control_true(self):
        schema = self._make_schema()
        entry = schema.perturbation_fields["control_flag"]
        result = preview_field_resolution(entry, {"guide_id": "NTC_1"})
        assert result == "true"

    def test_preview_recognize_control_false(self):
        schema = self._make_schema()
        entry = schema.perturbation_fields["control_flag"]
        result = preview_field_resolution(entry, {"guide_id": "TP53_1"})
        assert result == "false"

    def test_preview_dose_parse(self):
        schema = self._make_schema()
        entry = schema.perturbation_fields["dose"]
        result = preview_field_resolution(entry, {"dose_raw": "100nM"})
        assert result == "100"

    def test_preview_timepoint_parse(self):
        schema = self._make_schema()
        entry = schema.perturbation_fields["timepoint"]
        result = preview_field_resolution(entry, {"time_raw": "24h"})
        assert result == "24"

    def test_preview_null_field(self):
        schema = self._make_schema()
        # cell_context is null strategy
        entry = schema.context_fields.get("cell_context") or SchemaFieldEntry(
            source_fields=(), strategy="null", transforms=(), confidence="low", required=True
        )
        from perturb_data_lab.contracts import MISSING_VALUE_LITERAL
        result = preview_field_resolution(entry, {})
        assert result == MISSING_VALUE_LITERAL


class TestPreviewCellRow:
    """Test preview_cell_row end-to-end."""

    def test_preview_cell_row_full(self):
        schema = SchemaDocument.new_draft(
            dataset_id="row_preview",
            source_path="/fake.h5ad",
            dataset_metadata={},
            perturbation_fields={
                "perturbation_label": SchemaFieldEntry(
                    source_fields=("guide",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
            },
            context_fields={
                "cell_context": SchemaFieldEntry(
                    source_fields=("cell_type",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
            },
            feature_fields={},
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            feature_tokenization=FeatureTokenizationSpec(selected="gene_id", namespace="ensembl"),
            transform_catalog=TRANSFORM_CATALOG,
        )
        obs_row = {"guide": "TP53_1", "cell_type": "K562"}
        result = preview_cell_row(schema, obs_row)
        assert result["perturbation"]["perturbation_label"] == "TP53_1"
        assert result["context"]["cell_context"] == "K562"


class TestPreviewFeatureRow:
    """Test preview_feature_row end-to-end."""

    def test_preview_feature_row(self):
        schema = SchemaDocument.new_draft(
            dataset_id="feat_preview",
            source_path="/fake.h5ad",
            dataset_metadata={},
            perturbation_fields={},
            context_fields={},
            feature_fields={
                "feature_id": SchemaFieldEntry(
                    source_fields=("gene_id",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
                "feature_label": SchemaFieldEntry(
                    source_fields=("gene_symbol",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=False,
                ),
            },
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            feature_tokenization=FeatureTokenizationSpec(selected="gene_id", namespace="ensembl"),
            transform_catalog=TRANSFORM_CATALOG,
        )
        var_row = {"gene_id": "ENSG1", "gene_symbol": "TP53"}
        result = preview_feature_row(schema, var_row)
        assert result["feature_id"] == "ENSG1"
        assert result["feature_label"] == "TP53"


# ---------------------------------------------------------------------------
# Namespace compatibility tests
# ---------------------------------------------------------------------------


class TestNamespaceCompatibility:
    """Test check_namespace_compatibility."""

    def _make_ready_schema(self, namespace: str) -> SchemaDocument:
        # Create a status:ready schema by building directly, not via new_draft
        return SchemaDocument(
            kind="schema",
            contract_version="0.2.0",
            dataset_id="compat_test",
            source_path="/fake.h5ad",
            status="ready",
            dataset_metadata={},
            perturbation_fields={
                "perturbation_label": SchemaFieldEntry(
                    source_fields=("guide",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
            },
            context_fields={},
            feature_fields={
                "feature_id": SchemaFieldEntry(
                    source_fields=("gene_id",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
            },
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            feature_tokenization=FeatureTokenizationSpec(
                selected="gene_id",
                namespace=namespace,
            ),
            transform_catalog=TRANSFORM_CATALOG,
            materialization_notes=(),
        )

    def test_compatible_exact_match(self):
        schema = self._make_ready_schema("ensembl")
        ok, reason = check_namespace_compatibility(schema, "ensembl")
        assert ok
        assert reason == ""

    def test_incompatible_mismatch(self):
        schema = self._make_ready_schema("ensembl")
        ok, reason = check_namespace_compatibility(schema, "gene_symbol")
        assert not ok
        assert "mismatch" in reason
        assert "ensembl" in reason
        assert "gene_symbol" in reason

    def test_unset_namespace_rejected(self):
        for ns in ("unknown", "set-manually", ""):
            schema = self._make_ready_schema(ns)
            ok, reason = check_namespace_compatibility(schema, "ensembl")
            assert not ok
            assert "unknown" in reason or "set-manually" in reason or "namespace is ''" in reason

    def test_not_ready_schema_rejected(self):
        # status is draft, not ready - should be rejected
        schema = SchemaDocument.new_draft(
            dataset_id="compat_test",
            source_path="/fake.h5ad",
            dataset_metadata={},
            perturbation_fields={},
            context_fields={},
            feature_fields={
                "feature_id": SchemaFieldEntry(
                    source_fields=("gene_id",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
            },
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            feature_tokenization=FeatureTokenizationSpec(
                selected="gene_id",
                namespace="ensembl",
            ),
            transform_catalog=TRANSFORM_CATALOG,
        )
        ok, reason = check_namespace_compatibility(schema, "ensembl")
        # draft schema: should be rejected because not "ready"
        assert not ok
        assert "ready" in reason


# ---------------------------------------------------------------------------
# Validation tests with namespace gate
# ---------------------------------------------------------------------------


class TestValidationNamespaceGate:
    """Test that validate_schema_readiness catches unset namespaces."""

    def test_validation_rejects_unset_namespace(self, tmp_path: Path):
        schema = SchemaDocument.new_draft(
            dataset_id="ns_gate_test",
            source_path="/fake.h5ad",
            dataset_metadata={},
            perturbation_fields={
                "perturbation_label": SchemaFieldEntry(
                    source_fields=("guide",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
            },
            context_fields={},
            feature_fields={
                "feature_id": SchemaFieldEntry(
                    source_fields=("gene_id",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
            },
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            feature_tokenization=FeatureTokenizationSpec(
                selected="gene_id",
                namespace="unknown",
            ),
            transform_catalog=TRANSFORM_CATALOG,
        )
        # Manually set to ready
        import perturb_data_lab.inspectors.models as m

        schema = SchemaDocument(
            kind="schema",
            contract_version=schema.contract_version,
            dataset_id=schema.dataset_id,
            source_path=schema.source_path,
            status="ready",
            dataset_metadata=schema.dataset_metadata,
            perturbation_fields=schema.perturbation_fields,
            context_fields=schema.context_fields,
            feature_fields=schema.feature_fields,
            count_source=schema.count_source,
            feature_tokenization=schema.feature_tokenization,
            transform_catalog=schema.transform_catalog,
            materialization_notes=(),
        )
        schema_path = tmp_path / "schema.yaml"
        schema.write_yaml(schema_path)

        result = validate_schema_readiness(str(schema_path))
        assert not result.valid
        ns_violations = [v for v in result.violations if "namespace" in v.field]
        assert len(ns_violations) == 1
        assert "unknown" in ns_violations[0].reason

    def test_validation_accepts_valid_namespace(self, tmp_path: Path):
        schema = SchemaDocument.new_draft(
            dataset_id="valid_ns_test",
            source_path="/fake.h5ad",
            dataset_metadata={},
            perturbation_fields={
                "perturbation_label": SchemaFieldEntry(
                    source_fields=("guide",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
            },
            context_fields={},
            feature_fields={
                "feature_id": SchemaFieldEntry(
                    source_fields=("gene_id",),
                    strategy="source-field",
                    transforms=(),
                    confidence="high",
                    required=True,
                ),
            },
            count_source=CountSourceSpec(selected=".X", integer_only=True),
            feature_tokenization=FeatureTokenizationSpec(
                selected="gene_id",
                namespace="ensembl",
            ),
            transform_catalog=TRANSFORM_CATALOG,
        )
        # Set to ready manually
        schema = SchemaDocument(
            kind="schema",
            contract_version=schema.contract_version,
            dataset_id=schema.dataset_id,
            source_path=schema.source_path,
            status="ready",
            dataset_metadata=schema.dataset_metadata,
            perturbation_fields=schema.perturbation_fields,
            context_fields=schema.context_fields,
            feature_fields=schema.feature_fields,
            count_source=schema.count_source,
            feature_tokenization=schema.feature_tokenization,
            transform_catalog=schema.transform_catalog,
            materialization_notes=(),
        )
        schema_path = tmp_path / "schema.yaml"
        schema.write_yaml(schema_path)

        result = validate_schema_readiness(str(schema_path))
        assert result.valid
        assert len(result.violations) == 0


# ---------------------------------------------------------------------------
# Transform catalog completeness
# ---------------------------------------------------------------------------


class TestTransformCatalog:
    """Test that TRANSFORM_CATALOG includes Phase 5 new transforms."""

    def test_all_new_transforms_in_catalog(self):
        new_transforms = {
            "coalesce_values",
            "split_on_delimiter",
            "map_values",
            "dose_parse",
            "dose_unit",
            "timepoint_parse",
            "timepoint_unit",
        }
        catalog_names = {entry.name for entry in TRANSFORM_CATALOG}
        assert new_transforms.issubset(catalog_names), (
            f"New transforms not in catalog: {new_transforms - catalog_names}"
        )
