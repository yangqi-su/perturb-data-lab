"""Phase 5 tests for transform functions."""

from __future__ import annotations

import pytest

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
