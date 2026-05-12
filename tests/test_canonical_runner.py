"""Unit tests for canonicalization runner: transforms, gene mapping, vocab building."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perturb_data_lab.canonical.contract import (
    CANONICAL_OBS_MUST_HAVE,
    CANONICAL_VAR_MUST_HAVE,
    CanonicalVocab,
    CanonicalizationSchema,
    ConditionalCase,
    ExtensibleColumn,
    GeneMappingConfig,
    ObsColumnMapping,
    TransformRule,
    VarColumnMapping,
)
from perturb_data_lab.canonical.runner import (
    CanonicalizationRunner,
    CanonicalizationResult,
    build_canonical_vocab,
    run_canonicalization,
)
from perturb_data_lab.inspectors.transforms import TRANSFORM_CATALOG, get_transform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_obs_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a minimal raw-obs.parquet with cell_id and raw_fields JSON."""
    cell_ids = [r.get("cell_id", f"cell_{i:06d}") for i, r in enumerate(rows)]
    raw_fields = [json.dumps({k: v for k, v in r.items() if k != "cell_id"}) for r in rows]
    table = pa.table({
        "cell_id": pa.array(cell_ids, type=pa.string()),
        "dataset_id": pa.array(["test"] * len(rows), type=pa.string()),
        "dataset_release": pa.array(["test-release"] * len(rows), type=pa.string()),
        "raw_fields": pa.array(raw_fields, type=pa.string()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _make_raw_var_parquet(path: Path, feature_ids: list[str]) -> None:
    """Write a minimal raw-var.parquet with origin_index and feature_id."""
    table = pa.table({
        "origin_index": pa.array(list(range(len(feature_ids))), type=pa.int32()),
        "feature_id": pa.array(feature_ids, type=pa.string()),
        "raw_var": pa.array(["{}"] * len(feature_ids), type=pa.string()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _make_size_factor_parquet(path: Path, cell_ids: list[str], size_factors: list[float]) -> None:
    """Write a minimal size-factor.parquet."""
    table = pa.table({
        "cell_id": pa.array(cell_ids, type=pa.string()),
        "size_factor": pa.array(size_factors, type=pa.float64()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _make_minimal_schema(path: Path, dataset_id: str = "test") -> CanonicalizationSchema:
    """Create and write a minimal canonicalization schema covering all must-have fields."""
    # Fields that read from raw data
    _SOURCE_FIELDS = {
        "cell_id": "cell_id",
        "dataset_id": "dataset_id",
        "perturb_label": "guide_1",
        "perturb_type": "treatment",
        "cell_context": "cell_type",
        "cell_line_or_type": "cellline",
        "condition": "genotype",
        "batch_id": "batch",
        "donor_id": "donor_id",
        "size_factor": "size_factor",
    }
    # Fields that use a literal value
    _LITERAL_FIELDS = {
        "dataset_index": "0",
        "species": "human",
        "tissue": "unknown",
        "assay": "Perturb-seq",
        "disease_state": "healthy",
    }
    # Fields that are row-index
    _ROW_INDEX_FIELDS = {"global_row_index", "local_row_index"}
    # Fallback values per field
    _FALLBACKS: dict[str, str] = {
        "dose": "NA",
        "dose_unit": "NA",
        "timepoint": "NA",
        "timepoint_unit": "NA",
        "sex": "unknown",
        "donor_id": "unknown",
        "cell_context": "unknown",
        "cell_line_or_type": "unknown",
        "condition": "WT",
        "batch_id": "NA",
        "perturb_type": "untreated",
    }

    def _make_mapping(name: str) -> ObsColumnMapping:
        if name in _SOURCE_FIELDS:
            return ObsColumnMapping(
                canonical_name=name,
                strategy="source-field",
                source_column=_SOURCE_FIELDS[name],
                fallback=_FALLBACKS.get(name, "NA"),
            )
        if name in _LITERAL_FIELDS:
            return ObsColumnMapping(
                canonical_name=name,
                strategy="literal",
                literal_value=_LITERAL_FIELDS[name],
            )
        if name in _ROW_INDEX_FIELDS:
            return ObsColumnMapping(
                canonical_name=name,
                strategy="row-index",
            )
        # null strategy
        return ObsColumnMapping(
            canonical_name=name,
            strategy="null",
            fallback=_FALLBACKS.get(name, "NA"),
        )

    schema = CanonicalizationSchema(
        dataset_id=dataset_id,
        status="draft",
        obs_column_mappings=tuple(_make_mapping(name) for name in CANONICAL_OBS_MUST_HAVE),
        var_column_mappings=(
            VarColumnMapping(canonical_name="origin_index", strategy="passthrough"),
            VarColumnMapping(
                canonical_name="gene_id",
                strategy="source-field",
                source_column="feature_id",
            ),
            VarColumnMapping(
                canonical_name="canonical_gene_id",
                strategy="gene-mapping",
                enabled=False,
                engine="identity",
            ),
            VarColumnMapping(canonical_name="global_id", strategy="auto"),
        ),
        gene_mapping=GeneMappingConfig(enabled=False, engine="identity"),
    )
    schema.write_yaml(path)
    return schema


def _replace_obs_mapping(
    schema: CanonicalizationSchema,
    mapping: ObsColumnMapping,
) -> CanonicalizationSchema:
    return CanonicalizationSchema(
        kind=schema.kind,
        contract_version=schema.contract_version,
        dataset_id=schema.dataset_id,
        status=schema.status,
        description=schema.description,
        obs_column_mappings=tuple(
            mapping if existing.canonical_name == mapping.canonical_name else existing
            for existing in schema.obs_column_mappings
        ),
        obs_extensible=schema.obs_extensible,
        var_column_mappings=schema.var_column_mappings,
        var_extensible=schema.var_extensible,
        gene_mapping=schema.gene_mapping,
        notes=schema.notes,
    )


def _run_with_custom_obs_mapping(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    mapping: ObsColumnMapping,
) -> CanonicalizationResult:
    raw_obs = tmp_path / "raw-obs.parquet"
    raw_var = tmp_path / "raw-var.parquet"
    sf_path = tmp_path / "size-factor.parquet"
    schema_path = tmp_path / "canonicalization-schema.yaml"
    out_root = tmp_path / "output"

    cell_ids = [row.get("cell_id", f"cell_{index}") for index, row in enumerate(rows)]
    _make_raw_obs_parquet(raw_obs, rows)
    _make_raw_var_parquet(raw_var, ["gene1", "gene2"])
    _make_size_factor_parquet(sf_path, cell_ids, [1.0] * len(rows))
    base_schema = _make_minimal_schema(schema_path)
    schema = _replace_obs_mapping(base_schema, mapping)
    schema.write_yaml(schema_path)

    runner = CanonicalizationRunner(
        raw_obs_path=raw_obs,
        raw_var_path=raw_var,
        size_factor_path=sf_path,
        schema_path=schema_path,
        output_root=out_root,
    )
    return runner.run()


# ---------------------------------------------------------------------------
# Transform application tests
# ---------------------------------------------------------------------------


class TestTransformApplication:
    """Test that individual transforms work as expected through the runner."""

    def test_transform_catalog_and_dispatch_include_phase4_scalars(self):
        required = {
            "map_control_labels",
            "strip_whitespace",
            "replace_empty_with_null",
            "normalize_case",
            "map_values",
            "regex_sub",
            "strip_prefix",
            "strip_suffix",
            "strip_guide_suffix",
            "split_on_delimiter",
            "dose_parse",
            "dose_unit",
            "timepoint_parse",
            "timepoint_unit",
            "normalize_time_unit",
            "normalize_dose_unit",
            "strip_ensembl_version",
            "normalize_boolean",
        }
        catalog_names = {entry.name for entry in TRANSFORM_CATALOG}
        assert required.issubset(catalog_names)
        assert all(get_transform(name) is not None for name in required)

    def test_strip_suffix_transform(self):
        """strip_suffix removes a known suffix."""
        result = CanonicalizationRunner._apply_transforms(
            "CTRL_sg1",
            transforms=(TransformRule(name="strip_suffix", args={"suffix": "_sg1"}),),
        )
        assert result == "CTRL"

    def test_strip_prefix_transform(self):
        """strip_prefix removes a known prefix."""
        result = CanonicalizationRunner._apply_transforms(
            "prefix_GENE",
            transforms=(TransformRule(name="strip_prefix", args={"prefix": "prefix_"}),),
        )
        assert result == "GENE"

    def test_regex_sub_transform(self):
        """regex_sub applies a regex substitution."""
        result = CanonicalizationRunner._apply_transforms(
            "GENE_v1",
            transforms=(TransformRule(name="regex_sub", args={"pattern": "_v\\d+", "replacement": ""}),),
        )
        assert result == "GENE"

    def test_normalize_case_lower(self):
        """normalize_case lowercases values."""
        result = CanonicalizationRunner._apply_transforms(
            "TP53",
            transforms=(TransformRule(name="normalize_case", args={"mode": "lower"}),),
        )
        assert result == "tp53"

    def test_normalize_case_upper(self):
        """normalize_case uppercases values."""
        result = CanonicalizationRunner._apply_transforms(
            "tp53",
            transforms=(TransformRule(name="normalize_case", args={"mode": "upper"}),),
        )
        assert result == "TP53"

    def test_strip_whitespace_transform(self):
        """strip_whitespace trims leading and trailing spaces."""
        result = CanonicalizationRunner._apply_transforms(
            "  TP53  ",
            transforms=(TransformRule(name="strip_whitespace", args={}),),
        )
        assert result == "TP53"

    def test_replace_empty_with_null_uses_fallback(self):
        """replace_empty_with_null converts empty strings into fallback values."""
        result = CanonicalizationRunner._apply_transforms(
            "   ",
            transforms=(TransformRule(name="replace_empty_with_null", args={}),),
            fallback="FALLBACK",
        )
        assert result == "FALLBACK"

    def test_map_control_labels_case_insensitive_and_whitespace_aware(self):
        """map_control_labels matches normalized candidate labels and emits ctrl."""
        result = CanonicalizationRunner._apply_transforms(
            "  Non-Targeting  ",
            transforms=(TransformRule(
                name="map_control_labels",
                args={"candidates": ["ctrl", "ntc", "non-targeting"]},
            ),),
        )
        assert result == "ctrl"

    def test_map_control_labels_preserves_unmatched_values(self):
        """map_control_labels leaves non-control labels unchanged."""
        result = CanonicalizationRunner._apply_transforms(
            "TP53",
            transforms=(TransformRule(
                name="map_control_labels",
                args={"candidates": ["ctrl", "ntc", "non-targeting"]},
            ),),
        )
        assert result == "TP53"

    def test_strip_guide_suffix_transform(self):
        """strip_guide_suffix removes common perturbation guide suffixes."""
        result = CanonicalizationRunner._apply_transforms(
            "NTC_sg1",
            transforms=(TransformRule(name="strip_guide_suffix", args={}),),
        )
        assert result == "NTC"

    def test_dose_parse_extracts_numeric(self):
        """dose_parse extracts the numeric part of dose strings."""
        result = CanonicalizationRunner._apply_transforms(
            "100nM",
            transforms=(TransformRule(name="dose_parse", args={}),),
        )
        assert result == "100"

    def test_dose_unit_extracts_unit(self):
        """dose_unit extracts and normalizes the unit."""
        result = CanonicalizationRunner._apply_transforms(
            "100nM",
            transforms=(TransformRule(name="dose_unit", args={}),),
        )
        assert result == "nm"

    def test_normalize_dose_unit_normalizes_standalone_unit(self):
        """normalize_dose_unit normalizes unit-only metadata values."""
        result = CanonicalizationRunner._apply_transforms(
            "μM",
            transforms=(TransformRule(name="normalize_dose_unit", args={}),),
        )
        assert result == "um"

    def test_timepoint_parse_extracts_numeric(self):
        """timepoint_parse extracts the numeric part."""
        result = CanonicalizationRunner._apply_transforms(
            "24h",
            transforms=(TransformRule(name="timepoint_parse", args={}),),
        )
        assert result == "24"

    def test_timepoint_unit_extracts_unit(self):
        """timepoint_unit extracts and normalizes the time unit."""
        result = CanonicalizationRunner._apply_transforms(
            "48 hr",
            transforms=(TransformRule(name="timepoint_unit", args={}),),
        )
        assert result == "h"

    def test_map_values(self):
        """map_values remaps via a lookup table."""
        result = CanonicalizationRunner._apply_transforms(
            "Homo sapiens",
            transforms=(TransformRule(
                name="map_values",
                args={"mapping": {"Homo sapiens": "human", "Mus musculus": "mouse"}},
            ),),
        )
        assert result == "human"

    def test_map_values_unchanged_when_not_in_mapping(self):
        """map_values returns the original value when not in the mapping."""
        result = CanonicalizationRunner._apply_transforms(
            "Rattus norvegicus",
            transforms=(TransformRule(
                name="map_values",
                args={"mapping": {"Homo sapiens": "human"}},
            ),),
        )
        assert result == "Rattus norvegicus"

    def test_split_on_delimiter(self):
        """split_on_delimiter splits and extracts one part."""
        result = CanonicalizationRunner._apply_transforms(
            "TP53+MDM2",
            transforms=(TransformRule(name="split_on_delimiter", args={"delimiter": "+", "part": 0}),),
        )
        assert result == "TP53"

    def test_split_on_delimiter_part1(self):
        """split_on_delimiter extracts the second part."""
        result = CanonicalizationRunner._apply_transforms(
            "TP53+MDM2",
            transforms=(TransformRule(name="split_on_delimiter", args={"delimiter": "+", "part": 1}),),
        )
        assert result == "MDM2"

    def test_normalize_time_unit_normalizes_standalone_unit(self):
        """normalize_time_unit normalizes unit-only metadata values."""
        result = CanonicalizationRunner._apply_transforms(
            "Hours",
            transforms=(TransformRule(name="normalize_time_unit", args={}),),
        )
        assert result == "h"

    def test_strip_ensembl_version(self):
        """strip_ensembl_version removes trailing Ensembl version numbers."""
        result = CanonicalizationRunner._apply_transforms(
            "ENSG00000141510.18",
            transforms=(TransformRule(name="strip_ensembl_version", args={}),),
        )
        assert result == "ENSG00000141510"

    @pytest.mark.parametrize(
        ("raw_value", "expected"),
        [
            ("Yes", "true"),
            (" 0 ", "false"),
            ("FALSE", "false"),
        ],
    )
    def test_normalize_boolean(self, raw_value: str, expected: str):
        """normalize_boolean handles common truthy and falsy spellings."""
        result = CanonicalizationRunner._apply_transforms(
            raw_value,
            transforms=(TransformRule(name="normalize_boolean", args={}),),
        )
        assert result == expected

    def test_normalize_boolean_preserves_unmatched_values(self):
        """normalize_boolean leaves unknown values unchanged."""
        result = CanonicalizationRunner._apply_transforms(
            "maybe",
            transforms=(TransformRule(name="normalize_boolean", args={}),),
        )
        assert result == "maybe"

    def test_chained_transforms(self):
        """Multiple transforms are applied in order."""
        result = CanonicalizationRunner._apply_transforms(
            "CTRL_sg1",
            transforms=(
                TransformRule(name="strip_suffix", args={"suffix": "_sg1"}),
                TransformRule(name="normalize_case", args={"mode": "lower"}),
            ),
        )
        assert result == "ctrl"

    def test_chained_control_normalization(self):
        """Guide suffix stripping can feed into control-label normalization."""
        result = CanonicalizationRunner._apply_transforms(
            "NTC_sg1",
            transforms=(
                TransformRule(name="strip_guide_suffix", args={}),
                TransformRule(name="map_control_labels", args={"candidates": ["ntc"]}),
            ),
        )
        assert result == "ctrl"

    def test_null_value_falls_back(self):
        """Null-like values produce the fallback."""
        result = CanonicalizationRunner._apply_transforms(
            "NA",
            transforms=(TransformRule(name="strip_suffix", args={"suffix": "_x"}),),
            fallback="FALLBACK",
        )
        assert result == "FALLBACK"

    def test_unknown_transform_skipped(self):
        """Unknown transform names are silently skipped."""
        result = CanonicalizationRunner._apply_transforms(
            "hello",
            transforms=(TransformRule(name="nonexistent_transform", args={}),),
        )
        assert result == "hello"

    def test_failing_transform_uses_fallback(self):
        """Transform failures warn and fall back instead of propagating."""
        result = CanonicalizationRunner._apply_transforms(
            "TP53",
            transforms=(TransformRule(name="normalize_case", args={"mode": "snake"}),),
            fallback="FALLBACK",
        )
        assert result == "FALLBACK"


# ---------------------------------------------------------------------------
# Gene mapping tests
# ---------------------------------------------------------------------------


class TestGeneMapping:
    """Test gene ID mapping engines in isolation."""

    def test_identity_mapping(self):
        """Identity mapping returns each gene_id as its own canonical."""
        from perturb_data_lab.canonical.runner import _gene_map_identity
        result = _gene_map_identity(["gene1", "gene2", "gene3"])
        assert result == {"gene1": "gene1", "gene2": "gene2", "gene3": "gene3"}

    def test_file_mapping(self, tmp_path: Path):
        """Mapping file maps gene_ids to canonical_gene_ids."""
        from perturb_data_lab.canonical.runner import _gene_map_file

        mapping_file = tmp_path / "mapping.tsv"
        mapping_file.write_text("gene1\tENSG0001\ngene2\tENSG0002\n")

        result = _gene_map_file(["gene1", "gene2", "gene3"], mapping_file=str(mapping_file))
        assert result == {"gene1": "ENSG0001", "gene2": "ENSG0002", "gene3": "gene3"}

    def test_file_mapping_comments_skipped(self, tmp_path: Path):
        """Lines starting with # are skipped."""
        from perturb_data_lab.canonical.runner import _gene_map_file

        mapping_file = tmp_path / "mapping.tsv"
        mapping_file.write_text("# comment\n# another\ngene1\tENSG0001\n")

        result = _gene_map_file(["gene1", "gene2"], mapping_file=str(mapping_file))
        assert result == {"gene1": "ENSG0001", "gene2": "gene2"}

    def test_file_mapping_empty_lines_skipped(self, tmp_path: Path):
        """Empty lines are skipped."""
        from perturb_data_lab.canonical.runner import _gene_map_file

        mapping_file = tmp_path / "mapping.tsv"
        mapping_file.write_text("\ngene1\tENSG0001\n\n")

        result = _gene_map_file(["gene1"], mapping_file=str(mapping_file))
        assert result == {"gene1": "ENSG0001"}


# ---------------------------------------------------------------------------
# End-to-end canonicalization tests
# ---------------------------------------------------------------------------


class TestCanonicalizationRunner:
    """End-to-end tests with synthetic parquet files."""

    def test_single_dataset_obs_output(self, tmp_path: Path):
        """Runner produces canonical obs with all must-have columns."""
        raw_obs = tmp_path / "raw-obs.parquet"
        raw_var = tmp_path / "raw-var.parquet"
        sf_path = tmp_path / "size-factor.parquet"
        schema_path = tmp_path / "canonicalization-schema.yaml"
        out_root = tmp_path / "output"

        _make_raw_obs_parquet(raw_obs, [
            {"cell_id": "cell_0", "guide_1": "TP53", "treatment": "Drug_A",
             "cell_type": "B_cells", "cellline": "CL1", "genotype": "WT",
             "batch": "Batch_1", "donor_id": "D001", "passage": 3, "n_features": 1000},
            {"cell_id": "cell_1", "guide_1": "CTRL", "treatment": "DMSO",
             "cell_type": "T_cells", "cellline": "CL2", "genotype": "WT",
             "batch": "Batch_2", "donor_id": "D002", "passage": 5, "n_features": 1200},
        ])
        _make_raw_var_parquet(raw_var, ["gene1", "gene2", "gene3"])
        _make_size_factor_parquet(sf_path, ["cell_0", "cell_1"], [0.95, 1.05])
        _make_minimal_schema(schema_path, dataset_id="test")

        runner = CanonicalizationRunner(
            raw_obs_path=raw_obs,
            raw_var_path=raw_var,
            size_factor_path=sf_path,
            schema_path=schema_path,
            output_root=out_root,
        )
        result = runner.run()

        assert result.dataset_id == "test"
        assert result.obs_rows == 2
        assert result.var_rows == 3

        # Verify obs parquet
        obs = pq.read_table(str(result.obs_path))
        assert set(obs.column_names) >= set(CANONICAL_OBS_MUST_HAVE)
        assert obs.schema.field("global_row_index").type == pa.int64()
        assert obs.schema.field("dataset_index").type == pa.int32()
        assert obs.schema.field("local_row_index").type == pa.int64()
        assert obs.schema.field("size_factor").type == pa.float64()

        # Check specific values
        assert obs.column("cell_id")[0].as_py() == "cell_0"
        assert obs.column("dataset_id")[0].as_py() == "test"
        assert obs.column("dataset_index")[0].as_py() == 0
        assert obs.column("perturb_label")[0].as_py() == "TP53"
        assert obs.column("perturb_type")[0].as_py() == "Drug_A"
        assert obs.column("cell_context")[0].as_py() == "B_cells"
        assert obs.column("cell_line_or_type")[0].as_py() == "CL1"
        assert obs.column("condition")[0].as_py() == "WT"
        assert obs.column("species")[0].as_py() == "human"

        # Safe nullable fields now use real nulls
        assert obs.column("dose")[0].as_py() is None
        assert obs.column("dose_unit")[0].as_py() is None
        assert obs.column("timepoint")[0].as_py() is None
        assert obs.column("timepoint_unit")[0].as_py() is None
        assert obs.column("sex")[0].as_py() == "unknown"

        # Row index columns
        assert obs.column("global_row_index")[0].as_py() == 0
        assert obs.column("local_row_index")[0].as_py() == 0
        assert obs.column("global_row_index")[1].as_py() == 1

        # Size factors
        assert obs.column("size_factor")[0].as_py() == pytest.approx(0.95)

        # No warnings
        assert result.warnings == ()

    def test_var_output_identity_mapping(self, tmp_path: Path):
        """Runner produces canonical var with identity gene mapping."""
        raw_obs = tmp_path / "raw-obs.parquet"
        raw_var = tmp_path / "raw-var.parquet"
        sf_path = tmp_path / "size-factor.parquet"
        schema_path = tmp_path / "canonicalization-schema.yaml"
        out_root = tmp_path / "output"

        _make_raw_obs_parquet(raw_obs, [
            {"cell_id": "cell_0", "guide_1": "TP53", "treatment": "Drug_A",
             "cell_type": "B_cells", "cellline": "CL1", "genotype": "WT",
             "batch": "Batch_1", "donor_id": "D001"},
        ])
        _make_raw_var_parquet(raw_var, ["geneA", "geneB", "geneC"])
        _make_size_factor_parquet(sf_path, ["cell_0"], [1.0])
        _make_minimal_schema(schema_path)

        runner = CanonicalizationRunner(
            raw_obs_path=raw_obs,
            raw_var_path=raw_var,
            size_factor_path=sf_path,
            schema_path=schema_path,
            output_root=out_root,
        )
        result = runner.run()

        var = pq.read_table(str(result.var_path))
        assert set(var.column_names) >= set(CANONICAL_VAR_MUST_HAVE)

        # origin_index
        assert [var.column("origin_index")[i].as_py() for i in range(3)] == ["0", "1", "2"]

        # gene_id == feature_id
        assert var.column("gene_id")[0].as_py() == "geneA"

        # canonical_gene_id == gene_id (identity)
        assert var.column("canonical_gene_id")[0].as_py() == "geneA"

        # global_id assigned consecutively to unique canonical_gene_ids
        assert var.column("global_id")[0].as_py() == "0"

    def test_release_free_output_filenames(self, tmp_path: Path):
        """Canonical outputs use release-free filenames."""
        raw_obs = tmp_path / "raw-obs.parquet"
        raw_var = tmp_path / "raw-var.parquet"
        schema_path = tmp_path / "canonicalization-schema.yaml"
        out_root = tmp_path / "output"

        _make_raw_obs_parquet(raw_obs, [
            {"cell_id": "c1", "guide_1": "g1", "treatment": "Drug_A",
             "cell_type": "B", "cellline": "CL", "genotype": "WT",
             "batch": "B1", "donor_id": "D1"},
        ])
        _make_raw_var_parquet(raw_var, ["geneX"])
        _make_minimal_schema(schema_path)
        sf_path = tmp_path / "size-factor.parquet"
        _make_size_factor_parquet(sf_path, ["c1"], [1.0])

        runner = CanonicalizationRunner(
            raw_obs_path=raw_obs,
            raw_var_path=raw_var,
            size_factor_path=sf_path,
            schema_path=schema_path,
            output_root=out_root,
        )
        result = runner.run()

        assert result.obs_path.name == "canonical-obs.parquet"
        assert result.var_path.name == "canonical-var.parquet"

    def test_missing_required_column_raises(self, tmp_path: Path):
        """Runner raises if a required canonical column is not produced."""
        raw_obs = tmp_path / "raw-obs.parquet"
        raw_var = tmp_path / "raw-var.parquet"
        schema_path = tmp_path / "canonicalization-schema.yaml"
        out_root = tmp_path / "output"

        # Schema missing some required obs columns
        schema = CanonicalizationSchema(
            dataset_id="test",
            status="draft",
            obs_column_mappings=(
                # Only 2 columns — will miss the other 20 required fields
                ObsColumnMapping(canonical_name="cell_id", strategy="null", fallback="NA"),
                ObsColumnMapping(canonical_name="global_row_index", strategy="row-index"),
            ),
            var_column_mappings=(
                VarColumnMapping(canonical_name="origin_index", strategy="passthrough"),
                VarColumnMapping(canonical_name="gene_id", strategy="source-field", source_column="feature_id"),
                VarColumnMapping(canonical_name="canonical_gene_id", strategy="gene-mapping", enabled=False, engine="identity"),
                VarColumnMapping(canonical_name="global_id", strategy="auto"),
            ),
        )
        schema.write_yaml(schema_path)

        _make_raw_obs_parquet(raw_obs, [{"cell_id": "c1"}])
        _make_raw_var_parquet(raw_var, ["g1"])
        sf_path = tmp_path / "size-factor.parquet"
        _make_size_factor_parquet(sf_path, ["c1"], [1.0])

        runner = CanonicalizationRunner(
            raw_obs_path=raw_obs,
            raw_var_path=raw_var,
            size_factor_path=sf_path,
            schema_path=schema_path,
            output_root=out_root,
        )
        with pytest.raises(ValueError, match="Missing required canonical obs columns"):
            runner.run()

    def test_extensible_columns_present(self, tmp_path: Path):
        """Extensible columns declared in schema appear in output."""
        raw_obs = tmp_path / "raw-obs.parquet"
        raw_var = tmp_path / "raw-var.parquet"
        schema_path = tmp_path / "canonicalization-schema.yaml"
        out_root = tmp_path / "output"

        _make_raw_obs_parquet(raw_obs, [
            {"cell_id": "c1", "guide_1": "g1", "treatment": "Drug_A",
             "cell_type": "B", "cellline": "CL", "genotype": "WT",
             "batch": "B1", "donor_id": "D1", "extra_col": "value1"},
        ])
        _make_raw_var_parquet(raw_var, ["geneX"])
        sf_path = tmp_path / "size-factor.parquet"
        _make_size_factor_parquet(sf_path, ["c1"], [1.0])

        base = _make_minimal_schema(schema_path)
        # Re-create with extensible
        schema = CanonicalizationSchema(
            dataset_id=base.dataset_id,
            status=base.status,
            obs_column_mappings=base.obs_column_mappings,
            obs_extensible=(
                ExtensibleColumn(raw_source_column="extra_col", canonical_name="extra"),
            ),
            var_column_mappings=base.var_column_mappings,
            gene_mapping=base.gene_mapping,
        )
        schema.write_yaml(schema_path)

        runner = CanonicalizationRunner(
            raw_obs_path=raw_obs,
            raw_var_path=raw_var,
            size_factor_path=sf_path,
            schema_path=schema_path,
            output_root=out_root,
        )
        result = runner.run()

        obs = pq.read_table(str(result.obs_path))
        assert "extra" in obs.column_names
        assert obs.column("extra")[0].as_py() == "value1"


class TestMultiColumnStrategies:
    def test_coalesce_uses_first_present_value_and_applies_transforms(self, tmp_path: Path):
        result = _run_with_custom_obs_mapping(
            tmp_path,
            rows=[
                {"cell_id": "c1", "target_gene": None, "perturbation": "", "guide_id": "TP53_sg1", "treatment": "Drug_A", "cell_type": "B", "cellline": "CL1", "genotype": "WT", "batch": "B1", "donor_id": "D1"},
                {"cell_id": "c2", "target_gene": None, "perturbation": " NTC ", "guide_id": "NTC_sg2", "treatment": "Drug_B", "cell_type": "B", "cellline": "CL1", "genotype": "WT", "batch": "B1", "donor_id": "D1"},
                {"cell_id": "c3", "target_gene": "KRAS", "perturbation": "NTC", "guide_id": "KRAS_sg3", "treatment": "Drug_C", "cell_type": "B", "cellline": "CL1", "genotype": "WT", "batch": "B1", "donor_id": "D1"},
            ],
            mapping=ObsColumnMapping(
                canonical_name="perturb_label",
                strategy="coalesce",
                source_columns=("target_gene", "perturbation", "guide_id"),
                transforms=(
                    TransformRule(name="strip_guide_suffix", args={}),
                    TransformRule(name="map_control_labels", args={"candidates": ["ntc"]}),
                ),
            ),
        )

        obs = pq.read_table(str(result.obs_path))
        assert obs.column("perturb_label").to_pylist() == ["TP53", "ctrl", "KRAS"]

    def test_join_combines_columns_then_runs_output_transforms(self, tmp_path: Path):
        result = _run_with_custom_obs_mapping(
            tmp_path,
            rows=[
                {"cell_id": "c1", "cellline": "CL1", "treatment": "Drug_A", "time_raw": "24H", "guide_1": "TP53", "cell_type": "B", "genotype": "WT", "batch": "B1", "donor_id": "D1"},
                {"cell_id": "c2", "cellline": "CL2", "treatment": "DMSO", "time_raw": None, "guide_1": "CTRL", "cell_type": "T", "genotype": "WT", "batch": "B2", "donor_id": "D2"},
            ],
            mapping=ObsColumnMapping(
                canonical_name="condition",
                strategy="join",
                source_columns=("cellline", "treatment", "time_raw"),
                separator=":",
                skip_nulls=True,
                transforms=(TransformRule(name="normalize_case", args={"mode": "lower"}),),
            ),
        )

        obs = pq.read_table(str(result.obs_path))
        assert obs.column("condition").to_pylist() == ["cl1:drug_a:24h", "cl2:dmso"]

    def test_template_renders_with_configurable_missing_value_behavior(self, tmp_path: Path):
        result = _run_with_custom_obs_mapping(
            tmp_path,
            rows=[
                {"cell_id": "c1", "cellline": "CL1", "treatment": "Drug_A", "time_raw": "24H", "guide_1": "TP53", "cell_type": "B", "genotype": "WT", "batch": "B1", "donor_id": "D1"},
                {"cell_id": "c2", "cellline": "CL2", "treatment": "DMSO", "time_raw": None, "guide_1": "CTRL", "cell_type": "T", "genotype": "WT", "batch": "B2", "donor_id": "D2"},
            ],
            mapping=ObsColumnMapping(
                canonical_name="condition",
                strategy="template",
                template="{cellline}:{treatment}:{time_raw}",
                missing_value_behavior="literal",
                missing_value="unknown",
                transforms=(TransformRule(name="normalize_case", args={"mode": "lower"}),),
            ),
        )

        obs = pq.read_table(str(result.obs_path))
        assert obs.column("condition").to_pylist() == [
            "cl1:drug_a:24h",
            "cl2:dmso:unknown",
        ]

    def test_conditional_resolves_cases_in_order_then_applies_transforms(self, tmp_path: Path):
        result = _run_with_custom_obs_mapping(
            tmp_path,
            rows=[
                {"cell_id": "c1", "is_control": "yes", "target_gene": None, "guide_id": "NTC_sg1", "guide_1": "NTC_sg1", "treatment": "Drug_A", "cell_type": "B", "cellline": "CL1", "genotype": "WT", "batch": "B1", "donor_id": "D1"},
                {"cell_id": "c2", "is_control": "no", "target_gene": "KRAS", "guide_id": "KRAS_sg2", "guide_1": "KRAS_sg2", "treatment": "Drug_B", "cell_type": "B", "cellline": "CL1", "genotype": "WT", "batch": "B1", "donor_id": "D1"},
                {"cell_id": "c3", "is_control": "no", "target_gene": None, "guide_id": "EGFR_sg3", "guide_1": "EGFR_sg3", "treatment": "Drug_C", "cell_type": "B", "cellline": "CL1", "genotype": "WT", "batch": "B1", "donor_id": "D1"},
            ],
            mapping=ObsColumnMapping(
                canonical_name="perturb_label",
                strategy="conditional",
                cases=(
                    ConditionalCase(
                        source_column="is_control",
                        predicate="equals",
                        value="yes",
                        result_literal="NTC",
                    ),
                    ConditionalCase(
                        source_column="target_gene",
                        predicate="not_null",
                        result_source_column="target_gene",
                    ),
                ),
                default_source_column="guide_id",
                transforms=(
                    TransformRule(name="strip_guide_suffix", args={}),
                    TransformRule(name="map_control_labels", args={"candidates": ["ntc"]}),
                ),
            ),
        )

        obs = pq.read_table(str(result.obs_path))
        assert obs.column("perturb_label").to_pylist() == ["ctrl", "KRAS", "EGFR"]

    def test_missing_multi_column_source_raises_actionable_error(self, tmp_path: Path):
        raw_obs = tmp_path / "raw-obs.parquet"
        raw_var = tmp_path / "raw-var.parquet"
        sf_path = tmp_path / "size-factor.parquet"
        schema_path = tmp_path / "canonicalization-schema.yaml"
        out_root = tmp_path / "output"

        _make_raw_obs_parquet(raw_obs, [
            {"cell_id": "c1", "guide_1": "TP53", "treatment": "Drug_A", "cell_type": "B", "cellline": "CL1", "genotype": "WT", "batch": "B1", "donor_id": "D1"},
        ])
        _make_raw_var_parquet(raw_var, ["gene1"])
        _make_size_factor_parquet(sf_path, ["c1"], [1.0])

        base_schema = _make_minimal_schema(schema_path)
        schema = _replace_obs_mapping(
            base_schema,
            ObsColumnMapping(
                canonical_name="condition",
                strategy="join",
                source_columns=("cellline", "missing_column"),
                separator=":",
            ),
        )
        schema.write_yaml(schema_path)

        runner = CanonicalizationRunner(
            raw_obs_path=raw_obs,
            raw_var_path=raw_var,
            size_factor_path=sf_path,
            schema_path=schema_path,
            output_root=out_root,
        )

        with pytest.raises(ValueError, match="join mapping for 'condition' references missing raw obs columns: missing_column"):
            runner.run()


# ---------------------------------------------------------------------------
# Vocab building tests
# ---------------------------------------------------------------------------


class TestVocabBuilding:
    """Test CanonicalVocab construction and merging."""

    def test_per_dataset_vocab(self, tmp_path: Path):
        """Runner extracts per-dataset vocab from canonical tables."""
        raw_obs = tmp_path / "raw-obs.parquet"
        raw_var = tmp_path / "raw-var.parquet"
        schema_path = tmp_path / "canonicalization-schema.yaml"
        out_root = tmp_path / "output"

        _make_raw_obs_parquet(raw_obs, [
            {"cell_id": "c1", "guide_1": "TP53", "treatment": "Drug_A",
             "cell_type": "B_cells", "cellline": "CL1", "genotype": "WT",
             "batch": "B1", "donor_id": "D1"},
            {"cell_id": "c2", "guide_1": "CTRL", "treatment": "DMSO",
             "cell_type": "T_cells", "cellline": "CL2", "genotype": "WT",
             "batch": "B2", "donor_id": "D2"},
            {"cell_id": "c3", "guide_1": "TP53", "treatment": "Drug_A",
             "cell_type": "B_cells", "cellline": "CL1", "genotype": "Het",
             "batch": "B1", "donor_id": "D1"},
        ])
        _make_raw_var_parquet(raw_var, ["geneA", "geneB", "geneC"])
        sf_path = tmp_path / "size-factor.parquet"
        _make_size_factor_parquet(sf_path, ["c1", "c2", "c3"], [1.0, 1.0, 1.0])
        _make_minimal_schema(schema_path)

        runner = CanonicalizationRunner(
            raw_obs_path=raw_obs,
            raw_var_path=raw_var,
            size_factor_path=sf_path,
            schema_path=schema_path,
            output_root=out_root,
        )
        result = runner.run()

        vocab = result.vocab
        # perturb_label should have 2 unique values
        assert vocab.obs_categories["perturb_label"] == ["CTRL", "TP53"]
        # perturb_type should have 2
        assert vocab.obs_categories["perturb_type"] == ["DMSO", "Drug_A"]
        # cell_context should have 2
        assert "B_cells" in vocab.obs_categories.get("cell_context", [])
        # gene vocab
        assert vocab.global_vocab_size == 3
        assert len(vocab.gene_id_mappings) == 3

    def test_build_canonical_vocab_merges(self):
        """build_canonical_vocab merges two per-dataset vocabs."""
        v1 = CanonicalVocab()
        v1.obs_categories["perturb_label"] = ["CTRL", "TP53"]
        v1.obs_categories["cell_type"] = ["B_cells"]
        v1.global_vocab_size = 100

        v2 = CanonicalVocab()
        v2.obs_categories["perturb_label"] = ["CTRL", "EGFR_KO"]
        v2.obs_categories["cell_type"] = ["T_cells"]
        v2.global_vocab_size = 150

        merged = build_canonical_vocab([v1, v2])
        assert merged.obs_categories["perturb_label"] == ["CTRL", "EGFR_KO", "TP53"]
        assert sorted(merged.obs_categories["cell_type"]) == ["B_cells", "T_cells"]

    def test_build_canonical_vocab_writes_yaml(self, tmp_path: Path):
        """build_canonical_vocab writes the vocab YAML when given an output path."""
        v1 = CanonicalVocab()
        v1.obs_categories["perturb_label"] = ["CTRL"]
        v1.var_categories["canonical_gene_id"] = ["gene1", "gene2"]
        v1.global_vocab_size = 2

        out_path = tmp_path / "canonical-vocab.yaml"
        built = build_canonical_vocab([v1], output_path=out_path)

        assert out_path.exists()
        content = out_path.read_text()
        assert "canonical-vocab" in content
        assert "perturb_label" in content
        assert "CTRL" in content


# ---------------------------------------------------------------------------
# Convenience function test
# ---------------------------------------------------------------------------


class TestRunCanonicalization:
    """Test the convenience entry point."""

    def test_run_canonicalization_returns_result(self, tmp_path: Path):
        """run_canonicalization is equivalent to constructing and running."""
        raw_obs = tmp_path / "raw-obs.parquet"
        raw_var = tmp_path / "raw-var.parquet"
        sf_path = tmp_path / "size-factor.parquet"
        schema_path = tmp_path / "canonicalization-schema.yaml"
        out_root = tmp_path / "output"

        _make_raw_obs_parquet(raw_obs, [
            {"cell_id": "c1", "guide_1": "g1", "treatment": "Drug_A",
             "cell_type": "B", "cellline": "CL", "genotype": "WT",
             "batch": "B1", "donor_id": "D1"},
        ])
        _make_raw_var_parquet(raw_var, ["geneX"])
        _make_size_factor_parquet(sf_path, ["c1"], [1.0])
        _make_minimal_schema(schema_path)

        result = run_canonicalization(
            dataset_id="test",
            raw_obs_path=raw_obs,
            raw_var_path=raw_var,
            size_factor_path=sf_path,
            schema_path=schema_path,
            output_root=out_root,
        )
        assert isinstance(result, CanonicalizationResult)
        assert result.dataset_id == "test"
        assert result.obs_rows == 1
