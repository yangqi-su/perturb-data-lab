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


# ---------------------------------------------------------------------------
# Transform application tests
# ---------------------------------------------------------------------------


class TestTransformApplication:
    """Test that individual transforms work as expected through the runner."""

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

    def test_timepoint_parse_extracts_numeric(self):
        """timepoint_parse extracts the numeric part."""
        result = CanonicalizationRunner._apply_transforms(
            "24h",
            transforms=(TransformRule(name="timepoint_parse", args={}),),
        )
        assert result == "24"

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
            release_id="test-release",
        )
        result = runner.run()

        assert result.dataset_id == "test"
        assert result.obs_rows == 2
        assert result.var_rows == 3

        # Verify obs parquet
        obs = pq.read_table(str(result.obs_path))
        assert set(obs.column_names) >= set(CANONICAL_OBS_MUST_HAVE)

        # Check specific values
        assert obs.column("cell_id")[0].as_py() == "cell_0"
        assert obs.column("dataset_id")[0].as_py() == "test"
        assert obs.column("perturb_label")[0].as_py() == "TP53"
        assert obs.column("perturb_type")[0].as_py() == "Drug_A"
        assert obs.column("cell_context")[0].as_py() == "B_cells"
        assert obs.column("cell_line_or_type")[0].as_py() == "CL1"
        assert obs.column("condition")[0].as_py() == "WT"
        assert obs.column("species")[0].as_py() == "human"

        # NA-filled null strategy columns
        assert obs.column("dose")[0].as_py() == "NA"
        assert obs.column("sex")[0].as_py() == "unknown"

        # Row index columns
        assert obs.column("global_row_index")[0].as_py() == "0"
        assert obs.column("local_row_index")[0].as_py() == "0"
        assert obs.column("global_row_index")[1].as_py() == "1"

        # Size factors
        assert obs.column("size_factor")[0].as_py() == "0.95"

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
            release_id="test-release",
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

    def test_custom_release_id(self, tmp_path: Path):
        """Custom release_id is used in output filenames."""
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
            release_id="my-custom-release",
        )
        result = runner.run()

        assert "my-custom-release-canonical-obs.parquet" in str(result.obs_path)
        assert "my-custom-release-canonical-var.parquet" in str(result.var_path)

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
            release_id="test-release",
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
            release_id="test-release",
        )
        result = runner.run()

        obs = pq.read_table(str(result.obs_path))
        assert "extra" in obs.column_names
        assert obs.column("extra")[0].as_py() == "value1"


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
            release_id="test-release",
        )
        assert isinstance(result, CanonicalizationResult)
        assert result.dataset_id == "test"
        assert result.obs_rows == 1
