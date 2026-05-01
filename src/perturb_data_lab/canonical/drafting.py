"""AI-assisted schema drafting helper for canonicalization-schema.yaml generation.

Reads raw obs/var column names and emits a draft ``CanonicalizationSchema``
with heuristic mappings, transform suggestions, and uncertainty flags.

Heuristics are explicit and auditable — no black-box LLM calls.
"""

from __future__ import annotations

import re
from typing import Any

from .contract import (
    CANONICAL_OBS_MUST_HAVE,
    CANONICAL_VAR_MUST_HAVE,
    CanonicalizationSchema,
    ExtensibleColumn,
    GeneMappingConfig,
    ObsColumnMapping,
    TransformRule,
    VarColumnMapping,
)
from ..contracts import MISSING_VALUE_LITERAL

# ---------------------------------------------------------------------------
# Known column-name aliases: canonical field → raw column name candidates
# (ordered by preference — first match wins)
# ---------------------------------------------------------------------------

_CANONICAL_OBS_ALIASES: dict[str, tuple[str, ...]] = {
    "assay": ("assay", "protocol", "technology", "platform", "assay_type"),
    "batch_id": (
        "batch", "batch_id", "batch_name", "sample_batch", "library_batch",
        "sequencing_batch", "plate_id", "pool_id",
    ),
    "cell_context": (
        "cell_type", "cell_context", "cluster", "cell_ontology", "cell_label",
        "cell_type_label", "celltype",
    ),
    "cell_id": (
        "cell_id", "cell_barcode", "barcode", "cell_name", "cell_bc",
        "barcode_sequence",
    ),
    "cell_line_or_type": (
        "cellline", "cell_line", "cell_line_or_type", "cell_line_name",
        "line", "cell_line_id",
    ),
    "condition": (
        "genotype", "condition", "genetic_background", "background",
        "strain", "genetic_condition",
    ),
    "dataset_id": (
        "dataset_id", "study_id", "dataset", "source_dataset",
    ),
    "disease_state": (
        "disease_state", "disease", "health_status", "condition_status",
        "pathology", "disease_status",
    ),
    "donor_id": (
        "donor_id", "donor", "subject_id", "patient_id", "sample_donor_id",
        "individual_id",
    ),
    "dose": (
        "dose", "concentration", "amount", "dose_value", "dosage",
        "dose_concentration",
    ),
    "dose_unit": (
        "dose_unit", "concentration_unit", "unit", "dosage_unit",
    ),
    "perturb_label": (
        "guide_1", "perturb_label", "guide", "sgrna", "guide_id",
        "perturbation", "perturbation_name", "target_gene", "target_label",
    ),
    "perturb_type": (
        "treatment", "perturb_type", "perturbation_type", "intervention_type",
        "moa", "treatment_type", "intervention",
    ),
    "sex": ("sex", "gender", "biological_sex"),
    "size_factor": (
        "size_factor", "size_factors", "sf", "norm_factor",
        "library_size", "total_counts",
    ),
    "species": ("species", "organism", "source_organism"),
    "timepoint": (
        "timepoint", "time", "treatment_time", "exposure_time",
        "duration", "time_point",
    ),
    "timepoint_unit": (
        "timepoint_unit", "time_unit", "duration_unit",
    ),
    "tissue": ("tissue", "organ", "tissue_type", "source_tissue"),
}
"""Canonical obs field name → ordered list of candidate raw column names."""

_CANONICAL_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "origin_index": ("origin_index", "index", "row_index", "feature_index"),
    "gene_id": (
        "feature_id", "gene_id", "gene", "gene_name", "gene_symbol",
        "symbol", "feature_name", "gene_identifier",
    ),
}
"""Canonical var field name → ordered list of candidate raw column names."""

# ---------------------------------------------------------------------------
# Fields that require fixed strategies regardless of column availability
# ---------------------------------------------------------------------------

_ROW_INDEX_FIELDS: frozenset[str] = frozenset({
    "global_row_index",
    "local_row_index",
})

_LITERAL_DEFAULTS: dict[str, str] = {
    "dataset_index": "0",
    "species": "human",
    "tissue": "unknown",
    "assay": "unknown",
    "disease_state": "healthy",
    "sex": "unknown",
}
"""Canonical fields that default to a literal when no raw column is found."""

_NULL_FALLBACKS: dict[str, str] = {
    "dose": MISSING_VALUE_LITERAL,
    "dose_unit": MISSING_VALUE_LITERAL,
    "timepoint": MISSING_VALUE_LITERAL,
    "timepoint_unit": MISSING_VALUE_LITERAL,
    "cell_context": "unknown",
    "cell_line_or_type": "unknown",
    "condition": "WT",
    "batch_id": MISSING_VALUE_LITERAL,
    "donor_id": "unknown",
    "perturb_type": "untreated",
    "perturb_label": MISSING_VALUE_LITERAL,
    "cell_id": MISSING_VALUE_LITERAL,
    "dataset_id": MISSING_VALUE_LITERAL,
}
"""Fallback values when strategy is ``null`` and no literal default applies."""

# ---------------------------------------------------------------------------
# Gene ID pattern detection
# ---------------------------------------------------------------------------

_ENSEMBL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^ENS[A-Z]*G\d+", re.IGNORECASE),
    re.compile(r"^ENSMUSG\d+", re.IGNORECASE),
    re.compile(r"^ENS[A-Z]*T\d+", re.IGNORECASE),
    re.compile(r"^ENSDARG\d+", re.IGNORECASE),
    re.compile(r"^WBGene\d+", re.IGNORECASE),  # WormBase (non-Ensembl but similar)
)


def _looks_ensembl(value: str) -> bool:
    """Return True if *value* looks like an Ensembl-style identifier."""
    for pat in _ENSEMBL_PATTERNS:
        if pat.match(value):
            return True
    return False


# ---------------------------------------------------------------------------
# Column name matching utilities
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    """Lowercase and strip underscores for fuzzy comparison."""
    return s.lower().replace("_", "")


def _exact_match(canonical: str, raw_columns: list[str]) -> str | None:
    """Return the raw column that exactly matches *canonical* (case-insensitive)."""
    cl = canonical.lower()
    for col in raw_columns:
        if col.lower() == cl:
            return col
    return None


def _normalized_match(canonical: str, raw_columns: list[str]) -> str | None:
    """Return the raw column whose normalized form matches *canonical*."""
    canon_norm = _normalize(canonical)
    for col in raw_columns:
        if _normalize(col) == canon_norm:
            return col
    return None


def _substring_match(canonical: str, raw_columns: list[str]) -> str | None:
    """Return a raw column where *canonical* is a substring, or vice versa."""
    canon_lower = canonical.lower()
    for col in raw_columns:
        col_lower = col.lower()
        if canon_lower in col_lower or col_lower in canon_lower:
            return col
    return None


def _alias_match(canonical: str, raw_columns: list[str]) -> str | None:
    """Return the first raw column matching a known alias for *canonical*."""
    aliases = _CANONICAL_OBS_ALIASES.get(canonical, ())
    raw_lower = {col.lower(): col for col in raw_columns}
    for alias in aliases:
        if alias.lower() in raw_lower:
            return raw_lower[alias.lower()]
    return None


def _alias_substring_match(canonical: str, raw_columns: list[str]) -> str | None:
    """Return a raw column that contains a known alias for *canonical* as a substring.

    Used as a fallback when exact alias match fails but raw column names
    contain alias text surrounded by prefixes/suffixes (e.g. ``x_batch``
    matching alias ``batch`` for canonical ``batch_id``).
    """
    aliases = _CANONICAL_OBS_ALIASES.get(canonical, ())
    for col in raw_columns:
        col_lower = col.lower()
        for alias in aliases:
            alias_lower = alias.lower()
            # Check both directions: alias in col or col in alias
            # Also try normalized (no underscores) to catch prefix-suffixed variants
            if alias_lower in col_lower or col_lower in alias_lower:
                return col
            alias_norm = _normalize(alias)
            col_norm = _normalize(col)
            if alias_norm and col_norm and (alias_norm in col_norm or col_norm in alias_norm):
                return col
    return None


def find_obs_column(canonical: str, raw_columns: list[str]) -> str | None:
    """Find the best raw obs column for a canonical field.

    Matching priority:
    1. Exact match (case-insensitive)
    2. Normalized match (strip underscores, lowercase)
    3. Known alias exact match
    4. Known alias substring match
    5. Canonical-name substring match
    6. No match → None
    """
    # 1. Exact match
    match = _exact_match(canonical, raw_columns)
    if match is not None:
        return match

    # 2. Normalized match
    match = _normalized_match(canonical, raw_columns)
    if match is not None:
        return match

    # 3. Known alias exact match
    match = _alias_match(canonical, raw_columns)
    if match is not None:
        return match

    # 4. Known alias substring match
    match = _alias_substring_match(canonical, raw_columns)
    if match is not None:
        return match

    # 5. Canonical-name substring match
    match = _substring_match(canonical, raw_columns)
    if match is not None:
        return match

    return None


# ---------------------------------------------------------------------------
# Transform suggestion heuristics
# ---------------------------------------------------------------------------


def _suggest_transforms(
    canonical: str,
    raw_column: str,
    sampled_values: list[str] | None = None,
) -> tuple[TransformRule, ...]:
    """Suggest transforms based on column name and optional value samples.

    Detection heuristics:
    - Column name ending in ``_sgN`` → suggests ``strip_suffix``
    - Value patterns matching dose notation (e.g. ``100nM``) → suggests
      ``dose_parse`` + ``dose_unit`` for appropriate canonical fields
    - Value patterns matching time notation (e.g. ``24h``) → suggests
      ``timepoint_parse`` + ``timepoint_unit`` for appropriate fields
    """
    transforms: list[TransformRule] = []

    # Suffix detection from column name patterns
    # (e.g. guide_1_sg1, perturbation_label_sg2)
    if "_sg" in raw_column.lower() or canonical == "perturb_label":
        # Only suggest strip_suffix if the column name itself ends with _sgN
        # (this is a column-name heuristic)
        pass  # Column-name-only transform detection is limited; use hints

    # If sampled values are provided, run value-based detection
    if sampled_values:
        _dose_count = 0
        _time_count = 0
        for v in sampled_values[:50]:  # Sample up to 50 values
            if _looks_like_dose(v):
                _dose_count += 1
            if _looks_like_time(v):
                _time_count += 1

        if _dose_count > len(sampled_values[:50]) * 0.5:
            if canonical == "dose":
                transforms.append(TransformRule(name="dose_parse", args={}))
            elif canonical == "dose_unit":
                transforms.append(TransformRule(name="dose_unit", args={}))

        if _time_count > len(sampled_values[:50]) * 0.5:
            if canonical == "timepoint":
                transforms.append(TransformRule(name="timepoint_parse", args={}))
            elif canonical == "timepoint_unit":
                transforms.append(TransformRule(name="timepoint_unit", args={}))

    return tuple(transforms)


_DOSE_RE = re.compile(r"^[0-9.]+\s*(nM|uM|μM|mM|mg/kg|μg/kg|ug/kg)\s*$", re.IGNORECASE)
_TIME_RE = re.compile(r"^[0-9.]+\s*(h|hr|hrs|d|day|days|m|min|mins)\s*$", re.IGNORECASE)


def _looks_like_dose(value: str) -> bool:
    return bool(_DOSE_RE.match(str(value).strip()))


def _looks_like_time(value: str) -> bool:
    return bool(_TIME_RE.match(str(value).strip()))


# ---------------------------------------------------------------------------
# Gene mapping inference
# ---------------------------------------------------------------------------


def _infer_gene_mapping(
    gene_id_column: str | None,
    sampled_gene_ids: list[str] | None = None,
) -> GeneMappingConfig:
    """Infer gene mapping configuration from column name and value samples.

    - If gene IDs look like Ensembl IDs (``ENSG...``), enable ``gget`` engine.
    - If gene IDs look like gene symbols (alphanumeric, no ENS prefix),
      use ``identity`` engine.
    - If no information is available, default to ``identity``.
    """
    if sampled_gene_ids:
        ensembl_count = sum(1 for gid in sampled_gene_ids if _looks_ensembl(str(gid)))
        total = len(sampled_gene_ids)
        if total > 0 and ensembl_count / total > 0.5:
            return GeneMappingConfig(
                enabled=True,
                engine="gget",
                source_namespace="ensembl_gene_id",
                target_namespace="gene_symbol",
            )

    # Check column name for hints
    if gene_id_column is not None:
        col_lower = gene_id_column.lower()
        if any(pat.search(col_lower) for pat in _ENSEMBL_PATTERNS):
            return GeneMappingConfig(
                enabled=True,
                engine="gget",
                source_namespace="ensembl_gene_id",
                target_namespace="gene_symbol",
            )
        if "ensembl" in col_lower or "ensg" in col_lower:
            return GeneMappingConfig(
                enabled=True,
                engine="gget",
                source_namespace="ensembl_gene_id",
                target_namespace="gene_symbol",
            )

    # Default: identity (gene symbols)
    return GeneMappingConfig(
        enabled=True,
        engine="identity",
        source_namespace="gene_symbol",
        target_namespace="gene_symbol",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def draft_canonicalization_schema(
    dataset_id: str,
    obs_columns: list[str],
    var_columns: list[str],
    hints: dict[str, Any] | None = None,
) -> CanonicalizationSchema:
    """Generate a draft ``CanonicalizationSchema`` from raw column names.

    Parameters
    ----------
    dataset_id : str
        Stable dataset identifier (e.g. ``dummy_00``).
    obs_columns : list of str
        Raw obs column names available from the dataset.
    var_columns : list of str
        Raw var column names available from the dataset.
    hints : dict or None
        Optional overrides and sampled values:
        - ``dataset_index``: literal value for dataset_index
        - ``species``: literal value override
        - ``tissue``: literal value override
        - ``assay``: literal value override
        - ``disease_state``: literal value override
        - ``sex``: literal value override
        - ``sampled_gene_ids``: list of sample gene_id values
        - ``sampled_obs_values``: dict of column_name → list of sample values

    Returns
    -------
    CanonicalizationSchema
        A draft schema with ``status="draft"`` and uncertainty notes.
    """
    hints = hints or {}
    notes: list[str] = []

    # --- Obs mappings ---
    obs_mappings: list[ObsColumnMapping] = []
    mapped_obs_raw: set[str] = set()

    for canonical_name in CANONICAL_OBS_MUST_HAVE:
        mapping = _draft_obs_mapping(canonical_name, obs_columns, hints, notes)
        obs_mappings.append(mapping)
        if mapping.source_column is not None:
            mapped_obs_raw.add(mapping.source_column)

    # --- Var mappings ---
    var_mappings: list[VarColumnMapping] = []
    mapped_var_raw: set[str] = set()

    gene_id_col: str | None = None
    for canonical_name in CANONICAL_VAR_MUST_HAVE:
        mapping = _draft_var_mapping(canonical_name, var_columns, hints, notes)
        var_mappings.append(mapping)
        if mapping.source_column is not None:
            mapped_var_raw.add(mapping.source_column)
        if canonical_name == "gene_id":
            gene_id_col = mapping.source_column

    # --- Gene mapping inference ---
    sampled_gene_ids = hints.get("sampled_gene_ids")
    gene_mapping = _infer_gene_mapping(gene_id_col, sampled_gene_ids)
    if gene_mapping.engine == "identity" and not sampled_gene_ids:
        notes.append(
            "[uncertain] gene_mapping: no sampled gene_id values provided; "
            "assuming gene symbol identity mapping. If gene IDs are "
            "Ensembl-style, set gene_mapping.engine='gget'."
        )
    elif gene_mapping.engine == "identity":
        notes.append(
            "[inferred] gene_mapping: sampled gene_id values look like "
            "gene symbols → using identity mapping."
        )
    elif gene_mapping.engine == "gget":
        notes.append(
            "[inferred] gene_mapping: sampled gene_id values look like "
            "Ensembl IDs → using gget mapping (requires gget installed)."
        )

    # --- Extensible columns ---
    obs_extensible: list[ExtensibleColumn] = []
    for col in obs_columns:
        if col not in mapped_obs_raw and col.lower() not in {"cell_id"}:
            obs_extensible.append(ExtensibleColumn(raw_source_column=col))
    if obs_extensible:
        ext_names = [e.raw_source_column for e in obs_extensible]
        notes.append(
            f"[info] {len(obs_extensible)} obs extensible columns auto-detected: "
            f"{', '.join(ext_names)}. Review and remove noise columns before "
            f"materializing."
        )

    var_extensible: list[ExtensibleColumn] = []
    for col in var_columns:
        if col not in mapped_var_raw and col.lower() not in {"origin_index"}:
            var_extensible.append(ExtensibleColumn(raw_source_column=col))
    if var_extensible:
        ext_names = [e.raw_source_column for e in var_extensible]
        notes.append(
            f"[info] {len(var_extensible)} var extensible columns auto-detected: "
            f"{', '.join(ext_names)}."
        )

    # --- Check which required fields are still null/filled ---
    null_fields = [
        m.canonical_name for m in obs_mappings
        if m.strategy == "null" and m.fallback == MISSING_VALUE_LITERAL
    ]
    if null_fields:
        notes.append(
            f"[action-needed] {len(null_fields)} obs fields have no heuristic "
            f"mapping and will be filled with NA: {', '.join(null_fields)}. "
            f"Review these fields before advancing to materialization."
        )

    literal_fields = [
        m for m in obs_mappings
        if m.strategy == "literal"
    ]
    for lf in literal_fields:
        notes.append(
            f"[review] '{lf.canonical_name}' set to literal "
            f"'{lf.literal_value}'. Verify this value is correct for "
            f"this dataset."
        )

    schema = CanonicalizationSchema(
        dataset_id=dataset_id,
        status="draft",
        description=f"Auto-drafted canonicalization schema for {dataset_id}",
        obs_column_mappings=tuple(obs_mappings),
        obs_extensible=tuple(obs_extensible),
        var_column_mappings=tuple(var_mappings),
        var_extensible=tuple(var_extensible),
        gene_mapping=gene_mapping,
        notes=tuple(notes),
    )
    schema.validate()
    return schema


# ---------------------------------------------------------------------------
# Per-field drafting helpers
# ---------------------------------------------------------------------------


def _draft_obs_mapping(
    canonical_name: str,
    obs_columns: list[str],
    hints: dict[str, Any],
    notes: list[str],
) -> ObsColumnMapping:
    """Draft a single obs column mapping."""

    # Row-index fields
    if canonical_name in _ROW_INDEX_FIELDS:
        return ObsColumnMapping(
            canonical_name=canonical_name,
            strategy="row-index",
        )

    # Literal defaults (with hint overrides)
    if canonical_name in _LITERAL_DEFAULTS:
        literal_value = hints.get(canonical_name, _LITERAL_DEFAULTS[canonical_name])
        return ObsColumnMapping(
            canonical_name=canonical_name,
            strategy="literal",
            literal_value=str(literal_value),
        )

    # Size factor is always source-field (numeric, handled by runner)
    if canonical_name == "size_factor":
        return ObsColumnMapping(
            canonical_name=canonical_name,
            strategy="source-field",
            source_column="size_factor",
        )

    # Special case: cell_id — passthrough from raw (always present per contract)
    if canonical_name == "cell_id":
        match = find_obs_column(canonical_name, obs_columns)
        if match:
            return ObsColumnMapping(
                canonical_name=canonical_name,
                strategy="source-field",
                source_column=match,
                fallback=_NULL_FALLBACKS.get(canonical_name, MISSING_VALUE_LITERAL),
            )
        # If cell_id not found in obs_columns, flag it
        notes.append(
            f"[uncertain] 'cell_id' not found in raw obs columns; "
            f"using passthrough with fallback."
        )
        return ObsColumnMapping(
            canonical_name=canonical_name,
            strategy="passthrough",
            fallback=_NULL_FALLBACKS.get(canonical_name, MISSING_VALUE_LITERAL),
        )

    # For dataset_id: always passthrough if present, else source-field
    if canonical_name == "dataset_id":
        match = find_obs_column(canonical_name, obs_columns)
        if match:
            return ObsColumnMapping(
                canonical_name=canonical_name,
                strategy="source-field",
                source_column=match,
                fallback=_NULL_FALLBACKS.get(canonical_name, MISSING_VALUE_LITERAL),
            )
        notes.append(
            f"[uncertain] 'dataset_id' not found in raw obs columns; "
            f"will fall back to schema dataset_id."
        )
        return ObsColumnMapping(
            canonical_name=canonical_name,
            strategy="null",
            fallback=_NULL_FALLBACKS.get(canonical_name, MISSING_VALUE_LITERAL),
        )

    # General heuristics for remaining fields
    match = find_obs_column(canonical_name, obs_columns)

    if match is not None:
        # Heuristic match found
        confidence = "exact" if match.lower() == canonical_name.lower() else "heuristic"
        if confidence == "heuristic":
            notes.append(
                f"[heuristic] '{canonical_name}' → '{match}' "
                f"(not an exact match; verify this mapping)."
            )

        sampled_values = None
        if hints and "sampled_obs_values" in hints:
            sv = hints["sampled_obs_values"]
            if isinstance(sv, dict):
                sampled_values = sv.get(match)

        transforms = _suggest_transforms(canonical_name, match, sampled_values)
        if transforms:
            t_names = [t.name for t in transforms]
            notes.append(
                f"[suggested] '{canonical_name}' → transforms: {t_names} "
                f"(auto-detected from value patterns; verify before applying)."
            )

        return ObsColumnMapping(
            canonical_name=canonical_name,
            strategy="source-field",
            source_column=match,
            transforms=transforms,
            fallback=_NULL_FALLBACKS.get(canonical_name, MISSING_VALUE_LITERAL),
        )

    # No match found — use null strategy
    notes.append(
        f"[no-match] '{canonical_name}' has no heuristic mapping in raw "
        f"columns; using null/fallback. Review and provide a "
        f"source column or literal value."
    )
    return ObsColumnMapping(
        canonical_name=canonical_name,
        strategy="null",
        fallback=_NULL_FALLBACKS.get(canonical_name, MISSING_VALUE_LITERAL),
    )


def _draft_var_mapping(
    canonical_name: str,
    var_columns: list[str],
    hints: dict[str, Any],
    notes: list[str],
) -> VarColumnMapping:
    """Draft a single var column mapping."""

    # origin_index — passthrough if present
    if canonical_name == "origin_index":
        # Check for alias match first
        aliases = _CANONICAL_VAR_ALIASES.get(canonical_name, ())
        raw_lower = {col.lower(): col for col in var_columns}
        for alias in aliases:
            if alias.lower() in raw_lower:
                return VarColumnMapping(
                    canonical_name=canonical_name,
                    strategy="passthrough",
                    source_column=raw_lower[alias.lower()],
                )
        # Default: passthrough by canonical name
        return VarColumnMapping(
            canonical_name=canonical_name,
            strategy="passthrough",
        )

    # gene_id — map from feature_id or similar
    if canonical_name == "gene_id":
        # Use multi-step matching: exact → normalized → alias exact → alias substring
        # 1. Exact match
        match = _exact_match(canonical_name, var_columns)
        if match is not None:
            return VarColumnMapping(
                canonical_name=canonical_name,
                strategy="source-field",
                source_column=match,
                fallback=MISSING_VALUE_LITERAL,
            )

        # 2. Normalized match (strip underscores)
        match = _normalized_match(canonical_name, var_columns)
        if match is not None:
            return VarColumnMapping(
                canonical_name=canonical_name,
                strategy="source-field",
                source_column=match,
                fallback=MISSING_VALUE_LITERAL,
            )

        # 3. Known alias exact match
        aliases = _CANONICAL_VAR_ALIASES.get(canonical_name, ())
        raw_lower = {col.lower(): col for col in var_columns}
        for alias in aliases:
            if alias.lower() in raw_lower:
                return VarColumnMapping(
                    canonical_name=canonical_name,
                    strategy="source-field",
                    source_column=raw_lower[alias.lower()],
                    fallback=MISSING_VALUE_LITERAL,
                )

        # 4. Alias substring / normalized match
        for col in var_columns:
            col_lower = col.lower()
            col_norm = _normalize(col)
            for alias in aliases:
                alias_lower = alias.lower()
                alias_norm = _normalize(alias)
                if (alias_lower in col_lower or col_lower in alias_lower or
                        (alias_norm and col_norm and (alias_norm in col_norm or col_norm in alias_norm))):
                    notes.append(
                        f"[heuristic] var 'gene_id' → '{col}' "
                        f"(alias substring/normalized match; verify)."
                    )
                    return VarColumnMapping(
                        canonical_name=canonical_name,
                        strategy="source-field",
                        source_column=col,
                        fallback=MISSING_VALUE_LITERAL,
                    )

        # 5. Canonical-name substring match as last resort
        match = _substring_match(canonical_name, var_columns)
        if match:
            notes.append(
                f"[heuristic] var 'gene_id' → '{match}' "
                f"(substring match; verify)."
            )
            return VarColumnMapping(
                canonical_name=canonical_name,
                strategy="source-field",
                source_column=match,
                fallback=MISSING_VALUE_LITERAL,
            )

        notes.append(
            "[no-match] var 'gene_id' has no heuristic mapping; "
            "using null strategy. Must be manually resolved."
        )
        return VarColumnMapping(
            canonical_name=canonical_name,
            strategy="null",
            fallback=MISSING_VALUE_LITERAL,
        )

    # canonical_gene_id — always gene-mapping
    if canonical_name == "canonical_gene_id":
        return VarColumnMapping(
            canonical_name=canonical_name,
            strategy="gene-mapping",
            enabled=False,  # Will be set by top-level gene_mapping
            engine="identity",
        )

    # global_id — always auto (assigned by runner)
    if canonical_name == "global_id":
        return VarColumnMapping(
            canonical_name=canonical_name,
            strategy="auto",
        )

    # Fallback for any unexpected var field
    notes.append(f"[no-match] var '{canonical_name}' has no drafting logic.")
    return VarColumnMapping(
        canonical_name=canonical_name,
        strategy="null",
        fallback=MISSING_VALUE_LITERAL,
    )
