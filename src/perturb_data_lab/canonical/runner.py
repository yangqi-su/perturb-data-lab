"""Canonicalization runner: transforms raw obs/var sidecars into canonical parquets.

``CanonicalizationRunner`` reads raw parquet sidecars and a per-dataset
``canonicalization-schema.yaml``, applies the declared column mappings and
transforms, and writes ``{release_id}-canonical-obs.parquet`` and
``{release_id}-canonical-var.parquet``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts import CONTRACT_VERSION, MISSING_VALUE_LITERAL
from ..inspectors import transforms as _xforms
from .contract import (
    CANONICAL_OBS_MUST_HAVE,
    CANONICAL_VAR_MUST_HAVE,
    CanonicalVocab,
    CanonicalizationSchema,
    ExtensibleColumn,
    ObsColumnMapping,
    VarColumnMapping,
)

__all__ = [
    "CanonicalizationRunner",
    "CanonicalizationResult",
    "run_canonicalization",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transform dispatch registry
# ---------------------------------------------------------------------------

_transform_registry: dict[str, Callable[..., str]] = {
    "strip_prefix": _xforms.strip_prefix,
    "strip_suffix": _xforms.strip_suffix,
    "regex_sub": _xforms.regex_sub,
    "normalize_case": _xforms.normalize_case,
    "dose_parse": _xforms.dose_parse,
    "dose_unit": _xforms.dose_unit,
    "timepoint_parse": _xforms.timepoint_parse,
    "timepoint_unit": _xforms.timepoint_unit,
    "map_values": _xforms.map_values,
    "split_on_delimiter": _xforms.split_on_delimiter,
}


# ---------------------------------------------------------------------------
# Gene mapping engines
# ---------------------------------------------------------------------------


def _gene_map_identity(gene_ids: list[str], **_kwargs: Any) -> dict[str, str]:
    """Identity mapping: each ``gene_id`` maps to itself."""
    return {gid: gid for gid in gene_ids}


def _gene_map_file(gene_ids: list[str], mapping_file: str, **_: Any) -> dict[str, str]:
    """Read a tab-separated mapping file and return ``gene_id → canonical_gene_id``."""
    mapping: dict[str, str] = {}
    with open(mapping_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                mapping[parts[0].strip()] = parts[1].strip()
    result: dict[str, str] = {}
    for gid in gene_ids:
        result[gid] = mapping.get(gid, gid)
    return result


def _gene_map_gget(gene_ids: list[str], **kwargs: Any) -> dict[str, str]:
    """Use ``gget.convert()`` to map gene identifiers.

    Falls back to identity mapping if gget is unavailable or if any
    conversion fails.
    """
    try:
        import gget  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("gget not installed; falling back to identity gene mapping")
        return _gene_map_identity(gene_ids)

    source_ns = kwargs.get("source_namespace", "gene_symbol")
    target_ns = kwargs.get("target_namespace", "gene_symbol")

    if source_ns == target_ns:
        return _gene_map_identity(gene_ids)

    # gget.convert takes lists and returns dict mapping
    try:
        result_map = gget.convert(gene_ids, source=source_ns, target=target_ns)
        if isinstance(result_map, dict):
            return {
                gid: str(result_map.get(gid, gid))
                for gid in gene_ids
            }
    except Exception as exc:
        logger.warning(
            "gget.convert failed for %d gene IDs (%s); falling back to identity",
            len(gene_ids),
            exc,
        )

    return _gene_map_identity(gene_ids)


_GENE_MAP_ENGINES: dict[str, Callable[..., dict[str, str]]] = {
    "identity": _gene_map_identity,
    "mapping_file": _gene_map_file,
    "gget": _gene_map_gget,
}


# ---------------------------------------------------------------------------
# CanonicalizationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalizationResult:
    """Summary of a single-dataset canonicalization run."""

    dataset_id: str
    release_id: str
    obs_path: Path
    var_path: Path
    obs_rows: int
    var_rows: int
    vocab: CanonicalVocab = field(default_factory=CanonicalVocab)
    gene_mapping_used: str = "identity"
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# CanonicalizationRunner
# ---------------------------------------------------------------------------


class CanonicalizationRunner:
    """Transform raw obs/var sidecars into canonical parquet files.

    Parameters
    ----------
    raw_obs_path : str | Path
        Path to the raw obs sidecar (``{release_id}-raw-obs.parquet``).
    raw_var_path : str | Path
        Path to the raw var sidecar (``{release_id}-raw-var.parquet``).
    size_factor_path : str | Path | None
        Path to the size-factor sidecar (``{release_id}-size-factor.parquet``).
        When ``None``, size_factor defaults to 1.0.
    schema_path : str | Path
        Path to the per-dataset ``canonicalization-schema.yaml``.
    output_root : str | Path
        Directory where canonical parquets will be written.
    release_id : str | None
        Release identifier used in output filenames.  Defaults to
        ``{dataset_id}-release``.
    """

    def __init__(
        self,
        raw_obs_path: str | Path,
        raw_var_path: str | Path,
        size_factor_path: str | Path | None,
        schema_path: str | Path,
        output_root: str | Path,
        release_id: str | None = None,
    ):
        self._raw_obs_path = Path(raw_obs_path)
        self._raw_var_path = Path(raw_var_path)
        self._size_factor_path = Path(size_factor_path) if size_factor_path else None
        self._schema_path = Path(schema_path)
        self._output_root = Path(output_root)

        self._schema = CanonicalizationSchema.from_yaml_file(self._schema_path)
        self._release_id = release_id or f"{self._schema.dataset_id}-release"

        self._warnings: list[str] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> CanonicalizationResult:
        """Execute the canonicalization and return a result summary."""
        self._warnings = []
        self._output_root.mkdir(parents=True, exist_ok=True)

        # --- Load raw sidecars ---
        obs_df, var_df, size_factors = self._load_sidecars()

        # --- Canonicalize obs ---
        obs_table = self._canonicalize_obs(obs_df, size_factors)
        obs_path = self._output_root / f"{self._release_id}-canonical-obs.parquet"
        pq.write_table(obs_table, obs_path)
        logger.info("Wrote canonical obs: %s (%d rows)", obs_path, obs_table.num_rows)

        # --- Canonicalize var ---
        var_table = self._canonicalize_var(var_df)
        var_path = self._output_root / f"{self._release_id}-canonical-var.parquet"
        pq.write_table(var_table, var_path)
        logger.info("Wrote canonical var: %s (%d rows)", var_path, var_table.num_rows)

        # --- Build per-dataset vocab ---
        vocab = self._build_vocab(obs_table, var_table)

        result = CanonicalizationResult(
            dataset_id=self._schema.dataset_id,
            release_id=self._release_id,
            obs_path=obs_path,
            var_path=var_path,
            obs_rows=obs_table.num_rows,
            var_rows=var_table.num_rows,
            vocab=vocab,
            gene_mapping_used=self._schema.gene_mapping.engine,
            warnings=tuple(self._warnings),
        )
        return result

    # ------------------------------------------------------------------
    # Sidecar loading
    # ------------------------------------------------------------------

    def _load_sidecars(self) -> tuple[dict[str, list[Any]], dict[str, list[Any]], list[float]]:
        """Load raw obs, raw var, and size-factor parquets into dict-of-columns form.

        Returns
        -------
        (obs_columns, var_columns, size_factors)
            ``obs_columns`` is a dict mapping column name → list of values.
            ``raw_fields`` JSON is parsed and merged into the dict.
            ``var_columns`` is the same shape for var.
            ``size_factors`` is a list of float values in cell order.
        """
        # --- Raw obs ---
        obs_table = pq.read_table(str(self._raw_obs_path))
        obs_columns: dict[str, list[Any]] = {}
        for col_name in obs_table.column_names:
            obs_columns[col_name] = obs_table.column(col_name).to_pylist()

        # Parse raw_fields JSON and merge
        if "raw_fields" in obs_columns:
            raw_json_list = obs_columns.pop("raw_fields")
            for i, raw_str in enumerate(raw_json_list):
                if raw_str:
                    obj = json.loads(raw_str)
                    for k, v in obj.items():
                        if k not in obs_columns:
                            obs_columns[k] = [None] * obs_table.num_rows
                        obs_columns[k][i] = v
                else:
                    # Ensure all existing nested keys have None at this row
                    # so array lengths stay consistent
                    pass
        n_obs = obs_table.num_rows

        # --- Size factors ---
        size_factors: list[float]
        if self._size_factor_path and self._size_factor_path.exists():
            sf_table = pq.read_table(str(self._size_factor_path))
            # Verify cell_id ordering matches raw obs
            sf_cell_ids = sf_table.column("cell_id").to_pylist()
            obs_cell_ids = obs_columns.get("cell_id", [])
            if sf_cell_ids != obs_cell_ids:
                # Attempt to align by cell_id
                sf_map = dict(zip(sf_cell_ids, sf_table.column("size_factor").to_pylist()))
                size_factors = [float(sf_map.get(cid, 1.0)) for cid in obs_cell_ids]
            else:
                size_factors = [float(v) for v in sf_table.column("size_factor").to_pylist()]
        else:
            size_factors = [1.0] * n_obs

        # --- Raw var ---
        var_table = pq.read_table(str(self._raw_var_path))
        var_columns: dict[str, list[Any]] = {}
        for col_name in var_table.column_names:
            var_columns[col_name] = var_table.column(col_name).to_pylist()

        # Parse raw_var JSON and merge
        if "raw_var" in var_columns:
            raw_json_list = var_columns.pop("raw_var")
            for i, raw_str in enumerate(raw_json_list):
                if raw_str:
                    obj = json.loads(raw_str)
                    for k, v in obj.items():
                        if k not in var_columns:
                            var_columns[k] = [None] * var_table.num_rows
                        var_columns[k][i] = v

        return obs_columns, var_columns, size_factors

    # ------------------------------------------------------------------
    # Obs canonicalization
    # ------------------------------------------------------------------

    def _canonicalize_obs(
        self,
        obs_raw: dict[str, list[Any]],
        size_factors: list[float],
    ) -> pa.Table:
        """Build a canonical obs Parquet table from raw columns and a schema."""
        schema = self._schema
        n_rows = len(next(iter(obs_raw.values()), []))
        columns: dict[str, pa.Array] = {}

        for mapping in schema.obs_column_mappings:
            col = self._resolve_obs_column(mapping, obs_raw, size_factors, n_rows)
            columns[mapping.canonical_name] = col

        # Extensible columns
        for ext in schema.obs_extensible:
            raw_col = obs_raw.get(ext.raw_source_column, [])
            if raw_col:
                values = [str(v) if v is not None else MISSING_VALUE_LITERAL for v in raw_col]
            else:
                values = [MISSING_VALUE_LITERAL] * n_rows
                self._warnings.append(
                    f"obs extensible column '{ext.canonical_name}' not found in "
                    f"raw sidecar; filling with {MISSING_VALUE_LITERAL}"
                )
            columns[ext.canonical_name] = pa.array(values, type=pa.string())

        # Validate required columns
        available = set(columns.keys())
        missing = CanonicalizationRunner._default_obs_schema().missing_required(available)
        if missing:
            raise ValueError(
                f"Missing required canonical obs columns: {missing}. "
                f"Schema at {self._schema_path} must cover all must-have fields."
            )

        return pa.table(columns)

    def _resolve_obs_column(
        self,
        mapping: ObsColumnMapping,
        obs_raw: dict[str, list[Any]],
        size_factors: list[float],
        n_rows: int,
    ) -> pa.Array:
        """Resolve a single obs column mapping to a PyArrow string array."""
        strategy = mapping.strategy

        if strategy == "literal":
            value = mapping.literal_value if mapping.literal_value is not None else mapping.fallback
            return pa.array([value] * n_rows, type=pa.string())

        if strategy == "passthrough":
            # Copy the column as-is; must exist in raw data
            if mapping.canonical_name in obs_raw:
                raw_vals = obs_raw[mapping.canonical_name]
                return pa.array(
                    [str(v) if v is not None else mapping.fallback for v in raw_vals],
                    type=pa.string(),
                )
            # Try the source_column if canonical_name not found
            src = mapping.source_column or mapping.canonical_name
            raw_vals = obs_raw.get(src, [])
            if raw_vals:
                return pa.array(
                    [str(v) if v is not None else mapping.fallback for v in raw_vals],
                    type=pa.string(),
                )
            self._warnings.append(
                f"passthrough column '{mapping.canonical_name}' not found; "
                f"filling with {mapping.fallback}"
            )
            return pa.array([mapping.fallback] * n_rows, type=pa.string())

        if strategy == "row-index":
            return pa.array([str(i) for i in range(n_rows)], type=pa.string())

        if strategy == "null":
            return pa.array([mapping.fallback] * n_rows, type=pa.string())

        if strategy == "source-field":
            if mapping.canonical_name == "size_factor":
                # Special case: size_factor is numeric, stored separately
                return pa.array(
                    [str(sf) for sf in size_factors],
                    type=pa.string(),
                )

            src_col = mapping.source_column or mapping.canonical_name
            raw_vals = obs_raw.get(src_col, [])
            if not raw_vals:
                self._warnings.append(
                    f"source-field '{mapping.canonical_name}' references missing "
                    f"column '{src_col}'; filling with {mapping.fallback}"
                )
                return pa.array([mapping.fallback] * n_rows, type=pa.string())

            resolved: list[str] = []
            for v in raw_vals:
                resolved.append(
                    self._apply_transforms(
                        str(v) if v is not None else MISSING_VALUE_LITERAL,
                        mapping.transforms,
                        fallback=mapping.fallback,
                    )
                )
            return pa.array(resolved, type=pa.string())

        raise ValueError(f"Unhandled obs strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # Var canonicalization
    # ------------------------------------------------------------------

    def _canonicalize_var(
        self,
        var_raw: dict[str, list[Any]],
    ) -> pa.Table:
        """Build a canonical var Parquet table from raw columns and schema."""
        schema = self._schema
        n_rows = len(next(iter(var_raw.values()), []))
        columns: dict[str, pa.Array] = {}

        # Collect gene_ids first (needed for gene mapping)
        gene_ids: list[str] = []
        gene_id_mapping: VarColumnMapping | None = None
        for mapping in schema.var_column_mappings:
            if mapping.canonical_name == "gene_id":
                gene_id_mapping = mapping
                break

        # Resolve raw gene_ids
        if gene_id_mapping is not None:
            src = gene_id_mapping.source_column or "gene_id"
            raw_vals = var_raw.get(src, [])
            for v in raw_vals:
                gene_ids.append(str(v) if v is not None else MISSING_VALUE_LITERAL)
        else:
            gene_ids = [MISSING_VALUE_LITERAL] * n_rows

        # Run gene mapping to get canonical_gene_id → gene_id mapping
        gene_map: dict[str, str] = {}
        mapping_config = schema.gene_mapping
        if mapping_config.enabled and mapping_config.engine != "identity":
            engine_fn = _GENE_MAP_ENGINES.get(mapping_config.engine, _gene_map_identity)
            gene_map = engine_fn(
                list(set(gene_ids)),
                mapping_file=mapping_config.mapping_file or "",
                source_namespace=mapping_config.source_namespace,
                target_namespace=mapping_config.target_namespace,
            )
        else:
            gene_map = {gid: gid for gid in gene_ids}

        # Assign global_ids (consecutive ints to unique canonical_gene_ids)
        # Deterministic: sorted by canonical_gene_id
        unique_canonical = sorted(set(gene_map.values()))
        canonical_to_global: dict[str, int] = {
            cg: idx for idx, cg in enumerate(unique_canonical)
        }

        # Resolve each var column
        for mapping in schema.var_column_mappings:
            if mapping.canonical_name == "origin_index":
                # passthrough or source-field from raw var
                col_vals: list[str] = []
                if mapping.strategy == "passthrough":
                    raw_vals = var_raw.get("origin_index", [])
                    col_vals = [
                        str(v) if v is not None else mapping.fallback
                        for v in raw_vals
                    ]
                elif mapping.strategy == "source-field":
                    src = mapping.source_column or "origin_index"
                    raw_vals = var_raw.get(src, [])
                    col_vals = [
                        self._apply_transforms(
                            str(v) if v is not None else MISSING_VALUE_LITERAL,
                            mapping.transforms,
                            fallback=mapping.fallback,
                        )
                        for v in raw_vals
                    ]
                columns["origin_index"] = pa.array(col_vals, type=pa.string())

            elif mapping.canonical_name == "gene_id":
                columns["gene_id"] = pa.array(gene_ids, type=pa.string())

            elif mapping.canonical_name == "canonical_gene_id":
                mapped_ids = [gene_map.get(gid, gid) for gid in gene_ids]
                columns["canonical_gene_id"] = pa.array(mapped_ids, type=pa.string())

            elif mapping.canonical_name == "global_id":
                global_ids = [
                    str(canonical_to_global.get(gene_map.get(gid, gid), -1))
                    for gid in gene_ids
                ]
                columns["global_id"] = pa.array(global_ids, type=pa.string())

            elif mapping.strategy == "source-field":
                src = mapping.source_column or mapping.canonical_name
                raw_vals = var_raw.get(src, [])
                resolved = [
                    self._apply_transforms(
                        str(v) if v is not None else MISSING_VALUE_LITERAL,
                        mapping.transforms,
                        fallback=mapping.fallback,
                    )
                    for v in raw_vals
                ] if raw_vals else [mapping.fallback] * n_rows
                columns[mapping.canonical_name] = pa.array(resolved, type=pa.string())

            elif mapping.strategy == "literal":
                val = mapping.literal_value or mapping.fallback
                columns[mapping.canonical_name] = pa.array([val] * n_rows, type=pa.string())

            elif mapping.strategy == "passthrough":
                raw_vals = var_raw.get(mapping.canonical_name, [])
                resolved = [
                    str(v) if v is not None else mapping.fallback
                    for v in raw_vals
                ] if raw_vals else [mapping.fallback] * n_rows
                columns[mapping.canonical_name] = pa.array(resolved, type=pa.string())

            elif mapping.strategy == "null":
                columns[mapping.canonical_name] = pa.array(
                    [mapping.fallback] * n_rows, type=pa.string()
                )

            else:
                self._warnings.append(
                    f"Unhandled var strategy '{mapping.strategy}' for "
                    f"column '{mapping.canonical_name}'; using fallback"
                )
                columns[mapping.canonical_name] = pa.array(
                    [mapping.fallback] * n_rows, type=pa.string()
                )

        # Extensible var columns
        for ext in schema.var_extensible:
            raw_col = var_raw.get(ext.raw_source_column, [])
            if raw_col:
                values = [str(v) if v is not None else MISSING_VALUE_LITERAL for v in raw_col]
            else:
                values = [MISSING_VALUE_LITERAL] * n_rows
                self._warnings.append(
                    f"var extensible column '{ext.canonical_name}' not found"
                )
            columns[ext.canonical_name] = pa.array(values, type=pa.string())

        # Validate
        available = set(columns.keys())
        missing = CanonicalizationRunner._default_var_schema().missing_required(available)
        if missing:
            raise ValueError(
                f"Missing required canonical var columns: {missing}"
            )

        return pa.table(columns)

    # ------------------------------------------------------------------
    # Transform application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transforms(
        value: str,
        transforms: tuple,
        fallback: str = MISSING_VALUE_LITERAL,
    ) -> str:
        """Apply an ordered list of ``TransformRule`` objects to a string value.

        Returns *fallback* if any intermediate value is null-like.
        """
        for rule in transforms:
            fn = _transform_registry.get(rule.name)
            if fn is None:
                logger.warning(
                    "Unknown transform '%s'; skipping (value=%r)", rule.name, value
                )
                continue
            try:
                value = fn(value, **rule.args)
            except Exception as exc:
                logger.warning("Transform '%s' failed: %s; using fallback", rule.name, exc)
                return fallback
        if _is_null_like_str(value):
            return fallback
        return value

    # ------------------------------------------------------------------
    # Vocab building
    # ------------------------------------------------------------------

    def _build_vocab(
        self,
        obs_table: pa.Table,
        var_table: pa.Table,
    ) -> CanonicalVocab:
        """Extract per-dataset vocabulary from canonical tables."""
        vocab = CanonicalVocab()

        # Obs categories — collect unique values per string column
        skip_obs = {"global_row_index", "local_row_index", "size_factor"}
        for col_name in obs_table.column_names:
            if col_name in skip_obs:
                continue
            col = obs_table.column(col_name)
            if col.type == pa.string():
                uniq = sorted(set(
                    v for v in col.to_pylist()
                    if v and v != MISSING_VALUE_LITERAL
                ))
                if uniq:
                    vocab.obs_categories[col_name] = uniq

        # Var categories
        skip_var = {"origin_index", "global_id"}
        for col_name in var_table.column_names:
            if col_name in skip_var:
                continue
            col = var_table.column(col_name)
            if col.type == pa.string():
                uniq = sorted(set(
                    v for v in col.to_pylist()
                    if v and v != MISSING_VALUE_LITERAL
                ))
                if uniq:
                    vocab.var_categories[col_name] = uniq

        # Gene ID mappings: gene_id → canonical_gene_id
        if "gene_id" in var_table.column_names and "canonical_gene_id" in var_table.column_names:
            gids = var_table.column("gene_id").to_pylist()
            cgids = var_table.column("canonical_gene_id").to_pylist()
            for gid, cgid in zip(gids, cgids):
                if gid and gid != MISSING_VALUE_LITERAL:
                    vocab.gene_id_mappings[gid] = cgid

        # Global vocab size
        if "canonical_gene_id" in var_table.column_names:
            uniq_cg = set(
                v for v in var_table.column("canonical_gene_id").to_pylist()
                if v and v != MISSING_VALUE_LITERAL
            )
            vocab.global_vocab_size = len(uniq_cg)

        return vocab

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_obs_schema() -> "CanonicalObsSchema":
        from .contract import CanonicalObsSchema
        return CanonicalObsSchema()

    @staticmethod
    def _default_var_schema() -> "CanonicalVarSchema":
        from .contract import CanonicalVarSchema
        return CanonicalVarSchema()


# ---------------------------------------------------------------------------
# Corpus-level vocab builder
# ---------------------------------------------------------------------------


def build_canonical_vocab(
    per_dataset_vocabs: list[CanonicalVocab],
    output_path: str | Path | None = None,
) -> CanonicalVocab:
    """Merge per-dataset vocabs into a corpus-global ``CanonicalVocab``.

    Parameters
    ----------
    per_dataset_vocabs : list of CanonicalVocab
        One vocab per dataset in canonicalization order.
    output_path : Path or None
        If given, write as ``canonical-vocab.yaml``.

    Returns
    -------
    CanonicalVocab
        Merged vocabulary with sorted, deduplicated values.
    """
    merged = CanonicalVocab()

    for vocab in per_dataset_vocabs:
        for cat, vals in vocab.obs_categories.items():
            existing = set(merged.obs_categories.get(cat, []))
            existing.update(vals)
            merged.obs_categories[cat] = sorted(existing)

        for cat, vals in vocab.var_categories.items():
            existing = set(merged.var_categories.get(cat, []))
            existing.update(vals)
            merged.var_categories[cat] = sorted(existing)

        merged.gene_id_mappings.update(vocab.gene_id_mappings)

    # Global vocab size: count unique canonical_gene_ids across all
    if "canonical_gene_id" in merged.var_categories:
        merged.global_vocab_size = len(merged.var_categories["canonical_gene_id"])
    else:
        merged.global_vocab_size = 0

    if output_path:
        _write_vocab_yaml(merged, Path(output_path))

    return merged


def _write_vocab_yaml(vocab: CanonicalVocab, path: Path) -> None:
    """Serialize a ``CanonicalVocab`` to YAML."""
    import yaml

    doc: dict[str, Any] = {
        "kind": "canonical-vocab",
        "contract_version": CONTRACT_VERSION,
        "global_vocab_size": vocab.global_vocab_size,
        "obs_categories": dict(vocab.obs_categories),
        "var_categories": dict(vocab.var_categories),
        "gene_id_mappings_count": len(vocab.gene_id_mappings),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_canonicalization(
    dataset_id: str,
    raw_obs_path: str | Path,
    raw_var_path: str | Path,
    size_factor_path: str | Path | None,
    schema_path: str | Path,
    output_root: str | Path,
    release_id: str | None = None,
) -> CanonicalizationResult:
    """Single-dataset canonicalization entry point.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (must match schema's ``dataset_id``).
    raw_obs_path : Path
        Raw obs sidecar parquet.
    raw_var_path : Path
        Raw var sidecar parquet.
    size_factor_path : Path or None
        Size-factor sidecar parquet.
    schema_path : Path
        ``canonicalization-schema.yaml`` for this dataset.
    output_root : Path
        Output directory for canonical parquets.
    release_id : str or None
        Release identifier for output naming.
    """
    runner = CanonicalizationRunner(
        raw_obs_path=raw_obs_path,
        raw_var_path=raw_var_path,
        size_factor_path=size_factor_path,
        schema_path=schema_path,
        output_root=output_root,
        release_id=release_id,
    )
    return runner.run()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NA_LITERALS: frozenset[str] = frozenset({"", "na", "n/a", "none", "null", "nan", ".", "-"})


def _is_null_like_str(value: str | None) -> bool:
    """Return True for values treated as null/empty."""
    if value is None:
        return True
    lowered = str(value).strip().lower()
    if not lowered:
        return True
    return lowered in _NA_LITERALS
