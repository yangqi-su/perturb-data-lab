"""Phase 6 CLI: top-level multi-command interface for the perturb-data-lab workflow.

Commands
--------
inspect batch   Run h5ad inspection over a batch config.
schema validate Validate a schema YAML for materialization readiness.
schema preview  Preview resolved canonical values on sampled rows from a schema.
materialize    Materialize a single dataset into a backend storage format.
corpus create  Create a new corpus with a declared backend.
corpus append  Append a materialized dataset to an existing corpus.
corpus validate  Validate a corpus for logical completeness.

All subcommands are thin wrappers over the existing typed API.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .inspectors.models import InspectionBatchConfig, SchemaDocument
from .inspectors.schema_utils import explain_schema, preview_cell_row, preview_feature_row
from .inspectors.workflow import run_batch


# ---------------------------------------------------------------------------
# inspect batch
# ---------------------------------------------------------------------------


def _add_inspect_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--config", required=True, help="Path to the YAML batch inspection config."
    )
    sub.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of datasets to inspect concurrently.",
    )


def _cmd_inspect(args: argparse.Namespace) -> None:
    config = InspectionBatchConfig.from_yaml_file(Path(args.config))
    manifest = run_batch(config, workers=args.workers)
    print(
        f"[inspect] wrote manifest {Path(manifest.output_root) / 'inspection-manifest.yaml'}"
    )


# ---------------------------------------------------------------------------
# schema validate
# ---------------------------------------------------------------------------


def _add_schema_validate_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "schema_path",
        help="Path to the schema YAML file to validate.",
    )
    sub.add_argument(
        "--corpus-namespace",
        default=None,
        help="Optional corpus namespace to also check namespace compatibility against.",
    )


def _cmd_schema_validate(args: argparse.Namespace) -> None:
    from .materializers.validation import validate_schema_readiness
    from .inspectors.models import SchemaDocument

    schema_path = Path(args.schema_path)
    if not schema_path.exists():
        print(f"[error] schema not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    schema = SchemaDocument.from_yaml_file(schema_path)
    result = validate_schema_readiness(str(schema_path))

    if result.valid:
        print(f"[schema validate] PASS — {schema_path} is ready for materialization")
    else:
        print(f"[schema validate] FAIL — {schema_path}", file=sys.stderr)
        for v in result.violations:
            print(f"  [{v.section}.{v.field}] {v.reason}", file=sys.stderr)
        sys.exit(1)

    # Optional namespace compatibility check
    if args.corpus_namespace:
        from .schema_utils import check_namespace_compatibility

        ok, reason = check_namespace_compatibility(schema, args.corpus_namespace)
        if ok:
            print(
                f"[namespace] OK — schema namespace '{schema.feature_tokenization.namespace}' "
                f"is compatible with corpus namespace '{args.corpus_namespace}'"
            )
        else:
            print(
                f"[namespace] INCOMPATIBLE — schema namespace "
                f"'{schema.feature_tokenization.namespace}' != corpus namespace "
                f"'{args.corpus_namespace}'",
                file=sys.stderr,
            )
            print(f"  {reason}", file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# schema preview
# ---------------------------------------------------------------------------


def _add_schema_preview_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "schema_path",
        help="Path to the schema YAML file.",
    )
    sub.add_argument(
        "--sample",
        default=None,
        help=(
            "Path to a small h5ad file to sample rows from for preview. "
            "If not provided, uses example values from the schema."
        ),
    )
    sub.add_argument(
        "--n-rows",
        type=int,
        default=5,
        help="Number of rows to sample from the h5ad for preview. Default: 5.",
    )


def _cmd_schema_preview(args: argparse.Namespace) -> None:
    import anndata as ad
    import pandas as pd

    schema_path = Path(args.schema_path)
    if not schema_path.exists():
        print(f"[error] schema not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    schema = SchemaDocument.from_yaml_file(schema_path)

    # Print schema explanation
    exp = explain_schema(schema)
    print(f"\n=== Schema Explanation: {exp.dataset_id} ===")
    print(f"Status     : {exp.status}")
    print(f"Namespace  : {exp.namespace} ({exp.namespace_status})")
    if exp.namespace_issue:
        print(f"  {exp.namespace_issue}")
    print(f"Fields     : {exp.field_count} total, {exp.resolved_count} resolved, "
          f"{exp.unresolved_count} unresolved, {exp.null_required_count} null+required")
    if exp.readiness_blockers:
        print("Blockers:")
        for b in exp.readiness_blockers:
            print(f"  - {b}")

    print(f"\n--- Per-field status ---")
    for fe in exp.field_explanations:
        status_mark = "✓" if fe.status == "resolved" else "✗"
        null_note = " [null+required]" if (fe.status == "null_strategy" and fe.required) else ""
        print(f"  {status_mark} [{fe.section}.{fe.field}] strategy={fe.strategy}  "
              f"confidence={fe.confidence} sources={fe.source_fields}{null_note}")
        if fe.issue and (fe.status != "resolved" or fe.required):
            print(f"      issue: {fe.issue}")
        if fe.resolution_hint and fe.status != "resolved":
            print(f"      hint: {fe.resolution_hint}")

    # If a sample h5ad is provided, show resolved values on real rows
    if args.sample:
        sample_path = Path(args.sample)
        if not sample_path.exists():
            print(f"\n[error] sample h5ad not found: {sample_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\n--- Sample row preview (from {sample_path.name}) ---")
        adata = ad.read_h5ad(str(sample_path), backed="r")
        obs_mem = adata.obs.to_memory() if hasattr(adata.obs, "to_memory") else adata.obs
        n_rows = min(args.n_rows, len(obs_mem))

        # Sample evenly spaced rows
        indices = list(range(0, len(obs_mem), max(1, len(obs_mem) // n_rows)))[:n_rows]

        for i in indices:
            obs_row = obs_mem.iloc[i].to_dict()
            cell_id = str(obs_mem.index[i])
            result = preview_cell_row(schema, obs_row)
            print(f"\n  cell_id={cell_id}")
            for section_name, fields in [("perturbation", result["perturbation"])]:
                for k, v in fields.items():
                    print(f"    {section_name}.{k} = {v}")

        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

        # Also preview feature rows if var is available
        var_mem = adata.var.to_memory() if hasattr(adata.var, "to_memory") else adata.var
        print(f"\n--- Sample feature rows ---")
        n_feat = min(args.n_rows, len(var_mem))
        feat_indices = list(range(0, len(var_mem), max(1, len(var_mem) // n_feat)))[:n_feat]
        for i in feat_indices:
            var_row = var_mem.iloc[i].to_dict()
            feat_id = str(var_mem.index[i])
            result = preview_feature_row(schema, var_row)
            print(f"\n  feature={feat_id}")
            for k, v in result.items():
                print(f"    feature.{k} = {v}")


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------


def _add_materialize_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--source",
        required=True,
        help="Path to the source h5ad file.",
    )
    sub.add_argument(
        "--review-bundle",
        required=True,
        help=(
            "Path to the Stage 1 dataset-summary.yaml that gates this materialization. "
            "This is the only required gating artifact; no schema.yaml is used."
        ),
    )
    sub.add_argument(
        "--backend",
        required=True,
        choices=["arrow-parquet", "arrow-ipc", "webdataset", "zarr", "lance"],
        help="Storage backend for this materialization.",
    )
    sub.add_argument(
        "--topology",
        default="federated",
        choices=["federated", "aggregate"],
        help="Corpus topology. Default: federated.",
    )
    sub.add_argument(
        "--release-id",
        required=True,
        help="Release identifier for this dataset version (e.g., v0.1).",
    )
    sub.add_argument(
        "--dataset-id",
        required=True,
        help="Stable dataset identifier.",
    )
    sub.add_argument(
        "--output-root",
        required=True,
        help="Root directory for all metadata and matrix outputs.",
    )
    sub.add_argument(
        "--corpus-index",
        default=None,
        help=(
            "Path to corpus-index.yaml for corpus registration. "
            "If provided with --register, the dataset is registered with the corpus "
            "after materialization. If not provided, --register cannot be used."
        ),
    )
    sub.add_argument(
        "--corpus-id",
        default=None,
        help="Corpus identifier. Required when registering a new corpus.",
    )
    sub.add_argument(
        "--register",
        action="store_true",
        help=(
            "If set, automatically register this dataset with the corpus ledger "
            "after successful materialization. Requires --corpus-index to be set."
        ),
    )
    sub.add_argument(
        "--rerun-stage1",
        action="store_true",
        help=(
            "If set, rerun Stage 1 inspection before materialization as a preflight step "
            "and use the resulting dataset-summary.yaml as the gating artifact."
        ),
    )
    sub.add_argument(
        "--n-hvg",
        type=int,
        default=2000,
        help="Number of top-dispersion genes to select as HVGs. Default: 2000.",
    )


def _cmd_materialize(args: argparse.Namespace) -> None:
    from .materializers import Stage2Materializer
    from .materializers.models import OutputRoots

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[error] source h5ad not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    review_bundle_path = Path(args.review_bundle)
    if not args.rerun_stage1 and not review_bundle_path.exists():
        print(
            f"[error] review bundle not found: {review_bundle_path}; "
            "pass --rerun-stage1 to run Stage 1 as preflight",
            file=sys.stderr,
        )
        sys.exit(1)

    output_roots = OutputRoots(
        metadata_root=str(Path(args.output_root) / "meta"),
        matrix_root=str(Path(args.output_root) / "matrix"),
    )

    if args.register and args.corpus_index is None:
        print(
            "[error] --register requires --corpus-index to be set",
            file=sys.stderr,
        )
        sys.exit(1)

    materializer = Stage2Materializer(
        source_path=str(source_path),
        review_bundle_path=str(review_bundle_path),
        output_roots=output_roots,
        release_id=args.release_id,
        dataset_id=args.dataset_id,
        backend=args.backend,
        topology=args.topology,
        rerun_stage1=args.rerun_stage1,
        n_hvg=args.n_hvg,
        corpus_index_path=args.corpus_index,
        corpus_id=args.corpus_id,
        register=args.register,
    )

    manifest = materializer.materialize()

    print(f"[materialize] done — {args.dataset_id}/{args.release_id}")
    print(f"  backend     : {manifest.backend}")
    print(f"  topology     : {manifest.topology}")
    print(f"  count source: {manifest.count_source.selected}")
    print(f"  integer_verified: {manifest.integer_verified}")
    print(f"  cell_count  : {manifest.cell_count}")
    print(f"  feature_count: {manifest.feature_count}")
    print(f"  manifest    : {Path(manifest.outputs.metadata_root) / 'materialization-manifest.yaml'}")
    if manifest.corpus_registration is not None:
        reg = manifest.corpus_registration
        print(f"  corpus_id   : {reg.corpus_id}")
        print(f"  is_create   : {reg.is_create}")
        print(f"  dataset_index: {reg.dataset_index}")
        print(f"  global range: [{reg.global_start}, {reg.global_end})")
        print(f"  ledger      : {reg.ledger_path}")


# ---------------------------------------------------------------------------
# corpus create
# ---------------------------------------------------------------------------


def _add_corpus_create_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--backend",
        required=True,
        choices=["arrow-parquet", "arrow-ipc", "webdataset", "zarr", "lance"],
        help="Backend for this corpus.",
    )
    sub.add_argument(
        "--output",
        required=True,
        help="Output path for corpus-index.yaml.",
    )
    sub.add_argument(
        "--corpus-id",
        default=None,
        help="Optional corpus identifier. Defaults to a generated ID.",
    )


def _cmd_corpus_create(args: argparse.Namespace) -> None:
    from .materializers import update_corpus_index
    from .materializers.models import DatasetJoinRecord, GlobalMetadataDocument
    from .materializers.emission_spec import CorpusEmissionSpec
    from .contracts import CONTRACT_VERSION

    output_path = Path(args.output)
    if output_path.exists():
        print(f"[error] corpus-index already exists: {output_path}", file=sys.stderr)
        sys.exit(1)

    # Create an empty corpus index (no placeholder records).
    # The first real dataset is added via `materialize --corpus-index` in create_new mode.
    # An empty index still records the corpus backend declaration.
    corpus_id = args.corpus_id or "perturb-data-lab-v0"
    updated = update_corpus_index(
        output_path,
        DatasetJoinRecord(
            dataset_id="__placeholder__must_materialize_first__",
            release_id="__placeholder__",
            join_mode="create_new",
            manifest_path="/dev/null",
            cell_count=0,
        ),
        backend=args.backend,
    )
    # Immediately overwrite with a clean empty index (no placeholder record)
    from .materializers.models import CorpusIndexDocument
    empty = CorpusIndexDocument(
        kind="corpus-index",
        contract_version=CONTRACT_VERSION,
        corpus_id=corpus_id,
        global_metadata={"backend": args.backend},
        datasets=(),
    )
    empty.write_yaml(output_path)

    print(f"[corpus create] {corpus_id} backend={args.backend}")
    print(f"  corpus-index: {output_path}")
    print(f"  note: no datasets yet — run 'materialize --corpus-index {output_path}' to add the first dataset")

    # Write initial emission spec
    emission_spec = CorpusEmissionSpec(corpus_id=corpus_id)
    emission_spec_path = output_path.parent / "corpus-emission-spec.yaml"
    emission_spec.write_yaml(emission_spec_path)
    print(f"  emission-spec: {emission_spec_path}")


# ---------------------------------------------------------------------------
# corpus append
# ---------------------------------------------------------------------------


def _add_corpus_append_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--corpus-index",
        required=True,
        help="Path to the corpus-index.yaml.",
    )
    sub.add_argument(
        "--manifest",
        required=True,
        help="Path to the dataset's materialization-manifest.yaml.",
    )


def _cmd_corpus_append(args: argparse.Namespace) -> None:
    from .materializers import update_corpus_index
    from .materializers.models import MaterializationManifest, DatasetJoinRecord

    corpus_index_path = Path(args.corpus_index)
    if not corpus_index_path.exists():
        print(f"[error] corpus-index not found: {corpus_index_path}", file=sys.stderr)
        sys.exit(1)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[error] manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = MaterializationManifest.from_yaml_file(manifest_path)
    record = DatasetJoinRecord(
        dataset_id=manifest.dataset_id,
        release_id=manifest.release_id,
        join_mode="append_routed",
        manifest_path=str(manifest_path),
        cell_count=manifest.cell_count,
    )
    updated = update_corpus_index(corpus_index_path, record)  # backend already set in corpus
    print(f"[corpus append] done")
    print(f"  corpus-index: {corpus_index_path}")
    print(f"  added       : {manifest.dataset_id}/{manifest.release_id}")
    print(f"  datasets    : {[d.dataset_id for d in updated.datasets]}")


# ---------------------------------------------------------------------------
# corpus validate
# ---------------------------------------------------------------------------


def _add_corpus_validate_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "corpus_index",
        help="Path to the corpus-index.yaml to validate.",
    )
    sub.add_argument(
        "--backend",
        choices=["arrow-parquet", "arrow-ipc", "webdataset", "zarr", "lance"],
        help="Optional: verify the corpus backend matches this value.",
    )


def _cmd_corpus_validate(args: argparse.Namespace) -> None:
    from .materializers.models import CorpusIndexDocument

    corpus_index_path = Path(args.corpus_index)
    if not corpus_index_path.exists():
        print(f"[error] corpus-index not found: {corpus_index_path}", file=sys.stderr)
        sys.exit(1)

    corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)

    corpus_root = corpus_index_path.parent

    print(f"\n=== Corpus Validation: {corpus.corpus_id} ===")
    print(f"Datasets ({len(corpus.datasets)}):")
    violations: list[str] = []

    for ds in corpus.datasets:
        manifest_path = corpus_root / ds.manifest_path
        exists = "✓" if manifest_path.exists() else "✗ MISSING"
        print(f"  {exists} {ds.dataset_id}/{ds.release_id}  [{ds.join_mode}]")
        if not manifest_path.exists():
            violations.append(f"manifest missing for {ds.dataset_id}: {ds.manifest_path}")

        # Validate manifest if it exists
        if manifest_path.exists() and manifest_path.name.endswith("-manifest.yaml"):
            try:
                from .materializers.models import MaterializationManifest

                mnf = MaterializationManifest.from_yaml_file(manifest_path)
                print(f"         backend={mnf.backend}  count={mnf.count_source.selected}")

                # Backend consistency check
                if args.backend and mnf.backend != args.backend:
                    violations.append(
                        f"backend mismatch in {ds.dataset_id}: "
                        f"expected {args.backend}, got {mnf.backend}"
                    )
            except Exception as e:
                violations.append(f"failed to load manifest for {ds.dataset_id}: {e}")

    # Check tokenizer (Phase 3: tokenizer is no longer required for corpus validation)
    tokenizer_path = corpus_root / "tokenizer.json"
    tokenizer_exists = tokenizer_path.exists()
    print(f"\nTokenizer: {'✓' if tokenizer_exists else '✗ (optional since Phase 3)'} {tokenizer_path}")
    # Tokenizer is optional — corpus feature set is now maintained by canonicalize-meta

    # Check emission spec
    emission_spec_path = corpus_root / "corpus-emission-spec.yaml"
    emission_exists = emission_spec_path.exists()
    print(f"Emission spec: {'✓' if emission_exists else '✗ MISSING'} {emission_spec_path}")

    if violations:
        print(f"\n[corpus validate] FAIL — {len(violations)} issue(s):")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print(f"\n[corpus validate] PASS")


# ---------------------------------------------------------------------------
# canonicalize-meta
# ---------------------------------------------------------------------------


def _add_canonicalize_meta_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--corpus-index",
        required=True,
        help="Path to the corpus-index.yaml.",
    )


def _cmd_canonicalize_meta(args: argparse.Namespace) -> None:
    from .materializers import run_canonicalize_meta

    corpus_index_path = Path(args.corpus_index)
    if not corpus_index_path.exists():
        print(f"[error] corpus-index not found: {corpus_index_path}", file=sys.stderr)
        sys.exit(1)

    result = run_canonicalize_meta(corpus_index_path)

    print(f"[canonicalize-meta] done — {result.corpus_id}")
    print(f"  datasets      : {result.datasets}")
    print(f"  total_cells   : {result.total_cells}")
    print(f"  cell_ranges   : {len(result.cell_ranges)} datasets with global ranges")
    for r in result.cell_ranges:
        print(f"    {r.dataset_id}/{r.release_id}: [{r.global_start}, {r.global_end}) count={r.count}")
    print(f"  corpus_cell_meta    : {result.corpus_cell_meta_path}")
    print(f"  corpus_feature_meta : {result.corpus_feature_meta_path}")
    print(f"  feature_set         : {result.feature_set_path}")


def _add_stage2_materialize_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--source",
        required=True,
        help="Path to the source h5ad file.",
    )
    sub.add_argument(
        "--review-bundle",
        required=True,
        help=(
            "Path to the Stage 1 dataset-summary.yaml that gates this materialization. "
            "This is the only required gating artifact; no schema.yaml is used."
        ),
    )
    sub.add_argument(
        "--output-root",
        required=True,
        help="Root directory for all metadata and matrix outputs.",
    )
    sub.add_argument(
        "--release-id",
        required=True,
        help="Release identifier for this dataset version (e.g., v0.1).",
    )
    sub.add_argument(
        "--dataset-id",
        required=True,
        help="Stable dataset identifier.",
    )
    sub.add_argument(
        "--backend",
        default="arrow-parquet",
        choices=["arrow-parquet", "arrow-ipc", "webdataset", "zarr", "lance"],
        help="Storage backend for this materialization. Default: arrow-parquet.",
    )
    sub.add_argument(
        "--topology",
        default="federated",
        choices=["federated", "aggregate"],
        help="Corpus topology. Default: federated.",
    )
    sub.add_argument(
        "--rerun-stage1",
        action="store_true",
        help=(
            "If set, rerun Stage 1 inspection before materialization as a preflight step "
            "and use the resulting dataset-summary.yaml as the gating artifact."
        ),
    )
    sub.add_argument(
        "--n-hvg",
        type=int,
        default=2000,
        help="Number of top-dispersion genes to select as HVGs. Default: 2000.",
    )
    sub.add_argument(
        "--corpus-index",
        default=None,
        help=(
            "Path to corpus-index.yaml for corpus registration. "
            "If provided with --register, the dataset is registered with the corpus "
            "after materialization. If not provided, --register cannot be used."
        ),
    )
    sub.add_argument(
        "--corpus-id",
        default=None,
        help="Corpus identifier. Required when registering a new corpus.",
    )
    sub.add_argument(
        "--register",
        action="store_true",
        help=(
            "If set, automatically register this dataset with the corpus ledger "
            "after successful materialization. Requires --corpus-index to be set."
        ),
    )


def _cmd_stage2_materialize(args: argparse.Namespace) -> None:
    from .materializers import Stage2Materializer
    from .materializers.models import OutputRoots

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[error] source h5ad not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    review_bundle_path = Path(args.review_bundle)
    if not args.rerun_stage1 and not review_bundle_path.exists():
        print(
            f"[error] review bundle not found: {review_bundle_path}; "
            "pass --rerun-stage1 to run Stage 1 as preflight",
            file=sys.stderr,
        )
        sys.exit(1)

    output_roots = OutputRoots(
        metadata_root=str(Path(args.output_root) / "meta"),
        matrix_root=str(Path(args.output_root) / "matrix"),
    )

    if args.register and args.corpus_index is None:
        print(
            "[error] --register requires --corpus-index to be set",
            file=sys.stderr,
        )
        sys.exit(1)

    materializer = Stage2Materializer(
        source_path=str(source_path),
        review_bundle_path=str(review_bundle_path),
        output_roots=output_roots,
        release_id=args.release_id,
        dataset_id=args.dataset_id,
        backend=args.backend,
        topology=args.topology,
        rerun_stage1=args.rerun_stage1,
        n_hvg=args.n_hvg,
        corpus_index_path=args.corpus_index,
        corpus_id=args.corpus_id,
        register=args.register,
    )

    manifest = materializer.materialize()

    print(f"[stage2-materialize] done — {args.dataset_id}/{args.release_id}")
    print(f"  backend     : {manifest.backend}")
    print(f"  topology    : {manifest.topology}")
    print(f"  count source: {manifest.count_source.selected}")
    print(f"  integer_verified: {manifest.integer_verified}")
    print(f"  cell_count  : {manifest.cell_count}")
    print(f"  feature_count: {manifest.feature_count}")
    print(f"  manifest    : {Path(manifest.outputs.metadata_root) / 'materialization-manifest.yaml'}")
    if manifest.corpus_registration is not None:
        reg = manifest.corpus_registration
        print(f"  corpus_id   : {reg.corpus_id}")
        print(f"  is_create   : {reg.is_create}")
        print(f"  dataset_index: {reg.dataset_index}")
        print(f"  global range: [{reg.global_start}, {reg.global_end})")
        print(f"  ledger      : {reg.ledger_path}")


# ---------------------------------------------------------------------------
# Main CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="perturb-data-lab: inspect, materialize, and validate multi-dataset corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # inspect batch
    p_inspect = sub.add_parser("inspect", help="Run h5ad inspection batch.")
    _add_inspect_args(p_inspect)

    # schema validate
    p_sv = sub.add_parser("schema-validate", help="Validate schema for materialization readiness.")
    _add_schema_validate_args(p_sv)

    # schema preview
    p_sp = sub.add_parser("schema-preview", help="Preview resolved canonical values from a schema.")
    _add_schema_preview_args(p_sp)

    # materialize
    p_mat = sub.add_parser("materialize", help="Materialize a dataset into a backend storage format.")
    _add_materialize_args(p_mat)

    # corpus create
    p_ccreate = sub.add_parser("corpus-create", help="Create a new empty corpus with a declared backend.")
    _add_corpus_create_args(p_ccreate)

    # corpus append
    p_cap = sub.add_parser("corpus-append", help="Append a materialized dataset to a corpus.")
    _add_corpus_append_args(p_cap)

    # corpus validate
    p_cv = sub.add_parser("corpus-validate", help="Validate a corpus for logical completeness.")
    _add_corpus_validate_args(p_cv)

    # canonicalize-meta
    p_cm = sub.add_parser("canonicalize-meta", help="Rebuild corpus canonical metadata from raw artifacts.")
    _add_canonicalize_meta_args(p_cm)

    # stage2-materialize — schema-independent, Stage-1-gated path
    p_s2 = sub.add_parser(
        "stage2-materialize",
        help=(
            "Materialize a dataset using the Stage 2 schema-independent path "
            "(gated by Stage 1 dataset-summary.yaml, no schema.yaml required)."
        ),
    )
    _add_stage2_materialize_args(p_s2)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "schema-validate":
        _cmd_schema_validate(args)
    elif args.command == "schema-preview":
        _cmd_schema_preview(args)
    elif args.command == "materialize":
        _cmd_materialize(args)
    elif args.command == "corpus-create":
        _cmd_corpus_create(args)
    elif args.command == "corpus-append":
        _cmd_corpus_append(args)
    elif args.command == "corpus-validate":
        _cmd_corpus_validate(args)
    elif args.command == "canonicalize-meta":
        _cmd_canonicalize_meta(args)
    elif args.command == "stage2-materialize":
        _cmd_stage2_materialize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
