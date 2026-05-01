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

from .inspectors.models import InspectionBatchConfig
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
    elif args.command == "materialize":
        _cmd_materialize(args)
    elif args.command == "corpus-create":
        _cmd_corpus_create(args)
    elif args.command == "corpus-append":
        _cmd_corpus_append(args)
    elif args.command == "corpus-validate":
        _cmd_corpus_validate(args)
    elif args.command == "stage2-materialize":
        _cmd_stage2_materialize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
