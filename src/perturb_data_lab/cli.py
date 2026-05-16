"""CLI: top-level multi-command interface for the perturb-data-lab workflow.

Commands
--------
inspect          Run h5ad inspection over a batch config, or directly for one dataset.
materialize      Materialize datasets with --mode {create,append} into a backend.
                 Supports single (--source) and bulk (--input-dir, --input-list)
                 input. Outputs go under --output-corpus.
draft-schema     Auto-draft canonicalization schemas from corpus-local inspection summaries.
canonicalize     Canonicalize using corpus-local final schemas.
backfill-hvg     Backfill canonical hvg.parquet files for an existing Lance corpus.
corpus-validate  Validate a corpus for logical completeness.
corpus-gc        Garbage-collect orphaned Lance fragments not in the corpus ledger.

Removed (Phase 2 consolidation):
  stage2-materialize → merged into materialize
  corpus-create → merged into materialize --mode create
  corpus-append  → merged into materialize --mode append
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .inspectors.models import InspectionBatchConfig
from .inspectors.workflow import run_batch
from .materializers.paths import resolve_corpus_paths


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BACKEND_CHOICES = ["lance", "zarr"]
TOPOLOGY_CHOICES = ["federated", "aggregate"]


@dataclass(frozen=True)
class _DatasetInput:
    """A single dataset input parsed from CLI flags."""
    source: str
    dataset_id: str
    review_bundle: str


def _parse_input_list_csv(csv_path: Path) -> list[_DatasetInput]:
    """Parse a CSV input list with columns: source, dataset_id, review_bundle."""
    inputs: list[_DatasetInput] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row")
        missing = {"source", "dataset_id", "review_bundle"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            inputs.append(
                _DatasetInput(
                    source=row["source"].strip(),
                    dataset_id=row["dataset_id"].strip(),
                    review_bundle=row["review_bundle"].strip(),
                )
            )
    if not inputs:
        raise ValueError(f"CSV file {csv_path} contains no dataset rows")
    return inputs


def _scan_input_dir(
    input_dir: Path,
    *,
    review_bundle_dir: Path | None = None,
) -> list[_DatasetInput]:
    """Scan a directory for .h5ad files and build _DatasetInput entries.

    dataset_id is derived from the filename stem
    (e.g. dummy_00_counts -> dummy_00_counts).
    Review bundles are expected at:
        {review_bundle_dir or input_dir}/{dataset_id}-summary.yaml
    """
    h5ad_files = sorted(p for p in input_dir.iterdir() if p.suffix == ".h5ad")
    if not h5ad_files:
        raise FileNotFoundError(f"No .h5ad files found in {input_dir}")

    bundle_dir = review_bundle_dir or input_dir
    inputs: list[_DatasetInput] = []
    for h5ad_path in h5ad_files:
        dataset_id = h5ad_path.stem
        bundle_path = bundle_dir / f"{dataset_id}-summary.yaml"
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Review bundle not found for {dataset_id}: expected {bundle_path}. "
                "Use --review-bundle-dir to specify an alternate location, "
                "or use --rerun-stage1 to run inspection as preflight."
            )
        inputs.append(
            _DatasetInput(
                source=str(h5ad_path),
                dataset_id=dataset_id,
                review_bundle=str(bundle_path),
            )
        )
    return inputs


def _resolve_inputs(args: argparse.Namespace) -> list[_DatasetInput]:
    """Resolve and validate the dataset inputs from CLI flags.

    Returns a list of _DatasetInput entries in processing order.
    """
    sources: list[_DatasetInput] = []
    has_single = getattr(args, "source", None) is not None
    has_input_list = getattr(args, "input_list", None) is not None
    has_input_dir = getattr(args, "input_dir", None) is not None

    n_input_modes = sum([bool(has_single), bool(has_input_list), bool(has_input_dir)])
    if n_input_modes == 0:
        print(
            "[error] must specify one of: --source, --input-list, or --input-dir",
            file=sys.stderr,
        )
        sys.exit(1)
    if n_input_modes > 1:
        print(
            "[error] only one of --source, --input-list, --input-dir may be used at a time",
            file=sys.stderr,
        )
        sys.exit(1)

    if has_single:
        if not getattr(args, "dataset_id", None):
            print("[error] --dataset-id is required with --source", file=sys.stderr)
            sys.exit(1)
        review_bundle = getattr(args, "review_bundle", None)
        if not review_bundle:
            print("[error] --review-bundle is required with --source", file=sys.stderr)
            sys.exit(1)
        sources.append(
            _DatasetInput(
                source=args.source,
                dataset_id=args.dataset_id,
                review_bundle=review_bundle,
            )
        )
    elif has_input_list:
        list_path = Path(args.input_list)
        if not list_path.exists():
            print(f"[error] input list not found: {list_path}", file=sys.stderr)
            sys.exit(1)
        sources = _parse_input_list_csv(list_path)
    elif has_input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"[error] input directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)
        bundle_dir = (
            Path(args.review_bundle_dir)
            if getattr(args, "review_bundle_dir", None)
            else None
        )
        sources = _scan_input_dir(input_dir, review_bundle_dir=bundle_dir)

    # Validate each source exists
    for ds in sources:
        src = Path(ds.source)
        if not src.exists():
            print(f"[error] source h5ad not found: {src}", file=sys.stderr)
            sys.exit(1)
        if not getattr(args, "rerun_stage1", False):
            bundle = Path(ds.review_bundle)
            if not bundle.exists():
                print(
                    f"[error] review bundle not found: {bundle}; "
                    "pass --rerun-stage1 to run Stage 1 as preflight",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Check for duplicate dataset_ids within the batch
    seen_ids: set[str] = set()
    for ds in sources:
        if ds.dataset_id in seen_ids:
            print(
                f"[error] duplicate dataset_id in inputs: {ds.dataset_id}",
                file=sys.stderr,
            )
            sys.exit(1)
        seen_ids.add(ds.dataset_id)

    return sources


def _is_aggregate(backend: str, topology: str) -> bool:
    """Determine if the combination implies aggregate topology."""
    return _resolve_effective_topology(backend, topology) == "aggregate"


def _resolve_effective_topology(backend: str, topology: str) -> str:
    """Resolve the effective topology from backend + requested topology.

    Phase 5 removes the old Lance-specific aggregate forcing so federated Lance
    can be selected explicitly and through the public CLI defaults.
    """
    if topology not in TOPOLOGY_CHOICES:
        raise ValueError(
            f"unsupported topology {topology!r}; expected one of {TOPOLOGY_CHOICES}"
        )
    return topology


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def _add_inspect_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--config",
        default=None,
        help="Path to the YAML batch inspection config.",
    )
    sub.add_argument(
        "--source",
        default=None,
        help="Path to a single source h5ad file (direct mode).",
    )
    sub.add_argument(
        "--dataset-id",
        default=None,
        help="Stable dataset identifier (direct mode).",
    )
    sub.add_argument(
        "--output-dir",
        default=None,
        help="Directory where direct inspect writes dataset-summary.yaml.",
    )
    sub.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of datasets to inspect concurrently.",
    )


def _cmd_inspect(args: argparse.Namespace) -> None:
    from .inspectors.workflow import inspect_target
    from .inspectors.models import InspectionTarget

    has_config = args.config is not None
    has_direct_flags = any(
        value is not None for value in (args.source, args.dataset_id, args.output_dir)
    )

    if has_config and has_direct_flags:
        print(
            "[inspect] use either --config (batch mode) or "
            "--source/--dataset-id/--output-dir (direct mode), not both.",
            file=sys.stderr,
        )
        sys.exit(1)

    if has_config:
        config = InspectionBatchConfig.from_yaml_file(Path(args.config))
        manifest = run_batch(config, workers=args.workers)
        print(
            f"[inspect] wrote manifest "
            f"{Path(manifest.output_root) / 'inspection-manifest.yaml'}"
        )
        return

    if not (args.source and args.dataset_id and args.output_dir):
        print(
            "[inspect] direct mode requires --source, --dataset-id, and --output-dir",
            file=sys.stderr,
        )
        sys.exit(1)

    source = Path(args.source)
    if not source.exists():
        print(f"[inspect] source h5ad not found: {source}", file=sys.stderr)
        sys.exit(1)

    artifacts = inspect_target(
        InspectionTarget(
            dataset_id=args.dataset_id,
            source_path=str(source),
            source_release=args.dataset_id,
        ),
        Path(args.output_dir),
    )
    print(f"[inspect] wrote review bundle {artifacts.review_bundle}")


# ---------------------------------------------------------------------------
# materialize (unified create / append)
# ---------------------------------------------------------------------------


def _add_materialize_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--mode",
        required=True,
        choices=["create", "append"],
        help=(
            "Materialization mode. 'create' starts a new corpus and materializes "
            "the first dataset (plus any subsequent ones). 'append' adds datasets "
            "to an existing corpus."
        ),
    )

    # Input group
    input_group = sub.add_argument_group("input source (choose one)")
    input_group.add_argument(
        "--source",
        default=None,
        help="Path to a single source h5ad file.",
    )
    input_group.add_argument(
        "--input-list",
        default=None,
        help=(
            "Path to a CSV file listing datasets to materialize. "
            "Required columns: source, dataset_id, review_bundle."
        ),
    )
    input_group.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Directory to scan for .h5ad files. Review bundles are expected "
            "alongside as {stem}-summary.yaml. Use --review-bundle-dir for "
            "alternate locations."
        ),
    )
    sub.add_argument(
        "--review-bundle-dir",
        default=None,
        help="Directory containing review bundles when using --input-dir.",
    )

    # Single-dataset identifiers (only used with --source)
    sub.add_argument(
        "--dataset-id",
        default=None,
        help="Stable dataset identifier (required with --source).",
    )
    # Review bundle for single dataset mode
    sub.add_argument(
        "--review-bundle",
        default=None,
        help="Path to the Stage 1 dataset-summary.yaml (required with --source).",
    )

    # Corpus output
    sub.add_argument(
        "--output-corpus",
        required=True,
        help=(
            "Root directory for the corpus "
            "(contains corpus-index.yaml, datasets, and matrix artifacts)."
        ),
    )

    # Backend / topology
    sub.add_argument(
        "--backend",
        default=None,
        choices=BACKEND_CHOICES,
        help=(
            "Storage backend for this materialization. Required for --mode create; "
            "auto-detected from corpus-index.yaml for --mode append when omitted."
        ),
    )
    sub.add_argument(
        "--topology",
        default=None,
        choices=TOPOLOGY_CHOICES,
        help=(
            "Corpus topology. Optional for --mode append (auto-detected from corpus metadata)."
        ),
    )

    # Optional corpus identifier
    sub.add_argument(
        "--corpus-id",
        default="perturb-data-lab-v0",
        help="Corpus identifier (default: perturb-data-lab-v0).",
    )

    # Materializer options
    sub.add_argument(
        "--rerun-stage1",
        action="store_true",
        help=(
            "If set, rerun Stage 1 inspection before materialization as a preflight "
            "and use the resulting dataset-summary.yaml as the gating artifact."
        ),
    )
    sub.add_argument(
        "--n-hvg",
        type=int,
        default=2000,
        help="Number of top-dispersion genes to select as HVGs (default: 2000).",
    )

    # Dry-run
    sub.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the execution plan without writing any data.",
    )


def _materialize_dataset(
    ds: _DatasetInput,
    args: argparse.Namespace,
    *,
    backend: str,
    corpus_root: Path,
    effective_topology: str,
    mode: str,
    dataset_index: int,
    global_row_start: int,
    writer_state: dict | None,
    is_last_dataset: bool,
    total_datasets: int,
    dry_run: bool,
    display_index: int = 0,
) -> tuple[int, dict | None]:
    """Materialize a single dataset and return (next_global_row_start, next_writer_state).

    In aggregate mode, writer_state is carried between calls.
    """
    from .materializers import Stage2Materializer
    from .materializers.models import OutputRoots

    if dry_run:
        print(
            f"  [{display_index + 1}/{total_datasets}] DRY-RUN: "
            f"{ds.dataset_id} source={ds.source}"
        )
        # Simulate a cell count of 0 for dry-run planning
        return (global_row_start, writer_state)

    # Build output roots under corpus
    resolved_paths = resolve_corpus_paths(
        topology=effective_topology,
        corpus_root=corpus_root,
        dataset_id=ds.dataset_id,
    )
    output_roots = OutputRoots(
        metadata_root=str(resolved_paths.meta_root),
        matrix_root=str(resolved_paths.matrix_root),
    )

    corpus_index_path = str(corpus_root / "corpus-index.yaml")

    print(
        f"\n  [{display_index + 1}/{total_datasets}] materializing "
        f"{ds.dataset_id}"
    )
    print(f"    source: {ds.source}")
    print(f"    mode: {mode}")

    materializer = Stage2Materializer(
        source_path=ds.source,
        review_bundle_path=ds.review_bundle,
        output_roots=output_roots,
        dataset_id=ds.dataset_id,
        backend=backend,
        topology=effective_topology,
        rerun_stage1=args.rerun_stage1,
        n_hvg=getattr(args, "n_hvg", 2000),
        corpus_index_path=corpus_index_path,
        corpus_id=args.corpus_id,
        register=True,
        mode=mode,
        dataset_index=dataset_index,
        global_row_start=global_row_start,
        writer_state=writer_state,
        _is_last_dataset=is_last_dataset,
    )

    manifest = materializer.materialize()

    print(f"    cell_count: {manifest.cell_count}")
    print(f"    feature_count: {manifest.feature_count}")
    print(f"    route: {manifest.route}")
    if manifest.corpus_registration is not None:
        reg = manifest.corpus_registration
        print(f"    corpus registration: {reg.corpus_id} is_create={reg.is_create}")
        print(f"    global range: [{reg.global_start}, {reg.global_end})")

    # Return updated state
    next_global_start = global_row_start + manifest.cell_count
    next_writer_state = materializer.writer_state
    return (next_global_start, next_writer_state)


def _load_corpus_index(corpus_root: Path):
    from .materializers.models import CorpusIndexDocument

    corpus_index_path = corpus_root / "corpus-index.yaml"
    if not corpus_index_path.exists():
        raise FileNotFoundError(f"corpus-index.yaml not found at {corpus_index_path}")
    return CorpusIndexDocument.from_yaml_file(corpus_index_path)


def _infer_backend_topology_from_corpus(corpus_root: Path) -> tuple[str, str]:
    from .materializers.models import MaterializationManifest

    corpus = _load_corpus_index(corpus_root)
    gmeta = corpus.global_metadata or {}

    backend = gmeta.get("backend")
    topology = gmeta.get("topology")

    if (backend is None or topology is None) and corpus.datasets:
        first_manifest_path = corpus_root / corpus.datasets[0].manifest_path
        if first_manifest_path.exists():
            manifest = MaterializationManifest.from_yaml_file(first_manifest_path)
            backend = backend or manifest.backend
            topology = topology or manifest.topology

    if backend is None:
        raise ValueError(
            f"cannot auto-detect backend from {corpus_root / 'corpus-index.yaml'}"
        )
    if topology is None:
        topology = "aggregate" if backend == "lance" else "federated"

    return str(backend), str(topology)


def _cmd_materialize(args: argparse.Namespace) -> None:
    """Unified materialize command supporting create/append, single/bulk, dry-run."""
    from .materializers.registration import corpus_exists as _corpus_exists

    # --- Resolve and validate inputs ---
    sources = _resolve_inputs(args)
    total = len(sources)
    sources_note = (
        f"{total} dataset(s): {', '.join(d.dataset_id for d in sources)}"
    )

    # --- Resolve corpus root ---
    corpus_root = Path(args.output_corpus).resolve()
    corpus_index_path = corpus_root / "corpus-index.yaml"
    effective_backend: str
    effective_topology: str
    existing_corpus = None

    # --- Determine actual mode based on corpus existence ---
    effective_mode: str
    if args.mode == "create":
        if args.backend is None:
            print(
                "[error] --backend is required with --mode create",
                file=sys.stderr,
            )
            sys.exit(1)
        if _corpus_exists(corpus_root):
            print(
                f"[error] --mode create but corpus already exists at {corpus_root}. "
                "Use --mode append to add datasets, or remove the existing corpus.",
                file=sys.stderr,
            )
            sys.exit(1)
        effective_mode = "create"
        effective_backend = args.backend
        requested_topology = args.topology or "federated"
        effective_topology = _resolve_effective_topology(
            effective_backend,
            requested_topology,
        )
    else:  # append
        if not _corpus_exists(corpus_root):
            print(
                f"[error] --mode append but no corpus found at {corpus_root}. "
                "Use --mode create to create a new corpus.",
                file=sys.stderr,
            )
            sys.exit(1)
        effective_mode = "append"
        existing_corpus = _load_corpus_index(corpus_root)
        detected_backend, detected_topology = _infer_backend_topology_from_corpus(
            corpus_root
        )
        if args.backend is not None and args.backend != detected_backend:
            print(
                f"[error] --backend={args.backend} does not match existing corpus backend "
                f"{detected_backend}",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.topology is not None and args.topology != detected_topology:
            print(
                f"[error] --topology={args.topology} does not match existing corpus topology "
                f"{detected_topology}",
                file=sys.stderr,
            )
            sys.exit(1)
        effective_backend = detected_backend
        effective_topology = _resolve_effective_topology(
            detected_backend,
            detected_topology,
        )

    aggregate = _is_aggregate(effective_backend, effective_topology)

    # --- Check for duplicate dataset_ids against existing corpus (append mode) ---
    if effective_mode == "append":
        assert existing_corpus is not None
        existing_ids = {d.dataset_id for d in existing_corpus.datasets}
        for ds in sources:
            if ds.dataset_id in existing_ids:
                print(
                    f"[error] dataset_id '{ds.dataset_id}' already exists in corpus at "
                    f"{corpus_root}. Refusing to overwrite.",
                    file=sys.stderr,
                )
                sys.exit(1)

    # --- Print execution plan ---
    print(
        f"[materialize] mode={effective_mode}  backend={effective_backend}  "
        f"topology={effective_topology}"
    )
    print(f"  corpus: {corpus_root}")
    print(f"  inputs: {sources_note}")
    if aggregate:
        print("  aggregate topology: writer_state will be passed between datasets")

    if args.dry_run:
        print("\n[dry-run] execution plan validated -- no data will be written")
        print(f"  Would materialize {total} dataset(s) into {corpus_root}")
        print(f"  Corpus ID: {args.corpus_id}")
        # Walk through and validate
        next_global = 0
        next_dataset_index = 0
        if effective_mode == "append":
            assert existing_corpus is not None
            next_global = sum(ds.cell_count for ds in existing_corpus.datasets)
            next_dataset_index = len(existing_corpus.datasets)
        writer_state: dict | None = None
        for i, ds in enumerate(sources):
            mode_name = "create" if i == 0 and effective_mode == "create" else "append"
            is_last = aggregate and (i == total - 1)
            next_global, writer_state = _materialize_dataset(
                ds, args,
                backend=effective_backend,
                corpus_root=corpus_root,
                effective_topology=effective_topology,
                mode=mode_name,
                dataset_index=next_dataset_index + i,
                global_row_start=next_global,
                writer_state=writer_state,
                is_last_dataset=is_last,
                total_datasets=total,
                dry_run=True,
                display_index=i,
            )
        print(f"  Estimated total cells: {next_global}")
        return

    # --- Execute materialization ---
    corpus_root.mkdir(parents=True, exist_ok=True)
    next_global = 0
    next_dataset_index = 0
    if effective_mode == "append":
        assert existing_corpus is not None
        next_global = sum(ds.cell_count for ds in existing_corpus.datasets)
        next_dataset_index = len(existing_corpus.datasets)
    writer_state: dict | None = None
    success_count = 0

    for i, ds in enumerate(sources):
        mode_name = "create" if i == 0 and effective_mode == "create" else "append"
        is_last = aggregate and (i == total - 1)
        try:
            next_global, writer_state = _materialize_dataset(
                ds, args,
                backend=effective_backend,
                corpus_root=corpus_root,
                effective_topology=effective_topology,
                mode=mode_name,
                dataset_index=next_dataset_index + i,
                global_row_start=next_global,
                writer_state=writer_state,
                is_last_dataset=is_last,
                total_datasets=total,
                dry_run=False,
                display_index=i,
            )
            success_count += 1
        except Exception as e:
            print(
                f"\n[materialize] ERROR materializing "
                f"{ds.dataset_id}: {e}",
                file=sys.stderr,
            )
            print(
                f"[materialize] {success_count}/{total} dataset(s) succeeded before "
                "failure. Prior datasets remain registered and valid. "
                "Orphaned Lance fragments (if any) can be cleaned up with 'corpus-gc'.",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- Summary ---
    print(f"\n[materialize] done -- {success_count}/{total} dataset(s) materialized")
    print(f"  corpus: {corpus_root}")
    print(f"  corpus-index: {corpus_index_path}")
    print(f"  total cells: {next_global}")

    # Write emission spec if this was a create
    if effective_mode == "create":
        from .materializers.emission_spec import CorpusEmissionSpec

        spec = CorpusEmissionSpec(corpus_id=args.corpus_id)
        spec_path = corpus_root / "corpus-emission-spec.yaml"
        spec.write_yaml(spec_path)
        print(f"  emission-spec: {spec_path}")


# ---------------------------------------------------------------------------
# corpus-validate (kept)
# ---------------------------------------------------------------------------


def _add_corpus_validate_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "corpus_index",
        help="Path to the corpus-index.yaml to validate.",
    )
    sub.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
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
        exists = "\u2713" if manifest_path.exists() else "\u2717 MISSING"
        print(f"  {exists} {ds.dataset_id}  [{ds.join_mode}]")
        if not manifest_path.exists():
            violations.append(
                f"manifest missing for {ds.dataset_id}: {ds.manifest_path}"
            )

        # Validate manifest if it exists
        if manifest_path.exists() and manifest_path.name.endswith("-manifest.yaml"):
            try:
                from .materializers.models import MaterializationManifest

                mnf = MaterializationManifest.from_yaml_file(manifest_path)
                print(
                    f"         backend={mnf.backend}  "
                    f"count={mnf.count_source.selected}"
                )

                # Backend consistency check
                if args.backend and mnf.backend != args.backend:
                    violations.append(
                        f"backend mismatch in {ds.dataset_id}: "
                        f"expected {args.backend}, got {mnf.backend}"
                    )
            except Exception as e:
                violations.append(
                    f"failed to load manifest for {ds.dataset_id}: {e}"
                )

    # Check emission spec
    emission_spec_path = corpus_root / "corpus-emission-spec.yaml"
    emission_exists = emission_spec_path.exists()
    emission_status = "\u2713" if emission_exists else "\u2717 MISSING"
    print(f"Emission spec: {emission_status} {emission_spec_path}")

    if violations:
        print(f"\n[corpus validate] FAIL -- {len(violations)} issue(s):")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("\n[corpus validate] PASS")


# ---------------------------------------------------------------------------
# corpus-gc (new)
# ---------------------------------------------------------------------------


def _add_corpus_gc_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "corpus_root",
        help="Path to the corpus root directory (containing corpus-index.yaml).",
    )
    sub.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report orphaned fragments; do not delete anything.",
    )


def _cmd_corpus_gc(args: argparse.Namespace) -> None:
    """Garbage-collect orphaned Lance fragments not in the corpus ledger.

    Lance writes incremental fragments (data/defragmentation files) that may
    become orphaned if a materialization fails mid-bulk-append. This command:
    1. Loads the corpus ledger to identify all registered datasets.
    2. Scans the corpus root for dataset directories not in the ledger.
    3. Reports (and optionally removes) orphaned dataset directories.

    This is a conservative GC: only per-dataset directories (with meta/ or
    matrix/) that do not appear in the corpus index are considered orphaned.
    Lance internal fragment files within a registered dataset are left alone.
    """
    corpus_root = Path(args.corpus_root).resolve()
    corpus_index_path = corpus_root / "corpus-index.yaml"
    if not corpus_index_path.exists():
        print(f"[error] corpus-index not found at {corpus_index_path}", file=sys.stderr)
        sys.exit(1)

    from .materializers.models import CorpusIndexDocument

    corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)
    registered_ids = {d.dataset_id for d in corpus.datasets}

    print(f"[corpus-gc] scanning {corpus_root}")
    print(f"  registered datasets: {len(registered_ids)}")
    if registered_ids:
        for rid in sorted(registered_ids):
            print(f"    - {rid}")

    # Scan for dataset directories not in the ledger
    orphaned: list[Path] = []
    known_metadata = {
        "corpus-index.yaml", "corpus-ledger.parquet",
        "global-metadata.yaml", "corpus-emission-spec.yaml",
    }

    for entry in sorted(corpus_root.iterdir()):
        if not entry.is_dir():
            continue
        entry_name = entry.name

        # Skip known metadata files
        if entry_name in known_metadata or entry_name.startswith("."):
            continue

        if entry_name in registered_ids:
            continue

        # Check if it looks like a dataset directory
        has_meta = (entry / "meta").is_dir()
        has_matrix = (entry / "matrix").is_dir()
        if has_meta or has_matrix:
            orphaned.append(entry)

    if not orphaned:
        print("\n[corpus-gc] no orphaned dataset directories found")
        return

    print(f"\n[corpus-gc] found {len(orphaned)} orphaned dataset directories:")
    for op in orphaned:
        print(f"  - {op}")

    if args.dry_run:
        print(
            "\n[corpus-gc] dry-run: no directories were deleted. "
            "Run without --dry-run to remove orphaned entries."
        )
        return

    # Remove orphaned directories
    for op in orphaned:
        try:
            shutil.rmtree(op)
            print(f"[corpus-gc] removed: {op}")
        except Exception as e:
            print(
                f"[corpus-gc] WARNING: failed to remove {op}: {e}",
                file=sys.stderr,
            )

    print(f"\n[corpus-gc] done -- removed {len(orphaned)} orphaned directories")


# ---------------------------------------------------------------------------
# backfill-hvg
# ---------------------------------------------------------------------------


def _add_backfill_hvg_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--corpus-root",
        required=True,
        help="Path to the existing Lance corpus root (containing corpus-index.yaml).",
    )
    sub.add_argument(
        "--dataset-id",
        action="append",
        default=None,
        help="Optional dataset_id to backfill; may be repeated.",
    )
    sub.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional alternate root where hvg.parquet files are written. "
            "Defaults to in-place writes under the corpus root."
        ),
    )
    sub.add_argument(
        "--chunk-rows",
        type=int,
        default=50_000,
        help="Number of corpus rows to request per backfill batch (default: 50000).",
    )
    sub.add_argument(
        "--n-hvg",
        type=int,
        default=2000,
        help="Fallback default_n_hvg when the manifest does not already record one.",
    )
    sub.add_argument(
        "--summary-json",
        default=None,
        help="Optional path where a JSON backfill summary should be written.",
    )
    sub.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing hvg.parquet instead of failing.",
    )
    sub.add_argument(
        "--no-update-manifests",
        action="store_false",
        dest="update_manifests",
        help="Do not rewrite materialization-manifest.yaml with hvg_ranking_path/default_n_hvg.",
    )
    sub.set_defaults(update_manifests=True)


def _cmd_backfill_hvg(args: argparse.Namespace) -> None:
    from .materializers import backfill_hvg_rankings_for_corpus

    corpus_root = Path(args.corpus_root).resolve()
    if not corpus_root.exists():
        print(f"[error] corpus root not found: {corpus_root}", file=sys.stderr)
        sys.exit(1)

    try:
        summary = backfill_hvg_rankings_for_corpus(
            corpus_root,
            dataset_ids=args.dataset_id,
            output_root=args.output_root,
            chunk_rows=args.chunk_rows,
            n_hvg=args.n_hvg,
            overwrite=args.overwrite,
            update_manifests=args.update_manifests,
            progress_callback=print,
        )
    except Exception as exc:
        print(f"[backfill-hvg] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.summary_json:
        summary.write_json(args.summary_json)
        print(f"[backfill-hvg] summary-json: {Path(args.summary_json).resolve()}")

    print("[backfill-hvg] done")
    print(f"  corpus_root: {summary.corpus_root}")
    print(f"  topology: {summary.topology}")
    print(f"  datasets: {summary.dataset_count}")
    print(f"  output_root: {summary.output_root}")
    print(f"  update_manifests: {summary.update_manifests}")
    for dataset in summary.datasets:
        print(
            "  - "
            f"{dataset.dataset_id}: cells={dataset.cell_count} "
            f"features={dataset.feature_count} rows={dataset.row_count} "
            f"output={dataset.output_path} sha256={dataset.sha256[:12]}"
        )


# ---------------------------------------------------------------------------
# draft-schema (Phase 3)
# ---------------------------------------------------------------------------


def _add_draft_schema_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--corpus",
        required=True,
        help="Path to the corpus root directory (containing corpus-index.yaml).",
    )
    sub.add_argument(
        "--force-all",
        action="store_true",
        help="Redraft even if draft-schema.yaml already exists.",
    )


def _dataset_paths_for_topology(
    *,
    corpus_root: Path,
    topology: str,
    dataset_id: str,
) -> tuple[Path, Path]:
    resolved = resolve_corpus_paths(
        topology=topology,
        corpus_root=corpus_root,
        dataset_id=dataset_id,
    )
    return resolved.meta_root, resolved.canonical_meta_root


def _cmd_draft_schema(args: argparse.Namespace) -> None:
    from .canonical import draft_canonicalization_schema
    from .inspectors.models import DatasetSummaryDocument

    corpus_root = Path(args.corpus).resolve()
    corpus = _load_corpus_index(corpus_root)
    _, topology = _infer_backend_topology_from_corpus(corpus_root)

    drafted = 0
    skipped_existing = 0
    skipped_canonicalized = 0

    for ds in corpus.datasets:
        dataset_id = ds.dataset_id
        meta_root, canonical_meta_root = _dataset_paths_for_topology(
            corpus_root=corpus_root,
            topology=topology,
            dataset_id=dataset_id,
        )
        canonical_obs = canonical_meta_root / "canonical-obs.parquet"
        canonical_var = canonical_meta_root / "canonical-var.parquet"
        if canonical_obs.exists() and canonical_var.exists():
            skipped_canonicalized += 1
            continue

        summary_path = meta_root / "dataset-summary.yaml"
        draft_path = meta_root / "draft-schema.yaml"

        if not summary_path.exists():
            print(
                f"[draft-schema] missing dataset summary for {dataset_id}: {summary_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        if draft_path.exists() and not args.force_all:
            skipped_existing += 1
            continue

        summary = DatasetSummaryDocument.from_yaml_file(summary_path)
        sampled_obs_values = {
            field.name: list(field.examples)
            for field in summary.obs_fields
        }
        sampled_var_values = {
            field.name: list(field.examples)
            for field in summary.var_fields
        }
        sampled_gene_ids: list[str] = []
        for field in summary.var_fields:
            lowered = field.name.lower()
            if lowered in {
                "feature_id",
                "gene_id",
                "gene",
                "gene_name",
                "gene_symbol",
                "symbol",
                "feature_name",
                "gene_identifier",
            }:
                sampled_gene_ids = list(field.examples)
                break
        schema = draft_canonicalization_schema(
            dataset_id=dataset_id,
            obs_columns=[field.name for field in summary.obs_fields],
            var_columns=[field.name for field in summary.var_fields],
            hints={
                "sampled_obs_values": sampled_obs_values,
                "sampled_var_values": sampled_var_values,
                "sampled_gene_ids": sampled_gene_ids,
                "obs_field_profiles": {
                    field.name: {
                        "dtype": field.dtype,
                        "null_count": field.null_count,
                        "sampled_unique_values": field.sampled_unique_values,
                    }
                    for field in summary.obs_fields
                },
                "control_label_candidates": [
                    {
                        "column": candidate.column,
                        "candidate_values": list(candidate.candidate_values),
                        "suggested_output": candidate.suggested_output,
                        "confidence": candidate.confidence,
                        "reason": candidate.reason,
                    }
                    for candidate in summary.control_label_candidates
                ],
            },
        )
        schema.write_yaml(draft_path)
        drafted += 1
        print(f"[draft-schema] wrote {draft_path}")

    print(
        f"[draft-schema] done — drafted={drafted} "
        f"skipped_existing={skipped_existing} "
        f"skipped_canonicalized={skipped_canonicalized}"
    )


# ---------------------------------------------------------------------------
# canonicalize (Phase 3 simplified CLI)
# ---------------------------------------------------------------------------


def _add_canonicalize_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--corpus",
        required=True,
        help="Path to the corpus root directory (containing corpus-index.yaml).",
    )
    sub.add_argument(
        "--dataset-id",
        default=None,
        help="Optional single dataset identifier; if omitted, canonicalize all with final-schema.yaml.",
    )
    sub.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover schemas and validate inputs without running canonicalization.",
    )


def _resolve_sidecars_from_corpus(
    dataset_id: str,
    corpus_root: Path,
) -> tuple[str, str, str | None]:
    """Look up *dataset_id* in the corpus index and return raw sidecar paths.

    Returns ``(raw_obs_path, raw_var_path, size_factor_path)`` where
    the paths are read from the dataset's ``materialization-manifest.yaml``.
    """
    from .materializers.models import (
        CorpusIndexDocument,
        MaterializationManifest,
    )

    corpus_index_path = corpus_root / "corpus-index.yaml"
    if not corpus_index_path.exists():
        raise FileNotFoundError(
            f"corpus-index.yaml not found at {corpus_index_path}"
        )

    corpus = CorpusIndexDocument.from_yaml_file(corpus_index_path)
    for ds in corpus.datasets:
        if ds.dataset_id == dataset_id:
            manifest_path = corpus_root / ds.manifest_path
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"materialization manifest not found: {manifest_path}"
                )
            manifest = MaterializationManifest.from_yaml_file(manifest_path)
            raw_obs = manifest.raw_cell_meta_path
            raw_var = manifest.raw_feature_meta_path
            size_factor = manifest.size_factor_parquet_path

            if not raw_obs:
                raise ValueError(
                    f"raw_cell_meta_path missing in manifest for {dataset_id}"
                )
            if not raw_var:
                raise ValueError(
                    f"raw_feature_meta_path missing in manifest for {dataset_id}"
                )
            raw_obs_path = Path(raw_obs)
            raw_var_path = Path(raw_var)
            size_factor_path = Path(size_factor) if size_factor else None
            if not raw_obs_path.is_absolute():
                raw_obs_path = corpus_root / raw_obs_path
            if not raw_var_path.is_absolute():
                raw_var_path = corpus_root / raw_var_path
            if size_factor_path is not None and not size_factor_path.is_absolute():
                size_factor_path = corpus_root / size_factor_path
            return (
                str(raw_obs_path),
                str(raw_var_path),
                str(size_factor_path) if size_factor_path is not None else None,
            )

    raise ValueError(
        f"dataset_id '{dataset_id}' not found in corpus index at {corpus_index_path}"
    )


def _cmd_canonicalize(args: argparse.Namespace) -> None:
    """Execute the canonicalize command (bulk or incremental)."""
    from .canonical import (
        CanonicalizationResult,
        run_canonicalization,
    )

    corpus_root = Path(args.corpus).resolve()
    corpus = _load_corpus_index(corpus_root)
    _, topology = _infer_backend_topology_from_corpus(corpus_root)

    all_dataset_ids = [ds.dataset_id for ds in corpus.datasets]
    if args.dataset_id is not None:
        if args.dataset_id not in set(all_dataset_ids):
            print(
                f"[canonicalize] dataset_id '{args.dataset_id}' not found in corpus",
                file=sys.stderr,
            )
            sys.exit(1)
        dataset_ids = [args.dataset_id]
    else:
        dataset_ids = all_dataset_ids

    schemas: list[tuple[str, Path, Path]] = []
    for dataset_id in dataset_ids:
        meta_root, canonical_meta_root = _dataset_paths_for_topology(
            corpus_root=corpus_root,
            topology=topology,
            dataset_id=dataset_id,
        )
        schema_path = meta_root / "final-schema.yaml"
        if schema_path.exists():
            schemas.append((dataset_id, schema_path, canonical_meta_root))
            continue
        if args.dataset_id is not None:
            print(
                f"[canonicalize] final schema not found for {dataset_id}: {schema_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    if not schemas:
        print(
            "[canonicalize] no finalized schemas found (expected meta/<dataset>/final-schema.yaml)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  corpus: {corpus_root}")
    print(f"  dry-run: {args.dry_run}")
    print(f"  datasets ({len(schemas)}): {', '.join(s[0] for s in schemas)}")

    # --- Process each dataset ---
    results: list[CanonicalizationResult] = []
    failures: list[tuple[str, str]] = []

    for dataset_id, schema_path, output_root in schemas:
        print(f"\n  [{dataset_id}]")
        try:
            raw_obs, raw_var, size_factor = _resolve_sidecars_from_corpus(
                dataset_id, corpus_root
            )
            print(f"    raw_obs: {raw_obs}")
            print(f"    raw_var: {raw_var}")
            if size_factor:
                print(f"    size_factor: {size_factor}")
            else:
                print("    size_factor: (none — defaults to 1.0)")
            print(f"    final_schema: {schema_path}")
            print(f"    canonical_output: {output_root}")

            if args.dry_run:
                print(f"    [dry-run] would canonicalize → {output_root}")
                continue

            result = run_canonicalization(
                dataset_id=dataset_id,
                raw_obs_path=raw_obs,
                raw_var_path=raw_var,
                size_factor_path=size_factor,
                schema_path=schema_path,
                output_root=output_root,
            )
            results.append(result)
            print(
                f"    OK  obs={result.obs_rows} rows  var={result.var_rows} rows  "
                f"vocab_size={result.vocab.global_vocab_size}"
            )
            if result.warnings:
                for w in result.warnings:
                    print(f"    WARNING: {w}")
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            failures.append((dataset_id, str(exc)))
            # Continue with remaining datasets on failure (bulk mode resilience)
            continue

    # --- Summary ---
    succeeded = len(results)
    failed = len(failures)
    total = succeeded + failed
    print(f"\n[canonicalize] done — {succeeded}/{total} dataset(s) canonicalized")
    if failures:
        print(f"  {failed} dataset(s) failed:")
        for ds_id, err in failures:
            print(f"    - {ds_id}: {err}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "perturb-data-lab: inspect, materialize, and validate "
            "multi-dataset corpora."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Run h5ad inspection batch.")
    _add_inspect_args(p_inspect)

    # materialize (unified create/append)
    p_mat = sub.add_parser(
        "materialize",
        help="Materialize datasets into a backend storage format (create or append).",
    )
    _add_materialize_args(p_mat)

    # draft-schema
    p_draft = sub.add_parser(
        "draft-schema",
        help="Auto-draft canonicalization schemas from corpus-local inspection summaries.",
    )
    _add_draft_schema_args(p_draft)

    # canonicalize (new — Phase 3)
    p_can = sub.add_parser(
        "canonicalize",
        help="Transform raw obs/var sidecars into canonical parquet files.",
    )
    _add_canonicalize_args(p_can)

    # backfill-hvg
    p_backfill_hvg = sub.add_parser(
        "backfill-hvg",
        help="Backfill canonical hvg.parquet files for an existing Lance corpus.",
    )
    _add_backfill_hvg_args(p_backfill_hvg)

    # corpus-validate (kept)
    p_cv = sub.add_parser(
        "corpus-validate", help="Validate a corpus for logical completeness."
    )
    _add_corpus_validate_args(p_cv)

    # corpus-gc (new)
    p_gc = sub.add_parser(
        "corpus-gc", help="Garbage-collect orphaned dataset directories from a corpus."
    )
    _add_corpus_gc_args(p_gc)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "materialize":
        _cmd_materialize(args)
    elif args.command == "draft-schema":
        _cmd_draft_schema(args)
    elif args.command == "canonicalize":
        _cmd_canonicalize(args)
    elif args.command == "backfill-hvg":
        _cmd_backfill_hvg(args)
    elif args.command == "corpus-validate":
        _cmd_corpus_validate(args)
    elif args.command == "corpus-gc":
        _cmd_corpus_gc(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
