"""CLI: top-level multi-command interface for the perturb-data-lab workflow.

Commands
--------
inspect          Run h5ad inspection over a batch config.
materialize      Materialize datasets with --mode {create,append} into a backend.
                 Supports single (--source) and bulk (--input-dir, --input-list)
                 input. Outputs go under --output-corpus.
canonicalize     Transform raw obs/var sidecars into canonical parquet files.
                 Supports bulk (--schema-dir) and incremental (--dataset-id) modes.
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

BACKEND_CHOICES = ["arrow-parquet", "arrow-ipc", "webdataset", "zarr", "lance"]
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

    Lance is contractually aggregate-only in Stage 2 corpus materialization.
    If the caller keeps the parser default (``topology='federated'``) while
    selecting ``backend='lance'``, we route as aggregate to ensure all datasets
    append into ``{corpus}/matrix/aggregated-cells.lance``.
    """
    if backend == "lance":
        return "aggregate"
    return topology


# ---------------------------------------------------------------------------
# inspect
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
        f"[inspect] wrote manifest "
        f"{Path(manifest.output_root) / 'inspection-manifest.yaml'}"
    )


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
        required=True,
        choices=BACKEND_CHOICES,
        help="Storage backend for this materialization.",
    )
    sub.add_argument(
        "--topology",
        default="federated",
        choices=TOPOLOGY_CHOICES,
        help="Corpus topology (default: federated).",
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
    corpus_root: Path,
    effective_topology: str,
    mode: str,
    dataset_index: int,
    global_row_start: int,
    writer_state: dict | None,
    is_last_dataset: bool,
    total_datasets: int,
    dry_run: bool,
) -> tuple[int, dict | None]:
    """Materialize a single dataset and return (next_global_row_start, next_writer_state).

    In aggregate mode, writer_state is carried between calls.
    """
    from .materializers import Stage2Materializer
    from .materializers.models import OutputRoots

    if dry_run:
        print(
            f"  [{dataset_index + 1}/{total_datasets}] DRY-RUN: "
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
        f"\n  [{dataset_index + 1}/{total_datasets}] materializing "
        f"{ds.dataset_id}"
    )
    print(f"    source: {ds.source}")
    print(f"    mode: {mode}")

    materializer = Stage2Materializer(
        source_path=ds.source,
        review_bundle_path=ds.review_bundle,
        output_roots=output_roots,
        dataset_id=ds.dataset_id,
        backend=args.backend,
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

    effective_topology = _resolve_effective_topology(args.backend, args.topology)
    aggregate = _is_aggregate(args.backend, args.topology)

    # --- Determine actual mode based on corpus existence ---
    effective_mode: str
    if args.mode == "create":
        if _corpus_exists(corpus_root):
            print(
                f"[error] --mode create but corpus already exists at {corpus_root}. "
                "Use --mode append to add datasets, or remove the existing corpus.",
                file=sys.stderr,
            )
            sys.exit(1)
        effective_mode = "create"
    else:  # append
        if not _corpus_exists(corpus_root):
            print(
                f"[error] --mode append but no corpus found at {corpus_root}. "
                "Use --mode create to create a new corpus.",
                file=sys.stderr,
            )
            sys.exit(1)
        effective_mode = "append"

    # --- Check for duplicate dataset_ids against existing corpus (append mode) ---
    if effective_mode == "append":
        from .materializers.models import CorpusIndexDocument

        existing = CorpusIndexDocument.from_yaml_file(corpus_index_path)
        existing_ids = {d.dataset_id for d in existing.datasets}
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
        f"[materialize] mode={effective_mode}  backend={args.backend}  "
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
        writer_state: dict | None = None
        for i, ds in enumerate(sources):
            mode_name = "create" if i == 0 and effective_mode == "create" else "append"
            is_last = aggregate and (i == total - 1)
            next_global, writer_state = _materialize_dataset(
                ds, args,
                corpus_root=corpus_root,
                effective_topology=effective_topology,
                mode=mode_name,
                dataset_index=i,
                global_row_start=next_global,
                writer_state=writer_state,
                is_last_dataset=is_last,
                total_datasets=total,
                dry_run=True,
            )
        print(f"  Estimated total cells: {next_global}")
        return

    # --- Execute materialization ---
    corpus_root.mkdir(parents=True, exist_ok=True)
    next_global = 0
    writer_state: dict | None = None
    success_count = 0

    for i, ds in enumerate(sources):
        mode_name = "create" if i == 0 and effective_mode == "create" else "append"
        is_last = aggregate and (i == total - 1)
        try:
            next_global, writer_state = _materialize_dataset(
                ds, args,
                corpus_root=corpus_root,
                effective_topology=effective_topology,
                mode=mode_name,
                dataset_index=i,
                global_row_start=next_global,
                writer_state=writer_state,
                is_last_dataset=is_last,
                total_datasets=total,
                dry_run=False,
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

    # Check tokenizer
    tokenizer_path = corpus_root / "tokenizer.json"
    tokenizer_exists = tokenizer_path.exists()
    tokenizer_status = "\u2713" if tokenizer_exists else "\u2717 (optional since Phase 3)"
    print(f"\nTokenizer: {tokenizer_status} {tokenizer_path}")

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
        "tokenizer.json",
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
# canonicalize (new — Phase 3)
# ---------------------------------------------------------------------------


def _add_canonicalize_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--schema-dir",
        default=None,
        help=(
            "Directory containing finalized canonicalization-schema YAML files "
            "(bulk mode). Each schema's dataset_id is read from the YAML and "
            "matched against the corpus index; raw sidecar paths are discovered "
            "automatically from each dataset's materialization-manifest.yaml."
        ),
    )
    sub.add_argument(
        "--dataset-id",
        default=None,
        help="Single dataset identifier (incremental mode; requires --schema).",
    )
    sub.add_argument(
        "--schema",
        default=None,
        help=(
            "Path to a single canonicalization-schema.yaml "
            "(incremental mode; requires --dataset-id)."
        ),
    )
    sub.add_argument(
        "--corpus",
        required=True,
        help="Path to the corpus root directory (containing corpus-index.yaml).",
    )
    sub.add_argument(
        "--output-dir",
        required=True,
        help="Directory where canonical parquet files will be written.",
    )
    sub.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover schemas and validate inputs without running canonicalization.",
    )


def _discover_schemas_from_dir(schema_dir: Path) -> list[tuple[str, Path]]:
    """Scan *schema_dir* for ``*canonicalization-schema.yaml`` files.

    Returns a list of ``(dataset_id, schema_path)`` pairs sorted by
    dataset_id.  The ``dataset_id`` is read from each schema's YAML
    content, not from the filename.
    """
    from .canonical import CanonicalizationSchema

    candidates = sorted(schema_dir.glob("*canonicalization-schema.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No *canonicalization-schema.yaml files found in {schema_dir}"
        )
    schemas: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for sp in candidates:
        schema = CanonicalizationSchema.from_yaml_file(sp)
        if schema.dataset_id in seen:
            print(
                f"[canonicalize] WARNING: duplicate dataset_id '{schema.dataset_id}' "
                f"in {sp}; skipping duplicate schema",
                file=sys.stderr,
            )
            continue
        seen.add(schema.dataset_id)
        schemas.append((schema.dataset_id, sp))
    return schemas


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
            return (raw_obs, raw_var, size_factor)

    raise ValueError(
        f"dataset_id '{dataset_id}' not found in corpus index at {corpus_index_path}"
    )


def _cmd_canonicalize(args: argparse.Namespace) -> None:
    """Execute the canonicalize command (bulk or incremental)."""
    from .canonical import (
        CanonicalizationResult,
        CanonicalVocab,
        build_canonical_vocab,
        run_canonicalization,
    )

    corpus_root = Path(args.corpus).resolve()
    output_dir = Path(args.output_dir).resolve()

    # --- Resolve mode ---
    has_schema_dir = args.schema_dir is not None
    has_dataset_id = args.dataset_id is not None
    has_schema = args.schema is not None

    if has_schema_dir:
        if has_dataset_id or has_schema:
            print(
                "[canonicalize] --schema-dir is for bulk mode and cannot be "
                "combined with --dataset-id/--schema.",
                file=sys.stderr,
            )
            sys.exit(1)
        schema_dir = Path(args.schema_dir)
        if not schema_dir.is_dir():
            print(
                f"[canonicalize] schema directory not found: {schema_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        schemas = _discover_schemas_from_dir(schema_dir)
        print(
            f"[canonicalize] bulk mode: discovered {len(schemas)} schema(s) "
            f"in {schema_dir}"
        )
    elif has_dataset_id and has_schema:
        schema_path = Path(args.schema)
        if not schema_path.exists():
            print(
                f"[canonicalize] schema file not found: {schema_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        schemas = [(args.dataset_id, schema_path)]
        print(
            f"[canonicalize] incremental mode: dataset_id={args.dataset_id} "
            f"schema={schema_path}"
        )
    else:
        print(
            "[canonicalize] must specify either --schema-dir (bulk) or "
            "both --dataset-id and --schema (incremental).",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Print plan ---
    print(f"  corpus: {corpus_root}")
    print(f"  output-dir: {output_dir}")
    print(f"  dry-run: {args.dry_run}")
    if schemas:
        ds_ids = [s[0] for s in schemas]
        print(f"  datasets ({len(ds_ids)}): {', '.join(ds_ids)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Process each dataset ---
    results: list[CanonicalizationResult] = []
    failures: list[tuple[str, str]] = []

    for dataset_id, schema_path in schemas:
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

            if args.dry_run:
                print(f"    [dry-run] would canonicalize → {output_dir}")
                continue

            result = run_canonicalization(
                dataset_id=dataset_id,
                raw_obs_path=raw_obs,
                raw_var_path=raw_var,
                size_factor_path=size_factor,
                schema_path=schema_path,
                output_root=output_dir,
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

    # --- Merge and write corpus-level vocab ---
    if not args.dry_run and results:
        per_dataset_vocabs = [r.vocab for r in results]
        corpus_vocab_path = output_dir / "corpus-vocab.yaml"
        merged = build_canonical_vocab(
            per_dataset_vocabs,
            output_path=corpus_vocab_path,
        )
        print(
            f"\n[canonicalize] merged vocab: {corpus_vocab_path}  "
            f"global_vocab_size={merged.global_vocab_size}"
        )

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

    # canonicalize (new — Phase 3)
    p_can = sub.add_parser(
        "canonicalize",
        help="Transform raw obs/var sidecars into canonical parquet files.",
    )
    _add_canonicalize_args(p_can)

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
    elif args.command == "canonicalize":
        _cmd_canonicalize(args)
    elif args.command == "corpus-validate":
        _cmd_corpus_validate(args)
    elif args.command == "corpus-gc":
        _cmd_corpus_gc(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
