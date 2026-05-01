from __future__ import annotations

from dataclasses import dataclass


CONTRACT_VERSION = "0.3.0"
MISSING_VALUE_LITERAL = "NA"
REQUIRED_ARTIFACTS = (
    "canonical-perturbation-metadata",
    "canonical-context-metadata",
    "feature-registry",
    "dataset-summary",
    "schema",
    "materialization-manifest",
    "corpus-index",
    "global-metadata",
)


@dataclass(frozen=True)
class CanonicalField:
    name: str
    description: str
    required: bool = False
    missing_literal: str = MISSING_VALUE_LITERAL


@dataclass(frozen=True)
class ArtifactContract:
    name: str
    scope: str
    review_format: str
    runtime_model: str
    owner_repo: str
    purpose: str


@dataclass(frozen=True)
class RepoBlueprint:
    repo_name: str
    python_package: str
    purpose: str
    owns_paths: tuple[str, ...]
    excludes: tuple[str, ...]


@dataclass(frozen=True)
class BackendCriterion:
    name: str
    measurement: str
    reason: str


@dataclass(frozen=True)
class GitWorkflowPolicy:
    init_repo: bool
    branch_policy: str
    commit_rule: str
    milestone_examples: tuple[str, ...]


@dataclass(frozen=True)
class Phase1Blueprint:
    perturbation_fields: tuple[CanonicalField, ...]
    context_fields: tuple[CanonicalField, ...]
    artifact_contracts: tuple[ArtifactContract, ...]
    projects: tuple[RepoBlueprint, ...]
    backend_rubric: tuple[BackendCriterion, ...]
    git_policy: GitWorkflowPolicy

    def validate(self) -> None:
        artifact_names = tuple(contract.name for contract in self.artifact_contracts)
        if artifact_names != REQUIRED_ARTIFACTS:
            raise ValueError(f"artifact order mismatch: {artifact_names}")
        for field in self.perturbation_fields + self.context_fields:
            if field.missing_literal != MISSING_VALUE_LITERAL:
                raise ValueError(
                    f"field {field.name} does not use {MISSING_VALUE_LITERAL}"
                )
        for contract in self.artifact_contracts:
            if contract.review_format != "yaml":
                raise ValueError(
                    f"artifact {contract.name} must remain YAML for review"
                )
        if not self.git_policy.init_repo:
            raise ValueError("phase 1 requires git initialization for each repo")


def build_phase1_blueprint() -> Phase1Blueprint:
    blueprint = Phase1Blueprint(
        perturbation_fields=(
            CanonicalField(
                "perturbation_label",
                "Human-readable perturbation label.",
                required=True,
            ),
            CanonicalField(
                "perturbation_type",
                "CRISPR, compound, cytokine, or control class.",
                required=True,
            ),
            CanonicalField("target_id", "Stable target identifier when available."),
            CanonicalField("target_label", "Raw target symbol or fallback label."),
            CanonicalField(
                "control_flag", "Marks control-like interventions.", required=True
            ),
            CanonicalField("dose", "Perturbation dose value after unit normalization."),
            CanonicalField("dose_unit", "Unit paired with the canonical dose."),
            CanonicalField("timepoint", "Elapsed perturbation time."),
            CanonicalField("timepoint_unit", "Unit paired with the timepoint."),
            CanonicalField(
                "combination_key", "Stable `+`-joined key for multi-perturbation rows."
            ),
        ),
        context_fields=(
            CanonicalField(
                "dataset_id", "Dataset-level stable identifier.", required=True
            ),
            CanonicalField(
                "dataset_release",
                "Immutable processed release identifier.",
                required=False,
            ),
            CanonicalField(
                "cell_context", "Primary training context label.", required=True
            ),
            CanonicalField("cell_line_or_type", "Cell line or cell type label."),
            CanonicalField("species", "Source species label."),
            CanonicalField("tissue", "Source tissue or organ label."),
            CanonicalField("assay", "Assay description."),
            CanonicalField(
                "condition",
                "Baseline or treatment context outside the perturbation key.",
            ),
            CanonicalField("batch_id", "Technical batch identifier."),
            CanonicalField("donor_id", "Donor identifier when applicable."),
            CanonicalField("sex", "Sex label when available."),
            CanonicalField("disease_state", "Disease or health state label."),
        ),
        artifact_contracts=(
            ArtifactContract(
                name="canonical-perturbation-metadata",
                scope="cell",
                review_format="yaml",
                runtime_model="CanonicalPerturbationFields",
                owner_repo="perturb-data-lab",
                purpose="Defines additive perturbation metadata fields and NA behavior.",
            ),
            ArtifactContract(
                name="canonical-context-metadata",
                scope="cell",
                review_format="yaml",
                runtime_model="CanonicalContextFields",
                owner_repo="perturb-data-lab",
                purpose="Defines additive context metadata fields and NA behavior.",
            ),
            ArtifactContract(
                name="feature-registry",
                scope="corpus",
                review_format="yaml",
                runtime_model="FeatureRegistryManifest",
                owner_repo="perturb-data-lab",
                purpose="Tracks append-only feature vocabulary and namespace provenance.",
            ),
            ArtifactContract(
                name="dataset-summary",
                scope="dataset",
                review_format="yaml",
                runtime_model="DatasetSummaryDocument",
                owner_repo="perturb-data-lab",
                purpose="Stores lightweight inspector evidence before materialization.",
            ),
            ArtifactContract(
                name="schema",
                scope="dataset",
                review_format="yaml",
                runtime_model="CanonicalizationSchema",
                owner_repo="perturb-data-lab",
                purpose="Maps raw source fields into canonical cell and feature fields with inline null markers for unresolved entries.",
            ),
            ArtifactContract(
                name="materialization-manifest",
                scope="release",
                review_format="yaml",
                runtime_model="MaterializationManifest",
                owner_repo="perturb-data-lab",
                purpose="Captures count source, route, release outputs, and provenance.",
            ),
            ArtifactContract(
                name="corpus-index",
                scope="corpus",
                review_format="yaml",
                runtime_model="CorpusIndexDocument",
                owner_repo="perturb-data-lab",
                purpose="Lists immutable dataset releases and their join mode.",
            ),
            ArtifactContract(
                name="global-metadata",
                scope="global",
                review_format="yaml",
                runtime_model="GlobalMetadataDocument",
                owner_repo="perturb-data-lab",
                purpose="Captures corpus-wide defaults, schema versions, and registry pointers.",
            ),
        ),
        projects=(
            RepoBlueprint(
                repo_name="perturb-data-lab",
                python_package="perturb_data_lab",
                purpose="Owns h5ad inspection, YAML review artifacts, materialization, loaders, and manifests.",
                owns_paths=(
                    "src/perturb_data_lab/inspectors",
                    "src/perturb_data_lab/contracts.py",
                    "src/perturb_data_lab/materializers",
                    "src/perturb_data_lab/loaders",
                    "examples/contracts",
                    "docs",
                ),
                excludes=(
                    "benchmark execution harnesses",
                    "GPU throughput reports",
                ),
            ),
            RepoBlueprint(
                repo_name="perturb-backend-benchmark",
                python_package="perturb_backend_benchmark",
                purpose="Owns backend benchmark scenarios, runners, metrics capture, and ranking reports.",
                owns_paths=(
                    "src/perturb_backend_benchmark/config",
                    "src/perturb_backend_benchmark/runners",
                    "src/perturb_backend_benchmark/reports",
                    "examples",
                    "docs",
                ),
                excludes=(
                    "dataset inspection logic",
                    "canonical materialization code",
                ),
            ),
        ),
        backend_rubric=(
            BackendCriterion(
                "build_cost",
                "time and complexity to build each backend output",
                "Avoid expensive default paths.",
            ),
            BackendCriterion(
                "read_throughput",
                "samples/sec and batches/sec",
                "Keep model training fed.",
            ),
            BackendCriterion(
                "worker_scaling",
                "1, 4, and 8 worker behavior",
                "Detect loader contention.",
            ),
            BackendCriterion(
                "random_access",
                "cost of sparse random reads",
                "Supports sampler flexibility.",
            ),
            BackendCriterion(
                "sequential_streaming",
                "steady-state sequential scan rate",
                "Supports iterable training loops.",
            ),
            BackendCriterion(
                "storage_footprint",
                "bytes on disk for canonical releases",
                "Constrain corpus cost.",
            ),
            BackendCriterion(
                "join_complexity",
                "difficulty of create_new and append flows",
                "Prefer maintainable onboarding.",
            ),
        ),
        git_policy=GitWorkflowPolicy(
            init_repo=True,
            branch_policy="local main branch is enough for phase-scoped work until collaboration requires more",
            commit_rule="commit after each milestone-sized, reviewable change set; do not batch unrelated work",
            milestone_examples=(
                "bootstrap contracts and repo layout",
                "add inspector proposal workflow",
                "wire materialization manifests",
                "record benchmark harness changes",
            ),
        ),
    )
    blueprint.validate()
    return blueprint


BLUEPRINT = build_phase1_blueprint()
