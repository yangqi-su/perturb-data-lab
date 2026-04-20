from __future__ import annotations

import argparse
from pathlib import Path

from .models import InspectionBatchConfig
from .workflow import run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lightweight h5ad inspection workflows."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the YAML batch inspection config."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of datasets to inspect concurrently.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = InspectionBatchConfig.from_yaml_file(Path(args.config))
    manifest = run_batch(config, workers=args.workers)
    print(
        f"[inspect] wrote manifest {Path(manifest.output_root) / 'inspection-manifest.yaml'}"
    )


if __name__ == "__main__":
    main()
