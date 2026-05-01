#!/usr/bin/env python3
"""Standalone script for one-off canonicalization-schema.yaml generation.

Usage::

    python scripts/draft_canonicalization_schema.py \\
        --dataset-id dummy_00 \\
        --obs-cols guide_1 treatment cell_type cellline genotype batch \\
                    donor_id dataset_id cell_id \\
        --var-cols feature_id origin_index \\
        --output schemas/dummy_00/canonicalization-schema.yaml

    # With hints override
    python scripts/draft_canonicalization_schema.py \\
        --dataset-id dummy_00 \\
        --obs-cols guide_1 treatment cell_type ... \\
        --var-cols feature_id origin_index \\
        --hint species=mouse \\
        --hint dataset_index=3 \\
        --output schemas/dummy_00/canonicalization-schema.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from perturb_data_lab.canonical.drafting import draft_canonicalization_schema


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft a canonicalization-schema.yaml from raw column names.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stable dataset identifier (e.g. dummy_00).",
    )
    parser.add_argument(
        "--obs-cols",
        nargs="+",
        required=True,
        help="Space-separated list of raw obs column names.",
    )
    parser.add_argument(
        "--var-cols",
        nargs="+",
        required=True,
        help="Space-separated list of raw var column names.",
    )
    parser.add_argument(
        "--hint",
        nargs="+",
        default=[],
        action="extend",
        help="Key=value hints to override defaults "
             "(e.g. --hint species=mouse --hint dataset_index=5).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for the generated YAML file.",
    )

    args = parser.parse_args()

    # Parse hints
    hints: dict[str, str] = {}
    for h in args.hint:
        if "=" not in h:
            print(f"Warning: hint '{h}' does not contain '='; skipping.", file=sys.stderr)
            continue
        key, _, value = h.partition("=")
        hints[key.strip()] = value.strip()

    # Generate schema
    schema = draft_canonicalization_schema(
        dataset_id=args.dataset_id,
        obs_columns=args.obs_cols,
        var_columns=args.var_cols,
        hints=hints,
    )

    output_path = Path(args.output)
    if output_path.suffix not in (".yaml", ".yml"):
        print(
            f"Warning: output path '{output_path}' does not end with .yaml; "
            f"writing as-is.",
            file=sys.stderr,
        )

    schema.write_yaml(output_path)
    print(f"Drafted canonicalization schema → {output_path}")
    print(f"  Dataset: {schema.dataset_id}")
    print(f"  Status:  {schema.status}")
    print(f"  Obs mappings: {len(schema.obs_column_mappings)}")
    print(f"  Var mappings: {len(schema.var_column_mappings)}")
    print(f"  Obs extensible: {len(schema.obs_extensible)}")
    print(f"  Var extensible: {len(schema.var_extensible)}")
    print(f"  Gene mapping: {schema.gene_mapping.engine}")
    print(f"  Notes: {len(schema.notes)}")
    if schema.notes:
        for note in schema.notes:
            print(f"    - {note}")


if __name__ == "__main__":
    main()
