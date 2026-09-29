#!/usr/bin/env python3
"""Inspect and validate a marine-waste dataset.

Usage:
    python scripts/inspect_dataset.py --data-root /path/to/TrashCan
"""
import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from marine_waste_cil.config import ensure_output_dirs, OUTPUT_DIR
from marine_waste_cil.data.dataset import inspect_dataset, validate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a TrashCan / marine waste dataset on disk."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Where to write the inspection JSON report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code if the dataset is missing required splits or has corrupted images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.data_root)
    if not root.exists():
        print(f"ERROR: dataset root does not exist: {root}", file=sys.stderr)
        return 1
    try:
        if args.strict:
            report = validate_dataset(str(root), require_splits=False)
        else:
            report = inspect_dataset(str(root))
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    report.pprint()
    output_dirs = ensure_output_dirs(args.output_dir)
    report_path = Path(output_dirs["results"]) / "dataset_inspection.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report.to_dict(), fp, indent=2)
    print(f"\nInspection report written to: {report_path}")
    if args.strict and report.corrupted_images:
        print(
            f"ERROR: strict mode: {len(report.corrupted_images)} corrupted image(s) detected.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
