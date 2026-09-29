#!/usr/bin/env python3

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from marine_waste_cil.data.dataset import SUPPORTED_EXTENSIONS, verify_dataset_structure


def parse_args():
    parser = argparse.ArgumentParser(description="Verify marine waste dataset structure")
    parser.add_argument("--data-root", type=str, required=True, help="Path to the dataset root")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    if not root.exists():
        print(f"ERROR: dataset root does not exist: {root}", file=sys.stderr)
        return 1

    summary = verify_dataset_structure(str(root))
    if summary["layout"] == "unknown":
        print("ERROR: dataset layout is not supported. Expected either class directories or train/val/test splits.", file=sys.stderr)
        return 1

    print("Dataset summary")
    print(f"  layout: {summary['layout']}")
    print(f"  file extensions found: {summary['file_extensions']}")
    print(f"  total readable images: {summary['total_images']}")
    print(f"  corrupted/unreadable images: {len(summary['corrupted_images'])}")
    if summary["corrupted_images"]:
        print("  corrupted files:")
        for path in summary["corrupted_images"][:10]:
            print(f"    - {path}")

    if summary["layout"] == "official_split":
        print("  split counts:")
        for split in ("train", "val", "test"):
            counts = summary.get(f"{split}_per_class", {})
            total = sum(counts.values())
            print(f"    {split}: {total}")
    else:
        print(f"  classes discovered: {list(summary['class_counts'].keys())}")
        print("  class image counts:")
        for cls, count in sorted(summary["class_counts"].items()):
            print(f"    {cls}: {count}")

    if summary["corrupted_images"]:
        print("ERROR: dataset contains unreadable or corrupted image files.", file=sys.stderr)
        return 1

    print("SUCCESS: dataset structure is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
