#!/usr/bin/env python3
"""Optional debugging utility for generating a tiny synthetic dataset.
This must never be used by the real experimental pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def generate_dummy_dataset(output_root: str, classes: list[str] | None = None, images_per_class: int = 8) -> None:
    root = Path(output_root)
    classes = classes or ["plastic", "metal", "glass"]
    for cls in classes:
        class_dir = root / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(images_per_class):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[..., 0] = 120 + idx * 5
            image[..., 1] = 60 + idx * 7
            image[..., 2] = 200 - idx * 3
            Image.fromarray(image).save(class_dir / f"{cls}_{idx}.png")
    print(f"Dummy synthetic dataset generated at {root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic smoke-test data only")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--classes", nargs="*", default=["plastic", "metal", "glass"])
    parser.add_argument("--images-per-class", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_dummy_dataset(args.output_root, args.classes, args.images_per_class)


if __name__ == "__main__":
    main()
