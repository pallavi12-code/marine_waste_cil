#!/usr/bin/env python3
"""Prepare a video-group-disjoint TrashCan-Material classification dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


CLASS_ORDER = [
    "trash_plastic",
    "trash_metal",
    "trash_wood",
    "trash_etc",
    "animal_fish",
    "rov",
]
STAGES = [
    ["trash_plastic", "trash_metal"],
    ["trash_wood", "trash_etc"],
    ["animal_fish", "rov"],
]
MIN_PURE_IMAGES = 100
MIN_SOURCE_VIDEOS = 15
SPLIT_SEED = 42
VIDEO_PATTERN = re.compile(r"^(vid_\d+)_")

PROJECT_CLASS_REVIEW = {
    "plastic": {"material_label": "trash_plastic", "status": "direct"},
    "metal": {"material_label": "trash_metal", "status": "direct"},
    "glass": {"material_label": None, "status": "not_available"},
    "paper": {"material_label": "trash_paper", "status": "support_too_low"},
    "rubber": {"material_label": "trash_rubber", "status": "support_too_low"},
    "cloth": {"material_label": "trash_fabric", "status": "renamed_synonym_below_video_support_threshold"},
    "wood": {"material_label": "trash_wood", "status": "direct"},
    "foam": {"material_label": None, "status": "not_available"},
    "rope": {"material_label": "trash_fishing_gear", "status": "merged_category_support_too_low"},
    "net": {"material_label": "trash_fishing_gear", "status": "merged_category_support_too_low"},
    "fish": {"material_label": "animal_fish", "status": "renamed_taxonomic_label"},
    "crab": {"material_label": "animal_crab", "status": "support_too_low"},
    "seagrass": {"material_label": "plant", "status": "broader_category_support_too_low"},
    "seaweed": {"material_label": "plant", "status": "broader_category_support_too_low"},
    "jellyfish": {"material_label": None, "status": "not_available_as_exact_label"},
    "shell": {"material_label": "animal_shells", "status": "ambiguous_semantics_support_too_low"},
    "coral": {"material_label": None, "status": "not_available_as_exact_label"},
    "rock": {"material_label": None, "status": "not_available"},
    "rov": {"material_label": "rov", "status": "direct"},
    "other": {"material_label": "trash_etc", "status": "narrower_trash_catchall"},
}


def _read_coco_split(root: Path, split: str) -> tuple[dict[str, Any], dict[int, str]]:
    annotation_path = root / f"instances_{split}_trashcan.json"
    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    categories = {int(category["id"]): str(category["name"]) for category in data["categories"]}
    return data, categories


def _video_id(file_name: str) -> str:
    match = VIDEO_PATTERN.match(file_name)
    if match is None:
        raise ValueError(f"Image filename does not encode a source video: {file_name}")
    return match.group(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary_path.replace(path)


def prepare_dataset(raw_root: Path, output_root: Path) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    output_root = output_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"TrashCan-Material root not found: {raw_root}")
    if output_root.exists():
        raise FileExistsError(
            f"Processed output already exists; refusing to overwrite it: {output_root}"
        )

    records: list[dict[str, Any]] = []
    category_object_counts: Counter[str] = Counter()
    pure_image_counts: Counter[str] = Counter()
    source_videos_by_category: dict[str, set[str]] = defaultdict(set)
    source_image_counts: Counter[str] = Counter()
    image_dimensions: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    mixed_class_images = 0
    duplicate_names: set[str] = set()
    seen_names: set[str] = set()

    for split in ("train", "val"):
        data, categories = _read_coco_split(raw_root, split)
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data["annotations"]:
            category_name = categories[int(annotation["category_id"])]
            category_object_counts[category_name] += 1
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        for image in data["images"]:
            file_name = str(image["file_name"])
            source_image_counts[split] += 1
            if file_name in seen_names:
                duplicate_names.add(file_name)
            seen_names.add(file_name)
            video_id = _video_id(file_name)
            image_annotations = annotations_by_image.get(int(image["id"]), [])
            if not image_annotations:
                raise ValueError(f"Image has no object annotations: {file_name}")

            source_path = raw_root / split / file_name
            if not source_path.is_file():
                raise FileNotFoundError(f"Annotation image is missing: {source_path}")
            try:
                with Image.open(source_path) as image_file:
                    image_file.verify()
                with Image.open(source_path) as image_file:
                    actual_size = image_file.size
                    actual_format = image_file.format
            except Exception as exc:
                raise ValueError(f"Invalid image {source_path}: {exc}") from exc
            metadata_size = (int(image["width"]), int(image["height"]))
            if actual_size != metadata_size:
                raise ValueError(
                    f"Image dimensions differ from COCO metadata: {source_path} "
                    f"actual={actual_size} metadata={metadata_size}"
                )
            image_dimensions[f"{actual_size[0]}x{actual_size[1]}"] += 1
            extensions[(actual_format or source_path.suffix.lstrip(".")).lower()] += 1

            image_categories = {
                categories[int(annotation["category_id"])]
                for annotation in image_annotations
            }
            if len(image_categories) != 1:
                mixed_class_images += 1
                continue

            category_name = next(iter(image_categories))
            pure_image_counts[category_name] += 1
            source_videos_by_category[category_name].add(video_id)
            records.append(
                {
                    "file_name": file_name,
                    "class_name": category_name,
                    "source_video": video_id,
                    "original_split": split,
                    "raw_relative_path": f"{split}/{file_name}",
                    "image_width": int(image["width"]),
                    "image_height": int(image["height"]),
                    "annotation_count": len(image_annotations),
                }
            )

    if duplicate_names:
        raise ValueError(
            f"Duplicate image filenames across official splits: {sorted(duplicate_names)[:10]}"
        )

    qualifying_classes = sorted(
        category_name
        for category_name, image_count in pure_image_counts.items()
        if image_count >= MIN_PURE_IMAGES
        and len(source_videos_by_category[category_name]) >= MIN_SOURCE_VIDEOS
    )
    if qualifying_classes != sorted(CLASS_ORDER):
        raise ValueError(
            "Observed classes do not match the locked TrashCan protocol. "
            f"qualifying={qualifying_classes}, configured={sorted(CLASS_ORDER)}"
        )
    records = [record for record in records if record["class_name"] in CLASS_ORDER]
    records.sort(key=lambda record: record["file_name"])

    labels = np.asarray([record["class_name"] for record in records])
    groups = np.asarray([record["source_video"] for record in records])
    outer_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SPLIT_SEED
    )
    test_indices = next(outer_splitter.split(np.zeros(len(records)), labels, groups))[1]
    remaining_indices = np.setdiff1d(np.arange(len(records)), test_indices)
    validation_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SPLIT_SEED + 100
    )
    validation_relative_indices = next(
        validation_splitter.split(
            np.zeros(len(remaining_indices)),
            labels[remaining_indices],
            groups[remaining_indices],
        )
    )[1]
    validation_indices = remaining_indices[validation_relative_indices]
    train_indices = np.setdiff1d(remaining_indices, validation_indices)
    split_indices = {
        "train": train_indices,
        "val": validation_indices,
        "test": test_indices,
    }
    split_for_index = {
        int(index): split
        for split, indices in split_indices.items()
        for index in indices
    }

    split_groups = {
        split: {records[int(index)]["source_video"] for index in indices}
        for split, indices in split_indices.items()
    }
    if (
        split_groups["train"] & split_groups["val"]
        or split_groups["train"] & split_groups["test"]
        or split_groups["val"] & split_groups["test"]
    ):
        raise AssertionError("Source-video leakage remains in the generated splits")

    final_records: list[dict[str, Any]] = []
    class_counts_by_split: dict[str, Counter[str]] = {
        split: Counter() for split in split_indices
    }
    content_hashes: dict[str, str] = {}
    for index, record in enumerate(records):
        split = split_for_index[index]
        source_path = raw_root / record["raw_relative_path"]
        content_hash = _sha256(source_path)
        previous = content_hashes.get(content_hash)
        if previous is not None:
            raise ValueError(
                f"Byte-identical images found in the selected dataset: {previous}, {record['file_name']}"
            )
        content_hashes[content_hash] = record["file_name"]
        processed_path = output_root / split / record["class_name"] / record["file_name"]
        final_records.append(
            {
                **record,
                "split": split,
                "image_path": str(processed_path),
                "sha256": content_hash,
            }
        )
        class_counts_by_split[split][record["class_name"]] += 1

    for split, class_counts in class_counts_by_split.items():
        missing = sorted(set(CLASS_ORDER) - set(class_counts))
        if missing:
            raise ValueError(f"Split {split} has no samples for configured classes: {missing}")

    counts_payload = {
        split: dict(sorted(counts.items()))
        for split, counts in class_counts_by_split.items()
    }
    stage_payload = {
        "dataset_variant": "TrashCan-Material",
        "class_order": CLASS_ORDER,
        "stages": [
            {
                "stage": stage_index + 1,
                "new_classes": stage,
                "cumulative_classes": [
                    name for earlier in STAGES[: stage_index + 1] for name in earlier
                ],
            }
            for stage_index, stage in enumerate(STAGES)
        ],
    }
    mapping_payload = {
        "dataset_variant": "TrashCan-Material",
        "selected_native_categories": CLASS_ORDER,
        "selection_rule": {
            "image_label": "Keep an image only if every COCO annotation has the same category; retain the image once even when it contains multiple same-category objects.",
            "mixed_category_images": "Excluded; they are not duplicated into multiple class samples.",
            "minimum_pure_images": MIN_PURE_IMAGES,
            "minimum_distinct_source_videos": MIN_SOURCE_VIDEOS,
        },
        "project_class_review": PROJECT_CLASS_REVIEW,
    }
    manifest_payload = {
        "protocol_version": 1,
        "raw_dataset_root": str(raw_root),
        "dataset_variant": "TrashCan-Material",
        "official_splits_used_as_sources": ["train", "val"],
        "split_method": "StratifiedGroupKFold over source video IDs parsed from vid_######_ filenames. First 1/5 fold is held-out test; a second 1/5 fold of the remaining groups is validation.",
        "split_seed": SPLIT_SEED,
        "validation_seed": SPLIT_SEED + 1,
        "class_order": CLASS_ORDER,
        "class_counts": counts_payload,
        "source_video_counts": {
            split: len(groups) for split, groups in split_groups.items()
        },
        "samples": final_records,
    }
    selected_counts = Counter(record["class_name"] for record in final_records)
    inspection_payload = {
        "dataset_variant": "TrashCan-Material",
        "raw_dataset_root": str(raw_root),
        "source_images_by_official_split": dict(sorted(source_image_counts.items())),
        "source_image_count": sum(source_image_counts.values()),
        "mixed_category_images_excluded": mixed_class_images,
        "single_category_images_before_support_filter": sum(pure_image_counts.values()),
        "selected_images": len(final_records),
        "selected_class_counts_before_split": dict(sorted(selected_counts.items())),
        "selected_class_counts_by_split": counts_payload,
        "selected_source_video_groups_by_split": {
            split: len(groups) for split, groups in split_groups.items()
        },
        "official_category_object_annotation_counts": dict(sorted(category_object_counts.items())),
        "source_image_dimensions": dict(sorted(image_dimensions.items())),
        "source_file_formats": dict(sorted(extensions.items())),
        "exact_duplicate_images_in_selected_set": 0,
        "video_group_intersections": {"train_val": 0, "train_test": 0, "val_test": 0},
        "class_mapping": mapping_payload,
        "cil_protocol": stage_payload,
    }

    output_root.mkdir(parents=True, exist_ok=False)
    for split in split_indices:
        for class_name in CLASS_ORDER:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)
    for record in final_records:
        destination = Path(record["image_path"])
        source_path = raw_root / record["raw_relative_path"]
        destination.symlink_to(source_path)

    _atomic_json(output_root / "split_manifest.json", manifest_payload)
    _atomic_json(output_root / "class_mapping.json", mapping_payload)
    _atomic_json(output_root / "cil_stages.json", stage_payload)
    _atomic_json(output_root / "dataset_inspection.json", inspection_payload)
    return inspection_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("dataset/material_version"),
        help="TrashCan-Material root; this directory is read-only to this script.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/TrashCan_processed"),
        help="New processed dataset location. Existing directories are never overwritten.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare_dataset(args.raw_root, args.output_root)
    print(json.dumps(report, indent=2))
    print(f"Processed dataset created at: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())