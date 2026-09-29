#!/usr/bin/env python3
"""Build a concise report from an already completed TrashCan experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "overall_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_report(artifact_root: Path, processed_root: Path) -> dict[str, Any]:
    results_root = artifact_root / "results"
    config = _read_json(artifact_root / "locked_configuration.json")
    inspection = _read_json(processed_root / "dataset_inspection.json")
    mapping = _read_json(processed_root / "class_mapping.json")
    stage_protocol = _read_json(processed_root / "cil_stages.json")
    cil_summary = _read_json(results_root / "cil_summary.json")["stages"]
    gradcam = _read_json(artifact_root / "gradcam" / "manifest.json")

    stages = []
    for stage_index in range(len(stage_protocol["stages"])):
        stage_number = stage_index + 1
        test_metrics = _read_json(results_root / f"stage_{stage_number}_metrics.json")
        validation_metrics = _read_json(
            results_root / f"validation_stage_{stage_number}_metrics.json"
        )
        stages.append(
            {
                "stage": stage_number,
                "new_classes": stage_protocol["stages"][stage_index]["new_classes"],
                "cumulative_classes": stage_protocol["stages"][stage_index]["cumulative_classes"],
                "test_metrics": {
                    field: test_metrics[field] for field in METRIC_FIELDS
                },
                "per_class_accuracy": {
                    test_metrics["labels"][entry["class_idx"]]: entry["accuracy"]
                    for entry in test_metrics["per_class"]
                },
                "per_class_metrics": test_metrics["per_class"],
                "confusion_matrix": test_metrics["confusion_matrix"],
                "confusion_matrix_labels": test_metrics["labels"],
                "cil_metrics": cil_summary[stage_index],
                "old_class_accuracy_defined": bool(cil_summary[stage_index]["old_classes"]),
                "validation_metrics": {
                    field: validation_metrics[field] for field in METRIC_FIELDS
                },
            }
        )

    report = {
        "experiment_id": config["experiment_id"],
        "dataset": {
            "source": config["dataset"]["source"],
            "variant": inspection["dataset_variant"],
            "raw_path": inspection["raw_dataset_root"],
            "processed_path": str(processed_root),
            "source_image_count": inspection["source_image_count"],
            "selected_image_count": inspection["selected_images"],
            "excluded_mixed_category_images": inspection["mixed_category_images_excluded"],
            "classes": config["classes"],
            "image_level_rule": mapping["selection_rule"],
            "class_mapping": mapping["project_class_review"],
            "split_method": config["dataset"]["split_method"],
            "split_seed": config["dataset"]["split_seed"],
            "split_image_counts": config["dataset"]["split_image_counts"],
            "split_video_counts": config["dataset"]["split_video_counts"],
            "source_video_overlap": config["dataset"]["source_video_overlap"],
            "exact_duplicate_images": config["dataset"]["exact_duplicate_images"],
        },
        "cil_protocol": stage_protocol,
        "training_configuration": config["locked_training_configuration"],
        "validation_tuning": config["validation_tuning"],
        "stages": stages,
        "final_cumulative_test_accuracy": stages[-1]["test_metrics"]["overall_accuracy"],
        "final_cumulative_test_macro_f1": stages[-1]["test_metrics"]["macro_f1"],
        "repeatability": {
            "final_seed_count": 1,
            "seeds": [config["locked_training_configuration"]["seed"]],
            "limitation": "Only one locked full run was completed on the available local compute budget; no mean or standard deviation is reported.",
        },
        "gradcam": {
            "manifest": str(artifact_root / "gradcam" / "manifest.json"),
            "sample_count": gradcam["total_samples"],
            "correct_count": sum(sample["correct"] for sample in gradcam["samples"]),
            "incorrect_count": sum(not sample["correct"] for sample in gradcam["samples"]),
            "samples": gradcam["samples"],
        },
        "limitations": [
            "ImageNet weights were not cached locally; internet downloads were prohibited, so ResNet18 used random initialization.",
            "2,722 multi-category images were excluded to preserve a defensible single-label image-classification target.",
            "The held-out test split has only 10 trash_wood images; that class's test accuracy is high-variance.",
            "Only one final random seed was run.",
            "The final stage-3 test macro-F1 is modest and substantial forgetting remains; results should be reported with these limitations.",
            "Stage 1 has no old classes; the pipeline serializes old-class accuracy as 0.0 for that stage, which should be interpreted as not applicable.",
        ],
        "commands": [
            "python scripts/prepare_trashcan.py --raw-root dataset/material_version --output-root data/TrashCan_processed_seed42",
            "python verify_dataset.py --data-root data/TrashCan_processed_seed42",
            "python main.py --full-experiment --data-root data/TrashCan_processed_seed42 --output-dir artifacts/real_trashcan_experiment --epochs 2 --batch-size 16 --learning-rate 0.0001 --memory-size 400 --memory-budget-per-class 20 --temperature 2.0 --kd-weight 1.0 --device mps --no-pretrained --seed 42",
            "python main.py --gradcam --checkpoint artifacts/real_trashcan_experiment/checkpoints/stage_3.pt --data-root data/TrashCan_processed_seed42 --output-dir artifacts/real_trashcan_experiment --samples-per-class 1 --device mps --no-pretrained --seed 42",
        ],
        "output_locations": {
            "checkpoints": str(artifact_root / "checkpoints"),
            "results": str(results_root),
            "gradcam": str(artifact_root / "gradcam"),
            "training_log": str(artifact_root / "logs" / "train.log"),
        },
    }
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    training = report["training_configuration"]
    lines = [
        "# TrashCan CIL Experiment Report",
        "",
        f"- Dataset: {dataset['source']} / {dataset['variant']}",
        f"- Images: {dataset['selected_image_count']} selected of {dataset['source_image_count']} source images",
        f"- Classes: {', '.join(dataset['classes'])}",
        f"- Split: {dataset['split_image_counts']} (video-group disjoint)",
        f"- Model: {training['architecture']}; ImageNet pretrained={training['pretrained_imagenet']}; device={training['device']}",
        f"- Train: {training['epochs_per_stage']} epochs/stage, batch={training['batch_size']}, LR={training['learning_rate']}, replay/class={training['replay_budget_per_class']}, KD T={training['distillation_temperature']}, KD weight={training['distillation_weight']}",
        "",
        "| Stage | Test accuracy | Macro F1 | Weighted F1 | Old accuracy | New accuracy | Mean forgetting | Mean retention |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report["stages"]:
        test = stage["test_metrics"]
        cil = stage["cil_metrics"]
        old_accuracy = (
            f"{cil['old_class_accuracy']:.4f}"
            if stage["old_class_accuracy_defined"]
            else "N/A"
        )
        lines.append(
            f"| {stage['stage']} | {test['overall_accuracy']:.4f} | {test['macro_f1']:.4f} | {test['weighted_f1']:.4f} | {old_accuracy} | {cil['new_class_accuracy']:.4f} | {cil['mean_forgetting']:.4f} | {cil['mean_retention']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Final cumulative test accuracy: {report['final_cumulative_test_accuracy']:.4f}",
            f"Final cumulative test macro-F1: {report['final_cumulative_test_macro_f1']:.4f}",
            "",
            "## Limitations",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            f"Checkpoints: `{report['output_locations']['checkpoints']}`",
            f"Results: `{report['output_locations']['results']}`",
            f"Grad-CAM: `{report['output_locations']['gradcam']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/real_trashcan_experiment"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/TrashCan_processed_seed42"))
    args = parser.parse_args()
    report = build_report(args.artifact_root, args.processed_root)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    (args.artifact_root / "research_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.artifact_root / "research_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(f"Wrote {args.artifact_root / 'research_report.json'}")
    print(f"Wrote {args.artifact_root / 'research_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())