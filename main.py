#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from marine_waste_cil.config import (
    CIL_STAGES,
    DATA_ROOT,
    DEFAULT_TRASHCAN_INSTANCE_CLASSES,
    OUTPUT_DIR,
    RANDOM_SEED,
    build_runtime_config,
    ensure_output_dirs,
    get_classes_up_to_stage,
    get_logger,
    save_class_order,
    save_experiment_config,
    set_seed,
    validate_cil_stages_against_dataset,
)
from marine_waste_cil.data.dataset import (
    build_samples_for_official_split,
    build_samples_from_directory,
    build_samples_from_coco,
    class_distribution_from_samples,
    discover_dataset_layout,
    get_class_names_in_order,
    inspect_dataset,
    validate_dataset,
    validate_no_split_leak,
)
from marine_waste_cil.data.transforms import build_default_transforms, build_eval_transforms
from marine_waste_cil.explainability.gradcam import generate_gradcam_for_dataset
from marine_waste_cil.training.trainer import CILTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Marine Waste Class-Incremental Learning (CIL) pipeline."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.getenv("MARINE_CIL_DATA_ROOT", DATA_ROOT),
        help="Path to the dataset root directory.",
    )
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Artifacts output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per stage.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--memory-size", type=int, default=None, help="Total replay memory size.")
    parser.add_argument("--memory-budget-per-class", type=int, default=None, help="Exemplars per class.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or mps.")
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Use inverse-frequency weighted sampling for training batches.",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--temperature", type=float, default=None, help="KD temperature.")
    parser.add_argument("--kd-weight", type=float, default=None, help="KD loss weight.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--inspect-dataset", action="store_true", help="Inspect and validate dataset.")
    action.add_argument("--train", action="store_true", help="Run the 3-stage CIL training pipeline.")
    action.add_argument("--evaluate", action="store_true", help="Evaluate a checkpoint on the test split.")
    action.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualizations.")
    action.add_argument("--full-experiment", action="store_true", help="Run the complete end-to-end experiment.")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Train and report validation metrics without opening test images.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file for evaluation, gradcam, or resume.")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint.")
    parser.add_argument("--stage", type=int, default=None, help="Target stage for evaluation.")
    parser.add_argument("--samples-per-class", type=int, default=3, help="Grad-CAM samples per class.")
    return parser.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "DATA_ROOT": args.data_root,
        "OUTPUT_DIR": args.output_dir,
        "RANDOM_SEED": args.seed,
    }
    if args.epochs is not None:
        overrides["NUM_EPOCHS_PER_STAGE"] = args.epochs
    if args.batch_size is not None:
        overrides["BATCH_SIZE"] = args.batch_size
    if args.learning_rate is not None:
        overrides["LEARNING_RATE"] = args.learning_rate
    if args.memory_size is not None:
        overrides["REPLAY_MEMORY_SIZE"] = args.memory_size
    if args.memory_budget_per_class is not None:
        overrides["MEMORY_BUDGET_PER_CLASS"] = args.memory_budget_per_class
    if args.num_workers is not None:
        overrides["NUM_WORKERS"] = args.num_workers
    if args.balanced_sampling:
        overrides["BALANCED_SAMPLING"] = True
    if args.device is not None:
        overrides["DEVICE"] = args.device
    if args.no_pretrained:
        overrides["USE_PRETRAINED"] = False
    if args.temperature is not None:
        overrides["DISTILLATION_TEMPERATURE"] = args.temperature
    if args.kd_weight is not None:
        overrides["DISTILLATION_WEIGHT"] = args.kd_weight
    return overrides


def _discover_and_build_samples(data_root: str, seed: int) -> Dict[str, Any]:
    layout_info = discover_dataset_layout(data_root)
    if layout_info["layout"] == "unknown":
        raise ValueError(f"Unsupported dataset layout at {data_root}")
    if layout_info["layout"] == "official_split":
        class_names = list(layout_info["classes"])
        data_by_split = build_samples_for_official_split(data_root, class_names)
    elif layout_info["layout"] == "coco_annotations":
        class_names = list(layout_info["classes"])
        data_by_split = build_samples_from_coco(data_root, class_names, random_seed=seed)
    else:
        class_names = get_class_names_in_order(data_root)
        data_by_split = build_samples_from_directory(data_root, class_names, random_seed=seed)
    return {"layout": layout_info["layout"], "class_names": class_names, "data_by_split": data_by_split}


def cmd_inspect_dataset(args: argparse.Namespace) -> int:
    data_root = args.data_root
    print(f"Inspecting dataset at: {data_root}")
    try:
        report = validate_dataset(data_root, require_splits=False)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    report.pprint()
    output_dirs = ensure_output_dirs(args.output_dir)
    json_path = os.path.join(output_dirs["results"], "dataset_inspection.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report.to_dict(), fp, indent=2)
    print(f"Dataset report saved to: {json_path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    return cmd_full_experiment(args, evaluate_only=False)


def cmd_full_experiment(args: argparse.Namespace, evaluate_only: bool = False) -> int:
    set_seed(args.seed)
    logger = get_logger()
    overrides = _apply_cli_overrides(args)
    runtime = build_runtime_config(overrides)
    data_root = runtime["DATA_ROOT"]
    output_dir = runtime["OUTPUT_DIR"]
    output_dirs = ensure_output_dirs(output_dir)
    logger.info(f"Dataset root: {data_root}")
    logger.info(f"Output directory: {output_dir}")
    try:
        validate_dataset(data_root, require_splits=True)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        logger.error(str(e))
        return 1
    discovered = _discover_and_build_samples(data_root, int(runtime["RANDOM_SEED"]))
    discovered_classes = discovered["class_names"]
    data_by_split = discovered["data_by_split"]
    try:
        validate_cil_stages_against_dataset(
            discovered_classes, CIL_STAGES, DEFAULT_TRASHCAN_INSTANCE_CLASSES
        )
    except ValueError as e:
        logger.error("Dataset does not match the locked TrashCan CIL protocol: %s", e)
        return 1
    else:
        stages = [list(s) for s in CIL_STAGES]
    class_order = [class_name for stage in stages for class_name in stage]
    validate_no_split_leak(
        data_by_split["train"], data_by_split["val"], data_by_split["test"]
    )
    trainer = CILTrainer(
        dataset_root=data_root,
        class_names=class_order,
        stages=stages,
        config=runtime,
        device=runtime.get("DEVICE"),
        output_dir=output_dir,
    )
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
    save_experiment_config(
        os.path.join(output_dirs["results"], "experiment_config.json"),
        trainer.config,
        class_order,
    )
    save_class_order(os.path.join(output_dirs["root"], "class_order.json"), class_order)
    train_transform = build_default_transforms(image_size=int(runtime.get("IMAGE_SIZE", 224)))
    logger.info("Distribution: %s", class_distribution_from_samples(data_by_split))
    eval_transform = build_eval_transforms(image_size=int(runtime.get("IMAGE_SIZE", 224)))
    result = trainer.run_full_pipeline(
        data_by_split,
        transform=train_transform,
        eval_transform=eval_transform,
        evaluate_test=not args.validation_only,
    )
    summary = {
        "stages": [],
    }
    summary_results = (
        result.get("validation_cil_summary", [])
        if args.validation_only
        else result.get("cil_summary", [])
    )
    for r in summary_results:
        summary["stages"].append(
            {
                "stage": r["stage_index"] + 1,
                "overall_accuracy": r["overall_accuracy"],
                "old_class_accuracy": r["old_class_accuracy"],
                "new_class_accuracy": r["new_class_accuracy"],
                "average_incremental_accuracy": r["average_incremental_accuracy"],
                "mean_forgetting": r["mean_forgetting"],
                "mean_retention": r["mean_retention"],
            }
        )
    summary_name = "validation_summary.json" if args.validation_only else "final_summary.json"
    summary_path = os.path.join(output_dirs["results"], summary_name)
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    logger.info("Training complete. Final summary: %s", summary)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    overrides = _apply_cli_overrides(args)
    runtime = build_runtime_config(overrides)
    runtime["USE_PRETRAINED"] = False
    data_root = runtime["DATA_ROOT"]
    output_dir = runtime["OUTPUT_DIR"]
    ensure_output_dirs(output_dir)
    ckpt_path = args.checkpoint
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("ERROR: --checkpoint is required and must exist for evaluation.", file=sys.stderr)
        return 1
    discovered = _discover_and_build_samples(data_root, int(runtime["RANDOM_SEED"]))
    class_names = discovered["class_names"]
    stages = [list(s) for s in CIL_STAGES]
    trainer = CILTrainer(
        dataset_root=data_root,
        class_names=class_names,
        stages=stages,
        config=runtime,
        device=runtime.get("DEVICE"),
        output_dir=output_dir,
    )
    loaded = trainer.load_checkpoint(ckpt_path)
    stage_idx = int(loaded.get("stage", len(stages) - 1))
    if args.stage is not None:
        stage_idx = args.stage - 1
    seen_classes = get_classes_up_to_stage(stages, min(stage_idx, len(stages) - 1))
    seen_class_to_idx = {c: i for i, c in enumerate(seen_classes)}
    test_samples = [
        s for s in discovered["data_by_split"].get("test", []) if s.class_name in seen_classes
    ]
    eval_transform = build_eval_transforms(image_size=int(runtime.get("IMAGE_SIZE", 224)))
    metrics = trainer.evaluate(
        test_samples, seen_class_to_idx, transform=eval_transform
    )
    metrics.pop("samples", None)
    metrics.pop("predictions", None)
    metrics.pop("targets", None)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


def cmd_gradcam(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    overrides = _apply_cli_overrides(args)
    runtime = build_runtime_config(overrides)
    runtime["USE_PRETRAINED"] = False
    data_root = runtime["DATA_ROOT"]
    output_dir = runtime["OUTPUT_DIR"]
    output_dirs = ensure_output_dirs(output_dir)
    ckpt_path = args.checkpoint
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("ERROR: --checkpoint is required and must exist for Grad-CAM.", file=sys.stderr)
        return 1
    discovered = _discover_and_build_samples(data_root, int(runtime["RANDOM_SEED"]))
    class_names = discovered["class_names"]
    stages = [list(s) for s in CIL_STAGES]
    trainer = CILTrainer(
        dataset_root=data_root,
        class_names=class_names,
        stages=stages,
        config=runtime,
        device=runtime.get("DEVICE"),
        output_dir=output_dir,
    )
    trainer.load_checkpoint(ckpt_path)
    test_samples = discovered["data_by_split"].get("test", [])
    eval_transform = build_eval_transforms(image_size=int(runtime.get("IMAGE_SIZE", 224)))
    eval_transform = build_eval_transforms(image_size=int(runtime.get("IMAGE_SIZE", 224)))
    last_stage = len(stages) - 1
    seen_classes = get_classes_up_to_stage(stages, last_stage)
    seen_class_to_idx = {c: i for i, c in enumerate(seen_classes)}
    result = generate_gradcam_for_dataset(
        trainer.model,
        [s for s in test_samples if s.class_name in seen_classes],
        seen_class_to_idx,
        output_dir=output_dirs["gradcam"],
        transform=eval_transform,
        samples_per_class=args.samples_per_class,
        device=trainer.device,
        stage_index=last_stage,
    )
    print(json.dumps({"gradcam_count": result["count"], "manifest": result["manifest"]}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.inspect_dataset:
        return cmd_inspect_dataset(args)
    if args.evaluate:
        return cmd_evaluate(args)
    if args.gradcam:
        return cmd_gradcam(args)
    if args.train or args.full_experiment:
        return cmd_train(args)
    if args.resume:
        args.train = True
        return cmd_train(args)
    print("No action specified. Use one of:\n"
          "  --inspect-dataset\n"
          "  --train / --full-experiment\n"
          "  --evaluate --checkpoint <path>\n"
          "  --gradcam --checkpoint <path>\n"
          "Use --help for all options.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
