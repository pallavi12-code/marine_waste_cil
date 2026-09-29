import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

USE_PRETRAINED = True

DEFAULT_TRASHCAN_INSTANCE_CLASSES = [
    "trash_plastic",
    "trash_metal",
    "trash_wood",
    "trash_etc",
    "animal_fish",
    "rov",
]

CIL_STAGES = [
    ["trash_plastic", "trash_metal"],
    ["trash_wood", "trash_etc"],
    ["animal_fish", "rov"],
]

STAGE_CLASS_COUNTS = [len(stage) for stage in CIL_STAGES]
CUMULATIVE_CLASS_COUNTS = [sum(STAGE_CLASS_COUNTS[: i + 1]) for i in range(len(STAGE_CLASS_COUNTS))]

DATA_ROOT = os.getenv(
    "MARINE_CIL_DATA_ROOT",
    os.path.join(os.getcwd(), "data", "TrashCan_processed_seed42"),
)
OUTPUT_DIR = os.path.join(os.getcwd(), "artifacts", "real_trashcan_experiment")
BATCH_SIZE = 32
NUM_EPOCHS_PER_STAGE = 3
LEARNING_RATE = 1e-4
REPLAY_MEMORY_SIZE = 400
MEMORY_BUDGET_PER_CLASS = 20
DISTILLATION_TEMPERATURE = 2.0
DISTILLATION_WEIGHT = 1.0
CLASSIFICATION_WEIGHT = 1.0
RANDOM_SEED = 42
IMAGE_SIZE = 224
NUM_WORKERS = 0
BALANCED_SAMPLING = False
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
    else "cpu"
)
GRADCAM_SAMPLES_PER_CLASS = 3


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "marine_waste_cil", log_dir: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_output_dirs(output_dir: str = OUTPUT_DIR) -> Dict[str, str]:
    base = Path(output_dir)
    dirs = {
        "root": str(base),
        "checkpoints": str(base / "checkpoints"),
        "results": str(base / "results"),
        "confusion_matrices": str(base / "results"),
        "gradcam": str(base / "gradcam"),
        "logs": str(base / "logs"),
        "plots": str(base / "results"),
    }
    for d in dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    return dirs


def validate_cil_stages_against_dataset(
    discovered_classes: Sequence[str],
    stages: Sequence[Sequence[str]],
    configured_class_list: Optional[Sequence[str]] = None,
) -> None:
    discovered = list(dict.fromkeys(discovered_classes))
    if not discovered:
        raise ValueError("No classes were discovered under the provided dataset path.")

    if configured_class_list is not None:
        configured = list(dict.fromkeys(configured_class_list))
        if set(configured) != set(discovered):
            missing_in_configured = sorted(set(discovered) - set(configured))
            missing_in_discovered = sorted(set(configured) - set(discovered))
            msg_parts = ["Configured class list does not match discovered dataset classes."]
            if missing_in_configured:
                msg_parts.append(f"Discovered but not configured: {missing_in_configured}")
            if missing_in_discovered:
                msg_parts.append(f"Configured but not discovered: {missing_in_discovered}")
            raise ValueError(" ".join(msg_parts))

    flat = [name for stage in stages for name in stage]
    if len(flat) != len(set(flat)):
        duplicates = [n for n in set(flat) if flat.count(n) > 1]
        raise ValueError(f"Duplicate class names are present in CIL_STAGES: {duplicates}")

    missing = sorted({name for name in flat if name not in discovered})
    if missing:
        raise ValueError(
            "CIL_STAGES contains classes not found in the dataset: "
            f"missing={missing}. Discovered classes={discovered}"
        )

    seen: set[str] = set()
    for stage_idx, stage in enumerate(stages):
        stage_set = set(stage)
        overlaps = seen & stage_set
        if overlaps:
            raise ValueError(
                f"Class appears in multiple stages: stage {stage_idx} overlaps previous stages on {sorted(overlaps)}"
            )
        seen |= stage_set

    uncovered = sorted(set(discovered) - set(flat))
    if uncovered:
        raise ValueError(
            "Some discovered classes are not assigned to any stage. "
            f"Unassigned={uncovered}."
        )


def build_runtime_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "DATA_ROOT": os.getenv("MARINE_CIL_DATA_ROOT", DATA_ROOT),
        "OUTPUT_DIR": OUTPUT_DIR,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS_PER_STAGE": NUM_EPOCHS_PER_STAGE,
        "LEARNING_RATE": LEARNING_RATE,
        "REPLAY_MEMORY_SIZE": REPLAY_MEMORY_SIZE,
        "MEMORY_BUDGET_PER_CLASS": MEMORY_BUDGET_PER_CLASS,
        "DISTILLATION_TEMPERATURE": DISTILLATION_TEMPERATURE,
        "DISTILLATION_WEIGHT": DISTILLATION_WEIGHT,
        "CLASSIFICATION_WEIGHT": CLASSIFICATION_WEIGHT,
        "RANDOM_SEED": RANDOM_SEED,
        "IMAGE_SIZE": IMAGE_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "BALANCED_SAMPLING": BALANCED_SAMPLING,
        "DEVICE": DEVICE,
        "USE_PRETRAINED": USE_PRETRAINED,
        "CIL_STAGES": [list(stage) for stage in CIL_STAGES],
        "MODEL_NAME": "resnet18",
        "GRADCAM_SAMPLES_PER_CLASS": GRADCAM_SAMPLES_PER_CLASS,
        "PYTHON_VERSION": sys.version,
        "PYTORCH_VERSION": torch.__version__,
        "TORCHVISION_VERSION": _get_torchvision_version(),
        "CUDA_AVAILABLE": torch.cuda.is_available(),
        "MPS_AVAILABLE": bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ),
        "GPU_NAME": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    return cfg


def _get_torchvision_version() -> str:
    try:
        import torchvision
        return torchvision.__version__
    except Exception:
        return "unknown"


def save_experiment_config(path: str, cfg: Dict[str, Any], class_order: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(cfg)
    payload["CLASS_ORDER"] = list(class_order)
    payload["CUMULATIVE_CLASS_COUNTS"] = CUMULATIVE_CLASS_COUNTS
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=str)


def load_class_order(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        return [str(x) for x in data.get("class_order", data.get("CLASS_ORDER", []))]
    raise ValueError(f"Unknown class_order file format in {path}")


def save_class_order(path: str, class_order: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(list(class_order), fp, indent=2)


def resolve_stage_class_order(class_names: Sequence[str]) -> List[str]:
    ordered = [name for name in class_names if name]
    if not ordered:
        raise ValueError("No class names were supplied to stage ordering.")
    return ordered


def get_classes_up_to_stage(stages: Sequence[Sequence[str]], stage_index: int) -> List[str]:
    result: List[str] = []
    for i in range(min(stage_index + 1, len(stages))):
        result.extend(list(stages[i]))
    return result


def get_new_classes_for_stage(stages: Sequence[Sequence[str]], stage_index: int) -> List[str]:
    return list(stages[stage_index])


def get_old_classes_for_stage(stages: Sequence[Sequence[str]], stage_index: int) -> List[str]:
    result: List[str] = []
    for i in range(stage_index):
        result.extend(list(stages[i]))
    return result
