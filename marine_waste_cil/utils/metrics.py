from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(y_true, y_pred, labels) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    label_list = list(labels)
    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=label_list, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=label_list, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=label_list, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=label_list)
    per_class_accuracy: List[float] = []
    for idx, lbl in enumerate(label_list):
        mask = y_true_arr == lbl
        if mask.any():
            correct = (y_pred_arr[mask] == lbl).sum()
            per_class_accuracy.append(float(correct) / float(mask.sum()))
        else:
            per_class_accuracy.append(0.0)
    report: Dict[str, Any] = {
        "overall_accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": [],
        "confusion_matrix": cm.tolist(),
    }
    for idx, label in enumerate(label_list):
        report["per_class"].append(
            {
                "class_idx": int(label),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "accuracy": float(per_class_accuracy[idx]),
            }
        )
    return report


def compute_cil_metrics(
    stage_metrics_history: Sequence[Dict[str, Any]],
    stage_class_splits: Sequence[Dict[str, Sequence[int]]],
    class_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    stage_metrics_history[i] contains per-class metrics (with class index as label integer)
    after stage i.
    stage_class_splits[i] = {"old": [...], "new": [...]} class indices for that stage.
    """
    results: List[Dict[str, Any]] = []
    peak_accuracy_per_class: Dict[int, float] = {}
    previous_stage_accuracy: Dict[int, float] = {}
    for stage_idx, stage_metrics in enumerate(stage_metrics_history):
        per_class = stage_metrics.get("per_class", [])
        current_accuracy: Dict[int, float] = {}
        for entry in per_class:
            class_idx = int(entry["class_idx"])
            current_accuracy[class_idx] = float(entry.get("accuracy", 0.0))
        for c, a in current_accuracy.items():
            if c not in peak_accuracy_per_class or a > peak_accuracy_per_class[c]:
                peak_accuracy_per_class[c] = a
        splits = stage_class_splits[stage_idx]
        old_classes = [int(c) for c in splits.get("old", [])]
        new_classes = [int(c) for c in splits.get("new", [])]
        all_classes = sorted(set(old_classes) | set(new_classes) | set(current_accuracy.keys()))
        overall_acc = float(stage_metrics.get("overall_accuracy", 0.0))
        old_accs = [current_accuracy.get(c, 0.0) for c in old_classes]
        new_accs = [current_accuracy.get(c, 0.0) for c in new_classes]
        old_class_accuracy = float(np.mean(old_accs)) if old_accs else 0.0
        new_class_accuracy = float(np.mean(new_accs)) if new_accs else 0.0
        all_accs = [current_accuracy.get(c, 0.0) for c in all_classes]
        avg_inc_accuracy = float(np.mean(all_accs)) if all_accs else 0.0
        forgetting_per_class: Dict[int, float] = {}
        retention_per_class: Dict[int, float] = {}
        for c in old_classes:
            prev = previous_stage_accuracy.get(c, peak_accuracy_per_class.get(c, 0.0))
            cur = current_accuracy.get(c, 0.0)
            forgetting_per_class[c] = max(0.0, prev - cur)
            if prev > 0:
                retention_per_class[c] = cur / prev
            else:
                retention_per_class[c] = 1.0 if cur >= 0 else 0.0
        forgetting_values = list(forgetting_per_class.values())
        mean_forgetting = float(np.mean(forgetting_values)) if forgetting_values else 0.0
        max_forgetting = float(np.max(forgetting_values)) if forgetting_values else 0.0
        retention_values = list(retention_per_class.values())
        mean_retention = float(np.mean(retention_values)) if retention_values else 1.0
        if class_names is not None:
            def name_of(c: int) -> str:
                return class_names[c] if 0 <= c < len(class_names) else f"class_{c}"
            forgetting_named = {name_of(c): float(v) for c, v in forgetting_per_class.items()}
            retention_named = {name_of(c): float(v) for c, v in retention_per_class.items()}
        else:
            forgetting_named = {str(c): float(v) for c, v in forgetting_per_class.items()}
            retention_named = {str(c): float(v) for c, v in retention_per_class.items()}
        results.append(
            {
                "stage_index": stage_idx,
                "overall_accuracy": overall_acc,
                "old_class_accuracy": old_class_accuracy,
                "new_class_accuracy": new_class_accuracy,
                "average_incremental_accuracy": avg_inc_accuracy,
                "mean_forgetting": mean_forgetting,
                "max_forgetting": max_forgetting,
                "mean_retention": mean_retention,
                "per_class_forgetting": forgetting_named,
                "per_class_retention": retention_named,
                "old_classes": [int(c) for c in old_classes],
                "new_classes": [int(c) for c in new_classes],
            }
        )
        previous_stage_accuracy = dict(current_accuracy)
    return results


def labels_to_indices(labels: Sequence[str], class_to_index: Dict[str, int]) -> np.ndarray:
    return np.asarray([class_to_index[name] for name in labels], dtype=int)


def flatten_stage_metrics(stage_metrics: Iterable[dict]) -> List[dict]:
    return [item for item in stage_metrics]


def save_metrics_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=str)


def save_metrics_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_per_class_metrics_csv(path: str, per_class_entries: Sequence[Dict[str, Any]], class_names: Optional[Sequence[str]] = None) -> None:
    if not per_class_entries:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for e in per_class_entries:
        row = dict(e)
        if class_names is not None:
            idx = int(row.get("class_idx", -1))
            if 0 <= idx < len(class_names):
                row["class_name"] = class_names[idx]
        rows.append(row)
    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_confusion_matrix_numeric(path_base: str, cm: np.ndarray, class_names: Optional[Sequence[str]] = None) -> None:
    base = Path(path_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(base) + ".npy", cm)
    with open(str(base) + ".csv", "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        header = ["class"]
        if class_names:
            header.extend(list(class_names))
        else:
            header.extend([f"pred_{i}" for i in range(cm.shape[1])])
        writer.writerow(header)
        for i in range(cm.shape[0]):
            row_name = class_names[i] if class_names else f"true_{i}"
            writer.writerow([row_name] + [int(x) for x in cm[i].tolist()])
