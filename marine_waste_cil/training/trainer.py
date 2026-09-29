from __future__ import annotations

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from marine_waste_cil.config import (
    build_runtime_config,
    ensure_output_dirs,
    get_classes_up_to_stage,
    get_logger,
    get_new_classes_for_stage,
    get_old_classes_for_stage,
    save_class_order,
    save_experiment_config,
    set_seed,
)
from marine_waste_cil.data.dataset import (
    MarineWasteDataset,
    SampleRecord,
    validate_no_split_leak,
)
from marine_waste_cil.memory.replay_buffer import (
    ReplayBuffer,
    select_herding_exemplars_iterative,
)
from marine_waste_cil.models.backbone import ResNet18ForCIL
from marine_waste_cil.training.distillation import (
    CILTotalLoss,
    clone_as_teacher,
    freeze_model,
    teacher_params_frozen_check,
)
from marine_waste_cil.utils.metrics import (
    compute_cil_metrics,
    compute_classification_metrics,
    save_confusion_matrix_numeric,
    save_metrics_csv,
    save_metrics_json,
    save_per_class_metrics_csv,
)
from marine_waste_cil.utils.plotting import (
    save_accuracy_curve,
    save_confusion_matrix_heatmap,
    save_forgetting_barplot,
    save_retention_barplot,
)


class CILTrainer:
    def __init__(
        self,
        dataset_root: str,
        class_names: Sequence[str],
        stages: Sequence[Sequence[str]],
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.config = build_runtime_config(config or {})
        if output_dir:
            self.config["OUTPUT_DIR"] = output_dir
        self.dataset_root = dataset_root
        self.class_names = list(class_names)
        self.stages = [list(stage) for stage in stages]
        staged_classes = [name for stage in self.stages for name in stage]
        if not self.stages or not staged_classes:
            raise ValueError("At least one non-empty CIL stage is required.")
        if len(staged_classes) != len(set(staged_classes)):
            duplicates = [n for n in set(staged_classes) if staged_classes.count(n) > 1]
            raise ValueError(f"CIL stages contain duplicate class names: {duplicates}")
        if set(staged_classes) != set(self.class_names):
            raise ValueError(
                f"CIL stages and class_names must contain the same classes. "
                f"staged={set(staged_classes)}, names={set(self.class_names)}"
            )
        self.class_names = staged_classes
        self.device = torch.device(device or self.config["DEVICE"])
        self.output_dirs = ensure_output_dirs(self.config["OUTPUT_DIR"])
        self.logger = get_logger(log_dir=self.output_dirs["logs"])
        seed = int(self.config.get("RANDOM_SEED", 42))
        set_seed(seed)
        self.replay_buffer = ReplayBuffer(
            budget_per_class=int(self.config["MEMORY_BUDGET_PER_CLASS"]),
            total_budget=self.config.get("REPLAY_MEMORY_SIZE"),
        )
        first_stage_count = len(self.stages[0])
        pretrained = bool(self.config.get("USE_PRETRAINED", True))
        self.model = ResNet18ForCIL(num_classes=first_stage_count, pretrained=pretrained)
        self.model.to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=float(self.config["LEARNING_RATE"])
        )
        self.criterion = CILTotalLoss(
            classification_weight=float(self.config.get("CLASSIFICATION_WEIGHT", 1.0)),
            distillation_weight=float(self.config["DISTILLATION_WEIGHT"]),
            temperature=float(self.config["DISTILLATION_TEMPERATURE"]),
        )
        self.stage_history: List[Dict[str, Any]] = []
        self.stage_metrics_history: List[Dict[str, Any]] = []
        self.stage_class_splits: List[Dict[str, List[int]]] = []
        self.class_to_index: Dict[str, int] = {c: i for i, c in enumerate(self.class_names)}
        self.start_stage = 0

    def _make_dataloader(
        self,
        sample_list: Sequence[SampleRecord],
        transform=None,
        shuffle: bool = True,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        bs = batch_size or int(self.config["BATCH_SIZE"])
        num_workers = int(self.config.get("NUM_WORKERS", 0))
        pin = str(self.device) == "cuda"
        dataset = MarineWasteDataset(sample_list, transform=transform)
        sampler = None
        use_balanced_sampling = bool(self.config.get("BALANCED_SAMPLING", False))
        if shuffle and sample_list and use_balanced_sampling:
            class_counts: Dict[str, int] = {}
            for sample in sample_list:
                class_counts[sample.class_name] = class_counts.get(sample.class_name, 0) + 1
            sample_weights = [
                1.0 / class_counts[sample.class_name] for sample in sample_list
            ]
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_list), replacement=True
            )
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin,
        )

    def save_checkpoint(
        self,
        path: str,
        stage_index: int,
        class_to_index: Dict[str, int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": int(self.config["NUM_EPOCHS_PER_STAGE"]),
            "stage": stage_index,
            "class_names": list(self.class_names),
            "class_to_index": dict(class_to_index),
            "class_order": list(self.class_names),
            "config": dict(self.config),
            "metrics": dict(self.stage_metrics_history[-1]) if self.stage_metrics_history else {},
            "replay_memory": self.replay_buffer.as_dict(),
            "stage_history": list(self.stage_history),
            "stage_metrics_history": list(self.stage_metrics_history),
            "stage_class_splits": list(self.stage_class_splits),
            "start_stage": stage_index + 1,
            "extra": extra or {},
        }
        torch.save(payload, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        self.logger.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        if "class_order" in ckpt:
            self.class_names = list(ckpt["class_order"])
        elif "class_names" in ckpt:
            self.class_names = list(ckpt["class_names"])
        self.class_to_index = {c: i for i, c in enumerate(self.class_names)}
        state_dict = ckpt["model_state_dict"]
        ckpt_num_classes = int(state_dict.get("classifier.weight", torch.empty(self.model.num_classes, 512)).shape[0])
        if ckpt.get("class_to_index"):
            ckpt_num_classes = max(ckpt_num_classes, len(ckpt["class_to_index"]))
        if "stage" in ckpt and isinstance(self.stages, (list, tuple)):
            stage_idx = int(ckpt["stage"])
            if stage_idx < len(self.stages):
                seen_classes = get_classes_up_to_stage(self.stages, stage_idx)
                ckpt_num_classes = max(ckpt_num_classes, len(seen_classes))
        if ckpt_num_classes != self.model.num_classes:
            if ckpt_num_classes > self.model.num_classes:
                self.model.expand_head(ckpt_num_classes)
            else:
                self.model.classifier = nn.Linear(512, ckpt_num_classes).to(self.device)
                self.model.num_classes = ckpt_num_classes
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                self.logger.warning(f"Could not restore optimizer state: {e}")
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=float(self.config["LEARNING_RATE"])
                )
        if "replay_memory" in ckpt:
            rm = ckpt["replay_memory"]
            for cls, keys in rm.items():
                self.replay_buffer._store[cls] = list(keys)
        if "stage_metrics_history" in ckpt:
            self.stage_metrics_history = list(ckpt["stage_metrics_history"])
        if "stage_history" in ckpt:
            self.stage_history = list(ckpt["stage_history"])
        if "stage_class_splits" in ckpt:
            self.stage_class_splits = list(ckpt["stage_class_splits"])
        self.start_stage = int(ckpt.get("start_stage", 0))
        self.logger.info(f"Checkpoint loaded. Resume from stage {self.start_stage}")
        return ckpt

    def expand_classifier_for_stage(self, stage_index: int) -> None:
        current_class_count = self.model.num_classes
        seen_classes = get_classes_up_to_stage(self.stages, stage_index)
        target_class_count = len(seen_classes)
        if target_class_count > current_class_count:
            old_weights = self.model.get_classifier_weights()
            self.model.expand_head(target_class_count)
            new_weights = self.model.get_classifier_weights()
            assert torch.allclose(
                new_weights[:old_weights.shape[0]], old_weights, atol=1e-6
            ), "Old classifier weights were not preserved during expansion"
            self.logger.info(
                f"Expanded classifier: {current_class_count} -> {target_class_count}"
            )
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=float(self.config["LEARNING_RATE"])
            )

    def _extract_features_for_herding(
        self, samples: Sequence[SampleRecord], transform=None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        self.model.eval()
        loader = self._make_dataloader(samples, transform=transform, shuffle=False, batch_size=min(32, len(samples) if samples else 1))
        all_features: List[np.ndarray] = []
        all_labels: List[str] = []
        all_paths: List[str] = []
        with torch.no_grad():
            for batch in loader:
                images, targets, paths = batch
                images = images.to(self.device)
                feats = self.model.extract_features(images)
                all_features.append(feats.detach().cpu().numpy())
                all_labels.extend(list(targets))
                all_paths.extend(list(paths))
        if not all_features:
            return np.zeros((0, 512), dtype=float), [], []
        return np.concatenate(all_features, axis=0), all_labels, all_paths

    def select_exemplars_for_stage(
        self, stage_index: int, train_samples: Sequence[SampleRecord], transform=None
    ) -> Dict[str, List[str]]:
        stage_classes = get_new_classes_for_stage(self.stages, stage_index)
        budget = int(self.config["MEMORY_BUDGET_PER_CLASS"])
        results: Dict[str, List[str]] = {}
        if not stage_classes:
            return results
        new_class_samples = [
            sample for sample in train_samples if sample.class_name in stage_classes
        ]
        features, labels, paths = self._extract_features_for_herding(
            new_class_samples, transform=transform
        )
        for cls in stage_classes:
            selected = select_herding_exemplars_iterative(
                features, labels, cls, paths, budget
            )
            results[cls] = selected
            self.replay_buffer.update_from_herding(cls, selected, budget=budget)
        return results

    def _reduce_old_exemplars(self, stage_index: int) -> None:
        seen_classes = get_classes_up_to_stage(self.stages, stage_index)
        if not seen_classes:
            return
        budget_per_class = max(1, int(self.config["MEMORY_BUDGET_PER_CLASS"]))
        total_budget = self.config.get("REPLAY_MEMORY_SIZE")
        if total_budget is not None:
            per_class_allowance = max(1, total_budget // len(seen_classes))
            budget_per_class = min(budget_per_class, per_class_allowance)
        for cls in seen_classes:
            self.replay_buffer.reduce_class_budget(cls, budget_per_class)

    def _build_stage_training_samples(
        self,
        stage_index: int,
        data_by_split: Dict[str, List[SampleRecord]],
    ) -> Tuple[List[SampleRecord], Dict[str, SampleRecord]]:
        new_classes = get_new_classes_for_stage(self.stages, stage_index)
        seen_classes = get_classes_up_to_stage(self.stages, stage_index)
        train_all = data_by_split.get("train", [])
        new_samples = [s for s in train_all if s.class_name in new_classes]
        replay_path_to_sample: Dict[str, SampleRecord] = {}
        for s in train_all:
            if s.class_name in seen_classes:
                replay_path_to_sample[os.path.abspath(s.image_path)] = s
                replay_path_to_sample[s.image_path] = s
        replay_samples: List[SampleRecord] = []
        for _, key in self.replay_buffer.get_all_samples():
            abs_key = os.path.abspath(key)
            match = replay_path_to_sample.get(key) or replay_path_to_sample.get(abs_key)
            if match is not None:
                replay_samples.append(match)
        combined = list(new_samples) + list(replay_samples)
        return combined, replay_path_to_sample

    def train_stage(
        self,
        stage_index: int,
        data_by_split: Dict[str, List[SampleRecord]],
        transform=None,
    ) -> Dict[str, Any]:
        logger = self.logger
        new_classes = get_new_classes_for_stage(self.stages, stage_index)
        old_classes = get_old_classes_for_stage(self.stages, stage_index)
        seen_classes = get_classes_up_to_stage(self.stages, stage_index)
        seen_class_to_index = {c: i for i, c in enumerate(seen_classes)}
        old_class_count = len(old_classes)
        stage_samples, _ = self._build_stage_training_samples(stage_index, data_by_split)
        train_split = [s for s in data_by_split.get("train", []) if s.class_name in seen_classes]
        val_split = [s for s in data_by_split.get("val", []) if s.class_name in seen_classes]
        validate_no_split_leak(train_split, val_split, [])
        self.expand_classifier_for_stage(stage_index)
        teacher: Optional[nn.Module] = None
        if stage_index > 0 and old_class_count > 0:
            teacher = clone_as_teacher(self.model, device=self.device)
            assert teacher_params_frozen_check(teacher), "Teacher model parameters must be frozen"
            self.expand_classifier_for_stage(stage_index)
        self.model.train()
        num_epochs = int(self.config["NUM_EPOCHS_PER_STAGE"])
        batch_size = int(self.config["BATCH_SIZE"])
        replay_sample_count = sum(
            1 for s in stage_samples if s.class_name not in new_classes
        )
        logger.info(
            f"Stage {stage_index + 1}: "
            f"classes={len(seen_classes)} (new={len(new_classes)}, old={len(old_classes)}), "
            f"train samples={len(stage_samples)} (replay={replay_sample_count}), "
            f"val={len(val_split)}, epochs={num_epochs}"
        )
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_loss_ce = 0.0
            epoch_loss_kd = 0.0
            epoch_total = 0.0
            num_batches = 0
            correct = 0
            total_samples = 0
            dataloader = self._make_dataloader(
                stage_samples, transform=transform, shuffle=True, batch_size=batch_size
            )
            pbar = tqdm(
                dataloader,
                desc=f"Stage {stage_index + 1} Epoch {epoch + 1}/{num_epochs}",
                leave=False,
                disable=None,
            )
            for batch in pbar:
                images, targets, paths = batch
                images = images.to(self.device)
                try:
                    y = torch.tensor(
                        [seen_class_to_index[t] for t in targets],
                        device=self.device,
                        dtype=torch.long,
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"Training sample has class outside the seen stages: {exc.args[0]}"
                    ) from exc
                self.optimizer.zero_grad()
                logits = self.model(images)
                teacher_logits = None
                if teacher is not None and old_class_count > 0:
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                loss, stats = self.criterion(
                    logits,
                    y,
                    teacher_logits=teacher_logits,
                    old_class_count=old_class_count,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss_ce += stats["ce"]
                epoch_loss_kd += stats["kd"]
                epoch_total += stats["total"]
                num_batches += 1
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total_samples += int(y.shape[0])
                pbar.set_postfix(
                    loss=f"{stats['total']:.3f}",
                    ce=f"{stats['ce']:.3f}",
                    kd=f"{stats['kd']:.3f}",
                    acc=f"{correct / max(1, total_samples):.3f}",
                )
            pbar.close()
            if num_batches > 0:
                avg_ce = epoch_loss_ce / num_batches
                avg_kd = epoch_loss_kd / num_batches
                avg_total = epoch_total / num_batches
                epoch_acc = correct / max(1, total_samples)
                logger.info(
                    f"  Epoch {epoch + 1}/{num_epochs}: "
                    f"loss={avg_total:.4f}, ce={avg_ce:.4f}, kd={avg_kd:.4f}, train_acc={epoch_acc:.4f}"
                )
        self.model.eval()
        if stage_index == 0:
            pass
        if old_class_count > 0:
            self._reduce_old_exemplars(stage_index)
        exemplars = self.select_exemplars_for_stage(stage_index, train_split, transform=transform)
        self.logger.info(f"  Selected exemplars for stage: { {k: len(v) for k, v in exemplars.items()} }")
        old_indices = [seen_class_to_index[c] for c in old_classes]
        new_indices = [seen_class_to_index[c] for c in new_classes]
        self.stage_class_splits.append({"old": old_indices, "new": new_indices})
        train_acc = correct / max(1, total_samples) if total_samples else 0.0
        elapsed = time.time() - start_time
        stage_record = {
            "stage_index": stage_index,
            "seen_classes": list(seen_classes),
            "new_classes": list(new_classes),
            "old_classes": list(old_classes),
            "class_to_index": dict(seen_class_to_index),
            "num_train_samples": len(stage_samples),
            "num_val_samples": len(val_split),
            "replay_samples_count": replay_sample_count,
            "num_epochs": num_epochs,
            "train_accuracy_last_epoch": float(train_acc),
            "elapsed_seconds": float(elapsed),
            "exemplar_counts": {k: len(v) for k, v in exemplars.items()},
        }
        self.stage_history.append(stage_record)
        return stage_record

    def evaluate(
        self,
        samples: Sequence[SampleRecord],
        class_to_index: Dict[str, int],
        transform=None,
        include_labels: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not samples:
            return {
                "overall_accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "weighted_precision": 0.0,
                "weighted_recall": 0.0,
                "weighted_f1": 0.0,
                "per_class": [],
                "confusion_matrix": [],
                "labels": [],
            }
        self.model.eval()
        all_targets: List[int] = []
        all_preds: List[int] = []
        all_paths: List[str] = []
        labels_list = list(class_to_index.keys())
        if include_labels is not None:
            labels_list = list(include_labels)
        label_indices = [class_to_index[label] for label in labels_list]
        dataloader = self._make_dataloader(
            samples, transform=transform, shuffle=False, batch_size=int(self.config["BATCH_SIZE"])
        )
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False, disable=None):
                images, targets, paths = batch
                images = images.to(self.device)
                logits = self.model(images)
                preds = logits.argmax(dim=1).cpu().numpy()
                unknown = [target for target in targets if target not in class_to_index]
                if unknown:
                    raise ValueError(
                        f"Evaluation samples contain unknown classes: {sorted(set(unknown))}"
                    )
                target_indices = np.array(
                    [class_to_index[target] for target in targets], dtype=int
                )
                all_targets.extend(target_indices.tolist())
                all_preds.extend(preds.tolist())
                all_paths.extend(list(paths))
        metrics = compute_classification_metrics(all_targets, all_preds, label_indices)
        metrics["labels"] = labels_list
        metrics["samples"] = list(all_paths)
        metrics["predictions"] = list(all_preds)
        metrics["targets"] = list(all_targets)
        return metrics

    def validate_configuration(self, discovered_classes: Sequence[str]) -> None:
        if not discovered_classes:
            raise ValueError("No dataset classes discovered; cannot train.")
        if set(self.class_names) != set(discovered_classes):
            raise ValueError(
                f"Class mismatch: configured {self.class_names}, discovered {list(discovered_classes)}"
            )

    def _save_evaluation_outputs(
        self,
        stage_index: int,
        metrics: Dict[str, Any],
        seen_classes: Sequence[str],
        prefix: str = "",
    ) -> None:
        results_dir = self.output_dirs["results"]
        stage_label = f"{prefix}stage_{stage_index + 1}"
        save_metrics_json(
            os.path.join(results_dir, f"{stage_label}_metrics.json"),
            metrics,
        )
        per_class_entries = metrics.get("per_class", [])
        if per_class_entries:
            save_per_class_metrics_csv(
                os.path.join(results_dir, f"{stage_label}_per_class_metrics.csv"),
                per_class_entries,
                class_names=list(seen_classes),
            )
        cm = np.asarray(metrics.get("confusion_matrix", []))
        if cm.size > 0:
            cm_path_base = os.path.join(results_dir, f"confusion_matrix_{stage_label}")
            save_confusion_matrix_numeric(cm_path_base, cm, class_names=list(seen_classes))
            save_confusion_matrix_heatmap(
                cm,
                labels=list(seen_classes),
                save_path=os.path.join(results_dir, f"confusion_matrix_{stage_label}.png"),
                title=f"Confusion Matrix - Stage {stage_index + 1} ({len(seen_classes)} classes)",
            )

    def _save_cil_outputs(
        self,
        cil_results: Sequence[Dict[str, Any]],
        prefix: str = "",
    ) -> None:
        results_dir = self.output_dirs["results"]
        save_metrics_json(
            os.path.join(results_dir, f"{prefix}cil_summary.json"),
            {"stages": list(cil_results)},
        )
        rows: List[Dict[str, Any]] = []
        for r in cil_results:
            rows.append(
                {
                    "stage_index": r["stage_index"],
                    "overall_accuracy": r["overall_accuracy"],
                    "old_class_accuracy": r["old_class_accuracy"],
                    "new_class_accuracy": r["new_class_accuracy"],
                    "average_incremental_accuracy": r["average_incremental_accuracy"],
                    "mean_forgetting": r["mean_forgetting"],
                    "max_forgetting": r["max_forgetting"],
                    "mean_retention": r["mean_retention"],
                }
            )
        save_metrics_csv(os.path.join(results_dir, f"{prefix}summary_by_stage.csv"), rows)
        stage_labels = [f"Stage {r['stage_index'] + 1}" for r in cil_results]
        if len(stage_labels) >= 1:
            save_accuracy_curve(
                stage_labels,
                [r["overall_accuracy"] for r in cil_results],
                os.path.join(results_dir, f"{prefix}accuracy_across_stages.png"),
                ylabel="Overall Accuracy",
                title="Overall Accuracy Across CIL Stages",
            )
            save_accuracy_curve(
                stage_labels,
                [r["old_class_accuracy"] for r in cil_results],
                os.path.join(results_dir, f"{prefix}old_class_accuracy_across_stages.png"),
                ylabel="Old Class Accuracy",
                title="Old Class Accuracy Across Stages",
            )
            save_accuracy_curve(
                stage_labels,
                [r["average_incremental_accuracy"] for r in cil_results],
                os.path.join(results_dir, f"{prefix}avg_incremental_accuracy_across_stages.png"),
                ylabel="Average Incremental Accuracy",
                title="Average Incremental Accuracy Across Stages",
            )
        last = cil_results[-1]
        forgetting_names = list(last.get("per_class_forgetting", {}).keys())
        forgetting_vals = [last["per_class_forgetting"].get(k, 0.0) for k in forgetting_names]
        if forgetting_names:
            save_forgetting_barplot(
                forgetting_names,
                forgetting_vals,
                os.path.join(results_dir, f"{prefix}forgetting_summary_last_stage.png"),
            )
        retention_names = list(last.get("per_class_retention", {}).keys())
        retention_vals = [last["per_class_retention"].get(k, 0.0) for k in retention_names]
        if retention_names:
            save_retention_barplot(
                retention_names,
                retention_vals,
                os.path.join(results_dir, f"{prefix}retention_summary_last_stage.png"),
            )

    def run_full_pipeline(
        self,
        data_by_split: Dict[str, List[SampleRecord]],
        transform=None,
        eval_transform=None,
        evaluate_test: bool = True,
    ) -> Dict[str, Any]:
        set_seed(int(self.config.get("RANDOM_SEED", 42)))
        save_experiment_config(
            os.path.join(self.output_dirs["results"], "experiment_config.json"),
            self.config,
            self.class_names,
        )
        save_class_order(
            os.path.join(self.output_dirs["root"], "class_order.json"),
            self.class_names,
        )
        num_stages = len(self.stages)
        for stage_index in range(self.start_stage, num_stages):
            self.logger.info(f"========== Stage {stage_index + 1} / {num_stages} ==========")
            seen_classes = get_classes_up_to_stage(self.stages, stage_index)
            seen_class_to_index = {c: i for i, c in enumerate(seen_classes)}
            validation_samples = [
                s
                for s in data_by_split.get("val", [])
                if s.class_name in seen_classes
            ]
            self.train_stage(stage_index, data_by_split, transform=transform)
            validation_metrics = self.evaluate(
                validation_samples,
                seen_class_to_index,
                transform=eval_transform or transform,
            )
            self.stage_metrics_history.append(validation_metrics)
            self.logger.info(
                f"Stage {stage_index + 1} validation: "
                f"overall_acc={validation_metrics['overall_accuracy']:.4f}, "
                f"macro_f1={validation_metrics['macro_f1']:.4f}, "
                f"weighted_f1={validation_metrics['weighted_f1']:.4f}"
            )
            self._save_evaluation_outputs(
                stage_index, validation_metrics, seen_classes, prefix="validation_"
            )
            ckpt_path = os.path.join(
                self.output_dirs["checkpoints"], f"stage_{stage_index + 1}.pt"
            )
            if evaluate_test:
                self.save_checkpoint(ckpt_path, stage_index, seen_class_to_index)

        validation_cil_results = compute_cil_metrics(
            self.stage_metrics_history,
            self.stage_class_splits,
            class_names=self.class_names,
        )
        self._save_cil_outputs(validation_cil_results, prefix="validation_")

        test_metrics_history: List[Dict[str, Any]] = []
        cil_results: List[Dict[str, Any]] = []
        if evaluate_test:
            final_model = self.model
            for stage_index in range(num_stages):
                ckpt_path = os.path.join(
                    self.output_dirs["checkpoints"], f"stage_{stage_index + 1}.pt"
                )
                if not os.path.isfile(ckpt_path):
                    raise FileNotFoundError(
                        f"Missing trained stage checkpoint for final test evaluation: {ckpt_path}"
                    )
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                seen_classes = get_classes_up_to_stage(self.stages, stage_index)
                stage_model = ResNet18ForCIL(
                    num_classes=len(seen_classes), pretrained=False
                ).to(self.device)
                stage_model.load_state_dict(checkpoint["model_state_dict"])
                self.model = stage_model
                seen_class_to_index = {c: i for i, c in enumerate(seen_classes)}
                test_samples = [
                    sample
                    for sample in data_by_split.get("test", [])
                    if sample.class_name in seen_classes
                ]
                test_metrics = self.evaluate(
                    test_samples,
                    seen_class_to_index,
                    transform=eval_transform or transform,
                )
                test_metrics_history.append(test_metrics)
                self._save_evaluation_outputs(stage_index, test_metrics, seen_classes)
            self.model = final_model
            cil_results = compute_cil_metrics(
                test_metrics_history,
                self.stage_class_splits,
                class_names=self.class_names,
            )
            self._save_cil_outputs(cil_results)

        for r in cil_results:
            self.logger.info(
                f"CIL Stage {r['stage_index'] + 1}: "
                f"overall={r['overall_accuracy']:.4f}, "
                f"old={r['old_class_accuracy']:.4f}, new={r['new_class_accuracy']:.4f}, "
                f"forgetting={r['mean_forgetting']:.4f}, retention={r['mean_retention']:.4f}"
            )
        return {
            "stage_metrics": list(self.stage_metrics_history),
            "test_stage_metrics": test_metrics_history,
            "validation_cil_summary": validation_cil_results,
            "cil_summary": cil_results,
            "output_dirs": dict(self.output_dirs),
        }


class StageManager:
    def __init__(self, trainer: CILTrainer):
        self.trainer = trainer

    def run(
        self, data_by_split: Dict[str, List[SampleRecord]], transform=None
    ) -> bool:
        self.trainer.run_full_pipeline(data_by_split, transform=transform)
        return True
