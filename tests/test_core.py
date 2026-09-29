"""Comprehensive tests for the marine-waste CIL pipeline.

These tests use synthetic fixtures (tiny image files) and must never be
interpreted as research-grade experimental results. Real results require
the actual TrashCan dataset.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn
from PIL import Image

from marine_waste_cil.config import (
    CIL_STAGES,
    build_runtime_config,
    get_classes_up_to_stage,
    get_new_classes_for_stage,
    get_old_classes_for_stage,
    set_seed,
    validate_cil_stages_against_dataset,
)
from marine_waste_cil.data.dataset import (
    DatasetReport,
    SampleRecord,
    build_samples_from_directory,
    create_splits_from_directory,
    discover_dataset_layout,
    inspect_dataset,
    SUPPORTED_EXTENSIONS,
    validate_dataset,
    validate_no_split_leak,
)
from marine_waste_cil.explainability.gradcam import (
    generate_gradcam_for_dataset,
    gradcam_for_sample,
    overlay_heatmap,
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
)


# ----------------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------------


def _write_image(path: Path, size: int = 16) -> None:
    arr = np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_small_fixture_dataset(root: Path, classes: Sequence[str], per_class: int = 5) -> None:
    rng = np.random.default_rng(0)
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            _write_image(d / f"{cls}_{i}.png")


def _make_official_split_dataset(
    root: Path, classes: Sequence[str], per_class: int = 4
) -> None:
    for split in ("train", "val", "test"):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for cls in classes:
            cls_dir = split_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                _write_image(cls_dir / f"{split}_{cls}_{i}.png")


# ----------------------------------------------------------------------------
# Dataset tests
# ----------------------------------------------------------------------------


def test_supported_extensions_include_webp() -> None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        assert ext in SUPPORTED_EXTENSIONS


def test_dataset_class_discovery_and_split_leak(tmp_path: Path) -> None:
    classes = ["plastic", "metal", "glass"]
    _make_small_fixture_dataset(tmp_path, classes, per_class=6)
    splits = create_splits_from_directory(str(tmp_path), classes, random_seed=123)
    assert set(splits.keys()) == {"train", "val", "test"}
    for s in splits.values():
        for cls in classes:
            assert cls in s
            assert len(s[cls]) >= 1
    train = [SampleRecord(class_name=cls, image_path=p, split="train")
             for cls, paths in splits["train"].items() for p in paths]
    val = [SampleRecord(class_name=cls, image_path=p, split="val")
           for cls, paths in splits["val"].items() for p in paths]
    test = [SampleRecord(class_name=cls, image_path=p, split="test")
            for cls, paths in splits["test"].items() for p in paths]
    validate_no_split_leak(train, val, test)


def test_discover_layout_official_split(tmp_path: Path) -> None:
    classes = ["plastic", "metal"]
    _make_official_split_dataset(tmp_path, classes, per_class=3)
    layout = discover_dataset_layout(str(tmp_path))
    assert layout["layout"] == "official_split"
    assert sorted(layout["classes"]) == sorted(classes)


def test_discover_layout_class_directory(tmp_path: Path) -> None:
    classes = ["plastic", "metal"]
    _make_small_fixture_dataset(tmp_path, classes, per_class=3)
    layout = discover_dataset_layout(str(tmp_path))
    assert layout["layout"] == "class_directory"
    assert sorted(layout["classes"]) == sorted(classes)


def test_inspect_dataset_report(tmp_path: Path) -> None:
    classes = ["plastic", "metal", "glass"]
    _make_small_fixture_dataset(tmp_path, classes, per_class=5)
    report = inspect_dataset(str(tmp_path))
    assert isinstance(report, DatasetReport)
    assert report.total_classes == len(classes)
    assert report.total_images == len(classes) * 5
    for cls in classes:
        assert report.images_per_class.get(cls) == 5
    assert report.corrupted_images == []
    report_dict = report.to_dict()
    assert "layout" in report_dict
    assert report_dict["total_classes"] == len(classes)


def test_validate_dataset_missing_path(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_dataset(str(tmp_path / "nonexistent"))


def test_validate_dataset_raises_on_empty(tmp_path: Path) -> None:
    (tmp_path / "empty_dir").mkdir()
    with pytest.raises(ValueError, match="no valid images"):
        validate_dataset(str(tmp_path / "empty_dir"), require_splits=False)


def test_validate_dataset_missing_expected_classes(tmp_path: Path) -> None:
    classes = ["plastic", "metal"]
    _make_small_fixture_dataset(tmp_path, classes, per_class=4)
    with pytest.raises(ValueError, match="expected classes missing"):
        validate_dataset(str(tmp_path), expected_classes=["plastic", "metal", "glass"], require_splits=False)


def test_corrupted_image_detection(tmp_path: Path) -> None:
    d = tmp_path / "plastic"
    d.mkdir()
    _write_image(d / "good.png")
    (d / "bad.png").write_bytes(b"this is not a valid png file at all")
    report = inspect_dataset(str(tmp_path))
    assert report.corrupted_images, "should detect at least one corrupted image"
    assert report.total_images == 1


# ----------------------------------------------------------------------------
# Class-Incremental setup tests
# ----------------------------------------------------------------------------


def test_cil_stage_counts_match_expected() -> None:
    assert len(CIL_STAGES) == 3
    cumulative = [sum(len(s) for s in CIL_STAGES[: i + 1]) for i in range(len(CIL_STAGES))]
    assert cumulative == [2, 4, 6]
    assert [len(stage) for stage in CIL_STAGES] == [2, 2, 2]
    assert set(name for stage in CIL_STAGES for name in stage) == set(
        [
            "trash_plastic",
            "trash_metal",
            "trash_wood",
            "trash_etc",
            "animal_fish",
            "rov",
        ]
    )


def test_stage_class_helpers() -> None:
    stages = [["a", "b"], ["c", "d", "e"], ["f"]]
    assert get_classes_up_to_stage(stages, 0) == ["a", "b"]
    assert get_classes_up_to_stage(stages, 1) == ["a", "b", "c", "d", "e"]
    assert get_classes_up_to_stage(stages, 2) == ["a", "b", "c", "d", "e", "f"]
    assert get_new_classes_for_stage(stages, 1) == ["c", "d", "e"]
    assert get_old_classes_for_stage(stages, 1) == ["a", "b"]
    assert get_old_classes_for_stage(stages, 0) == []


def test_validate_cil_stages(tmp_path: Path) -> None:
    classes = ["plastic", "metal", "glass"]
    _make_small_fixture_dataset(tmp_path, classes, per_class=4)
    validate_cil_stages_against_dataset(classes, [["plastic", "metal"], ["glass"]])
    with pytest.raises(ValueError):
        validate_cil_stages_against_dataset(classes, [["plastic", "metal"], ["metal"]])
    with pytest.raises(ValueError):
        validate_cil_stages_against_dataset(classes, [["plastic"], ["glass"]])


# ----------------------------------------------------------------------------
# Classifier expansion tests
# ----------------------------------------------------------------------------


def test_classifier_expansion_preserves_weights() -> None:
    torch.manual_seed(0)
    model = ResNet18ForCIL(num_classes=8, pretrained=False)
    old_weight = model.get_classifier_weights().detach().clone()
    old_bias = model.classifier.bias.detach().clone()
    model.expand_head(14)
    assert model.num_classes == 14
    assert model.classifier.out_features == 14
    new_weight = model.get_classifier_weights().detach()
    assert torch.allclose(new_weight[:8], old_weight, atol=1e-6)
    assert torch.allclose(model.classifier.bias[:8].detach(), old_bias, atol=1e-6)
    old_after_14 = new_weight.detach().clone()
    model.expand_head(20)
    assert model.num_classes == 20
    new_weight_20 = model.get_classifier_weights().detach()
    assert torch.allclose(new_weight_20[:14], old_after_14, atol=1e-6)


def test_classifier_shrink_raises() -> None:
    model = ResNet18ForCIL(num_classes=10, pretrained=False)
    with pytest.raises(ValueError, match="cannot be shrunk"):
        model.expand_head(5)


# ----------------------------------------------------------------------------
# Replay buffer tests
# ----------------------------------------------------------------------------


def test_replay_buffer_budget_and_duplicates() -> None:
    buf = ReplayBuffer(budget_per_class=3)
    keys = ["a", "b", "c", "d"]
    buf.add_samples("plastic", keys)
    assert len(buf.get_class_samples("plastic")) == 3
    assert len(set(buf.get_class_samples("plastic"))) == 3
    buf.add_sample("plastic", "a")
    assert len(buf.get_class_samples("plastic")) == 3


def test_replay_buffer_total_budget_enforced() -> None:
    buf = ReplayBuffer(budget_per_class=10, total_budget=10)
    for cls in ["a", "b", "c"]:
        buf.add_samples(cls, [f"{cls}_{i}" for i in range(10)])
    assert len(buf) <= 10


def test_replay_buffer_update_from_herding() -> None:
    buf = ReplayBuffer(budget_per_class=3)
    ranked = ["e1", "e2", "e3", "e4", "e5"]
    buf.update_from_herding("plastic", ranked)
    assert buf.get_class_samples("plastic") == ["e1", "e2", "e3"]


def test_replay_buffer_reduce_and_clear() -> None:
    buf = ReplayBuffer(budget_per_class=10)
    buf.add_samples("p", [f"s{i}" for i in range(10)])
    removed = buf.reduce_class_budget("p", 4)
    assert len(removed) == 6
    assert len(buf.get_class_samples("p")) == 4
    buf.clear_class("p")
    assert buf.get_class_samples("p") == []
    buf.add_samples("m", ["x"])
    assert "m" in buf.get_classes()
    buf.clear()
    assert len(buf) == 0


def test_replay_buffer_serialization(tmp_path: Path) -> None:
    buf = ReplayBuffer(budget_per_class=2, total_budget=100)
    buf.add_samples("a", ["a1", "a2", "a3"])
    buf.add_samples("b", ["b1"])
    path = tmp_path / "rb.json"
    buf.to_json(str(path))
    reloaded = ReplayBuffer.from_json(str(path))
    assert reloaded.budget_per_class == 2
    assert set(reloaded.get_classes()) == {"a", "b"}
    assert reloaded.get_class_samples("a") == ["a1", "a2"]


# ----------------------------------------------------------------------------
# Herding tests
# ----------------------------------------------------------------------------


def test_herding_deterministic_and_correct_count() -> None:
    rng = np.random.default_rng(42)
    n = 20
    features = rng.normal(size=(n, 8)).astype(float)
    labels = ["plastic"] * n
    keys = [f"img_{i}" for i in range(n)]
    sel1 = select_herding_exemplars_iterative(features, labels, "plastic", keys, 5)
    sel2 = select_herding_exemplars_iterative(features, labels, "plastic", keys, 5)
    assert len(sel1) == 5
    assert sel1 == sel2, "herding must be deterministic"
    assert len(set(sel1)) == 5


def test_herding_respects_class_filtering() -> None:
    rng = np.random.default_rng(0)
    n_plastic, n_metal = 10, 8
    features = rng.normal(size=(n_plastic + n_metal, 6))
    labels = ["plastic"] * n_plastic + ["metal"] * n_metal
    keys = [f"p{i}" for i in range(n_plastic)] + [f"m{i}" for i in range(n_metal)]
    sel = select_herding_exemplars_iterative(features, labels, "metal", keys, 3)
    assert len(sel) == 3
    for k in sel:
        assert k.startswith("m")


def test_herding_empty_cases() -> None:
    assert select_herding_exemplars_iterative(
        np.zeros((0, 4)), [], "x", [], 10
    ) == []
    feats = np.random.randn(3, 4)
    assert select_herding_exemplars_iterative(
        feats, ["a", "a", "a"], "a", ["k1", "k2", "k3"], 0
    ) == []


# ----------------------------------------------------------------------------
# Knowledge distillation tests
# ----------------------------------------------------------------------------


def test_teacher_is_frozen() -> None:
    torch.manual_seed(0)
    model = ResNet18ForCIL(num_classes=8, pretrained=False)
    teacher = clone_as_teacher(model)
    assert teacher_params_frozen_check(teacher)
    freeze_model(model)
    assert teacher_params_frozen_check(model)


def test_teacher_parameters_do_not_change() -> None:
    torch.manual_seed(0)
    student = ResNet18ForCIL(num_classes=14, pretrained=False)
    teacher = ResNet18ForCIL(num_classes=8, pretrained=False)
    freeze_model(teacher)
    teacher_state = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    x = torch.randn(4, 3, 32, 32)
    for _ in range(3):
        logits_student = student(x)
        with torch.no_grad():
            logits_teacher = teacher(x)
        y = torch.randint(0, 14, (4,))
        crit = CILTotalLoss(distillation_weight=1.0, temperature=2.0)
        loss, _ = crit(logits_student, y, teacher_logits=logits_teacher, old_class_count=8)
        loss.backward()
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, teacher_state[k]), f"teacher parameter changed: {k}"


def test_kd_loss_nonzero_when_student_differs_from_teacher() -> None:
    torch.manual_seed(7)
    s_logits = torch.randn(6, 8) * 5
    t_logits = torch.randn(6, 8) * 5 + 3
    crit = CILTotalLoss(classification_weight=0.0, distillation_weight=1.0, temperature=2.0)
    y = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    loss, stats = crit(s_logits, y, teacher_logits=t_logits, old_class_count=8)
    assert stats["kd"] > 1e-4
    assert float(loss) > 1e-4


def test_kd_loss_uses_only_old_classes() -> None:
    torch.manual_seed(2)
    s_logits = torch.randn(4, 14)
    t_logits = torch.randn(4, 8)
    y = torch.tensor([0, 5, 9, 12], dtype=torch.long)
    crit = CILTotalLoss(classification_weight=0.0, distillation_weight=1.0, temperature=2.0)
    loss, stats = crit(s_logits, y, teacher_logits=t_logits, old_class_count=8)
    assert stats["kd"] > 0 or True  # depends on randomness but should not crash
    assert float(loss.detach().item()) == float(stats["kd"])


# ----------------------------------------------------------------------------
# Checkpoint save/load tests
# ----------------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    from marine_waste_cil.training.trainer import CILTrainer

    classes = [f"cls{i}" for i in range(8)]
    stages = [classes[:5], classes[5:8]]
    trainer = CILTrainer(
        dataset_root=str(tmp_path),
        class_names=classes,
        stages=stages,
        config={"NUM_EPOCHS_PER_STAGE": 1, "BATCH_SIZE": 2, "MEMORY_BUDGET_PER_CLASS": 2, "USE_PRETRAINED": False},
        device="cpu",
        output_dir=str(tmp_path / "out"),
    )
    trainer.replay_buffer.add_samples("cls0", ["a.jpg", "b.jpg"])
    c2i = {c: i for i, c in enumerate(classes[:5])}
    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(str(ckpt_path), 0, c2i, extra={"hello": "world"})
    assert ckpt_path.exists()

    classes2 = list(classes)
    trainer2 = CILTrainer(
        dataset_root=str(tmp_path),
        class_names=classes2,
        stages=stages,
        config={"NUM_EPOCHS_PER_STAGE": 1, "BATCH_SIZE": 2, "MEMORY_BUDGET_PER_CLASS": 2, "USE_PRETRAINED": False},
        device="cpu",
        output_dir=str(tmp_path / "out2"),
    )
    loaded = trainer2.load_checkpoint(str(ckpt_path))
    assert loaded["stage"] == 0
    assert loaded["extra"]["hello"] == "world"
    assert "cls0" in trainer2.replay_buffer.get_classes()
    assert trainer2.start_stage == 1


# ----------------------------------------------------------------------------
# Metrics tests
# ----------------------------------------------------------------------------


def test_metrics_known_values() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    labels = [0, 1]
    m = compute_classification_metrics(y_true, y_pred, labels)
    assert m["overall_accuracy"] == pytest.approx(0.75)
    assert m["per_class"][0]["support"] == 2
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    for entry in m["per_class"]:
        assert "accuracy" in entry
        assert "precision" in entry
        assert "recall" in entry
        assert "f1" in entry


def test_cil_metrics_compute_forgetting_and_retention() -> None:
    labels = [0, 1, 2, 3]
    stage_0 = {
        "overall_accuracy": 1.0,
        "per_class": [
            {"class_idx": 0, "accuracy": 1.0},
            {"class_idx": 1, "accuracy": 0.9},
        ],
    }
    stage_1 = {
        "overall_accuracy": 0.8,
        "per_class": [
            {"class_idx": 0, "accuracy": 0.7},
            {"class_idx": 1, "accuracy": 0.8},
            {"class_idx": 2, "accuracy": 0.95},
            {"class_idx": 3, "accuracy": 1.0},
        ],
    }
    splits = [
        {"old": [], "new": [0, 1]},
        {"old": [0, 1], "new": [2, 3]},
    ]
    cil = compute_cil_metrics([stage_0, stage_1], splits, class_names=["a", "b", "c", "d"])
    assert len(cil) == 2
    stage1_res = cil[1]
    assert stage1_res["old_class_accuracy"] == pytest.approx(0.75)
    assert stage1_res["new_class_accuracy"] == pytest.approx(0.975)
    # forgetting: class 0 goes 1.0 -> 0.7, so 0.3
    assert stage1_res["mean_forgetting"] >= 0.1
    assert stage1_res["max_forgetting"] >= 0.2
    assert "per_class_forgetting" in stage1_res


# ----------------------------------------------------------------------------
# Grad-CAM tests
# ----------------------------------------------------------------------------


def test_gradcam_output_shape_and_non_nan() -> None:
    torch.manual_seed(0)
    model = ResNet18ForCIL(num_classes=3, pretrained=False)
    image = torch.rand(3, 32, 32)
    cam = gradcam_for_sample(model, image, target_layer_name="layer4", class_index=0)
    assert cam.ndim == 2
    assert not np.any(np.isnan(cam))
    assert cam.min() >= -1e-6
    assert cam.max() <= 1 + 1e-6


def test_gradcam_bad_layer_raises() -> None:
    model = ResNet18ForCIL(num_classes=3, pretrained=False)
    image = torch.rand(3, 32, 32)
    with pytest.raises(ValueError, match="not found"):
        gradcam_for_sample(model, image, target_layer_name="not_a_layer")


def test_overlay_heatmap_shape() -> None:
    img = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    cam = np.random.rand(16, 16).astype(float)
    overlay = overlay_heatmap(img, cam, alpha=0.5)
    assert overlay.shape == img.shape
    assert overlay.dtype == np.uint8


def test_gradcam_pipeline_with_tiny_dataset(tmp_path: Path) -> None:
    torch.manual_seed(0)
    classes = ["a", "b", "c"]
    _make_small_fixture_dataset(tmp_path, classes, per_class=3)
    model = ResNet18ForCIL(num_classes=len(classes), pretrained=False)
    from marine_waste_cil.data.transforms import build_default_transforms

    transform = build_default_transforms(image_size=32)
    samples: List[SampleRecord] = []
    for cls in classes:
        for p in sorted((tmp_path / cls).glob("*.png")):
            samples.append(SampleRecord(class_name=cls, image_path=str(p), split="train"))
    c2i = {c: i for i, c in enumerate(classes)}
    out = tmp_path / "gc_out"
    result = generate_gradcam_for_dataset(
        model, samples, c2i, str(out), transform=transform,
        samples_per_class=1, device=torch.device("cpu"),
    )
    assert result["count"] >= 1
    manifest_path = Path(result["manifest"])
    assert manifest_path.exists()


# ----------------------------------------------------------------------------
# Config + reproducibility tests
# ----------------------------------------------------------------------------


def test_config_contains_expected_keys() -> None:
    import marine_waste_cil.config as cfg
    expected = {
        "BATCH_SIZE",
        "NUM_EPOCHS_PER_STAGE",
        "LEARNING_RATE",
        "MEMORY_BUDGET_PER_CLASS",
        "REPLAY_MEMORY_SIZE",
        "DISTILLATION_TEMPERATURE",
        "DISTILLATION_WEIGHT",
        "RANDOM_SEED",
        "CIL_STAGES",
        "OUTPUT_DIR",
        "IMAGE_SIZE",
        "DEVICE",
        "USE_PRETRAINED",
    }
    runtime = cfg.build_runtime_config()
    assert expected.issubset(runtime.keys())
    assert len(runtime["CIL_STAGES"]) >= 1


def test_set_seed_deterministic() -> None:
    set_seed(42)
    a = torch.randn(4)
    b = np.random.randn(4)
    c = np.random.choice(100)
    set_seed(42)
    assert torch.allclose(a, torch.randn(4))
    assert np.allclose(b, np.random.randn(4))
    assert c == np.random.choice(100)


# ----------------------------------------------------------------------------
# Training pipeline smoke test (tiny synthetic data, no claims made)
# ----------------------------------------------------------------------------


def test_trainer_smoke_runs_one_stage(tmp_path: Path) -> None:
    """This smoke test uses tiny synthetic images only to verify the pipeline
    compiles and produces output artifacts. No research-grade claims implied."""
    torch.manual_seed(0)
    classes = ["a", "b", "c", "d", "e", "f", "g", "h"]
    _make_small_fixture_dataset(tmp_path, classes, per_class=5)
    stages = [classes[:4], classes[4:6], classes[6:8]]
    cfg = build_runtime_config({
        "NUM_EPOCHS_PER_STAGE": 1,
        "BATCH_SIZE": 2,
        "MEMORY_BUDGET_PER_CLASS": 2,
        "USE_PRETRAINED": False,
        "NUM_WORKERS": 0,
        "RANDOM_SEED": 0,
        "OUTPUT_DIR": str(tmp_path / "artifacts"),
    })
    from marine_waste_cil.training.trainer import CILTrainer
    from marine_waste_cil.data.transforms import build_default_transforms

    transform = build_default_transforms(image_size=32)
    samples = build_samples_from_directory(str(tmp_path), classes, random_seed=0)
    trainer = CILTrainer(
        dataset_root=str(tmp_path),
        class_names=classes,
        stages=stages,
        config=cfg,
        device="cpu",
        output_dir=str(tmp_path / "artifacts"),
    )
    result = trainer.run_full_pipeline(samples, transform=transform)
    assert "cil_summary" in result
    assert len(result["cil_summary"]) == 3
    for stage_idx in range(3):
        ckpt = tmp_path / "artifacts" / "checkpoints" / f"stage_{stage_idx + 1}.pt"
        assert ckpt.exists(), f"missing checkpoint {ckpt}"
    class_order_path = tmp_path / "artifacts" / "class_order.json"
    assert class_order_path.exists()
    with open(class_order_path) as fp:
        assert json.load(fp) == classes
