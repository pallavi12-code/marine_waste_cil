from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from marine_waste_cil.data.dataset import SampleRecord


def gradcam_for_sample(
    model,
    image_tensor: torch.Tensor,
    target_layer_name: str = "layer4",
    class_index: Optional[int] = None,
) -> np.ndarray:
    model.eval()
    if not hasattr(model, "backbone"):
        raise ValueError("Model must have a backbone attribute for Grad-CAM.")
    if image_tensor.ndim != 3:
        raise ValueError("image_tensor must have shape [C, H, W]")
    layer = getattr(model.backbone, target_layer_name, None)
    if layer is None:
        raise ValueError(f"Target layer '{target_layer_name}' not found on the model backbone.")
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def _forward_hook(module, inputs, output):
        activations.append(output)

    def _backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = layer.register_forward_hook(_forward_hook)
    handle_b = layer.register_full_backward_hook(_backward_hook)
    try:
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device).unsqueeze(0)
        image_tensor.requires_grad_(True)
        logits = model(image_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        if not 0 <= class_index < logits.shape[1]:
            raise ValueError(
                f"class_index {class_index} is outside the model output range {logits.shape[1]}"
            )
        model.zero_grad()
        logits[:, class_index].backward(retain_graph=True)
        if not activations or not gradients:
            raise ValueError("Grad-CAM could not register activations or gradients.")
        weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[-1]).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam_np = cam[0, 0].detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        cam_np = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)
        return cam_np
    finally:
        handle_f.remove()
        handle_b.remove()


def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if cam.shape[:2] != image_np.shape[:2]:
        cam = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_np.copy()
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    blended = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    return blended


def tensor_to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    if image_tensor.ndim != 3:
        raise ValueError("image_tensor must have shape [C, H, W] or [1, C, H, W]")
    mean = np.array([0.485, 0.456, 0.406], dtype=float).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=float).reshape(3, 1, 1)
    img = image_tensor.detach().cpu().numpy()
    img = img * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


def generate_gradcam_grid_figure(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    true_class: str,
    pred_class: str,
    confidence: float,
    save_path: str,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original\nTrue: {true_class}")
    axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nPred: {pred_class}\nConfidence: {confidence:.3f}"
    )
    axes[2].axis("off")
    fig.suptitle(
        f"Grad-CAM Visualization\nTrue: {true_class} | Pred: {pred_class} | Conf: {confidence:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_gradcam_for_dataset(
    model,
    samples: Sequence[SampleRecord],
    class_to_index: Dict[str, int],
    output_dir: str,
    transform,
    samples_per_class: int = 3,
    target_layer_name: str = "layer4",
    device: Optional[torch.device] = None,
    stage_index: Optional[int] = None,
) -> Dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    model.to(device)
    index_to_class = {v: k for k, v in class_to_index.items()}
    samples_by_class: Dict[str, List[SampleRecord]] = {}
    for s in samples:
        if s.class_name not in class_to_index:
            continue
        samples_by_class.setdefault(s.class_name, []).append(s)
    selected_by_class: Dict[str, List[SampleRecord]] = {}
    for cls, class_samples in samples_by_class.items():
        selected: List[SampleRecord] = []
        selected_outcomes: set[bool] = set()
        for sample in class_samples:
            try:
                image = Image.open(sample.image_path).convert("RGB")
            except Exception:
                continue
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = int(model(tensor).argmax(dim=1).item())
            is_correct = prediction == class_to_index[cls]
            if is_correct not in selected_outcomes:
                selected.append(sample)
                selected_outcomes.add(is_correct)
            if len(selected) >= samples_per_class:
                break
        selected_by_class[cls] = selected
    manifest: List[Dict[str, Any]] = []
    selected_count = 0
    pbar = tqdm(
        total=min(samples_per_class * len(samples_by_class), len(samples)),
        desc="Grad-CAM",
        leave=False,
        disable=None,
    )
    try:
        for cls, picked in sorted(selected_by_class.items()):
            for s_idx, sample in enumerate(picked):
                try:
                    pil_img = Image.open(sample.image_path).convert("RGB")
                except Exception as e:
                    continue
                tensor = transform(pil_img).unsqueeze(0).to(device)
                tensor_sq = tensor[0]
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = int(logits.argmax(dim=1).item())
                    confidence = float(probs[0, pred_idx].item())
                pred_class = index_to_class.get(pred_idx, f"unknown_{pred_idx}")
                target_idx = class_to_index.get(cls, pred_idx)
                try:
                    cam = gradcam_for_sample(
                        model,
                        tensor_sq.detach().cpu(),
                        target_layer_name=target_layer_name,
                        class_index=target_idx,
                    )
                except Exception:
                    continue
                orig_np = np.array(pil_img)
                if cam.shape[:2] != orig_np.shape[:2]:
                    cam_resized = cv2.resize(cam, (orig_np.shape[1], orig_np.shape[0]))
                else:
                    cam_resized = cam
                overlay = overlay_heatmap(orig_np, cam_resized, alpha=0.5)
                heatmap_rgb = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
                safe_cls = "".join(c if c.isalnum() or c in "-_" else "_" for c in cls)
                filename = f"gradcam_{safe_cls}_{s_idx + 1}.png"
                save_path = os.path.join(output_dir, filename)
                generate_gradcam_grid_figure(
                    orig_np,
                    cam_resized,
                    overlay,
                    true_class=cls,
                    pred_class=pred_class,
                    confidence=confidence,
                    save_path=save_path,
                )
                manifest.append(
                    {
                        "file": filename,
                        "image_path": sample.image_path,
                        "true_class": cls,
                        "predicted_class": pred_class,
                        "stage": stage_index + 1 if stage_index is not None else None,
                        "prediction_status": "correct" if cls == pred_class else "incorrect",
                        "confidence": confidence,
                        "correct": cls == pred_class,
                    }
                )
                selected_count += 1
                pbar.update(1)
    finally:
        pbar.close()
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "total_samples": selected_count,
                "samples_per_class": samples_per_class,
                "stage": stage_index + 1 if stage_index is not None else None,
                "samples": manifest,
            },
            fp,
            indent=2,
        )
    return {"count": selected_count, "manifest": manifest_path}
