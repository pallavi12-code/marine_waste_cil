#!/usr/bin/env python3
"""Single-image inference script for Marine Waste CIL."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marine_waste_cil.config import CIL_STAGES, build_runtime_config
from marine_waste_cil.data.dataset import discover_dataset_layout, get_class_names_in_order
from marine_waste_cil.data.transforms import build_eval_transforms
from marine_waste_cil.models.backbone import ResNet18ForCIL


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image prediction using CIL checkpoint.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint file.")
    parser.add_argument("--data-root", type=str, default="data/TrashCan_processed_seed42", help="Path to dataset root.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda.")
    return parser.parse_args()


def predict_single_image(image_path: str, checkpoint_path: str, data_root: str, device: str = None):
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(ckpt_p, map_location=device)
    class_names = ckpt.get("class_order", ckpt.get("class_names", [c for stage in CIL_STAGES for c in stage]))

    model = ResNet18ForCIL(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = build_eval_transforms(image_size=224)
    image = Image.open(img_p).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(logits.argmax(dim=1).item())
        predicted_class = class_names[pred_idx]
        confidence = float(probs[pred_idx].item())

    prob_dict = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

    print("\n" + "=" * 40)
    print("  Marine Waste CIL Single-Image Prediction")
    print("=" * 40)
    print(f"Image Path     : {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence     : {confidence:.2%}")
    print("\nClass Probabilities:")
    for cls, prob in prob_dict.items():
        print(f"  - {cls:<15}: {prob:.2%}")
    print("=" * 40 + "\n")

    return {
        "image_path": image_path,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict,
    }


def main():
    args = parse_args()
    predict_single_image(args.image_path, args.checkpoint, args.data_root, device=args.device)


if __name__ == "__main__":
    main()
