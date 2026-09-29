from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torchvision import models


class ResNet18ForCIL(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, target_layer: str = "layer4"):
        super().__init__()
        self.num_classes = num_classes
        self.target_layer = target_layer
        self._features = None
        self._gradients = None

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

        self.backbone_layer = getattr(self.backbone, target_layer, None)
        if self.backbone_layer is None:
            raise ValueError(f"Target layer '{target_layer}' not found in ResNet18")

    def activations_hook(self, grad):
        self._gradients = grad

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        self._features = x
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        feats = torch.flatten(self.backbone.avgpool(x), 1)
        return feats

    def forward(self, x: torch.Tensor):
        feats = self.forward_features(x)
        logits = self.classifier(feats)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            feats = self.forward_features(x)
        return feats

    def get_last_conv_layer(self):
        return self.backbone.layer4

    def get_gradients(self):
        return self._gradients

    def get_activations(self):
        return self._features

    def expand_head(self, new_num_classes: int) -> "ResNet18ForCIL":
        old_weight = self.classifier.weight.detach().clone()
        old_bias = self.classifier.bias.detach().clone() if self.classifier.bias is not None else None
        old_count = old_weight.shape[0]
        if new_num_classes < old_count:
            raise ValueError("Classifier head cannot be shrunk during CIL training.")
        if new_num_classes == old_count:
            return self
        device = old_weight.device
        in_features = old_weight.shape[1]
        new_classifier = nn.Linear(in_features, new_num_classes).to(device)
        nn.init.kaiming_normal_(new_classifier.weight, nonlinearity="linear")
        if new_classifier.bias is not None:
            nn.init.zeros_(new_classifier.bias)
        with torch.no_grad():
            new_classifier.weight[:old_count, :] = old_weight
            if old_bias is not None and new_classifier.bias is not None:
                new_classifier.bias[:old_count] = old_bias
        self.classifier = new_classifier
        self.num_classes = new_num_classes
        return self

    def get_classifier_weights(self) -> torch.Tensor:
        return self.classifier.weight.detach().clone()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def set_trainable(self, train_classifier_only: bool = False) -> None:
        if train_classifier_only:
            self.freeze_backbone()
            for p in self.classifier.parameters():
                p.requires_grad = True
        else:
            self.unfreeze_backbone()
            for p in self.classifier.parameters():
                p.requires_grad = True


class GradCAM:
    def __init__(self, model: ResNet18ForCIL):
        self.model = model
        self.gradients = None
        self.activations = None
        self._forward_handle = model.backbone.layer4.register_forward_hook(self._forward_hook)
        self._backward_handle = model.backbone.layer4.register_full_backward_hook(self._backward_hook)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _forward_hook(self, module, input, output):
        self.activations = output

    def __call__(self, input_tensor: torch.Tensor, class_index: Optional[int] = None):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        if not 0 <= class_index < logits.shape[1]:
            raise ValueError(f"class_index {class_index} is outside model output range {logits.shape[1]}")
        self.model.zero_grad()
        logits[:, class_index].backward(retain_graph=True)
        gradients = self.gradients
        activations = self.activations
        if gradients is None or activations is None:
            raise ValueError("Grad-CAM could not capture gradients or activations from the target layer.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam[0, 0]
        cam = cam.detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        denom = cam_max - cam_min
        cam = (cam - cam_min) / (denom + 1e-8)
        return cam

    def close(self):
        self._forward_handle.remove()
        self._backward_handle.remove()
