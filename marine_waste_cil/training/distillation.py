from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, alpha: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        old_class_count: Optional[int] = None,
    ) -> torch.Tensor:
        if old_class_count is not None and old_class_count > 0:
            if old_class_count > student_logits.shape[1]:
                raise ValueError(
                    f"old_class_count ({old_class_count}) exceeds student logits dim ({student_logits.shape[1]})"
                )
            if old_class_count > teacher_logits.shape[1]:
                raise ValueError(
                    f"old_class_count ({old_class_count}) exceeds teacher logits dim ({teacher_logits.shape[1]})"
                )
            student_old = student_logits[:, :old_class_count]
            teacher_old = teacher_logits[:, :old_class_count]
        else:
            student_old = student_logits
            teacher_old = teacher_logits

        student_log_probs = F.log_softmax(student_old / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_old / self.temperature, dim=1)
        kd_loss = self.kl_div(student_log_probs, teacher_probs)
        kd_loss = kd_loss * (self.temperature**2)
        return self.alpha * kd_loss


class CILTotalLoss(nn.Module):
    def __init__(
        self,
        classification_weight: float = 1.0,
        distillation_weight: float = 1.0,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.distillation_weight = distillation_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = DistillationLoss(temperature=temperature, alpha=1.0)

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        old_class_count: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce = self.ce_loss(student_logits, targets)
        loss = self.classification_weight * ce
        stats: Dict[str, float] = {
            "ce": float(ce.detach().cpu().item()),
            "kd": 0.0,
            "total": float(loss.detach().cpu().item()),
        }
        if teacher_logits is not None and old_class_count is not None and old_class_count > 0:
            kd = self.kd_loss(student_logits, teacher_logits, old_class_count=old_class_count)
            stats["kd"] = float(kd.detach().cpu().item())
            loss = loss + self.distillation_weight * kd
            stats["total"] = float(loss.detach().cpu().item())
        return loss, stats


def freeze_model(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def clone_as_teacher(model: nn.Module, device: Optional[torch.device] = None) -> nn.Module:
    teacher = type(model)(
        num_classes=getattr(model, "num_classes", 10),
        pretrained=False,
    )
    teacher.load_state_dict(model.state_dict())
    if device is not None:
        teacher = teacher.to(device)
    freeze_model(teacher)
    return teacher


def teacher_params_frozen_check(model: nn.Module) -> bool:
    for name, p in model.named_parameters():
        if p.requires_grad:
            return False
    return True
