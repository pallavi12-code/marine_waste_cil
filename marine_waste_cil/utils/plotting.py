from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_heatmap(
    cm: np.ndarray,
    labels: Sequence[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.75))
    )
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if val == 0:
                continue
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_curve(
    stage_names: Sequence[str],
    values: Sequence[float],
    save_path: str,
    ylabel: str = "Accuracy",
    title: str = "Accuracy Across Stages",
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(stage_names)), values, marker="o", linewidth=2, color="tab:blue")
    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names)
    ax.set_xlabel("Stage")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_forgetting_barplot(
    classes: Sequence[str],
    forgetting: Sequence[float],
    save_path: str,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if not classes:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No old classes in this stage", ha="center", va="center")
        fig.savefig(save_path, dpi=180)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.5), 5))
    bars = ax.bar(range(len(classes)), forgetting, color="tab:orange", edgecolor="black")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Forgetting (previous - current)")
    ax.set_title("Forgetting Across Classes")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, forgetting):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_retention_barplot(
    classes: Sequence[str],
    retention: Sequence[float],
    save_path: str,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if not classes:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.5), 5))
    ax.bar(range(len(classes)), retention, color="tab:green", edgecolor="black")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Retention (current / previous)")
    ax.set_title("Retention Across Classes")
    ax.set_ylim(0.0, 1.1)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
