from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, budget_per_class: int = 20, total_budget: Optional[int] = None):
        self.budget_per_class = budget_per_class
        self.total_budget = total_budget
        self._store: Dict[str, List[str]] = defaultdict(list)

    def add_sample(self, class_name: str, sample_key: str) -> None:
        if sample_key in self._store[class_name]:
            return
        self._store[class_name].append(sample_key)
        if len(self._store[class_name]) > self.budget_per_class:
            self._store[class_name] = self._store[class_name][: self.budget_per_class]
        self._enforce_total_budget()

    def add_samples(self, class_name: str, sample_keys: Sequence[str]) -> None:
        seen = set(self._store[class_name])
        for key in sample_keys:
            if key not in seen:
                self._store[class_name].append(key)
                seen.add(key)
        if len(self._store[class_name]) > self.budget_per_class:
            self._store[class_name] = self._store[class_name][: self.budget_per_class]
        self._enforce_total_budget()

    def _enforce_total_budget(self) -> None:
        if self.total_budget is None:
            return
        total = len(self)
        if total <= self.total_budget:
            return
        classes_with_exemplars = [c for c, v in self._store.items() if v]
        per_class_allowed = max(1, self.total_budget // max(1, len(classes_with_exemplars)))
        for c in classes_with_exemplars:
            if len(self._store[c]) > per_class_allowed:
                self._store[c] = self._store[c][:per_class_allowed]

    def get_class_samples(self, class_name: str) -> List[str]:
        return list(self._store.get(class_name, []))

    def get_all_samples(self) -> List[Tuple[str, str]]:
        result: List[Tuple[str, str]] = []
        for cls in sorted(self._store.keys()):
            for key in self._store[cls]:
                result.append((cls, key))
        return result

    def sample_batch(self, class_name: str, batch_size: int) -> List[str]:
        samples = self._store.get(class_name, [])
        if not samples:
            return []
        if len(samples) <= batch_size:
            return samples[:]
        rng = np.random.default_rng()
        idxs = rng.choice(len(samples), size=batch_size, replace=False)
        return [samples[i] for i in sorted(idxs)]

    def as_dict(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in sorted(self._store.items())}

    def update_from_herding(
        self, class_name: str, ranked_keys: Sequence[str], budget: Optional[int] = None
    ) -> None:
        limit = self.budget_per_class if budget is None else budget
        kept: List[str] = []
        seen: set[str] = set()
        for key in ranked_keys:
            if key not in seen:
                kept.append(key)
                seen.add(key)
            if len(kept) >= limit:
                break
        self._store[class_name] = kept
        self._enforce_total_budget()

    def reduce_class_budget(self, class_name: str, new_budget: int) -> List[str]:
        if new_budget < 0:
            raise ValueError("new_budget must be non-negative")
        current = self._store.get(class_name, [])
        if len(current) > new_budget:
            removed = current[new_budget:]
            self._store[class_name] = current[:new_budget]
            return removed
        return []

    def get_classes(self) -> List[str]:
        return sorted([c for c, v in self._store.items() if v])

    def clear(self) -> None:
        self._store.clear()

    def clear_class(self, class_name: str) -> None:
        if class_name in self._store:
            self._store[class_name] = []

    def __len__(self) -> int:
        return sum(len(v) for v in self._store.values())

    def __contains__(self, item: Tuple[str, str]) -> bool:
        cls, key = item
        return key in self._store.get(cls, [])

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "budget_per_class": self.budget_per_class,
            "total_budget": self.total_budget,
            "store": self.as_dict(),
        }
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ReplayBuffer":
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        buf = cls(
            budget_per_class=int(payload.get("budget_per_class", 20)),
            total_budget=payload.get("total_budget"),
        )
        for cls, keys in payload.get("store", {}).items():
            buf._store[cls] = list(keys)
        return buf


def select_herding_exemplars_iterative(
    features: np.ndarray,
    labels: Sequence[Any],
    class_name: Any,
    sample_keys: Sequence[str],
    budget_per_class: int,
) -> List[str]:
    """
    Genuine iCaRL-style herding.

    Iteratively selects exemplars so that the mean of selected features
    best approximates the class mean. This is the standard algorithm from
    the iCaRL paper, not a simple distance-from-mean ranking.
    """
    if budget_per_class <= 0 or not sample_keys:
        return []
    labels_list = list(labels)
    keys_list = list(sample_keys)
    if len(features) != len(labels_list) or len(labels_list) != len(keys_list):
        raise ValueError("features, labels, and sample_keys must have the same length")

    class_entries: List[Tuple[str, np.ndarray]] = []
    seen_keys: set[str] = set()
    for key, label, feat in zip(keys_list, labels_list, features):
        if label != class_name:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        class_entries.append((key, np.asarray(feat, dtype=float)))

    if not class_entries:
        return []

    budget = min(budget_per_class, len(class_entries))
    unique_keys = [k for k, _ in class_entries]
    class_features = np.stack([f for _, f in class_entries], axis=0)
    if class_features.ndim != 2:
        raise ValueError("features must be 2-dimensional")

    class_mean = class_features.mean(axis=0)
    n = class_features.shape[0]
    selected_indices: List[int] = []
    selected_mask = np.zeros(n, dtype=bool)
    for _ in range(budget):
        current_count = len(selected_indices)
        if current_count == 0:
            sum_selected = np.zeros_like(class_mean)
        else:
            sum_selected = class_features[selected_mask].sum(axis=0)
        candidates_mask = ~selected_mask
        candidate_idx = np.where(candidates_mask)[0]
        candidate_means = (sum_selected[None, :] + class_features[candidate_idx]) / (current_count + 1)
        diffs = candidate_means - class_mean[None, :]
        distances = (diffs**2).sum(axis=1)
        best_local = int(np.argmin(distances))
        best_global = candidate_idx[best_local]
        selected_indices.append(int(best_global))
        selected_mask[best_global] = True

    selected_keys = [unique_keys[i] for i in selected_indices]
    return selected_keys


def select_herding_exemplars(
    features,
    labels,
    class_name,
    sample_keys,
    budget_per_class,
):
    """
    Backward-compatible wrapper that delegates to the iterative herding algorithm.
    """
    features_arr = np.asarray(features, dtype=float)
    return select_herding_exemplars_iterative(
        features_arr, labels, class_name, sample_keys, budget_per_class
    )


def compute_class_means(
    features: np.ndarray, labels: Sequence[Any]
) -> Dict[Any, np.ndarray]:
    class_to_rows: Dict[Any, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        class_to_rows[lab].append(idx)
    means: Dict[Any, np.ndarray] = {}
    for lab, rows in class_to_rows.items():
        means[lab] = features[rows].mean(axis=0)
    return means
