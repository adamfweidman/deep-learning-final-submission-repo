"""Validation metrics."""

from __future__ import annotations

from typing import Sequence


def accuracy(preds: Sequence[int], targets: Sequence[int]) -> float:
    if len(preds) != len(targets):
        raise ValueError(f"length mismatch: preds={len(preds)} targets={len(targets)}")
    if not preds:
        return 0.0
    correct = sum(int(p) == int(t) for p, t in zip(preds, targets))
    return correct / len(preds)
