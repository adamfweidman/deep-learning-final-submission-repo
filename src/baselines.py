"""Trivial label-only baselines for sanity checks.

These are diagnostic floors: any real model must beat them on
validation. They use no images and no model — they exist so the smoke
path can produce a number and a submission without loading a VLM.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


def predict_always_a(df: pd.DataFrame) -> list[int]:
    """Always pick choice index 0."""
    return [0] * len(df)


def predict_majority_letter(train_df: pd.DataFrame, df: pd.DataFrame) -> list[int]:
    """Pick the choice index that is most often correct in train."""
    counts = Counter(int(a) for a in train_df["answer"])
    if not counts:
        return [0] * len(df)
    most_common_idx = counts.most_common(1)[0][0]
    out = []
    for nc in df["num_choices"].astype(int):
        out.append(most_common_idx if most_common_idx < nc else 0)
    return out


def baseline_predictions(name: str, df: pd.DataFrame, *, train_df: pd.DataFrame | None = None) -> list[int]:
    if name == "always_a":
        return predict_always_a(df)
    if name == "majority_letter":
        if train_df is None:
            raise ValueError("majority_letter requires train_df")
        return predict_majority_letter(train_df, df)
    raise ValueError(f"unknown baseline: {name}")
