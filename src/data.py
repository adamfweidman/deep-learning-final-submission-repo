"""CSV schema validation, choice parsing, and image-path resolution.

The competition data lives under ``data/`` (a symlink layout — see
``DATA.md``). CSV ``image_path`` values are relative to that directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

LABELED_REQUIRED = {
    "id",
    "image_path",
    "question",
    "choices",
    "num_choices",
    "answer",
}
TEST_REQUIRED = {
    "id",
    "image_path",
    "question",
    "choices",
    "num_choices",
}


@dataclass
class SplitReport:
    split: str
    rows: int
    images_resolved: int
    images_missing: int
    missing_columns: list[str]
    sample_missing_paths: list[str]
    num_choices_min: int
    num_choices_max: int
    answers_in_range: bool | None  # None for unlabeled (test) split


def _parse_choices(value) -> list:
    if isinstance(value, list):
        return value
    return json.loads(value)


def load_split(csv_path: str | Path, *, labeled: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["choices"] = df["choices"].apply(_parse_choices)
    df["num_choices"] = df["num_choices"].astype(int)
    if labeled:
        df["answer"] = df["answer"].astype(int)
    return df


def validate_split(
    df: pd.DataFrame,
    *,
    split: str,
    data_dir: Path,
    labeled: bool,
    sample_check: int | None = None,
) -> SplitReport:
    required = LABELED_REQUIRED if labeled else TEST_REQUIRED
    missing_columns = sorted(required - set(df.columns))

    if sample_check is None:
        check_df = df
    else:
        check_df = df.head(sample_check)

    missing_paths: list[str] = []
    resolved = 0
    for rel in check_df["image_path"]:
        if (data_dir / rel).exists():
            resolved += 1
        else:
            if len(missing_paths) < 5:
                missing_paths.append(str(rel))
    missing = len(check_df) - resolved

    answers_in_range: bool | None = None
    if labeled:
        answers_in_range = bool(
            ((df["answer"] >= 0) & (df["answer"] < df["num_choices"])).all()
        )

    return SplitReport(
        split=split,
        rows=int(len(df)),
        images_resolved=int(resolved),
        images_missing=int(missing),
        missing_columns=missing_columns,
        sample_missing_paths=missing_paths,
        num_choices_min=int(df["num_choices"].min()),
        num_choices_max=int(df["num_choices"].max()),
        answers_in_range=answers_in_range,
    )


def resolve_image_paths(df: pd.DataFrame, data_dir: Path) -> Iterable[Path]:
    return (data_dir / rel for rel in df["image_path"])
