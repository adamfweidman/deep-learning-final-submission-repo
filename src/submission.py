"""Submission CSV writer.

Strict invariants enforced before writing:

- columns are exactly ``id`` and ``answer`` (in that order);
- one row per test id, no duplicates, no missing ids;
- every ``answer`` is a non-negative integer within ``num_choices``
  for the matching test row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


class SubmissionError(ValueError):
    pass


def write_submission(
    predictions: Mapping[str, int],
    test_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    test_ids = list(test_df["id"])
    test_id_set = set(test_ids)
    pred_id_set = set(predictions.keys())

    missing = test_id_set - pred_id_set
    if missing:
        raise SubmissionError(f"{len(missing)} test ids missing predictions; first: {sorted(missing)[:3]}")
    extra = pred_id_set - test_id_set
    if extra:
        raise SubmissionError(f"{len(extra)} predictions for ids not in test.csv; first: {sorted(extra)[:3]}")

    nc_by_id = dict(zip(test_df["id"], test_df["num_choices"].astype(int)))
    rows = []
    for tid in test_ids:
        ans = int(predictions[tid])
        nc = nc_by_id[tid]
        if ans < 0 or ans >= nc:
            raise SubmissionError(f"id={tid} answer={ans} out of range [0, {nc})")
        rows.append((tid, ans))

    df = pd.DataFrame(rows, columns=["id", "answer"])
    df.to_csv(out, index=False)
    return out
