"""Majority-vote ensemble across multiple submission CSVs.

For each test id, take the modal answer across input submissions.
On ties, fall back to the FIRST input file's answer (i.e. the
priority candidate).

Usage:
  python scripts/majority_vote.py \
    --inputs /path/to/submission_A.csv /path/to/submission_B.csv ... \
    --out /path/to/output_dir/

Writes:
  - submission_majority.csv
  - majority_report.json (per-input agreement, n_votes_at_max)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="submission CSVs to vote on (priority first)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(Path(args.data_dir) / "test.csv")
    test_df["num_choices"] = test_df["num_choices"].astype(int)
    test_ids = test_df["id"].astype(str).tolist()

    inputs = []
    for p in args.inputs:
        df = pd.read_csv(p)
        if list(df.columns) != ["id", "answer"]:
            raise SystemExit(f"{p}: expected columns ['id','answer'], got {list(df.columns)}")
        m = dict(zip(df["id"].astype(str), df["answer"].astype(int)))
        if set(m.keys()) != set(test_ids):
            raise SystemExit(f"{p}: id set differs from test.csv")
        inputs.append((p, m))

    nc_by_id = dict(zip(test_df["id"].astype(str), test_df["num_choices"]))

    rows = []
    n_unanimous = 0
    n_tie = 0
    pairwise_agree = {p: 0 for p, _ in inputs}
    priority_used_on_tie = 0
    for tid in test_ids:
        votes = [m[tid] for _, m in inputs]
        ctr = Counter(votes)
        top, top_n = ctr.most_common(1)[0]
        ties = [a for a, c in ctr.items() if c == top_n]
        if len(ctr) == 1:
            n_unanimous += 1
        if len(ties) > 1:
            n_tie += 1
            top = inputs[0][1][tid]  # priority candidate's vote
            priority_used_on_tie += 1
        if top < 0 or top >= nc_by_id[tid]:
            raise SystemExit(f"id={tid} ensemble pred={top} out of range for num_choices={nc_by_id[tid]}")
        rows.append((tid, int(top)))
        for p, m in inputs:
            if m[tid] == top:
                pairwise_agree[p] += 1

    sub = pd.DataFrame(rows, columns=["id", "answer"])
    sub_path = out_dir / "submission_majority.csv"
    sub.to_csv(sub_path, index=False)

    report = {
        "n_inputs": len(inputs),
        "n_test_rows": len(test_ids),
        "n_unanimous": n_unanimous,
        "n_tie": n_tie,
        "priority_used_on_tie": priority_used_on_tie,
        "agree_with_majority": pairwise_agree,
        "answer_distribution": Counter(int(a) for a in sub["answer"]),
        "submission_path": str(sub_path),
    }
    # Counter -> dict for JSON
    report["answer_distribution"] = {int(k): int(v) for k, v in report["answer_distribution"].items()}
    (out_dir / "majority_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
