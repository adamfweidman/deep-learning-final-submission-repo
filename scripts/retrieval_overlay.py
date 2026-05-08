"""Apply pHash + question-similarity direct-copy overlay on top of a
base submission (e.g. V_β's submission.csv).

Usage:
  python scripts/retrieval_overlay.py \
    --base-submission /scratch/.../runs/2026-05-06-infer-vb-best/submission.csv \
    --base-val-scores /scratch/.../runs/2026-05-06-infer-vb-best/val_scores.json \
    --hamming-thresh 2 --qsim-thresh 0.85 --require-choice-match \
    --out /scratch/.../runs/2026-05-06-retrieval-overlay/

The script:
  1. Reads `data/{train,val,test}.csv`.
  2. Computes/caches pHashes under data/phash_cache/.
  3. Loads base submission (test-side predictions).
  4. Reconstructs base val predictions from a score dump or from
     `--base-val-preds` (a CSV).
  5. Applies retrieval overlay on val, prints tier breakdown,
     compares vs base val accuracy.
  6. Applies retrieval on test if val gates pass; writes
     `submission_retrieval.csv` and a JSON report.

Gates (pass to write test submission unless `--force`):
  - combined val acc must not regress > 0.5 pp vs base val acc.
  - direct_copy slice acc must be >= 0.80 (when n_direct_copy >= 10).
  - direct_copy slice acc must beat the BASE model's accuracy on
    that same slice (project rule from the iter-5 brief).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root so ``src.*`` imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval import (
    OverlayResult,
    apply_retrieval,
    get_or_compute_hashes,
    tier_breakdown,
)


def _load_split(p):
    df = pd.read_csv(p)
    df["choices"] = df["choices"].apply(json.loads)
    df["num_choices"] = df["num_choices"].astype(int)
    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(int, errors="ignore")
    return df


def _load_val_preds_from_scores(score_path: Path) -> dict[str, int]:
    payload = json.loads(score_path.read_text())
    return {rec["id"]: int(rec["pred"]) for rec in payload["records"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base-submission", required=True, help="path to base submission.csv (test-side)")
    ap.add_argument("--base-val-scores", help="path to val_scores.json (preferred)")
    ap.add_argument("--base-val-preds", help="alt: CSV with id,answer columns for val predictions")
    ap.add_argument("--out", required=True, help="output dir for retrieval submission + report")
    ap.add_argument("--phash-cache-dir", default=None,
                    help="default: <data-dir>/phash_cache")
    ap.add_argument("--hamming-thresh", type=int, default=2)
    ap.add_argument("--qsim-thresh", type=float, default=0.85)
    ap.add_argument("--require-choice-match", action="store_true", default=True)
    ap.add_argument("--no-require-choice-match", dest="require_choice_match", action="store_false")
    ap.add_argument("--gate-tol-pp", type=float, default=0.5)
    ap.add_argument("--tier1-min-acc", type=float, default=0.80)
    ap.add_argument("--force", action="store_true", help="write submission even if gates fail")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.phash_cache_dir) if args.phash_cache_dir else (data_dir / "phash_cache")

    print("[retrieval] loading splits", flush=True)
    train_df = _load_split(data_dir / "train.csv")
    val_df = _load_split(data_dir / "val.csv")
    test_df = _load_split(data_dir / "test.csv")
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)

    print("[retrieval] computing/loading pHashes", flush=True)
    train_hashes = get_or_compute_hashes(train_df, data_dir, "train", cache_dir=cache_dir)
    val_hashes = get_or_compute_hashes(val_df, data_dir, "val", cache_dir=cache_dir)
    test_hashes = get_or_compute_hashes(test_df, data_dir, "test", cache_dir=cache_dir)

    # Base test predictions
    base_test = pd.read_csv(args.base_submission)
    if list(base_test.columns) != ["id", "answer"]:
        raise SystemExit(f"base submission has unexpected columns {list(base_test.columns)}")
    base_test_pred_map = dict(zip(base_test["id"].astype(str), base_test["answer"].astype(int)))
    base_test_preds = [int(base_test_pred_map[str(i)]) for i in test_df["id"]]

    # Base val predictions
    if args.base_val_scores:
        base_val_pred_map = _load_val_preds_from_scores(Path(args.base_val_scores))
    elif args.base_val_preds:
        df = pd.read_csv(args.base_val_preds)
        base_val_pred_map = dict(zip(df["id"].astype(str), df["answer"].astype(int)))
    else:
        raise SystemExit("Need either --base-val-scores or --base-val-preds")
    base_val_preds = [int(base_val_pred_map[str(i)]) for i in val_df["id"]]

    val_gold = [int(a) for a in val_df["answer"]]
    base_val_acc = sum(int(p == g) for p, g in zip(base_val_preds, val_gold)) / len(val_df)
    print(f"[retrieval] base val_acc = {base_val_acc:.4f} (n={len(val_df)})", flush=True)

    print("[retrieval] applying overlay on val", flush=True)
    val_res = apply_retrieval(
        val_df, val_hashes, base_val_preds,
        train_df=train_df, train_hashes=train_hashes,
        hamming_thresh=args.hamming_thresh, qsim_thresh=args.qsim_thresh,
        require_choice_match=args.require_choice_match,
    )
    val_brk = tier_breakdown(val_res, gold=val_gold)
    overlay_val_acc = val_brk["overall_overlay_acc"]
    delta_pp = (overlay_val_acc - base_val_acc) * 100
    print(f"[retrieval] overlay val_acc = {overlay_val_acc:.4f}  Δ = {delta_pp:+.2f} pp", flush=True)
    print(f"[retrieval] val tier counts: {val_brk['tier_counts']}", flush=True)
    print(f"[retrieval] val tier overlay acc: {val_brk['tier_overlay_acc']}", flush=True)

    # Slice-specific base accuracy: how does the BASE model do on
    # exactly the rows we'd direct-copy? (project rule: direct_copy
    # must beat V_β on its own slice, not just clear 0.80.)
    direct_idx = [i for i, t in enumerate(val_res.tiers) if t == "direct_copy"]
    n_dc = len(direct_idx)
    if n_dc:
        dc_overlay = sum(int(val_res.preds[i] == val_gold[i]) for i in direct_idx) / n_dc
        dc_base    = sum(int(base_val_preds[i] == val_gold[i]) for i in direct_idx) / n_dc
        print(f"[retrieval] direct_copy slice n={n_dc}: overlay={dc_overlay:.4f} vs base={dc_base:.4f} (Δ={(dc_overlay-dc_base)*100:+.2f} pp)", flush=True)
    else:
        dc_overlay = dc_base = 0.0

    # Gate
    gate_pass = True
    gate_msgs = []
    if delta_pp < -args.gate_tol_pp:
        gate_pass = False
        gate_msgs.append(f"combined Δ {delta_pp:+.2f} pp < tol {-args.gate_tol_pp:.2f}")
    if n_dc >= 10:
        if dc_overlay < args.tier1_min_acc:
            gate_pass = False
            gate_msgs.append(f"direct_copy acc {dc_overlay:.3f} < {args.tier1_min_acc}")
        if dc_overlay <= dc_base:
            gate_pass = False
            gate_msgs.append(f"direct_copy overlay {dc_overlay:.3f} ≤ base {dc_base:.3f} on same slice")
    else:
        gate_msgs.append(f"direct_copy n={n_dc} < 10 (skipping accuracy gate)")

    print(f"[retrieval] GATES: {'PASS' if gate_pass else 'FAIL'}  ({'; '.join(gate_msgs) or 'none'})", flush=True)

    report = {
        "base_submission": str(args.base_submission),
        "hparams": {
            "hamming_thresh": args.hamming_thresh,
            "qsim_thresh": args.qsim_thresh,
            "require_choice_match": args.require_choice_match,
        },
        "base_val_acc": base_val_acc,
        "overlay_val_acc": overlay_val_acc,
        "delta_pp": delta_pp,
        "val": val_brk,
        "direct_copy_slice": {
            "n": n_dc,
            "overlay_acc": dc_overlay,
            "base_acc": dc_base,
            "overlay_minus_base_pp": (dc_overlay - dc_base) * 100,
        },
        "gates_pass": gate_pass,
        "gate_msgs": gate_msgs,
    }

    if not gate_pass and not args.force:
        print("[retrieval] gates failed; NOT writing test submission. Use --force to override.", flush=True)
        report_path = out_dir / "retrieval_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"[retrieval] report -> {report_path}", flush=True)
        return 1

    print("[retrieval] applying overlay on test", flush=True)
    test_res = apply_retrieval(
        test_df, test_hashes, base_test_preds,
        train_df=train_df, train_hashes=train_hashes,
        hamming_thresh=args.hamming_thresh, qsim_thresh=args.qsim_thresh,
        require_choice_match=args.require_choice_match,
    )
    test_brk = tier_breakdown(test_res)

    sub = pd.DataFrame({
        "id": test_df["id"].astype(str),
        "answer": pd.Series(test_res.preds, dtype=int),
    })
    # Validate range and uniqueness.
    for i, row in test_df.iterrows():
        if not (0 <= int(test_res.preds[i]) < int(row["num_choices"])):
            raise SystemExit(f"row {i} (id={row['id']}): pred {test_res.preds[i]} out of range for num_choices={row['num_choices']}")
    sub_path = out_dir / "submission_retrieval.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[retrieval] wrote {sub_path}", flush=True)
    print(f"[retrieval] test tier counts: {test_brk['tier_counts']}", flush=True)
    overrides = sum(1 for i, t in enumerate(test_res.tiers) if t == "direct_copy" and test_res.preds[i] != base_test_preds[i])
    print(f"[retrieval] test direct_copy overrides vs base: {overrides}", flush=True)

    # Distribution before/after
    before = pd.Series(base_test_preds).value_counts().sort_index().to_dict()
    after = pd.Series(test_res.preds).value_counts().sort_index().to_dict()
    report.update({
        "test": test_brk,
        "test_distribution_before": {int(k): int(v) for k, v in before.items()},
        "test_distribution_after": {int(k): int(v) for k, v in after.items()},
        "test_overrides_vs_base": overrides,
        "submission_path": str(sub_path),
    })
    (out_dir / "retrieval_report.json").write_text(json.dumps(report, indent=2))
    print(f"[retrieval] report -> {out_dir / 'retrieval_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
