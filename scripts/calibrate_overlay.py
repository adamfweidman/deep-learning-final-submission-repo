"""Validation-fitted calibration on top of a base submission.

Two strategies, evaluated independently and combined on val. Apply
to test only if val improves over the base.

  Strategy A (per-letter additive bias):
    Fit `delta[A..E]` = -log frequency of each letter being CORRECT
    on val. At inference, add delta[i] to logit[i] before argmax.
    Approximate, but with 1048 val rows and 4-5 free parameters it is
    well-regularized.

  Strategy B (per-num_choices additive bias):
    Same but conditioned on num_choices ∈ {2,3,4,5}. 4×5 = 20 params,
    still small.

  Combination: pick whichever yields higher val accuracy. Both are
  cheap to evaluate exhaustively over a tiny grid of damping factors.

Usage:
  python scripts/calibrate_overlay.py \
    --val-scores /scratch/.../runs/.../val_scores.json \
    --test-scores /scratch/.../runs/.../test_scores.json \
    --base-submission /scratch/.../runs/.../submission.csv \
    --out /scratch/.../runs/<attempt-id>/

The script reads val_scores.json (which has gold labels), fits the
calibration, computes overlay-val_acc, applies to test, writes
calibration_report.json + submission_calibrated.csv.

Gates: calibrated val_acc >= base val_acc (no regression). If gate
fails, no test submission is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHOICE_LETTERS = "ABCDEFGHIJ"


def _load_scores(path: Path) -> list[dict]:
    return json.loads(path.read_text())["records"]


def _scores_to_array(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (logprobs[N×5], num_choices[N], ids[N])."""
    N = len(records)
    lp = np.full((N, 5), -np.inf, dtype=np.float64)
    nc = np.zeros(N, dtype=np.int64)
    ids: list[str] = []
    for i, r in enumerate(records):
        for j, s in enumerate(r["scores"]):
            if s is not None:
                lp[i, j] = float(s)
        nc[i] = int(r["num_choices"])
        ids.append(str(r["id"]))
    return lp, nc, ids


def _argmax_with_mask(lp: np.ndarray, nc: np.ndarray) -> np.ndarray:
    """Argmax restricted to first nc[i] candidates per row."""
    out = np.zeros(len(lp), dtype=np.int64)
    for i in range(len(lp)):
        out[i] = int(np.argmax(lp[i, : nc[i]]))
    return out


def _accuracy(preds: np.ndarray, gold: np.ndarray) -> float:
    return float((preds == gold).mean())


def fit_per_letter_bias(val_lp: np.ndarray, val_nc: np.ndarray, val_gold: np.ndarray, *,
                       grid: list[float]) -> tuple[float, np.ndarray, float]:
    """Try per-letter additive bias `delta[i] * scale` for scale in grid.

    delta is the negative log of the empirical letter-prediction
    frequency (so adding delta de-emphasizes over-predicted letters).

    Returns (best_scale, best_delta, best_val_acc).
    """
    base_pred = _argmax_with_mask(val_lp, val_nc)
    pred_freq = np.bincount(base_pred, minlength=5).astype(np.float64)
    pred_freq = pred_freq / pred_freq.sum()
    # Smoothing to avoid log 0.
    delta = -np.log(np.maximum(pred_freq, 1e-3))
    delta = delta - delta.mean()  # mean-center; absolute scale absorbed into `scale`.

    best_scale, best_acc = 0.0, _accuracy(base_pred, val_gold)
    for s in grid:
        adj = val_lp + s * delta[None, :]
        pred = _argmax_with_mask(adj, val_nc)
        acc = _accuracy(pred, val_gold)
        if acc > best_acc:
            best_scale, best_acc = s, acc
    return best_scale, delta, best_acc


def fit_per_nc_letter_bias(val_lp: np.ndarray, val_nc: np.ndarray, val_gold: np.ndarray, *,
                           grid: list[float]) -> tuple[float, dict, float]:
    """Per-num_choices conditional letter bias.

    Returns (best_scale, {nc: delta}, best_val_acc).
    """
    base_pred = _argmax_with_mask(val_lp, val_nc)
    deltas: dict[int, np.ndarray] = {}
    for n in [2, 3, 4, 5]:
        sel = (val_nc == n)
        if sel.sum() == 0:
            deltas[n] = np.zeros(5)
            continue
        nc_pred = base_pred[sel]
        f = np.bincount(nc_pred, minlength=5).astype(np.float64)
        f = f / max(1.0, f.sum())
        d = -np.log(np.maximum(f, 1e-3))
        d = d - d.mean()
        deltas[n] = d

    best_scale, best_acc = 0.0, _accuracy(base_pred, val_gold)
    for s in grid:
        adj = val_lp.copy()
        for n in [2, 3, 4, 5]:
            sel = (val_nc == n)
            if sel.any():
                adj[sel] = val_lp[sel] + s * deltas[n][None, :]
        pred = _argmax_with_mask(adj, val_nc)
        acc = _accuracy(pred, val_gold)
        if acc > best_acc:
            best_scale, best_acc = s, acc
    return best_scale, deltas, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-scores", required=True)
    ap.add_argument("--test-scores", required=True)
    ap.add_argument("--base-submission", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[calibrate] loading val + test score dumps", flush=True)
    val_recs = _load_scores(Path(args.val_scores))
    test_recs = _load_scores(Path(args.test_scores))
    val_lp, val_nc, val_ids = _scores_to_array(val_recs)
    test_lp, test_nc, test_ids = _scores_to_array(test_recs)
    val_gold = np.array([int(r["gold"]) for r in val_recs])

    base_pred = _argmax_with_mask(val_lp, val_nc)
    base_val_acc = _accuracy(base_pred, val_gold)
    print(f"[calibrate] base val_acc = {base_val_acc:.4f} (n={len(val_lp)})", flush=True)

    grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    print("[calibrate] strategy A: per-letter additive bias", flush=True)
    sA, deltaA, accA = fit_per_letter_bias(val_lp, val_nc, val_gold, grid=grid)
    print(f"  best scale={sA}, best val_acc={accA:.4f}, delta={np.round(deltaA, 3).tolist()}", flush=True)

    print("[calibrate] strategy B: per-num_choices letter bias", flush=True)
    sB, deltasB, accB = fit_per_nc_letter_bias(val_lp, val_nc, val_gold, grid=grid)
    print(f"  best scale={sB}, best val_acc={accB:.4f}", flush=True)

    if accB > accA:
        winner, scale, val_acc = "per_nc", sB, accB
        deltas = {int(n): d.tolist() for n, d in deltasB.items()}
    else:
        winner, scale, val_acc = "per_letter", sA, accA
        deltas = {"all": deltaA.tolist()}

    print(f"[calibrate] winner: {winner}, val_acc={val_acc:.4f} (Δ={(val_acc - base_val_acc)*100:+.2f} pp)", flush=True)

    if val_acc < base_val_acc:
        print("[calibrate] FAIL: calibrated val_acc < base val_acc; not writing test submission", flush=True)
        report = {
            "base_val_acc": base_val_acc,
            "calibrated_val_acc": val_acc,
            "delta_pp": (val_acc - base_val_acc) * 100,
            "winner": winner,
            "scale": scale,
            "deltas": deltas,
            "gates_pass": False,
        }
        (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2))
        return 1

    # Apply to test
    test_pred_base = _argmax_with_mask(test_lp, test_nc)
    if winner == "per_letter":
        test_lp_cal = test_lp + scale * np.asarray(deltas["all"])[None, :]
    else:
        test_lp_cal = test_lp.copy()
        for n in [2, 3, 4, 5]:
            sel = (test_nc == n)
            if sel.any():
                test_lp_cal[sel] = test_lp[sel] + scale * np.asarray(deltas[n])[None, :]
    test_pred_cal = _argmax_with_mask(test_lp_cal, test_nc)

    overrides = int((test_pred_cal != test_pred_base).sum())

    # Build submission
    test_df = pd.read_csv(data_dir / "test.csv")
    pred_map = {tid: int(p) for tid, p in zip(test_ids, test_pred_cal.tolist())}
    rows = [(str(i), pred_map[str(i)]) for i in test_df["id"]]
    sub = pd.DataFrame(rows, columns=["id", "answer"])
    # Validate range
    test_df["num_choices"] = test_df["num_choices"].astype(int)
    nc_by_id = dict(zip(test_df["id"].astype(str), test_df["num_choices"]))
    for _, r in sub.iterrows():
        nc = nc_by_id[str(r["id"])]
        if not (0 <= int(r["answer"]) < nc):
            raise SystemExit(f"id={r['id']} answer={r['answer']} out of range for num_choices={nc}")
    sub_path = out_dir / "submission_calibrated.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[calibrate] wrote {sub_path}", flush=True)

    # Also write calibrated val_scores (preds replaced with calibrated
    # argmax) so downstream retrieval overlay can chain on top.
    if winner == "per_letter":
        val_lp_cal = val_lp + scale * np.asarray(deltas["all"])[None, :]
    else:
        val_lp_cal = val_lp.copy()
        for n in [2, 3, 4, 5]:
            sel = (val_nc == n)
            if sel.any():
                val_lp_cal[sel] = val_lp[sel] + scale * np.asarray(deltas[n])[None, :]
    val_pred_cal = _argmax_with_mask(val_lp_cal, val_nc)
    val_records_cal = []
    for i, r in enumerate(val_recs):
        rec = dict(r)
        rec["pred"] = int(val_pred_cal[i])
        val_records_cal.append(rec)
    vsc_path = out_dir / "val_scores_calibrated.json"
    vsc_path.write_text(json.dumps({"records": val_records_cal}))
    print(f"[calibrate] wrote {vsc_path} (for chaining retrieval overlay)", flush=True)

    report = {
        "base_val_acc": base_val_acc,
        "calibrated_val_acc": val_acc,
        "delta_pp": (val_acc - base_val_acc) * 100,
        "winner": winner,
        "scale": scale,
        "deltas": deltas,
        "test_overrides_vs_base": overrides,
        "submission_path": str(sub_path),
        "gates_pass": True,
    }
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
