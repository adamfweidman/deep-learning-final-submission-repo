"""pHash + question-similarity + choice-match retrieval overlay.

Ported from override.ipynb. The intended use is as a late-stage
overlay on top of a base submission produced by an MCF-trained
adapter:

  1. Compute pHash of every train, val, test image.
  2. For each val/test row, find tightest train neighbor by Hamming
     distance.
  3. If Hamming ≤ ``hamming_thresh`` AND question text similarity
     (difflib SequenceMatcher) ≥ ``qsim_thresh`` AND
     choice sets match (when ``require_choice_match=True``):
     direct-copy the train neighbor's answer (remapped to test row's
     choice ordering).
  4. Otherwise, fall back to the base submission's prediction.

Validate on val first: combined val accuracy must not regress vs the
base submission and the direct-copy slice should clear 80 % accuracy.

Hashes are pixel-derived (no model dependency); cache them under
``data/phash_cache/`` so subsequent runs skip the ~30-90 s hashing
step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import imagehash
import pandas as pd
from PIL import Image


def _load_split(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["choices"] = df["choices"].apply(json.loads)
    df["num_choices"] = df["num_choices"].astype(int)
    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(int, errors="ignore")
    return df


def _phash_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"{split}_phash.json"


def _hash_one(img_path: Path) -> imagehash.ImageHash:
    with Image.open(img_path) as im:
        return imagehash.phash(im.convert("RGB"))


def hashes_for_df(df: pd.DataFrame, data_dir: Path, *, label: str = "") -> list[imagehash.ImageHash]:
    out: list[imagehash.ImageHash] = []
    for i, p in enumerate(df["image_path"]):
        out.append(_hash_one(data_dir / p))
        if (i + 1) % 500 == 0 and label:
            print(f"  [{label}] {i+1}/{len(df)} hashed", flush=True)
    return out


def get_or_compute_hashes(
    df: pd.DataFrame, data_dir: Path, split: str, *, cache_dir: Path
) -> list[imagehash.ImageHash]:
    cache_path = _phash_path(cache_dir, split)
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if payload.get("image_paths") == df["image_path"].tolist():
                print(f"  [{split}] using cached hashes ({len(payload['hashes_hex'])}) from {cache_path}", flush=True)
                return [imagehash.hex_to_hash(h) for h in payload["hashes_hex"]]
            print(f"  [{split}] cache image_paths drift; recomputing", flush=True)
        except Exception as e:
            print(f"  [{split}] cache load error ({e}); recomputing", flush=True)
    print(f"  [{split}] computing pHashes for {len(df)} images...", flush=True)
    hashes = hashes_for_df(df, data_dir, label=split)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "image_paths": df["image_path"].tolist(),
        "hashes_hex": [str(h) for h in hashes],
    }))
    print(f"  [{split}] cached -> {cache_path}", flush=True)
    return hashes


@dataclass
class Match:
    train_idx: int
    hamming: int
    qsim: float
    choice_match: bool


def find_best_match(
    qhash: imagehash.ImageHash,
    qquestion: str,
    qchoices: list[str],
    *,
    train_hashes: list[imagehash.ImageHash],
    train_df: pd.DataFrame,
    hamming_thresh: int,
) -> Match | None:
    qchoices_set = set(qchoices)
    cands: list[Match] = []
    for ti, th in enumerate(train_hashes):
        h = th - qhash
        if h <= hamming_thresh:
            train_q = str(train_df.iloc[ti]["question"])
            qsim = SequenceMatcher(None, str(qquestion), train_q).ratio()
            cm = qchoices_set == set(train_df.iloc[ti]["choices"])
            cands.append(Match(ti, h, qsim, cm))
    if not cands:
        return None
    cands.sort(key=lambda m: (m.hamming, -m.qsim, 0 if m.choice_match else 1))
    return cands[0]


@dataclass
class OverlayResult:
    preds: list[int]
    tiers: list[str]            # 'direct_copy' | 'mcf_fallback' | 'mcf_no_match'
    info: list[Match | None]


def apply_retrieval(
    df: pd.DataFrame,
    df_hashes: list[imagehash.ImageHash],
    base_preds: list[int],
    *,
    train_df: pd.DataFrame,
    train_hashes: list[imagehash.ImageHash],
    hamming_thresh: int,
    qsim_thresh: float,
    require_choice_match: bool,
) -> OverlayResult:
    preds: list[int] = []
    tiers: list[str] = []
    info: list[Match | None] = []
    for i in range(len(df)):
        row = df.iloc[i]
        m = find_best_match(
            df_hashes[i], row["question"], row["choices"],
            train_hashes=train_hashes, train_df=train_df,
            hamming_thresh=hamming_thresh,
        )
        if m is None:
            preds.append(base_preds[i]); tiers.append("mcf_no_match"); info.append(None); continue
        gate = (
            m.hamming <= hamming_thresh
            and m.qsim >= qsim_thresh
            and (m.choice_match or not require_choice_match)
        )
        if gate:
            train_choices = train_df.iloc[m.train_idx]["choices"]
            train_ans_idx = int(train_df.iloc[m.train_idx]["answer"])
            train_ans_text = train_choices[train_ans_idx]
            if train_ans_text in row["choices"]:
                preds.append(int(row["choices"].index(train_ans_text)))
                tiers.append("direct_copy")
                info.append(m)
                continue
        preds.append(base_preds[i])
        tiers.append("mcf_fallback")
        info.append(m)
    return OverlayResult(preds=preds, tiers=tiers, info=info)


def tier_breakdown(result: OverlayResult, gold: list[int] | None = None) -> dict:
    counts = {"direct_copy": 0, "mcf_fallback": 0, "mcf_no_match": 0}
    overlay_correct = {k: 0 for k in counts}
    for i, t in enumerate(result.tiers):
        counts[t] += 1
        if gold is not None and result.preds[i] == gold[i]:
            overlay_correct[t] += 1
    out = {
        "n_total": len(result.preds),
        "tier_counts": counts,
        "tier_pct": {k: (v / max(1, len(result.preds))) for k, v in counts.items()},
    }
    if gold is not None:
        out["tier_overlay_acc"] = {
            k: (overlay_correct[k] / max(1, counts[k])) for k in counts
        }
        out["overall_overlay_acc"] = sum(int(p == g) for p, g in zip(result.preds, gold)) / max(1, len(gold))
    return out
