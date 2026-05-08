"""Project entrypoint.

Run with:

    python -m src.run --config configs/<name>.yaml

The YAML's ``run.mode`` selects the path:

- ``smoke``    — schema validation + image resolution + label-only
                 baseline on validation + sample-style submission.
- ``baseline`` — same as smoke but reports the baseline name explicitly
                 in the run summary; alias kept for clarity in the
                 journal.
- ``preflight`` — HF model cache preflight; downloads to ``/scratch``,
                  loads processor/model once, reports parameter counts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

# Avoid the HF tokenizers Rust-parallelism + DataLoader-fork deadlock
# observed on RunA smoke 8096599 (process idle at 0% CPU/GPU after 25
# min). Must be set before transformers imports.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from src.config import Config, load_config, resolved_output_dir
from src.data import load_split, validate_split
from src.metrics import accuracy
from src.submission import write_submission


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass


def _summary_path(out_dir: Path) -> Path:
    return out_dir / "run_summary.json"


def _write_resolved_config(cfg: Config, out_dir: Path) -> None:
    train_dict = asdict(cfg.train)
    # tuple → list so the JSON round-trip is identity
    train_dict["target_modules"] = list(train_dict["target_modules"])
    train_dict["unfreeze_modules"] = list(train_dict["unfreeze_modules"])
    payload = {
        "attempt_id": cfg.attempt_id,
        "level": cfg.level,
        "experiment": asdict(cfg.experiment),
        "model": asdict(cfg.model),
        "data": asdict(cfg.data),
        "run": asdict(cfg.run),
        "train": train_dict,
        "preflight": asdict(cfg.preflight),
        "resources": cfg.resources,
        "notes": cfg.notes,
    }
    (out_dir / "config.resolved.json").write_text(json.dumps(payload, indent=2))


def _run_smoke(cfg: Config, out_dir: Path) -> dict:
    from src.baselines import baseline_predictions

    data_dir = Path(cfg.data.data_dir)
    train_df = load_split(cfg.data.train_csv, labeled=True)
    val_df = load_split(cfg.data.val_csv, labeled=True)
    test_df = load_split(cfg.data.test_csv, labeled=False)

    n_check = cfg.data.n_examples
    reports = [
        validate_split(train_df, split="train", data_dir=data_dir, labeled=True, sample_check=n_check),
        validate_split(val_df, split="val", data_dir=data_dir, labeled=True, sample_check=n_check),
        validate_split(test_df, split="test", data_dir=data_dir, labeled=False, sample_check=n_check),
    ]
    for r in reports:
        if r.missing_columns:
            raise RuntimeError(f"{r.split}: missing required columns {r.missing_columns}")
        if r.images_missing > 0:
            raise RuntimeError(
                f"{r.split}: {r.images_missing} image paths failed to resolve under {data_dir}; "
                f"first missing: {r.sample_missing_paths}"
            )
        if r.answers_in_range is False:
            raise RuntimeError(f"{r.split}: at least one answer index outside [0, num_choices)")

    val_preds = baseline_predictions(cfg.run.baseline, val_df, train_df=train_df)
    val_acc = accuracy(val_preds, val_df["answer"].astype(int).tolist())

    test_preds = baseline_predictions(cfg.run.baseline, test_df, train_df=train_df)
    pred_map = dict(zip(test_df["id"], test_preds))
    submission_path = write_submission(pred_map, test_df, out_dir / "submission.csv")

    summary = {
        "mode": cfg.run.mode,
        "attempt_id": cfg.attempt_id,
        "baseline": cfg.run.baseline,
        "val_accuracy": val_acc,
        "submission_path": str(submission_path),
        "splits": [asdict(r) for r in reports],
    }
    return summary


def _run_zero_shot(cfg: Config, out_dir: Path) -> dict:
    from src.zero_shot import evaluate_and_predict

    data_dir = Path(cfg.data.data_dir)
    val_df = load_split(cfg.data.val_csv, labeled=True)
    test_df = load_split(cfg.data.test_csv, labeled=False)

    res = evaluate_and_predict(
        val_df,
        test_df,
        data_dir=data_dir,
        model_id=cfg.model.name,
        cache_dir=cfg.preflight.hf_cache_dir,
        dtype_str=cfg.model.dtype,
        n_val=cfg.run.n_val,
        n_test=cfg.run.n_test,
        img_size=cfg.run.img_size,
        batch_size=cfg.run.batch_size,
    )

    val_acc = accuracy(res["val_preds"], res["val_targets"])

    pred_map = dict(zip(res["test_ids"], res["test_preds"]))
    submission_path: str | None = None
    if cfg.run.n_test is None:
        submission_path = str(write_submission(pred_map, test_df, out_dir / "submission.csv"))

    summary = {
        "mode": "zero_shot",
        "attempt_id": cfg.attempt_id,
        "val_accuracy": val_acc,
        "n_val": res["n_val"],
        "n_test": res["n_test"],
        "submission_path": submission_path,
        "device": res["device"],
        "dtype": res["dtype"],
        "model_id": cfg.model.name,
    }
    return summary


def _run_infer(cfg: Config, out_dir: Path) -> dict:
    """Load model + LoRA adapter offline; run val + test inference; write submission."""
    import os

    cache_dir = Path(cfg.preflight.hf_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    from src.zero_shot import predict_zero_shot

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        cfg.model.dtype, torch.bfloat16
    )

    data_dir = Path(cfg.data.data_dir)
    val_df = load_split(cfg.data.val_csv, labeled=True)
    test_df = load_split(cfg.data.test_csv, labeled=False)

    processor = AutoProcessor.from_pretrained(cfg.model.name, local_files_only=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base = AutoModelForVision2Seq.from_pretrained(
        cfg.model.name, torch_dtype=dtype, local_files_only=True,
    )
    if cfg.run.adapter_path:
        model = PeftModel.from_pretrained(base, cfg.run.adapter_path, is_trainable=False)
        print(f"[infer] loaded adapter: {cfg.run.adapter_path}", flush=True)
    else:
        model = base
        print("[infer] no adapter_path provided; running base model", flush=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    bs = cfg.run.batch_size or 48
    nw = cfg.train.eval_num_workers

    # Score dump: capture per-row A-E log-probs + metadata so downstream
    # calibration/ensembling has everything it needs.
    val_preds, val_scores = predict_zero_shot(
        val_df, data_dir=data_dir, processor=processor, model=model,
        img_size=cfg.run.img_size, batch_size=bs, num_workers=nw,
        return_scores=True,
    )
    val_targets = [int(a) for a in val_df["answer"]]
    val_acc = accuracy(val_preds, val_targets)
    print(f"[infer] val_accuracy={val_acc:.4f} (n={len(val_df)})", flush=True)

    _write_score_dump(out_dir / "val_scores.json", val_df, val_scores, val_preds, with_gold=True)

    test_preds, test_scores = predict_zero_shot(
        test_df, data_dir=data_dir, processor=processor, model=model,
        img_size=cfg.run.img_size, batch_size=bs, num_workers=nw,
        return_scores=True,
    )
    _write_score_dump(out_dir / "test_scores.json", test_df, test_scores, test_preds, with_gold=False)

    pred_map = dict(zip(test_df["id"], test_preds))
    submission_path = str(write_submission(pred_map, test_df, out_dir / "submission.csv"))
    print(f"[infer] submission written: {submission_path}", flush=True)

    return {
        "mode": "infer",
        "attempt_id": cfg.attempt_id,
        "adapter_path": cfg.run.adapter_path,
        "val_accuracy": val_acc,
        "submission_path": submission_path,
        "val_scores_path": str(out_dir / "val_scores.json"),
        "test_scores_path": str(out_dir / "test_scores.json"),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": cfg.model.dtype,
    }


def _write_score_dump(path, df, scores, preds, *, with_gold: bool) -> None:
    """Serialize per-row scoring info for downstream calibration/ensembling.

    Schema (one record per row):
      id, num_choices, scores[A..E] (log-prob, -inf for unused),
      pred (int 0..n-1), gold (int or null),
      task, grade, subject, topic, category, skill (if present)
    """
    import json as _json

    metadata_cols = ["task", "grade", "subject", "topic", "category", "skill"]
    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        rec = {
            "id": str(row["id"]),
            "num_choices": int(row["num_choices"]),
            "scores": [(s if s != float("-inf") else None) for s in scores[i]],
            "pred": int(preds[i]),
        }
        if with_gold and "answer" in df.columns:
            rec["gold"] = int(row["answer"])
        for col in metadata_cols:
            if col in df.columns:
                v = row.get(col)
                rec[col] = None if (v is None or (isinstance(v, float) and v != v)) else str(v)
        records.append(rec)
    path.write_text(_json.dumps({"records": records}, indent=None))
    print(f"[infer] wrote score dump: {path}", flush=True)


def _run_train(cfg: Config, out_dir: Path) -> dict:
    from dataclasses import asdict as _asdict

    from src.training import train_lora

    data_dir = Path(cfg.data.data_dir)
    train_df = load_split(cfg.data.train_csv, labeled=True)
    val_df = load_split(cfg.data.val_csv, labeled=True)
    test_df = load_split(cfg.data.test_csv, labeled=False)

    report = train_lora(
        attempt_id=cfg.attempt_id,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        data_dir=data_dir,
        model_id=cfg.model.name,
        cache_dir=cfg.preflight.hf_cache_dir,
        out_dir=out_dir,
        dtype_str=cfg.model.dtype,
        n_train=cfg.train.n_train,
        n_val_eval=cfg.train.n_val_eval,
        n_test=cfg.train.n_test,
        img_size=cfg.run.img_size,
        batch_size=cfg.train.batch_size,
        grad_accum=cfg.train.grad_accum,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        lora_r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=tuple(cfg.train.target_modules),
        target_modules_regex=cfg.train.target_modules_regex,
        unfreeze_modules=tuple(cfg.train.unfreeze_modules),
        use_dora=cfg.train.use_dora,
        use_rslora=cfg.train.use_rslora,
        choice_permute=cfg.train.choice_permute,
        include_lecture=cfg.train.include_lecture,
        include_hint=cfg.train.include_hint,
        use_chat_template=cfg.train.use_chat_template,
        include_solution_train=cfg.train.include_solution_train,
        use_val_for_training=cfg.train.use_val_for_training,
        eval_batch_size=cfg.train.eval_batch_size,
        eval_num_workers=cfg.train.eval_num_workers,
        train_num_workers=cfg.train.train_num_workers,
        train_prefetch_factor=cfg.train.train_prefetch_factor,
        seed=cfg.run.seed,
        log_every=cfg.train.log_every,
        eval_every_epoch=cfg.train.eval_every_epoch,
        early_stop_patience=cfg.train.early_stop_patience,
        write_submission_csv=cfg.train.write_submission_csv,
        submission_threshold=cfg.train.submission_threshold,
        resume_from=cfg.train.resume_from,
    )
    return {"mode": "train", "attempt_id": cfg.attempt_id, **_asdict(report)}


def _run_preflight(cfg: Config, out_dir: Path) -> dict:
    from src.model_preflight import run_preflight

    report = run_preflight(
        cfg.model.name,
        cache_dir=cfg.preflight.hf_cache_dir,
        download_first=cfg.preflight.download_first,
        snapshot_only=cfg.preflight.snapshot_only,
        dtype_str=cfg.model.dtype,
        output_dir=out_dir,
    )
    return {"mode": "preflight", "attempt_id": cfg.attempt_id, **asdict(report)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="deep-learning-final entrypoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _seed_all(cfg.run.seed)

    out_dir = resolved_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(cfg, out_dir)

    mode = cfg.run.mode
    if mode in ("smoke", "baseline"):
        summary = _run_smoke(cfg, out_dir)
    elif mode == "preflight":
        summary = _run_preflight(cfg, out_dir)
    elif mode == "zero_shot":
        summary = _run_zero_shot(cfg, out_dir)
    elif mode == "train":
        summary = _run_train(cfg, out_dir)
    elif mode == "infer":
        summary = _run_infer(cfg, out_dir)
    else:
        raise SystemExit(f"unknown run.mode: {mode!r} (expected smoke|baseline|preflight|zero_shot|train|infer)")

    _summary_path(out_dir).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
