"""LoRA fine-tuning of SmolVLM-500M-Instruct on the choice-letter task.

Strategy:

- Vision tower frozen. LoRA on all 7 Llama target modules of the
  language tower (q/k/v/o + gate/up/down). With ``r=8`` and the
  500M's `text_config` (hidden 960, intermediate 2560, 32 layers),
  trainable params ≈ 4.34M, comfortably under the 5M competition cap.
- Loss: standard causal-LM cross-entropy with labels masked to a
  single position — the answer-letter token (`` A``, `` B``, …) at the
  end of the prompt. Everything else is `-100`.
- Optimizer: AdamW + cosine LR with linear warmup. Mixed precision
  via bf16 autocast on l40s.

Inference at the end of training reuses
``src.zero_shot.predict_zero_shot`` so the val/test path is identical
to the zero-shot floor, only the model weights differ.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

from src.data import load_split
from src.metrics import accuracy
from src.prompt import CHOICE_LETTERS, build_prompt
from src.submission import write_submission
from src.zero_shot import _candidate_token_ids, _load_image, predict_zero_shot


@dataclass
class TrainReport:
    attempt_id: str
    n_train: int
    n_val_eval: int
    n_test: int
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    lora_r: int
    lora_alpha: int
    target_modules: list[str]
    trainable_params: int
    total_params: int
    final_train_loss: float
    val_accuracy: float
    best_val_accuracy: float
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    epoch_history: list[dict]
    submission_path: str | None
    adapter_dir: str
    best_adapter_dir: str
    device: str
    dtype: str


class _TrainDataset:
    """Worker-side: load PIL image + build prompt + grab target id.

    Each item is small (string + PIL.Image + int), so the DataLoader
    overhead is negligible. The processor (image_processor + tokenizer)
    runs on the main thread inside the training step; that step still
    pays a CPU cost but GPU forward/backward overlaps with worker
    prefetch, which is the main lever vs the all-main-thread loop
    that triggered the GPU-low-util kill on job 8055893.

    If ``choice_permute`` is True, each ``__getitem__`` returns a
    randomly-permuted choice ordering with the answer index updated
    accordingly. This is both a regularizer (the model can't learn
    "A is most often correct") and a free ~factorial-augmentation of
    the training set — for a 4-choice problem each row appears in
    4! = 24 distinct orderings across epochs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        data_dir: Path,
        img_size: int,
        cand_ids: list[int],
        choice_permute: bool = False,
        include_lecture: bool = True,
        include_hint: bool = True,
        include_solution: bool = False,
        use_chat_template: bool = False,
        processor=None,
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.img_size = img_size
        self.cand_ids = cand_ids
        self.choice_permute = choice_permute
        self.include_lecture = include_lecture
        self.include_hint = include_hint
        self.include_solution = include_solution
        self.use_chat_template = use_chat_template
        self.processor = processor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        import random as _r

        row = self.df.iloc[i]
        if self.choice_permute:
            choices = list(row["choices"])
            old_ans = int(row["answer"])
            n = len(choices)
            perm = list(range(n))
            _r.shuffle(perm)
            new_choices = [choices[k] for k in perm]
            new_ans = perm.index(old_ans)
            row = row.copy()
            row["choices"] = new_choices
            row["answer"] = new_ans
        if self.use_chat_template and self.processor is not None:
            from src.prompt import build_chat_prompt
            prompt = build_chat_prompt(
                self.processor, row, include_answer=True,
                include_lecture=self.include_lecture,
                include_hint=self.include_hint,
                include_solution=self.include_solution,
            )
        else:
            prompt = build_prompt(
                row, include_answer=True,
                include_lecture=self.include_lecture,
                include_hint=self.include_hint,
                include_solution=self.include_solution,
            )
        image = _load_image(self.data_dir / row["image_path"], self.img_size)
        target_id = self.cand_ids[int(row["answer"])]
        return prompt, image, target_id


def _train_collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


def _build_inputs_with_labels(processor, prompts, images, target_ids, device):
    """Mask labels so the loss is on a single token: the answer letter.

    For the manual prompt the answer letter IS the last non-pad
    token (the prompt ends with `Answer: A`). For the chat template
    the answer letter is followed by `<end_of_utterance>`, so we
    search backward for the actual cand_id token. Falls back to
    last_pos if the search fails (defensive).
    """
    import torch

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attn = inputs["attention_mask"]
    labels = torch.full_like(input_ids, -100)
    last_pos = attn.sum(dim=1) - 1
    for b in range(len(prompts)):
        target_id = int(target_ids[b])
        last = int(last_pos[b].item())
        # Prefer the explicit cand_id position so the chat-template
        # case (where last is `<end_of_utterance>`) doesn't put the
        # label on the EOS token.
        pos = last
        for k in range(last, max(-1, last - 6), -1):
            if int(input_ids[b, k].item()) == target_id:
                pos = k
                break
        labels[b, pos] = target_id
    return inputs, labels


def _set_offline_env(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def train_lora(
    *,
    attempt_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_dir: Path,
    model_id: str,
    cache_dir: str | Path,
    out_dir: Path,
    dtype_str: str = "bfloat16",
    n_train: int | None = None,
    n_val_eval: int | None = None,
    n_test: int | None = None,
    img_size: int = 224,
    batch_size: int = 4,
    grad_accum: int = 2,
    epochs: int = 2,
    lr: float = 2e-4,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.0,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
    target_modules_regex: str = "",
    unfreeze_modules: tuple[str, ...] = (),
    use_dora: bool = False,
    use_rslora: bool = False,
    choice_permute: bool = False,
    include_lecture: bool = True,
    include_hint: bool = True,
    include_solution_train: bool = False,
    use_chat_template: bool = False,
    use_val_for_training: bool = False,
    eval_batch_size: int = 48,
    eval_num_workers: int = 4,
    train_num_workers: int = 4,
    train_prefetch_factor: int = 2,
    seed: int = 42,
    log_every: int = 10,
    eval_every_epoch: bool = True,
    early_stop_patience: int = 1,
    write_submission_csv: bool = True,
    submission_threshold: float = 0.5,
    resume_from: str = "",
) -> TrainReport:
    _set_offline_env(Path(cache_dir))

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForVision2Seq, AutoProcessor

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_str, torch.bfloat16)

    print(f"[train] loading processor + model ({model_id})", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    device = next(model.parameters()).device

    # Freeze everything; LoRA will mark its own modules trainable.
    for p in model.parameters():
        p.requires_grad_(False)

    target_modules_arg = target_modules_regex if target_modules_regex else list(target_modules)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules_arg,
        use_dora=use_dora,
        use_rslora=use_rslora,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Optional: unfreeze additional named parameters (e.g. the
    # cross-modal connector). PEFT's modules_to_save would do this
    # too, but a substring match is simpler when we don't know the
    # exact module path; the runtime cap-assert catches anything
    # over budget.
    if unfreeze_modules:
        added = 0
        unfrozen_param_count = 0
        for name, p in model.named_parameters():
            if any(pat in name for pat in unfreeze_modules):
                if not p.requires_grad:
                    p.requires_grad_(True)
                    added += 1
                    unfrozen_param_count += p.numel()
        print(
            f"[train] unfreeze_modules patterns={list(unfreeze_modules)} "
            f"added {added} param tensors ({unfrozen_param_count:,} params)",
            flush=True,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params > 5_000_000:
        raise RuntimeError(
            f"LoRA trainable params {trainable_params} exceed the 5M competition cap."
        )

    cand_ids = _candidate_token_ids(processor)

    # Seeded random subsets (replaces .head() per process rule).
    # Plain random sampling — for proxy-scale work the
    # num_choices stratification gain is small and the groupby-apply
    # was eating the num_choices column on some pandas versions.
    rng_seed = seed
    def _stratified_head(df, k):
        if k is None or k >= len(df):
            return df
        return df.sample(n=k, random_state=rng_seed).reset_index(drop=True)

    if use_val_for_training:
        # Final-fit on train+val (paper-scout iter-5 Q1: +1-2 pp). val
        # is exhausted as a hold-out; use the same epoch count picked
        # from the best train-only run.
        combined = pd.concat([train_df, val_df], ignore_index=True)
        train_slice = combined if n_train is None else _stratified_head(combined, int(n_train))
        # Eval still happens on val for sanity; just expect val_acc to
        # be very high (training leaked into eval) — this is for
        # final-fit, not hparam selection.
        val_slice = val_df if n_val_eval is None else _stratified_head(val_df, int(n_val_eval))
    else:
        train_slice = train_df if n_train is None else _stratified_head(train_df, int(n_train))
        val_slice = val_df if n_val_eval is None else _stratified_head(val_df, int(n_val_eval))
    test_slice = test_df if n_test is None else _stratified_head(test_df, int(n_test))

    train_ds = _TrainDataset(
        train_slice, data_dir=data_dir, img_size=img_size, cand_ids=cand_ids,
        choice_permute=choice_permute,
        include_lecture=include_lecture, include_hint=include_hint,
        include_solution=include_solution_train,
        use_chat_template=use_chat_template, processor=processor,
    )
    n = len(train_ds)
    steps_per_epoch = math.ceil(n / batch_size)
    update_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    total_update_steps = update_steps_per_epoch * epochs
    warmup_steps = max(1, int(total_update_steps * warmup_ratio))

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_update_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    print(
        f"[train] n={n} bs={batch_size} grad_accum={grad_accum} "
        f"steps/epoch={steps_per_epoch} updates/epoch={update_steps_per_epoch} "
        f"epochs={epochs} total_updates={total_update_steps} warmup={warmup_steps} "
        f"train_workers={train_num_workers} eval_workers={eval_num_workers}",
        flush=True,
    )

    model.train()
    final_loss = float("nan")
    update_step = 0
    optim.zero_grad(set_to_none=True)
    t_start = time.time()

    val_targets = [int(a) for a in val_slice["answer"]]

    best_val_acc = -1.0
    best_epoch = -1
    epochs_since_best = 0
    epoch_history: list[dict] = []
    start_epoch = 0
    stopped_early = False

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    best_adapter_dir = out_dir / "adapter_best"
    best_adapter_dir.mkdir(parents=True, exist_ok=True)
    latest_adapter_dir = out_dir / "adapter_latest"
    latest_adapter_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

    # Resume from explicit path or from in-place latest checkpoint.
    candidate_resume = resume_from or (str(ckpt_path) if ckpt_path.exists() else "")
    if candidate_resume and Path(candidate_resume).exists():
        print(f"[train] resuming from {candidate_resume}", flush=True)
        ckpt = torch.load(candidate_resume, map_location=device, weights_only=False)
        # Reload latest adapter weights into model
        adapter_to_reload = ckpt.get("adapter_dir") or str(latest_adapter_dir)
        if Path(adapter_to_reload).exists():
            try:
                model.load_adapter(adapter_to_reload, adapter_name="default", is_trainable=True)
                print(f"[train] adapter reloaded from {adapter_to_reload}", flush=True)
            except Exception as e:
                print(f"[train] WARN: adapter reload failed ({e}); continuing with fresh adapter", flush=True)
        optim.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        update_step = int(ckpt["update_step"])
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))
        best_epoch = int(ckpt.get("best_epoch", -1))
        epochs_since_best = int(ckpt.get("epochs_since_best", 0))
        epoch_history = list(ckpt.get("epoch_history", []))
        rng_torch = ckpt.get("rng_torch")
        if rng_torch is not None:
            torch.set_rng_state(rng_torch)
        rng_cuda = ckpt.get("rng_cuda")
        if rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_cuda)
        print(f"[train] resumed at start_epoch={start_epoch} update_step={update_step} "
              f"best_val_acc={best_val_acc:.4f}@ep{best_epoch}", flush=True)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        # Per-epoch reproducible shuffle + worker-friendly DataLoader.
        epoch_seed = seed + epoch
        gen = torch.Generator()
        gen.manual_seed(epoch_seed)
        loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_num_workers,
            prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
            collate_fn=_train_collate,
            persistent_workers=False,
            generator=gen,
        )

        for batch_idx, (prompts, images, target_ids) in enumerate(loader):
            inputs, labels = _build_inputs_with_labels(processor, prompts, images, target_ids, device)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                out = model(**inputs, labels=labels)
                loss = out.loss / grad_accum

            loss.backward()
            final_loss = float(loss.item() * grad_accum)

            is_last_in_epoch = (batch_idx + 1) == len(loader)
            if ((batch_idx + 1) % grad_accum) == 0 or is_last_in_epoch:
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                update_step += 1
                if log_every and update_step % log_every == 0:
                    lr_now = sched.get_last_lr()[0]
                    elapsed = time.time() - t_start
                    print(
                        f"[train] ep={epoch+1}/{epochs} upd={update_step}/{total_update_steps} "
                        f"loss={final_loss:.4f} lr={lr_now:.2e} t={elapsed:.0f}s",
                        flush=True,
                    )

        if eval_every_epoch:
            t_eval = time.time()
            model.eval()
            val_preds_ep = predict_zero_shot(
                val_slice, data_dir=data_dir, processor=processor, model=model,
                img_size=img_size, batch_size=eval_batch_size,
                progress_every=0, num_workers=eval_num_workers,
                include_lecture=include_lecture, include_hint=include_hint,
                use_chat_template=use_chat_template,
            )
            val_acc_ep = accuracy(val_preds_ep, val_targets)
            model.train()
            ep_time = time.time() - epoch_start
            eval_time = time.time() - t_eval
            print(
                f"[train] EPOCH {epoch+1}/{epochs} val_acc={val_acc_ep:.4f} "
                f"loss={final_loss:.4f} ep_t={ep_time:.0f}s eval_t={eval_time:.0f}s",
                flush=True,
            )
            epoch_history.append({
                "epoch": epoch + 1, "val_acc": val_acc_ep,
                "final_loss": final_loss, "epoch_time_s": ep_time,
            })
            if val_acc_ep > best_val_acc:
                best_val_acc = val_acc_ep
                best_epoch = epoch + 1
                epochs_since_best = 0
                model.save_pretrained(str(best_adapter_dir))
                print(f"[train] new best val_acc={val_acc_ep:.4f}; saved adapter to {best_adapter_dir}", flush=True)
            else:
                epochs_since_best += 1

        # Per-epoch checkpoint (always: covers the no-eval branch too).
        # Saving the latest adapter + optimizer/scheduler/rng is enough
        # to resume from the next epoch if the job is killed mid-run.
        model.save_pretrained(str(latest_adapter_dir))
        torch.save({
            "epoch": epoch,
            "update_step": update_step,
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "epochs_since_best": epochs_since_best,
            "epoch_history": epoch_history,
            "adapter_dir": str(latest_adapter_dir),
        }, ckpt_path)

        if eval_every_epoch and epochs_since_best >= early_stop_patience and best_val_acc >= 0:
            print(
                f"[train] early stop: no val_acc improvement in {epochs_since_best} epoch(s); "
                f"best={best_val_acc:.4f} at epoch {best_epoch}",
                flush=True,
            )
            stopped_early = True
            break

    # Save final adapter (last epoch's weights, regardless of best)
    model.save_pretrained(str(adapter_dir))
    print(f"[train] final adapter saved to {adapter_dir}", flush=True)

    # Decide which adapter wins. If we have epoch history, load best
    # adapter back; if best_val_acc was set at the last epoch, model
    # is already at best. Otherwise reload best.
    if eval_every_epoch and best_epoch > 0 and best_epoch != (len(epoch_history)):
        from peft import PeftModel
        # PEFT replaces in-place via load_adapter when present.
        try:
            model.load_adapter(str(best_adapter_dir), adapter_name="default", is_trainable=False)
            print(f"[train] reloaded best adapter (epoch {best_epoch})", flush=True)
        except Exception as e:
            print(f"[train] WARN: failed to reload best adapter: {e}", flush=True)

    # Final eval (val) to record under val_accuracy in the report.
    model.eval()
    val_preds = predict_zero_shot(
        val_slice, data_dir=data_dir, processor=processor, model=model,
        img_size=img_size, batch_size=eval_batch_size, num_workers=eval_num_workers,
        include_lecture=include_lecture, include_hint=include_hint,
        use_chat_template=use_chat_template,
    )
    val_acc = accuracy(val_preds, val_targets)
    print(f"[train] FINAL val_accuracy={val_acc:.4f} (n={len(val_slice)})", flush=True)

    # Test eval only after val signal exists (zero-shot floor was 0.558;
    # don't waste ~3 min on a submission that won't beat it).
    submission_path: str | None = None
    if write_submission_csv and n_test is None and len(test_slice) > 0:
        if val_acc < submission_threshold:
            print(
                f"[train] skip test eval / submission: val_acc={val_acc:.4f} "
                f"< submission_threshold={submission_threshold:.4f}; "
                f"saved adapters can be reloaded later via mode=infer",
                flush=True,
            )
        else:
            test_preds = predict_zero_shot(
                test_slice, data_dir=data_dir, processor=processor, model=model,
                img_size=img_size, batch_size=eval_batch_size, num_workers=eval_num_workers,
                include_lecture=include_lecture, include_hint=include_hint,
                use_chat_template=use_chat_template,
            )
            pred_map = dict(zip(test_slice["id"], test_preds))
            submission_path = str(write_submission(pred_map, test_df, out_dir / "submission.csv"))
            print(f"[train] submission written: {submission_path}", flush=True)

    report = TrainReport(
        attempt_id=attempt_id,
        n_train=len(train_slice),
        n_val_eval=len(val_slice),
        n_test=len(test_slice),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        trainable_params=int(trainable_params),
        total_params=int(total_params),
        final_train_loss=final_loss,
        val_accuracy=float(val_acc),
        best_val_accuracy=float(best_val_acc if best_val_acc >= 0 else val_acc),
        best_epoch=int(best_epoch),
        epochs_completed=int(len(epoch_history) if eval_every_epoch else epochs),
        stopped_early=bool(stopped_early),
        epoch_history=epoch_history,
        submission_path=submission_path,
        adapter_dir=str(adapter_dir),
        best_adapter_dir=str(best_adapter_dir),
        device=str(device),
        dtype=dtype_str,
    )

    (out_dir / "train_report.json").write_text(json.dumps(asdict(report), indent=2))
    return report
