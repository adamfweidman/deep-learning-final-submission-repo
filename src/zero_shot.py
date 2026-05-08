"""Zero-shot multiple-choice scoring for SmolVLM.

For each example, run a single forward pass with the prompt ending in
``Answer:`` and read the logit at the next-token position. We compare
the log-prob of the leading-space letter token for each available
choice (`` A``, `` B``, …) and predict argmax.

This avoids generation drift seen in the starter notebook (the model
sometimes continues with unrelated text instead of a clean letter).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

from src.prompt import CHOICE_LETTERS, build_prompt


def _candidate_token_ids(processor) -> list[int]:
    tok = processor.tokenizer
    ids: list[int] = []
    for letter in CHOICE_LETTERS:
        encoded = tok.encode(" " + letter, add_special_tokens=False)
        if not encoded:
            ids.append(-1)
        else:
            ids.append(int(encoded[0]))
    return ids


def _load_image(path: Path, img_size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    # img_size=0 means native resolution — let the processor's
    # image_processor handle resizing/tiling. SmolVLM was designed
    # around max_image_size=512 with image splitting on; pre-resizing
    # to 224 was discarding the spatial info that splitting needs.
    if img_size > 0:
        img = img.resize((img_size, img_size), Image.BICUBIC)
    return img


class _InferenceImageDataset:
    """Yields (prompt, image, num_choices) so a DataLoader worker can
    do PIL decode + resize off the main thread.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        data_dir: Path,
        img_size: int,
        include_lecture: bool = True,
        include_hint: bool = True,
        use_chat_template: bool = False,
        processor=None,
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.img_size = img_size
        self.include_lecture = include_lecture
        self.include_hint = include_hint
        self.use_chat_template = use_chat_template
        self.processor = processor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        if self.use_chat_template and self.processor is not None:
            from src.prompt import build_chat_prompt
            prompt = build_chat_prompt(
                self.processor, row, include_answer=False,
                include_lecture=self.include_lecture,
                include_hint=self.include_hint,
            )
        else:
            prompt = build_prompt(
                row, include_answer=False,
                include_lecture=self.include_lecture,
                include_hint=self.include_hint,
            )
        image = _load_image(self.data_dir / row["image_path"], self.img_size)
        return prompt, image, int(row["num_choices"])


def _inference_collate(batch):
    prompts = [b[0] for b in batch]
    images = [b[1] for b in batch]
    num_choices = [b[2] for b in batch]
    return prompts, images, num_choices


def predict_zero_shot(
    df: pd.DataFrame,
    *,
    data_dir: Path,
    processor,
    model,
    img_size: int = 224,
    batch_size: int = 8,
    progress_every: int = 50,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    include_lecture: bool = True,
    include_hint: bool = True,
    use_chat_template: bool = False,
    return_scores: bool = False,
) -> list[int] | tuple[list[int], list[list[float]]]:
    """Batched zero-shot scoring.

    PIL image decode happens on DataLoader worker processes so it
    overlaps with GPU forward passes (observed: 1048-row eval dropped
    from 534 s to ~150 s when ``num_workers=4`` was applied; the GPU
    was at 0 % utilization without it).

    The processor pads to the longest prompt in the batch (right-side
    padding by default for SmolVLM's tokenizer). For each item, we
    take the logit at the last *non-pad* position via the
    attention_mask, then compare candidate-letter token log-probs.
    """
    import torch

    device = next(model.parameters()).device
    cand_ids = _candidate_token_ids(processor)
    preds: list[int] = []
    all_scores: list[list[float]] = []  # per-row [A, B, C, D, E] log-prob (filled to 5 with -inf for unused)
    n = len(df)

    if num_workers > 0:
        loader = torch.utils.data.DataLoader(
            _InferenceImageDataset(
                df, data_dir=data_dir, img_size=img_size,
                include_lecture=include_lecture, include_hint=include_hint,
                use_chat_template=use_chat_template, processor=processor,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=_inference_collate,
            persistent_workers=False,
        )
        batch_iter = ((i * batch_size, b) for i, b in enumerate(loader))
    else:
        def _legacy():
            for start in range(0, n, batch_size):
                slice_df = df.iloc[start : start + batch_size]
                prompts = []
                for _, r in slice_df.iterrows():
                    if use_chat_template and processor is not None:
                        from src.prompt import build_chat_prompt
                        prompts.append(build_chat_prompt(
                            processor, r, include_answer=False,
                            include_lecture=include_lecture, include_hint=include_hint,
                        ))
                    else:
                        prompts.append(build_prompt(
                            r, include_answer=False,
                            include_lecture=include_lecture, include_hint=include_hint,
                        ))
                images = [_load_image(data_dir / r["image_path"], img_size) for _, r in slice_df.iterrows()]
                num_choices = [int(r["num_choices"]) for _, r in slice_df.iterrows()]
                yield start, (prompts, images, num_choices)
        batch_iter = _legacy()

    for start, (prompts, images, num_choices) in batch_iter:

        # No truncation: SmolVLM expands <image> into 1088 tokens per
        # image, and processor-side truncation cuts that span and
        # crashes the integrity check. Cap text via build_prompt's
        # max_context_chars instead.
        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        with torch.inference_mode():
            out = model(**inputs)
            attn = inputs.get("attention_mask")
            if attn is not None:
                last_pos = attn.sum(dim=1) - 1
            else:
                last_pos = torch.full(
                    (len(prompts),), out.logits.shape[1] - 1, device=out.logits.device
                )
            idx = last_pos.view(-1, 1, 1).expand(-1, 1, out.logits.shape[-1])
            last_logits = out.logits.gather(1, idx).squeeze(1)
            log_probs = torch.log_softmax(last_logits, dim=-1).float().cpu()

        del out, last_logits, inputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        for b in range(len(prompts)):
            nc = num_choices[b]
            row_lp = log_probs[b]
            scores = [float("-inf") if cand_ids[j] < 0 else row_lp[cand_ids[j]].item() for j in range(nc)]
            preds.append(int(max(range(nc), key=lambda k: scores[k])))
            if return_scores:
                # Pad to 5 candidates with -inf so downstream callers
                # can stack into a uniform N×5 array.
                padded = list(scores) + [float("-inf")] * (5 - nc)
                all_scores.append(padded)

        done = len(preds)
        if progress_every and (done % progress_every < batch_size or done == n):
            print(f"[zero_shot] {done}/{n} done", flush=True)

    if return_scores:
        return preds, all_scores
    return preds


def evaluate_and_predict(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    data_dir: Path,
    model_id: str,
    cache_dir: str | Path,
    dtype_str: str = "float32",
    n_val: int | None = None,
    n_test: int | None = None,
    img_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 4,
) -> dict:
    """Load SmolVLM offline, score val (subset), and predict test (subset)."""
    import os

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(Path(cache_dir) / "transformers")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_str, torch.float32)

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    val_slice = val_df if n_val is None else val_df.head(int(n_val)).reset_index(drop=True)
    test_slice = test_df if n_test is None else test_df.head(int(n_test)).reset_index(drop=True)

    val_preds = predict_zero_shot(
        val_slice, data_dir=data_dir, processor=processor, model=model,
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
    )
    test_preds = predict_zero_shot(
        test_slice, data_dir=data_dir, processor=processor, model=model,
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
    )

    return {
        "val_ids": list(val_slice["id"]),
        "val_preds": val_preds,
        "val_targets": [int(a) for a in val_slice["answer"]],
        "test_ids": list(test_slice["id"]),
        "test_preds": test_preds,
        "n_val": len(val_slice),
        "n_test": len(test_slice),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": dtype_str,
    }
