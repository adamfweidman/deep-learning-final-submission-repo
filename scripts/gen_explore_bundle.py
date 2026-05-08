"""Generate 8 short exploratory configs for the iter-3 broad-search bundle.

Each writes a YAML to configs/explore/. All variants share:
- 1000 train rows × 2 epochs
- 400 val eval rows
- no test / no submission
- l40s_public, bs=4, grad_accum=2, bf16
- 4 train workers + 4 eval workers, ~32 GB RAM
- runtime budget ~25-30 min

Variants differ along one axis each; bundle tests diverse hypotheses
about what's holding val_acc below 0.80.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

OUT_DIR = Path("configs/explore")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = """\
attempt_id: "{aid}"
level: "exploratory"

experiment:
  name: "{name}"
  question: "{question}"
  branch: "{branch}"

model:
  name: "HuggingFaceTB/SmolVLM-500M-Instruct"
  dtype: "bfloat16"
  max_tokens: 8192

data:
  data_dir: "data"
  train_csv: "data/train.csv"
  val_csv: "data/val.csv"
  test_csv: "data/test.csv"
  sample_submission_csv: "data/sample_submission.csv"

run:
  mode: "train"
  output_dir: "/scratch/${{USER}}/deep-learning-final/runs/{aid}"
  seed: {seed}
  baseline: "always_a"
  img_size: {img_size}

train:
  n_train: 1000
  n_val_eval: 400
  n_test: 0
  epochs: 2
  batch_size: 4
  grad_accum: 2
  lr: {lr}
  warmup_ratio: 0.05
  weight_decay: {weight_decay}
  lora_r: {lora_r}
  lora_alpha: {lora_alpha}
  lora_dropout: {lora_dropout}
{target_lines}
  choice_permute: true
  include_lecture: {include_lecture}
  include_hint: true
  eval_batch_size: 48
  eval_num_workers: 4
  train_num_workers: 4
  log_every: 25
  eval_every_epoch: true
  early_stop_patience: 2
  write_submission_csv: false

preflight:
  hf_cache_dir: "/scratch/${{USER}}/deep-learning-final/hf_cache"
  download_first: false
  snapshot_only: false
  offline_after: true

resources: {{}}

notes: |
  {notes}
"""


def cfg(aid, name, question, branch, *, seed=42, img_size=224,
        lr=2.0e-4, weight_decay=0.0, lora_r=8, lora_alpha=8,
        lora_dropout=0.05, target_modules_regex=None,
        target_modules=None, include_lecture=True, notes=""):
    if target_modules_regex:
        target_lines = f'  target_modules_regex: "{target_modules_regex}"'
    elif target_modules:
        target_lines = "  target_modules:\n    - " + "\n    - ".join(target_modules)
    else:
        target_lines = "  # default target_modules: q/k/v/o + gate/up/down (text + vision attention via PEFT pattern match)"
    body = BASE.format(
        aid=aid, name=name, question=question, branch=branch,
        seed=seed, img_size=img_size, lr=lr,
        weight_decay=weight_decay, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, target_lines=target_lines,
        include_lecture=str(include_lecture).lower(), notes=notes,
    )
    (OUT_DIR / f"{aid}.yaml").write_text(body)
    print(f"wrote {OUT_DIR / aid}.yaml")


# Text-only attention regex (used by several variants to avoid burning
# the budget on SiglipVisionModel q/k/v/o):
TEXT_ATTN_RE = r"^.*text_model\\.layers\\.\\d+\\.self_attn\\.(q_proj|k_proj|v_proj|o_proj)$"
TEXT_FULL_RE = (
    r"^.*text_model\\.layers\\.\\d+\\.(self_attn|mlp)\\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)

# E1: r=8 q+v text-only — minimal LoRA capacity
cfg(
    "2026-05-06-explore-e1-r8qv-textonly", "explore-e1-r8qv-textonly",
    "Does minimal r=8 q+v text-only LoRA underperform the full 7-module r=8 baseline at proxy scale?",
    "diagnose", lora_r=8, lora_alpha=8,
    target_modules_regex=r"^.*text_model\\.layers\\.\\d+\\.self_attn\\.(q_proj|v_proj)$",
    notes="Tests bias toward minimal capacity; ~819k trainable.",
)

# E2: r=32 q-only text — narrow but deep
cfg(
    "2026-05-06-explore-e2-r32q-text", "explore-e2-r32q-text",
    "Does r=32 q-only (deep capacity in a single matrix) outperform r=8 distributed?",
    "broaden", lora_r=32, lora_alpha=32,
    target_modules_regex=r"^.*text_model\\.layers\\.\\d+\\.self_attn\\.q_proj$",
    notes="Per-layer params: 32 layers x 32 x (960+960) = ~1.97M trainable.",
)

# E4: drop lecture — prompt minimization
cfg(
    "2026-05-06-explore-e4-no-lecture", "explore-e4-no-lecture",
    "Does removing the lecture context (often verbose paragraphs) help or hurt?",
    "diagnose", include_lecture=False,
    notes="Tests if lecture content distracts the model from question+choices.",
)

# E6: img_size=384 — visual detail
cfg(
    "2026-05-06-explore-e6-img384", "explore-e6-img384",
    "Does upping the image resize from 224 to 384 (more visual detail) help diagrams/maps?",
    "broaden", img_size=384,
    notes="SmolVLM processor handles arbitrary input sizes; image-token expansion may grow.",
)

# E8: lr=4e-4 — more aggressive
cfg(
    "2026-05-06-explore-e8-lr4e4", "explore-e8-lr4e4",
    "Does lr=4e-4 (2x baseline) cleanly beat lr=2e-4 at proxy scale?",
    "diagnose", lr=4.0e-4,
    notes="Per Llama LoRA literature, 1e-4 to 5e-4 is typical; 2e-4 may have been conservative.",
)

# E9: weight_decay=0.05 — regularizer
cfg(
    "2026-05-06-explore-e9-wd05", "explore-e9-wd05",
    "Does weight_decay=0.05 mitigate the over-fitting drift observed at epoch 3+ in 8055893?",
    "diagnose", weight_decay=0.05,
    notes="LoRA usually doesn't need WD, but on small data + small adapter it sometimes helps.",
)

# E11: lora_dropout=0.2 — different regularizer
cfg(
    "2026-05-06-explore-e11-drop20", "explore-e11-drop20",
    "Does lora_dropout=0.2 (4x baseline 0.05) regularize better?",
    "diagnose", lora_dropout=0.2,
    notes="Higher dropout slows fitting; with permutation-augmented data it's a soft generalization knob.",
)

# E12: r=4 all 7 modules — ultra-low capacity
cfg(
    "2026-05-06-explore-e12-r4all", "explore-e12-r4all",
    "Does r=4 distributed across all 7 Llama modules outperform r=8 q+v in the same total-parameter ballpark?",
    "diagnose", lora_r=4, lora_alpha=4,
    notes="Per-layer params at r=4 all-7: ~2.17M trainable.",
)
