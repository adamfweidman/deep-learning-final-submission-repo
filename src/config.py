"""YAML config loader.

Preserves the loop-lab schema (``attempt_id``, ``level``,
``experiment.*``, ``model.*``, ``data.*``, ``run.*``, ``notes``) so the
same YAML file can be referenced by ``runs/registry.jsonl`` events.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentBlock:
    name: str
    question: str = ""
    branch: str = "exploit"


@dataclass(frozen=True)
class ModelBlock:
    name: str
    dtype: str = "float32"
    max_tokens: int = 8192


@dataclass(frozen=True)
class DataBlock:
    data_dir: str = "data"
    train_csv: str = "data/train.csv"
    val_csv: str = "data/val.csv"
    test_csv: str = "data/test.csv"
    sample_submission_csv: str = "data/sample_submission.csv"
    n_examples: int | None = None


@dataclass(frozen=True)
class RunBlock:
    mode: str = "smoke"
    output_dir: str = ""
    seed: int = 42
    baseline: str = "always_a"
    n_val: int | None = None
    n_test: int | None = None
    img_size: int = 224
    batch_size: int = 8
    adapter_path: str = ""        # for mode=infer or mode=train resume


@dataclass(frozen=True)
class TrainBlock:
    n_train: int | None = None
    n_val_eval: int | None = None
    n_test: int | None = None
    epochs: int = 2
    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    target_modules_regex: str = ""        # if set, used as PEFT regex (overrides target_modules)
    unfreeze_modules: tuple[str, ...] = ()  # additional non-LoRA trainable patterns (e.g. "connector")
    use_dora: bool = False                # PEFT DoRA: weight-decomposed LoRA
    use_rslora: bool = False              # PEFT rsLoRA: alpha/sqrt(r) scaling
    choice_permute: bool = False
    include_lecture: bool = True
    include_hint: bool = True
    use_chat_template: bool = False
    include_solution_train: bool = False  # add `solution` to TRAIN-side context only (not eval)
    use_val_for_training: bool = False    # final-fit: train on train+val concatenated
    eval_batch_size: int = 48
    eval_num_workers: int = 4
    train_num_workers: int = 4
    train_prefetch_factor: int = 2
    log_every: int = 10
    eval_every_epoch: bool = True
    early_stop_patience: int = 1
    write_submission_csv: bool = True
    submission_threshold: float = 0.5
    resume_from: str = ""


@dataclass(frozen=True)
class PreflightBlock:
    hf_cache_dir: str = "/scratch/${USER}/hf_cache"
    download_first: bool = True
    snapshot_only: bool = False
    offline_after: bool = True


@dataclass(frozen=True)
class Config:
    attempt_id: str
    level: str
    experiment: ExperimentBlock
    model: ModelBlock
    data: DataBlock
    run: RunBlock
    preflight: PreflightBlock = field(default_factory=PreflightBlock)
    train: TrainBlock = field(default_factory=TrainBlock)
    resources: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    raw = _expand(raw)

    return Config(
        attempt_id=raw["attempt_id"],
        level=raw.get("level", "exploratory"),
        experiment=ExperimentBlock(**raw.get("experiment", {"name": "unnamed"})),
        model=ModelBlock(**raw.get("model", {"name": "HuggingFaceTB/SmolVLM-500M-Instruct"})),
        data=DataBlock(**raw.get("data", {})),
        run=RunBlock(**raw.get("run", {})),
        preflight=PreflightBlock(**raw.get("preflight", {})),
        train=TrainBlock(**{
            **raw.get("train", {}),
            **(
                {"target_modules": tuple(raw["train"]["target_modules"])}
                if "train" in raw and "target_modules" in raw["train"]
                else {}
            ),
            **(
                {"unfreeze_modules": tuple(raw["train"]["unfreeze_modules"])}
                if "train" in raw and "unfreeze_modules" in raw["train"]
                else {}
            ),
        }),
        resources=raw.get("resources", {}),
        notes=raw.get("notes", ""),
    )


def resolved_output_dir(cfg: Config) -> Path:
    out = cfg.run.output_dir
    if not out:
        user = os.environ.get("USER", "user")
        out = f"/scratch/{user}/deep-learning-final/runs/{cfg.attempt_id}"
    return Path(out)
