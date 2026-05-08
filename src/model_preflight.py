"""HF model cache preflight for SmolVLM-500M-Instruct.

Designed for SLURM-first execution. The preflight has two phases that
are independently controllable from the YAML config:

1. **Download phase** (``download_first=True``) — call
   ``snapshot_download`` to populate the cache under ``/scratch``. Skip
   this on a compute node that has no internet by setting
   ``download_first: false`` in the config (or by submitting with
   ``HF_HUB_OFFLINE=1`` already set), and run the download once on a
   login or interactive node first.

2. **Offline-load verification** — always run, regardless of phase 1.
   Reloads the processor and model with
   ``HF_HUB_OFFLINE=1`` + ``TRANSFORMERS_OFFLINE=1`` and
   ``local_files_only=True`` so the failure mode of "cache exists but
   isn't usable from a no-internet batch run" is caught here, not
   inside a long training job.

The report records ``cache_hit``, ``offline_load_ok``, parameter
counts, and the env vars to export for downstream SLURM jobs.

``torch`` / ``transformers`` are imported lazily so the smoke path can
run with only the main project dependencies installed.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PreflightReport:
    model_id: str
    cache_dir: str
    snapshot_path: str
    cache_hit: bool
    offline_load_ok: bool
    total_params: int
    trainable_params: int
    trainable_within_5m_budget: bool
    dtype: str
    offline_env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _set_hf_env(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    transformers_cache = cache_dir / "transformers"
    transformers_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _offline_env(cache_dir: Path) -> dict[str, str]:
    return {
        "HF_HOME": str(cache_dir),
        "TRANSFORMERS_CACHE": str(cache_dir / "transformers"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


@contextmanager
def _offline_mode():
    prior = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def count_trainable(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_all(model) -> None:
    """Freeze every parameter. Default for the preflight: 0 trainable.

    Real adapter wiring (LoRA / partial unfreeze) belongs in a separate
    module so the preflight stays a pure infrastructure check.
    """
    for p in model.parameters():
        p.requires_grad_(False)


def _try_local_snapshot(model_id: str, cache_dir: Path) -> str | None:
    """Return the local snapshot path if it exists, else None.

    Mirrors ``huggingface_hub.try_to_load_from_cache`` semantics but
    returns the snapshot directory path. If ``snapshot_download`` has
    already been run for this repo, the directory will exist under
    ``<cache>/transformers/models--<org>--<name>/snapshots/<rev>``.
    """
    org, name = model_id.split("/", 1)
    base = cache_dir / "transformers" / f"models--{org}--{name}" / "snapshots"
    if not base.exists():
        return None
    snapshots = [p for p in base.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(snapshots[0])


def run_preflight(
    model_id: str,
    cache_dir: str | Path,
    *,
    download_first: bool = True,
    snapshot_only: bool = False,
    dtype_str: str = "float32",
    output_dir: str | Path | None = None,
) -> PreflightReport:
    cache_dir = Path(cache_dir)
    _set_hf_env(cache_dir)
    notes: list[str] = []

    snapshot_path: str = ""
    if download_first:
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir / "transformers"),
        )
        notes.append("download_phase=ran")
    else:
        local = _try_local_snapshot(model_id, cache_dir)
        if local is None:
            raise RuntimeError(
                f"download_first=False but no local snapshot found for {model_id} "
                f"under {cache_dir}/transformers. Run preflight once with "
                f"download_first=true on a network-enabled node first."
            )
        snapshot_path = local
        notes.append("download_phase=skipped (cache hit)")

    cache_hit = bool(snapshot_path) and Path(snapshot_path).exists()

    if snapshot_only:
        report = PreflightReport(
            model_id=model_id,
            cache_dir=str(cache_dir),
            snapshot_path=snapshot_path,
            cache_hit=cache_hit,
            offline_load_ok=False,
            total_params=0,
            trainable_params=0,
            trainable_within_5m_budget=True,
            dtype=dtype_str,
            offline_env=_offline_env(cache_dir),
            notes=[*notes, "snapshot_only=true (no model load attempted)"],
        )
    else:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype_str, torch.float32)

        with _offline_mode():
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token

            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            freeze_all(model)
            total, trainable = count_trainable(model)

        notes.append("offline_load=ran (HF_HUB_OFFLINE=1, local_files_only=True)")

        report = PreflightReport(
            model_id=model_id,
            cache_dir=str(cache_dir),
            snapshot_path=snapshot_path,
            cache_hit=cache_hit,
            offline_load_ok=True,
            total_params=int(total),
            trainable_params=int(trainable),
            trainable_within_5m_budget=trainable <= 5_000_000,
            dtype=dtype_str,
            offline_env=_offline_env(cache_dir),
            notes=notes,
        )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "preflight_report.json").write_text(json.dumps(asdict(report), indent=2))

    return report
