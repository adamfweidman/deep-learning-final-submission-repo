# Pixels to Predictions — Final Project (CS-GY 6953 / ECE-GY 7123)

Public code release. Multimodal multiple-choice reasoning on the **Pixels to Predictions: DL Vision Challenge** Kaggle competition. Backbone: `HuggingFaceTB/SmolVLM-500M-Instruct`, fine-tuned with LoRA under a 5M trainable-parameter cap, with a pHash-based retrieval overlay at inference time.

**Author:** Adam Weidman (solo).

## Headline numbers

| Pipeline | Local val | Public LB |
|---|---:|---:|
| Always-A baseline | 0.331 | — |
| Zero-shot SmolVLM-500M | 0.558 | — |
| LoRA r=8 / all 7 modules / α=16 (lora-001) | 0.7767 | — |
| **V_β** (r=8, all 7, α=8, choice-permute) | **0.8073** | **0.86** ← best |
| iter4-vd (V_β + lr=4e-4) | 0.8225 | 0.84 |
| iter4-vd + retrieval (h=4, qsim=0.85) | 0.8483 | 0.86 |
| finfit (train+val) | 0.8674 (leaky) | 0.823 |
| finfit + retrieval (h=4) | 0.8874 (leaky) | 0.849 |

**Headline:** plain V_β at 0.86 is the top public-LB result. Two submissions we expected to do better — finfit and finfit + retrieval — actually scored lower on public despite higher local val. See the report for analysis.

## Repo layout

```
src/                Python package (training, inference, retrieval, prompts, …)
configs/            YAML hyperparameter configs (one per experiment)
scripts/            CLI helpers: retrieval overlay, calibration, majority vote
sbatch/             SLURM scripts used on NYU HPC
notebooks/          Inference-only notebook (Colab/Kaggle-runnable)
PROJECT.md          Task description (constraints, data layout, eval metric)
DATA.md             Local data layout (symlinks over kagglehub cache)
pyproject.toml      uv-managed deps (Python 3.12)
requirements.txt    pip-installable mirror
```

## Setup

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Download the competition data (writes to ~/.cache/kagglehub)
python -c "import kagglehub; print(kagglehub.competition_download('pixels-to-predictions'))"

# 3. Symlink into ./data/ following the layout in DATA.md
#    (or set DATA_DIR env var; src/config.py respects it)

# 4. Pre-cache the SmolVLM checkpoint (one-time)
python -m src.run --config configs/preflight.yaml
```

## Reproducing the headline results

All commands assume `./data/` follows `DATA.md`. Outputs land in `runs/<attempt_id>/`.

### Train V_β (best non-retrieval adapter, val 0.8073 → public 0.86)

```bash
python -m src.run --config configs/train_lora_v3b_allmod_permute.yaml
```

5 epochs, ~3.5 h on a single L40S. Best adapter saves to `runs/<attempt_id>/adapter_best/`.

### Train final-fit (train ∪ val, expected best leaderboard)

```bash
python -m src.run --config configs/train_lora_final_fit.yaml
```

### Inference + submission CSV

```bash
# Set adapter_path in configs/infer_vb_best.yaml to your adapter_best path, then:
python -m src.run --config configs/infer_vb_best.yaml
```

Writes `runs/2026-05-06-infer-vb-best/submission.csv`.

### Retrieval overlay (the +2 pp post-processor)

```bash
python scripts/retrieval_overlay.py \
  --base-submission runs/2026-05-06-infer-vb-best/submission.csv \
  --hamming-thresh 4 \
  --qsim-thresh 0.85 \
  --require-choice-match \
  --out runs/retrieval-vbeta-h4/
```

### Inference notebook (Colab/Kaggle-runnable)

`notebooks/submission-notebook.ipynb` is **inference-only**: loads the adapter from Drive, generates `submission.csv`, runs offline. Suitable for the Kaggle judge.

## Trained adapter weights

- **V_β `adapter_best/`** (val 0.8073, public LB 0.86): see `WEIGHTS.md` (link added at submission time).
- **final-fit `adapter_best/`**: see `WEIGHTS.md`.

> If the link is unreachable, please email afw8937@nyu.edu.

## Reproducibility

- **Fixed seeds.** `src/run.py:_seed_all` seeds `random`, `numpy`, `torch` (CPU + CUDA), `transformers.set_seed`, `PYTHONHASHSEED`, and forces cuDNN deterministic. Default seed is 42 for inference and 43 for V_β training (matches the seed in `configs/train_lora_v3b_allmod_permute.yaml`).
- **Pinned deps.** `requirements.txt` and `pyproject.toml` pin exact versions of `torch==2.11.0` and `transformers==4.57.6`.
- **Offline inference.** Inference path (`mode=infer` in YAML) sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`; works without internet once the cache is populated.
- **DataLoader determinism.** Per-epoch generators seeded with `seed + epoch`. `TOKENIZERS_PARALLELISM=false` set before transformers import to avoid the fork deadlock observed on HPC.
- **Expected numbers.** With the V_β config above and seed 43, val_accuracy at epoch 4 should be 0.8073 ± 0.005.

## Notes

- Trained on NYU HPC (`l40s_public` partition) under the 5M trainable-parameter cap. Inference fits comfortably on a single L40S, T4 (Colab Free), or P100 (Kaggle Free).
- Code includes the AI-tooling disclosure required by the assignment; see report `\section*{AI Tooling Disclosure}` for details.
- This repo intentionally excludes the internal experiment journal (`artifacts/EXPERIMENTS.md`), append-only registry (`runs/registry.jsonl`), and exploratory notebook (`override.ipynb`) used during development. The full development history is documented in the report.
