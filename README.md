# Pixels to Predictions — Final Project (CS-GY 6953 / ECE-GY 7123)

Public code release. Multimodal multiple-choice reasoning on the **Pixels to Predictions: DL Vision Challenge** Kaggle competition. Backbone: `HuggingFaceTB/SmolVLM-500M-Instruct`, fine-tuned with LoRA under a 5M trainable-parameter cap, with a pHash-based retrieval overlay at inference time.

**Author:** Adam Weidman (solo).

## Headline numbers

| Pipeline | Local val | Public LB |
|---|---:|---:|
| Always-A baseline | 0.331 | — |
| Zero-shot SmolVLM-500M | 0.558 | — |
| LoRA r=8 / all 7 modules / α=16 (lora-001) | 0.7767 | — |
| V_β (r=8, all 7, α=8, choice-permute) | 0.8073 | 0.86 |
| iter4-vd (V_β + lr=4e-4) | 0.8225 | 0.84 |
| **iter4-vd + retrieval (h=4, qsim=0.85)** | **0.8483** | **0.861** ← best |
| finfit (train+val) | 0.8674 (leaky) | 0.823 |
| finfit + retrieval (h=4) | 0.8874 (leaky) | 0.849 |

**Headline:** the best public-LB result is **iter4-vd + retrieval at 0.861** (a higher-LR LoRA with the perceptual-hash retrieval overlay). Plain V_β at 0.86 is a close second. Two submissions we expected to do better — finfit and finfit + retrieval — actually scored lower on public despite higher local val. See the report for the analysis.

## Quickstart: reproduce the 0.861 submission

```bash
# 1. install deps
pip install -r requirements.txt

# 2. download competition data
python -c "import kagglehub; print(kagglehub.competition_download('pixels-to-predictions'))"
# then symlink into ./data/ per DATA.md

# 3. download the trained iter4-vd adapter and unzip into ./adapter_best/
#    https://drive.google.com/file/d/1H9PaPqCkqgfRtIRiNhHy4e5lf58VGJ63/view?usp=sharing

# 4. run inference (writes runs/infer-iter4-vd/submission.csv + val_scores.json)
python -m src.run --config configs/infer_iter4_vd_local.yaml

# 5. apply retrieval overlay (writes runs/retrieval-vd-h4q085/submission_retrieval.csv)
python scripts/retrieval_overlay.py \
    --base-submission runs/infer-iter4-vd/submission.csv \
    --base-val-scores runs/infer-iter4-vd/val_scores.json \
    --hamming-thresh 4 --qsim-thresh 0.85 --require-choice-match \
    --out runs/retrieval-vd-h4q085/
```

The retrieval overlay auto-builds the pHash cache at `data/phash_cache/{train,val,test}_phash.json` on first run (~30–90 s) and reuses it on subsequent runs.

The `configs/infer_iter4_vd_local.yaml` quickstart config is configured with all-local paths (`./adapter_best/`, `./hf_cache/`, `runs/`). The other configs in `configs/` (e.g. `infer_iter4_vd_best.yaml`, `train_lora_*.yaml`) use HPC-style `/scratch/${USER}/...` paths and are kept for transparency about the exact experiments run; edit `adapter_path`, `output_dir`, and `preflight.hf_cache_dir` to local paths if you want to use them on a non-HPC system.

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

### Inference notebook (Colab/Kaggle-runnable)

`notebooks/submission-notebook.ipynb` is **inference-only**: loads the adapter, generates `submission.csv`, then optionally applies the retrieval overlay. Set `ADAPTER_PATH` to your local adapter directory and `DATA_DIR` to your local data directory before running.

## Trained adapter weights

The iter4-vd adapter (the base for the 0.861 winning submission):

**https://drive.google.com/file/d/1H9PaPqCkqgfRtIRiNhHy4e5lf58VGJ63/view?usp=sharing**

Download and unzip into `./adapter_best/`. The Drive link is set to "Anyone with the link can view." If you hit a permission wall, please email afw8937@nyu.edu.

## Reproducibility

- **Fixed seeds.** `src/run.py:_seed_all` seeds `random`, `numpy`, `torch` (CPU + CUDA), `transformers.set_seed`, `PYTHONHASHSEED`, and forces cuDNN deterministic. Default seed is 42 for inference and 43 for V_β training (matches the seed in `configs/train_lora_v3b_allmod_permute.yaml`).
- **Pinned deps.** `requirements.txt` pins `torch==2.11.0`, `transformers==4.57.6`, `peft`, and `accelerate` to exact versions used during training.
- **Offline inference.** Inference path (`mode=infer` in YAML) sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`; works without internet once the cache is populated.
- **DataLoader determinism.** Per-epoch generators seeded with `seed + epoch`. `TOKENIZERS_PARALLELISM=false` set before transformers import to avoid the fork deadlock observed on HPC.
- **Expected numbers.** With the V_β config above and seed 43, val_accuracy at epoch 4 should be 0.8073 ± 0.005.

## Notes

- Trained on NYU HPC (`l40s_public` partition) under the 5M trainable-parameter cap. Inference fits comfortably on a single L40S, T4 (Colab Free), or P100 (Kaggle Free).
- Code includes the AI-tooling disclosure required by the assignment; see report `\section*{AI Tooling Disclosure}` for details.
- This repo intentionally excludes the internal experiment journal (`artifacts/EXPERIMENTS.md`), append-only registry (`runs/registry.jsonl`), and exploratory notebook (`override.ipynb`) used during development. The full development history is documented in the report.
