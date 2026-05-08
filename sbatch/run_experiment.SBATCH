#!/bin/bash
#SBATCH --account=torch_pr_62_tandon_priority
#SBATCH --partition=l40s_public
#SBATCH --job-name=experiment
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
# Frugal-resources rule (.claude/CLAUDE.md): one l40s_public GPU is
# the default for light GPU work on this project. SmolVLM-500M with a
# 5M trainable-parameter cap fits trivially in 48 GB. Do NOT bump to
# h100/h200 — those are off-limits for this project.
#
# --mem=32G (not 16G): proxies 8067349 / 8067351 OOM-killed at end of
# epoch 2 with --mem=16G + 4 train_workers + 4 eval_workers; each
# DataLoader worker forks the parent's RAM (model + processor) and
# the system mem ceiling tripped before final eval. 32G is the
# smallest credible figure; keep cpus/gpu/walltime as small as ever.

# Parametric SBATCH template. Submit with:
#
#   sbatch --export=ALL,CONFIG=configs/<name>.yaml \
#     sbatch/run_experiment.SBATCH
#
# Per docs/design.md sec 10: experiment-specific parameters belong in
# the YAML config, not in this file. Edit this file only when the way
# the project launches on the cluster changes (modules, env, account,
# partition wiring), not for every new attempt.
#
# Override partition/account at submit time via additional --export
# vars or sbatch flags rather than editing this file.

set -euo pipefail

if [[ -z "${CONFIG:-}" ]]; then
    echo "ERROR: CONFIG env var is required (path to YAML config)." >&2
    exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: CONFIG file not found: $CONFIG" >&2
    exit 2
fi

mkdir -p "/scratch/$USER/logs"

# Standard env (matches .claude/skills/hpc/SKILL.md).
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT="/scratch/$USER/$(basename "$PWD")/venv"
export UV_CACHE_DIR="/scratch/$USER/.uv-cache"
export HF_HOME="/scratch/$USER/$(basename "$PWD")/hf_cache"

# Source project .env safely so subprocesses see the variables.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Bypass `source venv/bin/activate` and call the venv's python
# directly. With `set -euo pipefail` the activate script can leave
# PATH unmodified, which lets the system python3.9 win and crashes
# on first import (observed: job 8046008, ModuleNotFoundError numpy).
PROJECT_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
if [[ ! -x "$PROJECT_PYTHON" ]]; then
    echo "ERROR: expected venv python at $PROJECT_PYTHON not found" >&2
    exit 3
fi
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"

echo "[sbatch] host=$(hostname) job=${SLURM_JOB_ID:-no-slurm} config=$CONFIG"
echo "[sbatch] git: $(git rev-parse --abbrev-ref HEAD)@$(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
echo "[sbatch] python=$($PROJECT_PYTHON -V)"

# The single CONFIG path is the only required argument; everything
# else lives in the YAML.
"$PROJECT_PYTHON" -m src.run --config "$CONFIG"
