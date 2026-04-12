#!/bin/bash
#SBATCH --job-name=llmge_ci
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-gpu 16G
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# pgrep -f '[/]server\.sh' > /dev/null || bash ./server.sh

if [ ! -f "sota/Titanic/data/train.csv" ]; then
  (
    cd sota/Titanic || exit 1
    uv run ./pull_data.sh
  )
fi

if [ ! -f "sota/Titanic/data/processed_train.csv" ]; then
  (
    cd sota/Titanic || exit 1
    uv run preprocess.py
  )
fi

set -euo pipefail

echo "==== SLURM JOB START ===="
hostname
date
echo "PWD: $(pwd)"

# Make sure the report directory exists in the repo checkout
mkdir -p tests/results

# CUDA / nvjitlink path
export LD_LIBRARY_PATH="$HOME/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

# Ensure uv is available
command -v uv >/dev/null 2>&1 || { echo "uv is not installed or not on PATH"; exit 1; }

# Sync dependencies from pyproject.toml / uv.lock
uv sync


# Run tests inside the uv-managed environment and generate JUnit XML
uv run pytest tests/test_*.py --junitxml=tests/results/report.xml

echo "==== SLURM JOB END ===="
date