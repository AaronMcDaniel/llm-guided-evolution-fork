#!/bin/bash
#SBATCH --job-name=llm_opt
#SBATCH -t 8:00:00
#SBATCH --mem-per-gpu 16G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|H200|V100-16GB|V100-32GB|RTX6000|A40|L40S"
echo "launching LLM Guided Evolution"
hostname
module load cuda
module load anaconda3
module load uv
export CUDA_VISIBLE_DEVICES=0
export MKL_THREADING_LAYER=GNU 
export SERVER_HOSTNAME=$(hostname)
uv run python run_improved.py point_transformers_test
