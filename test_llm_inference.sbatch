#!/bin/bash
#SBATCH --job-name=llm_inf_test
#SBATCH --time=08:00:00

#SBATCH -G 2 
#SBATCH -C 'A100-80GB|H100'
#SBATCH --mem-per-gpu 80G
#SBATCH -c 24


cd $SLURM_SUBMIT_DIR

module load cuda/12
module load anaconda3

conda activate llmIslandsEnv
conda info

# Set the TOKENIZERS_PARALLELISM environment variable if needed
# export TOKENIZERS_PARALLELISM=false

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/huggingface

# Run Python script
python test_llm_inference.py