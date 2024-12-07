#!/bin/bash
#SBATCH --job-name=LLMTest
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=08:00:00
#SBATCH -oReport-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda/12
module load anaconda3

conda activate llmIntegration #ur local environment
conda info

export HF_HOME=/storage/ice1/0/1/gmiao8/.cache/huggingface

python master.py testIslands --islands 3
