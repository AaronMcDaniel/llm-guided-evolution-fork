#!/bin/bash
#SBATCH --job-name=llm_server
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:2
#SBATCH -G 2
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
echo "launching LLM Server"
hostname
module load cuda/12.2.2
source /storage/ice1/8/2/amcdaniel39/llm-guided-evolution-fork/.venv/bin/activate
export SERVER_HOSTNAME=$(hostname)
HOSTNAME_FILE=/storage/ice1/8/2/amcdaniel39/llm-guided-evolution-fork/hostname.log
echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Starting LLM server on host: $SERVER_HOSTNAME"
uvicorn server:app --host $SERVER_HOSTNAME --port 8000 --workers 1 --no-access-log 
