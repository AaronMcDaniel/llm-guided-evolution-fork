#!/bin/bash
#SBATCH --job-name=llmge_ci
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-gpu 16G
#SBATCH --gres=gpu:1
#SBATCH -C "H100|H200"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Exit immediately if a command fails.
# Treat unset variables as errors.
# Make pipeline failures visible.
set -euo pipefail

echo "==== SLURM JOB START ===="
hostname
date
echo "PWD: $(pwd)"

# Submit the LLM server as a separate Slurm job.
echo "Submitting LLM server job..."
SERVER_JOB_ID=$(sbatch --parsable server.sh)

echo "Submitted server job: $SERVER_JOB_ID"

# Wait for server.sh to write hostname.log.
echo "Waiting for hostname.log..."

for i in {1..60}; do
    if [ -s hostname.log ]; then
        SERVER_HOSTNAME=$(cat hostname.log)
        echo "Server hostname: $SERVER_HOSTNAME"
        break
    fi

    echo "hostname.log not ready yet... ($i/60)"
    sleep 10
done

# If hostname.log was never created, the server probably failed to start
# or is still stuck in the Slurm queue.

if [ ! -s hostname.log ]; then
    echo "ERROR: hostname.log was not created"
    sacct -j "$SERVER_JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed
    exit 1
fi

# Read the server hostname and construct the server URL.
# This must match the port used in server.sh:
# uv run uvicorn server:app --host $SERVER_HOSTNAME --port 8137

SERVER_HOSTNAME=$(cat hostname.log)
SERVER_URL="http://${SERVER_HOSTNAME}:8137/"

echo "Waiting for LLM server at $SERVER_URL"


# Wait until the server responds successfully.

for i in {1..120}; do
    if curl -fsS --connect-timeout 5 --max-time 10 "$SERVER_URL" > /dev/null; then
        echo "LLM server is ready"
        break
    fi

    echo "LLM server not ready yet... ($i/120)"
    sleep 10
done

# Final readiness check.
# If this fails, stop the test job instead of running tests against
# a server that is not ready.
curl -fsS --connect-timeout 5 --max-time 10 "$SERVER_URL" > /dev/null || {
    echo "ERROR: LLM server failed readiness check"
    sacct -j "$SERVER_JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed
    exit 1
}


# Make sure the report directory exists in the repo checkout
mkdir -p tests/results

# CUDA / nvjitlink path
export LD_LIBRARY_PATH="$HOME/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

# Ensure uv is available
command -v uv >/dev/null 2>&1 || { echo "uv is not installed or not on PATH"; exit 1; }

# Sync dependencies from pyproject.toml / uv.lock
uv sync

# Run tests inside the uv-managed environment and generate JUnit XML
#v run pytest tests/test_*.py --junitxml=tests/results/report.xml

#Added -v and the capture option  for verbose output and to allow print statement from benchmark test to display in xml report and slurm output 
uv run pytest tests/test_*.py -v --capture=tee-sys --junitxml=tests/results/report.xml

echo "==== SLURM JOB END ===="
date