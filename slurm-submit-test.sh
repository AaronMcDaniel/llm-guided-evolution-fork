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

set -euo pipefail

echo "==== SLURM JOB START ===="
hostname
date
echo "PWD: $(pwd)"

# Remove stale hostname file from previous runs.
rm -f hostname.log

# Start the LLM server in the background.
# This allows the script to continue to the readiness check and tests.
echo "Starting LLM server in background..."
bash ./server.sh &

# Save the process ID of the server.
SERVER_PID=$!
echo "Server process PID: $SERVER_PID"

# Stop the server when this job exits.
cleanup() {
    echo "Stopping LLM server process: $SERVER_PID"
    kill "$SERVER_PID" || true
}
trap cleanup EXIT

# Wait for server.sh to write hostname.log.
echo "Waiting for hostname.log..."

for i in {1..60}; do
    if [ -s hostname.log ]; then
        SERVER_HOSTNAME=$(cat hostname.log)
        echo "Server hostname: $SERVER_HOSTNAME"
        break
    fi

    echo "hostname.log not ready yet... ($i/60)"

    # Check whether the server process already died.
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server process exited before writing hostname.log"
        exit 1
    fi

    sleep 10
done

if [ ! -s hostname.log ]; then
    echo "ERROR: hostname.log was not created"
    exit 1
fi

SERVER_HOSTNAME=$(cat hostname.log)
SERVER_PORT=8137
SERVER_URL="http://${SERVER_HOSTNAME}:${SERVER_PORT}/"

echo "Waiting for LLM server at $SERVER_URL"

# Wait until the HTTP server responds successfully.
for i in {1..120}; do
    if curl -fsS --connect-timeout 5 --max-time 10 "$SERVER_URL" > /dev/null; then
        echo "LLM server is ready"
        break
    fi

    echo "LLM server not ready yet... ($i/120)"

    # If the server process died, stop immediately.
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server process exited before becoming ready"
        exit 1
    fi

    sleep 10
done

# Final readiness check.
if ! curl -fsS --connect-timeout 5 --max-time 10 "$SERVER_URL" > /dev/null; then
    echo "ERROR: LLM server failed readiness check"
    exit 1
fi

echo "LLM server is verified and ready."

# Export this so your tests can use it.
export LLM_SERVER_HOST="$SERVER_HOSTNAME"
export LLM_SERVER_PORT="$SERVER_PORT"
export LLM_SERVER_URL="$SERVER_URL"

# Make sure the report directory exists in the repo checkout
mkdir -p tests/results

# CUDA / nvjitlink path
export LD_LIBRARY_PATH="$HOME/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

# Ensure uv is available
command -v uv >/dev/null 2>&1 || { echo "uv is not installed or not on PATH"; exit 1; }

# Sync dependencies from pyproject.toml / uv.lock
uv sync

# Run tests inside the uv-managed environment and generate JUnit XML
#uv run pytest tests/test_*.py --junitxml=tests/results/report.xml

#including "junit_logging=all" will ensure print statements are included
uv run pytest tests/test_*.py -s --junitxml=tests/results/report.xml -o "junit_logging=all"

echo "==== SLURM JOB END ===="
date