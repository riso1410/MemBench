#!/bin/bash
# Wait for the full-benchmark driver to finish, requeue errored pairs,
# rerun the driver to fill them, then emit final adjusted results.
set -u
cd ~/MemBench
export PATH="$HOME/.local/bin:$PATH" MEMBENCH_ROOT=$HOME/MemBench SWB_ROOT=$HOME/SWE-bench-Live     DOCKER_BIN=podman DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock     MEMBENCH_CONFIG=configs/claude_code_qwen_pectra.toml MEM0_TELEMETRY=false

while pgrep -fx '.*python3 scripts/run_full_benchmark.py.*' >/dev/null; do sleep 300; done
echo "[finisher] driver done $(date)" >> runs/full/driver.log
python3 scripts/requeue_errored.py >> runs/full/driver.log 2>&1
python3 scripts/run_full_benchmark.py >> runs/full/driver.log 2>&1
python3 scripts/adjusted_results.py | tee runs/full/adjusted_results_final.txt
echo "[finisher] adjusted results written $(date)" >> runs/full/driver.log
