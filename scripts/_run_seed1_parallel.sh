#!/bin/bash
cd "$HOME/MemBench"
export PATH="$HOME/.local/bin:$HOME/.opencode/bin:$PATH"
export MEMBENCH_ROOT=$HOME/MemBench SWB_ROOT=$HOME/SWE-bench-Live
export DOCKER_BIN=podman DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
export MEM0_TELEMETRY=false MEMBENCH_ARM_CONCURRENCY=4
export MEMBENCH_CONFIG=configs/claude_code_qwen_pectra.toml
exec python3 scripts/run_cross_session.py --seed 1 --out-root runs/cs_claude_seed1 \
  --repos deepset-ai/haystack,instructlab/instructlab,run-llama/llama_deploy,pdm-project/pdm,beeware/briefcase,cyclotruc/gitingest
