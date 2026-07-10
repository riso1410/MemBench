#!/bin/bash
# ponytail: sequential 2-phase seed matrix; one box, vllm is the shared bottleneck
set -u
cd "$HOME/MemBench"
export PATH="$HOME/.local/bin:$HOME/.opencode/bin:$PATH"
export MEMBENCH_ROOT=$HOME/MemBench SWB_ROOT=$HOME/SWE-bench-Live
export DOCKER_BIN=podman DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock MEM0_TELEMETRY=false
REPOS="deepset-ai/haystack,instructlab/instructlab,run-llama/llama_deploy,pdm-project/pdm,beeware/briefcase,cyclotruc/gitingest"
log(){ echo "[$(date +%F\ %T)] $*"; }

log "starting vllm servers"
tmux new-session -d -s vllm 'cd ~/vllm-serve && PATH="$HOME/vllm-serve/.venv/bin:/usr/local/cuda/bin:$PATH" vllm serve cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit --enable-auto-tool-choice --tool-call-parser qwen3_coder --host 127.0.0.1 --max-model-len 65536 --served-model-name qwen3-coder-30b --gpu-memory-utilization 0.80 2>&1 | tee -a ~/vllm-serve/vllm.log'
n=0; until curl -sf localhost:8000/health >/dev/null; do sleep 15; n=$((n+1)); [ $n -gt 80 ] && { log "FATAL: vllm never became healthy"; exit 1; }; done
tmux new-session -d -s vllm-embed '~/vllm-serve/.venv/bin/vllm serve Qwen/Qwen3-Embedding-0.6B --served-model-name qwen3-embedding --host 127.0.0.1 --port 8001 --runner pooling --convert embed --gpu-memory-utilization 0.08 --max-model-len 4096 2>&1 | tee -a ~/vllm-serve/vllm-embed.log'
n=0; until curl -sf localhost:8001/health >/dev/null; do sleep 10; n=$((n+1)); [ $n -gt 60 ] && { log "FATAL: embed vllm never healthy"; exit 1; }; done
log "vllm healthy"

log "opencode smoke test"
OC_OK=1
timeout 1200 "$HOME/.local/bin/uv" run --python 3.13 python -m membench run \
  --config configs/opencode_qwen_pectra.toml --instances /tmp/opencode_smoke_inst.jsonl \
  --output /tmp/opencode_smoke_pred.jsonl --adapter none || OC_OK=0
if [ $OC_OK -eq 1 ] && grep -q '"status": *"ok"' /tmp/opencode_smoke_pred.jsonl 2>/dev/null; then
  log "opencode smoke OK"
else
  OC_OK=0; log "OPENCODE SMOKE FAILED — phase 2 will be skipped"
fi

run_phase(){ # $1=prefix $2=config
  for s in 1 2 3; do
    root="runs/cs_${1}_seed$s"; mkdir -p "$root"
    tmux new-session -d -s "mb-${1}$s" "cd $HOME/MemBench && MEMBENCH_CONFIG=$2 python3 scripts/run_cross_session.py --seed $s --out-root $root --repos '$REPOS' > $root/driver.log 2>&1"
    log "launched mb-${1}$s -> $root"
  done
  while :; do
    done_n=0
    for s in 1 2 3; do grep -q 'CROSS-SESSION-COMPLETE' "runs/cs_${1}_seed$s/driver.log" 2>/dev/null && done_n=$((done_n+1)); done
    [ $done_n -eq 3 ] && break
    pgrep -f "run_cross_session.py" >/dev/null || { log "WARN: no drivers alive but only $done_n/3 complete"; break; }
    sleep 300
  done
  log "phase $1 drivers finished"
  for s in 1 2 3; do
    .venv/bin/python scripts/evaluate_cross_session.py --run-root "runs/cs_${1}_seed$s" && log "eval done cs_${1}_seed$s" || log "WARN eval failed cs_${1}_seed$s"
  done
}

log "PHASE 1: claude_code x 3 seeds"
run_phase claude configs/claude_code_qwen_pectra.toml
if [ $OC_OK -eq 1 ]; then
  log "PHASE 2: opencode x 3 seeds"
  run_phase oc configs/opencode_qwen_pectra.toml
else
  log "phase 2 skipped (smoke failed)"
fi

log "cleanup: stopping vllm (free VRAM)"
tmux send-keys -t vllm C-c 2>/dev/null; tmux send-keys -t vllm-embed C-c 2>/dev/null; sleep 15
tmux kill-session -t vllm 2>/dev/null; tmux kill-session -t vllm-embed 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
log "MATRIX-COMPLETE"
