#!/bin/bash
cd /Users/riso1410/Projects/MemBench
for a in none raw_rag structured claude_mem mem0 graphiti graphify; do
  ok=0
  for i in 1 2 3 4 5 6; do
    curl -s -m 5 http://localhost:8000/health -o /dev/null && curl -s -m 5 http://localhost:8001/v1/models -o /dev/null && { ok=1; break; }
    sleep 10
  done
  [ "$ok" = 1 ] || { echo "TUNNEL-DOWN before $a"; exit 2; }
  echo "=== ARM $a $(date +%H:%M:%S) ==="
  uv run --python 3.13 python -m membench run \
    --config configs/claude_code_qwen.toml \
    --instances dataset/swebench_live/instances.jsonl \
    --output "runs/swb_ccq_$a/predictions.jsonl" \
    --adapter "$a" 2>&1 | grep -v warning
done
echo "SWB-ARMS-COMPLETE $(date +%H:%M:%S)"
