#!/bin/bash
# Score MemBench SWE-bench-Live predictions with the official Docker harness.
# Usage: score_swb_arms.sh <arm> [<arm> ...]
set -u
MB=/Users/riso1410/Projects/MemBench
SWB=/Users/riso1410/Projects/SWE-bench-Live
for a in "$@"; do
  echo "=== SCORE $a $(date +%H:%M:%S) ==="
  python3 - "$a" <<'PYEOF'
import json, sys
arm = sys.argv[1]
preds = {}
for line in open(f"/Users/riso1410/Projects/MemBench/runs/swb_ccq_{arm}/predictions.jsonl"):
    d = json.loads(line)
    patch = d.get("model_patch") or ""
    # strip pycache/binary noise the workspace diff picks up
    keep, block, skip = [], [], False
    for l in patch.splitlines(keepends=True):
        if l.startswith("diff --git"):
            if block and not skip: keep += block
            block, skip = [l], ("__pycache__" in l or l.rstrip().endswith(".pyc"))
        else:
            block.append(l)
    if block and not skip: keep += block
    preds[d["instance_id"]] = {"model_patch": "".join(keep)}
out = f"/Users/riso1410/Projects/MemBench/runs/swb_ccq_{arm}/patches.json"
json.dump(preds, open(out, "w"), indent=1)
print(f"wrote {out} ({len(preds)} patches)")
PYEOF
  cd "$SWB" && .venv/bin/python -m evaluation.evaluation \
    --dataset SWE-bench-Live/SWE-bench-Live --split lite \
    --platform linux \
    --patch_dir "$MB/runs/swb_ccq_$a/patches.json" \
    --output_dir "$MB/runs/swb_ccq_$a/docker_eval" \
    --workers 2 --overwrite 1 2>&1 | tail -8
done
echo "SWB-SCORE-COMPLETE"
