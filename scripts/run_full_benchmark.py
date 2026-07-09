"""Task-by-task full-benchmark driver.

For each instance: run all 7 memory arms (agents on qwen via tunnel), then
score all 7 patches with the official SWE-bench-Live Docker harness while the
instance's image is cached, then remove the image. Fully resumable: skips
(instance, arm) pairs that already have a prediction / verdict.

Run under caffeinate in tmux:
  tmux new -d -s membench-full 'caffeinate -dims python3 scripts/run_full_benchmark.py \
      > runs/full/driver.log 2>&1'
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MB = Path(os.environ.get("MEMBENCH_ROOT", "/Users/riso1410/Projects/MemBench"))
SWB = Path(os.environ.get("SWB_ROOT", "/Users/riso1410/Projects/SWE-bench-Live"))
DOCKER_BIN = os.environ.get("DOCKER_BIN", "docker")
CONFIG = os.environ.get("MEMBENCH_CONFIG", "configs/claude_code_qwen.toml")
ARMS = ["none", "raw_rag", "structured", "claude_mem", "mem0", "graphiti", "graphify"]
DATASET = MB / "dataset/swebench_live_full/instances.jsonl"
RUNS = MB / "runs/full"
AGENT_TIMEOUT = 2400
EVAL_TIMEOUT = 3600


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def endpoints_ok() -> bool:
    for url in ("http://localhost:8000/health", "http://localhost:8001/v1/models"):
        r = subprocess.run(["curl", "-s", "-m", "5", "-o", "/dev/null", url])
        if r.returncode != 0:
            return False
    return True


def ensure_endpoints():
    for i in range(30):  # up to 15 min
        if endpoints_ok():
            return
        log(f"endpoints down, retry {i+1}/30")
        time.sleep(30)
    log("ENDPOINTS-DEAD, exiting for operator")
    sys.exit(3)


def image_for(iid: str) -> str:
    return f"starryzhang/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest"


def load_done_predictions(arm: str) -> set[str]:
    p = RUNS / arm / "predictions.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["instance_id"] for l in p.read_text().splitlines() if l.strip()}


def load_verdicts() -> dict[tuple[str, str], bool]:
    p = RUNS / "verdicts.jsonl"
    out = {}
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                out[(d["instance_id"], d["arm"])] = d["resolved"]
    return out


def run_agent(inst: dict, arm: str) -> None:
    iid = inst["instance_id"]
    arm_dir = RUNS / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    tmp_inst = arm_dir / f".tmp_{iid}.jsonl"
    tmp_pred = arm_dir / f".tmp_{iid}_pred.jsonl"
    # template_dir resolves relative to the instances file's directory —
    # absolutize it since the tmp file lives elsewhere
    inst = {**inst, "workspace": {**inst["workspace"],
            "template_dir": str(DATASET.parent / inst["workspace"]["template_dir"])}}
    tmp_inst.write_text(json.dumps(inst) + "\n")
    cmd = ["uv", "run", "--python", "3.13", "python", "-m", "membench", "run",
           "--config", CONFIG,
           "--instances", str(tmp_inst),
           "--output", str(tmp_pred), "--adapter", arm]
    try:
        r = subprocess.run(cmd, cwd=MB, capture_output=True, text=True, timeout=AGENT_TIMEOUT)
        line = tmp_pred.read_text().strip() if tmp_pred.exists() else ""
        if not line:
            line = json.dumps({"instance_id": iid, "status": "error",
                               "error": f"runner rc={r.returncode}: {r.stderr[-300:]}",
                               "resolved": None, "wall_time_sec": 0})
    except subprocess.TimeoutExpired:
        line = json.dumps({"instance_id": iid, "status": "error",
                           "error": "agent timeout", "resolved": None,
                           "wall_time_sec": AGENT_TIMEOUT})
    with (arm_dir / "predictions.jsonl").open("a") as fh:
        fh.write(line.splitlines()[0] + "\n")
    tmp_inst.unlink(missing_ok=True)
    tmp_pred.unlink(missing_ok=True)


def strip_patch(patch: str) -> str:
    keep, block, skip = [], [], False
    for l in patch.splitlines(keepends=True):
        if l.startswith("diff --git"):
            if block and not skip:
                keep += block
            block, skip = [l], ("__pycache__" in l or l.rstrip().endswith(".pyc"))
        else:
            block.append(l)
    if block and not skip:
        keep += block
    return "".join(keep)


def score(iid: str, arm: str) -> bool | None:
    pred = None
    for l in (RUNS / arm / "predictions.jsonl").read_text().splitlines():
        d = json.loads(l)
        if d["instance_id"] == iid:
            pred = d
    if pred is None or pred.get("status") != "ok":
        return None
    patch = strip_patch(pred.get("model_patch") or "")
    if not patch.strip():
        return False
    tmp = RUNS / arm / f".tmp_{iid}_patch.json"
    tmp.write_text(json.dumps({iid: {"model_patch": patch}}))
    out_dir = RUNS / arm / "docker_eval" / iid
    try:
        subprocess.run(
            [str(SWB / ".venv/bin/python"), "-m", "evaluation.evaluation",
             "--dataset", "SWE-bench-Live/SWE-bench-Live", "--split", "lite",
             "--platform", "linux", "--patch_dir", str(tmp),
             "--output_dir", str(out_dir), "--workers", "1", "--overwrite", "1"],
            cwd=SWB, capture_output=True, text=True, timeout=EVAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    finally:
        tmp.unlink(missing_ok=True)
    results = out_dir / "results.json"
    if not results.exists():
        return None
    return iid in json.load(results.open()).get("success_ids", [])


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    instances = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    done_preds = {arm: load_done_predictions(arm) for arm in ARMS}
    verdicts = load_verdicts()
    log(f"driver start: {len(instances)} instances x {len(ARMS)} arms")
    for n, inst in enumerate(instances):
        iid = inst["instance_id"]
        # 1. agents
        for arm in ARMS:
            if iid in done_preds[arm]:
                continue
            ensure_endpoints()
            t0 = time.time()
            run_agent(inst, arm)
            done_preds[arm].add(iid)
            log(f"[{n+1}/{len(instances)}] agent {arm}/{iid} {time.time()-t0:.0f}s")
        # 2. scoring (image cached across arms)
        need = [a for a in ARMS if (iid, a) not in verdicts]
        for arm in need:
            t0 = time.time()
            resolved = score(iid, arm)
            verdicts[(iid, arm)] = resolved
            with (RUNS / "verdicts.jsonl").open("a") as fh:
                fh.write(json.dumps({"instance_id": iid, "arm": arm,
                                     "resolved": resolved}) + "\n")
            log(f"[{n+1}/{len(instances)}] score {arm}/{iid} -> {resolved} {time.time()-t0:.0f}s")
        # 3. drop the image
        if need:
            subprocess.run([DOCKER_BIN, "container", "prune", "-f"], capture_output=True)
            subprocess.run([DOCKER_BIN, "rmi", image_for(iid)], capture_output=True)
    log("FULL-BENCHMARK-COMPLETE")


if __name__ == "__main__":
    main()
