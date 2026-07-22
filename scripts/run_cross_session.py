"""True cross-session memory benchmark driver.

Modeled on scripts/run_full_benchmark.py (same resumability, endpoint checks, and
SWE-bench-Live Docker scoring flow). The difference is WHERE each memory arm's
memory comes from:

  run_full_benchmark: every task retrieves from a pre-mined git-history corpus
                      (the instance's memory_corpus.path) -- a proxy for memory.

  run_cross_session:  each memory arm keeps a PERSISTENT, per-(arm, repo) store
                      that starts EMPTY. Repos are processed as chronological
                      sequences; after each task the arm writes a memory record
                      (issue + final agent message + resulting patch + verdict)
                      into its store, so before task k the store holds only what
                      the agent produced on tasks 1..k-1 of that repo. This is the
                      "does memory the agent accumulates itself help later tasks?"
                      experiment. Pass --use-mined-corpus to fall back to the old
                      mined-corpus behaviour instead (default: off).

Repo/code state per task is unchanged from the single-task setup: setup_workspace
always makes a fresh checkout at the instance's base commit. Only MEMORY persists.

ADAPTER-INTERFACE COMPROMISE (uniform across arms, documented on purpose):
  The membership adapters expose retrieve()/write(), but every shipped write() is a
  no-op and each adapter only knows how to INGEST a corpus directory of jsonl files
  (documents.jsonl / project_memory.jsonl / events.jsonl). Rather than teach each
  arm a bespoke agent-memory-writing path, we implement the simple, uniform version:
  after each task we append one memory record to the arm's store as a corpus
  document (project_memory.jsonl for the `structured` arm, documents.jsonl for every
  other arm -- both formats every adapter's ingestion already reads), then point the
  next task's instance.memory_corpus.path at a fresh snapshot of that store. The
  snapshot dir name is unique per (arm, repo, k) so the caching adapters
  (mem0 / graphiti / graphify, which key their built stores by corpus dir name and
  skip re-ingestion via an .ingested marker) rebuild from the grown corpus each task
  instead of serving a stale first-task store. Cost: those three re-ingest the full
  accumulated corpus at every task (LLM extraction), so a long sequence is O(k^2)
  ingestion work for them -- inherent to honest cross-session, noted here.

Run under caffeinate in tmux (do NOT reuse the membench-full session name):
  tmux new -d -s membench-cross 'caffeinate -dims python3 scripts/run_cross_session.py \
      > runs/cross_session/driver.log 2>&1'
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MB = Path(os.environ.get("MEMBENCH_ROOT", Path(__file__).resolve().parent.parent))
SWB = Path(os.environ.get("SWB_ROOT", str(Path.home() / "SWE-bench-Live")))
DOCKER_BIN = os.environ.get("DOCKER_BIN", "docker")
CONFIG = os.environ.get("MEMBENCH_CONFIG", "configs/claude_code_qwen_pectra.toml")
# oracle is non-accumulating: it injects gold-edited file paths (E2 upper bound),
# so it skips the per-(arm,repo) store, snapshot, and write-back machinery below.
NON_ACCUM = ("none", "oracle")
ARMS = ["none", "oracle", "raw_rag", "structured", "claude_mem", "mem0", "graphiti", "graphify"]
MEMORY_ARMS = [a for a in ARMS if a not in NON_ACCUM]

DATASET_DIR = MB / "dataset/cross_session"
SEQUENCES = DATASET_DIR / "sequences.jsonl"
INSTANCES = DATASET_DIR / "instances.jsonl"
# template_dir / mined corpus paths in the instance dicts are relative to the
# original full dataset dir, not dataset/cross_session.
ORIG_DATASET = MB / "dataset/swebench_live_full"
RUNS = MB / "runs/cross_session"
AGENT_TIMEOUT = 2400
ARM_CONC = int(os.environ.get("MEMBENCH_ARM_CONCURRENCY", "4"))
EVAL_TIMEOUT = 3600

CORPUS_FILES = ("documents.jsonl", "project_memory.jsonl", "events.jsonl")


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


# --- endpoint / scoring helpers (copied from run_full_benchmark to stay decoupled) -

def endpoints_ok() -> bool:
    for url in ("http://localhost:8000/health", "http://localhost:8001/v1/models"):
        r = subprocess.run(["curl", "-s", "-m", "5", "-o", "/dev/null", url])
        if r.returncode != 0:
            return False
    return True


def ensure_endpoints() -> None:
    for i in range(30):  # up to 15 min
        if endpoints_ok():
            return
        log(f"endpoints down, retry {i+1}/30")
        time.sleep(30)
    log("ENDPOINTS-DEAD, exiting for operator")
    sys.exit(3)


def image_for(iid: str) -> str:
    return f"starryzhang/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest"


_LITTER_MD = re.compile(r"(SUMMARY|CHANGES|FIX_|IMPLEMENTATION|SOLUTION|NOTES?).*\.md$", re.I)


def _skip_block(block: list[str]) -> bool:
    head = block[0]
    if "__pycache__" in head or head.rstrip().endswith(".pyc"):
        return True
    # Only litter-filter NEW files; never touch modified files or new source files.
    if not any(b.startswith("new file mode") for b in block):
        return False
    path = head.rstrip().split(" b/", 1)[-1]  # 'diff --git a/x b/x' -> 'x'
    name = path.rsplit("/", 1)[-1]
    # ponytail: litter heuristic -- *.bak; SUMMARY/CHANGES/FIX_/IMPLEMENTATION/
    # SOLUTION/NOTES*.md; or a repo-root README.md introduced as a new file.
    return name.endswith(".bak") or bool(_LITTER_MD.search(name)) or path == "README.md"


def strip_patch(patch: str) -> str:
    keep, block = [], []
    for l in patch.splitlines(keepends=True):
        if l.startswith("diff --git"):
            if block and not _skip_block(block):
                keep += block
            block = [l]
        else:
            block.append(l)
    if block and not _skip_block(block):
        keep += block
    return "".join(keep)


def score(iid: str, arm: str) -> bool | None:
    pred = _load_prediction(iid, arm)
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


# --- persistent per-(arm, repo) memory store ------------------------------------

def _sanitize(repo: str) -> str:
    return repo.replace("/", "__")


def mem_dir(arm: str, repo: str) -> Path:
    """Canonical, growing store for one (arm, repo). Holds the accumulated corpus."""
    return RUNS / arm / "mem" / _sanitize(repo)


def snapshot_dir(arm: str, repo: str, k: int) -> Path:
    """Per-task snapshot pointed at by instance.memory_corpus.path for task k.

    Unique name per k defeats the caching adapters' .ingested short-circuit.
    """
    return RUNS / arm / "mem_snap" / f"{_sanitize(repo)}__k{k}"


def build_snapshot(arm: str, repo: str, k: int) -> Path:
    """Copy the store (tasks 1..k-1) into a fresh snapshot dir and return it."""
    src = mem_dir(arm, repo)
    dst = snapshot_dir(arm, repo, k)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for fn in CORPUS_FILES:
        if (src / fn).exists():
            shutil.copy2(src / fn, dst / fn)
    return dst


def _load_prediction(iid: str, arm: str) -> dict | None:
    path = RUNS / arm / "predictions.jsonl"
    pred = None
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                if d.get("instance_id") == iid:
                    pred = d
    return pred


def _mem_file(arm: str) -> str:
    # structured reads project_memory.jsonl; every other arm's ingestion reads
    # documents.jsonl. Write only the file the arm actually consumes.
    return "project_memory.jsonl" if arm == "structured" else "documents.jsonl"


def _record_text(inst: dict, pred: dict | None, resolved: bool | None = None) -> str:
    # No Docker verdict here: a deployed agent writing its own memory never has
    # the official resolved label, so leaking it would give retrieval an oracle
    # signal no real deployment carries. Keep only issue title, notes, and patch.
    issue = inst.get("issue", {})
    title = str(issue.get("title", ""))
    patch = str((pred or {}).get("model_patch") or "")
    final = str((pred or {}).get("prediction") or "")
    # opt-in: label records with their outcome so a prior FAILED attempt is
    # recallable as a NEGATIVE example instead of looking like a good fix. Uses
    # the scored verdict (an oracle label) -> only for the "learn-from-failure"
    # research condition, gated behind MEMBENCH_MEMORY_LABEL_OUTCOME=1.
    outcome = ""
    if os.environ.get("MEMBENCH_MEMORY_LABEL_OUTCOME") == "1" and resolved is not None:
        outcome = (
            "Outcome: RESOLVED (this patch fixed the issue and passed tests)\n"
            if resolved else
            "Outcome: FAILED (this patch did NOT resolve the issue / broke tests -- "
            "recall it to avoid repeating this approach, do not copy it)\n"
        )
    return (
        f"Issue: {title}\n"
        f"{outcome}"
        f"Agent notes:\n{final[:1200]}\n"
        f"Patch:\n{patch[:3000]}"
    )


def write_memory(arm: str, repo: str, inst: dict, resolved: bool | None = None) -> None:
    """Append one agent-written memory record for the just-finished task.

    Idempotent (keyed by cs_<iid>) so resumed runs rebuild the store in order
    without duplicating rows.
    """
    if arm in NON_ACCUM:
        return
    iid = inst["instance_id"]
    mid = f"cs_{iid}"
    fn = _mem_file(arm)
    path = mem_dir(arm, repo) / fn
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip() and json.loads(l).get("id") == mid:
                return  # already recorded
    pred = _load_prediction(iid, arm)
    # Don't pollute the store with thin/error rows: an errored (or missing)
    # prediction row, or one with neither a patch nor analysis text, is not a
    # usable memory. A real analysis with no patch is still kept.
    patch = str((pred or {}).get("model_patch") or "")
    final = str((pred or {}).get("prediction") or "")
    if pred is None or pred.get("status") != "ok" or (not patch.strip() and not final.strip()):
        return
    text = _record_text(inst, pred, resolved)
    created_at = str(inst.get("issue", {}).get("created_at", ""))
    if arm == "structured":
        row = {
            "id": mid, "kind": "task_outcome",
            "key": str(inst.get("issue", {}).get("title", "")),
            "value": text, "source": "cross_session_write",
            "created_at": created_at, "confidence": 1.0, "evidence": [iid],
        }
    else:
        row = {
            "id": mid,
            "title": f"Prior task {iid}: {inst.get('issue', {}).get('title', '')}",
            "text": text, "source": "cross_session_write", "created_at": created_at,
        }
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


# --- agent invocation -----------------------------------------------------------

def run_agent(inst: dict, arm: str, repo: str, k: int, use_mined: bool) -> None:
    iid = inst["instance_id"]
    arm_dir = RUNS / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    tmp_inst = arm_dir / f".tmp_{iid}.jsonl"
    tmp_pred = arm_dir / f".tmp_{iid}_pred.jsonl"

    workspace = {**inst["workspace"],
                 "template_dir": str(ORIG_DATASET / inst["workspace"]["template_dir"])}
    inst = {**inst, "workspace": workspace}
    if arm not in NON_ACCUM and not use_mined:
        # Point memory at this arm's persistent store (empty for k=1).
        snap = build_snapshot(arm, repo, k)
        inst = {**inst, "memory_corpus": {**inst.get("memory_corpus", {}),
                                          "path": str(snap)}}
    elif use_mined:
        # Keep the mined corpus but absolutize its (MB-relative) path.
        mc = inst.get("memory_corpus", {})
        if mc.get("path"):
            inst = {**inst, "memory_corpus": {**mc, "path": str(MB / mc["path"])}}

    tmp_inst.write_text(json.dumps(inst) + "\n")
    cmd = ["uv", "run", "--python", "3.13", "python", "-m", "membench", "run",
           "--config", CONFIG, "--instances", str(tmp_inst),
           "--output", str(tmp_pred), "--adapter", arm]
    # ponytail: isolate OpenCode's shared SQLite db per arm so concurrent arms
    # don't hit "database is locked"; harmless for agents that ignore XDG_DATA_HOME.
    oc_data = RUNS / ".opencode_data" / arm
    oc_data.mkdir(parents=True, exist_ok=True)
    agent_env = {**os.environ, "XDG_DATA_HOME": str(oc_data)}
    try:
        r = subprocess.run(cmd, cwd=MB, capture_output=True, text=True,
                           timeout=AGENT_TIMEOUT, env=agent_env)
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


# --- resumability ---------------------------------------------------------------

def load_done_predictions(arm: str) -> set[str]:
    p = RUNS / arm / "predictions.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["instance_id"] for l in p.read_text().splitlines() if l.strip()}


def load_verdicts() -> dict[tuple[str, str], bool]:
    p = RUNS / "verdicts.jsonl"
    out: dict[tuple[str, str], bool] = {}
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                out[(d["instance_id"], d["arm"])] = d["resolved"]
    return out


# --- planning / driver ----------------------------------------------------------

def load_plan() -> list[dict]:
    sequences = [json.loads(l) for l in SEQUENCES.read_text().splitlines() if l.strip()]
    inst_by_id = {json.loads(l)["instance_id"]: json.loads(l)
                  for l in INSTANCES.read_text().splitlines() if l.strip()}
    return sequences, inst_by_id


def dry_run(arms: list[str], repos: set[str] | None) -> None:
    sequences, inst_by_id = load_plan()
    if repos is not None:
        sequences = [s for s in sequences if s["repo"] in repos]
    n_tasks = 0
    print(f"{'repo':40} {'k':>3}  {'instance':45} arms")
    for seq in sequences:
        repo = seq["repo"]
        for k, iid in enumerate(seq["instance_ids"], 1):
            present = "OK" if iid in inst_by_id else "MISSING"
            print(f"{repo:40} {k:>3}  {iid:45} {','.join(arms)} [{present}]")
            n_tasks += 1
    total_runs = n_tasks * len(arms)
    est_min = total_runs * 8
    print()
    print(f"repos: {len(sequences)}  tasks(instances): {n_tasks}  arms: {len(arms)}")
    print(f"total agent runs (task x arm): {total_runs}")
    print(f"estimated agent time @ 8 min/run: {est_min} min "
          f"= {est_min/60:.1f} h = {est_min/60/24:.1f} days (scoring extra)")
    longest = max(sequences, key=lambda s: s["n"])
    print(f"longest sequence: {longest['repo']} ({longest['n']} tasks)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the (repo, k, instance, arm) schedule and exit")
    ap.add_argument("--use-mined-corpus", action="store_true",
                    help="retrieve from the mined git corpus instead of the "
                         "agent-accumulated store (default: off)")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of arms to run (default: all)")
    ap.add_argument("--repos", default=None,
                    help="comma-separated repo names to run (default: all)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for per-task arm order; N>0 also selects "
                         "runs/cross_session_seed{N} (N==0 keeps runs/cross_session)")
    ap.add_argument("--out-root", default=None,
                    help="override the runs root entirely (e.g. runs/cs_claude_seed1); "
                         "takes precedence over --seed. Absolute or MEMBENCH_ROOT-relative")
    args = ap.parse_args(argv)

    global RUNS
    if args.out_root:
        RUNS = Path(args.out_root)
        if not RUNS.is_absolute():
            RUNS = MB / args.out_root
    elif args.seed:
        RUNS = MB / f"runs/cross_session_seed{args.seed}"

    arms = [a for a in ARMS if a in set(args.arms.split(","))] if args.arms else list(ARMS)
    memory_arms = [a for a in arms if a not in NON_ACCUM]
    repos = set(args.repos.split(",")) if args.repos else None

    if args.dry_run:
        dry_run(arms, repos)
        return

    RUNS.mkdir(parents=True, exist_ok=True)
    sequences, inst_by_id = load_plan()
    if repos is not None:
        sequences = [s for s in sequences if s["repo"] in repos]
    done_preds = {arm: load_done_predictions(arm) for arm in arms}
    verdicts = load_verdicts()
    n_tasks = sum(len(s["instance_ids"]) for s in sequences)
    log(f"cross-session start: {len(sequences)} repos, {n_tasks} tasks x {len(arms)} arms"
        f" (use_mined_corpus={args.use_mined_corpus})")

    for seq in sequences:
        repo = seq["repo"]
        ids = seq["instance_ids"]
        for k, iid in enumerate(ids, 1):
            inst = inst_by_id[iid]
            # 1. agents (each memory arm retrieves from its own accumulated store).
            #    Shuffle arm order deterministically per (seed, iid). Arms are
            #    causally independent, so this is NOT a confound control; it just
            #    spreads machine-load / cache-warmth noise evenly across arms.
            arms_order = list(arms)
            random.Random(f"{args.seed}:{iid}").shuffle(arms_order)
            todo = [a for a in arms_order if iid not in done_preds[a]]
            if todo:
                ensure_endpoints()
                # ponytail: run arms concurrently so vLLM batches them and the GPU
                # stays fed during generation; arms are causally independent. Cap via
                # MEMBENCH_ARM_CONCURRENCY (default 4) to bound KV-cache pressure.
                def _run_arm(arm):
                    t0 = time.time()
                    run_agent(inst, arm, repo, k, args.use_mined_corpus)
                    log(f"[{repo} k={k}] agent {arm}/{iid} {time.time()-t0:.0f}s")
                with ThreadPoolExecutor(max_workers=min(ARM_CONC, len(todo))) as ex:
                    list(ex.map(_run_arm, todo))
                for arm in todo:
                    done_preds[arm].add(iid)
            # 2. scoring (image cached across arms)
            need = [a for a in arms if (iid, a) not in verdicts]
            for arm in need:
                t0 = time.time()
                resolved = score(iid, arm)
                verdicts[(iid, arm)] = resolved
                with (RUNS / "verdicts.jsonl").open("a") as fh:
                    fh.write(json.dumps({"instance_id": iid, "arm": arm, "repo": repo,
                                         "k": k, "resolved": resolved}) + "\n")
                log(f"[{repo} k={k}] score {arm}/{iid} -> {resolved} {time.time()-t0:.0f}s")
            # 3. drop the image
            if need:
                subprocess.run([DOCKER_BIN, "container", "prune", "-f"], capture_output=True)
                subprocess.run([DOCKER_BIN, "rmi", image_for(iid)], capture_output=True)
            # 4. write memory for every arm (in order, idempotent) so task k+1 in
            #    this repo sees tasks 1..k. Runs even for resumed/skipped tasks.
            for arm in memory_arms:
                write_memory(arm, repo, inst, verdicts.get((iid, arm)))
    log("CROSS-SESSION-COMPLETE")


if __name__ == "__main__":
    main()
