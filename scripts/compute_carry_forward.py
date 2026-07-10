#!/usr/bin/env python3
"""Carry-forward detection for cross-session sequences.

For each repo sequence (tasks ordered by k), decide whether task k's base
checkout already contains an earlier task j<k's fix. Method (per design doc):
compare gold patches vs base trees. Gold patches come from the SWE-bench-Live
HF dataset (local datasets have empty gold_patch); the base tree for task k is
the checked-out template repo at dataset/swebench_live_full/repos/<id_k>.

A fix is "carried forward" iff task j's gold patch REVERSE-applies cleanly to
task k's base tree (i.e. j's changes are already present there).

Run under the SWE-bench-Live venv (needs pyarrow):
    ~/SWE-bench-Live/.venv/bin/python scripts/compute_carry_forward.py

Output: dataset/cross_session/carry_forward.jsonl  +  stdout summary.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gold  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REPOS_DIR = REPO_ROOT / "dataset" / "swebench_live_full" / "repos"


def reverse_applies(patch_text, base_tree: Path):
    """True iff patch reverse-applies cleanly to base_tree (fix already present)."""
    if not patch_text.strip():
        return None  # no gold patch -> undecidable
    try:
        r = subprocess.run(
            ["patch", "-p1", "-R", "--dry-run", "--batch", "-d", str(base_tree)],
            input=patch_text, capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences", default=str(REPO_ROOT / "dataset/cross_session/sequences.jsonl"))
    ap.add_argument("--out", default=str(REPO_ROOT / "dataset/cross_session/carry_forward.jsonl"))
    args = ap.parse_args()

    gold = _gold.load_gold()
    seqs = [json.loads(l) for l in Path(args.sequences).read_text().splitlines() if l.strip()]

    rows = []
    per_repo = defaultdict(lambda: {"pairs": 0, "carried": 0, "undecidable": 0})
    for seq in seqs:
        repo = seq["repo"]
        ids = seq["instance_ids"]  # already ordered by k (k = index+1)
        for kt in range(1, len(ids)):          # k_to index (0-based); k>=2 => kt>=1
            id_to = ids[kt]
            base_tree = REPOS_DIR / id_to
            tree_ok = base_tree.is_dir()
            for kf in range(kt):               # every earlier task j<k
                id_from = ids[kf]
                patch = (gold.get(id_from) or {}).get("patch", "")
                if not tree_ok:
                    carried, method = None, "no_base_tree"
                elif not patch.strip():
                    carried, method = None, "no_gold_patch"
                else:
                    carried = reverse_applies(patch, base_tree)
                    method = "reverse_patch_apply" if carried is not None else "patch_error"
                rows.append({
                    "repo": repo,
                    "k_from": kf + 1, "k_to": kt + 1,
                    "id_from": id_from, "id_to": id_to,
                    "carried_forward": carried, "method": method,
                })
                per_repo[repo]["pairs"] += 1
                if carried is True:
                    per_repo[repo]["carried"] += 1
                elif carried is None:
                    per_repo[repo]["undecidable"] += 1

    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # summary
    tot = len(rows)
    carried = sum(1 for r in rows if r["carried_forward"] is True)
    undec = sum(1 for r in rows if r["carried_forward"] is None)
    # per-task rollup: task k carried if ANY earlier fix present
    task_carried = defaultdict(lambda: False)
    task_decided = defaultdict(lambda: False)
    for r in rows:
        if r["carried_forward"] is True:
            task_carried[r["id_to"]] = True
        if r["carried_forward"] is not None:
            task_decided[r["id_to"]] = True
    print(f"[carry_forward] wrote {args.out}")
    print(f"[carry_forward] pairs={tot} carried={carried} not={tot-carried-undec} undecidable={undec}")
    n_tasks = len(task_decided)
    n_task_carried = sum(1 for v in task_carried.values() if v)
    print(f"[carry_forward] tasks(k>=2) with >=1 decidable pair={n_tasks} "
          f"with >=1 carried-forward earlier fix={n_task_carried}")
    print("[carry_forward] per-repo (pairs/carried/undecidable):")
    for repo in sorted(per_repo):
        c = per_repo[repo]
        print(f"    {repo}: {c['pairs']}/{c['carried']}/{c['undecidable']}")
    # sample rows
    print("[carry_forward] sample rows:")
    for r in rows[:8]:
        print("   ", json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
