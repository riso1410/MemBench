#!/usr/bin/env python3
"""Heuristic memory-need labels for cross-session tasks.

For each task k>=2 in a repo sequence: does its gold fix touch any file that an
earlier task j<k's fix also touched (same repo)? If so the earlier session's
experience is plausibly relevant -> requires_memory=True.

Fix-file source: gold patches from the SWE-bench-Live HF dataset (preferred).
Fallback when gold is unavailable: the none-arm model_patch from the run.

Run under the SWE-bench-Live venv (needs pyarrow):
    ~/SWE-bench-Live/.venv/bin/python scripts/memory_need_labels.py

Output: dataset/cross_session/memory_need_labels.jsonl  +  stdout counts.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gold  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def none_patch_files(run_root: Path):
    """instance_id -> set(files) from none-arm model_patch, for fallback."""
    out = {}
    p = run_root / "none" / "predictions.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["instance_id"]] = _gold.gold_files(r.get("model_patch") or "")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences", default=str(REPO_ROOT / "dataset/cross_session/sequences.jsonl"))
    ap.add_argument("--run-root", default=str(REPO_ROOT / "runs/cross_session"),
                    help="used only for none-arm fallback when gold is missing")
    ap.add_argument("--out", default=str(REPO_ROOT / "dataset/cross_session/memory_need_labels.jsonl"))
    args = ap.parse_args()

    gold = _gold.load_gold()
    fallback = none_patch_files(Path(args.run_root))
    seqs = [json.loads(l) for l in Path(args.sequences).read_text().splitlines() if l.strip()]

    def files_for(iid):
        patch = (gold.get(iid) or {}).get("patch", "")
        if patch.strip():
            return _gold.gold_files(patch), "gold"
        if iid in fallback and fallback[iid]:
            return fallback[iid], "none_arm_fallback"
        return set(), "none"

    rows = []
    for seq in seqs:
        repo = seq["repo"]
        ids = seq["instance_ids"]
        earlier = []  # list of (iid, files)
        for k, iid in enumerate(ids, start=1):
            f, src = files_for(iid)
            if k >= 2:
                overlap = set()
                touched_by = []
                for (pid, pf) in earlier:
                    ov = f & pf
                    if ov:
                        overlap |= ov
                        touched_by.append(pid)
                rows.append({
                    "instance_id": iid, "repo": repo, "k": k,
                    "requires_memory": bool(overlap),
                    "overlap_files": sorted(overlap),
                    "earlier_tasks_overlapped": touched_by,
                    "fix_file_source": src,
                })
            earlier.append((iid, f))

    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    n = len(rows)
    req = sum(1 for r in rows if r["requires_memory"])
    from collections import Counter
    src = Counter(r["fix_file_source"] for r in rows)
    print(f"[memory_need] wrote {args.out}")
    print(f"[memory_need] tasks(k>=2)={n} requires_memory={req} not={n-req}")
    print(f"[memory_need] fix_file_source: {dict(src)}")
    print("[memory_need] requires_memory=True sample:")
    for r in [r for r in rows if r["requires_memory"]][:10]:
        print(f"    {r['instance_id']} (k={r['k']}) overlaps {r['earlier_tasks_overlapped']} on {r['overlap_files']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
