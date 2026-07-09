"""Analyze the cross-session run: does agent-accumulated memory help later tasks?

Reads runs/cross_session/verdicts.jsonl (each line has k = position in its repo
sequence) and each arm's predictions.jsonl (usage / cost / wall time, mirroring the
last 'result' line of the per-task trajectories). Stdlib only.

Prints, per arm:
  - solve rate overall and split by sequence position (k=1, k>=2, k>=3)
  - paired delta vs the `none` control on the instances both attempted
  - mean prompt/completion tokens, cost, and wall time
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

MB = Path(os.environ.get("MEMBENCH_ROOT", Path(__file__).resolve().parent.parent))
RUNS = MB / "runs/cross_session"
ARMS = ["none", "raw_rag", "structured", "claude_mem", "mem0", "graphiti", "graphify"]


def load_verdicts() -> list[dict]:
    p = RUNS / "verdicts.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_predictions(arm: str) -> dict[str, dict]:
    p = RUNS / arm / "predictions.jsonl"
    out: dict[str, dict] = {}
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                out[d["instance_id"]] = d  # last wins
    return out


def rate(vals: list) -> str:
    resolved = [v for v in vals if v is not None]
    if not resolved:
        return "   n/a    "
    r = sum(1 for v in resolved if v) / len(resolved)
    return f"{r*100:5.1f}% {sum(1 for v in resolved if v):>3}/{len(resolved):<3}"


def main() -> None:
    verdicts = load_verdicts()
    if not verdicts:
        print(f"no verdicts at {RUNS / 'verdicts.jsonl'} -- nothing to analyze")
        return

    # (arm) -> {iid: resolved}, and (arm) -> {iid: k}
    by_arm: dict[str, dict[str, object]] = defaultdict(dict)
    kpos: dict[str, int] = {}
    for v in verdicts:
        by_arm[v["arm"]][v["instance_id"]] = v["resolved"]
        kpos[v["instance_id"]] = v.get("k", 0)

    none_map = by_arm.get("none", {})

    print("=" * 92)
    print("SOLVE RATE BY SEQUENCE POSITION")
    print("=" * 92)
    print(f"{'arm':12} {'overall':>15} {'k=1':>15} {'k>=2':>15} {'k>=3':>15}")
    for arm in ARMS:
        m = by_arm.get(arm, {})
        if not m:
            continue
        overall = list(m.values())
        k1 = [r for iid, r in m.items() if kpos.get(iid) == 1]
        k2 = [r for iid, r in m.items() if kpos.get(iid, 0) >= 2]
        k3 = [r for iid, r in m.items() if kpos.get(iid, 0) >= 3]
        print(f"{arm:12} {rate(overall):>15} {rate(k1):>15} {rate(k2):>15} {rate(k3):>15}")

    print()
    print("=" * 92)
    print("PAIRED vs none (instances both attempted; +N = memory arm solved, none did not)")
    print("=" * 92)
    print(f"{'arm':12} {'n_paired':>9} {'arm_solved':>11} {'none_solved':>12} "
          f"{'arm_only':>9} {'none_only':>10} {'delta':>7}")
    for arm in ARMS:
        if arm == "none":
            continue
        m = by_arm.get(arm, {})
        common = [iid for iid in m if iid in none_map
                  and m[iid] is not None and none_map[iid] is not None]
        if not common:
            continue
        a_solved = sum(1 for iid in common if m[iid])
        n_solved = sum(1 for iid in common if none_map[iid])
        a_only = sum(1 for iid in common if m[iid] and not none_map[iid])
        n_only = sum(1 for iid in common if none_map[iid] and not m[iid])
        print(f"{arm:12} {len(common):>9} {a_solved:>11} {n_solved:>12} "
              f"{a_only:>9} {n_only:>10} {a_solved - n_solved:>+7}")

    print()
    print("=" * 92)
    print("COST / TOKENS / TIME PER ARM (mean over predictions)")
    print("=" * 92)
    print(f"{'arm':12} {'preds':>6} {'ok':>4} {'prompt_tok':>11} {'compl_tok':>10} "
          f"{'cost_usd':>10} {'wall_s':>8}")
    for arm in ARMS:
        preds = load_predictions(arm)
        if not preds:
            continue
        rows = list(preds.values())
        ok = [r for r in rows if r.get("status") == "ok"]
        def mean(key_fn, src=ok):
            xs = [key_fn(r) for r in src if key_fn(r) is not None]
            return sum(xs) / len(xs) if xs else 0.0
        pt = mean(lambda r: (r.get("usage") or {}).get("prompt_tokens"))
        ct = mean(lambda r: (r.get("usage") or {}).get("completion_tokens"))
        cost = mean(lambda r: r.get("estimated_cost_usd"))
        wall = mean(lambda r: r.get("wall_time_sec"), src=rows)
        print(f"{arm:12} {len(rows):>6} {len(ok):>4} {pt:>11.0f} {ct:>10.0f} "
              f"{cost:>10.5f} {wall:>8.1f}")


if __name__ == "__main__":
    main()
