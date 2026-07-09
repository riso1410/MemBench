"""Build the cross-session benchmark dataset from the existing 300 instances.

Groups the swebench_live_full instances by repo, keeps repos with >= MIN_SEQ
instances, orders each repo's instances chronologically by issue.created_at, and
emits:
  dataset/cross_session/sequences.jsonl  -- one line per repo: {repo, instance_ids, n}
  dataset/cross_session/instances.jsonl  -- the full instance dicts, flattened in
                                            (repo, chronological) order

No new task mining: this only re-partitions and re-orders the existing 300.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MB = Path(os.environ.get("MEMBENCH_ROOT", Path(__file__).resolve().parent.parent))
SRC = MB / "dataset/swebench_live_full/instances.jsonl"
OUT_DIR = MB / "dataset/cross_session"
MIN_SEQ = 3


def created_at(instance: dict) -> tuple:
    raw = str(instance.get("issue", {}).get("created_at", ""))
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return (0, datetime.strptime(raw, fmt).timestamp())
        except ValueError:
            continue
    try:
        return (0, datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return (1, raw)  # unparseable -> sort last, lexically stable


def main() -> None:
    instances = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for inst in instances:
        by_repo[inst["repo"]].append(inst)

    kept = {repo: rows for repo, rows in by_repo.items() if len(rows) >= MIN_SEQ}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seq_lines: list[str] = []
    flat_lines: list[str] = []
    total = 0
    for repo in sorted(kept):
        rows = sorted(kept[repo], key=created_at)
        ids = [r["instance_id"] for r in rows]
        seq_lines.append(json.dumps({"repo": repo, "n": len(ids), "instance_ids": ids}))
        flat_lines.extend(json.dumps(r) for r in rows)
        total += len(ids)

    (OUT_DIR / "sequences.jsonl").write_text("\n".join(seq_lines) + "\n")
    (OUT_DIR / "instances.jsonl").write_text("\n".join(flat_lines) + "\n")

    print(f"source instances: {len(instances)}  repos: {len(by_repo)}")
    print(f"kept repos (>= {MIN_SEQ} instances): {len(kept)}  instances: {total}")
    print(f"wrote {OUT_DIR / 'sequences.jsonl'}")
    print(f"wrote {OUT_DIR / 'instances.jsonl'}")
    longest = max(kept.items(), key=lambda kv: len(kv[1]))
    print(f"longest sequence: {longest[0]} ({len(longest[1])} tasks)")


if __name__ == "__main__":
    main()
