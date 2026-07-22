"""Build the oracle sidecars from the HF SWE-bench-Live gold patches.

The gold patch is NOT in the local dataset (gold_patch is blank in
dataset/**/instances.jsonl); it lives in the HF SWE-bench-Live dataset used at
scoring time. Run once (on the box with HF access, e.g. pectra) to produce both:

    dataset/cross_session/oracle_files.json    {instance_id: [gold-edited files]}
    dataset/cross_session/oracle_patches.json  {instance_id: "<gold unified diff>"}

    python3 scripts/build_oracle_files.py

The `oracle`/`oracle_files` arm injects the file LIST (perfect localization, the
E2 upper bound); the `oracle_strong` arm injects the DIFF itself (near-answer
upper bound, E2').
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MB = Path(__file__).resolve().parent.parent
OUT = MB / "dataset/cross_session/oracle_files.json"
OUT_PATCHES = MB / "dataset/cross_session/oracle_patches.json"
DATASET = "SWE-bench-Live/SWE-bench-Live"
SPLIT = "lite"

_DIFF_GIT = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+)$", re.MULTILINE)


def gold_files(patch: str) -> list[str]:
    """Files touched by a gold patch, in order, deduped."""
    seen: dict[str, None] = {}
    for m in _DIFF_GIT.finditer(patch or ""):
        seen.setdefault(m.group("b"), None)
    return list(seen)


def main() -> None:
    from datasets import load_dataset  # heavy import; only when actually building

    ids = {
        json.loads(l)["instance_id"]
        for l in (MB / "dataset/cross_session/instances.jsonl").read_text().splitlines()
        if l.strip()
    }
    ds = load_dataset(DATASET, split=SPLIT)
    out: dict[str, list[str]] = {}
    patches: dict[str, str] = {}
    for row in ds:
        iid = row["instance_id"]
        if iid in ids:
            patch = row.get("patch", "") or ""
            out[iid] = gold_files(patch)
            patches[iid] = patch
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    OUT_PATCHES.write_text(json.dumps(patches, indent=1))
    covered = sum(1 for v in out.values() if v)
    with_patch = sum(1 for v in patches.values() if v.strip())
    print(f"wrote {OUT}: {len(out)}/{len(ids)} instances, {covered} with files")
    print(f"wrote {OUT_PATCHES}: {len(patches)}/{len(ids)} instances, {with_patch} with diff")


def _selfcheck() -> None:
    patch = (
        "diff --git a/src/foo.py b/src/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/src/foo.py b/src/foo.py\n"  # dup
        "diff --git a/bar/baz.py b/bar/baz.py\n@@ -1 +1 @@\n-c\n+d\n"
    )
    assert gold_files(patch) == ["src/foo.py", "bar/baz.py"], gold_files(patch)
    assert gold_files("") == []


if __name__ == "__main__":
    _selfcheck()
    main()
