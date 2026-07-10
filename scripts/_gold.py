"""Shared gold-patch loader for cross-session analysis scripts.

Gold patches are empty in the local MemBench datasets, so we pull them from the
SWE-bench-Live HF dataset copy on disk (parquet). Reading parquet needs pyarrow,
which lives in the SWE-bench-Live venv:

    ~/SWE-bench-Live/.venv/bin/python scripts/<script>.py

load_gold() returns {instance_id: {"patch","base_commit","created_at"}}.
"""
import functools
import glob
import os

HF_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--SWE-bench-Live--SWE-bench-Live/snapshots"
)


def _parquet_files():
    return sorted(glob.glob(os.path.join(HF_SNAP, "*", "data", "full-*.parquet")))


@functools.lru_cache(maxsize=1)
def load_gold():
    import pyarrow.parquet as pq  # only needed here; SWE-bench-Live venv has it

    g = {}
    for f in _parquet_files():
        t = pq.read_table(f, columns=["instance_id", "patch", "base_commit", "created_at"])
        for r in t.to_pylist():
            g[r["instance_id"]] = r
    return g


def gold_files(patch):
    """Set of repo-relative file paths a unified diff touches (b/ side)."""
    files = set()
    for line in (patch or "").splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b = parts[3]
                files.add(b[2:] if b.startswith("b/") else b)
        elif line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            p = line[4:].strip().split("\t")[0]
            files.add(p[2:] if p.startswith("b/") else p)
    return files
