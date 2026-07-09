"""Build a MemBench dataset from N SWE-bench-Live lite instances.

Workspace = repo checkout at base_commit (no .git). Memory corpus = commit
history and docs strictly before the instance's created_at. Local oracle is
empty: resolution is scored by the official SWE-bench-Live Docker harness.

Run with the SWE-bench-Live venv python (needs `datasets`).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PICKS = [
    "aws-cloudformation__cfn-lint-3798",
    "python-babel__babel-1141",
    "projectmesa__mesa-2394",
    "falconry__falcon-2366",
    "pvlib__pvlib-python-2249",
]
OUT = Path("dataset/swebench_live")
N_COMMITS = 25


def sh(*args, cwd=None):
    r = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}: {r.stderr[-400:]}")
    return r.stdout


def main():
    from datasets import load_dataset

    ds = {r["instance_id"]: r for r in load_dataset("SWE-bench-Live/SWE-bench-Live", split="lite")}
    OUT.mkdir(parents=True, exist_ok=True)
    instances = []
    for iid in PICKS:
        r = ds[iid]
        repo_dir = OUT / "repos" / iid
        corpus_dir = OUT / "memory_corpora" / iid
        if not repo_dir.exists():
            tmp = OUT / "repos" / f"_{iid}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            sh("git", "clone", "--filter=blob:none", f"https://github.com/{r['repo']}.git", str(tmp))
            sh("git", "checkout", "-q", r["base_commit"], cwd=tmp)
            # mine memory corpus from history strictly before base_commit
            log = sh("git", "log", f"-{N_COMMITS + 1}", "--format=%H%x1f%aI%x1f%s%x1f%b%x1e",
                     r["base_commit"], cwd=tmp)
            events, project_memory = [], []
            entries = [e for e in log.split("\x1e") if e.strip()][1:]  # skip base commit itself
            for i, entry in enumerate(entries):
                sha, date, subject, body = (entry.strip("\n").split("\x1f") + ["", "", ""])[:4]
                events.append({"id": f"commit_{sha[:10]}", "type": "commit",
                               "title": subject, "body": body.strip(), "created_at": date})
                project_memory.append({"id": f"pm_{sha[:10]}", "kind": "episodic",
                                       "key": subject, "value": body.strip() or subject,
                                       "source": "git_history", "created_at": date,
                                       "confidence": 0.7, "evidence": [f"commit_{sha[:10]}"]})
            documents = []
            for doc in ("README.md", "README.rst", "CONTRIBUTING.md", "docs/index.md"):
                p = tmp / doc
                if p.is_file():
                    documents.append({"id": f"doc_{doc}", "title": doc,
                                      "text": p.read_text(errors="ignore")[:8000],
                                      "created_at": ""})
            corpus_dir.mkdir(parents=True, exist_ok=True)
            for name, rows in (("events.jsonl", events), ("project_memory.jsonl", project_memory),
                               ("documents.jsonl", documents)):
                (corpus_dir / name).write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            shutil.rmtree(tmp / ".git")
            tmp.rename(repo_dir)
        protected = [d for d in ("tests", "test") if (repo_dir / d).is_dir()]
        instances.append({
            "instance_id": iid,
            "repo": r["repo"],
            "base_commit": r["base_commit"],
            "issue": {"title": (r["problem_statement"].splitlines() or [""])[0][:200],
                      "body": r["problem_statement"],
                      "created_at": str(r["created_at"])},
            "gold_patch": "", "test_patch": "",
            "workspace": {"template_dir": f"repos/{iid}", "protected_paths": protected},
            "oracle": {"mode": "swebench_live_docker", "fail_to_pass": [], "pass_to_pass": []},
            "memory_corpus": {"path": f"dataset/swebench_live/memory_corpora/{iid}",
                              "cutoff_time": str(r["created_at"]),
                              "allowed_sources": ["git_history", "docs_at_or_before_cutoff"]},
            "memory_need_labels": {"requires_memory": False, "memory_type": [],
                                   "evidence_items": [], "category": "unlabeled"},
            "budgets": {"max_wall_time_sec": 1800, "max_input_tokens": 200000,
                        "max_output_tokens": 20000, "max_cost_usd": 5.0},
        })
    (OUT / "instances.jsonl").write_text("\n".join(json.dumps(x) for x in instances) + "\n")
    print(f"built {len(instances)} instances -> {OUT/'instances.jsonl'}")


if __name__ == "__main__":
    sys.exit(main())
