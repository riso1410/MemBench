"""Build MemBench instances for the FULL SWE-bench-Live lite split (300 tasks).

Per instance: workspace checkout at base_commit (git archive from a cached
partial clone) + memory corpus mined from pre-base_commit history. Resumable:
skips instances already built. Run with the SWE-bench-Live venv python.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("dataset/swebench_live_full")
CACHE = OUT / ".repo_cache"
N_COMMITS = 25


def sh(*args, cwd=None, timeout=600):
    r = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])}...: {r.stderr[-300:]}")
    return r.stdout


def cached_clone(repo: str) -> Path:
    d = CACHE / repo.replace("/", "__")
    if not (d / "HEAD").exists() and not (d / ".git").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        sh("git", "clone", "--filter=blob:none", "--bare",
           f"https://github.com/{repo}.git", str(d), timeout=1800)
    return d


def build_one(r) -> dict | None:
    iid = r["instance_id"]
    repo_dir = OUT / "repos" / iid
    corpus_dir = OUT / "memory_corpora" / iid
    if not (repo_dir / ".built").exists():
        git = cached_clone(r["repo"])
        shutil.rmtree(repo_dir, ignore_errors=True)
        repo_dir.mkdir(parents=True)
        # checkout via archive (no .git in workspace)
        tar = subprocess.Popen(["git", "--git-dir", str(git), "archive", r["base_commit"]],
                               stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(repo_dir)], stdin=tar.stdout, check=True)
        tar.wait()
        if tar.returncode != 0:
            # commit not in clone (force-push etc.) — fetch it explicitly
            sh("git", "--git-dir", str(git), "fetch", "origin", r["base_commit"], timeout=1800)
            tar = subprocess.Popen(["git", "--git-dir", str(git), "archive", r["base_commit"]],
                                   stdout=subprocess.PIPE)
            subprocess.run(["tar", "-x", "-C", str(repo_dir)], stdin=tar.stdout, check=True)
            tar.wait()
        # memory corpus from history strictly before base_commit
        log = sh("git", "--git-dir", str(git), "log", f"-{N_COMMITS + 1}",
                 "--format=%H%x1f%aI%x1f%s%x1f%b%x1e", r["base_commit"])
        events, pm = [], []
        for entry in [e for e in log.split("\x1e") if e.strip()][1:]:
            sha, date, subject, body = (entry.strip("\n").split("\x1f") + ["", "", ""])[:4]
            events.append({"id": f"commit_{sha[:10]}", "type": "commit", "title": subject,
                           "body": body.strip(), "created_at": date})
            pm.append({"id": f"pm_{sha[:10]}", "kind": "episodic", "key": subject,
                       "value": body.strip() or subject, "source": "git_history",
                       "created_at": date, "confidence": 0.7,
                       "evidence": [f"commit_{sha[:10]}"]})
        docs = []
        for doc in ("README.md", "README.rst", "CONTRIBUTING.md"):
            p = repo_dir / doc
            if p.is_file():
                docs.append({"id": f"doc_{doc}", "title": doc,
                             "text": p.read_text(errors="ignore")[:8000], "created_at": ""})
        corpus_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in (("events.jsonl", events), ("project_memory.jsonl", pm),
                           ("documents.jsonl", docs)):
            (corpus_dir / name).write_text(
                "\n".join(json.dumps(x) for x in rows) + ("\n" if rows else ""))
        (repo_dir / ".built").write_text("ok")
    protected = [d for d in ("tests", "test") if (repo_dir / d).is_dir()]
    return {
        "instance_id": iid, "repo": r["repo"], "base_commit": r["base_commit"],
        "issue": {"title": (r["problem_statement"].splitlines() or [""])[0][:200],
                  "body": r["problem_statement"], "created_at": str(r["created_at"])},
        "gold_patch": "", "test_patch": "",
        "workspace": {"template_dir": f"repos/{iid}", "protected_paths": protected},
        "oracle": {"mode": "swebench_live_docker", "fail_to_pass": [], "pass_to_pass": []},
        "memory_corpus": {"path": f"dataset/swebench_live_full/memory_corpora/{iid}",
                          "cutoff_time": str(r["created_at"]),
                          "allowed_sources": ["git_history", "docs_at_or_before_cutoff"]},
        "memory_need_labels": {"requires_memory": False, "memory_type": [],
                               "evidence_items": [], "category": "unlabeled"},
        "budgets": {"max_wall_time_sec": 1800, "max_input_tokens": 200000,
                    "max_output_tokens": 20000, "max_cost_usd": 5.0},
    }


def main():
    from datasets import load_dataset

    ds = load_dataset("SWE-bench-Live/SWE-bench-Live", split="lite")
    OUT.mkdir(parents=True, exist_ok=True)
    instances, failed = [], []
    for i, r in enumerate(ds):
        try:
            inst = build_one(r)
            instances.append(inst)
            print(f"[{i+1}/{len(ds)}] {r['instance_id']} ok", flush=True)
        except Exception as e:
            failed.append(r["instance_id"])
            print(f"[{i+1}/{len(ds)}] {r['instance_id']} FAILED: {str(e)[:150]}", flush=True)
    (OUT / "instances.jsonl").write_text("\n".join(json.dumps(x) for x in instances) + "\n")
    (OUT / "build_failures.json").write_text(json.dumps(failed, indent=1))
    print(f"built {len(instances)} instances, {len(failed)} failures")


if __name__ == "__main__":
    sys.exit(main())
