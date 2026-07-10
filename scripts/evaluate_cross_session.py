#!/usr/bin/env python3
"""Post-run evaluation report for cross-session MemBench runs.

Stage A (always): aggregate verdicts + official eval reports into
    <run-root>/report.md and <run-root>/report.json  (no LLM).
Stage B (--llm, on by default, degrades gracefully): ask the local qwen
    model WHY each failed (instance_id, arm) failed and write
    <run-root>/failure_analysis.jsonl plus a section in report.md.

If the LLM endpoint is unreachable, Stage B is skipped with a warning and
Stage A still produces its outputs (exit 0), so the run pipeline never breaks.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3-coder-30b"
MAX_ANALYSES = 200
# Preferred column order; any arm not listed is appended alphabetically.
ARM_ORDER = ["none", "raw_rag", "structured", "claude_mem", "mem0", "graphiti", "graphify"]

CELL = {True: "✓", False: "✗", None: "∅"}  # ✓ ✗ ∅


# --------------------------------------------------------------------------- IO
def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_report(run_root: Path, arm: str, iid: str):
    p = run_root / arm / "docker_eval" / iid / iid / "report.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def read_tail(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(errors="replace")[-n:]
    except OSError:
        return ""


def failing_tests(report):
    """Return (fail_to_pass_failures, pass_to_pass_failures) from a report.json."""
    if not report:
        return [], []
    f2p = report.get("FAIL_TO_PASS", {})
    p2p = report.get("PASS_TO_PASS", {})
    f = f2p.get("failure", []) if isinstance(f2p, dict) else []
    p = p2p.get("failure", []) if isinstance(p2p, dict) else []
    return f, p


def order_arms(arms):
    known = [a for a in ARM_ORDER if a in arms]
    extra = sorted(a for a in arms if a not in ARM_ORDER)
    return known + extra


# --------------------------------------------------------------- dataset lookup
def load_dataset_index(run_root: Path):
    """instance_id -> {title, body}. Try a few likely dataset locations."""
    idx = {}
    # run_root is e.g. <repo>/runs/cross_session ; the repo root is 2 levels up.
    candidates = []
    repo_root = run_root
    for _ in range(4):
        repo_root = repo_root.parent
        candidates.append(repo_root / "dataset" / "swebench_live_full" / "instances.jsonl")
    for path in candidates:
        if path.exists():
            for row in load_jsonl(path):
                iss = row.get("issue") or {}
                if isinstance(iss, dict):
                    idx[row["instance_id"]] = {
                        "title": iss.get("title", ""),
                        "body": iss.get("body", ""),
                    }
            break
    return idx


# ------------------------------------------------------------------- aggregate
def aggregate(run_root: Path):
    verdicts = load_jsonl(run_root / "verdicts.jsonl")
    if not verdicts:
        print(f"[evaluate] no verdicts at {run_root/'verdicts.jsonl'}", file=sys.stderr)

    arms = order_arms({v["arm"] for v in verdicts})
    iid_meta = {}  # iid -> {repo, k}
    for v in verdicts:
        iid_meta.setdefault(v["instance_id"], {"repo": v.get("repo", ""), "k": v.get("k", 0)})

    # predictions per arm: iid -> row
    preds = {}
    for arm in arms:
        preds[arm] = {r["instance_id"]: r for r in load_jsonl(run_root / arm / "predictions.jsonl")}

    # verdict lookup
    vmap = {(v["instance_id"], v["arm"]): v.get("resolved") for v in verdicts}

    # arm summary
    arm_summary = {}
    for arm in arms:
        c = {"resolved": 0, "failed": 0, "null": 0}
        for v in verdicts:
            if v["arm"] != arm:
                continue
            r = v.get("resolved")
            c["resolved" if r is True else "failed" if r is False else "null"] += 1
        total = c["resolved"] + c["failed"] + c["null"]
        c["total"] = total
        c["rate"] = (c["resolved"] / total) if total else 0.0
        arm_summary[arm] = c

    # per repo x arm
    repos = sorted({m["repo"] for m in iid_meta.values()})
    repo_arm = defaultdict(lambda: defaultdict(lambda: {"resolved": 0, "total": 0}))
    for v in verdicts:
        repo = iid_meta[v["instance_id"]]["repo"]
        cell = repo_arm[repo][v["arm"]]
        cell["total"] += 1
        if v.get("resolved") is True:
            cell["resolved"] += 1

    # ordered iids: group by repo, then by k, then id
    ordered_iids = sorted(
        iid_meta, key=lambda i: (iid_meta[i]["repo"], iid_meta[i]["k"], i)
    )

    # failure list
    failures = []
    for iid in ordered_iids:
        for arm in arms:
            if vmap.get((iid, arm)) is not False:
                continue
            report = load_report(run_root, arm, iid)
            f2p_fail, p2p_fail = failing_tests(report)
            pred = preds.get(arm, {}).get(iid, {})
            patch = pred.get("model_patch") or ""
            failures.append({
                "instance_id": iid,
                "arm": arm,
                "repo": iid_meta[iid]["repo"],
                "k": iid_meta[iid]["k"],
                "fail_to_pass_failures": f2p_fail,
                "pass_to_pass_broken": len(p2p_fail),
                "patch_size": len(patch),
                "empty_patch": not patch.strip(),
                "report_present": report is not None,
            })

    return {
        "run_root": str(run_root),
        "arms": arms,
        "repos": repos,
        "iid_meta": iid_meta,
        "ordered_iids": ordered_iids,
        "vmap": {f"{i}|{a}": r for (i, a), r in vmap.items()},
        "arm_summary": arm_summary,
        "repo_arm": {r: {a: dict(c) for a, c in am.items()} for r, am in repo_arm.items()},
        "failures": failures,
        "_preds": preds,  # kept for stage B, stripped before json dump
    }


# ------------------------------------------------------------------ markdown A
def render_markdown_a(data):
    arms = data["arms"]
    L = []
    L.append("# Cross-session evaluation report\n")
    L.append(f"Run root: `{data['run_root']}`\n")

    # 1. arm summary
    L.append("## Arm summary\n")
    L.append("| arm | resolved | failed | null | total | resolve rate |")
    L.append("|---|---|---|---|---|---|")
    for arm in arms:
        c = data["arm_summary"][arm]
        L.append(f"| {arm} | {c['resolved']} | {c['failed']} | {c['null']} | "
                 f"{c['total']} | {c['rate']*100:.1f}% |")
    L.append("")

    # 2. per repo x arm
    L.append("## Per-repo x arm (resolved / total)\n")
    L.append("| repo | " + " | ".join(arms) + " |")
    L.append("|---|" + "|".join(["---"] * len(arms)) + "|")
    for repo in data["repos"]:
        cells = []
        for arm in arms:
            c = data["repo_arm"].get(repo, {}).get(arm)
            cells.append(f"{c['resolved']}/{c['total']}" if c else "-")
        L.append(f"| {repo} | " + " | ".join(cells) + " |")
    L.append("")

    # 3. task matrix
    L.append("## Task matrix\n")
    L.append("Cells: ✓ resolved  ✗ failed  ∅ null/missing\n")
    L.append("| instance_id | k | " + " | ".join(arms) + " |")
    L.append("|---|---|" + "|".join(["---"] * len(arms)) + "|")
    last_repo = None
    for iid in data["ordered_iids"]:
        repo = data["iid_meta"][iid]["repo"]
        if repo != last_repo:
            L.append(f"| **{repo}** | | " + " | ".join([""] * len(arms)) + " |")
            last_repo = repo
        k = data["iid_meta"][iid]["k"]
        cells = [CELL.get(data["vmap"].get(f"{iid}|{arm}")) for arm in arms]
        L.append(f"| {iid} | {k} | " + " | ".join(cells) + " |")
    L.append("")

    # 4. failure list
    L.append(f"## Failure list ({len(data['failures'])} failed cells)\n")
    L.append("| instance_id | arm | failing FAIL_TO_PASS (first 5) | p2p broken | "
             "patch bytes | empty | report |")
    L.append("|---|---|---|---|---|---|---|")
    for f in data["failures"]:
        tests = f["fail_to_pass_failures"][:5]
        more = len(f["fail_to_pass_failures"]) - len(tests)
        tstr = "<br>".join(t.split("::")[-1] for t in tests) if tests else "(none listed)"
        if more > 0:
            tstr += f"<br>(+{more} more)"
        L.append(
            f"| {f['instance_id']} | {f['arm']} | {tstr} | {f['pass_to_pass_broken']} | "
            f"{f['patch_size']} | {'yes' if f['empty_patch'] else ''} | "
            f"{'ok' if f['report_present'] else 'MISSING'} |"
        )
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------- LLM (B)
def endpoint_alive(timeout=3):
    try:
        import requests
        requests.get(ENDPOINT.replace("/chat/completions", "/models"), timeout=timeout)
        return True
    except Exception:
        return False


def ask_llm(prompt, timeout=60):
    import requests
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a terse senior engineer doing"
             " root-cause analysis of a failed automated code fix."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }
    r = requests.post(ENDPOINT, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


CATEGORIES = ["wrong-file", "incomplete-fix", "broke-other-tests",
              "no-patch", "misunderstood-issue", "test-env"]


def build_prompt(f, dataset_idx, run_root):
    iid, arm = f["instance_id"], f["arm"]
    meta = dataset_idx.get(iid, {})
    title = meta.get("title", "") or "(unknown)"
    body = (meta.get("body", "") or "")[:800]
    pred = f["_pred"]
    patch = (pred.get("model_patch") or "")[:3000]
    tests = "\n".join(f["fail_to_pass_failures"][:10]) or "(no FAIL_TO_PASS failures recorded)"
    log = read_tail(run_root / arm / "docker_eval" / iid / iid / "post_patch_log.txt", 1500)
    return (
        f"Issue title: {title}\n"
        f"Issue body (truncated):\n{body}\n\n"
        f"The agent produced this patch (truncated):\n```diff\n{patch}\n```\n\n"
        f"Failing FAIL_TO_PASS tests:\n{tests}\n\n"
        f"pass_to_pass regressions: {f['pass_to_pass_broken']}\n"
        f"empty patch: {f['empty_patch']}\n\n"
        f"Tail of pytest log after applying the patch:\n{log}\n\n"
        f"In 3-5 sentences explain WHY this fix failed. "
        f"Start with exactly one root-cause category from this list on the first line "
        f"as `category: <one-of {'/'.join(CATEGORIES)}>`, then the explanation."
    )


def parse_category(text):
    first = text.splitlines()[0].lower() if text else ""
    for c in CATEGORIES:
        if c in first:
            return c
    for c in CATEGORIES:
        if c in text.lower():
            return c
    return "uncategorized"


def run_stage_b(data, run_root):
    failures = data["failures"]
    if not endpoint_alive():
        print(f"[evaluate] WARNING: LLM endpoint {ENDPOINT} unreachable "
              f"-- skipping Stage B (qualitative review). Stage A outputs written.",
              file=sys.stderr)
        return None

    dataset_idx = load_dataset_index(run_root)
    todo = failures
    if len(todo) > MAX_ANALYSES:
        print(f"[evaluate] capping analyses at {MAX_ANALYSES} of {len(todo)} failures",
              file=sys.stderr)
        todo = todo[:MAX_ANALYSES]

    results = []
    out_path = run_root / "failure_analysis.jsonl"
    with out_path.open("w") as fh:
        for i, f in enumerate(todo, 1):
            f = dict(f)
            f["_pred"] = data["_preds"].get(f["arm"], {}).get(f["instance_id"], {})
            try:
                text = ask_llm(build_prompt(f, dataset_idx, run_root))
                cat = parse_category(text)
            except Exception as e:
                text = f"(LLM call failed: {e})"
                cat = "error"
            rec = {"instance_id": f["instance_id"], "arm": f["arm"],
                   "category": cat, "analysis": text}
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            print(f"[evaluate] analyzed {i}/{len(todo)} "
                  f"{f['instance_id']} [{f['arm']}] -> {cat}", file=sys.stderr)
    print(f"[evaluate] wrote {out_path}", file=sys.stderr)
    return results


def render_markdown_b(results):
    L = ["", "## Qualitative failure review\n",
         f"LLM ({MODEL}) root-cause analysis of {len(results)} failed cells.\n"]
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        L.append(f"### {cat} ({len(by_cat[cat])})\n")
        for r in by_cat[cat]:
            L.append(f"- **{r['instance_id']}** [{r['arm']}]: {r['analysis']}")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="runs/cross_session",
                    help="run root containing verdicts.jsonl and per-arm dirs")
    ap.add_argument("--llm", dest="llm", action="store_true", default=True,
                    help="run Stage B qualitative LLM review (default on)")
    ap.add_argument("--no-llm", dest="llm", action="store_false",
                    help="skip Stage B")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        print(f"[evaluate] run root not found: {run_root}", file=sys.stderr)
        return 2

    data = aggregate(run_root)

    # Stage A outputs
    md = render_markdown_a(data)
    b_results = None
    if args.llm:
        b_results = run_stage_b(data, run_root)
        if b_results:
            md += "\n" + render_markdown_b(b_results)

    (run_root / "report.md").write_text(md)
    dump = {k: v for k, v in data.items() if not k.startswith("_")}
    (run_root / "report.json").write_text(json.dumps(dump, indent=1))
    print(f"[evaluate] wrote {run_root/'report.md'} and {run_root/'report.json'}",
          file=sys.stderr)
    print(f"[evaluate] arms={len(data['arms'])} iids={len(data['ordered_iids'])} "
          f"failures={len(data['failures'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
