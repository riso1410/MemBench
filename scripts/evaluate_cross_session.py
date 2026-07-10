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
import math
import random
import statistics
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


# ============================================================================
# Reviewer-mandated extended analyses (Stage A2, no LLM).
#   a. k>=2 headline split (k=1 shown separately as noise control)
#   b. McNemar exact + paired bootstrap 95% CI, each memory arm vs none
#   c. injected retrieved-memory volume per task (chars/tokens)
#   d. citation parsing (did the agent cite retrieved memory ids?)
#   e. date-split (contamination gradient by issue created_at)
#   + carry-forward and memory-need stratification (external label files)
# ============================================================================

def find_dataset_dir(run_root: Path):
    p = run_root
    for _ in range(5):
        p = p.parent
        d = p / "dataset" / "cross_session"
        if d.exists():
            return d
    return None


def load_created_at(dataset_dir):
    """instance_id -> created_at string, from cross_session/instances.jsonl."""
    idx = {}
    if not dataset_dir:
        return idx
    for row in load_jsonl(dataset_dir / "instances.jsonl"):
        iss = row.get("issue") or {}
        if isinstance(iss, dict) and iss.get("created_at"):
            idx[row["instance_id"]] = iss["created_at"]
    return idx


def rate_row(iids, arm, vmap):
    res = fail = null = 0
    for iid in iids:
        v = vmap.get(f"{iid}|{arm}")
        if v is True:
            res += 1
        elif v is False:
            fail += 1
        else:
            null += 1
    total = res + fail + null
    return {"resolved": res, "failed": fail, "null": null, "total": total,
            "rate": (res / total) if total else 0.0}


def arm_table(iids, arms, vmap):
    return {arm: rate_row(iids, arm, vmap) for arm in arms}


def _resolved01(iid, arm, vmap):
    return 1 if vmap.get(f"{iid}|{arm}") is True else 0


def mcnemar_exact(pairs):
    """pairs: list of (none01, arm01). Two-sided exact binomial (sign test)."""
    b = sum(1 for n, a in pairs if a == 1 and n == 0)  # arm wins
    c = sum(1 for n, a in pairs if n == 1 and a == 0)  # none wins
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return b, c, min(1.0, p)


def bootstrap_ci(pairs, iters=10000, seed=1234):
    """Paired bootstrap 95% CI on rate difference (arm - none)."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0
    base = sum(a - nn for nn, a in pairs) / n
    rnd = random.Random(seed)
    diffs = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            nn, a = pairs[rnd.randrange(n)]
            s += a - nn
        diffs.append(s / n)
    diffs.sort()
    return base, diffs[int(0.025 * iters)], diffs[min(iters - 1, int(0.975 * iters))]


def significance(iids, arms, vmap, baseline="none"):
    out = {}
    for arm in arms:
        if arm == baseline:
            continue
        pairs = [(_resolved01(i, baseline, vmap), _resolved01(i, arm, vmap)) for i in iids]
        b, c, p = mcnemar_exact(pairs)
        diff, lo, hi = bootstrap_ci(pairs)
        out[arm] = {"n_pairs": len(pairs), "arm_wins": b, "none_wins": c,
                    "mcnemar_p": round(p, 4), "rate_diff": round(diff, 4),
                    "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)}
    return out


def injected_volume(iids, arms, preds):
    """Per arm: chars/token stats of retrieved_memory injected per task (k>=2)."""
    out = {}
    for arm in arms:
        chars, ntoks, nitems = [], [], []
        for iid in iids:
            rm = (preds.get(arm, {}).get(iid, {}) or {}).get("retrieved_memory") or []
            if not isinstance(rm, list):
                rm = []
            c = sum(len(str(it.get("text", ""))) for it in rm if isinstance(it, dict))
            chars.append(c)
            ntoks.append(c // 4)  # rough token estimate (~4 chars/token)
            nitems.append(len(rm))
        def med(x):
            return round(statistics.median(x), 1) if x else 0.0
        out[arm] = {
            "tasks": len(iids),
            "median_chars": med(chars), "max_chars": max(chars) if chars else 0,
            "median_est_tokens": med(ntoks), "max_est_tokens": max(ntoks) if ntoks else 0,
            "median_items": med(nitems), "max_items": max(nitems) if nitems else 0,
        }
    return out


def candidate_ids(item):
    ids = set()
    iid = str(item.get("item_id") or "")
    if iid:
        ids.add(iid)
        if iid.startswith("cs_"):
            ids.add(iid[3:])
    md = item.get("metadata") or {}
    for ev in (md.get("evidence") or []):
        ids.add(str(ev))
    return {s for s in ids if s and len(s) >= 6}


def citations(iids, arms, preds, vmap, baseline="none"):
    """Did the agent cite a retrieved memory id in its final message?"""
    out = {}
    for arm in arms:
        if arm == baseline:
            continue
        tasks_with_mem = cited = cited_res = cited_fail = 0
        for iid in iids:
            pr = preds.get(arm, {}).get(iid, {}) or {}
            rm = pr.get("retrieved_memory") or []
            if not isinstance(rm, list) or not rm:
                continue
            tasks_with_mem += 1
            msg = str(pr.get("prediction") or "").lower()
            ids = set()
            for it in rm:
                if isinstance(it, dict):
                    ids |= candidate_ids(it)
            if any(cid.lower() in msg for cid in ids):
                cited += 1
                if vmap.get(f"{iid}|{arm}") is True:
                    cited_res += 1
                else:
                    cited_fail += 1
        out[arm] = {
            "tasks_with_memory": tasks_with_mem,
            "cited": cited,
            "citation_rate": round(cited / tasks_with_mem, 3) if tasks_with_mem else 0.0,
            "cited_and_resolved": cited_res,
            "cited_and_failed": cited_fail,
        }
    return out


def date_split(iids, arms, vmap, created_at):
    dated = [(i, created_at[i]) for i in iids if i in created_at]
    if len(dated) < 2:
        return None
    dates = sorted(d for _, d in dated)
    median = dates[len(dates) // 2]
    early = [i for i, d in dated if d < median]
    late = [i for i, d in dated if d >= median]
    return {
        "median_date": median,
        "n_early": len(early), "n_late": len(late),
        "early": {arm: rate_row(early, arm, vmap) for arm in arms},
        "late": {arm: rate_row(late, arm, vmap) for arm in arms},
        "undated": len(iids) - len(dated),
    }


def load_label_map(path: Path, key, value):
    m = {}
    for r in load_jsonl(path):
        if key in r:
            m[r[key]] = r.get(value)
    return m


def carry_forward_map(dataset_dir):
    """id_to -> carried (True if any earlier fix present in its base)."""
    m = defaultdict(lambda: False)
    decided = set()
    if not dataset_dir:
        return {}, set()
    for r in load_jsonl(dataset_dir / "carry_forward.jsonl"):
        idt = r.get("id_to")
        if r.get("carried_forward") is True:
            m[idt] = True
        if r.get("carried_forward") is not None:
            decided.add(idt)
    return dict(m), decided


def stratify(iids, arms, vmap, group_of):
    """group_of: iid -> group label (or None to drop). Return per-group arm rates + lift."""
    groups = defaultdict(list)
    for iid in iids:
        g = group_of.get(iid)
        if g is not None:
            groups[g].append(iid)
    out = {}
    for g, gids in groups.items():
        tbl = rate_row(gids, "none", vmap)["rate"] if "none" in arms else 0.0
        out[str(g)] = {
            "n": len(gids),
            "arms": {arm: rate_row(gids, arm, vmap) for arm in arms},
            "lift_vs_none": {arm: round(rate_row(gids, arm, vmap)["rate"] - tbl, 4)
                             for arm in arms if arm != "none"},
        }
    return out


def analyze_extended(run_root: Path, data):
    arms = data["arms"]
    vmap = data["vmap"]
    preds = data["_preds"]
    kmeta = data["iid_meta"]
    all_iids = data["ordered_iids"]
    k2 = [i for i in all_iids if kmeta[i]["k"] >= 2]
    k1 = [i for i in all_iids if kmeta[i]["k"] < 2]

    dataset_dir = find_dataset_dir(run_root)
    created_at = load_created_at(dataset_dir)
    cf_map, cf_decided = carry_forward_map(dataset_dir)
    mn_map = load_label_map(dataset_dir / "memory_need_labels.jsonl",
                            "instance_id", "requires_memory") if dataset_dir else {}

    cf_group = {i: ("carried" if cf_map.get(i) else "not_carried")
                for i in k2 if i in cf_decided}
    mn_group = {i: ("requires_memory" if mn_map.get(i) else "no_memory_need")
                for i in k2 if i in mn_map}

    return {
        "n_k2": len(k2), "n_k1": len(k1),
        "headline_k2": arm_table(k2, arms, vmap),
        "noise_control_k1": arm_table(k1, arms, vmap),
        "all_k": arm_table(all_iids, arms, vmap),
        "significance_k2": significance(k2, arms, vmap),
        "injected_volume_k2": injected_volume(k2, arms, preds),
        "citations_k2": citations(k2, arms, preds, vmap),
        "date_split": date_split(all_iids, arms, vmap, created_at),
        "carry_forward_strat_k2": stratify(k2, arms, vmap, cf_group),
        "memory_need_strat_k2": stratify(k2, arms, vmap, mn_group),
        "labels_present": {
            "carry_forward": bool(cf_map or cf_decided),
            "memory_need": bool(mn_map),
            "created_at": bool(created_at),
        },
    }


def _rate_table_md(L, title, arms, table):
    L.append(f"### {title}\n")
    L.append("| arm | resolved | failed | null | total | resolve rate |")
    L.append("|---|---|---|---|---|---|")
    for arm in arms:
        c = table[arm]
        L.append(f"| {arm} | {c['resolved']} | {c['failed']} | {c['null']} | "
                 f"{c['total']} | {c['rate']*100:.1f}% |")
    L.append("")


def render_markdown_ext(ext, arms):
    L = ["", "## Extended analyses (reviewer-mandated)\n"]
    L.append(f"Primary headline is computed on **k>=2 tasks only** ({ext['n_k2']} tasks); "
             f"k=1 prompts are identical across arms and are shown separately as a "
             f"noise control ({ext['n_k1']} tasks).\n")

    # a. headline split
    L.append("## a. Headline resolve rates\n")
    _rate_table_md(L, "PRIMARY -- k>=2 tasks", arms, ext["headline_k2"])
    _rate_table_md(L, "Noise control -- k=1 tasks (identical prompts across arms)",
                   arms, ext["noise_control_k1"])
    _rate_table_md(L, "Secondary -- all k", arms, ext["all_k"])

    # b. significance
    L.append("## b. Significance vs `none` (k>=2, paired)\n")
    L.append("McNemar exact two-sided p on paired task verdicts; paired bootstrap "
             "(10k) 95% CI on the resolve-rate difference (arm - none). "
             "Null/missing verdicts are treated as unresolved.\n")
    L.append("| arm | arm_wins | none_wins | McNemar p | rate diff | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for arm in arms:
        s = ext["significance_k2"].get(arm)
        if not s:
            continue
        L.append(f"| {arm} | {s['arm_wins']} | {s['none_wins']} | {s['mcnemar_p']} | "
                 f"{s['rate_diff']*100:+.1f}pp | [{s['ci95_lo']*100:+.1f}, {s['ci95_hi']*100:+.1f}]pp |")
    L.append("")

    # c. injected volume
    L.append("## c. Injected retrieved-memory volume (k>=2, per task)\n")
    L.append("Chars/tokens of `retrieved_memory` injected per task. Tokens are a rough "
             "~4-chars/token estimate.\n")
    L.append("| arm | median chars | max chars | median est tokens | max est tokens | median items | max items |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in arms:
        v = ext["injected_volume_k2"][arm]
        L.append(f"| {arm} | {v['median_chars']:.0f} | {v['max_chars']} | "
                 f"{v['median_est_tokens']:.0f} | {v['max_est_tokens']} | "
                 f"{v['median_items']:.0f} | {v['max_items']} |")
    L.append("")

    # d. citations
    L.append("## d. Memory citation parsing (k>=2)\n")
    L.append("Fraction of memory-injected tasks whose final agent message cites at least "
             "one retrieved memory id (item_id, encoded prior instance id, or evidence id). "
             "The agent prompt instructs it to state which memory item ids it used.\n")
    L.append("| arm | tasks w/ memory | cited | citation rate | cited&resolved | cited&failed |")
    L.append("|---|---|---|---|---|---|")
    for arm in arms:
        c = ext["citations_k2"].get(arm)
        if not c:
            continue
        L.append(f"| {arm} | {c['tasks_with_memory']} | {c['cited']} | "
                 f"{c['citation_rate']*100:.1f}% | {c['cited_and_resolved']} | {c['cited_and_failed']} |")
    L.append("")

    # e. date split
    L.append("## e. Date split (contamination gradient)\n")
    ds = ext["date_split"]
    if not ds:
        L.append("_Insufficient created_at data for a date split._\n")
    else:
        L.append(f"Split at median issue date `{ds['median_date']}` "
                 f"(early n={ds['n_early']}, late n={ds['n_late']}, undated={ds['undated']}).\n")
        L.append("| arm | early rate | late rate |")
        L.append("|---|---|---|")
        for arm in arms:
            e = ds["early"][arm]
            la = ds["late"][arm]
            L.append(f"| {arm} | {e['resolved']}/{e['total']} ({e['rate']*100:.0f}%) | "
                     f"{la['resolved']}/{la['total']} ({la['rate']*100:.0f}%) |")
        L.append("")

    # carry-forward stratification
    L.append("## Carry-forward stratification (k>=2)\n")
    L.append("Tasks split by whether the task's base checkout already contains an earlier "
             "task's fix (`dataset/cross_session/carry_forward.jsonl`). Lift = arm rate - none rate.\n")
    _render_strat(L, ext["carry_forward_strat_k2"], arms)

    # memory-need stratification
    L.append("## Memory-need stratification (k>=2)\n")
    L.append("Tasks split by whether the gold fix touches a file an earlier task's fix "
             "also touched (`dataset/cross_session/memory_need_labels.jsonl`).\n")
    _render_strat(L, ext["memory_need_strat_k2"], arms)

    return "\n".join(L)


def _render_strat(L, strat, arms):
    if not strat:
        L.append("_Label file missing -- run the corresponding script first._\n")
        return
    for g in sorted(strat):
        s = strat[g]
        L.append(f"### {g} (n={s['n']})\n")
        L.append("| arm | resolved/total | rate | lift vs none |")
        L.append("|---|---|---|---|")
        for arm in arms:
            c = s["arms"][arm]
            lift = s["lift_vs_none"].get(arm)
            lstr = f"{lift*100:+.1f}pp" if lift is not None else "-"
            L.append(f"| {arm} | {c['resolved']}/{c['total']} | {c['rate']*100:.1f}% | {lstr} |")
        L.append("")


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

    # Stage A2: reviewer-mandated extended analyses (no LLM)
    ext = analyze_extended(run_root, data)
    md += "\n" + render_markdown_ext(ext, data["arms"])

    b_results = None
    if args.llm:
        b_results = run_stage_b(data, run_root)
        if b_results:
            md += "\n" + render_markdown_b(b_results)

    (run_root / "report.md").write_text(md)
    dump = {k: v for k, v in data.items() if not k.startswith("_")}
    dump["extended"] = ext
    (run_root / "report.json").write_text(json.dumps(dump, indent=1))
    print(f"[evaluate] wrote {run_root/'report.md'} and {run_root/'report.json'}",
          file=sys.stderr)
    print(f"[evaluate] arms={len(data['arms'])} iids={len(data['ordered_iids'])} "
          f"failures={len(data['failures'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
