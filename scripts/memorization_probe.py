#!/usr/bin/env python3
"""Contamination / memorization probe.

For each instance, ask the local qwen model to produce a fix patch given ONLY
the issue title+body and repo name -- no repo access, no memory. If the model
can reproduce the fix from parametric knowledge alone, the instance is likely
contaminated (seen in pretraining). We score the answer's similarity to the
gold patch (difflib ratio + token overlap) and flag verbatim file/line hits.

Gold patches come from the SWE-bench-Live HF dataset; fall back to the none-arm
successful model_patch when gold is unavailable.

Run under the SWE-bench-Live venv (needs pyarrow + requests); vllm must be up:
    ~/SWE-bench-Live/.venv/bin/python scripts/memorization_probe.py

Output: runs/memorization_probe.jsonl (resumable) + stdout distribution.
"""
import argparse
import difflib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gold  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3-coder-30b"


def load_instances(path: Path):
    return {json.loads(l)["instance_id"]: json.loads(l)
            for l in path.read_text().splitlines() if l.strip()}


def none_success_patches(run_root: Path):
    """instance_id -> model_patch for none-arm resolved tasks (gold fallback)."""
    out = {}
    p = run_root / "none" / "predictions.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("resolved") is True and (r.get("model_patch") or "").strip():
            out[r["instance_id"]] = r["model_patch"]
    return out


def ask_llm(title, body, repo, timeout=180):
    import requests
    prompt = (
        f"Repository: {repo}\n"
        f"Issue title: {title}\n\n"
        f"Issue body:\n{body}\n\n"
        "You do NOT have access to the repository. Based only on your own prior "
        "knowledge of this project, write the exact code patch that fixes this "
        "issue as a unified diff (```diff ... ```). Include real file paths and "
        "the exact changed lines you believe are correct."
    )
    body_req = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert contributor to open-source Python projects."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
    }
    r = requests.post(ENDPOINT, json=body_req, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def token_overlap(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def gold_added_lines(patch):
    return [l[1:].strip() for l in (patch or "").splitlines()
            if l.startswith("+") and not l.startswith("+++") and len(l[1:].strip()) >= 12]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", default=str(REPO_ROOT / "dataset/cross_session/instances.jsonl"))
    ap.add_argument("--run-root", default=str(REPO_ROOT / "runs/cross_session"))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs/memorization_probe.jsonl"))
    ap.add_argument("--only-run-instances", action="store_true", default=True,
                    help="probe only instances that appear in run-root/verdicts.jsonl (default)")
    ap.add_argument("--all", dest="only_run_instances", action="store_false",
                    help="probe every instance in --instances")
    args = ap.parse_args()

    inst = load_instances(Path(args.instances))
    gold = _gold.load_gold()
    fallback = none_success_patches(Path(args.run_root))

    ids = list(inst)
    if args.only_run_instances:
        vp = Path(args.run_root) / "verdicts.jsonl"
        run_ids = {json.loads(l)["instance_id"] for l in vp.read_text().splitlines() if l.strip()}
        ids = [i for i in ids if i in run_ids]

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["instance_id"])

    with out_path.open("a") as fh:
        for i, iid in enumerate(ids, 1):
            if iid in done:
                print(f"[probe] skip {iid} (already done)", file=sys.stderr)
                continue
            row = inst[iid]
            iss = row.get("issue") or {}
            ref = (gold.get(iid) or {}).get("patch", "") or fallback.get(iid, "")
            ref_src = "gold" if (gold.get(iid) or {}).get("patch") else (
                "none_arm" if iid in fallback else "none")
            try:
                ans = ask_llm(iss.get("title", ""), (iss.get("body", "") or "")[:6000], row.get("repo", ""))
            except Exception as e:
                print(f"[probe] {iid} LLM error: {e}", file=sys.stderr)
                continue
            sim = difflib.SequenceMatcher(None, ans, ref).ratio() if ref else None
            tok = token_overlap(ans, ref) if ref else None
            gfiles = _gold.gold_files(ref)
            file_hit = any(f in ans for f in gfiles) if gfiles else False
            line_hits = sum(1 for gl in gold_added_lines(ref) if gl in ans)
            rec = {
                "instance_id": iid, "repo": row.get("repo", ""),
                "ref_source": ref_src,
                "similarity": round(sim, 4) if sim is not None else None,
                "token_overlap": round(tok, 4) if tok is not None else None,
                "verbatim_file_hit": bool(file_hit),
                "gold_file_hits": sorted(f for f in gfiles if f in ans),
                "verbatim_line_hits": line_hits,
                "answer_chars": len(ans),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[probe] {i}/{len(ids)} {iid} sim={rec['similarity']} "
                  f"tok={rec['token_overlap']} file_hit={rec['verbatim_file_hit']} "
                  f"line_hits={line_hits}", file=sys.stderr)

    # summary distribution
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["instance_id"] in set(ids)]
    sims = sorted(r["similarity"] for r in rows if r["similarity"] is not None)
    print(f"\n[probe] wrote {out_path}  ({len(rows)} instances)")
    if sims:
        import statistics
        print(f"[probe] similarity: min={sims[0]:.3f} median={statistics.median(sims):.3f} "
              f"max={sims[-1]:.3f} mean={statistics.mean(sims):.3f}")
        fh = sum(1 for r in rows if r["verbatim_file_hit"])
        lh = sum(1 for r in rows if r["verbatim_line_hits"] > 0)
        print(f"[probe] verbatim_file_hit={fh}/{len(rows)}  any_verbatim_line={lh}/{len(rows)}")
        buckets = {"<0.1": 0, "0.1-0.2": 0, "0.2-0.3": 0, "0.3-0.5": 0, ">=0.5": 0}
        for s in sims:
            k = ("<0.1" if s < 0.1 else "0.1-0.2" if s < 0.2 else "0.2-0.3" if s < 0.3
                 else "0.3-0.5" if s < 0.5 else ">=0.5")
            buckets[k] += 1
        print(f"[probe] similarity buckets: {buckets}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
