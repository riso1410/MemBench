from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ..jsonl import read_jsonl
from ..schema import memory_corpus_path
from .base import MemoryItem
from .mem0_adapter import _store_key
from .scoring import lexical_score
from .structured import _limit_tokens

LLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL = "qwen3-coder-30b"


class GraphifyAdapter:
    """safishamsi/graphify pinned to local qwen: corpus -> knowledge graph -> BFS query.

    Construction renders the memory corpus as markdown and runs
    `graphify extract --backend openai` (semantic extraction on the vLLM endpoint).
    Retrieval runs `graphify query` over the resulting graph.json.
    """

    name = "graphify"

    def __init__(self, corpus_root: str | Path, top_k: int = 8, max_memory_tokens: int = 3000):
        self.corpus_root = Path(corpus_root)
        self.top_k = top_k
        self.max_memory_tokens = max_memory_tokens
        self.construction_time_sec: float = 0.0

    def _graph_for(self, corpus_dir: Path) -> Path | None:
        # Store OUTSIDE the repo: graphify extract honours .gitignore, so a store
        # under runs/ (gitignored) has every doc skipped ("found 0 docs") -> empty
        # graph -> failure. ~/.membench is outside the repo and carries no
        # .gitignore, so the rendered corpus is actually ingested (and it is not
        # /tmp, so benchmark artifacts are not scattered there). Keyed by _store_key.
        work_dir = Path.home() / ".membench" / "graphify_stores" / _store_key(corpus_dir)
        graph_path = work_dir / "graphify-out" / "graph.json"
        if graph_path.exists():
            return graph_path
        work_dir.mkdir(parents=True, exist_ok=True)
        n_records = 0
        for name, title_key, body_key in (
            ("project_memory.jsonl", "key", "value"),
            ("events.jsonl", "title", "body"),
            ("documents.jsonl", "title", "text"),
        ):
            path = corpus_dir / name
            if not path.exists():
                continue
            lines = []
            for row in read_jsonl(path):
                title = str(row.get(title_key, ""))
                body = str(row.get(body_key, row.get("content", "")))
                created = str(row.get("created_at", ""))
                lines.append(f"## {title}\n({created})\n\n{body}\n")
                n_records += 1
            (work_dir / f"{name.removesuffix('.jsonl')}.md").write_text("\n".join(lines))
        if n_records == 0:
            # Empty store (e.g. first task in a sequence): no memory yet. Return
            # no graph so retrieval yields nothing; construction_time stays 0.
            return None
        env = {
            **os.environ,
            "OPENAI_BASE_URL": LLM_BASE_URL,
            "OPENAI_MODEL": LLM_MODEL,
            "OPENAI_API_KEY": "dummy",
        }
        started = time.time()
        result = subprocess.run(
            ["graphify", "extract", str(work_dir), "--backend", "openai"],
            capture_output=True, text=True, timeout=600, env=env,
        )
        self.construction_time_sec = round(time.time() - started, 3)
        if result.returncode != 0 or not graph_path.exists():
            # Nothing extractable (e.g. terse records produced an empty graph).
            # Degrade to "no memory" instead of failing the whole task, so one
            # thin snapshot cannot cascade-fail the sequence.
            return None
        return graph_path

    def retrieve(self, instance: dict[str, Any], query: str) -> list[MemoryItem]:
        corpus_dir = memory_corpus_path(instance, self.corpus_root)
        if not corpus_dir.exists():
            return []
        graph_path = self._graph_for(corpus_dir)
        if graph_path is None:
            return []
        result = subprocess.run(
            [
                "graphify", "query", query,
                "--graph", str(graph_path),
                "--budget", str(self.max_memory_tokens),
            ],
            capture_output=True, text=True, timeout=120,
        )
        answer = result.stdout.strip()
        items: list[MemoryItem] = []
        if result.returncode == 0 and answer:
            items.append(
                MemoryItem(
                    item_id=f"graphify:{corpus_dir.name}",
                    text=answer,
                    source="graphify",
                    score=1.0,
                    metadata={"construction_time_sec": self.construction_time_sec},
                )
            )
        # The single BFS answer is tiny (~150 tokens) versus the text arms that
        # saturate the 2000-token budget. Append relevant node/edge summaries from
        # the graph and let the shared token limiter bind the budget, so graphify's
        # injection is comparable in volume to the other arms.
        items.extend(self._graph_summaries(graph_path, query, corpus_dir.name))
        return _limit_tokens(items, self.max_memory_tokens)

    def _graph_summaries(self, graph_path: Path, query: str, corpus_name: str) -> list[MemoryItem]:
        try:
            graph = json.loads(graph_path.read_text())
        except (OSError, ValueError):
            return []
        id_to_label = {
            str(n.get("id")): str(n.get("label") or n.get("name") or n.get("id"))
            for n in graph.get("nodes", []) if isinstance(n, dict) and "id" in n
        }
        summaries: list[str] = []
        for n in graph.get("nodes", []):
            if isinstance(n, dict) and "id" in n:
                label = id_to_label.get(str(n["id"]), "")
                src_file = str(n.get("source_file") or "")
                if label:
                    summaries.append(f"{label} — {src_file}".strip(" —") if src_file else label)
        # NetworkX <=3.1 serialises edges under "links"; graphify writes either.
        for e in graph.get("edges", []) or graph.get("links", []):
            if not isinstance(e, dict):
                continue
            src = id_to_label.get(str(e.get("source")), str(e.get("source", "")))
            tgt = id_to_label.get(str(e.get("target")), str(e.get("target", "")))
            rel = str(e.get("relation") or e.get("type") or "related to")
            if src and tgt:
                summaries.append(f"{src} --{rel}--> {tgt}")
        # The whole graph is already scoped to this repo's prior tasks, so rank
        # summaries by query relevance (best first) but keep them all — the shared
        # token limiter binds the budget, so a small graph injects fully and a large
        # one fills up to the same budget the text arms use. No score>0 gate: graph
        # labels are terse (CamelCase), so lexical overlap is often 0 yet relevant.
        scored = sorted(
            ((lexical_score(query, s), s) for s in dict.fromkeys(summaries)),
            key=lambda p: p[0], reverse=True,
        )
        return [
            MemoryItem(
                item_id=f"graphify:{corpus_name}:g{i}",
                text=text,
                source="graphify_graph",
                score=float(score),
                metadata={"construction_time_sec": self.construction_time_sec},
            )
            for i, (score, text) in enumerate(scored)
        ]

    def write(self, instance: dict[str, Any], record: dict[str, Any]) -> None:
        return None
