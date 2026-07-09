from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ..jsonl import read_jsonl
from ..schema import memory_corpus_path
from .base import MemoryItem

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
        work_dir = Path("runs/.graphify_stores") / corpus_dir.name
        graph_path = work_dir / "graphify-out" / "graph.json"
        if graph_path.exists():
            return graph_path
        work_dir.mkdir(parents=True, exist_ok=True)
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
            (work_dir / f"{name.removesuffix('.jsonl')}.md").write_text("\n".join(lines))
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
            raise RuntimeError(
                f"graphify extract failed ({result.returncode}): {result.stderr[-500:]}"
            )
        return graph_path

    def retrieve(self, instance: dict[str, Any], query: str) -> list[MemoryItem]:
        corpus_dir = memory_corpus_path(instance, self.corpus_root)
        if not corpus_dir.exists():
            return []
        graph_path = self._graph_for(corpus_dir)
        result = subprocess.run(
            [
                "graphify", "query", query,
                "--graph", str(graph_path),
                "--budget", str(self.max_memory_tokens),
            ],
            capture_output=True, text=True, timeout=120,
        )
        answer = result.stdout.strip()
        if result.returncode != 0 or not answer:
            return []
        return [
            MemoryItem(
                item_id=f"graphify:{corpus_dir.name}",
                text=answer,
                source="graphify",
                score=1.0,
                metadata={"construction_time_sec": self.construction_time_sec},
            )
        ]

    def write(self, instance: dict[str, Any], record: dict[str, Any]) -> None:
        return None
