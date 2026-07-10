from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..jsonl import read_jsonl
from ..schema import memory_corpus_path
from .base import MemoryItem
from .structured import _limit_tokens

LLM_BASE_URL = "http://127.0.0.1:8000/v1"
EMBED_BASE_URL = "http://127.0.0.1:8001/v1"


class Mem0Adapter:
    """Mem0 OSS pinned to local models: qwen3-coder-30b extraction, qwen3-embedding vectors.

    Runs the real mem0 pipeline: add() performs LLM fact extraction, search() is
    vector retrieval from a local (serverless) Qdrant store.
    """

    name = "mem0"

    def __init__(self, corpus_root: str | Path, top_k: int = 8, max_memory_tokens: int = 3000):
        self.corpus_root = Path(corpus_root)
        self.top_k = top_k
        self.max_memory_tokens = max_memory_tokens
        self.construction_time_sec: float = 0.0
        self._memories: dict[str, Any] = {}

    def _memory_for(self, corpus_dir: Path) -> Any:
        key = str(corpus_dir)
        if key in self._memories:
            return self._memories[key]
        import os

        # telemetry opens a second local qdrant client on a global ~/.mem0 path,
        # which deadlocks the 2nd corpus in the same process
        os.environ.setdefault("MEM0_TELEMETRY", "false")
        from mem0 import Memory

        store_dir = Path("runs/.mem0_stores") / _store_key(corpus_dir)
        config = {
            "llm": {
                "provider": "vllm",
                "config": {
                    "model": "qwen3-coder-30b",
                    "vllm_base_url": LLM_BASE_URL,
                    "api_key": "dummy",
                    "temperature": 0.0,
                    "max_tokens": 2000,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "qwen3-embedding",
                    "openai_base_url": EMBED_BASE_URL,
                    "api_key": "dummy",
                    # no embedding_dims: mem0 would forward dimensions= and vLLM 400s on it
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(store_dir / "qdrant"),
                    "collection_name": "membench",
                    "embedding_model_dims": 1024,
                    "on_disk": True,
                },
            },
        }
        memory = Memory.from_config(config)
        started = time.time()
        if not (store_dir / ".ingested").exists():
            for text, created_at in _corpus_texts(corpus_dir):
                memory.add(
                    [{"role": "user", "content": text}],
                    user_id="project",
                    metadata={"created_at": created_at},
                )
            store_dir.mkdir(parents=True, exist_ok=True)
            (store_dir / ".ingested").touch()
        self.construction_time_sec = round(time.time() - started, 3)
        self._memories[key] = memory
        return memory

    def retrieve(self, instance: dict[str, Any], query: str) -> list[MemoryItem]:
        corpus_dir = memory_corpus_path(instance, self.corpus_root)
        if not corpus_dir.exists():
            return []
        memory = self._memory_for(corpus_dir)
        results = memory.search(query, filters={"user_id": "project"}, limit=self.top_k)
        rows = results.get("results", []) if isinstance(results, dict) else (results or [])
        rows = [{"memory": r} if isinstance(r, str) else r for r in rows]
        rows = rows[: self.top_k]  # mem0 can return more than limit; enforce top_k uniformly
        items = [
            MemoryItem(
                item_id=str(row.get("id", "")),
                text=str(row.get("memory", "")),
                source="mem0",
                score=float(row.get("score") or 0.0),
                created_at=str((row.get("metadata") or {}).get("created_at", "")),
                metadata={"construction_time_sec": self.construction_time_sec},
            )
            for row in rows
        ]
        return _limit_tokens(items, self.max_memory_tokens)

    def write(self, instance: dict[str, Any], record: dict[str, Any]) -> None:
        return None


def _store_key(corpus_dir: Path) -> str:
    # scope store dir by absolute path so reruns/arms/repos never share vectors/graphs
    import hashlib
    return f"{corpus_dir.name}__{hashlib.sha1(str(corpus_dir.resolve()).encode()).hexdigest()[:10]}"


def _corpus_texts(corpus_dir: Path) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for name, title_key, body_key in (
        ("project_memory.jsonl", "key", "value"),
        ("events.jsonl", "title", "body"),
        ("documents.jsonl", "title", "text"),
    ):
        path = corpus_dir / name
        if not path.exists():
            continue
        for row in read_jsonl(path):
            title = str(row.get(title_key, ""))
            body = str(row.get(body_key, row.get("content", "")))
            text = f"{title}\n{body}".strip()[:10000]  # qwen3-embedding ctx is 4096 tokens
            texts.append((text, str(row.get("created_at", ""))))
    return texts
