from __future__ import annotations

from pathlib import Path
from typing import Any

from ..jsonl import read_jsonl
from ..schema import memory_corpus_path
from ..token_count import estimate_tokens
from .base import MemoryItem
from .scoring import lexical_score


class StructuredMemoryAdapter:
    name = "structured"

    def __init__(self, corpus_root: str | Path, top_k: int = 8, max_memory_tokens: int = 3000):
        self.corpus_root = Path(corpus_root)
        self.top_k = top_k
        self.max_memory_tokens = max_memory_tokens

    def retrieve(self, instance: dict[str, Any], query: str) -> list[MemoryItem]:
        corpus_dir = memory_corpus_path(instance, self.corpus_root)
        path = corpus_dir / "project_memory.jsonl"
        if not path.exists():
            return []

        scored: list[MemoryItem] = []
        for index, row in enumerate(read_jsonl(path)):
            text = _memory_text(row)
            score = lexical_score(query, text)
            if score <= 0:
                continue
            scored.append(
                MemoryItem(
                    item_id=str(row.get("id", f"{corpus_dir.name}:memory:{index}")),
                    text=text,
                    source=str(row.get("source", "project_memory")),
                    score=score,
                    created_at=str(row.get("created_at", "")),
                    metadata={
                        "kind": row.get("kind", ""),
                        "confidence": row.get("confidence", ""),
                        "evidence": row.get("evidence", []),
                    },
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return _limit_tokens(scored[: self.top_k], self.max_memory_tokens)

    def write(self, instance: dict[str, Any], record: dict[str, Any]) -> None:
        return None


def _memory_text(row: dict[str, Any]) -> str:
    kind = row.get("kind", "")
    key = row.get("key", "")
    value = row.get("value", "")
    return f"{kind}: {key}\n{value}".strip()


def _limit_tokens(items: list[MemoryItem], max_tokens: int) -> list[MemoryItem]:
    selected: list[MemoryItem] = []
    used = 0
    for item in items:
        cost = estimate_tokens(item.text)
        if selected and used + cost > max_tokens:
            break
        selected.append(item)
        used += cost
    return selected

