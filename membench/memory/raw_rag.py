from __future__ import annotations

from pathlib import Path
from typing import Any

from ..jsonl import read_jsonl
from ..schema import memory_corpus_path
from ..token_count import estimate_tokens
from .base import MemoryItem
from .scoring import lexical_score


class RawRagMemoryAdapter:
    name = "raw_rag"

    def __init__(self, corpus_root: str | Path, top_k: int = 8, max_memory_tokens: int = 3000):
        self.corpus_root = Path(corpus_root)
        self.top_k = top_k
        self.max_memory_tokens = max_memory_tokens

    def retrieve(self, instance: dict[str, Any], query: str) -> list[MemoryItem]:
        corpus_dir = memory_corpus_path(instance, self.corpus_root)
        rows: list[dict[str, Any]] = []
        for name in ("documents.jsonl", "events.jsonl"):
            path = corpus_dir / name
            if path.exists():
                rows.extend(read_jsonl(path))

        scored: list[MemoryItem] = []
        for index, row in enumerate(rows):
            text = _row_text(row)
            score = lexical_score(query, text)
            if score <= 0:
                continue
            scored.append(
                MemoryItem(
                    item_id=str(row.get("id", f"{corpus_dir.name}:{index}")),
                    text=text,
                    source=str(row.get("source", "")),
                    score=score,
                    created_at=str(row.get("created_at", "")),
                    metadata=row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {},
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return _limit_tokens(scored[: self.top_k], self.max_memory_tokens)

    def write(self, instance: dict[str, Any], record: dict[str, Any]) -> None:
        return None


def _row_text(row: dict[str, Any]) -> str:
    parts = []
    for key in ("title", "summary", "text", "body", "value"):
        value = row.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


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

