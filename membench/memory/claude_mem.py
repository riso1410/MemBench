from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from ..jsonl import read_jsonl
from ..schema import memory_corpus_path
from .base import MemoryItem
from .structured import _limit_tokens


class ClaudeMemAdapter:
    """claude-mem's retrieval pipeline: SQLite FTS5 over observation records, bm25-ranked.

    ponytail: replicates the plugin's storage+search layer (FTS5 MATCH, bm25 order)
    without its worker daemon or Claude-based extraction step — the corpus is already
    structured, so extraction is a no-op, same treatment as the structured adapter.
    """

    name = "claude_mem"

    def __init__(self, corpus_root: str | Path, top_k: int = 8, max_memory_tokens: int = 3000):
        self.corpus_root = Path(corpus_root)
        self.top_k = top_k
        self.max_memory_tokens = max_memory_tokens
        self.construction_time_sec: float = 0.0

    def _build_db(self, corpus_dir: Path) -> sqlite3.Connection:
        started = time.time()
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE VIRTUAL TABLE observations USING fts5(item_id, title, body, source, created_at)"
        )
        rows = []
        for name, title_key, body_key in (
            ("project_memory.jsonl", "key", "value"),
            ("events.jsonl", "title", "body"),
            ("documents.jsonl", "title", "text"),
        ):
            path = corpus_dir / name
            if not path.exists():
                continue
            for index, row in enumerate(read_jsonl(path)):
                rows.append(
                    (
                        str(row.get("id", f"{name}:{index}")),
                        str(row.get(title_key, "")),
                        str(row.get(body_key, row.get("content", ""))),
                        name.removesuffix(".jsonl"),
                        str(row.get("created_at", "")),
                    )
                )
        conn.executemany("INSERT INTO observations VALUES (?,?,?,?,?)", rows)
        self.construction_time_sec = round(time.time() - started, 3)
        return conn

    def retrieve(self, instance: dict[str, Any], query: str) -> list[MemoryItem]:
        corpus_dir = memory_corpus_path(instance, self.corpus_root)
        if not corpus_dir.exists():
            return []
        conn = self._build_db(corpus_dir)
        # FTS5 MATCH is syntax-sensitive; OR-join sanitized query terms like claude-mem does
        terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if len(t) > 2]
        if not terms:
            return []
        match = " OR ".join(dict.fromkeys(terms))
        cur = conn.execute(
            "SELECT item_id, title, body, source, created_at, bm25(observations) AS rank "
            "FROM observations WHERE observations MATCH ? ORDER BY rank LIMIT ?",
            (match, self.top_k),
        )
        items = [
            MemoryItem(
                item_id=item_id,
                text=f"{title}\n{body}".strip(),
                source=f"claude_mem:{source}",
                score=-rank,  # bm25 is lower-is-better
                created_at=created_at,
                metadata={"construction_time_sec": self.construction_time_sec},
            )
            for item_id, title, body, source, created_at, rank in cur.fetchall()
        ]
        conn.close()
        return _limit_tokens(items, self.max_memory_tokens)

    def write(self, instance: dict[str, Any], record: dict[str, Any]) -> None:
        return None
