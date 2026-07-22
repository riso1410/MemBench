from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .base import MemoryItem

# ponytail: sidecar built once by scripts/build_oracle_files.py; adapter ignores
# the corpus entirely and injects the gold-edited file list (perfect retrieval
# upper bound, E2). Not the diff -- localization, not the answer.
DEFAULT_ORACLE_FILES = "dataset/cross_session/oracle_files.json"


class OracleMemoryAdapter:
    name = "oracle"

    def __init__(self, corpus_root: str | Path = "", top_k: int = 8, max_memory_tokens: int = 3000):
        path = Path(os.environ.get("MEMBENCH_ORACLE_FILES", DEFAULT_ORACLE_FILES))
        self._files: dict[str, list[str]] = json.loads(path.read_text()) if path.exists() else {}

    def retrieve(self, instance: dict[str, Any], query: str) -> list[MemoryItem]:
        files = self._files.get(str(instance.get("instance_id", "")), [])
        if not files:
            return []
        text = (
            "From prior work on this repository, the fix for this issue belongs in "
            "these files:\n" + "\n".join(f"- {f}" for f in files)
        )
        return [MemoryItem(item_id="oracle_files", text=text, source="oracle", score=1.0)]

    def write(self, instance: dict[str, Any], record: dict[str, Any]) -> None:
        return None
