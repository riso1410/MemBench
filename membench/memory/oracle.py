from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .base import MemoryItem

# ponytail: sidecars built once by scripts/build_oracle_files.py; the adapter
# ignores the corpus entirely and injects an oracle hint (perfect-retrieval upper
# bounds). Two strengths:
#   mode="files" (E2, weak oracle):  the gold-edited file list -- localization,
#                                    not the answer.
#   mode="diff"  (E2', strong oracle): the gold unified diff itself -- the
#                                    near-answer upper bound.
DEFAULT_ORACLE_FILES = "dataset/cross_session/oracle_files.json"
DEFAULT_ORACLE_PATCHES = "dataset/cross_session/oracle_patches.json"


class OracleMemoryAdapter:
    name = "oracle"

    def __init__(self, corpus_root: str | Path = "", top_k: int = 8,
                 max_memory_tokens: int = 3000, mode: str = "files"):
        self.mode = mode
        if mode == "diff":
            path = Path(os.environ.get("MEMBENCH_ORACLE_PATCHES", DEFAULT_ORACLE_PATCHES))
            self._patches: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}
        else:
            path = Path(os.environ.get("MEMBENCH_ORACLE_FILES", DEFAULT_ORACLE_FILES))
            self._files: dict[str, list[str]] = json.loads(path.read_text()) if path.exists() else {}

    def retrieve(self, instance: dict[str, Any], query: str) -> list[MemoryItem]:
        iid = str(instance.get("instance_id", ""))
        if self.mode == "diff":
            diff = self._patches.get(iid, "")
            if not diff.strip():
                return []
            text = (
                "From prior work on this repository, the exact change that resolves "
                "this issue is:\n\n" + diff
            )
            return [MemoryItem(item_id="oracle_diff", text=text, source="oracle", score=1.0)]
        files = self._files.get(iid, [])
        if not files:
            return []
        text = (
            "From prior work on this repository, the fix for this issue belongs in "
            "these files:\n" + "\n".join(f"- {f}" for f in files)
        )
        return [MemoryItem(item_id="oracle_files", text=text, source="oracle", score=1.0)]

    def write(self, instance: dict[str, Any], record: dict[str, Any]) -> None:
        return None
