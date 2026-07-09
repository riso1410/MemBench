from __future__ import annotations

from .base import MemoryItem


class NoMemoryAdapter:
    name = "none"

    def retrieve(self, instance: dict, query: str) -> list[MemoryItem]:
        return []

    def write(self, instance: dict, record: dict) -> None:
        return None

