from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class MemoryItem:
    item_id: str
    text: str
    source: str = ""
    score: float = 0.0
    created_at: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if value["metadata"] is None:
            value["metadata"] = {}
        return value


class MemoryAdapter(Protocol):
    name: str

    def retrieve(self, instance: dict[str, Any], query: str) -> list[MemoryItem]:
        ...

    def write(self, instance: dict[str, Any], record: dict[str, Any]) -> None:
        ...

