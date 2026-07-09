from __future__ import annotations

import math
from typing import Any


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(str(message.get("role", "")))
        total += estimate_tokens(str(message.get("content", "")))
    return total

