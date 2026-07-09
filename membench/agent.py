from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory.base import MemoryItem
from .models import ChatModel, ModelResult
from .token_count import estimate_message_tokens


SYSTEM_PROMPT = """You are a coding agent being evaluated in MemBench.
Return a unified diff patch that fixes the issue. Use retrieved memory only when it is relevant and not stale.
If memory influenced the patch, cite the memory item ids after the patch under a 'Memory used:' section."""


@dataclass(frozen=True)
class AgentRun:
    messages: list[dict[str, str]]
    result: ModelResult
    prompt_tokens_estimated: int


class CodingAgent:
    def __init__(self, model: ChatModel):
        self.model = model

    def run(self, instance: dict[str, Any], memory_items: list[MemoryItem]) -> AgentRun:
        messages = build_messages(instance, memory_items)
        result = self.model.complete(messages)
        return AgentRun(
            messages=messages,
            result=result,
            prompt_tokens_estimated=estimate_message_tokens(messages),
        )


def build_query(instance: dict[str, Any]) -> str:
    issue = instance.get("issue", {})
    labels = instance.get("memory_need_labels", {})
    parts = [
        str(instance.get("repo", "")),
        str(issue.get("title", "")),
        str(issue.get("body", "")),
        " ".join(str(value) for value in labels.get("memory_type", []))
        if isinstance(labels.get("memory_type", []), list)
        else "",
    ]
    return "\n".join(part for part in parts if part)


def build_messages(instance: dict[str, Any], memory_items: list[MemoryItem]) -> list[dict[str, str]]:
    issue = instance.get("issue", {})
    memory_block = format_memory(memory_items)
    user_prompt = f"""Repository: {instance.get("repo")}
Base commit: {instance.get("base_commit")}

Issue title:
{issue.get("title", "")}

Issue body:
{issue.get("body", "")}

Retrieved memory:
{memory_block}

Task:
Produce a patch as a unified diff. Do not include unrelated changes."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def format_memory(items: list[MemoryItem]) -> str:
    if not items:
        return "(none)"
    blocks: list[str] = []
    for item in items:
        created_at = f" created_at={item.created_at}" if item.created_at else ""
        source = f" source={item.source}" if item.source else ""
        blocks.append(
            f"[{item.item_id}] score={item.score:.2f}{source}{created_at}\n{item.text}"
        )
    return "\n\n".join(blocks)

