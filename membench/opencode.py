from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .agent import format_memory
from .claude_code import PROMPT_TEMPLATE
from .config import AgentConfig
from .memory.base import MemoryItem


def run_opencode(
    instance: dict[str, Any],
    memory_items: list[MemoryItem],
    workspace: Path,
    agent_config: AgentConfig,
    trajectory_path: Path | None = None,
) -> dict[str, Any]:
    """Run the OpenCode CLI as a coding-agent backend.

    Mirrors run_claude_code's interface: edits source files in place inside
    `workspace`, and the runner captures the resulting diff via workspace_diff.
    Returns the same shape of prediction dict (content / usage / cost_usd /
    num_turns / duration_ms / tool_calls / trajectory_path / is_error).

    OpenCode has no turn cap flag (unlike Claude Code's --max-turns), so the
    only bound is the wall-time timeout. `--auto` auto-approves permissions so
    the run is non-interactive (Claude Code's bypassPermissions equivalent).
    `--format json` streams raw JSON events which we persist as the trajectory
    and parse best-effort for token usage / cost / tool calls.
    """
    issue = instance.get("issue", {})
    prompt = PROMPT_TEMPLATE.format(
        title=issue.get("title", ""),
        body=issue.get("body", ""),
        memory=format_memory(memory_items),
    )
    model = agent_config.opencode_model or agent_config.claude_model
    cmd = [
        agent_config.opencode_cmd,
        "run",
        prompt,
        "--format", "json",
        "--auto",  # auto-approve permissions; no interactive gate (headless)
    ]
    if model:
        cmd += ["--model", model]
    timeout = int(instance.get("budgets", {}).get("max_wall_time_sec", 1800))
    env = {**os.environ, **agent_config.env} if agent_config.env else None
    result = subprocess.run(
        cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout, env=env
    )
    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(result.stdout)

    events: list[Any] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # Fallback: some builds emit one JSON document rather than JSONL.
    if not events and result.stdout.strip():
        try:
            events.append(json.loads(result.stdout))
        except json.JSONDecodeError:
            pass

    if not events and result.returncode != 0:
        raise RuntimeError(
            f"opencode exited {result.returncode}: {result.stderr[-1000:]} "
            f"stdout: {result.stdout[-2000:]}"
        )

    texts, tool_calls, prompt_tokens, completion_tokens, cost = _parse_events(events)
    content = texts[-1] if texts else ""
    return {
        "content": content,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cost_usd": cost,
        "duration_ms": 0,
        "num_turns": len(texts),
        "tool_calls": tool_calls,
        "is_error": result.returncode != 0,
        "trajectory_path": str(trajectory_path) if trajectory_path else None,
    }


def _walk(obj: Any):
    """Yield every dict nested anywhere inside a JSON structure."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk(value)


def _parse_events(events: list[Any]) -> tuple[list[str], dict[str, int], int, int, float]:
    """Best-effort extraction from OpenCode's raw JSON event stream.

    OpenCode's event schema is not a stable public contract, so we search
    defensively: assistant text parts ({"type":"text","text":...}), tool parts
    ({"type":"tool","tool":name}), and per-message usage ({"tokens":{input,
    output}, "cost":..}). Missing fields collapse to zeros/empties rather than
    failing the run — token usage is informational; the scoreable artifact is
    the workspace diff captured by the runner.
    """
    texts: list[str] = []
    tool_calls: dict[str, int] = {}
    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0
    for node in _walk(events):
        ntype = node.get("type")
        if ntype == "text" and isinstance(node.get("text"), str) and node["text"].strip():
            texts.append(node["text"])
        elif ntype == "tool" and node.get("tool"):
            name = str(node.get("tool"))
            tool_calls[name] = tool_calls.get(name, 0) + 1
        tokens = node.get("tokens")
        if isinstance(tokens, dict):
            # OpenCode reports cumulative context as input; take the last seen.
            if tokens.get("input") is not None:
                prompt_tokens = int(tokens.get("input") or 0)
            completion_tokens += int(tokens.get("output") or 0)
        usage = node.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = int(
                usage.get("input_tokens", usage.get("prompt_tokens", prompt_tokens)) or prompt_tokens
            )
            completion_tokens = int(
                usage.get("output_tokens", usage.get("completion_tokens", completion_tokens))
                or completion_tokens
            )
        if isinstance(node.get("cost"), (int, float)):
            cost += float(node["cost"])
    return texts, tool_calls, prompt_tokens, completion_tokens, round(cost, 8)
