from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .agent import format_memory
from .config import AgentConfig
from .memory.base import MemoryItem


PROMPT_TEMPLATE = """Fix the following issue in this repository.

Issue title:
{title}

Issue body:
{body}

Retrieved memory from previous work on this project:
{memory}

Rules:
- Edit source files in place. Do not modify existing tests or add files under the repository's test directories.
- Do not commit. Leave your changes in the working tree.
- Use retrieved memory only when relevant and not stale. If memory influenced the fix, state which memory item ids you used."""


# Per-arm system-prompt appendix (opt-in via MEMBENCH_ARM_SYSTEM_PROMPTS=1).
# Cross-session-memory ablation: memory is otherwise only *passively* injected in
# the user prompt and the agent tends to ignore it (~98% of tokens are it reading
# source files). These appendices explicitly direct the agent to LEAN ON the
# retrieved memory. The `none` baseline gets no appendix (it has no memory).
_MEM_CORE = (
    "This task is part of a CROSS-SESSION MEMORY experiment. Your prompt contains "
    "RETRIEVED MEMORY accumulated from your own prior work on THIS repository. Before "
    "exploring the codebase from scratch, FIRST read that memory and let it steer you: "
    "which files to open, which fix pattern worked before, and which pitfalls (e.g. "
    "tests that previously broke) to avoid. Recall and reuse prior solutions rather "
    "than rediscovering them; cite the memory item ids you relied on. Some records "
    "may be labeled 'Outcome: FAILED' -- those are prior attempts that did NOT work "
    "(they broke tests or missed the fix); recall them as negative examples to avoid "
    "repeating that approach, not to copy."
)
ARM_SYSTEM_PROMPTS = {
    "none": "",
    "raw_rag": _MEM_CORE + " The memory is retrieved text snippets from past sessions.",
    "structured": _MEM_CORE + " The memory is a structured record of prior task outcomes (issue -> fix -> affected files).",
    "claude_mem": _MEM_CORE + " The memory is notes distilled from your prior Claude sessions on this repo.",
    "mem0": _MEM_CORE + " The memory comes from a mem0 long-term store; treat it as durable memory of how similar issues were resolved.",
    "graphiti": _MEM_CORE + " The memory comes from a temporal knowledge graph; use entity/relationship links to locate the right code and prior changes.",
    "graphify": _MEM_CORE + " The memory comes from a knowledge-graph summary; use it to orient on structure and prior fixes quickly.",
}


def run_claude_code(
    instance: dict[str, Any],
    memory_items: list[MemoryItem],
    workspace: Path,
    agent_config: AgentConfig,
    trajectory_path: Path | None = None,
    adapter: str = "none",
) -> dict[str, Any]:
    issue = instance.get("issue", {})
    prompt = PROMPT_TEMPLATE.format(
        title=issue.get("title", ""),
        body=issue.get("body", ""),
        memory=format_memory(memory_items),
    )
    cmd = [
        agent_config.claude_cmd,
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--permission-mode", agent_config.permission_mode,
        # headless mode never auto-compacts; unbounded runs 400 at the 65k model
        # context after ~70 turns. Successful demo runs take 8-16 turns.
        "--max-turns", "60",
        # ponytail: strip unneeded tool schemas; 23-tool baseline was ~19k tokens on a 64k model
        "--disallowedTools", "Task,CronCreate,CronDelete,CronList,EnterWorktree,ExitWorktree,NotebookEdit,ScheduleWakeup,SendMessage,Skill,TaskCreate,TaskGet,TaskList,TaskOutput,TaskStop,TaskUpdate,WebFetch,WebSearch,Workflow,DesignSync",
    ]
    if agent_config.claude_model:
        cmd += ["--model", agent_config.claude_model]
    # opt-in per-arm system-prompt appendix (cross-session-memory usage ablation)
    if os.environ.get("MEMBENCH_ARM_SYSTEM_PROMPTS") == "1":
        _extra = ARM_SYSTEM_PROMPTS.get(adapter, "")
        if _extra:
            cmd += ["--append-system-prompt", _extra]
    timeout = int(instance.get("budgets", {}).get("max_wall_time_sec", 1800))
    env = {**os.environ, **agent_config.env} if agent_config.env else None
    result = subprocess.run(
        cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout, env=env
    )
    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(result.stdout)
    events = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    payload = next(
        (e for e in reversed(events) if e.get("type") == "result"), None
    )
    # exit code 1 with a result event = bounded failure (max turns / model error),
    # still a scoreable attempt; only a missing result event is a harness crash.
    if payload is None:
        raise RuntimeError(
            f"claude code exited {result.returncode}: {result.stderr[-1000:]} "
            f"stdout: {result.stdout[-2000:]}"
        )
    tool_calls: dict[str, int] = {}
    thinking_blocks = 0
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "tool_use":
                name = block.get("name", "?")
                tool_calls[name] = tool_calls.get(name, 0) + 1
            elif block.get("type") == "thinking":
                thinking_blocks += 1
    usage = payload.get("usage") or {}
    prompt_tokens = (
        int(usage.get("input_tokens", 0))
        + int(usage.get("cache_read_input_tokens", 0))
        + int(usage.get("cache_creation_input_tokens", 0))
    )
    completion_tokens = int(usage.get("output_tokens", 0))
    return {
        "content": str(payload.get("result", "")),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cost_usd": float(payload.get("total_cost_usd", 0.0)),
        "duration_ms": int(payload.get("duration_ms", 0)),
        "api_duration_ms": int(payload.get("duration_api_ms", 0)),
        "num_turns": int(payload.get("num_turns", 0)),
        "subtype": str(payload.get("subtype", "")),
        "is_error": bool(payload.get("is_error", False)),
        "tool_calls": tool_calls,
        "thinking_blocks": thinking_blocks,
        "trajectory_path": str(trajectory_path) if trajectory_path else None,
    }
