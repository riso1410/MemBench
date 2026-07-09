from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .agent import CodingAgent, build_query
from .claude_code import run_claude_code
from .config import MemBenchConfig
from .jsonl import write_jsonl
from .memory import build_memory_adapter
from .models import build_model
from .schema import load_instances, validate_instances
from .workspace import (
    apply_patch,
    extract_diff,
    restore_protected_paths,
    run_tests,
    score_workspace,
    setup_workspace,
    workspace_diff,
)


def run_benchmark(
    config: MemBenchConfig,
    instances_path: str | Path,
    output_path: str | Path,
    limit: int = 0,
) -> dict[str, Any]:
    instances = load_instances(instances_path)
    errors = validate_instances(instances)
    if errors:
        raise ValueError("invalid instances:\n" + "\n".join(errors))
    instances_dir = Path(instances_path).resolve().parent

    max_instances = limit or config.run.max_instances
    if max_instances > 0:
        instances = instances[:max_instances]

    memory = build_memory_adapter(config.memory)
    model = build_model(config.model) if config.agent.backend != "claude_code" else None

    predictions: list[dict[str, Any]] = []
    started_at = time.time()
    for instance in instances:
        instance_started_at = time.time()
        try:
            predictions.append(
                _run_instance(
                    config, instance, instances_dir, memory, model, instance_started_at,
                    trajectory_dir=Path(output_path).resolve().parent / "trajectories",
                )
            )
        except Exception as exc:  # noqa: BLE001 - benchmark runs should record per-instance errors.
            predictions.append(
                {
                    "instance_id": instance.get("instance_id", ""),
                    "repo": instance.get("repo", ""),
                    "status": "error",
                    "error": str(exc),
                    "wall_time_sec": round(time.time() - instance_started_at, 3),
                }
            )

    write_jsonl(output_path, predictions)
    return {
        "instances": len(instances),
        "output_path": str(output_path),
        "wall_time_sec": round(time.time() - started_at, 3),
        "ok": sum(1 for row in predictions if row.get("status") == "ok"),
        "errors": sum(1 for row in predictions if row.get("status") == "error"),
        "resolved": sum(1 for row in predictions if row.get("resolved") is True),
    }


def _run_instance(
    config: MemBenchConfig,
    instance: dict[str, Any],
    instances_dir: Path,
    memory: Any,
    model: Any,
    instance_started_at: float,
    trajectory_dir: Path | None = None,
) -> dict[str, Any]:
    workspace = None
    oracle_precheck = None
    if isinstance(instance.get("workspace"), dict):
        workspace = setup_workspace(instance, instances_dir)
        pre = run_tests(workspace, list(instance.get("oracle", {}).get("fail_to_pass", [])))
        oracle_precheck = {"fail_to_pass_failed_before_fix": not pre["passed"]}

    query = build_query(instance)
    memory_items = memory.retrieve(instance, query)

    patch_applied = None
    if config.agent.backend == "claude_code":
        if workspace is None:
            raise ValueError("claude_code backend requires instance.workspace")
        trajectory_path = (
            trajectory_dir / f"{instance['instance_id']}.jsonl" if trajectory_dir else None
        )
        cc = run_claude_code(
            instance, memory_items, workspace, config.agent, trajectory_path=trajectory_path
        )
        content = cc["content"]
        usage = cc["usage"]
        cost_usd = cc["cost_usd"]
        agent_info: dict[str, Any] = {
            "backend": "claude_code",
            "model": config.agent.claude_model or "default",
            "num_turns": cc["num_turns"],
            "duration_ms": cc["duration_ms"],
            "api_duration_ms": cc["api_duration_ms"],
            "tool_calls": cc["tool_calls"],
            "thinking_blocks": cc["thinking_blocks"],
            "trajectory_path": cc["trajectory_path"],
            "subtype": cc.get("subtype", ""),
            "is_error": cc.get("is_error", False),
        }
    else:
        agent = CodingAgent(model)
        run = agent.run(instance, memory_items)
        content = run.result.content
        usage = run.result.usage
        cost_usd = _estimated_cost_usd(config.model, usage)
        agent_info = {"backend": "single_shot", "model": config.model.model}
        if workspace is not None:
            patch_applied = apply_patch(workspace, extract_diff(content))

    prediction: dict[str, Any] = {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "status": "ok",
        "agent": agent_info,
        "memory": {
            "adapter": config.memory.adapter,
            "top_k": config.memory.top_k,
            "max_memory_tokens": config.memory.max_memory_tokens,
        },
        "prediction": content,
        "usage": usage,
        "estimated_cost_usd": cost_usd,
        "resolved": None,
        "wall_time_sec": round(time.time() - instance_started_at, 3),
    }
    if patch_applied is not None:
        prediction["patch_applied"] = patch_applied

    if workspace is not None:
        restore_protected_paths(workspace, instance)
        scoring = score_workspace(workspace, instance)
        prediction["resolved"] = scoring["resolved"]
        prediction["tests"] = scoring
        prediction["oracle_precheck"] = oracle_precheck
        prediction["model_patch"] = workspace_diff(workspace)
        prediction["workspace"] = str(workspace)
        prediction["wall_time_sec"] = round(time.time() - instance_started_at, 3)

    if config.run.include_retrieved_memory_in_output:
        prediction["retrieved_memory"] = [item.to_dict() for item in memory_items]
    return prediction


def _estimated_cost_usd(model_config: Any, usage: dict[str, int]) -> float:
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    cost = (
        prompt_tokens * model_config.input_cost_per_million_tokens / 1_000_000
        + completion_tokens * model_config.output_cost_per_million_tokens / 1_000_000
    )
    return round(cost, 8)
