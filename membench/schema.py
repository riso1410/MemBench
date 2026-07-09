from __future__ import annotations

from pathlib import Path
from typing import Any

from .jsonl import read_jsonl


REQUIRED_INSTANCE_FIELDS = {
    "instance_id",
    "repo",
    "base_commit",
    "issue",
    "oracle",
    "memory_corpus",
    "memory_need_labels",
    "budgets",
}


def load_instances(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def validate_instance(instance: dict[str, Any], index: int = 0) -> list[str]:
    prefix = f"instance[{index}]"
    errors: list[str] = []

    missing = sorted(REQUIRED_INSTANCE_FIELDS - set(instance))
    if missing:
        errors.append(f"{prefix}: missing fields: {', '.join(missing)}")

    if not isinstance(instance.get("instance_id"), str) or not instance.get("instance_id"):
        errors.append(f"{prefix}: instance_id must be a non-empty string")

    if not isinstance(instance.get("repo"), str) or "/" not in instance.get("repo", ""):
        errors.append(f"{prefix}: repo must look like owner/name")

    issue = instance.get("issue")
    if not isinstance(issue, dict):
        errors.append(f"{prefix}: issue must be an object")
    else:
        for field in ("title", "body", "created_at"):
            if not isinstance(issue.get(field), str):
                errors.append(f"{prefix}: issue.{field} must be a string")

    oracle = instance.get("oracle")
    if not isinstance(oracle, dict):
        errors.append(f"{prefix}: oracle must be an object")
    else:
        if "fail_to_pass" in oracle and not isinstance(oracle["fail_to_pass"], list):
            errors.append(f"{prefix}: oracle.fail_to_pass must be a list")
        if "pass_to_pass" in oracle and not isinstance(oracle["pass_to_pass"], list):
            errors.append(f"{prefix}: oracle.pass_to_pass must be a list")

    memory_corpus = instance.get("memory_corpus")
    if not isinstance(memory_corpus, dict):
        errors.append(f"{prefix}: memory_corpus must be an object")
    else:
        if not isinstance(memory_corpus.get("cutoff_time"), str):
            errors.append(f"{prefix}: memory_corpus.cutoff_time must be a string")
        allowed_sources = memory_corpus.get("allowed_sources", [])
        if not isinstance(allowed_sources, list):
            errors.append(f"{prefix}: memory_corpus.allowed_sources must be a list")

    labels = instance.get("memory_need_labels")
    if not isinstance(labels, dict):
        errors.append(f"{prefix}: memory_need_labels must be an object")
    else:
        if "requires_memory" in labels and not isinstance(labels["requires_memory"], bool):
            errors.append(f"{prefix}: memory_need_labels.requires_memory must be a boolean")
        if "memory_type" in labels and not isinstance(labels["memory_type"], list):
            errors.append(f"{prefix}: memory_need_labels.memory_type must be a list")

    budgets = instance.get("budgets")
    if not isinstance(budgets, dict):
        errors.append(f"{prefix}: budgets must be an object")
    else:
        for field in ("max_wall_time_sec", "max_input_tokens", "max_output_tokens", "max_cost_usd"):
            if field in budgets and not isinstance(budgets[field], int | float):
                errors.append(f"{prefix}: budgets.{field} must be numeric")

    return errors


def validate_instances(instances: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, instance in enumerate(instances):
        errors.extend(validate_instance(instance, index=index))
        instance_id = instance.get("instance_id")
        if isinstance(instance_id, str):
            if instance_id in seen_ids:
                errors.append(f"instance[{index}]: duplicate instance_id {instance_id}")
            seen_ids.add(instance_id)
    return errors


def memory_corpus_path(instance: dict[str, Any], default_root: str | Path) -> Path:
    memory_corpus = instance.get("memory_corpus", {})
    explicit_path = memory_corpus.get("path") if isinstance(memory_corpus, dict) else None
    if explicit_path:
        return Path(explicit_path)
    return Path(default_root) / str(instance["instance_id"])

