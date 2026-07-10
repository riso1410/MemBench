from __future__ import annotations

from pathlib import Path
from typing import Any

from .jsonl import read_jsonl, write_json
from .schema import load_instances, validate_instances


def evaluate_predictions(
    instances_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    instances = load_instances(instances_path)
    errors = validate_instances(instances)
    if errors:
        raise ValueError("invalid instances:\n" + "\n".join(errors))

    predictions = read_jsonl(predictions_path)
    instance_ids = {instance["instance_id"] for instance in instances}
    prediction_by_id = {prediction.get("instance_id"): prediction for prediction in predictions}

    missing = sorted(instance_ids - set(prediction_by_id))
    extra = sorted(set(prediction_by_id) - instance_ids)
    ok_predictions = [row for row in predictions if row.get("status") == "ok"]
    error_predictions = [row for row in predictions if row.get("status") == "error"]

    usage = _sum_usage(ok_predictions)
    estimated_cost_usd = round(
        sum(float(row.get("estimated_cost_usd", 0.0)) for row in ok_predictions),
        8,
    )
    scored = [row for row in ok_predictions if row.get("resolved_local_unverified") is not None]
    resolved_count = sum(1 for row in scored if row.get("resolved_local_unverified") is True)
    wall_times = [float(row.get("wall_time_sec", 0.0)) for row in ok_predictions]
    report: dict[str, Any] = {
        "instances": len(instances),
        "predictions": len(predictions),
        "ok": len(ok_predictions),
        "errors": len(error_predictions),
        "missing_predictions": missing,
        "extra_predictions": extra,
        "usage": usage,
        "estimated_cost_usd": estimated_cost_usd,
        "scored": len(scored),
        "resolved": resolved_count,
        "resolve_rate": round(resolved_count / len(scored), 4) if scored else None,
        "mean_wall_time_sec": round(sum(wall_times) / len(wall_times), 3) if wall_times else None,
        "evaluation_status": "executable" if scored else "no_workspace_instances",
    }
    if output_path:
        write_json(output_path, report)
    return report


def _sum_usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for row in rows:
        usage = row.get("usage", {})
        if not isinstance(usage, dict):
            continue
        prompt_tokens += int(usage.get("prompt_tokens", 0))
        completion_tokens += int(usage.get("completion_tokens", 0))
        total_tokens += int(usage.get("total_tokens", 0))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
