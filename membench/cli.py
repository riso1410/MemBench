from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .evaluate import evaluate_predictions
from .runner import run_benchmark
from .schema import load_instances, validate_instances


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="membench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate an instances JSONL file.")
    validate_parser.add_argument("--instances", required=True)

    run_parser = subparsers.add_parser("run", help="Run the benchmark scaffold.")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--instances", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--adapter", default="", help="Override [memory].adapter from the config.")

    eval_parser = subparsers.add_parser("eval", help="Validate predictions and aggregate usage.")
    eval_parser.add_argument("--instances", required=True)
    eval_parser.add_argument("--predictions", required=True)
    eval_parser.add_argument("--output", default="")

    args = parser.parse_args(argv)

    if args.command == "validate":
        instances = load_instances(args.instances)
        errors = validate_instances(instances)
        if errors:
            for error in errors:
                print(error)
            return 1
        print(f"valid: {args.instances} ({len(instances)} instances)")
        return 0

    if args.command == "run":
        config = load_config(args.config)
        if args.adapter:
            config = replace(config, memory=replace(config.memory, adapter=args.adapter))
        summary = run_benchmark(
            config=config,
            instances_path=args.instances,
            output_path=args.output,
            limit=args.limit,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["errors"] == 0 else 1

    if args.command == "eval":
        output = Path(args.output) if args.output else None
        report = evaluate_predictions(
            instances_path=args.instances,
            predictions_path=args.predictions,
            output_path=output,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2

