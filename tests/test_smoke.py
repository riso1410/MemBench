from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from membench.agent import build_query
from membench.config import load_config
from membench.memory import build_memory_adapter
from membench.runner import run_benchmark
from membench.schema import load_instances, validate_instances
from membench.workspace import score_workspace, setup_workspace


ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def test_example_dataset_validates(self) -> None:
        instances = load_instances(ROOT / "dataset/examples/instances.jsonl")
        self.assertEqual(validate_instances(instances), [])

    def test_example_config_loads(self) -> None:
        config = load_config(ROOT / "configs/example.toml")
        self.assertEqual(config.model.provider, "dry_run")
        self.assertEqual(config.memory.adapter, "structured")

    def test_structured_memory_retrieves_expected_item(self) -> None:
        config = load_config(ROOT / "configs/example.toml")
        instance = load_instances(ROOT / "dataset/examples/instances.jsonl")[0]
        memory = build_memory_adapter(config.memory)
        items = memory.retrieve(instance, build_query(instance))
        item_ids = {item.item_id for item in items}
        self.assertIn("pm_001", item_ids)

    def test_dry_run_writes_prediction(self) -> None:
        config = load_config(ROOT / "configs/example.toml")
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "predictions.jsonl"
            summary = run_benchmark(
                config=config,
                instances_path=ROOT / "dataset/examples/instances.jsonl",
                output_path=output,
            )
            self.assertEqual(summary["errors"], 0)
            self.assertTrue(output.exists())


class WorkspaceScoringTests(unittest.TestCase):
    def test_unfixed_workspace_is_unresolved_and_fix_resolves(self) -> None:
        instance = load_instances(ROOT / "dataset/examples/instances.jsonl")[0]
        workspace = setup_workspace(instance, ROOT / "dataset/examples")

        before = score_workspace(workspace, instance)
        self.assertFalse(before["resolved_local_unverified"])
        self.assertTrue(before["pass_to_pass"]["passed"])

        ops = workspace / "calculator" / "decimal_ops.py"
        source = ops.read_text()
        fixed = source.replace(
            "def divide(a: str, b: str) -> Decimal:\n    return Decimal(a) / Decimal(b)",
            "def divide(a: str, b: str) -> Decimal:\n"
            "    try:\n"
            "        return Decimal(a) / Decimal(b)\n"
            "    except (DivisionByZero, InvalidOperation, ZeroDivisionError) as exc:\n"
            "        raise CalculatorError(f\"divide failed: {exc}\") from exc",
        )
        self.assertNotEqual(source, fixed)
        ops.write_text(fixed)

        after = score_workspace(workspace, instance)
        self.assertTrue(after["resolved_local_unverified"])


if __name__ == "__main__":
    unittest.main()

