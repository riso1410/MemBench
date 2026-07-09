# MemBench

MemBench is a proposed benchmark for testing whether long-term memory systems make coding agents measurably better after accounting for latency, token use, and implementation cost.

The initial research/design document is:

- [docs/agent_memory_survey_and_membench_design.md](docs/agent_memory_survey_and_membench_design.md)

Implementation notes:

- [docs/implementation.md](docs/implementation.md)

Quick start:

```bash
uv run --python 3.13 python -m membench validate --instances dataset/examples/instances.jsonl
uv run --python 3.13 python -m membench run --config configs/example.toml --instances dataset/examples/instances.jsonl --output runs/demo/predictions.jsonl
uv run --python 3.13 python -m membench eval --instances dataset/examples/instances.jsonl --predictions runs/demo/predictions.jsonl --output runs/demo/report.json
uv run --python 3.13 python -m unittest discover -s tests
```
