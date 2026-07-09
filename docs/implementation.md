# MemBench Implementation Notes

This repository now contains a minimal runnable scaffold for the benchmark proposed in `docs/agent_memory_survey_and_membench_design.md`.

The project is pinned to Python 3.13 through `.python-version`; run commands through `uv run`.

## Current Commands

Validate the example dataset:

```bash
uv run --python 3.13 python -m membench validate --instances dataset/examples/instances.jsonl
```

Run without an LLM:

```bash
uv run --python 3.13 python -m membench run \
  --config configs/example.toml \
  --instances dataset/examples/instances.jsonl \
  --output runs/demo/predictions.jsonl
```

Aggregate the dry-run output:

```bash
uv run --python 3.13 python -m membench eval \
  --instances dataset/examples/instances.jsonl \
  --predictions runs/demo/predictions.jsonl \
  --output runs/demo/report.json
```

Run smoke tests:

```bash
uv run --python 3.13 python -m unittest discover -s tests
```

## Model URL Setup

When a model endpoint is available, update `configs/example.toml`:

```toml
[model]
provider = "openai_compatible"
model = "your-model-name"
base_url = "http://localhost:8000/v1"
api_key_env = "OPENAI_API_KEY"
input_cost_per_million_tokens = 0.0
output_cost_per_million_tokens = 0.0
```

The client calls `POST /v1/chat/completions`. If the base URL already ends in `/v1` or `/v1/chat/completions`, the scaffold preserves that shape.

## Agent Backends

- `single_shot` (default): one prompt → unified diff, extracted and `git apply`-ed. Blind to repo contents; real repos need the agentic backend.
- `claude_code`: runs headless `claude -p --output-format json` inside the task workspace. Memory arms inject retrieved items into the prompt. Reports real cost, turns, and API duration.
- `claude_code` + qwen: `configs/claude_code_qwen.toml` sets `[agent.env]` (`ANTHROPIC_BASE_URL` → pectra vLLM tunnel, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192`) so the same Claude Code scaffold runs on qwen3-coder-30b. vLLM serves the Anthropic `/v1/messages` API natively but must be launched with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. See `docs/benchmark_env_setup.md`.

## Executable Scoring

Instances with a `workspace` field (template repo + pytest oracle) are scored for real: the runner copies the template to a temp dir, git-commits the base state, runs a fail-to-pass pre-check, lets the agent work, restores `protected_paths` (tests), then runs fail-to-pass + pass-to-pass and records `resolved`, per-test output, and the final `model_patch` diff. Run any arm with `--adapter none|raw_rag|structured`:

```bash
uv run --python 3.13 python -m membench run --config configs/claude_code.toml \
  --instances dataset/examples/instances.jsonl --output runs/cc_structured/predictions.jsonl \
  --adapter structured
```

## Implemented Memory Adapters

- `none`: disables memory.
- `raw_rag`: lexical retrieval over `documents.jsonl` and `events.jsonl`.
- `structured`: lexical retrieval over `project_memory.jsonl`.

The retrieval is intentionally simple. It gives us a reproducible baseline before adding vector search, rerankers, graph memory, or product integrations.

## Next Implementation Steps

1. Add a SWE-bench harness adapter that can apply model patches and run fail-to-pass/pass-to-pass tests.
2. Add dataset builders that mine GitHub issues/PRs into time-split memory corpora, including the carry-forward check (is task N's fix already in task N+1's base checkout?).
3. Add a `filesystem` memory adapter (plain files + grep-style search) — per the v2 design this is the strongest evidence-backed baseline, ahead of vector search.
4. Add a `reasoningbank`-style adapter: distilled strategy items from prior success and failure trajectories.
5. Add token/cost pricing config per model, plus memory-construction-time and steps-per-task metrics.
6. Add multi-seed / multi-ordering run support with paired per-instance output (needed for McNemar and paired bootstrap in `eval`).
7. Add memory-write evaluation after each task.
