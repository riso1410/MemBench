# Coding-Benchmark Environment Setup (SWE-bench-Live)

How the executable-evaluation environment is set up, following the official guides:

- SWE-bench harness reference: https://www.swebench.com/SWE-bench/reference/harness/
- SWE-bench-Live: https://github.com/microsoft/SWE-bench-Live (`evaluation/README.md`)

## Topology

| Piece | Where | Why |
| --- | --- | --- |
| Inference (qwen3-coder-30b, vLLM 0.24.0) | pectra (RTX 5090), tunnel `ssh -f -N -L 8000:127.0.0.1:8000 pectra` | Only box with a GPU. |
| Evaluation harness (Docker per-instance) | this Mac | pectra has no Docker (rootless podman only); Docker Desktop works here. Images are x86_64 → run under Rosetta emulation: correct but ~2-5x slower. |
| MemBench runner + agent (Claude Code) | this Mac | Talks to the tunnel. |

## One-time setup (already done, reproducible)

```bash
# 1. Clone with the RepoLaunch submodule (evaluation imports launch.core)
cd ~/Projects
git clone https://github.com/microsoft/SWE-bench-Live.git
cd SWE-bench-Live
git submodule update --init --depth 1

# 2. Venv + install (Python >= 3.10)
uv venv --python 3.12 .venv
uv pip install -e . --python .venv/bin/python
uv pip install -e ./launch --python .venv/bin/python   # gotcha: pip install -e . alone misses launch.core

# 3. Smoke-test with the gold patch on one lite instance (pulls the
#    per-instance image starryzhang/sweb.eval.x86_64.<instance> from DockerHub)
.venv/bin/python -m evaluation.evaluation \
  --dataset SWE-bench-Live/SWE-bench-Live --split lite \
  --instance_ids aws-cloudformation__cfn-lint-3798 \
  --platform linux --patch_dir gold \
  --output_dir logs/gold_smoke --workers 1 --overwrite 1
```

Datasets (HuggingFace): `SWE-bench-Live/SWE-bench-Live` splits `lite` (300), `verified` (500), `full` (1888); also `MultiLang` (743) and `Windows` (61).

## Running model predictions

Predictions go in a JSON file keyed by instance id:

```json
{"aws-cloudformation__cfn-lint-3798": {"model_patch": "<git diff>"}}
```

then `--patch_dir <that file>` instead of `gold`. MemBench's runner already emits
`model_patch` per instance in `predictions.jsonl`; the adapter that converts
JSONL → this JSON format is implementation step 1 in `implementation.md`.

## Resource notes (from the official docs)

- ~4 CPUs / 16 GB RAM per concurrent worker; use `--workers 4` max on this Mac.
- Images are pulled per instance (~1-3 GB each) — prune with `docker system prune` between splits.
- Docker "does not guarantee full isolation": rerun flaky verdicts (this is exactly the run-to-run variance MemBench's multi-seed design accounts for).

## Claude Code + qwen agent (the agent side of the benchmark)

`configs/claude_code_qwen.toml` routes Claude Code at the pectra vLLM server:
vLLM 0.24.0 natively serves the Anthropic Messages API at `/v1/messages`, so
`ANTHROPIC_BASE_URL=http://127.0.0.1:8000` + `--model qwen3-coder-30b` just works.
Two gotchas, both handled in the config:

- vLLM must be launched with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
  (done on pectra; without it Claude Code gets HTTP 400 on tool use).
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192` — Claude Code's default 32k output request
  overflows the 65536 model context.
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS=44000` — Claude Code assumes 200k context for
  unknown models and never auto-compacts; long agentic sessions then 400 at 65k.
  Declare well under the real window: a single large tool result (e.g. pytest
  output) can jump the transcript ~10k tokens past the compaction check.
- `total_cost_usd` reported by the claude CLI is fictitious for a local model
  (it prices by model name); treat cost as $0 / electricity, use token counts.
