# Memory Plugins for Claude Code — Named Comparison (2026-07-03)

Facts checked against primary sources on 2026-07-03 (GitHub API, official docs, local plugin install). Every published benchmark number in this space is vendor-produced; the two head-to-head disputes on record swing the same system by 20+ points depending on configuration. That gap is what MemBench fills.

## The contenders

| System | Integration with Claude Code | Storage / retrieval | Extraction LLM | Stars / release | License |
| --- | --- | --- | --- | --- | --- |
| **claude-mem** (thedotmack) | Native plugin: lifecycle hooks (SessionStart/PostToolUse/Stop) + MCP search tools + background worker | SQLite + FTS5 lexical, plus Chroma vectors (ONNX MiniLM) — vector sync broken on 13.9.x (#3107) | Pluggable: Claude (default haiku), Gemini, or any OpenAI-compatible endpoint via `CLAUDE_MEM_PROVIDER=openrouter` + custom base URL (≥13.9.x) | 85.6k ★, v13.9.3 (2026-07-03) | Apache-2.0 |
| **Graphiti** (getzep) — "graphify" | MCP server (streamable HTTP), Docker Compose + FalkorDB; no hooks — model must be told to call tools | Bi-temporal knowledge graph (valid_at/invalid_at), hybrid semantic+BM25+graph, no LLM at query time | Required for ingestion; OpenAI-compatible incl. vLLM via `api_base`. Docs warn small models break structured-output ingestion | 28.3k ★, v0.29.2 (2026-06-08) | Apache-2.0 |
| **Mem0** | Official cloud plugin (`mcp.mem0.ai`, needs MEM0_API_KEY) + hooks; local OpenMemory MCP being sunset | Vector store (Qdrant default) + built-in entity linking; **2026 algorithm is single-pass ADD-only — not the published paper's pipeline** | Dedicated `vllm` provider — cleanest local pinning | 60.0k ★, PyPI 2.0.11 (2026-07-01) | Apache-2.0 |
| **Letta** (MemGPT lineage) | `claude-subconscious` plugin (first-party) — but requires Letta Cloud and is a labeled non-production demo; no first-party MCP memory server | Self-edited memory blocks + archival vector memory + sleep-time agents | Model-agnostic, dedicated vLLM docs; memory quality degrades with small models per their own leaderboard | 23.6k ★, v0.16.8 (2026-05-14) | Apache-2.0 |
| **Cognee** | First-party MCP guide (`claude mcp add --transport http cognee ...`); no hooks | ECL pipeline → knowledge graph (Kuzu default) + vectors (LanceDB default) | LiteLLM under the hood; docs recommend ≥32B local models | 26.7k ★, v1.2.2 (2026-06-26) | Apache-2.0 |
| **Anthropic native** (memory tool, MEMORY.md, CLAUDE.md) | Built-in: API memory tool (GA), auto-memory MEMORY.md (first 200 lines/25KB loaded per session), CLAUDE.md convention | Plain files, agent-maintained | None — the main agent model does the work | n/a | n/a |
| **basic-memory** | Pure MCP server | Local markdown + wikilinks + semantic search | None | 3.4k ★ | AGPL-3.0 |
| **mcp-memory-service** | MCP + Claude Code hooks (moved to Codeberg) | SQLite, BM25+vector, local ONNX embeddings | None | PyPI v11.3.3 | Apache-2.0 |

## Benchmark claims and why to distrust them

- **Zep/Graphiti**: LongMemEval +18.5% over full-context at ~1.6k vs ~115k tokens (vendor). LoCoMo: Zep claims 84 / rebuttal-corrected 75.14 / Mem0's re-score 58.44 — three incompatible numbers for one system.
- **Mem0**: +26% over OpenAI memory on LoCoMo (vendor); Letta's filesystem+grep agent (74.0%) beat Mem0-graph (68.5%); Mem0's own paper shows full-context beating Mem0. Current shipped algorithm ≠ benchmarked algorithm.
- **Cognee**: self-published HotPotQA head-to-head (24 questions, LLM-judged) — too small to mean anything.
- **claude-mem**: zero formal evaluation exists, vendor "~10x token savings" unmeasured. Session-start injection ≈ 6-7k tokens on this machine's data. Known issues: unauthenticated localhost HTTP API (#1251), DB bloat, observation loss under lock contention.
- **Anthropic memory tool**: +39% on an internal agentic-search eval, 84% token reduction on a 100-turn eval — internal, not replicated.

## Fair-benchmark cohort for MemBench (pinned qwen3-coder-30b via vLLM)

Extraction-LLM-swappable (can all share the pectra endpoint): **claude-mem ≥13.9.x** (openrouter provider + custom base URL), **Mem0 OSS** (`vllm` provider), **Graphiti** (`api_base` + guided decoding), **Cognee** (LiteLLM). Zero-extraction control arm: **MEMORY.md/CLAUDE.md, basic-memory, mcp-memory-service** — no second LLM, backend-neutral by construction.

Not fairly pinnable: Mem0 cloud plugin (hosted MCP), Letta claude-subconscious (Letta Cloud only), Zep Cloud. Report separately if at all — this is the A7 feasibility tier from the main design doc.
