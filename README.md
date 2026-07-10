# MemBench

MemBench is a proposed benchmark for testing whether long-term memory systems make coding agents measurably better after accounting for latency, token use, and implementation cost.

The initial research/design document is:

- [docs/agent_memory_survey_and_membench_design.md](docs/agent_memory_survey_and_membench_design.md)

Implementation notes:

- [docs/implementation.md](docs/implementation.md)

The paper and pilot are at [docs/paper/membench.tex](docs/paper/membench.tex).

## Memory arms

Arms are named by mechanism, with the product that inspired each in parentheses:

- `none` — no memory
- `raw_rag` — raw-history RAG (lexical)
- `structured` — structured markdown memory (Cline-Memory-Bank / Cursor-rules pattern)
- `claude_mem` — FTS/BM25 observations (claude-mem-style)
- `mem0` — vector-extraction memory (Mem0 OSS)
- `graphiti` — graph-edge memory (Graphiti / FalkorDB)
- `graphify` — KG-query memory (graphify)

The product arms (`mem0`, `graphiti`, `claude_mem`) run in a standardized corpus-ingestion mode: the driver ingests the same time-split corpus and calls each system's retrieval interface. Their native incremental write / consolidation pipelines do **not** execute, so these arms measure retrieval mechanisms over a driver-authored store, not the products as deployed. Cost numbers for caching arms include an O(k²) re-ingestion artifact a live incremental store would not pay.

## Threats to validity (read before citing pilot numbers)

- **Contamination.** The pinned backbone Qwen3-Coder-30B (July 2025) postdates SWE-bench-Live's April-2025 freshness window, so eval instances plausibly lie in its training data. MemBench makes **no contamination-controlled claim** for this backbone; a memorization probe and date-split are the reported mitigations.
- **Repo scope.** 6 of 39 eligible repos, all Python, chosen for fast Docker evals — per-repo reporting is mandatory. Depth k≥8 rests on ≤3 repos, k≥12 on a single repo.
- **Harness tuning.** The 64k-window harness (compact window, 400-line read cap, tool trimming) is tuned on briefcase+haystack (41% of the eval set); results are an **upper-bound regime** for memory value (cf. EvoMemBench's context-size curve).
- **Sequences.** Chronological, not interdependence-mined; carry-forward and memory-need labels are computed offline and used to stratify, not assumed.
- **Products.** See the corpus-ingestion caveat above.

The closest prior work is SWE-Bench-CL (arXiv 2507.00014); MemBench differs via a Live post-2024 base, seven standardized memory arms, a carry-forward control, and per-arm cost accounting.

Quick start:

```bash
uv run --python 3.13 python -m membench validate --instances dataset/examples/instances.jsonl
uv run --python 3.13 python -m membench run --config configs/example.toml --instances dataset/examples/instances.jsonl --output runs/demo/predictions.jsonl
uv run --python 3.13 python -m membench eval --instances dataset/examples/instances.jsonl --predictions runs/demo/predictions.jsonl --output runs/demo/report.json
uv run --python 3.13 python -m unittest discover -s tests
```

## License

MemBench is released under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): free to use for scientific research and other noncommercial purposes; commercial use and selling are not permitted.
