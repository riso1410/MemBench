# Agent Memory Systems and a Coding-Focused MemBench Design

Date: 2026-07-02 (v2 — integrates benchmark-audit, variance, feasibility, and security research)

## Executive Answer

**Are agent memory systems worth using?**

Conditionally yes — but the honest reading of the evidence is "small, real, and much cheaper to get than the memory-platform vendors suggest."

The best direct evidence on coding tasks:

| Evidence | Result | Takeaway |
| --- | --- | --- |
| ReasoningBank on SWE-bench-Verified (Gemini-2.5-flash) | 34.2% → 38.8% resolve (+4.6pp), 2.8 fewer steps/task | Distilled strategy memory gives a modest but real lift, and saves exploration steps. Extracting lessons from failures (not just successes) is what beats simpler baselines. |
| SWE Context Bench (319 interdependent GitHub tasks) | 26.3% → 34.3% with oracle summary reuse (+8pp); best practical system 30.3% | Memory helps most when tasks are genuinely related. Compact summaries (~205 words) beat full trajectories (~25k words) decisively. |
| MemoryAgentBench (ICLR 2026) | Mem0 scored 3.4% on test-time learning vs 87.6% for plain long-context models; all systems ≤6% on conflict resolution; Mem0 memory construction took ~4 hours vs 0.11s for BM25 | Heavyweight memory platforms can be catastrophically worse than doing nothing clever. Conflict resolution (stale/contradictory memory) is unsolved by everyone. |
| Letta's own benchmark | A plain filesystem + grep baseline scored 74.0% on LoCoMo, beating Mem0-graph (68.5%) and other specialized systems | Simple tools the model already knows beat purpose-built memory APIs. |
| EvoMemBench (15 memory methods) | Memory benefit shrinks with context window: +14.5pp at 16K context, +7.8 at 64K, +8.5 at 128K | As long-context gets cheaper, the memory-system value proposition shrinks (but does not vanish). |
| STALE benchmark | Best frontier model: 55.2% accuracy at detecting invalidated memories | Agents cannot yet be trusted to reject outdated memory — staleness is an active harm channel, not a corner case. |

**Which memory system should I use?**

1. **Default: structured file-based memory** (markdown/JSON notes + grep/BM25 retrieval, curated summaries not raw transcripts). It is the strongest evidence-backed baseline (Letta filesystem result), costs nearly nothing, is auditable, and is trivially reproducible.
2. **If tasks repeat within a project: add ReasoningBank-style distilled episodic memory** — short strategy/lesson items extracted from both successful and failed trajectories, injected as a few hundred tokens. This is the only approach with a measured lift on a repository-level coding benchmark.
3. **Avoid heavyweight platforms (Mem0, Zep, graph memory) for coding workloads until independently benchmarked.** Vendor numbers are adversarial and have not survived audit: Zep's 84% LoCoMo claim fell to 58.4% under corrected evaluation (25.6pp inflation from category errors, modified prompts, and single-run reporting); Mem0's 26%-over-OpenAI claim is on a benchmark (LoCoMo) whose answer key has 6.4% errors and whose LLM judge accepts 62.8% of intentionally wrong answers. Meanwhile Mem0's memory construction is ~130,000x slower than BM25 and it scored 3.4% on test-time learning.
4. **Any memory system in production needs a security and staleness story.** Query-only memory-poisoning attacks (MINJA, AgentPoison) achieve 95-98% injection success and persist across sessions; models detect invalidated memories only ~55% of the time.

**The question MemBench must answer** is therefore not "does memory improve the average score?" but:

- What lift does memory give on tasks that *actually depend on prior project history*, at what token/cost overhead?
- Does a simple filesystem/structured baseline capture most of the value of specialized systems?
- Does the value survive against long-context and repo-RAG baselines?
- What is the harm rate from stale, distracting, or poisoned memory?
- Is any of this statistically distinguishable from harness noise (which is large — see Statistical Design)?

## What Counts as Memory

For MemBench, "memory" means persistent state outside the current task prompt that can be written, updated, retrieved, and audited across tasks.

| Type | Coding-agent examples | Expected value |
| --- | --- | --- |
| Semantic memory | Repo architecture, APIs, invariants, test commands, known flaky tests | Avoids rediscovering stable facts. |
| Episodic memory | Previous attempts, error traces, issue-fix trajectories, failed patches | Avoids repeated mistakes; reuses debugging paths. ReasoningBank shows distilled episodic items are the winning format. |
| Procedural memory | Team preferences, review rules, commit style, migration playbooks | Consistency across tasks. |
| Entity/graph memory | Module ownership, dependency edges, issue-to-PR-to-file relations | Only where relationships matter; expensive to build (Graphiti requires structured-output-capable models and warns against small models). |
| Raw transcript memory | Full previous conversations and logs | Empirically dominated: SWE Context Bench found full trajectories (24,765 words avg) underperform compact summaries (205 words avg). |

## Current Evidence: What We Now Know

The evidence base changed substantially in 2025-2026. Key findings, ordered by relevance:

### Memory can help on coding tasks — modestly

- **ReasoningBank** (mini-SWE-agent, bash-only, reproducible): +4.6pp resolve rate on SWE-bench-Verified over no-memory, 2.8 steps saved per task. Its edge over Synapse/AWM comes from extracting memory items from *both successful and failed* trajectories. Memory-aware test-time scaling (MaTTS) lifts 49.7 → 55.1 with k=5 parallel samples.
- **SWE Context Bench** (arXiv 2602.08316): purpose-built for context reuse — 319 interdependent tasks mined from GitHub cross-references across 51 repos. Oracle summary reuse: +8pp absolute (31% relative). Best practical system (Supermemory): +4pp. The oracle-vs-practical gap shows summary *quality* is the bottleneck.
- **Memory condensation study** (arXiv 2605.18854, OpenHands harness): compares MemGPT tiered memory, summarization, token compression, and prompt caching on coding agents — a reusable methodological template.

### Specialized memory platforms underperform simple baselines

- **MemoryAgentBench** (ICLR 2026, open source): Mem0 3.4% vs long-context 87.6% on test-time learning; every method ≤6% on multi-hop conflict resolution; construction latency Mem0 14,644s / Cognee 8,309s / BM25 0.11s; query latency Zep 155.1s vs LightMem 3.67s.
- **Letta filesystem baseline** beats Mem0-graph 74.0% vs 68.5% on LoCoMo using only grep/search/open tools — simpler tools are better represented in training data, so agents use them more effectively.
- A 12-system evaluation (arXiv 2606.24775; harness at github.com/OpenDataBox/MemoryData) covering Mem0, Zep, Cognee, Letta, MemOS, MemoryOS, A-MEM, LightMem, SimpleMem, MemTree, MemoChat provides the most complete neutral comparison methodology to date — but not on coding tasks and not on a fully open-weight stack.

### Vendor benchmark claims do not survive audit

- **Zep/LoCoMo dispute**: Zep claimed 84%; corrected evaluation found 58.44% ± 0.20%. Causes: wrongly included the excluded adversarial category, Zep-only prompt modifications, single-run reporting vs 10-run averages. A 25.6pp harness delta — larger than the ~7.5pp genuine spread between memory systems.
- **LoCoMo itself is unreliable**: 6.4% answer-key errors (99/1540 questions); the LLM judge accepts 62.8% of intentionally wrong answers.
- **ABC (Agentic Benchmark Checklist, arXiv 2507.02825)**: flawed task setup/reward design misestimates agent performance by up to 100% relative; applying the checklist to CVE-Bench cut overestimation 33%. MemBench should be built to pass ABC.

### Memory has failure modes that average scores hide

- **Staleness**: STALE (arXiv 2605.06527) — best model 55.2% at detecting invalidated memories; models accept outdated assumptions embedded in queries.
- **False-positive retrieval**: arXiv 2604.27283 shows superficially similar stack traces/errors trigger unsafe memory injection in coding agents; learned *abstention* (knowing when not to retrieve) achieved 0.0% false positives — gating is as important as retrieval.
- **Poisoning**: AgentPoison (NeurIPS 2024) and MINJA (arXiv 2503.03704) achieve 98.2% injection success through *query-only* interaction — no store access needed — and persist across sessions (~95% persistence). For coding agents the vector is realistic: crafted issue comments or PR text that a memory system ingests. CVE-2025-53773 (Copilot RCE via poisoned issue context) shows the class is already exploited in the wild.
- **Compliance**: GDPR erasure requires multi-tier cleanup (source DB + vector indexes + derived memories); ID/namespace mappings must be planned at ingestion time.

## Benchmark Landscape (updated)

### Do NOT build on SWE-bench Verified

OpenAI's February 23, 2026 audit of 138 consistent o3 failures found **59.4% have flawed test cases** (too narrow or too wide), and all tested frontier models **reproduce exact gold patches verbatim** — training-data contamination. OpenAI stopped reporting scores and recommends others do the same. Independent corroboration: Berkeley RDI, arXiv 2506.12286 (76% vs 53% memorization gap). Additional structural problems: Django is ~46% of the 500 instances, top-5 repos >80%, 50% of issues predate 2020, ~90% are sub-1-hour fixes.

Consequence for memory research: on a contaminated benchmark, "memory lift" is confounded with "better retrieval of memorized solutions." ReasoningBank's +4.6pp must be read with this caveat.

### Usable coding benchmarks/sources

| Benchmark | State (July 2026) | Use for MemBench |
| --- | --- | --- |
| **SWE-bench-Live** | 1,888 Python tasks / 223 repos, frozen Sept 2025 (monthly cadence stalled); MultiLang (743 tasks, 6 languages), Windows (61), OS-bench variants; MIT license; issues post-Jan-2024 only | Primary contamination-safe task source. Caveat: severe repo imbalance (pdm-project/pdm 35 tasks; most repos 1-2 tasks) — must stratify. |
| **SWE-rebench** | 21,000+ Python tasks, continuous automated collection with contamination monitoring | Bulk task source and freshness pipeline model. |
| **SWE-smith** | 50,137 synthetic instances, 128 repos, $1,360 total generation cost, 295 GB (vs SWE-bench's ~50-150 TB), ~20 human-hours | The economics template for *generating* memory-dependent task sequences cheaply. Combine-Bugs strategy validates at 96.9%. |
| **SWE-Gym** | 2,438 real Python training instances + released trajectories (ICML 2025) | Source of prior-trajectory data for building episodic memory corpora. |
| **SWE Context Bench** | 319 interdependent task instances from 1,100 base tasks, 51 repos | Closest existing benchmark to MemBench's goal; reuse its dependency-mining method (multi-issue PRs, cross-references). |
| Multi-stage benchmarks (FeatureBench, SWE-EVO, EvoCode-Bench, SWE-CI) | 100-200 tasks each; multi-commit feature chains, spec drift, 71-commit evolution windows | Design references for long-horizon memory stress; too expensive to adopt wholesale. |

### Memory benchmarks (for method, not tasks)

| Benchmark | Value | Limitation |
| --- | --- | --- |
| MemoryAgentBench (ICLR 2026) | Four competencies: Accurate Retrieval, Test-Time Learning, Long-Range Understanding, Conflict Resolution; "inject once, query many" efficiency design | Conversational, not executable code. |
| LoCoMo / LongMemEval | Long-conversation recall | LoCoMo demonstrably unreliable (see above); LLM-judge scoring gameable. |
| EvoMemBench | Memory-vs-context-length scaling curves; 15 methods | Not coding-specific. |
| τ-bench | pass^k reliability metric | Domain is retail/airline agents. |
| STALE | Memory-invalidation testing methodology | Personal-state domain. |

Conclusion unchanged but sharpened: **no coding-agent memory benchmark with executable oracles, contamination controls, and statistical power exists.** SWE Context Bench is closest but single-run and without cost accounting. That is the gap MemBench fills.

## Proposed MemBench Task Definition

Each task models a coding agent returning to a project after previous work: current issue + repository snapshot, plus (for memory arms) a persisted memory store built only from strictly-earlier data.

```json
{
  "instance_id": "repo__task_id",
  "repo": "owner/name",
  "base_commit": "sha_before_fix",
  "issue": {"title": "...", "body": "...", "created_at": "..."},
  "gold_patch": "...",
  "test_patch": "...",
  "oracle": {"fail_to_pass": ["test_a"], "pass_to_pass": ["test_b"]},
  "memory_corpus": {
    "cutoff_time": "...",
    "allowed_sources": ["past_issues", "past_prs", "past_ci_logs",
                        "past_agent_trajectories", "docs_at_or_before_cutoff"]
  },
  "memory_need_labels": {
    "requires_memory": true,
    "memory_type": ["semantic", "episodic"],
    "evidence_items": ["..."],
    "carry_forward_checked": true
  },
  "budgets": {"max_wall_time_sec": 1800, "max_input_tokens": 200000,
              "max_output_tokens": 20000, "max_cost_usd": 5.0}
}
```

### Task Categories

| Category | Example | Why it matters |
| --- | --- | --- |
| Hidden setup memory | Prior issue found tests only pass with a specific env var. | Memory avoids repeated setup exploration. |
| Repeated bug pattern | Same root cause in a different module months later. Real-world basis: duplicate bug reports are 12-30% of issues (VS Code/Thunderbird >25%, Mozilla up to 30%). | Episodic reuse; also the most ecologically valid category. |
| Project convention | Repo-specific API/migration style not visible in touched files. | Semantic/procedural memory. |
| Stale memory trap | Old architecture note invalidated by a refactor. Grounded in STALE's 55.2% detection ceiling. | Forgetting, timestamps, conflict handling. |
| Distractor memory | Many similar prior issues, one relevant. Grounded in arXiv 2604.27283 false-injection findings. | Retrieval precision and abstention. |
| Poisoned memory (adversarial split) | MINJA-style bridging text planted in a prior issue/PR comment. | Security dimension; must be a separate opt-in split. |
| No-memory-needed | Issue and repo contain everything. | Overhead and false-positive memory use. |
| Long-context substitute | Relevant history stuffed into a huge context. | Memory vs brute-force context (EvoMemBench predicts shrinking but nonzero gap). |

### The carry-forward validity control (critical)

Analysis of 40 consecutive Django task pairs found **70% of task N's fix lines are already present in task N+1's base checkout** (12.5% partial, 17.5% absent). If uncontrolled, "memory lift" is inflated: the agent isn't remembering — the answer is in the repo. Every task sequence must record, per pair, whether the earlier fix is present in the later checkout, and memory-lift analysis must be reported separately for carry-forward-present vs absent pairs. This control does not exist in any published memory evaluation.

## Dataset Collection Plan

### Sources and construction

- **Base tasks**: SWE-bench-Live (post-2024 issues, contamination-safe) + SWE-rebench for volume; mine interdependent sequences via SWE Context Bench's method (multi-issue PRs, cross-references, second-order dependency expansion).
- **Synthetic augmentation**: SWE-smith's approach (single fixed commit per repo, shared environment) makes generating *controlled* memory-dependent sequences cheap (~$1,360 for 50k instances; 96.9% validation yield for combined-bug strategy). Use it to build stale-trap and distractor cases where natural data is scarce.
- **Episodic corpora**: SWE-Gym's released trajectories + trajectories generated by our own baseline runs on earlier tasks in each sequence (clearly marked synthetic).
- **Repositories**: 8-10 repos minimum (not Django-dominated), per-repo task caps, chosen for active issue history, container-reproducible tests, permissive licenses.

### Time-split rules

Allowed memory data strictly precedes the task timestamp: closed issues/PRs, commits, review comments, release notes, docs, CI logs, prior-task trajectories. Disallowed: the gold patch, anything post-cutoff, benchmark annotations.

### Memory dependency labels

| Label | Definition |
| --- | --- |
| `M0` | No expected memory dependency. |
| `M1` | Memory may help; repo retrieval should suffice. |
| `M2` | Prior history contains a reusable fact; memory likely helps. |
| `M3` | Memory required or nearly required without expensive exploration. |
| `S` | Stale-memory trap: correct behavior requires rejecting old memory. |
| `D` | Distractor-heavy corpus. |
| `P` | Poisoned corpus (adversarial split only). |

Each `M2/M3/S/D/P` label must cite evidence events; human audit required (ABC checklist discipline — this is where SWE-bench Verified failed).

## Statistical Design (new — this decides whether results mean anything)

Measured reality on SWE-bench-style evals:

- Run-to-run pass@1 varies **2.2-6.0pp across identical reruns** (run-level SD 0.7-1.8pp), persisting at temperature 0 (Docker/test flakiness, API nondeterminism). Example: Devstral-2/nano-agent 63.8 ± 1.6%, range 60.6-66.6% over 10 runs.
- Typical memory-system effect sizes are 1-5pp — the same order as the noise.
- Anthropic's "Adding Error Bars to Evals" (arXiv 2411.00640): clustered standard errors run up to 3x naive SEs when instances cluster by repo; paired per-instance comparison is essential.
- The strongest prior coded-memory study (Stompy, Feb 2026) is n=1 per cell — its 22-32% cost-saving claims sit inside single-run noise. This is the norm, not the exception.

**Design requirements** (derived from power analysis):

| Parameter | Value | Rationale |
| --- | --- | --- |
| Task set | 150-200 tasks, 8-10 repos, per-repo caps | N=200 → run-level σ ≈ 2.4pp; counters Django-style clustering. |
| Seeds | 4-5 per condition | N=200 × 4 seeds ≈ 780 paired task-runs → 80% power for a 5pp difference. N=100 would need ~7 seeds. |
| Orderings | 3 task orderings, treated as a random factor | Ordering materially affects memory accumulation (AgentCL: 9.4% cross-method SD on compositional streams vs 3.0% naive). |
| Tests | McNemar on discordant pairs; paired bootstrap (10k resamples); repo-clustered bootstrap; mixed-effects over seed/ordering | Matches binary paired outcomes + clustering. |
| Metrics | pass@1 pooled, plus pass@k / pass^k envelope; learning curves (streaming accuracy, AULC), not just endpoints | pass^k decays as p^k (90% → 57% at k=8); memory's benefit may be *consistency*, invisible in means. The pass@5↔pass^5 envelope spans up to 24.9pp. |
| Oracles | Executable tests only; no LLM judges | LoCoMo judge accepted 62.8% of wrong answers. |
| Transparency | Publish all logs, prompts, per-instance binaries, dataset-generation code | HAL precedent: 2.5B tokens from 21,730 rollouts released. |

**Budget**: a 500-task run costs $14-38 depending on model and caching (~213M tokens); a 100-task streaming run ≈ $7. The full pilot design — 4 conditions × 5 seeds × 3 orderings = 60 runs — lands at **$400-800**. Grading infrastructure is fast (Epoch AI grades 500 tasks in 62-70 min on a 32-core runner). This is affordable; there is no excuse for n=1.

## Systems to Compare

Same agent scaffold (mini-SWE-agent recommended: ~100 lines, bash-only, no tool-calling API dependency, >74% on SWE-bench Verified with frontier models), same tool budget; only the memory layer changes.

| Arm | Description | Purpose |
| --- | --- | --- |
| `A0_no_memory` | Issue + repo tools only. | Required baseline. |
| `A1_long_context_history` | Relevant history stuffed into context within budget. | EvoMemBench says this is the toughest baseline to beat. |
| `A2_repo_rag` | BM25/vector retrieval over current repo only. | Separates repo search from longitudinal memory. |
| `A3_filesystem_memory` | Plain files + grep/search tools (Letta-baseline style). | The simple-tools champion; likely the practical winner. |
| `A4_structured_project_memory` | Curated typed facts/lessons, lexical retrieval. | Recommended practical baseline. |
| `A5_reasoningbank_episodic` | Distilled strategy items from prior success+failure trajectories. | The only approach with measured coding-task lift. |
| `A6_hybrid` | A4 + A5 + repo retrieval. | Expected pragmatic best. |
| `A7_product_memory` | Mem0 OSS / Letta / Cognee / Graphiti under identical constraints. | Tests platform claims neutrally. |

Memory **writes** are benchmarked, not just reads: post-task proposed writes are scored by later-task utility and audited for noise (write precision was hypothesized — and MemoryAgentBench construction-cost data confirms — to be a first-order bottleneck).

### Feasibility constraints on the product arm (from the pinning audit)

| Tier | Systems | Status |
| --- | --- | --- |
| 1 — fully pinnable to open weights | Mem0 OSS v2.0.11, Graphiti v0.29.2, Cognee v1.2.1, Letta v0.16.8 | OpenAI-compatible endpoints (vLLM/Ollama) supported; include in benchmark. |
| 2 — pinnable but stale | LangMem v0.0.30 (last release Oct 2025) | Stability risk; exclude or flag. |
| 3 — not fairly pinnable | claude-mem (Claude/Gemini/OpenRouter only), Anthropic memory tool (API-only) | Report separately; cannot enter the open-weight comparison without confounds. |

Known confounds to control:

- **Backbone structured-output capability changes rankings**: memory-extraction format-error rates go from 1.2% (gpt-4o-mini) to 30.4% (Qwen-2.5-3B). Graphiti explicitly warns against small models; Mem0 has 5+ open issues on empty memories with local models; Cognee silently falls back to OpenAI if a component is unconfigured. All arms must use the same backbone, verified to support JSON mode.
- **Nondeterminism must be disabled**: Mem0 `async_mode` (default true since v1.0.0), Letta sleep-time agents, Graphiti async ingestion.
- **Four unfair-comparison classes to refuse**: claude-mem vs OSS systems; Zep-cloud vs Graphiti-OSS; Mem0-platform vs Mem0-OSS; any comparison across backbones with different JSON-mode reliability.

## Metrics

### Primary quality

| Metric | Definition |
| --- | --- |
| `resolved` | Fail-to-pass tests pass and pass-to-pass tests still pass. |
| `memory_lift` | `resolved(memory) - resolved(no_memory)`, paired per instance. |
| `memory_specific_lift` | Lift on `M2/M3` only; reported separately for carry-forward-present vs absent pairs. |
| `harm_rate` | Memory fails where no-memory succeeds (paired). |
| `stale_memory_failure_rate` | Failures from trusting outdated memory on `S` tasks. |
| `pass^k` | All-k-trials consistency; tests whether memory improves reliability, not just means. |
| `poison_susceptibility` | On `P` split: fraction of poisoned records retrieved and acted upon. |

### Cost and efficiency

Unchanged from v1 (input/output tokens, memory tokens injected, retrieval calls, write counts, store size, wall time, `cost_per_resolved_task`, `marginal_cost_per_extra_resolution`) — plus:

| Metric | Definition | Why (evidence) |
| --- | --- | --- |
| `memory_construction_time` | Wall-clock to build the store from the corpus | Mem0 4h vs BM25 0.11s on identical input — this alone can decide the verdict. |
| `steps_per_task` | Agent iterations to resolution | ReasoningBank's clearest win was −2.8 steps. |
| `query_latency` | Per-retrieval latency | Zep 155s vs LightMem 3.7s per query. |

### Memory quality

`evidence_recall@k`, `evidence_precision@k`, `citation_rate`, `write_precision`, `write_staleness` — plus `abstention_accuracy`: on `M0/D` tasks, does the system correctly *not* inject memory (the 0.0%-false-positive gating result shows this is learnable and decisive).

### Decision metric

```text
net_value = quality_lift - λ_cost·extra_cost - λ_latency·extra_latency - λ_harm·harm_rate
```

Report Pareto frontiers. A system gaining 2pp at 5x cost loses to one gaining 1pp at 1.1x.

## Reproducibility Design

SWE-bench harness pattern: Dockerized per-instance environments, immutable snapshots, fail-to-pass/pass-to-pass oracles, JSONL predictions, re-runnable evaluation containers, public logs.

```text
dataset/
  instances.jsonl
  memory_corpora/<instance>/
    events.jsonl  documents.jsonl  gold_evidence.jsonl
  docker/images.lock
harness/
  run_agent.py  run_evaluation.py
  memory_adapters/   # none, filesystem, raw_rag, structured, reasoningbank,
                     # mem0_oss, letta, cognee, graphiti
  scoring/           # test_oracle, cost, memory_quality, poison
```

Model tiers:

| Tier | Purpose | Notes |
| --- | --- | --- |
| **Qwen3-Coder-30B** (vLLM 0.24.0 on pectra RTX 5090, `configs/pectra.toml`, pinned checksum) | Main reproducible tier — fixed choice | Must have reliable JSON mode (structured-output confound). Demo six-arm run showed it resolves 0/1 in single-shot mode — it must run inside the agentic scaffold (mini-SWE-agent-style loop), not one-prompt patch generation. |
| Small open-weight model | Does memory compensate for weak models? | Expect memory-system breakage (30% format errors) — that breakage is itself a result. |
| Frontier API model | Ceiling/product tier | Record exact model ID, date, decoding params. |
| Long-context frontier model | Memory-vs-context frontier | EvoMemBench curve replication on code. |

Anti-contamination: post-training-cutoff issues only (SWE-bench-Live rule: post-Jan-2024), time-split memory corpora, dataset-generation code published (not just JSON), canary tasks from private forks, memorization probes (ask the model for the patch with no repo access — arXiv 2506.12286 method) reported per instance.

## Security and Staleness Splits (new)

Memory is an attack surface, and MemBench should be the first coding benchmark to measure it:

- **Poisoning split (`P`)**: plant MINJA-style bridging content in pre-cutoff issues/PR comments; measure injection success into the store, retrieval rate, and action rate. Baseline expectation from literature: ~98% injection, ~77% attack success on undefended systems; Anthropic-style anti-injection framing blocks ~88% vs 74% undefended.
- **Staleness split (`S`)**: refactor-invalidated facts; measures the STALE gap (55.2% detection) in an executable setting.
- Report `poison_susceptibility` and `stale_memory_failure_rate` alongside quality — a system whose lift is +3pp but which persists poisoned instructions across sessions is not "worth it" in production.

## Hypotheses

1. Filesystem/structured memory captures ≥70% of the best system's lift at <10% of its cost (Letta-baseline result generalizes to code).
2. Distilled episodic memory (ReasoningBank-style) beats raw-transcript RAG on `M2/M3`; full trajectories lose to ~200-word summaries.
3. Memory lift on `M0` is ≈0 or negative; harm concentrates in `D`/`S` categories.
4. The long-context arm closes most of the gap at 128K context (EvoMemBench curve), but memory retains a step-count/cost advantage.
5. Write precision is the bottleneck: bad writes accumulate and degrade later tasks in a sequence.
6. Uncontrolled carry-forward would have inflated measured memory lift by 2x or more (testable directly from the paired analysis).
7. No system exceeds 10% conflict-resolution/stale-rejection accuracy gain over baseline (the field-wide ≤6% CR result holds on code).

## First Milestone (pilot)

- 100-150 tasks from 8-10 repos (SWE-bench-Live derived), sequences mined via cross-reference dependencies, carry-forward annotated per pair.
- Mix: ~30% `M0`, ~30% `M1`, ~25% `M2/M3`, ~15% `S`+`D`. (Poison split `P` deferred to phase 2.)
- Arms: `A0`, `A1`, `A2`, `A3`, `A5` (add `A7` products only if pilot shows headroom above `A3`).
- One pinned open-weight code model (Qwen3-Coder-30B on pectra — free, local, fixed) + one frontier model; 4-5 seeds × 3 orderings; ~$400-800 (frontier arm only; the qwen arm costs electricity).
- Report: paired memory lift with McNemar + clustered bootstrap CIs, harm rate, pass^k, cost/latency table, evidence recall, carry-forward-stratified lift.

**Decision rules:**

- If `A3`/`A5` don't beat `A0` by ≥5pp on `M2/M3` (paired, significant) while keeping harm ≤3pp on `M0/S`: memory is not worth it for this workload — publish that.
- If `A3` (filesystem) is within noise of `A5`/products: the answer is "use files + grep, skip the platforms" — publish that.
- Only if products show significant headroom above `A3` does a phase-2 platform bake-off make sense.

## Survey/Paper Structure

Title: *Are Agent Memory Systems Worth It for Coding Agents? A Statistically Powered, Contamination-Controlled Benchmark*

1. Introduction: memory is popular; coding-agent value is unproven; existing evidence is n=1, contaminated, or vendor-run.
2. Taxonomy: semantic, episodic, procedural, graph, raw transcript.
3. Why existing comparisons fail: SWE-bench Verified audit, Zep/LoCoMo dispute, ABC findings, variance literature, carry-forward hazard.
4. MemBench dataset: longitudinal repo tasks, time-split corpora, dependency labels, carry-forward control, adversarial splits.
5. Harness: SWE-bench-style executable evaluation + memory adapters + pinning matrix.
6. Statistical methodology: power analysis, paired tests, clustered errors, pass^k, learning curves.
7. Experiments: baselines vs structured vs episodic vs products, across model tiers and context lengths.
8. Results: lift/harm/cost frontier; staleness and poisoning susceptibility.
9. Recommendation by task profile and budget.
10. Limitations: OSS model drift, product API churn (Zep CE deprecated Apr 2025), task-selection bias, synthetic-sequence realism.

## Source Notes

Benchmark validity and statistics:

- OpenAI SWE-bench Verified audit (Feb 23, 2026): https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- Memorization gap in coding agents: https://arxiv.org/abs/2506.12286
- ABC checklist: https://arxiv.org/abs/2507.02825
- Adding Error Bars to Evals: https://arxiv.org/abs/2411.00640
- SWE-bench variance analyses: https://arxiv.org/pdf/2602.07150 , https://www.ai21.com/blog/scaling-agentic-evaluation-swe-bench/
- τ-bench (pass^k): https://arxiv.org/abs/2406.12045

Task sources:

- SWE-bench-Live: https://arxiv.org/abs/2505.23419 , https://huggingface.co/datasets/SWE-bench-Live
- SWE-rebench: https://swe-rebench.com/
- SWE-smith: https://swesmith.com/
- SWE-Gym: https://arxiv.org/abs/2412.21139
- SWE Context Bench: https://arxiv.org/abs/2602.08316
- SWE-bench harness reference: https://www.swebench.com/SWE-bench/reference/harness/
- mini-SWE-agent: https://github.com/SWE-agent/mini-swe-agent
- OpenHands: https://github.com/All-Hands-AI/OpenHands

Memory systems and evaluations:

- ReasoningBank: https://arxiv.org/abs/2509.25140
- MemoryAgentBench (ICLR 2026): https://arxiv.org/abs/2507.05257 , https://github.com/HUST-AI-HYZ/MemoryAgentBench
- 12-system evaluation: https://arxiv.org/abs/2606.24775 , https://github.com/OpenDataBox/MemoryData
- Letta filesystem-baseline post: https://www.letta.com/blog/benchmarking-ai-agent-memory/
- Zep/LoCoMo dispute: Mem0 GitHub issue (May 8, 2025); Mem0 paper: https://arxiv.org/abs/2504.19413
- EvoMemBench: https://arxiv.org/abs/2605.18421
- Memory condensation for coding agents: https://arxiv.org/abs/2605.18854
- STALE: https://arxiv.org/abs/2605.06527
- Selective memory retrieval / false injection: https://arxiv.org/abs/2604.27283
- MemGPT: https://arxiv.org/abs/2310.08560 ; Letta docs: https://docs.letta.com/guides/agents/memory
- Graphiti config constraints: https://help.getzep.com/graphiti/configuration/llm-configuration
- Anthropic memory tool: https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
- claude-mem: https://github.com/thedotmack/claude-mem

Security:

- AgentPoison (NeurIPS 2024): https://arxiv.org/abs/2407.12784
- MINJA: https://arxiv.org/abs/2503.03704
- CVE-2025-53773 (Copilot prompt-injection RCE), Trail of Bits, Aug 2025
