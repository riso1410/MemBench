# Naming: "MemBench" collision and candidate replacements

**Status: recommendation only. Nothing is renamed. The decision is the user's.**

## Why revisit the name

- **Direct collision.** "MemBench" is already taken by Tan et al., *MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents*, ACL 2025 Findings (arXiv:2506.21605) — same domain (agent memory evaluation). This is a hard clash, not a near-miss.
- **Crowded suffix.** The `-MemBench` construction is saturated: EvoMemBench, StreamMemBench, VehicleMemBench, AdaptMemBench (arXiv:1812.07778, memory-subsystem benchmarking), plus adjacent MemoryBench (arXiv:2510.17281) and MemGym. A new `-MemBench` name inherits the crowding.

A distinctive name avoids citation confusion and search-engine dilution against an ACL-published namesake.

## Candidates and arXiv-collision checks

Checks run via web search, July 2026. "Clear" = no same-domain paper/benchmark found under that exact name.

| Candidate | Collision check | Verdict |
| --- | --- | --- |
| **CodeMemBench** | No paper found under this exact name. But sits inside the crowded `-MemBench` family (AdaptMemBench is a memory-subsystem benchmark; MemoryBench, EvoMemBench adjacent). Descriptive but generic. | Usable, low distinctiveness |
| **SWE-Mem** | **Direct collision** — *SWE-MeM: Learning Adaptive Memory Management for Long-Horizon Coding Agents* (arXiv:2606.28434). Same domain (coding-agent memory). | **Rule out** |
| **MemWorth** | No paper found. Distinctive; foregrounds the paper's thesis ("is memory worth it?"). Downside: does not signal the coding/SWE domain. | Strong, thesis-forward |
| **CarrySWE** | No paper found (searched "CarrySWE", "Carry-SWE", + memory/agent qualifiers). Distinctive; signals the SWE domain **and** the carry-forward validity control, which is this benchmark's most novel methodological contribution. | **Strong, differentiator-forward** |
| **SessionBench-Code** | No direct "SessionBench" collision, but SessionIntentBench (arXiv:2507.20185, e-commerce) is nearby and the `-Bench` suffix is extremely crowded (LiveCodeBench, LoCoBench, etc.). Generic. | Usable, low distinctiveness |

## Recommendation

**CarrySWE.**

Rationale:
1. **Clear namespace** — no arXiv or benchmark collision found (unlike MemBench and SWE-Mem).
2. **Signals the domain** — the `SWE` element places it unambiguously in software-engineering-agent evaluation, where MemWorth is domain-neutral.
3. **Names the contribution** — "Carry" foregrounds the carry-forward control (the finding that ~70% of a task's fix is already in the next task's base checkout, which uncontrolled inflates "memory lift"). That control is what most distinguishes this work from SWE-Bench-CL and every vendor benchmark, so the name doubles as positioning.

Runner-up: **MemWorth**, if a thesis-forward, domain-neutral name is preferred over a differentiator-forward one. Avoid **SWE-Mem** (hard collision) and treat **CodeMemBench** / **SessionBench-Code** as safe-but-forgettable fallbacks.

Whatever is chosen, do a fresh exact-string arXiv + Google Scholar check immediately before first public posting; the 2026 memory-benchmark literature is moving fast.
