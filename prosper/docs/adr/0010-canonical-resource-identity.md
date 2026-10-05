---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0010: One canonical resource per guest allocation, validated by page tracking

## Context

Every cached GPU source must be proven unchanged before reuse. #3155 records the three proofs, tried
cheapest-first: the intra-submit GPU write journal, an armed page-protection watch, and a full
`memcmp` against a retained snapshot. A watch arms only after
`PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_HITS` consecutive successful full compares (default 3), so every
cached texture pays three complete scans of itself first. Measured on *The Plucky Squire*: 67.0 GB
of `memcmp` traffic in a 30 s window at the default, `__memcmp_evex_movbe` the largest symbol at
5.31% of CPU. This is the `PERF-P5` violation.

The established pattern is page-granular tracking owned by one memory model: yuzu's buffer and
texture caches keep a memory tracker that marks guest pages dirty on CPU write and invalidates every
cached object over them, so a clean page is reused with no byte comparison. It is a structural
reference only.

## Decision

1. Guest memory has one owner of "what changed": a page-granular tracker in `guest/memory` (ADR
   0003), written by CPU write-watch faults, GPU writebacks and DMA, and read by every cache.
2. A guest allocation has one canonical host resource identity. Several guest virtual mappings of
   the same physical allocation resolve to it, so a write through one alias invalidates the others.
3. A cache validates by asking the tracker whether any page under its source changed since it was
   built. A full compare remains only as the fallback where tracking is genuinely unavailable, and
   that fallback is counted (`host-copy-pressure`), not silent.
4. #3155's promotion threshold is the first step on this path, not the end state.

Adds spec rule `PERF-P9`.

## Consequences

Per-draw cost stops scaling with resource size. The hard part is fault-tracking coverage: memory
written by paths that do not fault (host-side copies, other processes' views) must report into the
tracker explicitly, and a missed path is a stale-content bug, so each new writer needs a test that
writes through it and asserts invalidation. `CONFIDENCE: MED` that one tracker can serve compute,
texture and buffer caches without per-cache exceptions.

## Alternatives considered

- Lowering the promotion threshold only (#3155): a large measured win and worth taking, but each
  cached texture still pays a full scan before tracking engages.
- Hashing instead of comparing: still O(size) per validation.

## Approval

Requires the project owner's acceptance and ADR 0003 (the tracker's home).
