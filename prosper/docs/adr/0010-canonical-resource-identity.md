---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0010: One canonical resource per guest allocation, validated by page tracking

## Context

Every cached compute source must be proven unchanged before reuse. Three proofs are tried
cheapest-first: the intra-submit GPU write journal, an armed page-protection watch, and a full
`memcmp` against a retained snapshot (#3155). This is where `PERF-P5` is violated, and #3155's
issue body overstated how. **Its own corrections, which this ADR follows, say:**

- The 67.0 GB vs 30.5 GB pair in the issue body was an instrument artifact: a `memcmp` interposer
  that dumps its tally periodically, read at 2^21 and 2^20 calls, so the two arms measured
  different lengths of run. Re-measured at 2^16 granularity in both orders, the default spends
  63-65 GB per window on *The Plucky Squire*, and `hits=2` is indistinguishable from it. Only
  `hits=1` moves the number (28-29 GB), and that value arms a watch before any validation has run.
  #3158, which lowered the default to 2, was closed for changing a default with no measured benefit.
- The reason the threshold cannot matter: the stability counter that gates promotion lives on the
  **cache entry**, advances only on a matching full compare, and is zeroed by every storage
  writeback and every changed acquisition. `PROSPER_WATCH_PROMOTE_CENSUS` found **94.8% of promotion
  decisions taken against a counter of 0**. With high texture turnover, an entry is evicted or
  rewritten before it ever earns credit, so the credit dies with the entry. "Three scans, then the
  watch" is not what happens; for the dominant population the watch never arms.
- #3440 fixed a demonstrated buffer case (result baselines evicting primary buffers, and unchanged
  publication resetting progress), after which a *Sonic* buffer population validates through the
  write watch at the unchanged default. The image-turnover population above is not settled by it.

The pattern the evidence points at is page-granular tracking owned by memory, not by caches:
yuzu's buffer and texture caches keep a memory tracker that marks guest pages dirty on CPU write and
invalidates whatever is cached over them, so a clean page is reused with no comparison and no
per-entry history to lose. It is a structural reference only.

## Decision

1. Guest memory has one owner of "what changed": a page-granular tracker in `guest/memory` (ADR
   0003), written by CPU write-watch faults, GPU writebacks and DMA, and read by every cache.
2. Whether a source is unchanged is a property of its **pages**, which outlive any cache entry.
   A new entry over pages that have not been written since they were last validated is reused
   without a compare, which is exactly the case the per-entry counter can never reach.
3. A guest allocation has one canonical host resource identity; several guest virtual mappings of
   the same allocation resolve to it, so a write through one alias invalidates the others.
4. A full compare remains only where tracking is genuinely unavailable, and that fallback is
   counted (`host-copy-pressure`), not silent.

Adds spec rule `PERF-P9`.

## Consequences

Validation cost stops depending on how long a cache entry survives, which is the property the
measured workloads lack. The hard part is coverage: every writer that does not fault (host-side
copies, GPU writebacks, DMA) must report into the tracker, and a missed path is a stale-content
bug, so each writer needs a test that writes through it and asserts invalidation. `CONFIDENCE: MED`
that one tracker can serve compute, texture and buffer caches without per-cache exceptions.

## Alternatives considered

- Tuning the promotion threshold: measured as a no-op for every value that waits for a validation
  (#3155); the only value that helps skips validation entirely.
- Keeping per-entry credit across eviction: still history on the wrong object; a page tracker
  carries the same information without depending on cache policy.
- Hashing instead of comparing: still O(size) per validation.

## Approval

Requires the project owner's acceptance and ADR 0003 (the tracker's home).
