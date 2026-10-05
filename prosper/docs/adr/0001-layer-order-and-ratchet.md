---
kind: adr
status: accepted
date: 2026-10-02
---

# ADR 0001: Freeze structural debt with a down-only ratchet and a fixed layer order

## Context

Structural costs -- title ids in shared code, raw `getenv` reads, blocking GPU syncs, oversized
files, the shipping backend under `tests/`, host-platform `#if` outside the seam, includes against
the intended dependency direction -- were documented in prose and kept growing, because prose has
no failing state. Fixing them all at once is not possible while several lanes edit the same files.

## Decision

`tools/ci/check_arch_ratchet.py` holds a per-file count for each structural cost. A count may go
down, never up, and CI gates on delta mode (only counts the change itself raised). The include
order of `prosper/src` is the `LAYER_ORDER` constant in that file; `docs/spec/layers.md` renders
it. Spec rules `LAY-1`, `LAY-2`, `LAY-3`, `PLAT-1`, `PLAT-2`, `TITLE-1`, `TITLE-2`, `CFG-1`,
`SIZE-1`, `PERF-P1` and `PERF-P2` cite the ratchet rules that enforce them.

## Consequences

Debt cannot grow while the moves that remove it are pending. A genuinely needed rise is a baseline
edit with a reviewer-visible note in the same PR. The ratchet is a text scan: it is a proxy that
can be evaded by a construct it does not match, which is why ADR 0008 proposes moving the layer
rule into the build.

## Alternatives considered

- A full-tree gate that fails on any count above baseline: rejected after it reddened unrelated PRs
  whenever `main`'s baseline went stale (#4173); delta mode replaced it.
- Fixing the debt before gating it: rejected, the moves span months and many lanes.

## Approval

Merged as #4199, #4198 (delta mode) and #4215 (2026-10-02) and enforced in the `Docs` CI job; recorded here as a
backfill.
