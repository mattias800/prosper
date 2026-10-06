---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0004: Move the shipping Vulkan backend out of tests/fixtures

## Context

The offscreen Vulkan backend that the shipping app renders through is
`prosper/tests/fixtures/render_runner.h` (16,394 lines on 61557f29), in namespace `prosper::test`,
with its `retained_depth_{array,cube}_gpu.h` companions. `frontends/shared/live/` reaches it through
`prosper::test::` and `#include "fixtures/..."`; both are baselined by the `test-dep` and
`fixture-include` ratchet rules. `tests/fixtures/AGENTS.md` carries a section headed "`render_runner.h`
is NOT test-only, and the directory name is the trap". It is also the most expensive header in the build to parse
(`docs/architecture/REFACTOR_PLAN_2026_09.md` § finding 1).

## Decision

1. Move `render_runner.h` and its two companions to `frontends/shared/backend/` with
   `tools/refactor/move_module.py`, a move-only PR (target-tree move 1). Not into `src/`: the file
   includes `frontends/shared/` twelve times, so `src/` would add twelve forbidden edges.
2. Rename the namespace `prosper::test` -> `prosper::gpu::backend` for the moved code in its own
   commit, verified by an `nm` symbol A/B (move 2).
3. Split the backend into translation units afterwards (move 3), measuring every per-draw split,
   because there is no LTO and a split can turn an inlined hot call into a real one.

## Consequences

The renderer can be found by looking at folders, `test-dep` and `fixture-include` fall towards
zero, and later work (the submit worker for `PERF-P1`/`PERF-P6`, ADR 0005's replay gate) targets a
backend with an honest name. The move conflicts with every lane editing the backend and must be
announced and landed quickly.

## Alternatives considered

- Move straight into `src/`: rejected for the twelve `src` -> `frontends` edges above
  (`ARCHITECTURE_TARGET_TREE.md` § Ruled out).
- Leave it and document the trap: the status quo, and the reason `#3210` had to say "this is not a
  test-only change" in its own body.

## Approval

Requires the project owner's acceptance (the target tree is a proposal). Accepting it unblocks
target-tree moves 1-3.
