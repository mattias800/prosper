---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0011: A typed GPU command representation between PM4 decode and the backend

## Context

`src/gpu/pm4/command_processor.cpp` decodes the guest's command stream and the executor
(`src/gpu/execute/`) realises draws and dispatches against the Vulkan backend. A representation of
the work already exists in pieces: `SubmitOperation` (`Draw`, `Dispatch`, `DmaCopy` with a
command order, `gpu/execute/gpu_execute.hpp`), the capture model in `gpu/capture/`, and a dependency
graph over operations and their guest accesses (`gpu/execute/gpu_dependency_graph.hpp`). None of
them is the single contract between decode and execution, so state resolution, barrier decisions and
backend calls are interleaved in very large bodies (`execute_item`, `render_draw_pass_rgba`,
`resolve_dynamic_fetch` in `docs/architecture/REFACTOR_PLAN_2026_09.md`).

DXVK and vkd3d-proton separate the API's state from the backend's commands, and compilers
generally separate a representation from the passes that transform it. Structural references only.

## Decision

`CONFIDENCE: LOW` on the exact shape; this ADR proposes the direction and a survey, not a format.

1. Decode produces an ordered list of typed operations per submit -- draw, dispatch, copy, clear,
   barrier/wait, label write, flip -- each with its resolved state and its guest reads and writes.
   `SubmitOperation` and the dependency graph are the starting point, extended rather than replaced.
2. Execution consumes only that list; the backend never sees PM4.
3. Deterministic passes run over the list before execution: redundant-state elimination, barrier
   inference from the recorded accesses, and batching of compatible operations. Each pass is a pure
   function with a replay test (ADR 0005) showing guest-visible output unchanged.
4. The list is what a capture records, so replay, capture and live execution share one format.

Adds spec rule `GPU-2`. A survey of which fields decode resolves today, and where the executor
re-derives state, precedes any code change and is recorded in this ADR's successor.

## Consequences

Execution becomes testable without PM4, optimisations become separable passes instead of edits to
thousand-line bodies, and a second backend becomes possible later without touching decode. It is
the largest of these proposals and depends on ADR 0004 (backend location) and ADR 0005 (replay).

## Alternatives considered

- Keep decode-to-execution direct and only split the large bodies: reduces file size, but leaves
  barrier and state decisions spread across the bodies that issue them.
- Adopt the capture format unchanged as the representation: it records what was submitted, not the
  resolved state execution needs.

## Approval

Requires the project owner's acceptance; the survey in point 4 may proceed beforehand, since it
changes no code.
