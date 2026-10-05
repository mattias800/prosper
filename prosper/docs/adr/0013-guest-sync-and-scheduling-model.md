---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0013: One guest synchronisation and scheduling model, with no thread-identity assumptions

## Context

`src/hle/sync/` already holds the shared machinery: guest-address futex wait/wake, interruptible
mutex/condition wrappers, quarantine-then-reclaim for destroyed sync objects (#2042, #2176), the
guest handle scheme, and the fiber and ULT schedulers. The GPU command processor waits on and wakes
guest completion labels through the same primitives -- which is correct (one waker, one wait), and
also one of the baselined `gpu` -> `hle` include inversions, because the machinery lives in the top
layer.

Fiber titles exposed the missing invariant. In *Uncharted: Legacy of Thieves Collection* fibers start
on the main thread and are resumed by workers, and three defects had one shape: code assuming the
host thread that enters an HLE call is the one that returns from it (#3615 fixed, #3638, #3623;
`docs/games/UNCHARTED_STATUS.md`).

## Decision

1. The guest sync and scheduling machinery moves from `hle/sync` to `guest/sync` (ADR 0003), so
   `gpu` reaches it without an inversion and `hle` keeps the `scePthread*` / `sceKernel*` handlers.
2. One wait model: every blocking wait -- guest futex, condition, semaphore, event flag, GPU label,
   fence retirement (ADR 0009) -- goes through it, carries the identity of what it waits for, and is
   visible to the `hle-blocking-wait` and `gpu-sync-wait` accounting.
3. Invariant: no state keyed by host thread may be assumed to survive across an HLE call that can
   switch fibers. Per-fiber state lives with the fiber's guest context; per-host-thread state is
   re-read after any call that can migrate.
4. Each fiber-migration defect becomes a test that migrates a fiber across host threads mid-call.

Adds spec rule `SYNC-1`.

## Consequences

A class of defect that so far surfaced one title at a time gets a rule and tests, and a `gpu` ->
`hle` inversion disappears by relocation. Which host-thread-keyed state exists today has not been
surveyed; that census is the first step and may find more instances of the #3638 shape.

## Alternatives considered

- Fix each fiber defect where found: what has happened so far, three times.
- Make the GPU keep its own waiting primitives: rejected by `hle/sync`'s own reasoning, a waker and
  its wait must not drift apart.

## Approval

Requires the project owner's acceptance and ADR 0003.
