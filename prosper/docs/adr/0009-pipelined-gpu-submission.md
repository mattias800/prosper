---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0009: Pipeline GPU submission, retiring guest-visible effects in stream order

## Context

The renderer runs synchronously inside the guest's submit. #3948 measured one 5.03 s F8 window of
*Grand Theft Auto V* gameplay: 1,175 ms of the executor thread spent waiting, every compute dispatch
a `vkQueueSubmit` followed immediately by `vkWaitForFences`, every graphics batch submitted and
waited in its destructor, and the GPU 5-15% busy (#3873). This is the `PERF-P1` and `PERF-P6`
violation recorded in `docs/spec/performance.md`. #3948 is a staged design for removing it; it is
a design, not a change, and no stage past 0 has landed.

The comparable established design is DXVK's command-stream thread: API calls are recorded into
chunks that a worker consumes, so the application thread never waits for the GPU unless it reads a
result. It is a structural reference only.

## Decision

Adopt #3948's staged design as the architecture for GPU submission, with its observation-point rule
as the invariant: a guest-visible effect -- a completion label write, an EOP event, a flip, a guest
memory writeback, a write-watch invalidation -- is applied only after all GPU work and writebacks
that precede it in the command stream, and in stream order. Everything else the executor waits for
today is internal and may be deferred.

1. Stage 0: always-on `GpuSyncWait` cost in the perf ledger and the `gpu-sync-wait` alarm (the
   alarm rule exists; the ADR fixes it as the stage-gate metric).
2. Stage 1: compute dispatches pipelined within a submit, completion records retired at the first
   observation point (RAW/WAW overlap, a cache lookup, a graphics import, a pend-queue write, submit
   end), in submission order.
3. Stage 2: graphics batch waits deferred to their next consumer.
4. Stage 3: the submit returns before the GPU finishes; a completion thread applies guest-visible
   effects in stream order. This is also where frame pacing belongs: a bounded number of frames in
   flight (`PERF-P6`), no separate runtime layer.
5. Every stage ships behind a default-off switch with a verification mode that retires inline and
   compares guest bytes, and is A/B'd on the reference workloads before its switch is removed.

Adds spec rule `PERF-P8` (stream-order retirement).

## Consequences

The executor stops serialising CPU and GPU work, which is the largest measured cost in the current
renderer. The risk sits in Stage 3 (label reuse, concurrent guest submits, write-watch ordering;
#312 was an early label write) and is why every stage needs ADR 0005's replay gate and ADR 0016's
regression gate in place first. Stage 1 restructures `live_compute.cpp`'s dispatch function; ADR
0004's move should land before it so the work happens in the backend's permanent home.

## Alternatives considered

- Drop individual waits that look unobserved: rejected in #3948 -- every compute dispatch ends in a
  CPU writeback that needs its fence, so there is no class of wait that can simply go.
- A general task scheduler for all GPU work: more than the measured problem needs.

## Approval

Requires the project owner's acceptance. Stage 0 needs none beyond what #3948 already proposes.
