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
violation recorded in `docs/spec/performance.md`. #3948 is a staged design for removing it.
Where it stands as of 2026-10-05: Stage 0 (the always-on sync-wait cost and the `gpu-sync-wait`
alarm) has landed; **Stage 2 merged as #3964 on 2026-09-29**, behind `PROSPER_GRAPHICS_DEFERRED_WAIT=1`
(default off), and its OFF/ON validation is still being posted on #3948; Stage 1 is not implemented.

The order changed on the way. #3948's "Stage 1 design refinement" comment measured Stage 1's ceiling
on *Grand Theft Auto V* gameplay: at most about 48% of dispatches can leave their completion pending,
bounding the saving at about 5% of wall time (probably under 3%), for a restructuring of the
~7,440-line `execute_item`. The graphics batch wait in front of every dispatch was the larger lever
(654 ms against 250 ms), so Stage 2 went first. This ADR adopts that order.

The comparable established design is DXVK's command-stream thread: API calls are recorded into
chunks that a worker consumes, so the application thread never waits for the GPU unless it reads a
result. It is a structural reference only.

## Decision

Adopt #3948's staged design as the architecture for GPU submission, with its observation-point rule
as the invariant: a guest-visible effect -- a completion label write, an EOP event, a flip, a guest
memory writeback, a write-watch invalidation -- is applied only after all GPU work and writebacks
that precede it in the command stream, and in stream order. Everything else the executor waits for
today is internal and may be deferred.

1. Stage 0 (landed): the always-on sync-wait cost and the `gpu-sync-wait` alarm are the stage-gate
   metric for everything after it.
2. Stage 2 (landed behind a switch, #3964): graphics batch waits deferred to their next CPU
   consumer. Its switch becomes the default, or is deleted, when #3948's validation settles it.
3. Stage 1, only after Stage 2 and only with write-after-write overlap measured too: compute
   dispatches pipelined within a submit, completion records retired at the first observation point
   (RAW/WAW overlap, a cache lookup, a graphics import, a pend-queue write, submit end), in submission
   order. It starts with a behaviour-free refactor that moves `execute_item`'s post-wait tail into
   its own function over an explicit state struct, as #3948 recommends.
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
regression gate in place first. Stage 1 restructures `live_compute.cpp`'s dispatch function for a
small measured ceiling, which is why it is ordered after Stage 2 and gated on a fresh measurement.

## Alternatives considered

- Drop individual waits that look unobserved: rejected in #3948 -- every compute dispatch ends in a
  CPU writeback that needs its fence, so there is no class of wait that can simply go.
- A general task scheduler for all GPU work: more than the measured problem needs.

## Approval

Requires the project owner's acceptance. Stage 0 needs none beyond what #3948 already proposes.
