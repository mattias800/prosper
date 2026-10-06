# Pipelined submission: design note for ADR 0009 Stage 1 (compute)

Status: proposal for the project owner. Nothing here changes behaviour; the only code is a pure model
and its tests. Written 2026-10-06 from *Assassin's Creed Black Flag Resynced* (`PPSA28183`) measurements.

## Why this note exists

ADR 0009 adopts #3948's staged design and orders Stage 1 (pipelined compute) after Stage 2, gated on a
fresh measurement. The measurement below comes from a title whose frame is mostly compute, and it
changes two things the ADR assumed: Stage 2 as merged is **not safe** on this title, and compute is
where the time is.

## Evidence (Windows, RTX 4070 SUPER, `prosper-app`, GPU present, default route, one run per row)

- A submit carries about 13 compute dispatches; each is set up, submitted, `vkWaitForFences`d and written
  back before the next starts. Compute is about 133 ms of CPU per submit (setup 48, writeback 42,
  dispatch wait 42, of which image writeback 36).
- The `gpu-sync-wait` alarm reads 75-170 % of the 33 ms budget every window. In a 5 s window 640-780 ms
  of 700-960 ms of waiting is compute (about 3 ms per dispatch), `gpu-busy` 16-18 %, `idle-in-wait` 11-13 %:
  the wait is mostly latency and serial CPU work, not a saturated GPU.
- Cheap local wins were taken first (#4635: shared constant-ring arena, block-granular refresh).
  What remains is per-dispatch serial work that only overlap can hide.
- **Stage 2 (`PROSPER_GRAPHICS_DEFERRED_WAIT=1`) breaks this title.** With it ON the run logs
  `WaitRegMem #n NOT satisfied at fold time ... dependency violated` and ends within about 10 s; OFF it
  runs normally. That is a violation of the ADR's own observation-point rule: a label the guest waits on
  was observed before the work that writes it retired. It is the first concrete failure of the invariant,
  and the reason the rule below is stated as an executable contract before any more waiting is removed.

## The contract

`frontends/shared/compute/retirement_queue.hpp` is the rule as code (PERF-P8):

1. An operation records the guest ranges it reads and writes.
2. A new operation is admitted without waiting only if it has no read-after-write, write-after-write or
   write-after-read overlap with a pending one. On a conflict, every pending operation up to the *last*
   conflicting one retires first, in order.
3. A guest-visible effect with no range (label write, EOP, flip, submit end) retires everything.
4. A guest-memory observation retires up to the last pending writer of its range (and pending readers, if
   it is a store).
5. Retirement is strictly in submission order, and `retire()` refuses anything but the current prefix.
6. Depth is bounded (PERF-P6).

`tests/shared/compute/test_retirement_queue.cpp` proves it two ways: a case per rule, and a model check
that runs 400 random operation streams sequentially (today's behaviour) and pipelined at depths 1, 2, 4 and
16, and requires every value an operation reads and every CPU observation to be identical. A control run
that skips conflict retirement must diverge, so the model check cannot pass vacuously. Disabling conflict
retirement in the queue fails four tests.

## Proposed integration (each step ships behind a default-off switch, with an inline-retire verify mode)

1. **Behaviour-free refactor**: split `execute_item`'s post-wait tail into a function over an explicit state
   struct (as #3948 recommends). No new switch; existing tests prove it.
2. **Record ranges**: have each dispatch report its guest read and write ranges (it already resolves them
   for the buffer and image caches). Feed them to a `RetirementQueue` that is *only observed*: log the ids
   it would have required to retire at each effect, and assert in the verify mode that waiting inline gives
   the same guest bytes. This finds missing observation points without changing timing.
3. **Defer the fence wait and writeback** for operations the queue says may stay pending, retiring at the
   queue's observation points. Verify mode retires inline and compares guest bytes per submit.
4. Fix Stage 2 against the same contract before its switch becomes default: a `WaitRegMem` fold is an
   observation of its label address.

## Open questions for the owner

- Which guest-visible effects exist besides labels, EOP, flips and writebacks? The contract lists the ones
  found; a missed one is exactly what Stage 2 hit.
- Is a verification mode that retires inline and byte-compares acceptable as the gate (ADR 0005 and 0016
  mention the replay and regression gates but not this)?
- Reference workloads: only Black Flag is available on this machine. The `perf-change` skill asks for GTA V,
  Outer Wilds and *The Messenger* too; step 3 should not land without them.

## What this note does not claim

No speed-up is measured here. The ceiling for Stage 1 on GTA V was bounded at about 5 % of wall time in
#3948; Black Flag's compute-heavy frame may be higher, but that is a hypothesis until step 2's observation
mode reports how many dispatches could actually have stayed pending.
