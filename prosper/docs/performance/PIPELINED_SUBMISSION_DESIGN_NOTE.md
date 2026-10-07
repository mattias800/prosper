---
kind: plan
status: current
---

# Pipelined submission: design note for ADR 0009 Stage 1 (compute)

Status: proposal for the project owner. Nothing here changes behaviour; the only code is a pure model
and its tests. Written 2026-10-06 from *Assassin's Creed Black Flag Resynced* (`PPSA28183`) measurements.
ADR 0009 is `status: proposed`; this note adopts nothing from it.

## Why this note exists

ADR 0009 adopts #3948's staged design and orders Stage 1 (pipelined compute) after Stage 2, gated on a
fresh measurement. The measurement below comes from a title whose frame is mostly compute. It supports
one thing the ADR did not say: for this title compute, not graphics, is where the CPU time goes.

## Evidence (Windows, RTX 4070 SUPER, `prosper-app`, GPU present, default route)

One run per row, from a build with #4635 (open, under review) applied. The breakdown in
`AC_BLACK_FLAG_STATUS.md` (#4632) is from the PR head of #4586 without #4635, which is why the two differ.

- A submit carries about 13 compute dispatches; each is set up, submitted, `vkWaitForFences`d and written
  back before the next starts. Compute is about 133 ms of CPU per submit: setup 48 ms, writeback 42 ms
  (of which image writeback 36 ms), dispatch wait 42 ms.
- The `gpu-sync-wait` alarm reads 75-170 % of the 33 ms budget every window. In a 5 s window 640-780 ms
  of 700-960 ms of waiting is compute (about 3 ms per dispatch), `gpu-busy` 16-18 %, `idle-in-wait` 11-13 %:
  the wait is mostly latency and serial CPU work, not a saturated GPU.
- Cheap local wins were tried first in #4635 (shared constant-ring arena, block-granular refresh). What
  remains is per-dispatch serial work that only overlap can hide.
- **Stage 2 (`PROSPER_GRAPHICS_DEFERRED_WAIT=1`): no failure observed.** An earlier version of this note
  claimed Stage 2 breaks the title. That was wrong. The log line it relied on, `WaitRegMem ... NOT
  satisfied at fold time ... dependency violated`, is the default fold's message (printed when
  `PROSPER_WAIT_DEFER`, a different switch, is unset); it appears without Stage 2, 40 lines per 45 s run.
  In 3 runs per arm (default, and with the Stage 2 switch set; 45 s, same binary) the arms behaved the
  same: the same line, no `pausing queue` lines, similar alarm and flip counts. So these runs neither
  show a failure nor show that deferral was exercised. One earlier ON run ended after about 10 s and was
  not reproduced. Whether Stage 2 is safe on this title is open; the observation is posted on #3948.

## The contract

`frontends/shared/compute/retirement_queue.hpp` is the rule as code (PERF-P8). It is a model: no production
code uses it yet.

1. An operation records the guest ranges it reads and writes.
2. A new operation is admitted without waiting only if it has no read-after-write, write-after-write or
   write-after-read overlap with a pending one. On a conflict, every pending operation up to the *last*
   conflicting one retires first, in order.
3. A guest-visible effect with no range (label write, EOP, flip, submit end) retires everything.
4. A guest-memory observation retires up to the last pending writer of its range (and pending readers, if
   it is a store).
5. Retirement is strictly in submission order, and `retire()` refuses anything but the current prefix.
6. Depth is bounded (PERF-P6).

`tests/shared/compute/test_retirement_queue.cpp` has a hand-written case per rule and a model check. The
model runs 400 random streams sequentially (today's behaviour) and pipelined at depths 1, 2, 4 and 16,
where a dispatch reads guest memory when the "GPU" executes it, at a random point between admission and
retirement, and applies its writes at retirement. It requires every dispatch result, every CPU read and
every effect's view of memory to be identical. Five negative tests switch one rule off *in the real queue*
(`RetirementPolicy`: conflict retirement, observation retirement, reader wait on a store, effect
retirement, the depth bound) and require the same check to diverge.

What the model cannot show: write-after-write between dispatches (in-order retirement already applies
writes in order, so that clause is redundant here and rests on its hand-written case), and anything
about the backend, since the backend does not use the queue yet.

## Proposed integration (each step ships behind a default-off switch, with an inline-retire verify mode)

1. **Behaviour-free refactor** (#4650): split `execute_item`'s post-wait tail into a function over an
   explicit state struct (as #3948 recommends). No new switch; existing tests prove it.
2. **Record ranges**: have each dispatch report its guest read and write ranges (it already resolves them
   for the buffer and image caches). Feed them to a `RetirementQueue` that is *only observed*: log the ids
   it would have required to retire at each effect, and assert in the verify mode that waiting inline gives
   the same guest bytes. This finds missing observation points without changing timing.
3. **Defer the fence wait and writeback** for operations the queue says may stay pending, retiring at the
   queue's observation points. Verify mode retires inline and compares guest bytes per submit (ADR 0009
   Decision point 5 already requires that mode for every stage).
4. If Stage 2 is adopted, apply the same rule to the `WaitRegMem` fold (a fold is an observation of its
   label address). That code is in `src/gpu/pm4/` and Stage 2's deferral in `src/gpu/execute/`, neither of
   which may include `frontends/` (LAY-2), so step 4 reuses the rule, not this type; a shared type would
   have to live under `src/gpu/execute/`.

## Open questions for the owner

- Which guest-visible effects exist besides labels, EOP, flips and writebacks? The contract lists the ones
  found; a missed one would break a title.
- Reference workloads: only Black Flag is available on this machine. The `perf-change` skill asks for GTA V,
  Outer Wilds and *The Messenger* too; step 3 should not land without them.

## Where the figures come from

One Windows run per row; the logs are on the author's machine and the summary is posted on #4631. Treat
them as indicative, not as a measurement protocol.

## Step 2 result: what pipelining would have allowed (observe-only, 2026-10-06)

`PROSPER_PIPELINE_OBSERVE=1` (diagnostic; changes nothing) feeds every completed dispatch's guest read and
write ranges, and the graphics spans and DMA copies the executor runs between dispatches, to the
`RetirementQueue` contract, and reports four models: ranges-only (optimistic: a graphics span is assumed not
to read a pending result) and barriers (pessimistic: every graphics span and DMA is a retirement point), each
at depth 4 and depth 64. "Overlapped" means the dispatch would have been admitted while an older one was still
pending; "wait-overlapped" is the share of today's measured dispatch time (record + submit + fence wait)
belonging to such dispatches. Read it with `prosper-app` closed through its window (a `timeout` kill loses the
exit report; a periodic report prints every 256 dispatches).

*Assassin's Creed Black Flag Resynced*, Windows/NVIDIA, `prosper-app`, default route, 1,792 dispatches in about
45 s, two runs with identical counts, 655 graphics spans/DMA copies seen between them:

| model | overlapped | wait-overlapped | max pending |
| --- | --- | --- | --- |
| ranges-only, depth 4 | 69.6% | 76.8% | 4 |
| ranges-only, depth 64 | 69.6% | 76.8% | 22 |
| barriers, depth 4 | 47.5% | 50.0% | 4 |
| barriers, depth 64 | 47.5% | 50.0% | 22 |

What it says, and what it does not:
- About half to three quarters of dispatch time belongs to dispatches that could run concurrently with an
  older one, so overlap exists. Most dispatches still conflict with an earlier one (conflict retirements are
  about 0.25 per dispatch with barriers, 0.9 without), so the depth that pays is shallow: depth 4 loses
  nothing against depth 64 in "overlapped".
- Dispatch time is only part of a submit's compute CPU time (about 20 ms of about 113 ms in the same title;
  setup and writeback are the larger parts). Hiding half to three quarters of it is about 10-15 ms per submit.
  **Overlapping the CPU setup and writeback of the next dispatch with the GPU running the previous one is not
  measured here**, and is where the larger gain would have to come from; this step cannot see it.
- Ranges come from the bindings' guest addresses; GPU-internal hazards (image-cache aliasing, renderer-owned
  targets reached through the live-target bridge) are not modelled, and the barrier model assumes every
  graphics span reads everything. The truth lies between the two rows.
- One title, one machine, one route. The compute-only submits Black Flag uses on its own mean the barrier
  model's graphics spans are rare there; a title that interleaves draws and dispatches will sit near the
  barrier row.

## What this note does not claim

No speed-up is measured here. The ceiling for Stage 1 on GTA V was bounded at about 5 % of wall time in
#3948; Black Flag's compute-heavy frame may be higher, but that is a hypothesis until step 2's observation
mode reports how many dispatches could actually have stayed pending.
