---
kind: adr
status: proposed
date: 2026-10-07
---

# ADR 0034: Overlap guest compute with graphics on a second Vulkan queue (async compute)

## Context

The PS5 GPU runs graphics (DCB) and asynchronous compute (ACB) streams on separate hardware queues,
and titles build their frames around that overlap. prosper executes all of it serially on one host
queue, and most of each submit is spent waiting on it.

What is true today (`origin/main` at `d4d43d8ed`):

- **ACB submits fold into the graphics queue model.** `sceAgcDriverSubmitAcb`
  (`src/hle/graphics/hle_agc.cpp:3068`) routes async-compute streams through "the same serialized
  queue model" as DCB submits; its comment (`:3064-3067`) keeps the named wrapper only so "independent
  hardware queue state" can be added later. The library exports 35 `sceAgcAcb*` entry points
  (`hle_agc.cpp:3567`), and ACB variants such as `sceAgcAcbDmaData` (`:957`) and
  `sceAgcAcbDispatchIndirect` (`:3564`) build the same packets as their DCB siblings.
- **The compute backend borrows the renderer's queue.** `live_compute.cpp:3266-3281` adopts the
  shared Vulkan context, including `shared.queue` and `shared.queue_family`, whenever that family
  supports compute. Only a headless compute-only process creates a private device (`:3414-3426`). So
  compute and graphics share one `VkQueue`, and the hardware cannot overlap them.
- **Each compute batch is submitted and waited inline.** `live_compute.cpp:13010-13015` submits with a
  fence and `:13033` waits on it before returning to the command processor. This is the `PERF-P1`
  violation that ADR 0009 Stage 1 addresses. Whatever ADR 0009 removes, one queue still means the
  GPU executes compute and graphics back to back.
- **The guest already says which work depends on which.** Cross-queue ordering on PS5 comes from the
  guest's own packets: end-of-pipe `ReleaseMem` label writes, `WaitRegMem`/`WAIT_MEM_64` polls on
  those labels, and ordered DMA. The command processor already decodes and orders these
  (`hle_agc.cpp:3064-3066`), so the dependency graph is available without a heuristic.

**Measurement.** A 2026-10-07 *Assassin's Creed IV: Black Flag* run on Windows with an RTX 4070
SUPER, using `prosper-app` in `--present-mode immediate`, reported:

```text
[perf-alarm] rule=gpu-sync-wait value=121.14 %budget ... wait=1171ms (compute 1124ms/377, graphics 47ms/145) ... gpu-busy=21%
```

So 96% of the sync wait (1,124 of 1,171 ms) is spent on compute, over 377 waits, while the GPU is
busy only 21% of the time. The executor serialises compute behind a GPU that is mostly idle. This is
one run of one title: it shows where the time goes, not what overlap would win. Only the A/B in the
Migration order can measure that.

**Reference designs.** None of these were copied (CLAUDE.md, evidence hierarchy). They show the
design space only.

- KytyPS5 (`038d2257de85`) runs a separate guest command processor for each ACB queue:
  `src/graphics/guest_gpu/graphicsRun.h:51-53,96` (`ComputeQueueBase = 0x20`, an array of compute
  `CommandProcessor`s), and `graphicsRun.cpp:153-193` gives each submission its own `queue_id`.
  Whether these reach separate *Vulkan* queues was not checked.
- shadPS4 (`afde63182a38`): searching `src/` for a compute queue family or a dedicated compute queue
  found only renderer-internal compute pipelines (`video_core/texture_cache/tile_manager.cpp`,
  `buffer_cache/fault_manager.cpp`). No guest-ACB-to-host-queue mapping was found. Unverified beyond
  that grep.
- vkd3d-proton maps D3D12 compute command queues to Vulkan compute queues, and DXVK submits on one
  queue with presentation done asynchronously. Both are from memory and unverified here.

## Decision

1. **Prerequisite:** this work starts only after ADR 0009 Stage 1 (compute waits deferred to their
   observation point) has landed and shipped by default. Stage 1 alone removes the inline wait.
   This ADR changes *which queue* the deferred work runs on. Doing both at once would make the A/B
   unattributable.
2. **A dedicated host compute queue.** When the physical device exposes a queue family with
   `VK_QUEUE_COMPUTE_BIT` and without `VK_QUEUE_GRAPHICS_BIT`, the shared device requests one queue
   from it at creation. Recent AMD (RADV) and NVIDIA drivers commonly expose such a family; this was
   not verified on the project's machines. If there is no such family, prosper keeps today's
   single-queue path, unchanged. The second queue is a host capability, never a requirement.
3. **What goes on it:** ACB streams (`sceAgcDriverSubmitAcb`), one host timeline per guest ACB queue,
   multiplexed onto the host compute queue. Compute inside a DCB stays on the graphics queue, unless
   the guest marks it independent with its own packets (a dispatch bracketed by a label release that
   the graphics stream does not wait on before its next dependent read). That case is a later step
   with its own evidence (step 5). prosper never decides independence by heuristic.
4. **Dependencies come only from the guest's own sync packets.** A `ReleaseMem`/EOP label write on
   one queue signals a value on that queue's timeline semaphore (Vulkan 1.2 core). A `WaitRegMem` on
   another queue that polls that label becomes a timeline wait on the matching value, where prosper
   can resolve the label to a pending signal. Otherwise it stays a CPU-side wait at the point ADR 0009
   already defines. Ordered DMA and guest-memory writebacks keep their current ordering.
5. **`PERF-P8` is unchanged and binds both queues.** A completion label, EOP event, flip, writeback or
   write-watch invalidation becomes guest-visible only after all work that precedes it *in its own
   guest stream* has retired. Work on different guest queues may complete in any order the guest's
   own sync allows, which is the PS5 contract. It may never be visible in an order the guest's sync
   forbids. The completion thread from ADR 0009 Stage 3 applies effects in per-stream order.
6. **Resources use `VK_SHARING_MODE_CONCURRENT`** across the graphics and compute families for
   buffers and images that both queues can reach. That avoids queue-family ownership transfers, whose
   release/acquire pairs would have to be inferred from guest packets that do not express them.
   Concurrent sharing can cost compression on some hardware (notably AMD DCC). That is one measured
   question for step 3, not a reason to model ownership.
7. **Behind a switch.** `PROSPER_ASYNC_COMPUTE_QUEUE` is a *guest-behaviour selector*: it changes when
   guest-visible completions can happen relative to each other. Default off. Its issue settles the
   default and deletes the switch. A verification mode submits to the second queue but waits on each
   signal inline, so ordering bugs can be separated from overlap wins.

No spec rule is added. `PERF-P8` already states the invariant, and `PERF-P6` (recording overlaps
execution) covers the goal. A rule like "independent guest queues execute concurrently" would be a
target this ADR cannot yet enforce.

## Consequences

- Easier: the executor stops serialising ACB work behind graphics when the GPU is idle, which is the
  shape the Black Flag run shows. The per-queue timelines also give `gpu_timeline` an honest per-queue
  view.
- Harder: two submission threads of work to validate. Ordering bugs become driver- and
  timing-dependent, and the Vulkan validation layers' synchronization validation becomes a required
  check for every change in this area. `gpu_replay` must record which guest queue each submit came
  from and replay with both a single queue and two queues. A replay that cannot show the two arms
  agree is not evidence (ADR 0005).
- The `gpu-sync-wait` alarm's compute/graphics split and `gpu-busy` are the instruments. A change that
  lowers compute wait but leaves `gpu-busy` flat has moved the wait, not removed it.
- On hosts without a dedicated compute family nothing changes, so the CI lavapipe runner exercises
  only the fallback. The two-queue path is local-only evidence, which the PR must say.

## Alternatives considered

- **Keep one queue (today).** It is simplest, and every ordering is trivially correct. It is rejected
  as the end state because the measured wait is compute-dominated while `gpu-busy` is 21%. It remains
  the fallback, and ADR 0009 Stage 1 may recover enough on its own. If Stage 1's A/B leaves compute
  wait small, this ADR should be closed rather than built.
- **Choose the queue for each dispatch by heuristic** (for example, put a dispatch on the compute
  queue if its bindings do not overlap the in-flight graphics resources). Rejected: it reorders
  guest-visible effects on an inference instead of on the guest's own sync. A wrong guess is a silent
  `PERF-P8` violation, and alias analysis over guest addresses is exactly the
  resource-identity problem ADR 0010 has not yet solved.
- **One Vulkan queue per guest ACB queue.** Hardware exposes few compute queues (often 1-8), and the
  guest can create more. A host timeline per guest queue on one host queue gives the same ordering
  with less driver variance.
- **Queue-family ownership transfers instead of concurrent sharing.** Rejected for now (Decision 6):
  the guest's packets do not say where a release/acquire pair belongs.

## Migration order

1. ADR 0009 Stage 1 lands and is default-on. Re-measure Black Flag and the 0009 reference workloads.
   Proceed only if compute still dominates `gpu-sync-wait`.
2. Census: count ACB vs DCB-embedded dispatches and the labels linking them, per reference title. If
   the guest uses few ACB submits, step 3's ceiling is small, and that is recorded.
3. Device: request the dedicated compute queue, switch resources to concurrent sharing, add the
   per-queue timeline semaphores. Same-binary A/B with the second queue unused, to price concurrent
   sharing alone.
4. Route ACB submits to the compute queue behind `PROSPER_ASYNC_COMPUTE_QUEUE`. Same-binary A/B on
   the ADR 0009 workloads plus Black Flag, both on Windows/NVIDIA and Linux/RADV, quoting `gpu-busy`,
   the compute/graphics wait split and `distinct` fps (`--present-mode immediate`). Add a `gpu_replay`
   two-arm check and a test that a label-ordered ACB/DCB pair observes the guest's order.
5. Only with evidence from step 4: guest-marked independent compute inside DCB streams.

## Open questions

- Do the titles behind the measured compute wait submit through ACB at all, or is it DCB-embedded
  compute? Step 2 answers this. If it is DCB-embedded compute, the gain depends on step 5.
- What is the cost of concurrent sharing on RADV (DCC) and NVIDIA for render targets that compute
  reads?
- Can a `WaitRegMem` always be resolved to a pending signal, or do guests poll labels written by the
  CPU or by DMA, which must stay CPU-side waits?
- How do the renderer's own compute passes (detiling, format conversion) choose a queue?

## Approval

The project owner accepts this ADR. Acceptance authorises steps 2-4 after ADR 0009 Stage 1. It does
not authorise step 5, which needs its own evidence and review.
