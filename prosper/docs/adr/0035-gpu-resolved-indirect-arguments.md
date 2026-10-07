---
kind: adr
status: proposed
date: 2026-10-07
---

# ADR 0035: Resolve indirect draws, dispatches and predicates on the GPU, without CPU readback

## Context

A PS5 title produces indirect arguments on the GPU: a compute pass writes a workgroup-count
triplet or a draw record, and a later packet in the same stream consumes it. prosper resolves most
of those arguments on the CPU today, which is either a wait for the producer (a `PERF-P1`
violation: an in-frame readback the guest does not observe) or a dropped operation.

What is true on `main` (d4d43d8ed) today:

- **Indirect compute dispatches have a device route (#3656).**
  `frontends/shared/live/indirect_dispatch.hpp` describes it: pin a retained GPU buffer that is
  authoritative for the 12 argument bytes, copy the triplet into a scratch record, run a
  one-invocation validation pass that zeroes a launch exceeding the device's per-axis limit, then
  `vkCmdDispatchIndirect`. When no authoritative buffer can be pinned, or the argument range aliases
  a resource of the dispatch itself, the backend counts a `host_fallbacks` and reads the triplet
  from guest memory (`live_compute.cpp:6948-6979`); an unreadable or misaligned address declines
  with `indirect-arguments-unreadable` (`live_compute.cpp:6970`).
- **Before the backend sees it, the executor refuses an indirect dispatch whose producer has not
  landed.** `gpu_executor.cpp:11741-11757` skips the dispatch, records
  `RealizationFailureReason::IndirectDependencies` and raises the always-on perf alarm
  `skipped-dispatches ... reasons=indirect-dependencies:N` (`perf_ledger.hpp:350`). The skip is
  fail-visible, which is correct, but the content it drops is real.
- **Indirect draws are resolved entirely on the CPU.** `resolve_indirect_draw_arguments`
  (`gpu_executor.cpp:10359-10444`) `memcpy`s the four-dword (`sceAgcDcbDrawIndirect`, #2929) or
  five-dword (`DRAW_INDEX_INDIRECT`) record from guest memory and turns it into a direct draw. It
  refuses `firstInstance != 0`, counts above `1 << 20`, and unreadable addresses, each with a
  rate-limited `[agc] ... indirect draw skipped` line. Reading guest memory is correct only when
  the producer is CPU-visible at that moment; a GPU-produced record is read stale or forces a wait.
  `IT_DRAW_INDEX_INDIRECT_MULTI` (`pm4_registers.hpp:37`) has no count-buffer route.
- **Predication is decided on the CPU at parse time.** `cond_indirect_buffer.cpp` reads the
  predicate from guest memory to choose a conditional branch, and refuses reserved modes (#4540);
  `SET_PREDICATION` polarity is packed by `pack_set_predication_control`
  (`cond_indirect_buffer.hpp:41`). A predicate written by a GPU occlusion query or compute pass is
  therefore the CPU's view of it, not the GPU's.

Observed evidence. In live Windows runs of *Assassin's Creed Black Flag Resynced* on 2026-10-07,
the `[perf-alarm]` lines reported `skipped-dispatches ... reasons=indirect-dependencies:N`. Those
counts were seen in that session and not recorded with a run manifest; the first migration step
below re-measures them (ADR 0023).

The guest argument layouts are already close to Vulkan's: the five-dword indexed record is
`VkDrawIndexedIndirectCommand` field-for-field and the four-dword record is `VkDrawIndirectCommand`.
What differs is semantics around them -- prosper's per-draw vertex-offset override, its
`index_base`, its refusal of `firstInstance`, and its guest-address-to-buffer mapping -- not the
byte layout.

Reference designs, structural only (CLAUDE.md evidence hierarchy, item 4):

- shadPS4 (`afde63182a38`) records `drawIndexedIndirectCount` / `drawIndirectCount` and
  `dispatchIndirect` straight from the guest's buffer
  (`src/video_core/renderer_vulkan/vk_rasterizer.cpp:237-381`), with no CPU read of the arguments.
- KytyPS5 (`038d2257de85`) references indirect only in its recompiler; no backend route was found.
- vkd3d-proton implements D3D12 `ExecuteIndirect` on `VK_EXT_device_generated_commands` (and
  earlier on the NV extension), from knowledge; not re-read for this ADR.

## Decision

**An indirect argument the guest produces on the GPU stays on the GPU.** The CPU reads it only
when the guest itself observes that value through a CPU-visible path, and a case no route below
covers is refused fail-visibly, never guessed.

1. **Core indirect commands first.** Indirect dispatches keep the #3656 route; indirect draws gain
   the same shape -- pin the authoritative buffer for the record, validate on the device, and
   record `vkCmdDrawIndirect` / `vkCmdDrawIndexedIndirect`. `DRAW_INDEX_INDIRECT_MULTI` with a
   count maps to `vkCmdDraw*IndirectCount` (Vulkan 1.2 core).
2. **Argument translation is a compute pre-pass, not a CPU read.** Where a guest record differs
   from the host one (prosper's vertex-offset override, `firstInstance` handling, clamping to device
   limits, a stride the host cannot express), a small compute pass rewrites the guest record into a
   host record in device memory, as #3656's validation pass already does for dispatches. The
   pre-pass also enforces every limit the CPU path enforces today, zeroing an out-of-range launch
   and counting it.
3. **Indirect dependencies are ordering, not refusal.** A producer that has not landed for this
   submit becomes a recorded pipeline barrier (`SHADER_WRITE` -> `INDIRECT_COMMAND_READ`) on the
   producer's buffer. `indirect-dependencies` remains a skip reason only where the producer is
   genuinely outside anything prosper has recorded.
4. **Predication becomes `VK_EXT_conditional_rendering`** where the guest predicate is a
   GPU-written 32-bit value over a span of draws and dispatches. Conditional *branches* over
   command streams (`COND_INDIRECT_BUFFER`) stay CPU-decided until a measured title needs more;
   where the predicate is GPU-produced and the branch matters, that is the case for item 5.
5. **`VK_EXT_device_generated_commands` only where the guest's command stream is itself
   GPU-generated** (a GPU-written PM4 or AGC command buffer, or state changes inside a
   multi-draw that core indirect cannot express). It is an optional host capability, never a
   requirement: without it, the case is refused with a named reason.
6. **Every route is observable without a log**, as `IndirectDispatchBackendStats` is today:
   device-resolved, host-fallback and rejected counts per route, and the existing perf-alarm skip
   reasons keep their names.

No spec rule is added. This ADR is how prosper meets the existing `PERF-P1` for indirect work;
a rule may follow once the draw route exists and an instrument can count its CPU reads.

## Consequences

- `resolve_indirect_draw_arguments` stops being the only path; it remains as the fallback for a
  record that is CPU-produced and CPU-visible, where reading it is correct and free.
- The resource layer must answer "which buffer is authoritative for these guest bytes, now" for
  draw records, as it does for dispatch triplets. That is ADR 0010's canonical resource identity;
  this ADR depends on it rather than adding a second answer.
- A GPU-resolved draw count is invisible to the CPU, so draw-count censuses and the retained-draw
  selection (`retained_draw_selected`) must not assume a known count. Diagnostics that need the
  value read it back off the frame, explicitly, under a diagnostic switch.
- Validation moves from CPU `if`s to shader code, and needs Vulkan-execution tests: a guest record
  produced by a compute pass in the same submit, an out-of-range count zeroed, `firstInstance != 0`,
  and a zero-count no-op. Each must fail with the device route removed.
- Conditional rendering and DGC add optional-feature checks to device setup; their absence must be
  a named refusal, not a silent CPU read.

## Alternatives considered

- **CPU readback with pipelined latency** (read the argument one or more frames late, or wait only
  for the producer). It still reads a value the guest does not observe, so it violates `PERF-P1`;
  it either waits in-frame or renders with last frame's counts, which is wrong content for any
  culling or streaming pass whose output changes per frame. Rejected.
- **Keep refusing** (`indirect-dependencies`, `[agc] indirect draw skipped`). Honest, but it drops
  GPU-driven culling and particle passes wholesale; the Black Flag alarms are exactly that loss.
  Kept only as the last resort in item 5.
- **DGC for everything.** Simpler conceptually, but DGC support is narrower than core indirect and
  heavier per sequence; core commands cover the measured cases.

## Migration order

By measured skip reason, largest first:

1. Re-measure: run Black Flag and two guarded titles with a manifest, and record the
   `indirect-dependencies`, `indirect-arguments-unreadable`, `host_fallbacks` and
   `indirect draw skipped` counts per frame, so the order below is evidence, not guess.
2. Indirect dependencies on dispatches: barrier instead of skip (item 3).
3. Indirect draws: device route with validation pre-pass (items 1-2).
4. `DRAW_INDEX_INDIRECT_MULTI` with a count buffer.
5. Predication via conditional rendering (item 4).
6. DGC, only if a title is measured issuing a GPU-generated command stream.

## Open questions

- Driver support for `VK_EXT_device_generated_commands` on RADV and NVIDIA's Windows driver is
  stated from knowledge (both were reported to expose it after its 2024 release) and is
  **unverified** here; check `vulkaninfo` on both development hosts before relying on it.
- Whether any title writes a PM4 command stream with the GPU; none is known.
- Whether the Black Flag skips are producers prosper never recorded (no fix here) or producers in
  the same submit (fixed by item 3). Step 1 decides.
- How `firstInstance != 0` should behave: the current refusal predates any evidence either way.

## Approval

The project owner accepts or rejects this ADR. Acceptance unblocks the draw-side device route and
makes the CPU read of a GPU-produced indirect argument a review finding under `PERF-P1`.
