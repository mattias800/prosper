# Grand Theft Auto V (`PPSA04263`, RAGE) — status

**Rung 3** on the bring-up ladder: routed gameplay entry with real GPU draws. The HUD, radar and
tutorial text render; the 3D world does not.

Tracker: **#1873**. Active frontier: **#2481**. Route: `scripts/gta5/reach-story-mode.pad`
(read its header — the flip timing is measured, not estimated, and the tab navigation needs four R1
presses for a reason).

Historical design note for the descriptor work: `docs/FLAT_LOAD_DESIGN.md`. Do not start from it; the
descriptor-array lift it describes is complete.

## Where the world went

As of 2026-08-14 the black world is **one compute program**, and that is established by A/B rather
than inferred:

```
PROSPER_COMPUTE_SKIP_PROGRAM=0x413dc6700
```

| arm | device losses | frame |
| --- | --- | --- |
| default | 1, at ~59 s — reproduced in every run | black + HUD |
| `0x413dc6700` skipped | **0** across a full 300 s route | sun, anamorphic lens flare, radar with street geometry and blips, first-person tutorial text |

`0x413dc6700` hangs the GPU into a RADV **hard recovery** at `queue-submit`. prosper then disables
live compute for the whole process, and every later dispatch and indirect draw in the frame is
refused — so the frame cannot recover even partially.

**Read that skipped-program frame carefully: it is not "the frame minus that dispatch".** A skip goes
through the same decline path as a real refusal, so it clears `producer_epoch_ok` and the next
`ParserStall` latches `indirect_dependencies_ok` for the rest of the submit. The frame is that frame
minus the dispatch *minus every indirect draw and dispatch after the next parser stall*. The zero
device losses and the appearance of real scene content are unaffected by this; what it costs is the
right to attribute the *remaining* blackness in that image to `0x413dc6700` alone.

What the program probably is, from what survives without it: 43 resources,
`threads=2063x1x1 local=256x1x1 groups=9x1x1`, 61,143 SPIR-V dwords, and sky/flare/radar render while
world geometry does not — the shape of a tiled deferred lighting or light-culling pass.

**The frontier is therefore exactly one sentence: `0x413dc6700` must execute without hanging.**

### Why this took so long to see

Until #2538 the live compute backend had **fifteen** `return false` paths in `execute_item()` that were
silent, `trace`-gated, or logged without naming the dispatch. The executor records such a refusal as
`RealizationFailureReason::Unknown`, and **only when a capture trace is active** — so on a default run
a refused dispatch left no record at all. (`ComputeExecutionDeclined`, which an earlier revision of
this file named, is a classification added later by #2536; it did not exist when the frame went
black.) A gameplay submit's 196 realization failures were 59 anonymous declines and 128 cascade
failures, and the cascade hid the cause:

1. a declined dispatch clears `producer_epoch_ok`
2. the next `ParserStall` latches `indirect_dependencies_ok` false for the **rest of the submit**
   (`gpu_executor.cpp:9609`)
3. every later indirect draw and indirect dispatch short-circuits to `IndirectDependencies` without
   being attempted (`:9354`, `:9422`)

GTA V's world is GPU-driven, so the indirect operations *are* the world. One refusal anywhere early in
a submit removes all of it. Several sessions went to the 72 **direct** draws that execute correctly at
3840x2160 with colour writes enabled — they were never the ones failing.

The first domino turned out to be an **empty guest kernel**: `0x413cea300`, whose entire body is one
`s_endpgm`, declared one raw buffer and was refused as `no-bindable-descriptor` 192+ times per run.
Fixed by proving emptiness from the raw stream (`rdna2_program_is_terminator_only`) and reporting such
a dispatch as a successful no-op. A/B: 192 declines to 0.

## Ruled out

One line per falsified hypothesis, the evidence that killed it, and where. **Read this before forming
a new one** — and note which entries are *solid* versus *void*, because a void result is not a
falsification.

- **The 72 direct draws are failing / are culled / have colour writes masked.** They execute at full
  3840x2160 with `effective=3f` and zero realization failures in the window that presents black. The
  world is drawn by *indirect* operations, which the latch above dropped untried. #2481.
- **Collapsed AABBs culling the world.** The reduction at `0x413ced900` computes a correct bounding
  box, `(-13.71, -23.65, 0)` to `(14.98, 16.95, 4.101)`. #2481.
- **`CB_TARGET_MASK=0` masking colour writes (the #1946 shape).** The main-view pass reports
  `effective=3f`. The pass that showed `target-mask=0 effective=00` was a **512x512 depth-only shadow
  atlas**, where that is correct — selected by a `MIN_DRAWS` filter, not by phase. Instrument trap 159.
- **The sky renders correctly offline and is lost live.** The `--draw-steps` composite that showed a
  blue gradient with a horizon band is prosper's **seed-miss placeholder**: already 100% non-black at
  step 4, before most draws run, and unchanged through step 78. Instrument trap 161.
- **The whole-frame abort on an unsupported storage image.** `render_draw_pass_rgba`
  (`tests/render_runner.h:3309`) does abandon an entire 4K frame on the first invalid storage-image
  contract, which is a real disproportionality — but gating it so only the offending draw is dropped
  left the black window unchanged. Reverted rather than landed. #2481.
- **A dark frame containing a dim scene.** Frames measuring 0.00% non-black at threshold 8/255 while
  12.7-13.5% of pixels are non-zero at luminance 1-2, with a CRC that changes every second, are
  **animated dither** — amplified 64x there is no structure, edge or geometry. Instrument trap 160.
- **The recompiler is the frontier.** `[compute] skip unsupported program` named 15 programs; retried
  offline against their own captured resource tables, **16 of 20 recompile cleanly**. Only 9 of 196
  failures in the classified capsule are `shader-recompile`. Instrument trap 157.
- **Unresolved descriptors on the hanging program.** All **43** of `0x413dc6700`'s resources resolve
  with real addresses and sizes; none is null. (The zero-address resources in an earlier dump belong to
  `0x413dc3400`, the neighbouring dispatch.) #2481.
- **The hang is an out-of-bounds write from a mis-sized buffer bound.** `RADV_DEBUG=hang` produced a
  report whose **`vm_fault.log` is 0 bytes** — no page fault, no VM fault. This was the leading
  suspect given #2529/#2535's history with `scalar_buffer_dword_count`. #2481.
- **The hang is a non-uniform `OpControlBarrier` deadlock.** The CFG dispatcher's continue block runs
  two workgroup-scope barriers per iteration and its LDS reduction covers only `[base..base+63]`,
  which looks exactly like a per-wave exit under a workgroup-wide barrier at `local=256`/`wave=64`.
  It is not: the loop-exit predicate reads **slot 260**, the whole-workgroup liveness result, so every
  invocation iterates together. #2481.
- **The hang is a non-uniform early exit.** The 14,370-line disassembled module contains exactly
  **one** `OpReturn`, at the end. No `OpKill`, none inside any of the three dispatcher loops. #2481.
- **The hang is a cyclic traversal table (183 cyclic starts).** **RETRACTED — see the retraction
  section above.** The 183 cycles are in `0x20f848417c`, the table read by the dispatch that
  *completed*; the hanging dispatch reads `0x20f848a240`, which is acyclic with a longest chain of 11
  steps. The measurement was taken from the wrong dispatch's binding. #2542.
- **A lost atomic corrupts the traversal table.** The table that hangs `0x413dc6700` is a linked list
  whose corruption is 61 **two-cycles** (`i` and `i+2` pointing at each other), the classic signature
  of a non-atomic concurrent insertion — two threads each linking to the other because both read the
  head before either wrote. Its producer `0x413ce3400` performs **no atomic operation of any kind**,
  and the strong form of that is structural rather than grep-shaped: its entire memory footprint is
  **29 buffer loads and stores** (`buffer_load_dword` x15, `buffer_load_dwordx2` x9,
  `buffer_store_dword` x4, `buffer_store_dwordx3` x1) and **zero `ds_*` instructions**, so there is no
  LDS family in which an atomic could hide. Zero `OpAtomic*` in its 3,881-line recompiled SPIR-V
  agrees. **Positive control, a different program in a different capsule**: `0x413ced900` contains
  `buffer_atomic_fmax` x3, `buffer_atomic_fmin` x3, `ds_max_f32` x3 and `ds_min_f32` x3 — so the
  instrument fires on the buffer family *and* the LDS family, and the zero is a real negative.
  **Note what the control also demonstrates:** a `(buffer|global|flat|ds)_atomic_*` pattern silently
  misses LDS atomics entirely, because RDNA2 spells them `ds_max_f32`, not `ds_atomic_max` — the
  control's own `ds_min/max_f32` would not have been found by it. That gap is why the claim above
  rests on the absence of the whole DS family rather than on an atomic-shaped pattern. #2542.
- **The producer/consumer ordering violation is `WAIT_REG_MEM` being barreled through.** The guest
  issues waits prosper cannot satisfy (`[agc] WaitRegMem … dependency violated … LABEL-UNMAPPED`, 40
  per route), and by default an unsatisfied wait does not pause the queue — which would let a
  consumer run before its producer's results land, exactly the symptom. **Falsified** with the
  opt-in barrier model `PROSPER_WAIT_DEFER=1` (#312): the device is still lost, at the same program
  and the same dispatch index. **Lever verified before reading the result** (instrument trap 164):
  the run logs 40 `pausing queue (deferred effects)` and the baseline logs 0, against 28 and 40
  `dependency violated` respectively, so the model was genuinely active. Note the recorded #312
  verdict against defaulting this ON was measured entirely on *Dragon Quest VII*'s heap corruption
  and says nothing about GTA V — this is an independent falsification, not a re-derivation of it.
  #2542.
- **The hang is an unconditionally infinite loop.** The same program executes successfully at dispatch
  38 and hangs at dispatch 39 of the same submit. Whatever spins is data-dependent. #2481.
- **Bounding the CFG dispatcher's trip count stops the hang.** Tried at 4096 and at 2^20; the device
  was lost both times. Note this cuts *against* the loop being unbounded rather than for it. #2481.

### Void, not falsified — do not cite these as settled

- **Portable wave64 emulation cost is the hang's mechanism.** A run with
  `PROSPER_NATIVE_COMPUTE_MULTIWAVE=1` still hung, which was reported as a falsification and **is
  not one**: the emitted module is **byte-identical** with and without the switch
  (`d04fd09b13408f9b4da7287fae34f692` in both arms and in a capsule from hours earlier), so the two
  runs are the same run. `[subgroup] … native=64 … multiwave=1` reports the resolved *config*, not the
  emitted lowering. The hypothesis is neither confirmed nor refuted — though it is now less likely on
  other grounds, since `native_subgroup_size` was apparently already 64, which suggests those barriers
  are the guest's own `s_barrier`s rather than emulation scaffolding. Reopening it needs a lever
  verified by module hash **before** its result is read. #2481.

## CONFIRMED: the hang is a non-terminating CFG DISPATCHER loop, and more than one program has it

`PROSPER_CFG_TRIP_BOUND=N` (new, diagnostic) forces every dispatcher loop out after N iterations. With
`N=100000` the run gets **past** `0x413dc6700` and dies much later at a **different** program:

| run | device lost at |
| --- | --- |
| default | `0x413dc6700` submit 4535 **dispatch 39** order 5642 |
| + empty-kernel fix | `0x413dc6700` submit 5968 **dispatch 39** order 17056 |
| + no buffer cache | `0x413dc6700` submit 5963 **dispatch 39** order 9755 |
| + `PROSPER_WAIT_DEFER=1` | `0x413dc6700` submit 4643 **dispatch 39** order 1206 |
| **+ `PROSPER_CFG_TRIP_BOUND=100000`** | **`0x413e14900`** submit 5954 **dispatch 52** order **24374** |

**Isolated control — one program instrumented, only the constant differs.** `PROSPER_CFG_TRIP_BOUND`
is targetable with `PROSPER_CFG_TRIP_BOUND_PROGRAM=0xADDR`, which leaves every other recompiled module
byte-identical. Both arms below instrument `0x413dc6700` and nothing else — the run log carries a
single arm line naming that one program:

| bound | target | outcome |
| --- | --- | --- |
| **4,096** | only `0x413dc6700` | **gets past it**, dies later at `0x413e14900` |
| **4,000,000,000** | only `0x413dc6700` | **dies at `0x413dc6700`** |

Same instrumentation, same module perturbation, different constant. That removes the two alternatives
an untargeted bound leaves open: it cannot be an *earlier* truncated shader feeding different data
downstream (no other shader is touched), and it cannot be SPIR-V perturbation changing RADV code
generation (both arms carry the identical counter). **`0x413dc6700`'s loop genuinely exceeds 4,096
iterations.**

What is still NOT established is *which* of its three emitted phases spins, and there is no
device-side witness that a cap was actually reached — the run log records that the feature armed, not
that it fired. A hit witness (program, phase ordinal, workgroup, last `pc_var`, trip count, read back
after the dispatch) is the next instrument, and it is what would turn this into a named
CFG-state-transition defect rather than a bounded-loop observation.

**Earlier, weaker control — the bound's value matters, not the counter's presence.** With the counter emitted
but the bound set to 4,000,000,000 (effectively no bound), the device is lost at `0x413dc6700`
dispatch 39 again, exactly as in the four unbounded runs. So the result is not codegen perturbation
from adding a Function variable to the loop:

| bound | outcome |
| --- | --- |
| none (counter not emitted) | dies at `0x413dc6700` dispatch 39 |
| **4,000,000,000** (counter emitted, never reached) | **dies at `0x413dc6700` dispatch 39** |
| 100,000 | gets past it |
| 4,096 | gets past it |

Four independent configurations all die at the same program and the same dispatch index. Bounding the
dispatcher is the only change that gets past it — so **the mechanism is a dispatcher loop that does
not terminate**, and `0x413e14900` hangs for a **different** reason — both 4,096 and 100,000 get past `0x413dc6700` and
then die at `0x413e14900` dispatch 52, so a dispatcher bound does not save it. Either it is not
dispatcher-emitted at all (a structured module's loops are unbounded by this switch), or its
non-termination is elsewhere.

Strictly the bound proves the loop exceeds 100,000 iterations rather than that it is infinite. For a
program whose only guest loop is bounded at **11** iterations on this dispatch's own data, either is a
defect: the dispatcher iterates once per guest basic block executed, and nothing in that guest program
justifies 100,000 of them.

**The trip bound is NOT a fix** — truncating a guest program's control flow produces wrong results by
construction. It is opt-in, says so when it arms, and exists to answer this one question.

This also supersedes an earlier note in this investigation claiming that dispatcher bounds of 4,096
and 2^20 "still lost the device". That note is contradicted by the table above and should not be
relied on.

### Where to look

The dispatcher exits when the whole-workgroup liveness slot (scratch slot `padded_lanes +
wave_count`) reads zero. Non-termination means at least one lane stays *active* forever — its
`active_var` never clears, or its next-`pc` never reaches an exit block. That is a recompiler
correctness question about `emit_cfg_state_machine`, independent of any resource or descriptor.

## RETRACTED: the cyclic-table root cause. The hanging dispatch's table is ACYCLIC.

Earlier revisions of this file, and several comments on #2542 and #2481, stated that `0x413dc6700`
hangs because its traversal table contains 183 cyclic start indices. **That is wrong**, and it was
wrong in the most embarrassing way: I measured the wrong buffer.

`0x413dc6700` runs many times per submit with **different tables**. Same program, same fetch PC,
different resolved base:

| dispatch | binding 8 @ `pc=0x5b` | outcome | cycles | longest chain |
| --- | --- | --- | --- | --- |
| `compute[37]`, source 38, order 16836 | `0x20f848417c` | **completed** | **183** | 58 |
| `failure[1]`, source 39, order 16841 | `0x20f848a240` | **HUNG** | **0** | **11** |

I dumped `--dump-compute-resource 37:8` believing it was the hanging dispatch's table. It is the
*succeeding* one's. The whole cyclic-table chain was built on the buffer belonging to the dispatch
that worked.

Two conclusions follow, and the second is the useful one:

- **The hanging dispatch's table is well-formed.** Zero cycles from all 2063 starts, longest chain
  **11 steps**, 19,038 total steps across every start. That loop cannot hang on this data — it is
  three orders of magnitude short of a watchdog timeout.
- **A table WITH 183 cycles was walked by a dispatch that completed.** So a cycle in this structure
  does not hang the shader either, and the loop model that predicted it must be incomplete — most
  likely the per-dispatch `s18` bound means the cyclic region (indices 412..1238) is not reachable
  from that dispatch's start set.

**So the cause of the hang is unknown again.** What survives is everything about the *consequence*:
`0x413dc6700` hangs deterministically at the same dispatch index, RADV hard-recovers, live compute is
disabled process-wide, the indirect latch drops every remaining draw, and skipping that one program
yields zero device losses and the first real scene content this title has produced. The mechanism
inside the shader is not established, and the "183 must become 0" oracle is void.

The pointer-chase loop is now *less* likely to be the hang: it is one of three dispatcher loops in the
module, and it is the one just shown to be bounded at 11 iterations on the hanging dispatch's own
data. The other two are unexamined.

## Other open defects

- **#2445** — specific lowercase glyphs (`r`, `s`, `m`) dropped from UI text: "Ente ing Sto y Mode".
  Surrounding text is intact, so it is per-glyph, not a font failure.
- **#2429** — the world cannot render on 32-wide devices: the EXEC-population-count fix requires
  `native_subgroup_size == wave_size == 64`.
- **#2428** (Windows) — frame-rate cliffs of ~60x (62 fps to 0.8 fps) within 60 frames at the gameplay
  transition.
- **#2424** (Windows) — `sceKernelBatchMap` `MAP_DIRECT` fails `ENOMEM` on a
  map-small/unmap/map-larger cycle. No Linux equivalent.

## Instruments worth knowing about here

- **`PROSPER_COMPUTE_SKIP_PROGRAM=0xADDR[,...]`** — decline named compute programs. A bisection tool,
  not a workaround. It announces itself at parse time and reports each skip through the decline census
  as `reason=skipped-by-selector`, so a diagnostic run can never be mistaken for a default one later.
  Ordered **after** the trace and SPIR-V dump, so "dump the module, skip the dispatch" works — which
  is how a recompiler change can be checked against a program that hangs the GPU, at zero device cost.
- **`[compute-decline]`** — every refusal in `execute_item` now names its reason, the dispatch, and a
  **running count**. The count is in the line deliberately: rate-limited diagnostics in this codebase
  have twice been read as censuses.
- **`PROSPER_SUBGROUP_LOG`** — the resolved native-subgroup contract. Read it as an *input*: it says
  what the config resolved to, not what the emitter produced.
- **`RADV_DEBUG=hang`** — the vendor tool, no prosper change needed. Writes a report to
  `~/radv_dumps_<pid>_<ts>` with `vm_fault.log`, `trace.log` (the last CP trace point) and
  `pipeline.log` (shader stats for the hung pipeline). `umr` is not installed here, so per-wave PCs
  are unavailable, and `DISASM` needs an additional RADV flag.

## Shared-GPU policy on this title

Every device-loss experiment here costs a hard recovery on a machine other agents are using. Allow at
most **one expected device loss per hypothesis**, stop that process immediately after the loss, and
investigate before the next routed run. `PROSPER_COMPUTE_SKIP_PROGRAM` plus the SPIR-V dump removes
the need for most of them.
