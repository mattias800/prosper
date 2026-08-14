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

### The trip bound measures ONE emitter — a null from it is "not measured"

`PROSPER_CFG_TRIP_BOUND` caps the **CFG dispatcher's** back edge. It does not touch either structured
loop emitter, so a program the structurizer accepts is *structurally unmeasurable* by it and reports
nothing whether or not it runs away. `0x413dc6700` happens to be covered — it reaches `role=terminal`
in the structurizer and is lowered by `emit_cfg_state_machine` — which is why its witness fired at all.

This is recorded because the emitter's own comment asserted the opposite for a while: a correction
saying the structured emitters are NOT covered was added directly above a surviving sentence claiming
all three were. Both are now pinned by assertions — a structured loop must be byte-identical when
armed, a dispatcher loop must not be.

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
- **Our `v_cmpx` / `s_cbranch_execz` lowering cannot exit the guest's pointer-chasing loop.**
  `0x413dc6700`'s whole 903-dword body contains exactly **one** backward branch, at guest pc97 back to
  pc88, and that loop's only exit is `s_cbranch_execz` after a `v_cmpx_ne_u32`. Nothing else in the
  loop writes EXEC, so if `v_cmpx` narrowed VCC instead of EXEC — plausible, because the e32
  encoding's destination field still reads as VCC and `shader_inspect` prints it as `special:106` —
  the loop could never end. **Falsified by a hand-built kernel of the identical shape**
  (`tests/test_cfg_trip_bound.cpp`): built by hand rather than derived from the capture or the
  recompiler, its body decrements the index instead of chasing a buffer, so only the control flow is
  on trial. On real Vulkan, lane *i* walks exactly *i* steps for all 128 lanes — per-lane EXEC
  narrowing and the cross-lane `execz` vote are both correct. #2542.
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

## CORRECTED AND CONVERGED: `0x413dc6700` dispatch 39 takes ~2 SECONDS, and the loss follows it

Measuring the DURATION of every compute fence wait, and reporting any wait over 100 ms even when it
succeeds, produced exactly one line in a whole route:

```
[compute] SLOW fence wait 2045.2 ms result=0 program=0x413dc6700 submit=8116 dispatch=39 order=14036 groups=9x1x1
[compute] fatal Vulkan device loss stage=queue-submit … program=0x413dc6700 submit=8116 dispatch=40 order=14041
```

**One abnormal wait in the entire run — 2,045 ms against sub-millisecond for everything else — on the
dispatch every other instrument has named, immediately followed by the device loss on the next
dispatch.**

Three independent instruments now converge on `0x413dc6700` dispatch 39: the trip-bound hit witness
(`trips=4096`), the fence-wait duration, and the loss ordering. **The witness's third field is
a guest PC, not a block ordinal** — see the correction below; it does not say the guest's loop
is the thing spinning.

### This corrects the section that used to be here

That section argued "no compute dispatch ever hangs", from **0 fence-wait timeouts across 705 waits**.
The count was right and the inference was wrong: the timeout is **30 seconds** and the event is **2
seconds**, so a dispatch can be three orders of magnitude slower than every other one and still never
trip it. A zero timeout count measures only "nothing exceeded 30 s" — it says nothing about whether
anything is pathologically slow, which is the actual question.

The alternative that motivated the check — that a context reset SIGNALS pending fences, so a killed
job returns `VK_SUCCESS` and looks instant — remains untested and is no longer needed to explain
anything.

What survives from that section: the loss is still reported at `queue-submit` on the *next* dispatch,
so the loss line alone never named the culprit. Instrument trap 170 stands as written about
attribution; only the "not compute's" conclusion drawn from it was wrong.

## Superseded: the earlier reframing

### 

Counted across every routed run in this investigation:

| observation | count |
| --- | --- |
| compute dispatches that entered their fence wait | **705** |
| compute dispatches that completed it | **705** |
| `[compute-decline] reason=queue-wait` (a 30-second fence timeout) | **0** |
| `[compute-decline] reason=queue-submit` (device ALREADY lost) | **28** |

`execute_item` submits and then waits on a fence with a **30-second** timeout, and reports a timeout as
`queue-wait`. That has never once fired. Every compute dispatch this title issues completes.

The 28 losses are all `stage=queue-submit`, which means `vkQueueSubmit` returned
`VK_ERROR_DEVICE_LOST` — the device was **already** dead when compute next submitted. The dispatch
named in that message is the first one *after* the reset, not the one that caused it.

**So the GPU hang is not caused by a compute dispatch.** It is almost certainly in the graphics
submission path, and the compute backend is a victim that discovers the dead device and then disables
itself process-wide — which is what drops every later indirect draw.

### What this overturns

- **"`0x413dc6700` hangs the GPU."** Its dispatcher loop genuinely runs past 4,096 iterations — the hit
  witness recorded that directly, at `trips=4096`, on dispatch 39 — but it still *finishes* inside the
  30-second fence. A long loop, not a hang.
- **"`0x413e14900` is a second hanging program."** It is a 753-dword module with **zero loops** running
  `threads=42x1x1 groups=1x1x1` with `result=ok`. It was never a candidate.
- **The trip-bound A/B matrix.** Each cell was a single run of a failure that varies run to run — the
  same build died at `0x413dc6700` dispatch 39 in one run and dispatch 40 in another, and reached
  `0x413e14900` in a third. Single-run cells cannot support the causal reading I gave them.

### What survives

- The cyclic traversal tables are real and measured at the correct timing. They make that loop very
  long. They do not, on this evidence, hang the GPU.
- The queue-2 indirect-argument recovery is real and deterministic: 50 of 64 skipped dispatches now
  execute.
- The consequence chain from the loss onward — compute disabled process-wide, `producer_epoch_ok`
  cleared, indirect latch, no world — is unchanged and still explains the black frame.

**The next investigation is the graphics submit path**, not the recompiler and not compute resources.

## FIXED: indirect compute dispatches on queue 2 had no argument base — 50 of 64 were skipped

**What the fix reliably changes:**

| | before | after |
| --- | --- | --- |
| `indirect dispatch skipped: unreadable arguments` | **50 of 64** | **0** |

That is deterministic and verifiable in every run: the skips are gone and those dispatches execute.

**What it does NOT reliably change, corrected after a second run.** The first run with the fix got past
`0x413dc6700` and died later at `0x413e14900` dispatch 52, and its frame showed sun, lens flare and
radar. I wrote that up as a before/after improvement. **A second run with the same build died at
`0x413dc6700` dispatch 40 again.** The hang is data-dependent, so a single run either side proves
nothing about it, and the frame content had already appeared in earlier *skip* runs — so neither the
survival nor the frame is attributable to this fix.

The honest statement is: the fix removes a real and deterministic class of dropped work; whether that
changes the hang is unmeasured, and would need repeated runs on both sides.

### The defect

`SET_BASE_INDIRECT_ARGS` sets one `indirect_compute_base`, and that is **per-fold** state. A PS5
process has one GPU virtual address space, but GTA V's async-compute queue carries `DispatchIndirect`
packets whose 32-bit payload is a full address *within the already-selected aperture* and no SetBase
of its own — so its base is zero and its arguments resolve to an unmapped low address.

Probed on **49 of 49** such dispatches before changing anything:

```
readable? low=0  aperture20=1  hi-dword=0
          (low=0xf8480120  ap20=0x20f8480120  hi64=0x21f8480120)
```

The raw low address is unmapped; `aperture | low` is mapped; folding the modifier as an ADDR_HI is
not. The recovery learns the aperture from any full indirect base and applies it **only** when the
address we would otherwise use is unreadable and the recovered one is readable — so a wrong aperture
leaves behaviour exactly as it was.

### What it does NOT fix

The world is still black, and the traversal tables are still cyclic in a large minority of reads
(944 cyclic vs 1,179 clean, against 806/1,782 before — proportionally better, not resolved). So the
skipped dispatches were **a** cause of lost work but not the whole cause of the cyclic structure. The
next device loss is `0x413e14900`, which a dispatcher bound does not save either.

## Superseded: the root-cause candidate as first written

`SET_BASE_INDIRECT_ARGS` sets one shared `indirect_compute_base` in the command processor. Logging
base, offset and queue separately at the `DispatchIndirect` site:

| queue | dispatches | base | outcome |
| --- | --- | --- | --- |
| **1** | 14 | `0x205b690f80` | resolve and run |
| **2** | **50** | **`0x0`** | `args = raw offset` -> unreadable -> **skipped** |

The queue-2 offsets are `0xf8480120`, `0xf8480160`, `0xf84801a0`, `0xf84801e0`, `0xf8480220` … stepping
by 0x40 — and the guest arena those dispatches belong to is at **`0x20f8480000`** (347,040 bytes; it is
the buffer the `s_endpgm` kernel declares). So the intended address is almost certainly
`0x20f8480120`, and what is missing is a `0x20_00000000` base that queue 2 never receives.

**50 of 64 indirect compute dispatches in one route are therefore skipped entirely**, with only
`[agc] indirect dispatch skipped: unreadable arguments at 0x…` to show for it — a message that prints
the SUM, so it cannot distinguish a bad offset from a base that was never set.

### Why this is very likely THE defect

It closes the chain that every other measurement in this document constrains:

1. `SET_BASE_INDIRECT_ARGS` for compute is seen only on queue 1; queue 2's folds start with base 0
2. queue-2 indirect dispatches resolve to an unmapped address and are skipped
3. the skipped dispatches are the maintenance passes for the traversal structure
4. the parent array degrades — **once, irreversibly** (each buffer transitions clean->cyclic exactly
   once and never recovers, which is what a half-applied union-find update looks like)
5. `0x413dc6700` walks the cyclic chain and never terminates — recorded directly by the trip-bound
   hit witness (its third field is a dispatch ordinal — see the correction below)
6. GPU watchdog -> RADV hard recovery -> live compute disabled process-wide
7. `producer_epoch_ok` cleared -> `indirect_dependencies_ok` latched -> every remaining indirect draw
   dropped untried -> **no world**

**Not yet proven**: that supplying the missing base makes the tables stay acyclic. That is the next
experiment, and the cycle census is already the oracle for it — the number to move is
`cyclic-roots`, per dispatch, at pre-dispatch timing.

## THE TABLE IS CYCLIC AT DISPATCH TIME — measured with the right instrument at the right moment

`PROSPER_COMPUTE_PARENT_WALK` reads the selected resource **immediately before `compute({item})`** —
the timing the capsule's post-submit snapshots never had — and models this exact loop
(`while (index != 0) index = (records[index] >> shift) & mask`), reporting cycles directly.

```
PROSPER_COMPUTE_PARENT_WALK=0x413dc6700:0x5b:3:0x07FFFFFF:64
```

**1,782 resolved reads across one route:**

| reads | cycles | cyclic roots (of 2,063) | oob roots | max depth |
| --- | --- | --- | --- | --- |
| 16 | 0 | 0 | 0 | 12..15 |
| 960 | 0 | 0 | **1,894** | 19 |
| **795** | **15** | **1,805** | 0 | 22 |
| 5 | 17 | **2,062** | 0 | 20 |
| 6 | 1 | 1,039 | 269 | — |

**806 of 1,782 dispatches receive a table in which 1,805-2,062 of 2,063 roots lead into a cycle.** The
guest loop cannot terminate on that data, so the dispatcher spinning past 4,096 iterations is the
*correct* behaviour for what it was handed. Combined with the hit witness, the chain is closed: bad
data in, non-terminating walk, GPU hang.

The 960 reads with `oob-roots=1894` terminate by the RDNA2 rule that an out-of-range `idxen` load
returns zero — those dispatches complete, which is why only some dispatches hang.

**Read the whole distribution, not the first rows.** The first three log lines of this run show
`cycles=0 cyclic-roots=0`, and stopping there gives exactly the opposite conclusion — "the data is
fine, so the emulation is broken". The clean reads are a real minority of the population, and they
come first.

### And it is ONE buffer, not a random failure

Correlating every resolved read by the table address it walked:

| table address | outcome | reads |
| --- | --- | --- |
| `0x20f848417c` | clean | **966** |
| `0x20f848417c` | cyclic | 6 |
| `0x20f848a240` | **cyclic** | **800** |
| `0x20f848a240` | clean | 10 |

`0x20f848a240` is cyclic in **800 of 810** reads (98.8%); `0x20f848417c` is clean in **966 of 972**
(99.4%). The program alternates between two traversal tables, and **one of them is systematically
corrupt while the other is systematically correct**. That is not a race or a timing artifact — it is
a property of one buffer.

The predecessor correlation carries no signal: both outcomes follow the same preceding dispatch
(`previous-code=0x413dc6700`, realized, mostly not executed) in the same proportion.

Note this is the same address whose *post-submit* snapshot I measured as acyclic, and used to retract
the cyclic-table hypothesis. At dispatch time it is cyclic. The original hypothesis was right in
substance; the capsule's snapshot timing is what made it look wrong.

### The writer is the CONSUMER ITSELF — this kernel corrupts its own structure

`0x413dc6700` both walks the table and writes it. Its own write-back lines say so:

```
[compute] execute submit=4310 dispatch=776 code=0x413dc6700 threads=2063x1x1 local=256x1x1 buffers=43
[compute]   writeback binding=4 addr=0x20f848a240 size=8252 changed=2062 hash=b24dd1c1…->514428f8…
```

It reads through one binding (`fetch=0x5b`, the loop load) and writes through another
(`binding 4`/`binding 5`, `fetch=0x35`/`0x41`), and the two tables swap roles between dispatches — a
double-buffered structure. Combined with the loop shape (chase a parent index to a root, then take the
**parity of the depth**, `v_and_b32 v1, 1, v2` compared against `s24`), this is the shape of a
**union-find with path compression**: walk to the root, then write compressed parents back.

That reframes the defect entirely. There is no upstream producer to blame — **the corruption is
produced by the same program that later chokes on it.** Dispatch N writes a malformed parent array;
a later dispatch reads it, finds a cycle, and never terminates.

It is consistent with the address split: `0x20f848a240` is the buffer this kernel *writes*, and it is
the one that is cyclic on later reads (800 of 810); `0x20f848417c` is clean 966 of 972.

**Correction to an attribution I nearly published.** I first read the trace as naming `0x413cf9a00`
the writer — a program that is genuinely skipped (`[compute] skip unsupported program 0x413cf9a00`) and
which sits on the recompiler-reject list, so it made a very attractive culprit. It was a parsing
error: my attribution regex latched onto the nearest `program 0x…` text, which included the skip line
itself. The real anchor is the `[compute] execute … code=…` line, and it says `0x413dc6700`.

So the next question is not "who failed to write this" but **"why does this kernel's own output
contain cycles"** — a store-path or algorithm-emulation question about `0x413dc6700`, not a missing
producer.

### What this settles, and what it reopens

Settled: the defect is **upstream data**, not the dispatcher's lowering of this loop. The recompiler
faithfully executes a walk that genuinely does not terminate.

Reopened: **why is the table cyclic?** The earlier producer investigation was measured on
post-submit snapshots and on the wrong dispatch's buffer, so it has to be redone against
`PROSPER_COMPUTE_PARENT_WALK` timings. `0x413ce3400` remains the only writer identified so far.

## CONFIRMED BY HIT RECORD: phase 0's dispatcher loop runs past 4,096 iterations on the hanging dispatch

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
generation (both arms carry the identical counter). This is targeted A/B evidence that phase 0's loop is implicated. It is **not** proof that the
cap fired — see the witness section below.

**Which phase spins — bisected.** `PROSPER_CFG_TRIP_BOUND_PHASE=K` bounds only the K-th dispatcher of
the selected program. Three runs, everything else byte-identical:

| phase bounded | guest pc range | outcome |
| --- | --- | --- |
| **0** | **0..116** | **gets past `0x413dc6700`**, dies later at `0x413e14900` |
| 1 | 117..129 | dies at `0x413dc6700` dispatch 39 |
| 2 | 130..902 | dies at `0x413dc6700` dispatch 39 |

**Only bounding phase 0 saves the device** (targeted A/B evidence, not proof of a cap hit). Phase 0 covers guest pc 0..116, and the program's only
loop is at **pc 88..97** — inside it. That is the pointer chase:

```
 88  v_mov_b32_e32     v2, s22
 89  v_cmpx_ne_u32_e32 0, v1        ; EXEC &= (v1 != 0), never restored in the loop
 90  s_cbranch_execz   7            ; exit when EXEC == 0
 91  buffer_load_dword v1, v1, s[0:3], 0 idxen
 95  v_bfe_u32         v1, v1, 3, 27
 97  s_branch          -10
```

So the runaway dispatcher is the one wrapping the EXEC-narrowing walk, and the walk's own data gives
a maximum chain of 11 steps. The defect is named to a program, a phase and a 117-instruction guest pc
range.

**The spinning dispatcher is the PORTABLE one, and that also explains the earlier void result.** The
emitted module for `0x413dc6700` contains **zero `OpGroupNonUniform*` instructions** — so
`b.native_subgroup_size` was 0 and the branch votes take the portable path: publish into
`vote_pending_var` / `vote_value_var` in the switch case, then reduce through LDS scratch behind two
workgroup barriers in the continue block.

That is forced, not incidental. `rdna2_to_spirv.cpp` sets `b.native_subgroup_size = 0` when
`partial_barrier_phases || exact_partial_dispatcher`, and this dispatch is both barrier-phased and
partial (`threads=2063`, `local=256` — the final workgroup carries 15 real threads of 256).

**This retires the `PROSPER_NATIVE_COMPUTE_MULTIWAVE` mystery recorded above as "void, not
falsified".** That switch only moves `config.native_subgroup_size`, which this line then overrides to
0 for exactly this shape — so the module *could not* change, and the byte-identical hashes were the
correct outcome rather than a broken lever. The result stays void as evidence about emulation cost,
but the reason is now known.

It also narrows where the defect can live: the vote machinery under suspicion is the portable
LDS-reduction path for a barrier-phased dispatcher with a partial final workgroup, not the native
subgroup path. Two things checked there and found correct: the vote mailboxes (`vote_pending_var`,
`vote_value_var`, …) are reset at the top of every dispatcher iteration, so a stale vote cannot carry
over; and each lane's LDS contribution is gated on its own `pending` bit.

**The device-side hit witness WORKS, and the cap fires — on the dispatch that hangs.** The shader
writes hit flag, phase, last block index and trip count into the top of the internal GDS buffer when a
cap runs out; the host reports and clears them per dispatch.

**Positive control first**: at bound **2** — which a 15-block dispatcher phase cannot satisfy — the
witness produces **1,606 hit records**. It fires.

At bound **4,096**, on the same program and phase:

```
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=9  trips=4096 submit=5547 dispatch=38
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=14 trips=4096 submit=5547 dispatch=39
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=9  trips=4096 submit=5547 dispatch=40
```

**Correction (2026-08-14): the third field is a dispatcher switch-case ORDINAL — `next-dispatch`.**
It was first printed as `last-block`, then briefly "corrected" to `next-guest-pc`; both were wrong,
in opposite directions, and the second was published on this branch before being caught. Every write
to the emitter's `pc_var` stores `dispatch_for_block[...]`, so despite the variable's name it is
neither a basic-block index nor a program counter. Instrument trap 172.

**What this does and does not settle.** The numbers 9, 14, 9 are ordinals into phase 0's dispatch
table. Whether ordinal 9 covers the guest's loop body at pc88..97 is a question about that table, not
something derivable by hand from the value — and the hand-mapping is what produced both errors. Each
phase now prints its map when a bound arms:

```text
[cfg-trip-bound] program 0x413dc6700 phase 0 dispatch map: 0:pc0..<N 1:pc... ...
```

so the ordinal resolves against emitted evidence rather than by hand.

**Resolved on a routed run, 2026-08-14 — and the original attribution was right.** Phase 0's map, and
the witness with the ordinal span it now records:

```text
[cfg-trip-bound] program 0x413dc6700 phase 0 dispatch map:
    0:pc0..<41  1:pc41..<50  2:pc50..<55  3:pc55..<61  4:pc61..<67  5:pc67..<73
    6:pc73..<76  7:pc76..<88  8:pc88..<91  9:pc91..<98  10:pc98..<103 ...
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 next-dispatch=9 trips=4096 dispatch-range=6..9
```

Ordinal **8 is `pc88..91`** — the loop header, `v_mov` / `v_cmpx_ne_u32` / `s_cbranch_execz`. Ordinal
**9 is `pc91..98`** — the loop body, `buffer_load_dword` / `s_add_i32` / `v_bfe_u32` / `s_branch`.
Across all 4,096 iterations the state machine visited **only ordinals 6..9**, i.e. the loop and the
block that falls into it. The guest's own loop is what spins, which is what the first reading of this
witness said before either mislabelling. `dispatch-range` is what makes it a measurement rather than a
one-sample inference: a span of four adjacent ordinals over 4,096 iterations is a cycle, not a pass.

11 hits across dispatches 38..48 of one submit — **including dispatch 39, the exact dispatch that
hangs on an unbounded build**. Last block is **14** on dispatch 39 and **9** on the other ten.

That is a direct record that the loop runs past 4,096 dispatcher iterations, replacing the earlier
inference from device survival. It also answers the reviewer's P1: the conclusion no longer rests on
comparing two modules that differ by a literal operand.

**An earlier version of this witness was inert, and the reason is worth keeping.** The SPIR-V declared
binding 127 while the resource table did not carry it, because the injection sat inside a conditional
resource-build region this program skips. The contract check then rejected the module and the dispatch
was **declined** — so the device survived because the program never ran, which is indistinguishable
from "the bound worked" in every artifact except one line reading
`skip invalid descriptor contract`. Its positive control at bound 2 produced zero hits, which is what
exposed it. Injecting at table finalization fixed it.

**A cheap hit-witness attempt that does NOT work, recorded so nobody repeats it.** Idea: run the same
phase-0 bound at 4,096 and 65,536 and compare the program's write-back hashes — if the loop terminates
naturally under both, identical inputs must give identical outputs, and a difference would witness
truncation. One dispatch in each run did share binding 4's input hash (`b24dd1c1ba122e6d`) and their
outputs differed (`514428f8…` vs `18c1fbdb…`), which looks like a hit.

It is not. Comparing **all** of that dispatch's inputs, **three of six differ** between the two runs
(bindings 3, 6 and 8) — the two runs are different live sessions on per-frame data, so the output
difference is fully explained without any truncation. The comparison cannot discriminate, and the
apparent agreement on one binding was coincidence.

A real witness has to be device-side, as originally specified: program, phase ordinal, workgroup, last
`pc_var` and trip count, written by the shader on a cap hit and read back after the dispatch.

What is still NOT established is, and there is no
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
