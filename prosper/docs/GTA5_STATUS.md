# Grand Theft Auto V (`PPSA04263`, RAGE) — status

**Rung 3** on the bring-up ladder: routed gameplay entry with real GPU draws. The HUD, radar and
tutorial text render; the 3D world does not.

Tracker: **#1873**. Active frontier: **#2481**. Route: `scripts/gta5/reach-story-mode.pad`
(read its header — the flip timing is measured, not estimated, and the tab navigation needs four R1
presses for a reason).

Historical design note for the descriptor work: `docs/FLAT_LOAD_DESIGN.md`. Do not start from it; the
descriptor-array lift it describes is complete.

## The current account — read this before anything below it

This document is layered: it grew as an investigation log, and several sections below are historical
transcripts kept for the evidence beside them. **Where a lower section disagrees with this one, this
one is current.** Each layer that was superseded now says so where it sits.

As of 2026-08-14, established and each measured rather than inferred:

1. **The missing world is one compute program.** `0x413dc6700` hangs the GPU into a RADV hard
   recovery. That disables live compute for the whole process, so every later indirect draw is
   dropped — and GTA V's world is GPU-driven, so those indirect operations *are* the world. Skipping
   it (`PROSPER_COMPUTE_SKIP_PROGRAM`) gives 0 device losses and the first real scene content.
2. **The hang is a non-terminating loop in that program.** Its 903-dword body contains exactly one
   backward branch (guest pc97 → pc88). The trip-bound witness fires there and reports that no
   invocation ever reached a dispatch ordinal past the loop body; the fence-wait duration
   for that dispatch is ~2,045 ms against sub-millisecond for every other dispatch in the route; the
   device loss follows on the next dispatch.
3. **The loop's data is cyclic at dispatch time.** Pre-dispatch, 806 of 1,782 reads receive a table
   in which 1,805–2,062 of 2,063 roots lead into a cycle. The guest loop cannot terminate on that.
4. **Our lowering of that loop shape is correct.** A hand-built kernel with the same
   `v_cmpx` / `s_cbranch_execz` / back-edge shape runs correctly on real Vulkan
   (`tests/test_cfg_trip_bound.cpp`), so the recompiler is not what fails to exit.

**So the open question is why the table is cyclic** — not whether the loop spins, and not whether we
lower it correctly.

**The writer is `0x413dc6700` itself.** Its own writeback line names the table it later reads
(`writeback binding=4 addr=0x20f848a240 changed=2062`), it reads through one binding and writes
through another, and the two tables swap roles between dispatches — a double-buffered union-find with
path compression. So this is a self-corrupting kernel, not a missing or refused producer.

`0x413ce3400` was an earlier attribution and is **superseded**: it is a writer of related state and it
is never declined on a routed run, so the "producer was refused" hypothesis is dead — but it was never
the question. Anything below that still reads as though an upstream producer must be found is
historical; the sections concerned now say so.

What is NOT established: which of `0x413dc6700`'s own stores introduces a cycle, and whether that is
the guest algorithm behaving correctly on inputs we produced wrongly upstream, or a defect in how we
execute its write-back path.

## RETRACTED 2026-08-14: "the corrupting write is `0x413dc6700`'s own" was one sample

The self-corrupting account was previously inferred from a writeback trace. It is now a direct
before/after on adjacent dispatches of the same program, in the same submit, on current master:

```text
disp 38  read  0x20f848417c  CLEAN   cycles=0  cyclic-roots=0     oob-roots=1385  max-depth=21
disp 38  execute ok, buffers=43, spirv=61143/177420afa4fd9c50
disp 38  writeback binding=4 addr=0x20f848a240 changed=2357
disp 39  read  0x20f848a240  CYCLIC  cycles=6  cyclic-roots=2062  oob-roots=0     max-depth=32
```

**RETRACTION.** The transition above is real and reproducible, and it does NOT establish cause. Two
observations from a wider census of the same instrumentation kill the inference:

- **A complete rewrite left the table CLEAN.** Submit 4312 dispatch 764 wrote 2,061 of 2,063 slots,
  and every read in that submit reports `cycles=0`. If this program's write corrupted the table, the
  most complete write in the route was the best chance to show it.
- **Tables flip to cyclic with NO write from this program.** Submit 4725 reads `0x20f848417c` with
  986 cyclic roots, and `0x413dc6700` writes nothing to it in that submit at all.

So a flip adjacent to its write is not evidence that the write caused it, because flips also happen
without one. What the original evidence established is that the program **writes the table it later
reads** — which was already known — plus one coincidence.

The generalisable error: a transition was observed next to a candidate cause, and adjacency was
treated as causation without checking whether the transition also occurs WITHOUT the cause. The
negative case is the whole test. Reproduce with `PROSPER_COMPUTELOG_CODE=0x413dc6700` plus
`PROSPER_COMPUTE_PARENT_WALK=0x413dc6700:0x5b:3:0x07FFFFFF:64` and read the interleaved
`parent-walk` / `execute` / `writeback` lines.

### There are TWO writers, and the second one was never in the picture

`PROSPER_COMPUTE_ADDRESS_WATCH=0x20f848417c` over a full route, 1,300 hits:

| program | bindings on the table (fetch pc) |
| --- | --- |
| `0x413dc6700` | 53, 65, **91** (the loop's read), 618, 629, 641, 653, 665, 677 |
| `0x413dc3400` | 597, 608, 620, 632, 644, 656 |

**`0x413dc3400` also writes this table**, through six bindings, and nothing in the investigation had
accounted for it. It is the program that runs immediately before the hanging dispatch — the walk
lines say `previous-code=0x413dc3400 previous-realized=1 previous-executed=1` — and it is on the
`role=terminal` list, so it is lowered through the CFG dispatcher as well.

This is exactly why the single-transition "confirmation" was wrong: flips with no write from
`0x413dc6700` are explained by a writer nobody was watching. On the one transition this run captured,
**both** programs touched the table since the previous read, so the correlation cannot yet separate
them — but the candidate set is now closed at two, which it was not before.

### FALSIFIED: our store INDICES are wrong

Grouped strictly per `(submit, dispatch)` — the ungrouped form of this analysis is meaningless and
produced a confident wrong answer first — every head a dispatch writes has its matching tail at the
very next slot:

```text
submit 4312 dispatch 764: 2061 slots written, 1031 heads, MISMATCHED=0
submit 5555 dispatch  38: 1721 slots written,  864 heads, MISMATCHED=0
```

So the pair-store index arithmetic is correct in our execution. Whatever produces an orphaned tail,
it is not this program emitting a head and a tail at non-adjacent slots.

### The corruption has a SHAPE: overlapping pair writes, not garbage

`PROSPER_COMPUTE_PARENT_WALK_DUMP` captured 138 cyclic tables. Every cycle in every one of them is a
**2-cycle**, and the records involved are structurally valid — correct tag, plausible index:

```text
rec[452] = 0x00000e32   tag=2  bit30=0  next=454     <== cycle
rec[453] = 0x40000e32   tag=2  bit30=1  next=454
rec[454] = 0x40000e22   tag=2  bit30=1  next=452     <== cycle
rec[455] = 0x00000e7a   tag=2  bit30=0  next=463
```

**The table is a sequence of PAIRS**: `rec[k+1] == rec[k] | 0x40000000`, same payload, bit 30 marking
the second element. 920 such pairs across 2,063 records, and the pairing is not parity-locked (523
begin at an even index, 397 at an odd one), so a pair is simply two consecutive slots.

A `bit30=1` record must therefore be immediately preceded by its `bit30=0` twin. In every cycle it is
not: above, slot 453 holds the *tail* of pair (452,453) while slot 454 holds a tail whose head is
gone. **Two writers claimed slot 453** — one writing the tail of (452,453), one the head of
(453,454) — and the survivor's orphaned partner at 454 points back at 452, closing the cycle.

So this is an **overlapping-allocation / lost-update** signature, and the earlier "61 two-cycles" note
was reading it correctly. What changes is that the falsification recorded against it was measured on
`0x413ce3400`'s instruction footprint, and the writer is `0x413dc6700`. Re-opened against the actual
writer.

Two candidates for where two writers get overlapping slots, neither yet tested:

- **The pair store itself.** The program contains 3 `buffer_store_dwordx2` and 5 `buffer_store_dwordx3`
  among its 23 stores, and a `dwordx2` writes exactly two consecutive dwords — a pair. An addressing
  error in the multi-dword store path (element versus byte, or an off-by-one base) shifts a pair by
  one slot and produces precisely this.
- **The slot allocator.** `ds_add_rtn_u32` at guest pc121 is a bump allocator: each thread atomically
  adds its size to an LDS counter and takes the old value as its base. Overlapping bases would do it.
  **Checked and currently NOT suspect:** the emitter lowers it to `OpAtomicIAdd` with
  `Scope_Workgroup` / `MemSem_WGAcqRel`, which is correct for `local=256` (four 64-wide waves), and
  the program has no global-memory atomics at all — 41 MUBUF ops, every one a plain load or store.

### What the same run also settles

- **Most dispatches write nothing.** 52 of 64 writebacks are `changed=0`; the substantial ones are
  `changed=2357`, `2061`, and a few single-digit updates. So the corrupting event is rare and
  identifiable, not a steady drift.
- **Both tables can be cyclic**, contrary to the earlier per-address reading: `0x20f848417c` is
  162 clean / 30 cyclic and `0x20f848a240` is 40 clean / 120 cyclic across 352 resolved reads. The
  asymmetry is real but it is not a property of the address.
- **The resource table varies per dispatch.** The same program address compiles to different modules
  (`spirv=58649/…` several distinct hashes, `61143/177420afa4fd9c50` on the big-write dispatch), and
  `buffers` ranges from 1 to 43. A single dispatch's resource picture is not the program's.

### The open question, now narrow

Why does that write produce cycles? Candidates, none yet tested:

1. the stored VALUES are wrong (our lowering of the store path computes the wrong record);
2. the stored INDICES are wrong (right values, wrong slots);
3. a concurrency effect at `threads=2063 local=256 groups=9` that the guest's algorithm tolerates on
   hardware and our lowering does not preserve.

The oracle is in place either way: `cyclic-roots` on the read immediately after the write is the
number to move, and dispatch 38 of a `0x413dc6700` pair is where to look.

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

### Diagnostics this investigation added, and what each is safe to conclude from

| switch | default | what it does | the trap it avoids |
| --- | --- | --- | --- |
| `PROSPER_CFG_TRIP_BOUND` + `_PROGRAM` + **required** `_PHASE` | off | caps the CFG dispatcher's back edge for one program and phase, and records a device-side hit witness | covers **only** the CFG dispatcher, so a null on a structurizer-accepted program means *not measured*; without `_PHASE` nothing is emitted, because one record cannot describe two phases |
| `PROSPER_COMPUTELOG_RAW` | off | writes a traced program's guest RDNA2 bytes for `tools/shader_inspect` | `PROSPER_SHADER_DUMP_SUCCESS` names files by hash, so recovering one program by address means hash-matching by hand |
| `PROSPER_INDIRECT_APERTURE_RECOVERY` | **off** | rebuilds a base-less queue-2 indirect argument address from the last-seen SetBase aperture | changes execution: the aperture is learned from any SetBase on any queue, and *mapped* is not *this is the argument buffer* |
| `PROSPER_INDIRECTLOG` | off | per-packet base/offset/queue, the three argument dwords, and an end-of-run outcome census | readability was probed and values were not, so a misread surfaced only as a `workgroup-count-limit` decline thousands of operations later |

The aperture recovery is opt-in on purpose and the trade is recorded rather than implied: **off, 50 of
64 indirect compute dispatches skip as "unreadable arguments"; on, 0 do**, and the probe found the raw
low address unmapped and `aperture | low` mapped on 49 of 49. That is real evidence about where those
arguments live and it is still not provenance — one process VA space can hold mapped allocations under
several high-32 prefixes, so accepting an address because 12 bytes are readable admits dispatching
group counts read out of an unrelated live allocation.

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
- **A lost atomic corrupts the traversal table.** **VOID against the current writer — the evidence
  below is about `0x413ce3400`, and the table's writer is `0x413dc6700` itself.** The corruption
  signature (61 two-cycles) still stands as an observation, and so does the instrument note at the end
  of this entry, which is why the entry is kept rather than deleted. What does not stand is the
  falsification: showing that `0x413ce3400` performs no atomic cannot rule out a lost-atomic
  write-path defect in a *different* program. To settle it, re-run the same footprint analysis on
  `0x413dc6700`'s own stores. Everything from here to the end of this bullet is that superseded
  argument. The table that hangs `0x413dc6700` is a linked list
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
  was lost both times, so a bound at those values does not rescue the frame. This is **not** in
  tension with the hit witness firing at 4,096 further down: the witness says the loop *reaches* the
  cap, and the cap then truncates that dispatch's control flow, which produces wrong results for
  every later consumer rather than a working frame. "The bound does not fix the title" and "the loop
  runs away" are both true. #2481.

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
(`trips=4096`, with no invocation ever reaching an ordinal past the loop body), the fence-wait
duration, and the loss ordering. The witness's fields are dispatcher
quantities; resolve an ordinal against the `dispatch map:` line the same phase prints.

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
not.

**That is evidence about where the arguments live, and it is not provenance — which is why the
recovery is opt-in.** This paragraph used to end "so a wrong aperture leaves behaviour exactly as it
was", and that claim does not hold: the aperture is learned from *any* SetBase on *any* queue, and one
process VA space can contain mapped allocations under several high-32 prefixes. A wrong aperture that
happens to land on a mapped allocation is accepted, and the dispatch then reads group counts out of
whatever lives there. Gated behind `PROSPER_INDIRECT_APERTURE_RECOVERY`, default off.

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
`PROSPER_COMPUTE_PARENT_WALK` timings.

**The sentence that used to close this paragraph — "`0x413ce3400` remains the only writer identified
so far" — contradicted the section it sits in and is withdrawn.** The writeback evidence directly
above names `0x413dc6700` as the writer of the table it later chokes on. Both readings were live in
this document at once, which is worse than either being wrong: they imply different next experiments
(find the producer vs. audit this kernel's own stores), and the reconciliation is at the top of the
file.

## CONFIRMED BY HIT RECORD: phase 0's dispatcher loop runs past 4,096 iterations on the hanging dispatch

`PROSPER_CFG_TRIP_BOUND=N` (diagnostic) forces a dispatcher loop out after N iterations. With
`N=100000` the run gets **past** `0x413dc6700` and dies much later at a **different** program:

> **The switch has since changed and these runs predate it.** It now bounds only the program and
> phase named by `PROSPER_CFG_TRIP_BOUND_PROGRAM` / `PROSPER_CFG_TRIP_BOUND_PHASE`, and **the phase
> selector is required** — armed without one it emits nothing and says so. The table below was taken
> when the bound applied to every dispatcher phase of the selected program, so reproduce it by naming
> a phase explicitly. The maps every phase prints when the bound is armed list what is available.

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

So the runaway dispatcher is the one wrapping the EXEC-narrowing walk. The defect is named to a
program, a phase, and a guest **pc** range — pc is a dword offset and RDNA2 instructions are variable
length, so an instruction count cannot be read off it; this line previously called the range
"117-instruction", which it never was.

(The "maximum chain of 11 steps" quoted here came from the capsule-timing measurement that the
retraction above supersedes. The pre-dispatch census is the current figure: 806 of 1,782 reads receive
a table in which 1,805-2,062 of 2,063 roots lead into a cycle.)

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
writes into the top of the internal GDS buffer when a cap runs out; the host prepares those dwords
before the selected dispatch, reports them after it, and then RESTORES the guest's original values,
so the shared GDS buffer is byte-identical afterwards for whatever dispatch uses it next. It touches
them only when a witness was actually **emitted** for that program — not merely when the selectors
accept it, since a structured loop or a phase ordinal the program lacks satisfies every selector and
emits nothing. **The current record is five dwords — hit flag, phase, highest trip count, and
the lowest and highest dispatcher switch-case ordinal visited — with the last three reduced across
invocations by device-scope atomics.** The per-invocation "last block index" this line used to name
no longer exists: it was one sample, it could not answer whether the dispatcher was cycling, and its
label was published wrongly twice (instrument trap 172).

**Positive control first**: at bound **2** — which a 15-block dispatcher phase cannot satisfy — the
witness produces **1,606 hit records**. It fires.

At bound **4,096**, on the same program and phase:

```
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=9  trips=4096 submit=5547 dispatch=38
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=14 trips=4096 submit=5547 dispatch=39
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 last-block=9  trips=4096 submit=5547 dispatch=40
    ^ HISTORICAL TRANSCRIPT. `last-block` is the label this run printed; the field is a dispatcher
      switch-case ordinal and the per-invocation form was removed rather than renamed again. The
      current line reports `trips` and `dispatch-range` only. See the correction directly below.
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

**Re-measured on a routed run, 2026-08-14, with the ATOMIC record — and the number changed.**

```text
[cfg-trip-bound] program 0x413dc6700 phase 0 dispatch map:
    0:pc0..<41  1:pc41..<50  2:pc50..<55  3:pc55..<61  4:pc61..<67  5:pc67..<73
    6:pc73..<76  7:pc76..<88  8:pc88..<91  9:pc91..<98  10:pc98..<103 ...
[cfg-trip-bound] HIT program=0x413dc6700 phase=0 trips=4096 dispatch-range=0..9   (5 hits, 1 loss)
```

**The earlier `dispatch-range=6..9` was an artifact of last-writer publication and is withdrawn.**
That record was written with plain stores from every invocation reaching the cap, so it reported one
invocation's local extremes; the reduction over all invocations is `0..9`. The claim built on it —
"the state machine visited only ordinals 6..9, so it is cycling in the loop" — does not survive,
because ordinal 0 is the program entry and every invocation passes through it. A ten-ordinal span is
not evidence of a cycle.

**What the corrected record does establish is the CEILING, and it is stronger.** The maximum ordinal
reached, over every invocation and workgroup, is **9** — the loop body at `pc91..98`. Ordinals 10..14
(`pc98..116`, everything after the loop) were **never reached by any lane**. So no invocation ever
left the loop, which is exactly the claim at issue, and it now rests on a true reduction rather than
on whichever invocation happened to write last.

The lesson is worth keeping separate from the conclusion: the conclusion survived, and the derivation
did not. A last-writer record produced a *narrower*, more striking span than the truth, and a narrower
span is the direction that reads as stronger evidence — which is why nobody questioned it.


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
