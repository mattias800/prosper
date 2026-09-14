# Uncharted: Legacy of Thieves Collection (PPSA05684) — status

**Rung 0 as of 2026-09-14.** The guest boots, runs its job system, mounts its content archives,
starts its world load, and executes **exactly five iterations of its game loop** before parking.
Nothing renders. Tracker: #3616.

**The Vulkan backend is not the frontier on this title, and that is measured rather than argued:**
the stall reproduces identically with `PROSPER_RENDER=0` (same five iterations, same wait, same
timing). **State that arm precisely, because the obvious paraphrase overstates it:** `PROSPER_RENDER`
turns off the *live renderer*; prosper still parses the guest's command buffers, still folds its
submits, and still runs its flip and event-queue emulation. So what the arm rules out is the
recompiler, the descriptor layer, the Vulkan submission path and the 14 skipped compute programs —
**not** prosper's flip/EOP *semantics*, which the guest's frame pacing does depend on. See
`## Ruled out`.

The title is a Naughty Dog engine with a **fiber-based job system** (`NdJob Fiber`, `NdJobWorkerThre`),
and that is the single most important thing to know before working on it: fibers start on the main
thread and are resumed by worker threads, so **any prosper mechanism that assumes "the host thread that
enters an HLE call is the host thread that returns from it" is wrong here.** Two defects of exactly
that shape have already been found (#3615, #3638), and #3623 is a third.

## Route

```bash
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
  tools/screenshot --seconds 25 --timeout 60 --count 1 --every 20 --out <dir> \
  <DUMP_ROOT>/PPSA05684-app0
```

No input script is needed to reach the current frontier.

Useful diagnostics on this title specifically:

| variable | what it answers |
| --- | --- |
| `PROSPER_FIBERLOG=1` | fiber init/run/yield with the owning **tid**, and which `sceFiberSwitch` branch refused |
| `PROSPER_TLSLOG=1` | `tid -> guest_tp` for every guest-TLS activation — the map that identifies a foreign TCB |
| `PROSPER_HWBP=<off> PROSPER_HWBP_ALLTHREADS=1` | execute breakpoint **on worker threads**; without `ALLTHREADS` a zero is void, not negative |
| `PROSPER_HWBP=0x57e829 PROSPER_HWBP_ALLTHREADS=1 PROSPER_HWBP_MAX=4000` | **the progression meter for this title**: one hit per game-loop iteration. Reads 5 on every arm so far, across four separate runs and two baselines, so a lever that does not move it did nothing |
| `PROSPER_HWWATCH_ABS=<addr> PROSPER_HWWATCH_ABS_ALLTHREADS=1` | who WROTE a guest slot, arming every guest thread rather than the main one (instrument trap 168's blind spot) |
| `PROSPER_POLLWATCH=<addr>` | when a guest slot changed, with elapsed time — bounds the stall before an exact instrument is pointed at it |
| `tools/re/guest_stacks.py <pid> --match <lo>-<hi>` | the call chain of a **parked fiber**, which no host-thread backtrace can show: when a job waits, its stack is in guest memory that no thread points at |
| `PROSPER_FIBER_DUMP_MS=<ms>` | every fiber's state on a timer, including the resume `rsp` — the one place the boundary between a parked fiber's live frames and the stale bytes below them is known exactly. Pair it with `guest_stacks.py` |
| `PROSPER_FIBERLOG=1` | now records **successful** `sceFiberSwitch` calls too; before that its zero was void (trap 282) |

## What works

- Boots and survives a 25 s capture with no faults, across 13 worker threads and 100 fibers.
- Mounts its real content archives: `bin.psarc`, `shaders.psarc`, `data.psarc`, `core.psarc`,
  `combo-common.psarc`, `common.psarc`, `sp-common.psarc`, `world-boat-intro.psarc`.
- Classifies its PlayGo chunks and starts its first world switch (`----- Switching world: from  to
  core`), loads its script modules, and runs its save-data and trophy initialisation.
- **Submits real command buffers.** With `PROSPER_GFXLOG=1`, a 25 s run folds **10 `SubmitDcb`** and
  **33 `SubmitAcb`** streams, including three `Flip` packets carrying buffer indices 0, 1 and 2.
  Zero draws in any of them.
- Runs **real GPU compute**: `[render-timing] compute calls=50 dispatches=50 avg_ms=3.34`, with
  137.2 MiB of image snapshots and 56.4 MiB of storage-image results copied back.
- 6,770 successful `sceFiberSwitch` calls in a 22 s run, still switching at the moment the capture
  ends: the job system stays alive and keeps cycling fibers. It simply has no work.

## The frontier: five game-loop iterations, then a job that never completes

The stall is located exactly, and the addresses below are module offsets (`eboot+`), stable across
runs:

| where | what |
| --- | --- |
| `0x57e9e0` | `common-game-loop-job.cpp`: the main guest thread kicks the game-loop job and then `usleep(10)`-polls a counter at `0x57eaab`. It never stops polling. |
| `0x57e7d0` | the game-loop job body, run once on a worker fiber. Its loop head is `0x57e829`. |
| `0x57e829` | **executes exactly 5 times**, every run. |
| `0x5a9ce0` | `GameLoopUpdate` — the per-frame job the loop runs, entered 5 times. |
| `0x5ab6de` | `GameLoopUpdate` kicks its own sub-job (`0x5a45c0`) at `common-game-loop.cpp:5081` and waits; reached and RETURNED FROM 5 times. |
| `0x5ab71c` | the frame-latency wait: `WaitForCounter(counter, -frameIndex, 1)`. Reached 5 times, returns 4 times. **This is where the title stops.** |

The counter is a guest object built by `InitializeFrameParams()` (`ndlib/frame-params.cpp`), with its
value word at `+0x30`. `PROSPER_POLLWATCH` gives its whole life:

```
+23.2ms    0 -> 1
+1292.4ms  1 -> 0
+1426.6ms  0 -> -1
+1486.6ms -1 -> -2
+1516.1ms -2 -> -3      <- and never again
```

Iteration *n* waits for `-(n-1)`, so the fifth waits for `-4` and the counter stops at `-3`.

`PROSPER_HWWATCH_ABS_ALLTHREADS` names the writer: **`eboot+0x13790c3`**, the job system's
counter-decrement primitive (`dec [rdx+0x30]` under the spinlock at `+0x20`), on five *different*
worker tids. So the counter is released by a **job completing**, not by the GPU — which is the same
answer the `PROSPER_RENDER=0` arm gives from the other direction.

Every job worker is parked in the work-scan loop at `0x1378508` finding all eight queues empty, so
nothing is runnable anywhere — the missing work was never enqueued, rather than enqueued and
starved.

### The wait chain, four levels deep

`PROSPER_FIBER_DUMP_MS` gives each parked fiber's exact resume `rsp`, which is what separates its
live frames from the stale bytes underneath; `tools/re/guest_stacks.py` then reads them. Two fibers
are parked deep, and together they make the whole chain:

```
main guest thread            usleep(10) poll                        counter "Err GameLoop()"        = 1
  -> FrameSpawnerJob #5      WaitForCounter at 0x5ab71c             frame-latency counter          = -3, needs -4
       -> render job #5      WaitForCounter at 0x565bb8             frame-slot counter (slot+0xc0) =  1, needs 0
            -> ???           never completes
```

A whole-heap census of the guest's counter objects (they carry a `__FILE__` pointer at `+0x00`, a
`__FUNCTION__` pointer at `+0x10`, a spinlock at `+0x20` and the value at `+0x30`) finds **12**
non-zero counters in 14.6 GiB, and every one is accounted for by the chain above plus
`AudioManager::Initialize` and three particle counters. So the thing the last job waits on is **not
a job counter** — every job counter in the process is either satisfied or one of these.

The frame-slot counter is *set* to 1 by the frame setup at `eboot+0x13939b0` and is never written
again (`PROSPER_HWWATCH_ABS_ALLTHREADS` sees writes `#1..#4` and then nothing), so the work
registered against it was never completed by anything.

### What clears it, and how far it gets

The same frame setup (`eboot+0x1391d30`, called from `FrameSpawnerJob` at `0x5ab627`) stores a
callback into the slot at `+0x80`: **`FrameCleanup`** (`eboot+0x1390ef0`, named by the string it is
registered with at `0x1390e0b`). That function takes the frame index, sets the slot's counter to 0
and wakes its waiters — it is the only writer that would release the render job.

`PROSPER_HWBP` on it, with arguments: **it runs exactly three times, for frame indices 0, 1 and 2,
and never again**, dispatched as an ordinary job (`ret=eboot+0x1378fdc`, the job dispatcher). Three
cleanups, three `GpuFlip`s, three buffer indices — and then the fourth of each is missing.

What kicks it is now measured rather than guessed. Run `PROSPER_EVLOG=1` and the breakpoint
together and they strictly alternate:

```
[ev]   GpuFlip t=0.123749 handle=0x1001 bufidx=0 ...
[hwbp] #1 rip=eboot+0x1390ef0  rdi=0x0   ret=eboot+0x1378fdc
[ev]   GpuFlip t=0.174088 handle=0x1001 bufidx=1 ...
[hwbp] #2 rip=eboot+0x1390ef0  rdi=0x1   ret=eboot+0x1378fdc
[ev]   GpuFlip t=0.191451 handle=0x1001 bufidx=2 ...
[hwbp] #3 rip=eboot+0x1390ef0  rdi=0x2   ret=eboot+0x1378fdc
```

**`FrameCleanup(N)` is driven one-for-one by the `GpuFlip` of frame N.** And the render job's wait
is on its OWN frame's slot: `PROSPER_HWBP_ARGS` at `0x565bb8` gives the counter objects in order —
`0x1500002080`, `0x1500002100`, `0x1500002180`, `0x1500002200` — and the fourth is exactly the one
the census reports still holding 1.

So the title stalls waiting for **its own fourth flip**, and there is no fourth flip because the
guest's fourth and fifth submissions carry no `Flip` packet: of ten `SubmitDcb` folds, only the
three 106-dword ones do. The guest's own submission thread agrees it is not behind — it sits in a
`usleep(1)` loop at `eboot+0x15a7cba` waiting for the "newest begun frame" global at
`eboot+0x211c278` to advance, and that global reads **5** at the stall, i.e. every frame the game
loop began has already been handed to it.

That reading was wrong on one point, and the correction moves the frontier. The guest DOES build
flips 3 and 4 — `PROSPER_HWBP` on `sceAgcDcbSetFlip`'s stub catches **five** calls, `bufidx` 0,1,2,0,1
with flipArg 0..4. Their DCBs are simply never submitted, and nothing is lost on the way in: guest
submit-wrapper calls and prosper folds agree exactly (12 = 12 in one run).

### Where the submission stops, and the guest state that gates it

The guest's own submission thread is not behind. It spins at `eboot+0x15a7cb0` on `usleep(1)`, and
its gate is a frame-slot structure (`[eboot+0x2500500]`, stride `0x6700`, 18 slots):

```
15a7cba: rax = [eboot+0x211c278]      ; newest BEGUN frame -- reads 5 at the stall
15a7cc4: jle back                      ; wait while it has caught up
15a7ccf: rax = [0x2500500] + r12*0x6700
15a7cdd: if slot.+0x48 == 0 -> back    ; three completion stamps must be present
15a7ce4: if slot.+0x50 == 0 -> back
15a7ceb: if slot.+0x58 == 0 -> back
15a7cf2: submit
```

and at the stall the slots read:

```
slot 0  +48=set  +50=set  +58=set   counter=0     <- fully complete
slot 1  +48=set  +50=set  +58=0     counter=0     <- MISSING the third stamp
slot 2  +48=set  +50=set  +58=0     counter=0     <- MISSING the third stamp
slot 3  +48=0    +50=0    +58=0     counter=1     <- never started
slot 4  +48=0    +50=0    +58=0     counter=1
slot 5  +48=0    +50=0    +58=0     counter=1
```

`+0x48` is written by the submission thread itself (`eboot+0x15a6f0a`). **`+0x58` is written by
`eboot+0x15b36b3`**, inside a loop that stamps every frame from a high-water at `eboot+0x2e1a8f8` up
to a completed-frame ordinal — and that ordinal comes from an imported call:

```
15b3229: call sceKernelGetEventData
15b3233: r14 = rax
15b3239: sar r14,0x10            ; completed frame = data >> 16
15b323d: cmp r14,[0x2e1a8f8]     ; refuse anything below the high-water
15b3266: [0x2e1a8f8] = r14 + 1   ; ...then stamp every slot up to it
```

**prosper posts `e.data = r.ident` for every AGC end-of-pipe event** (`hle_kernel_time.cpp`,
`eop_post_now`), and this title registers 58 EOP sources with idents `0x0`, `0x1` and `0x20..0x55` —
all below `0x10000`. So `data >> 16` is **always 0**: the high-water advances exactly once, to 1, and
every later event is refused. That is measured, not inferred: `[0x2e1a8f8]` reads **1** at the stall.

Whatever the right value is, a constant is not it — the consumer's arithmetic only makes sense
against a monotonic count. That is a genuine modelling gap in prosper's EOP contract, filed on its
own account.

**It is not, however, this title's blocker** — see `## Ruled out`. The consumer above is on the
**flip** queue, not the EOP one, and that is where the deadlock actually was.

### THE DEADLOCK IS BROKEN: the flip event's data must carry the flipArg in bits 16+

Reading the consumer's guard, rather than assuming which queue fed it, is what settled this:

```
eboot+0x15b31e4: call sceKernelGetEventFilter ; cmp eax,0xfffffff3   ; -13 == VideoOut, else retry
eboot+0x15b31f1: call sceKernelGetEventId     ; 1 -> retry, 3 -> …, else fall through
eboot+0x15b3229: call sceKernelGetEventData
eboot+0x15b3239: sar r14,0x10                 ; completed frame = data >> 16
```

So the ordinal comes from the **VideoOut flip event**, whose data prosper posts as the flipArg
*verbatim* (`prosper_eq_trigger_flip`). This title's flipArgs are its own small frame ordinals
(0, 1, 2, …), so `data >> 16` is always 0.

`PROSPER_FLIP_EVENT_DATA_SHIFT=1` posts `flipArg << 16` instead, and the title changes completely:

| | default | `PROSPER_FLIP_EVENT_DATA_SHIFT=1` |
| --- | --- | --- |
| game-loop iterations (25 s) | **5** | **1,552** |
| GPU flips (25 s) | **3** | **1,550** |
| over 270 s | deadlocked | **254,425 submits, 576,951 dispatches, 14,964 flips** |

The title goes from a hard deadlock at frame 5 to a live ~55 fps frame loop, completes its world
load, and reaches its intro movie. It is the single largest change this title has seen.

**It still renders nothing** — `draws_cum` stays 0 and every captured frame is uniform black — and
the reason is now a different, named gap: the title decodes its intro movie with
`libSceVdecsw`, the **GPU-compute software video decoder** (`gamelib\render\gui\movie-decoder-vdecsw.cpp`),
and all eight of its entry points are unimplemented:

```
sceVdecswQueryComputeMemoryInfo   sceVdecswQueryDecoderMemoryInfo   sceVdecswCreateDecoder
sceVdecswAllocateComputeQueue     sceVdecswSetDecodeInput           sceVdecswSetDecodeOutput
sceVdecswTrySyncDecodeInput       sceVdecswTrySyncDecodeOutput
```

so `TryFetchDecodedFrame` asserted `pNextDecodedFrame->m_frameBuffer.pFrameBuffer != NULL` forever.

### `libSceVdecsw` is implemented, and the movie DECODES

All eight entry points now answer. `libSceVdecsw` is `libSceVideodec2`'s software-decode sibling and
shares its structs — measured, not assumed: the title declares `0x18`, `0x10` and `0x30` for the
compute-memory, compute-queue and decode-input structs, which are `sizeof(VdecComputeMemory)`,
`sizeof(VdecComputeConfig)` and `sizeof(VdecInput)` exactly, and its `0x38` sync-output struct is
`sizeof(VdecOutput)`. Its decoder config is `0x50` against Videodec2's `0x48`, which
`vdecsw_adapt_config` translates.

Two corrections came out of the ASYNCHRONOUS shape, and both were caught by measurement rather than
reasoning:

- **The staged input is a queue, not a slot.** With a slot, the first access unit prosper ever saw
  carried `nal_unit_type` 1 exclusively, for the whole movie — which reads as "this stream has no
  parameter sets" and was a statement about prosper's own dropped writes. Queued, the head reads
  `7 8 6 5 5 …` (SPS, PPS, SEI, IDR slices).
- **The access unit must be COPIED at staging time.** The guest refills its bitstream buffer as soon
  as `SetDecodeInput` returns, so a pointer held until the poll is stale: the unit logged as
  SPS/PPS/SEI/IDR arrived at the backend microseconds later as slices only, and libavcodec reported
  that — correctly — as `non-existing PPS 0 referenced`. Two views of one pointer disagreeing is what
  named it.

Separately, the VA-API backend pointed libavcodec straight at guest memory; `avcodec.h` requires
`AV_INPUT_BUFFER_PADDING_SIZE` zero bytes past a packet, and the bytes after a guest access unit are
the next access unit.

Result: **61 decoded pictures out of 64 access units, 2560x1440 NV12, into the guest's own 12 MiB
frame buffer.** The title then runs on past the movie into its game-process update
(`UpdateProcesses : SaveTreasures`) and starts its **Iggy** UI runtime.

### The frontier: this title draws nothing, ever

Over a 300 s run it reaches **285,824 submits, 609,767 compute dispatches and 16,891 flips** — and
`draws_cum` is **0**, with every capture uniform black. It has no graphics draws at all; everything
it renders, it renders with compute. And the recompiler currently rejects **21** of its compute
programs — twenty as `mode=unresolved-operand`, one as `skip invalid descriptor contract`.

Nine minutes of runtime changes none of it: **494,621 submits, 1,018,573 compute dispatches, 29,283
flips, `draws_cum` 0**, one uniform-black capture.

That is the whole remaining gap, and by the charter's rule it is a fatal one rather than an
acceptable skip. These programs declare no sharps at all (`counts: ro=0 rw=0 samp=0 cbuf=0`) and
address every resource through the SRT.

**The named blocker, for whoever starts there.** Several of the rejections are
`fmt=5 op=0x4` — `s_load_dwordx16`, the bundled descriptor fetch. prosper does admit those, but only
at PCs certified by `proven_smem_x16_descriptor_loads` (`rdna2_emit_cfg.cpp:724`), and that proof is
deliberately linear:

> *"Alternate entries would require path-sensitive lifetime/provenance joins. Keep this first
> admission linear: hints, waits and barriers are transparent; every real scalar branch or indirect
> PC transfer makes the whole candidate ineligible."*

Uncharted's compute shaders have scalar control flow, so the proof bails before it examines a single
load and every x16 bundle in the title is ineligible. Admitting them needs the path-sensitive join
that comment defers — which is a design with its own tests, **not** a relaxation of the existing
guard. `tests/gpu/test_recompile_coverage.cpp:782` pins the current behaviour
(*"x16 load with an ordinary scalar/vector consumer remains fail-visible"*) and an earlier attempt in
this same investigation to widen the admission was reverted for exactly that reason.

The lever is **default OFF** on purpose. It changes a contract every title shares, prosper's current
model is pinned by `tests/hle/test_equeue_events.cpp`, and the evidence for changing it is one
title's consumer — so it waits on a cross-title snapshot pass, exactly as #2219's SDK gate does.

## Open blockers

- **#3634** — `scePlayGoGetLocus` refuses chunk ids outside prosper's discovered set and the title
  asserts on the refusal, 65 times in a 25 s run, from `gamelib\level\game-loading.cpp:1576`. Loudest
  current complaint and the natural next step. Not fatal: the guest's `int $0x41` is skipped.
- ~~**#3636** — `sceFiberOptParamInitialize` unregistered.~~ Implemented: it zeroes the 0x80-byte
  block the guest reserves for it (`lea r15,[rbp-0xe0]` against the next local at `[rbp-0x60]` at
  `eboot+0x1efe`). Measured against the loop meter afterwards: still 5 iterations, so it was a real
  gap and not this one.
- **#3623** — the import stub's host-`%fs` stash is per-guest-TCB while the value is per-host-thread.
- **#3638** — a migrated fiber that *returns* from its entry lands on the entering thread's stack.
  Latent: these fibers yield rather than return.
- **#3639** — the #3615 repair cannot detect its own failure, and its frame-offset constant has no
  automated guard.

## Ruled out

One line per hypothesis that was tested and died. Do not re-derive these.

- **"The `-1` fault is guest-TLS-related because it disappears under `PROSPER_NO_GUEST_FS=1`."**
  Falsified: that arm never reaches `NdJobWorkerThre` at all (0 mentions vs 1), dying earlier on three
  assertions, so its clean run says nothing about the fault. The conclusion happened to be *true* —
  it is a guest-TLS defect (#3615) — which is exactly why the reasoning had to be thrown away rather
  than kept: a right answer reached by a route that does not establish it is the hardest kind of wrong
  to notice. #3615.
- **"The `-1` is a field in a 64-byte job record loaded by `vmovdqu ymm1,[rdx+rax*1+0x20]` at
  `eboot+0x137868c`."** Falsified by a hardware breakpoint: **0 hits across 19 armed worker threads
  while the fault occurred**, with a positive control on the faulting instruction itself
  (`eboot+0x1378a60`) reporting hits in the same configuration. That instruction is never executed;
  the record actually comes from the drain block at `0x411379400`, reached via three far jumps that
  bypass the zeroing at `0x411378648`. The CFG had been reconstructed from a partial disassembly.
  #3615.
- **"`guest_execution_thread_enter` is never called on Linux, so `PROSPER_HWBP` cannot arm worker
  threads."** Falsified: it *is* called, at `hle_kernel.cpp:2206`, gated by `PROSPER_HWBP_ALLTHREADS`.
  Filed as #3633 and withdrawn the same hour; the grep that "established" the absence had been
  truncated by `head -6`. The surviving, much narrower point is the usage caution in the table above.
  Instrument trap 281.
- **"The shared thread pointer is installed by the fiber entry path (`call_fiber_entry`'s
  `write_fsbase(thread->guest_fs)`)."** Falsified: all 100 of those swaps are on the main thread and
  each installs that thread's *own* TCB — correct. The migration happens through the import stub's
  epilogue instead, on a plain `sceFiberRun` of an already-started fiber. #3615.
- ~~**`sceFiberSwitch` is not involved in the migration.** 0 switch events in a full run.~~
  **WITHDRAWN — the zero was the instrument.** `fiber_switch_impl` logged only its REFUSAL branches;
  a successful switch printed nothing, so "switched 6,770 times" and "never switched" produced the
  same empty evidence. With the success line added, a 22 s run logs **6,770 switches**, still going
  when the capture ends. The narrower claim that survives is the one #3615 actually rested on: the
  guest-TP migration observed there arrives through `sceFiberRun` with `started=1`, which the
  fiber-run log does record. Instrument trap 282.
- **"The AGC EOP event's data should carry the completed-FRAME ordinal (the guest's own flipArg) in
  bits 16+."** Implemented behind `PROSPER_EOP_FRAME_ORDINAL=1` and falsified. The lever demonstrably
  moves — its trace reports the ordinal advancing `0 -> 1 -> 2` as the three flips complete, and the
  accessor is linked — and the title still runs exactly 5 game-loop iterations and 3 flips, with the
  guest's high-water still stuck at 1. So either the consumer at `eboot+0x15b3229` is fed by a
  different event source than the AGC EOP queue, or bits 16+ carry something other than the flip
  ordinal. The lever stays, default OFF, so the A/B remains reproducible. #3616.
- **"The stall is the Vulkan backend."** Falsified: `PROSPER_RENDER=0` reproduces it exactly — 5
  game-loop iterations, the same counter trajectory, the same ~1.5 s. The 14 compute programs
  skipped as `mode=unresolved-operand` are therefore not the cause either, however much they need
  fixing on their own account. **This arm does NOT clear prosper's flip/EOP emulation**, which runs
  in both arms — see the note under the headline.
- **"It is the SDK-gated post-submit completion contract (#2219)."** This title requests **SDK 9**,
  so the gate is closed for it and the shape fits *ArcRunner* and *Crisis Core* exactly. It is still
  not the cause: `PROSPER_POST_SUBMIT_VISIBILITY=1` gives 5 iterations, against 5 for each of two
  baselines and 5 for `PROSPER_EOP_WATCHDOG_MS=200`.
- **"`sceAgcDriverRegisterWorkloadStream` returning 0 without an out-parameter starves the graphics
  path."** Falsified at the call site: the guest calls it once as `(1, "Entire Frame")` from its
  render init and **drops the result** — it is a Razor workload-stream NAME, not a handle the title
  keeps. Registering it honestly is still worth doing; it will not move this title.
- **"The title never reaches an AGC submit entry point."** Withdrawn — it was measured by a route
  that could not establish it (the unimplemented-NID census, which by construction says nothing
  about a NID that IS implemented). `PROSPER_GFXLOG=1` shows 10 `SubmitDcb` and 33 `SubmitAcb` folds
  in 25 s. What is true, and is the part worth keeping, is that **none of them contains a draw**.

## History

- **2026-09-14 (later)** — the frontier is located to one wait: five game-loop iterations, then
  `GameLoopUpdate` parks in the frame-latency `WaitForCounter` at `eboot+0x5ab71c`. Renderer ruled
  out by a `PROSPER_RENDER=0` arm. Rung unchanged.
- **2026-09-14** — `PROSPER_RESTORE_PATCHED_IMPORTS` removes the PlayGo blocker: 7 overwritten
  import stubs restored, `CheckPlayGoStatus` assertions 64 -> 0, `scePlayGoGetLocus` serviced 1,059
  times, archives unmounted 3 -> 1, and the guest reaches its world switch. Rung unchanged.
- **2026-09-14** — #3615 fixed (fiber guest-TP migration). Guest goes from faulting ~1 s into boot to
  surviving a full capture and loading content. Rung unchanged.
- Earlier — guest mutex ownership resolved on the calling host thread's TCB (`3c4e4db9a`), the first
  half of #3615.
