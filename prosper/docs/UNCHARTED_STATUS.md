# Uncharted: Legacy of Thieves Collection (PPSA05684) — status

**Rung 0 as of 2026-09-14.** The guest boots, runs its job system, mounts its content archives,
starts its world load, and executes **exactly five iterations of its game loop** before parking.
Nothing renders. Tracker: #3616.

**The renderer is not the frontier on this title, and that is measured rather than argued:** the
stall reproduces identically with `PROSPER_RENDER=0` (same five iterations, same wait, same
timing). Whatever holds this title is CPU-side — HLE, sync, or the job system — so a GPU
hypothesis needs new evidence before it is worth spending a run on. See `## Ruled out`.

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
- **"The stall is the renderer / the GPU / the flip path."** Falsified: `PROSPER_RENDER=0`
  reproduces it exactly — 5 game-loop iterations, the same counter trajectory, the same ~1.5 s. The
  14 compute programs skipped as `mode=unresolved-operand` are therefore not the cause either,
  however much they need fixing on their own account.
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
