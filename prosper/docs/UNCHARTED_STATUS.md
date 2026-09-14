# Uncharted: Legacy of Thieves Collection (PPSA05684) — status

**Rung 0 as of 2026-09-14.** The guest boots, runs its job system, and mounts its content archives.
Nothing renders. Tracker: #3616.

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

## What works

- Boots and survives a 25 s capture with no faults, across 13 worker threads and 100 fibers.
- Mounts its real content archives: `bin.psarc`, `shaders.psarc`, `data.psarc`, `core.psarc`,
  `combo-common.psarc`, `common.psarc`, `sp-common.psarc`, `world-boat-intro.psarc`.

## Open blockers

- **#3634** — `scePlayGoGetLocus` refuses chunk ids outside prosper's discovered set and the title
  asserts on the refusal, 65 times in a 25 s run, from `gamelib\level\game-loading.cpp:1576`. Loudest
  current complaint and the natural next step. Not fatal: the guest's `int $0x41` is skipped.
- **#3636** — `sceFiberOptParamInitialize` unregistered; the `return 0` default reports success
  without writing the out-parameter.
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
- **`sceFiberSwitch` is not involved in the migration.** 0 switch events in a full run; the
  work-stealing goes through `sceFiberRun` with `started=1`. #3615.

## History

- **2026-09-14** — #3615 fixed (fiber guest-TP migration). Guest goes from faulting ~1 s into boot to
  surviving a full capture and loading content. Rung unchanged.
- Earlier — guest mutex ownership resolved on the calling host thread's TCB (`3c4e4db9a`), the first
  half of #3615.
