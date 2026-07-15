# Dead Cells Progression Matrix

Last updated: 2026-07-15

This matrix separates guest progression from renderer output and frontend behavior. It exists because a
native-speed or sampled-render run reaching `PrisonStart` does not prove that the full synchronous renderer
reaches the same state. Track the current full-render stall in
[#723](https://github.com/mattias800/ps5ys/issues/723).

## Run It

Build `boot_trace`, `prosper-app`, and `gpu_timeline` from one commit, then run in native Windows PowerShell:

```powershell
cmake --build prosper/build-mingw-app --target boot_trace prosper-app gpu_timeline -j 8
prosper/scripts/dead-cells/progression-matrix.ps1
```

Use `-Mode headless-full` to run one row, `-BuildDir PATH` for another build, and `-OutputDir PATH` to retain
evidence at a chosen location. Every row gets a fresh save root, pad log, progress heartbeat, GPU timeline,
stdout/stderr logs, and an offline selector for the current gameplay semantic signature:

```text
target 738x420, target draw index 80:82, total draws 91:93, dispatches 8
```

The script writes `summary.json`. A selector match proves that the guest submitted the known gameplay-shaped
work; it does not by itself prove correct pixels. New image baselines still require human inspection.

## Rows And Semantics

| Row | Graphics work | Publication | Frontend | Correctness use |
| --- | --- | --- | --- | --- |
| `headless-full` | Every retained draw and dispatch | Every completed frame | `boot_trace`, paced silent audio | Full-render reference |
| `app-full` | Every retained draw and dispatch | Every completed frame | SDL video/audio/input | Compare frontend-only behavior |
| `headless-publish-10` | Every retained draw and dispatch | One in ten completed frames | `boot_trace` | Isolate publication/presentation pressure |
| `headless-render-every-10` | One in ten graphics submits; compute still runs | Every rendered frame | `boot_trace` | Diagnostic only |
| `headless-no-graphics` | No graphics backend; compute still runs | None | `boot_trace` | Guest/HLE progression diagnostic only |

`headless-full` and `app-full` share the loader, HLE, command folding, live renderer, and synchronous GPU
execution. Their intended differences are SDL window/swapchain consumption, real audio output, physical input,
and platform-dialog integration. They should reach the same semantic state under the same scripted input.

`PROSPER_PRESENT_EVERY=N` changes only `present_write_frame`: the live renderer executes every selected submit,
persistent render targets advance normally, and a valid output frame is simply not published on suppressed
iterations. This is the safe row for asking whether frontend publication is responsible for a progression
difference. It is not expected to improve a workload dominated by synchronous Vulkan execution or readback.

`PROSPER_RENDER_EVERY=N` is fundamentally different. On unselected draw submits it skips graphics realization
and Vulkan execution, while separately executing retained compute dispatches. It can change temporal render
targets, synchronization pressure, guest pacing, and frame-count timing. Never treat it as a correctness oracle.
The no-graphics row is even less representative; it is useful only to show that guest/HLE logic can progress
when synchronous graphics work is absent. Both diagnostic rows use the capture-safe route whose Circle hold lasts
through 300 seconds, so a slow transition cannot outlive the scripted skip input.

## Interpreting Divergence

1. If `headless-full` and `app-full` diverge, investigate frontend audio/input/dialog/presentation behavior.
2. If both full rows stall but `headless-publish-10` progresses, publication or frontend frame consumption is
   implicated even though GPU execution is unchanged.
3. If all full-execution rows stall while sampled/no-graphics rows progress, the boundary is synchronous GPU
   execution, guest pacing, or completion ordering. Skipping graphics has not proven a rendering bug.
4. Compare the first `Loading level PrisonStart`, `PARSEALL TOOK`, last `[progress]`, and semantic selector match.
   Do not infer a deadlock solely from a low FPS counter.

`boot_trace` historically wrote periodic BMPs whenever `PROSPER_RENDER=1`; the matrix sets
`PROSPER_NO_FRAME_DUMPS=1` so headless-full is not penalized by tool-only disk I/O.
