# Dead Cells Progression Matrix

Last updated: 2026-07-15

This matrix separates guest progression from renderer output and frontend behavior. It exists because a
native-speed or sampled-render run reaching `PrisonStart` does not prove that the full synchronous renderer
reaches the same state. Track the current full-render stall in
[#723](https://github.com/mattias800/ps5ys/issues/723) and the native-Windows render-disabled divergence in
[#768](https://github.com/mattias800/ps5ys/issues/768).

## Run It

Build `boot_trace`, `prosper-app`, and `gpu_timeline` from one commit, then run in native Windows PowerShell:

```powershell
cmake --build prosper/build-mingw-app --target boot_trace prosper-app gpu_timeline -j 8
prosper/scripts/dead-cells/progression-matrix.ps1
```

Use `-Mode headless-full` to run one row, `-BuildDir PATH` for another build, and `-OutputDir PATH` to retain
evidence at a chosen location. Every row gets a fresh save root, pad log, progress heartbeat, GPU timeline,
stdout/stderr logs, and an offline selector for the current native-Windows gameplay semantic signature:

```text
target 636x420, target draw index 77:85, total draws 91:94, dispatches 8
```

The script writes `summary.json`. `termination=timeout_killed` means the row remained alive until the configured
observation window and was intentionally stopped; its `exit_code` may therefore be null. `termination=process_exit`
retains the real process exit code. A selector match proves that the guest submitted the known gameplay-shaped
work; it does not by itself prove correct pixels. New image baselines still require human inspection.

The matrix is deliberately a native-Windows runner. Compare it with a WSL/Linux run from the same source tree
before describing a failed no-graphics row as a code regression. On 2026-07-15 the exact tree that remained in a
native-Windows 378-draw loading workload reached the sustained 92-94 draw / 8-dispatch gameplay signature under
WSL. A source bisect that mixed those hosts produced a false first-bad result. The host split was subsequently
root-caused to `sceSaveDataDialogUpdateStatus`: returning generic success meant status `NONE`, which Dead Cells
polled forever. The lifecycle HLE in #768 restores native-Windows progression in the no-GPU-work control.

Post-#767/#769 validation on 2026-07-15 reached the selector in both native-Windows diagnostic rows. With compute
disabled, the first match was at 28.35 seconds and the 38-second run contained 1,104 matches. With live compute
enabled, the first match was at 30.67 seconds and the run remained alive through the same timeout. This establishes
that SaveDataDialog progression is independent of rendering, while current compute execution no longer exits at
its first live dispatch. It does not resolve the full synchronous-render throughput problem tracked by #723.

## Rows And Semantics

| Row | Graphics work | Publication | Frontend | Correctness use |
| --- | --- | --- | --- | --- |
| `headless-full` | Every retained draw and dispatch | Every completed frame | `boot_trace`, paced silent audio | Full-render reference |
| `app-full` | Every retained draw and dispatch | Every completed frame | SDL video/audio/input | Compare frontend-only behavior |
| `headless-publish-10` | Every retained draw and dispatch | One in ten completed frames | `boot_trace` | Isolate publication/presentation pressure |
| `headless-render-every-10` | One in ten graphics submits; compute still runs | Every rendered frame | `boot_trace` | Diagnostic only |
| `headless-no-graphics` | No graphics backend; compute still runs | None | `boot_trace` | Guest/HLE progression diagnostic only |
| `headless-no-gpu-work` | No graphics or compute execution | None | `boot_trace` | Separate compute throughput from guest/HLE progression |

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
when synchronous graphics work is absent. These diagnostic rows use the capture-safe route whose Circle hold lasts
through 300 seconds, so a slow transition cannot outlive the scripted skip input.

`headless-no-gpu-work` additionally sets `PROSPER_NO_COMPUTE=1`. The timeline still records semantic dispatches,
but their writes do not occur. This row is not a correctness oracle: use it only when `headless-no-graphics`
remains slow or stuck and the question is whether synchronous compute throughput is consuming the guest submit
thread. If it progresses, repeat with compute enabled before blaming a non-GPU HLE contract.

## Interpreting Divergence

1. If `headless-full` and `app-full` diverge, investigate frontend audio/input/dialog/presentation behavior.
2. If both full rows stall but `headless-publish-10` progresses, publication or frontend frame consumption is
   implicated even though GPU execution is unchanged.
3. If all full-execution rows stall while sampled/no-graphics rows progress, the boundary is synchronous GPU
   execution, guest pacing, or completion ordering. Skipping graphics has not proven a rendering bug.
4. If `headless-no-gpu-work` progresses but `headless-no-graphics` does not, isolate live compute cost or side
   effects before investigating unrelated guest synchronization.
5. Compare the first `Loading level PrisonStart`, `PARSEALL TOOK`, last `[progress]`, and semantic selector match.
   Do not infer a deadlock solely from a low FPS counter.
6. If native Windows fails even in `headless-no-graphics`, compare the same source under WSL/Linux. #768 is the
   reference example: a system-dialog HLE poll looked like an endless loading render loop.

`boot_trace` historically wrote periodic BMPs whenever `PROSPER_RENDER=1`; the matrix sets
`PROSPER_NO_FRAME_DUMPS=1` so headless-full is not penalized by tool-only disk I/O.
