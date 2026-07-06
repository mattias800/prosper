# The render-loop frontier — post-boot-wall (2026-07-06)

**Status:** open (the next big frontier). The GfxDevice boot wall is RESOLVED (see
`GFXDEVICE_BRINGUP_PROBLEM.md`); the game now boots into asset loading + its frame loop but does not yet
render. This documents exactly where it is and what's needed, so the next session picks up cleanly.

## Where the game is now
- Boots through IL2CPP → GfxDevice bring-up → **loads `level0` + scene assets** (`sharedassets0.assets`,
  `unity_builtin_extra`, `globalgamemanagers`) → initializes audio (Ajm/AudioOut) → enters its frame loop.
- **Live and busy but not rendering:** ~48 threads idle in `k_wait_on_address` (job pool), a scheduler
  thread ping-ponging semaphores, an audio-mix thread (float accumulate at eboot+0x195d8e0). No draws.
- The **TRC R5089 "Forcing sce::Agc::suspendPoint" watchdog** (eboot+0x14eb200) fires periodically because
  no frame is presented (`(frame_counter − last_suspend) ≥ 3` forces a suspendPoint).

## What was fixed this session (render-loop heartbeat)
- The game drives frames via the **kqueue/kevent** model: `sceKernelCreateEqueue` ×4 ("Flip Event Queue
  GfxDeviceAgc", "EOP QUEUE", "UnityFTMFlipQueue", "eq to wait flip"), `sceVideoOutAddFlipEvent`, then
  `sceKernelWaitEqueue` for flip-completion events. Our equeue was fully stubbed → those queues never got
  events → two threads spun 18000× in WaitEqueue → no frame advanced.
- **FIXED (committed):** real equeue backend (`hle_kernel_time.cpp`) + `sceVideoOutAddFlipEvent/AddVblankEvent`
  registration + a ~60 Hz pump posting flip/vblank `SceKernelEvent`s + a blocking `WaitEqueue`. The
  flip-timing thread (eboot+0x14bd450) now cycles at ~60 Hz; the watchdog fires ~half as often. Gated
  `PROSPER_EVLOG` traces the handshake.

## The actual blocker (precisely located)
The **main thread is parked in Unity's PreloadManager** (the singleton at `eboot+0x1ff5228`, string
"Loading.PreloadManager"), in a **semaphore-gated work-queue consumer** entered via GfxDevice
`0xaee233 → 0xb06940`: `lock xadd -1,[singleton+0x130]` (count); if empty, block on the sem object
`[singleton+0x128]` via the generic wait helper `0x18a82a0` (a `sync_on_address` semaphore-with-timeout).
- **Verified with a HW write-watchpoint:** the count word is only ever decremented by the consumer itself
  — **no producer ever enqueues.** The main thread times out and re-waits forever.
- **Chain:** main thread → PreloadManager finishing `level0` → GfxDevice **GPU-resource upload** (create
  textures/buffers/pipelines for the loaded assets) → waits for the upload to complete on the GPU/render
  thread → never signaled (we don't execute GPU commands or post completion). No `SubmitDcb` fires, so the
  game is stalled *before* rendering, in the asset-upload step.

## What's needed next (the M4/M5 GPU-execution frontier)
Getting past `level0` load — and then to actual frames — requires **real AGC→Vulkan GPU command execution
+ completion signaling**, not another small fix:
1. **Execute submitted command buffers for real** (the back-half `run_command_buffer → GpuState` exists and
   renders offscreen in tests; wire it to a live Vulkan device + upload/execute the game's actual submits).
2. **Post GPU end-of-pipe (EOP) completion events** to the "EOP QUEUE" when a submit completes, so the
   GfxDevice upload wait returns and the PreloadManager producer enqueues the main-thread job.
3. Then the render loop should cycle: submit draws → EOP → flip → present → next frame.
- **Note:** the exact EOP registration mechanism isn't observable yet (the game submits nothing pre-upload),
  so implementing EOP requires either reaching the first submit or Gnm/AGC reference for the EOP-label +
  equeue-event association.

## Tooling added this session (all in `exec_image_linux.cpp`, gated, non-destructive)
- `PROSPER_HWBP=0xOFF` — race-free x86 HW execute breakpoint (perf_event_open) that logs registers +
  caller; safe on hot/multi-threaded code where the int3 logger races.
- `PROSPER_HWWATCH[=delta]` + `PROSPER_HWWATCH_REG=<reg>` — chained HW data write-watchpoint on
  `[reg+delta]`, armed on the first exec-bp hit; logs each writer's RIP + a `/proc/self/maps` classifier.
- `PROSPER_EVLOG` — traces the equeue + `sceVideoOut*` flip/vblank calls (how the handshake was mapped).

## Recommendation
The render milestone is the GPU-execution frontier — a large, multi-session build best done deliberately
(and it can't be fully validated until the game issues its first real submit). Meanwhile, reliable
committable progress toward full-game rendering = expanding RDNA2→SPIR-V recompiler coverage on the game's
real shaders (each spirv-val + execution verified). When GPU execution lands, more covered shaders = more
of the game renders.

## 2026-07-06 (overnight) — hard-stall confirmed; no easy crack; needs GPU execution
Exhaustively re-probed the stall. Findings:
- The last new AGC call before the stall, `libSceAgc::h9z6+0hEydk`, is called from `eboot+0x14eb1f2` —
  inside the TRC watchdog: it **IS `sce::Agc::suspendPoint`** (a per-frame checkpoint), not a blocker.
- "path … not considered suitable for apr reads" is a **benign** Unity async-page-read decision — it
  prints for files that loaded fine (globalgamemanagers etc.), not a stall indicator.
- gdb thread dumps during the stall show **every thread idle-waiting** (futex / nanosleep / audio-mix /
  scheduler) with **no thread in a read syscall** — the load is not actively progressing; it's a quiescent
  stall, and the PreloadManager work-queue producer never runs (count only ever decremented by the
  consumer). So it is NOT a lost-signal a punch could safely fix — the item is never produced (a punch
  would dequeue a phantom → crash).
**Conclusion:** the render-loop unblock is not crackable by AGC-call/file/semaphore shims in autonomous
iterations. It genuinely requires the **GPU-execution build** (execute submitted command buffers on a live
Vulkan device + write back GPU memory + post EOP/flip completion so Unity's asset-integration/upload
choreography advances) — a large, focused, likely reference/interactive effort — OR a deliberate
punch-through experiment done with a human watching each cascade. This is the clear next milestone for the
game's actual pixels; the recompiler and the offscreen `GpuState→frame` spine are both ready to receive
real draws the moment it lands.
