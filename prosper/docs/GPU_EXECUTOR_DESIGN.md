# GPU executor design — from submitted Dcb to on-screen frames

**Date:** 2026-07-06. **Goal:** execute the game's submitted AGC command buffers on a real Vulkan device
and signal GPU completion so the Unity render loop advances past its current stall to actual frames.
Grounded in two references in the parent folder: **shadPS4** (`src/video_core/amdgpu/liverpool.cpp`,
`src/core/libraries/gnmdriver/gnmdriver.cpp` — mature, PS4/Gnm) and **Kyty** (`source/emulator/src/Graphics/*`
— early, PS5/AGC-native; treat as a hint, cross-check against shadPS4).

## The completion mechanism (cross-validated by BOTH references)
A submit finishes and the CPU learns about it through **end-of-pipe (EOP) events**:
1. The Dcb contains a `RELEASE_MEM` / `EVENT_WRITE_EOP` packet: on GPU pipe-drain it **writes a fence value
   to a memory label** and optionally **raises a GPU interrupt**.
   - shadPS4 `liverpool.cpp`: `EventWriteEop` → `event_eop->SignalFence(...)` (the label write) + a lambda
     `IrqC::Signal(InterruptId::GfxEop)`; `ReleaseMem` → `SignalFence` + `IrqC::Signal(pipe_id)`.
   - Kyty `GraphicsRun`: the ReleaseMem packet dispatches to `GraphicsRenderWriteAtEndOfPipeWithInterrupt*`
     (label write) which, when a submit's work drains, calls `RenderContext::TriggerEopEvent()`.
2. The interrupt is routed to a **kevent** the app registered: `eq->TriggerEvent(ident=GfxEop=0x40,
   filter=EVFILT_GRAPHICS, udata)`. (shadPS4 `equeue.cpp::TriggerEvent`; Kyty `KernelTriggerEvent`.)
3. The app registered that source with **`sceGnmAddEqEvent`/`GraphicsAddEqEvent` (NID `b0xyllnVY-I`,
   id=`0x40`)** and a thread does `sceKernelWaitEqueue` on that queue; the trigger wakes it.
4. Flips are the same shape: a flip `RELEASE_MEM` variant / `SubmitFlip` presents the buffer and posts a
   **flip** kevent (shadPS4 `videoout/driver.cpp` `equeue->TriggerEvent(...)` on `GfxFlip`).

So completion = **{write the fence label} + {trigger the registered EOP/flip kevent}**. Both refs agree.

## Where prosper already is (the halves that exist)
- **Front half (done):** `agc_driver_submit_dcb` (NID `UglJIZjGssM`) replays the Dcb via
  `gpu::run_command_buffer` into a persistent `GpuState` (register files + draw list). The Dcb builders
  (`hle_agc.cpp`) capture the real packet args incl. ReleaseMem `dstGpuAddr`+value and the Flip.
- **Back half (done, offscreen):** `render_state`/`vk_translate` resolve a `GpuState` → Vulkan pipeline
  state; the recompiler turns the real shaders → SPIR-V (38/41, ~95%); `tests/render_runner.h` renders a
  `GpuState` to pixels on llvmpipe. `videoout_present.cpp` has `present_write_frame`/`present_readback`.
- **Missing:** a *live, persistent* device that executes the accumulated draws at submit time and presents,
  plus the **completion signaling** that both refs describe.

## The prosper-specific twist (must drive the design)
Our target (The Messenger) does **NOT** follow the textbook EOP-equeue path:
- It creates an "EOP QUEUE" equeue but **never registers an EOP event** on it (`b0xyllnVY-I` is never
  called — not in the unimplemented-call log, and we don't hook it) and **never `WaitEqueue`s** on it.
- It DOES register **flip** events (`sceVideoOutAddFlipEvent`) and its flip thread (eboot+0x14bd47f)
  successfully consumes ~1167 flip events from our pump.
- The stalled producers wait on **sync_on_address semaphores** (main/PreloadManager @0x18a83b5,
  GfxDevice @0xb0672a, worker @0x9385d7), NOT on the EOP equeue or a label poll we could see.

Interpretation: with the real AGC driver absent (HLE'd), the driver-internal step that converts "GPU submit
completed" into "post the Unity semaphore / flip-done" is **missing**. On hardware the AGC driver owns that;
in prosper the executor must emulate it. The earlier fence-label-write and EOP-queue-event shims failed
because (a) no CPU thread polls those labels and (b) nothing was registered on the EOP queue — consistent
with "the app expects the driver to do the completion plumbing".

## Design — staged, each stage independently testable
**Stage A — live device + execute on submit (no behavior risk; pure back-half wiring). DONE.**
Implemented as the Vulkan-agnostic core `execute_gpustate()` (`src/gpu/gpu_execute.hpp`) plus the live-submit
registry `set_submit_renderer`/`execute_and_present` (`src/gpu/gpu_executor.cpp`): whoever owns a device (the
runtime binary at startup, or a test) registers a `LiveRenderFn`; `agc_driver_submit_dcb` calls
`execute_and_present(state, present_width(), present_height())` after folding, gated on
`have_submit_renderer() && draws>0`, so it is inert on the game path until a device is wired yet fully
unit-tested (`test_gpu_execute`: GpuState → recompile+resolve+render → GREEN frame → `present_readback`
byte-for-byte, via the registry path). Deliberately kept `prosper_core` Vulkan-free (backend is a
`std::function`). Still open: promote a *persistent* device in the runtime binary and target `CB_COLOR0_BASE`
directly (currently the backend renders to its own attachment at videoout resolution).

**Stage B — honor the Dcb's memory-side effects (correct EOP semantics). DONE.**
`GpuState::apply` performs the writes the Dcb requests, since our CommandProcessor folds a submit
synchronously (GPU "done" the instant SubmitDcb returns → this IS the end-of-pipe moment):
- `RELEASE_MEM`/`EVENT_WRITE_EOP` (`honor_eop_write`): write the fence value to the label address, honoring
  `data_sel` (1=32-bit value, 2=64-bit value, 3=64-bit monotonic GPU clock). The AGC ABI is pinned to Kyty
  `GraphicsCbReleaseMem` (`buf, action, gcr_cntl, dst, cache_policy, address, data_sel, data, …`) — this
  resolved the earlier LOW-confidence "which arg is the value" (a5=address, stack arg 7=data_sel, stack arg
  8=the 64-bit value). `agc_cb_release_mem` now lays out `[0..1]=addr [2]=data_sel [3..4]=value [5]=action`.
- `WRITE_DATA` (`honor_write_data`): copy the inline dwords to the destination; `agc_dcb_write_data` now
  copies the real `data*`/`num_dwords` (Kyty `GraphicsDcbWriteData`) into the packet. `WAIT_REG_MEM`: no-op
  (the label is already written → condition satisfied).
Correct end-of-pipe semantics, on by default; `PROSPER_NO_EOP_WRITE=1` disables for bisection. Verified by
`test_eop_write` (data_sel 1/2/3 + WRITE_DATA + overflow-clamp). Replaces the old `PROSPER_AGC_FENCE` scaffold.

**Stage C — completion signaling + emulate the driver's post. (EOP hook DONE; game-crux is Stage D.)**
- DONE: hook `GraphicsAddEqEvent` (`b0xyllnVY-I`, `g_gnm_add_eq_event`) records `(eq, id, udata)` as an EOP
  source (`prosper_eq_add_eop`), and `agc_driver_submit_dcb` fires `prosper_eq_trigger_eop()` on each
  completed submit → posts `SceKernelEvent{ident=id, filter=GraphicsCore(-14), data=id, udata}` to every
  registered equeue (values pinned to shadPS4 `sceGnmAddEqEvent` + `equeue.h`). Harmless/inert if the game
  never registers one. Verified by `test_equeue_events` (register id=0x40 → submit → `WaitEqueue` returns
  the GfxEop event with the right ident/filter/data/udata).
- Since our game uses the flip path + semaphores, also ensure `SubmitFlip` posts a real flip-completion
  (present + flip kevent) — already partially done via the ~60 Hz pump; make it fire on the actual submit.
- The crux experiment (Stage D, needs the boot): determine what posts the PreloadManager semaphore on
  hardware. Instrument which thread would consume the EOP/flip event and post @0x18a83b5. Likely the
  GfxDevice render thread (0xb0672a) waits for upload completion; once the executor signals it, it posts
  main's semaphore. This is the one piece the EOP hook above does NOT cover for our specific target.

**Stage D — close the loop on the game.** With A–C, boot and watch the stall: does the GfxDevice thread
advance, submit draws (SubmitDcb with draws>0), and present? Iterate on the exact completion the producer
polls (add a HW read-watchpoint on the semaphore's backing word to find its writer). First real frame =
whatever the setup submit + first draws present.

## Verification (agentic-first, per project norm)
- Stage A/B: extend `test_agc_submit` — build a Dcb (clear + one triangle + ReleaseMem), submit, assert the
  rendered pixels (readback) AND the fence label was written. Runs on llvmpipe, no game needed.
- Stage C: unit-test the EOP-event trigger (register on an equeue, submit, assert `WaitEqueue` returns the
  0x40 event) — extends `test_equeue_events`.
- Stage D: the game boot itself (PROSPER_GFXLOG/EVLOG) — SubmitDcb-with-draws + a presented frame.

## Open question to resolve first (cheap, high-value)
Confirm the game's completion wiring before building Stage C's trigger: run the boot and check whether ANY
thread ever registers/wai­ts an EOP or flip event tied to the *upload* completion, vs. polling a label. A HW
read-watchpoint on the PreloadManager semaphore word (`PROSPER_HWWATCH` extended to reads) names its writer
— that writer is the exact producer we must drive. That pins Stage C precisely instead of guessing.
