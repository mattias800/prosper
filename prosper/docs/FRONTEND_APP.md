# prosper-app — the OS-integration frontend (design)

**Status:** **the app runs the game in a window, with audio out and controller in** (issue #164).
`prosper-app --dump <app0>` (built with `-DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON`)
boots the title, composites its GPU submits to an SDL3 window on the desktop, routes `sceAudioOut` to
the host via SDL3, and feeds a host controller into `libScePad`. P0 (present) + P0b (shared boot) +
the shared renderer (#177) + P1 (audio) + P2 (controllers) are landed; P3 is polish.

## What it is

`prosper-app` is a **separate frontend binary** that turns the headless prosper core into a real
desktop app: a window showing the game, audio out, controller in, and close-to-quit. It exists so
we can watch prosper run on a real GPU with sound — a human-perspective smoke test — **without
compromising the headless-first core**, which stays the source of truth for agentic/CI verification.

The frontend runs both as a Linux application (including through WSLg) and as a native MinGW Windows
application. Native Windows uses SDL3's Win32/WASAPI/XInput-HIDAPI backends and a Vulkan SDK; it boots
the same guest and renderer directly, without WSL. See `WINDOWS_PORT_HANDOFF.md` for the current build
and routed screenshot validation commands.

## First principle: the dependency arrow points one way

```
prosper_core   (static lib — ZERO windowing/surface/OS-device deps; headless; CI-tested)
     ▲ links
     │      audio_set_sink() · pad_set_backend() · present_* readback · run/request_stop()
frontends/prosper-app   (new binary — owns ALL OS integration, via SDL3)
```

The core never links SDL, a window system, or a presentation surface. The frontend depends on the
core; the core never depends on the frontend. Delete `frontends/` and CI is unaffected. This makes
"separate" and "headless-first" structural guarantees, not conventions to remember.

## The seam contract (three of four already exist)

The core already exposes exactly the injection points a frontend needs. The frontend *supplies*
implementations; the core calls them.

### 1. Audio — `AudioSink` (exists, `src/hle/audio.hpp`)
```cpp
struct AudioSink {
    virtual bool open(int port, const AudioPortInfo& info);        // configure a port
    virtual void output(int port, const void* pcm, int frames) = 0;// one interleaved-PCM grain
    virtual void set_volume(int port, uint32_t mask, const int* vols);
    virtual void close(int port);
};
void audio_set_sink(AudioSink*);   // nullptr restores the built-in silent/real-time sink
```
Frontend supplies an **SDL3 audio sink** (queue each `output()` grain to an SDL audio stream). The
`feat/audio-sdl3` branch is already most of this — coordinate with it, don't duplicate.

### 2. Controllers — `PadBackend` (exists, `src/input/pad.hpp`)
```cpp
struct PadBackend {
    virtual bool poll(int index, HostPadState& out) = 0;  // called on the guest's input thread; MUST be thread-safe
};
void pad_set_backend(PadBackend*);   // nullptr restores the neutral (no-device) backend
```
Frontend supplies an **SDL3 `GameController` backend**, mapping SDL buttons/axes → `HostPadState`.
The pad header already documents frontends installing this from the harness.

### 3. Video — the present readback (exists, `src/gpu/videoout_present.hpp`)
```cpp
bool     present_has_frame();
uint32_t present_width();  uint32_t present_height();
uint64_t present_count();                 // total flips presented (advances on guest flip)
size_t   present_readback(void* dst, size_t dst_cap);   // copy the presented frame's pixels (w*h*4)
```
On each guest flip the renderer writes the finished frame into the scanout buffer; `present_readback`
hands out those **CPU pixels**. The frontend polls `present_count()` for a new frame and blits the
readback into its swapchain. This is the whole video boundary — the frontend is a pure *consumer* of
finished frames.

### 4. Lifecycle — stop request exists; guest consumption remains
The frontend has the minimal, headless-agnostic control:
```cpp
// in the run harness / a small host lifecycle header
void prosper_request_stop();     // idempotent; sets a flag the guest run-loop checks
bool prosper_stop_requested();
```
`boot_trace` keeps its fixed-budget behavior for CI and the frontend calls `prosper_request_stop()`
on window close. `run_entry` does not consume the flag yet, so the current app cannot safely join a
booted guest. It flushes logs and calls `std::_Exit` instead; returning from `main` after detaching the
guest caused a reproducible Windows access violation when static teardown raced live guest threads.
The remaining lifecycle work is a guest flip-boundary check followed by a real join and normal
teardown (#352).

## The Vulkan-context decision: two contexts, frames cross as CPU pixels

**The frontend owns its own Vulkan context (instance + device + swapchain on the window surface).
The core keeps its existing headless render device. Frames cross the boundary as CPU pixels via
`present_readback`.** Rationale:

- The core's render device stays **surface-free and unchanged** — no swapchain dependency leaks into
  the renderer, no `#ifdef`, headless + CI identical.
- The seam already emits CPU pixels, so the frontend is a pure consumer — the cleanest boundary, and
  the same seam becomes a shared-memory frame ring if the two are ever split into separate processes.
- Cost = one readback + one upload per frame. The Messenger is 2D at 1080p (~8 MB/frame) — negligible.

**When to revisit:** only if a future 3D-heavy title makes the per-frame copy a real bottleneck. Then
unify on a single frontend-provided device and share the render target via `VK_KHR_external_memory`
(zero-copy) — but keep `present_readback` as the abstraction so headless still works. Not now.

## Present loop (sketch)

```
init:  SDL_Init(VIDEO|AUDIO|GAMECONTROLLER); create window;
       create VkInstance(+VK_KHR_surface, win32/wayland), VkSurfaceKHR, pick device, swapchain,
       one staging buffer + one sampled VkImage sized to present_width()×present_height();
       audio_set_sink(&sdl_sink); pad_set_backend(&sdl_pads);
       start the guest run-loop on its own thread.

frame: SDL_PollEvent → on SDL_QUIT / window-close: prosper_request_stop(); break;
       if present_count() advanced since last shown:
           n = present_readback(staging_mapped, cap);
           upload staging → VkImage; blit/scale VkImage → acquired swapchain image; vkQueuePresentKHR;
       else: small sleep / wait on a frame condvar to avoid spinning.

quit target: prosper_request_stop(); join guest thread; destroy swapchain/device/window; SDL_Quit().
quit today:  prosper_request_stop(); flush logs; direct process exit while the guest is still live.
```
Handle swapchain resize (`present_width/height` change or window resize → recreate). Vsync via
`VK_PRESENT_MODE_FIFO`.

## Threading

- **Guest run-loop thread**: drives the guest (as `boot_trace` does), renders into scanout on flip.
- **Main/UI thread**: SDL event pump + swapchain present. It only *reads* via `present_readback`
  (already mutex-guarded: renderer writes, present reads) — no new shared state.
- Audio/pad callbacks run on whatever thread the core calls them from; the SDL sink/backend must be
  thread-safe (the pad header already requires it).

Target shutdown ordering: `request_stop` → guest thread observes the flag at its loop boundary and
returns → join → tear down GPU/window. Until that check exists, direct process exit deliberately
skips frontend/HLE static teardown so it cannot race the detached guest.

## Target / build layout

- New dir `frontends/prosper-app/` with its own `main.cpp` + the SDL sink/backend.
- Its own CMake target linking `prosper_core`, gated by `-DPROSPER_APP=ON` (**default OFF**), and
  on `find_package(SDL3)` + `find_package(Vulkan)` being present. The core and all existing tests
  build and pass with the frontend absent — never a core/CI dependency.
- SDL3 unifies window + audio + gamepad in one dep, matching the existing SDL3 direction
  (`feat/audio-sdl3`, the gamepad frontend).

## Deployment on Windows

For native Windows, configure MinGW with `-DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON
-DPROSPER_PAD_SDL3=ON`, build `prosper-app.exe`, and verify the window/swapchain first with
`--test-pattern --frames 120`. `scripts/run-windows.ps1` performs that full configure/build/run path:

```powershell
.\prosper\scripts\run-windows.ps1 .\PPSA24651-app0
```

The complete manual PowerShell recipe is in `WINDOWS_PORT_HANDOFF.md`.

WSLg remains a useful alternate path for running the Linux build:

1. Prereq check: `vulkaninfo` in the WSL shell confirms a usable Vulkan device (vendor WSL ICD, else
   Mesa Dozen over D3D12). This is the one external dependency worth verifying up front.
2. Build the frontend in WSL with `-DBUILD_FRONTEND_APP=ON`.
3. Run it; WSLg surfaces the window on the Windows desktop with audio + a taskbar/Start entry.
4. Optional: a `.desktop` file (WSLg auto-registers Start-menu shortcuts) or a one-line `.bat`/`.lnk`
   launcher on the Windows side.

## Keeping it agentic-first

The human-facing app is **opt-in and additive**. The BMP/CRC headless path stays the default and the
CI source of truth. Nothing the frontend adds gates a core test. A future *frontend* smoke test, if
wanted, should use an offscreen/hidden surface and stay out of the default `ctest` set (needs a real
Vulkan device + display, which CI may lack).

## P0b status — boot via the shared `boot_program()` helper (option **(b)**, landed)

The chosen path was **(b): extract a shared `boot_program()` helper** both `boot_trace` and the
frontend call, rather than duplicate the boot glue.

- `src/host/boot_program.hpp/.cpp` (in `prosper_core`, Linux and Windows): links the fixed module set
  (honoring `PROSPER_NO_PSN`, dropping absent modules), registers the built-in HLE, maps images, sets
  up TLS/unwind/procparam, installs the import stubs + trap handler, registers the PSN/SaveData
  module-start ranges, and runs the dependent-module init_arrays. An `after_hle_registered` hook lets
  a caller install host frontends (audio sink / pad backend) at the exact point `boot_trace` always
  did. On success the caller registers its renderer and calls `run_entry`.
- `boot_trace` migrated to call it (behavior-preserving; verified it still boots + renders).
- `prosper-app` calls it too (`--dump <app0>`): the guest boots and runs on its own thread while the
  window owns present.

**Remaining for "show the game" — the composite renderer (render-frontier-owned):** booting is not
enough. Nothing reaches the window until a live renderer is registered via `set_submit_renderer` —
the ~350-line DrawItem→Vulkan lambda in `boot_trace` that composites each submit and calls
`present_write_frame`. Verified empirically: `prosper-app --dump …` boots the game and opens the
window but presents **0 frames** without it. That lambda is the render frontier's actively-evolving
code, so registering/extracting it is a **coordinated next step with that workstream**, not a
duplicate. Until then the app is fully functional via `--test-pattern` (and any external feeder of
`present_write_frame`).

## Phased plan

- **P0a — window + present** ✅ **done**: lifecycle hook, SDL3 window, Vulkan swapchain,
  present-from-readback, `SDL_QUIT`/Esc→stop. Verified with `--test-pattern`.
- **P0b — shared boot** ✅ **done**: `boot_program()` helper; `boot_trace` + `prosper-app` both call it.
- **renderer (#177)** ✅ **done**: shared `register_live_renderer` (frontends/shared); the window
  shows the composited game (verified `--dump … --frames 3`).
- **P1 — audio** ✅ **done**: `prosper-app` installs the SDL3 `AudioSink` (`sceAudioOut` → host).
- **P2 — controllers** ✅ **done**: installs the SDL3 `PadBackend` (host gamepad → `libScePad`).
- **P3 — polish** (in progress): resize/fullscreen, pause/quit UX, and present-mode/latency tuning.
  Native Windows build/run packaging is done via `scripts/run-windows.ps1`. Cooperative guest-stop
  at a flip boundary is a follow-up (today the
  guest thread is detached at window-close and reclaimed by process exit).

## Risks & open questions

- **Vulkan availability** — native Windows needs the Vulkan SDK at build time and a working host
  driver at runtime. WSLg additionally needs a usable WSL Vulkan ICD.
- **The run/stop hook touches the run harness** (`boot_trace`/how the guest loop is driven), which the
  render-frontier and audio workstreams also use — coordinate; keep the fixed-budget CI path intact.
- **`feat/audio-sdl3` overlap** — P1 must build on that branch, not fork it.
- **Present latency/vsync** is intentionally FIFO today; low-latency present-mode selection remains
  P3 work.
- **Later zero-copy** (`external_memory`) is deliberately deferred; the readback seam is the v1 answer.
- **area:** shared/host infrastructure — needs an `area:` decision and coordination before build.
