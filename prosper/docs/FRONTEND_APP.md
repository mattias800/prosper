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
     │      audio_set_sink() · pad_set_backend() · present frame lease · run/request_stop()
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

### 3. Video — rendered-frame ownership (exists, `src/gpu/videoout_present.hpp`)
```cpp
struct PresentFrameLease {
    uint64_t frame_seq;
    uint32_t width, height;
    std::shared_ptr<const std::vector<uint8_t>> rgba;
};
bool present_acquire_rendered_frame(PresentFrameLease& out);
size_t present_readback(void* dst, size_t dst_cap); // compatibility/capture copy
```
The renderer's selected scanout, the present layer, and `prosper-app` share one immutable CPU-pixel
allocation. Acquiring a lease does not copy the image, and its storage stays valid if the renderer
publishes a newer frame while the app uploads the old one. Screenshot, capture, replay, and legacy
readback APIs deliberately retain copying semantics. The frontend remains a pure *consumer* of finished
frames and polls `present_count()` for guest flip pacing.

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

## The Vulkan-context decision: two contexts, one shared CPU frame

**The frontend owns its own Vulkan context (instance + device + swapchain on the window surface).
The core keeps its existing headless render device. Frames cross the boundary through immutable shared
CPU storage acquired with `present_acquire_rendered_frame`.** Rationale:

- The core's render device stays **surface-free and unchanged** — no swapchain dependency leaks into
  the renderer, no `#ifdef`, headless + CI identical.
- The seam already emits CPU pixels, so the frontend is a pure consumer — the cleanest boundary, and
  the same seam becomes a shared-memory frame ring if the two are ever split into separate processes.
- Cost = the renderer's required GPU readback plus one frontend upload per frame. There are no
  renderer-cache-to-return, return-to-present, or present-to-app copies. The Messenger is 2D at 1080p
  (~8 MB/frame), but native
  Windows measurements showed that memory-type selection matters: uncached host-visible readback cost
  roughly 570 ms per frame on an RTX 4090, versus about 1.8 ms with `HOST_CACHED` memory.

**When to revisit:** direct image presentation remains worthwhile after the renderer stops rebuilding
pipelines and image/buffer objects per batch. At that point, unify on a single frontend-provided device
or share the render target via `VK_KHR_external_memory` (zero-copy), while retaining `present_readback`
for headless tests and screenshot tooling.

## Present loop (sketch)

```
init:  SDL_Init(VIDEO|AUDIO|GAMECONTROLLER); create window;
       create VkInstance(+VK_KHR_surface, win32/wayland), VkSurfaceKHR, pick device, swapchain,
       one staging buffer + one sampled VkImage sized to present_width()×present_height();
       audio_set_sink(&sdl_sink); pad_set_backend(&sdl_pads);
       start the guest run-loop on its own thread.

frame: SDL_PollEvent → on SDL_QUIT / window-close: prosper_request_stop(); break;
       F11 / Alt+Enter toggles borderless desktop fullscreen;
       pixel-size change → recreate the Vulkan swapchain before the next present;
       if present_count() advanced since last shown:
           lease = present_acquire_rendered_frame();
           upload lease.rgba → VkImage; blit/scale VkImage → acquired swapchain image; vkQueuePresentKHR;
       else: small sleep / wait on a frame condvar to avoid spinning.

quit target: prosper_request_stop(); join guest thread; destroy swapchain/device/window; SDL_Quit().
quit today:  prosper_request_stop(); flush logs; direct process exit while the guest is still live.
```
Handle swapchain resize (`present_width/height` change or window resize → recreate). Vsync via
`VK_PRESENT_MODE_FIFO`.

## Threading

- **Guest run-loop thread**: drives the guest (as `boot_trace` does), renders into scanout on flip.
- **Main/UI thread**: SDL event pump + swapchain present. It holds an immutable shared frame lease
  while copying to its Vulkan staging allocation; publication only swaps shared ownership under the
  present mutex.
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

### Live renderer performance diagnostics

Set `PROSPER_RENDER_TIMING=1` before launching to print aggregate timings every 25 operations:

```powershell
$env:PROSPER_RENDER_TIMING = '1'
.\prosper\scripts\run-windows.ps1 .\PPSA24651-app0 -NoBuild
```

The output separates guest-state realization, ordered graphics/compute execution, CPU resource
decode/detile, Vulkan target and pipeline setup, upload/recording, GPU fence wait, readback, cleanup,
and frontend publication. Cumulative `[render-timing]` lines show the whole run; `[render-window]`
lines show only the latest 25 submits/calls so a scene transition is immediately visible. The
`backend-submit` lines sum every Vulkan call made by those exact ordered submits, rather than mixing
an independent 25-call backend window across submit boundaries. Compare `measured` with `detail` and
its `other` remainder to verify that the subphase attribution covers the frontend's backend wall time.
The core window also reports graphics-shader cache hits, misses, bypasses, and actual miss compilation time.
The older independent 25-backend-call timing windows are disabled by default because their multi-line logging
can materially perturb the target call charged for printing them on Windows. Set
`PROSPER_BACKEND_TIMING_WINDOWS=1` only when that cross-submit legacy view or its memory-pool line is needed.
Use `PROSPER_RENDER_TIMING=detail` to additionally print individual texture decodes taking at least
0.5 ms (capped at 250 lines). Set `PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT=N` to begin those detail
lines at renderer submit N, so boot textures do not consume the cap before the scene under study. Each
line identifies local reuse, persistent hit/miss/invalidation, RTT sourcing, or an uncached decode. The
aggregate output also reports retained RTT bytes, decode-scratch capacity, and validation-scratch capacity;
use those figures before treating a process-memory increase as an unbounded cache.
Set `PROSPER_RTT_TIMING=1` with timing enabled to print one lightweight `[rtt-timing]` line per rendered target
group. It includes the submit number, target address, dimensions, draw count, and that target's exact Vulkan
phase breakdown without scanning pixels or printing every draw. `measured`, `detail`, and `other` show the
frontend-observed call duration, attributed backend phase sum, and remaining wrapper/logging cost for that
specific target. `PROSPER_RTTLOG=1` includes the same line plus
the more expensive visual RTT diagnostics. Use `PROSPER_RTTLOG_MIN_SUBMIT=N` and
`PROSPER_RTTLOG_MAX_SUBMIT=N` to bound either mode; unrestricted output can still perturb wall-clock routes.
For scene selection that is stable across runs, set `PROSPER_RTT_TIMING_MIN_DRAWS=N`. Lightweight records are
buffered for one ordered submit and emitted together only when its total backend draw count reaches N, so all
target calls are retained without logging the boot/menu workloads. For example, 300 selects the current
Dead Cells post-parse workload while excluding its 54-56-draw loading loop.
Selected lightweight records are formatted together and written to stderr in one batch at submit completion;
this preserves one parseable line per target without paying a Windows console write for every target.

Ordered graphics/compute submits retain a bounded journal of exact guest-memory ranges written by the
compute backend. A persistent texture validated in an earlier graphics span can therefore skip its repeated
full-byte scan when no later write overlaps it. Cross-submit texture validation currently uses exact byte
comparison on every platform. Windows protection-fault watches are deliberately unsupported: the Windows
exception dispatcher writes below the interrupted stack pointer before a vectored handler runs, which can
corrupt the guest's valid SysV red-zone locals. Timing therefore reports those watch attempts as `unknown`
and the remaining exact validation bytes; do not re-enable page-fault watches without an exception-delivery
mechanism that preserves all 128 guest red-zone bytes. For the same-submit A/B control, set
`PROSPER_NO_SUBMIT_TEXTURE_VALIDATION_REUSE=1`. Set
`PROSPER_AUDIT_SUBMIT_TEXTURE_VALIDATION_REUSE=1` to keep every comparison while checking and logging any
disagreement with the journal's unchanged decision; use that audit before extending writer coverage.
The instrumentation does not take clock samples when the variable is unset. This is the first tool
to use when the window presents correctly but a title is not interactive; do not infer a GPU
bottleneck from low FPS without the stage breakdown.

On native Windows, large aligned direct-memory views are demand-paged. Their committed 16 KiB pages
are retained in a mapping-generation-invalidated bitmap so repeated GPU resource reads do not repeat
the same `VirtualQuery` calls. Set `PROSPER_NO_SPARSE_DMEM_PAGE_CACHE=1` for a control run when
investigating this path; it is expected to be substantially slower in resource-heavy scenes. A
thread-local positive HLE mapping lookup can be disabled separately with
`PROSPER_NO_SPARSE_DMEM_ACCESS_CACHE=1`.

When the `tables=` bucket is unexpectedly large, set `PROSPER_STAGE_FOLD_PROFILE=1`. It ranks the
shader address and user-SGPR base pairs responsible for scalar table folding, including average and
maximum time, decoded instructions, dynamic fetches, SRT uses, guest readability checks, and the
decode/probe/interpreter split. It prints every 4096 folds by default; set
`PROSPER_STAGE_FOLD_PROFILE_CALLS=<N>` to change that window. The profiler is intended for short
diagnostic runs and takes no timing samples when disabled.

Transient Vulkan memory uses a bounded, exact-requirements pool because every backend call waits for its
fence before cleanup. The optional backend timing windows include `memory_pool` hits, misses, cached allocation count/bytes,
and budget-driven discards. The default budget is 512 MiB; override it with
`PROSPER_MEMORY_POOL_MB=<MiB>`, or set `PROSPER_NO_MEMORY_POOL=1` for an A/B run against direct
`vkAllocateMemory`/`vkFreeMemory`. The pool retains only allocation objects: images, buffers, views,
descriptors, and their layout/content rules keep their existing per-call lifetimes.

Within one already-synchronous backend call, identical immutable Vulkan contracts share objects instead of
recreating them for every draw. Storage buffers require the same nonzero guest address, size, and complete
captured bytes; synthetic resources with no identity remain distinct. Texture image views and samplers use
their complete image/view/swizzle/filter/address/LOD contract. Descriptor-set layouts and pipeline layouts
use their complete binding/layout contracts, and all sets allocate from one call-wide descriptor pool. Every
object remains call-local and is destroyed after the existing fence wait, so this adds no cross-submit
freshness assumption. Set `PROSPER_NO_BACKEND_RESOURCE_SHARE=1` to disable the keyed object reuse for an
A/B; the safe call-wide descriptor-pool consolidation remains enabled in both modes.

Host-visible storage buffers use a separate bounded pool across backend calls. By default, the backend
packs call-local logical uploads into non-overlapping slices of a persistently mapped host-coherent arena,
using the device's storage-buffer offset alignment. Each Vulkan descriptor keeps the exact aligned offset
and logical byte range, and every slice is rewritten before use. The backend waits for its fence before
returning arenas to the pool, so later uploads cannot race in-flight work. Arenas start at 1 MiB; set
`PROSPER_BACKEND_BUFFER_ARENA_KB=<KiB>` to tune that target or
`PROSPER_NO_BACKEND_BUFFER_ARENA=1` to restore one pooled buffer per logical upload. Oversized uploads and
the fallback path still use power-of-two capacity classes. The thread-local pool is capped at 4096 buffers
and 256 MiB by default; set `PROSPER_BACKEND_BUFFER_POOL_MB=<MiB>` to change the byte budget or
`PROSPER_NO_BACKEND_BUFFER_POOL=1` for the former create/map/destroy path. With renderer timing enabled,
`backend_buffer_pool` reports cumulative hits, misses, cached count/bytes, and evictions.

The persistent compute device has the same exact-requirements allocation pool with a separate 256 MiB
default budget (`PROSPER_COMPUTE_MEMORY_POOL_MB=<MiB>`). `PROSPER_NO_MEMORY_POOL=1` disables both
graphics and compute pools. Decoded texture scratch vectors are also retained across callbacks; they
are storage only, and every partial decode/read explicitly clears its unwritten tail.

Within one renderer callback, repeated guest-backed texture descriptions reuse the first decoded
pixel buffer. A bounded process-wide cache also covers guest-backed linear and tiled 2D sampled
textures in `Unorm8` component widths 1-4 and BC1-BC7 formats. Every cross-submit lookup validates
the exact source range before reusing decoded pixels: direct narrow sources compare their native
bytes, tiled sources retain the padded tiled range, and block-compressed sources use their exact
block dimensions and block size. Changed bytes, mapping/readability changes, or address reuse
invalidates the entry and creates a new content-version ID. Live render targets, storage images,
captured host backing, cube/volume textures, DCC surfaces, and other formats remain excluded. The
default budget is 1 GiB, which covers Evergate's measured 835 MiB working set; use
`PROSPER_TEXTURE_DECODE_CACHE_MB=<MiB>` to change it or
`PROSPER_NO_TEXTURE_DECODE_CACHE=1` for an A/B run. Timing reports cross-submit `texture_cache`
hits/misses/invalidations separately from `textures` (all texture uses) and `reused` (both local and
persistent decodes avoided).
Complete readable source ranges are compared directly against the cache's retained encoded bytes;
partial sparse ranges keep the guarded scratch-copy fallback. This preserves the same exact byte and
readability checks without copying large texture sources before comparing them. Set
`PROSPER_TEXTURE_VALIDATION_SCRATCH_COPY=1` for an A/B against the previous copy-then-compare path.

The Vulkan backend may retain an optimal sampled image only when that exact frontend validation
supplies a nonzero content-version ID. Cache hits skip image allocation, staging allocation, pixel
copy, transfer commands, and upload barriers. Exact image-view and sampler contracts over a retained
image remain resident with that image, under a 32-contract per-image bound; set
`PROSPER_NO_BACKEND_PERSISTENT_TEXTURE_BINDINGS=1` to retain images while restoring callback-local
bindings for an A/B. The image cache is bounded to 1 GiB and 1024 allocations by default. Set
`PROSPER_BACKEND_TEXTURE_CACHE_MB=<MiB>` to change the byte budget or
`PROSPER_NO_BACKEND_PERSISTENT_TEXTURES=1` for a forced-upload A/B. Backend timing reports
`persistent=hits/misses` and the current cache bytes. The frontend decoded-pixel budget and backend
device-image budget are separate: a hot immutable atlas can occupy space in both, trading bounded
residency for lower frame time.

Intermediate live color targets remain on the GPU by default. A later graphics pass that samples the
same guest target identity, extent, and format binds the retained image directly. Intermediate scanout
spans also defer readback until the final renderer callback; a final scanout pass reads the accumulated
target normally, while a submit that ends elsewhere materializes its cached scanout once before publishing.
Deferred scanouts are pinned against bounded target-cache eviction until that final callback.
Ordered DMA producers, captures, pixel diagnostics, same-target feedback, and other authoritative-readback
spans keep the CPU path. Guest GPU writes invalidate overlapping retained targets through the ordered write
observer. Set `PROSPER_NO_INTERMEDIATE_SCANOUT_DEFER=1` to restore per-span scanout readback for an A/B. Set
`PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1` for a complete frontend A/B, or
`PROSPER_NO_BACKEND_PERSISTENT_COLOR_TARGETS=1` to disable backend retention independently. The backend
cache defaults to 256 MiB and can be changed with `PROSPER_BACKEND_TARGET_CACHE_MB=<MiB>`.

Graphics RDNA2-to-SPIR-V results use a process-wide bounded cache (4096 entries and 128 MiB by
default). Its key contains the shader bytes and only the resource-table fields consumed by compilation;
current guest addresses and backing data remain on each draw and are never cached. Use
`PROSPER_NO_SHADER_CACHE=1` for a direct-recompiler A/B run or
`PROSPER_SHADER_CACHE_MB=<MiB>` to change the byte budget. `test_shader_recompile_cache` compares
every tested miss byte-for-byte with the direct recompiler and verifies that runtime-only resource
changes hit while descriptor-interface changes miss.

Shader code span, PC-relative dispatch metadata, and fragment interpolation discovery use a separate
64 MiB immutable-analysis cache. Every lookup validates the complete instruction/embedded-table span
byte-for-byte, so same-address shader patching invalidates the entry. Concrete user SGPRs, descriptor
table bytes, guest addresses, and resource backing remain per-draw inputs and are never reused by this
cache. Set `PROSPER_NO_SHADER_ANALYSIS_CACHE=1` for the direct-analysis A/B or
`PROSPER_SHADER_ANALYSIS_CACHE_MB=<MiB>` to change its budget.

The Vulkan backend retains graphics pipelines across render-target calls. Live draws carry a
process-unique, never-recycled identity from the exact shader cache, so a hot lookup does not copy or
hash complete SPIR-V modules. Capture, replay, and test draws without those identities fall back to an
exact full-module key. The rest of the key contains the descriptor-layout contract and every baked
fixed-function value; equality still compares the complete key after hashing. The cache is bounded to
1024 entries by default. Set `PROSPER_PIPELINE_CACHE_ENTRIES=<N>` to change the entry limit or
`PROSPER_NO_BACKEND_PIPELINE_CACHE=1` for a transient-pipeline A/B. With `PROSPER_RENDER_TIMING=1`,
the backend and submit-aligned windows report references, hits, misses, bypasses, current entries, and
evictions. A pipeline hit also skips temporary Vulkan shader-module creation.

Pipeline layouts use a separate exact cross-call cache keyed by every descriptor set's ordered binding,
type, count, and stage contract. Descriptor pools, descriptor sets, and descriptor-set layouts remain
call-local; Vulkan pipeline-layout compatibility follows the complete descriptor contract rather than the
temporary layout handle. The cache retains at most 256 layouts and evicts the least-recently-used entry not
referenced by the current backend call. Set `PROSPER_PIPELINE_LAYOUT_CACHE_ENTRIES=<N>` to change the bound
or `PROSPER_NO_BACKEND_PIPELINE_LAYOUT_CACHE=1` for a call-local A/B.

Do not cache `build_stage_table` results using only shader addresses and user-SGPR values. Descriptor
tables are reached through guest pointers, and their memory can change while every pointer/register
value remains identical. That experiment caused Messenger to remain on its initial loading screen and
was removed after an enabled/disabled A/B test. A future table cache needs explicit guest-memory
versioning or equivalent invalidation; the `tables=` timing bucket measures this work without caching it.

Windows readability checks retain positive results across synchronous submits only for explicitly
tracked, currently readable kernel-HLE mappings. This is mapping-topology reuse, not a table or
guest-content cache: HLE map, unmap, and protection operations advance a process generation and
discard all retained ranges before the next submit. Host-managed guest stacks, diagnostic mappings,
and other untracked regions are reused only within one submit. Guest bytes are still interpreted on
every fold. Set `PROSPER_NO_GUEST_READ_CACHE=1` for the uncached A/B path. The contract assumes memory
referenced by a synchronous GPU submit remains mapped until that submit returns, which was already
required before the cross-submit reuse was added.

Windows direct memory uses a delete-on-close sparse temporary file as its shared physical backing.
Mapped views are ordinary `MEM_COMMIT` file pages and can participate in cross-submit readability reuse;
the kernel demand-pages untouched storage without delivering a user-mode first-touch exception. This keeps
large logical arenas sparse and physical aliases coherent without letting Windows exception dispatch
overwrite guest SysV red-zone locals. If sparse backing cannot be created, direct-memory mapping fails
closed instead of silently returning to the unsafe `SEC_RESERVE` exception path.

The current renderer remains a deterministic readback-based implementation. It retains CPU-visible pixels
for screenshots and temporal RTT composition, so it is not the final zero-copy architecture. Issue #702
tracks persistent Vulkan resource/pipeline caching and direct image presentation beyond the initial
readback-memory, detile, compute-context, transient-memory-pool, scratch-reuse, and shader-cache fixes.
On the native Windows Messenger first level, the exact-byte texture cache improved the same-binary
workload from about 20 FPS to 24 FPS (resource construction 10.1-10.5 ms to about 3.9 ms). The remaining
roughly 36-38 ms submit time is primarily ordered graphics/compute backend work: four graphics spans,
four compute dispatches, synchronous fence waits/readbacks, and transient Vulkan pipelines/resources.
Further work should be validated against a 3D title and converge graphics/compute resource ownership;
title-specific 2D cache additions are not the current priority.
The complete A/B table, heavy-frame budget, corrected capture graph, rejected experiments, and handoff
decision are preserved in `RENDERER_PERFORMANCE_2026_07.md`.

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
- **P3 — polish** (in progress): resize/fullscreen, present-mode selection, and non-modal SaveData
  percentage progress presentation are implemented;
  pause/quit UX remains.
  Native Windows build/run packaging is done via `scripts/run-windows.ps1`. Cooperative guest-stop
  at a flip boundary is a follow-up (today the
  guest thread is detached at window-close and reclaimed by process exit).

## Risks & open questions

- **Vulkan availability** — native Windows needs the Vulkan SDK at build time and a working host
  driver at runtime. WSLg additionally needs a usable WSL Vulkan ICD.
- **The run/stop hook touches the run harness** (`boot_trace`/how the guest loop is driven), which the
  render-frontier and audio workstreams also use — coordinate; keep the fixed-budget CI path intact.
- **`feat/audio-sdl3` overlap** — P1 must build on that branch, not fork it.
- **Present latency/vsync** defaults to FIFO. `--present-mode mailbox` opts into low-latency vsync,
  and `--present-mode immediate` explicitly permits tearing; unsupported optional modes fall back to
  FIFO with a diagnostic.
- **Later zero-copy** (`external_memory`) is deliberately deferred; the readback seam is the v1 answer.
- **area:** shared/host infrastructure — needs an `area:` decision and coordination before build.
