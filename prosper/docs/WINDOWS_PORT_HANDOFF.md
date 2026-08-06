# Windows native port — handoff (2026-07-16)

For users of the prebuilt archive, start with `WINDOWS_RELEASE.md`. This document is the engineering
build, validation, and debugging handoff.

The Windows native core boots The Messenger (`PPSA24651`) through IL2CPP and repeated GC
stop-the-world cycles, drives real GPU draws, and presents 1920x1080 frames. The fence-field,
binary-file-read, and asynchronous-GC blockers were fixed in #672, #673, and #678 respectively.
Issue #683 adds and validates the SDL3/Vulkan window, audio and controller frontends plus the normal
sampled screenshot tool. Its native fresh-save route now reaches a fully lit first-level frame at
1920x1080 and exits cleanly. During that acceptance run, #688 exposed and fixed a stale Windows
readability stub that allowed a null dynamic-fetch table to reach a host dereference.

## What works today

The guest, on Windows (MinGW, native), reproducibly:
- Boots through SELF/ELF load → multi-module link → `module_start` init → guest `%fs` TLS.
- Spawns the full Unity thread ecosystem (Job.Worker 0–12, Background Job.Worker 0–15,
  AssetGarbageCollectorHelper ×13, Loading.AsyncRead, BatchDeleteObjects, …).
- Streams assets (positioned file IO), reaches VideoOut display setup (1920×1080, 3 buffers),
  registers flip/vblank/EOP equeues, and does an initial `GpuFlip` of buffer 0.
- Compiles 36 real shaders and submits GPU command buffers; the executor decodes real PM4
  (`WriteData`/`ReleaseMem`/`WaitRegMem`) and delivers flip/vblank/EOP events.
- Delivers IL2CPP's exception-type `0x1e` asynchronously on the requested target thread, allowing
  repeated GC stop-the-world suspend/resume cycles to complete.
- Runs the shared live Vulkan renderer and normal 1920x1080 present/readback path on an NVIDIA host.
- Builds SDL3 with native Win32 video, WASAPI audio, XInput/HIDAPI controllers, and Vulkan support.

### Current GC delivery model (#690)

Do not reintroduce `SuspendThread`/`SetThreadContext` as the normal exception path. Restoring a context
captured inside a native wait after waking it resumes stale `ntdll` state. Windows now queues the stop,
wakes the target's registered wait, and runs the guest handler cooperatively at an HLE boundary on a
dedicated alternate stack. `PROSPER_WIN_LEGACY_EXC=1` is the diagnostic opt-out.

Wait registrations are per nesting level, not merely per thread. The GC callback performs its own
semaphore waits, so a single thread-local registration lets the inner wait erase the still-live outer
wait. Condition interruption also increments its sequence before waking, which closes the wake-before-
`WaitOnAddress` race. Run `test_win_exception_delivery.exe` after touching any of these paths; its
nested-wait case deliberately holds the handler after the inner wait returns and proves the outer wait
is still discoverable. `PROSPER_APP_STALL_DUMP_MS=<milliseconds>` makes `prosper-app` dump the bounded
exception ring when presented-frame progress stops, and `PROSPER_VEHLOG=1` adds fatal worker context.

`PROSPER_SYNC_RING=<events>` retains the last N synchronisation events (wait-enter / wait-wake /
signal / broadcast / interrupt) in a lock-free ring, dumped by `prosper-app`'s timed guest-state dump
alongside the thread snapshot. It exists for the case a thread snapshot cannot explain: a TOTAL guest
deadlock, where the snapshot shows where all ~76 threads stopped but nothing shows how they got there,
and once the deadlock is complete there is nothing left running to log anything. Use it, not
`PROSPER_SYNCLOG`, when the failure is timing sensitive -- it is plain stores rather than `fprintf`,
so it does not serialise the very race being studied. Size it for the run: Blue Prince emits ~200k
events in the 100 s before it deadlocks, and a small ring drops exactly the objects of interest (the
ones whose last activity is OLDEST). The dump header reports the true total and the retained window,
so a truncated history is visible rather than silently passing as complete. Correlate it with the
snapshot using `tools/re/waitgraph.py`, which builds the wait-for graph -- each parked thread against
the thread that last woke its object -- and names the roots and any cycle.

Validation on 2026-07-16: 30 consecutive exception-suite passes and 12/12 fresh-save Blasphemous 2
Windows boots reached 360 presented frames with no stall or early exit. This validates boot/GC stability,
not the title-to-menu graphics transition; tiled 2D compute storage writeback remains tracked in #787.

Merged PRs that got it here (all `#ifdef _WIN32` — Linux/macOS unaffected, CI green on all 4 platforms):
- **#624** substrate: `exec_image_win.cpp` (VirtualAlloc mapping, VEH fault handler, SysV↔MS-x64 stub
  trampoline, host→guest SysV call trampoline `prosper_call_guest_sysv`, `__builtin_setjmp` recovery,
  TEB stack bounds), memory HLE (`hle_kernel_mem.cpp`), guest `%fs` TLS via FSGSBASE + VEH re-apply.
- **#628** lazy-commit reserved pages + guest worker-thread trampoline (SysV ABI + `%fs` TLS).
- **#655** build+init the live Vulkan renderer on Windows (CMake `find_package(Vulkan)` into `if(WIN32)`).
- **#658** reserve-alignment >64 KiB, worker-thread stack registration, sync diagnostics.
- **#663** `sceKernelWaitOnAddress` on the **native Win32 futex** (`WaitOnAddress`/`WakeByAddress*`) —
  fixed a lost-wakeup (global condition_variable + `notify_one`) that made the boot nondeterministic.
- **#665** positioned/vectored file IO (`pread`/`pwrite`/`*v`) via `ReadFile`/`WriteFile`+`OVERLAPPED`
  — unblocked Unity asset streaming (they'd returned -1).
- **#667** `futex_wake` (GPU RELEASE_MEM/EOP → `wake_label_waiters`) via native `WakeByAddressAll`
  (was a no-op on Windows) + native wait registers `futex_wait_enter/exit`.
- **#664** HDR support tracking issue (benign `sceSystemServiceGetHdrToneMapLuminance` stub).

## Resolved blocker (#672): GPU fence fields were garbage on Windows

The guest submits SubmitDcb #1 (a **sync-only** Dcb, 0 draws) and then wedges before submitting the
first **draw** Dcb. The executor logs:

```
[agc] WaitRegMem #0 NOT satisfied at fold time: [0x21a9fb40008]&0x0 = 0x0, func=3 ref=0x21a97f679a8 — dependency violated
```

Compare the very first `ReleaseMem`/`WaitRegMem` fence pair, Linux (renders 330 frames) vs Windows:

```
LINUX  : ReleaseMem action=0x28 dst=0x1 addr=0x…41da8 data_sel=0x2          data=0x1
         WaitRegMem  func=3 addr=0x…41da8 ref=0x1          mask=0xffffffff
WINDOWS: ReleaseMem action=0x28 dst=0x1 addr=0x…40008 data_sel=0x21a97f679a8 data=0x28
         WaitRegMem  func=3 addr=0x…40008 ref=0x21a97f679a8 mask=0x0
```

On Linux `data_sel=0x2, data=0x1, mask=0xffffffff` (a normal "write 1, wait for ==1" fence). On Windows
`data_sel`/`data`/`ref`/`mask` are garbage (`data_sel` is a heap *address*, `mask=0`). So the fence
writes the wrong value / the wait can never be satisfied → the guest's render thread never proceeds to
draws. **Rendering is gated entirely on decoding these fence fields correctly.**

### Confirmed root cause: stack-argument ABI in HLE handlers (args 7+)

The fence builders read their high arguments off the stack via `__builtin_frame_address(0)`:

- `agc_cb_release_mem` (`hle_agc.cpp` ~line 438): `volatile uint64_t* fp = __builtin_frame_address(0);
  uint64_t data_sel = fp[2], data = fp[3];`  (Sony ABI: `data_sel` = 7th arg, `data` = 8th arg, both
  passed on the **guest stack**, not in the 6 arg registers.)
- `agc_acb_dma_data` (~line 395): same pattern, `fp[2]`/`fp[3]` = `src_imm`/`num_bytes`.
- Likely the WaitRegMem builder and any other HLE reading a 7th/8th arg.

This `fp[2]/fp[3]` offset is **calibrated for the Linux import stub, which is a bare tail-`jmp`** — so
the guest's stack (return address at `[rsp]`, stack args 7,8 just above) is directly under the handler's
frame. **The Windows import stub is different**: `emit_impl` (`exec_image_win.cpp`) does
`sub rsp,0x38; mov [rsp+0x20],r8; …; call rax` — it converts SysV→MS-x64 and **`call`s** the handler,
inserting the stub's frame + 32-byte shadow space + the stub return address between the handler and the
guest's stack args. So in the handler, `__builtin_frame_address(0)` + `fp[2]/fp[3]` point into the
stub's shadow space / saved registers — **not** the guest's 7th/8th args. Hence garbage
`data_sel`/`data` (and the mislabeled `ref`/`mask`, which the same packet derives).

This is the same *class* as the already-fixed host→guest ABI bugs (`run_guest_inits`, the worker-thread
entry), but for the **guest→host** direction and specifically for **stack-passed args beyond the 6
registers**. The 6 register args are converted correctly by the stub (that's why everything else works);
only handlers that reach past arg 6 into the guest stack are affected.

### Implemented fix and validation

`emit_impl` now copies guest stack args 7-9 into the standard Microsoft-x64 outgoing argument slots;
Linux's guest-FS swap stub re-pushes the same three arguments with valid SysV alignment. Fixed-arity AGC
handlers take explicit args 7-9 instead of decoding compiler frames. `test_hle_stack_args` calls a real
generated stub with nine sentinels and checks alignment/return value; `test_agc_submit` checks the actual
ReleaseMem/WaitRegMem packet fields.

Native runtime validation: `data_sel=2 data=1`, `ref=1 mask=ffffffff`, no `NOT satisfied`, SubmitDcb #1
completes. The later #673 metadata and #678 GC-delivery blockers are also resolved; do not reopen the
solved fence investigation when diagnosing a later Windows failure.

## Full frontend build and validation (native Windows, MinGW)

The normal interactive path is now one command from the repository root. It configures the full
SDL3 video/audio/controller build when needed and then launches the title in the native window:

```powershell
.\prosper\scripts\run-windows.ps1 .\PPSA24651-app0
```

Use `-TestPattern -Frames 120` for a no-game swapchain smoke, or `-NoBuild` after the first build.
The explicit commands below remain the diagnostic recipe when compiler or CMake configuration needs
to be controlled directly.

Required tools are WinLibs MinGW-w64 UCRT `gcc/g++.exe`, Ninja, CMake, and a Vulkan SDK with
`VULKAN_SDK` set. Configure the window, SDL3 audio/controller, live renderer, and screenshot tool
directly from PowerShell:

```powershell
$wlb = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
cmake -S prosper -B prosper/build-mingw-app -G Ninja `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo `
  -DCMAKE_C_COMPILER="$wlb\gcc.exe" -DCMAKE_CXX_COMPILER="$wlb\g++.exe" `
  -DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON `
  -DGAME_DUMP=<DUMP_ROOT>/PPSA24651-app0
cmake --build prosper/build-mingw-app -j 8
```

Smoke-test the actual SDL window and Vulkan swapchain without a game, then run the game frontend:

```powershell
prosper/build-mingw-app/prosper-app.exe --test-pattern --frames 120
$env:PROSPER_GUEST_ARGS = '-force-gfx-direct'
prosper/build-mingw-app/prosper-app.exe --dump <DUMP_ROOT>/PPSA24651-app0
```

For unattended evidence, `screenshot.exe` uses Windows Imaging Component to write normal PNGs and
creates missing output directories. The following is the measured fresh-save acceptance route. It
samples every ten seconds to keep the evidence compact while retaining boot, menu, loading, fade, and
fully lit first-level milestones:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$env:PROSPER_PAD_SCRIPT = '@<REPO_ROOT>/prosper/scripts/messenger/reach-first-level-windows.pad'
$env:PROSPER_PAD_SCRIPT_LOG = '1'
$env:PROSPER_SAVEDATA_DIR = "$env:TEMP/prosper-messenger-$stamp"
prosper/build-mingw-app/screenshot.exe <DUMP_ROOT>/PPSA24651-app0 `
  --seconds 10 --count 36 --out "$env:USERPROFILE/Downloads/messenger-windows-$stamp" `
  --min-distinct-frames 25 --min-pixel-distinct-frames 10 --require-composited-frame
```

Use `--seconds 1 --count 360` when a complete one-frame-per-second sequence is more useful than a
compact acceptance sample. The validated 2026-07-14 run completed 36/36 captures at 360.3 seconds,
reported 36 distinct source frames and 23 distinct pixel frames, and showed the fully lit level in
the final PNG with process exit 0.

Use a new `PROSPER_SAVEDATA_DIR` for every fresh-save route validation. The manifest and PNG count
prove capture health; inspect the final PNGs too, because delivered input does not prove that the
intended game state was reached. Wall-time routes are renderer-speed sensitive, so use the dedicated
Windows route for native full rendering and keep `PROSPER_PAD_SCRIPT_LOG=1` enabled.

## Core diagnostics build

This smaller Git Bash recipe remains useful when only `boot_trace` and verbose core diagnostics are
needed. It shares the same compiler and Vulkan SDK prerequisites as the full build above:

```bash
WLB="$HOME/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin"
export PATH="$WLB:$PATH"   # + the Ninja package bin
cmake -S prosper -B prosper/build-mingw-vk -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER="$WLB/gcc.exe" -DCMAKE_CXX_COMPILER="$WLB/g++.exe" \
  -DGAME_DUMP=<DUMP_ROOT>/PPSA24651-app0     # VULKAN_SDK must be in env
cmake --build prosper/build-mingw-vk --target boot_trace -j8
```

Run from Git Bash (guest-fs is default-on on Windows):

```bash
cd prosper/build-mingw-vk
PROSPER_GFXLOG=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
  PROSPER_FRAME_DIR=/some/dir ./boot_trace.exe <DUMP_ROOT>/PPSA24651-app0
```

Performance diagnostics: set `PROSPER_RENDER_TIMING=1` for aggregate graphics/compute stage timings, or
`PROSPER_RENDER_TIMING=detail` to include slow individual texture decodes. The output and bucket definitions
are documented in `FRONTEND_APP.md`; rolling `[render-window]` lines cover the latest 25 operations rather
than averaging away the current scene. Use `PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT=N` to defer detailed
texture lines until the scene being profiled. Backend output also reports transient Vulkan memory-pool statistics;
use `PROSPER_NO_MEMORY_POOL=1` for a direct allocate/free A/B run, or `PROSPER_MEMORY_POOL_MB=<MiB>` to
override the default 512 MiB graphics budget. Compute allocations use a separate 256 MiB default budget,
overridable with `PROSPER_COMPUTE_MEMORY_POOL_MB=<MiB>`. Graphics shader translation is cached by shader
bytes plus descriptor-interface semantics; `PROSPER_NO_SHADER_CACHE=1` disables it for comparison and
`PROSPER_SHADER_CACHE_MB=<MiB>` overrides its 128 MiB budget. Timing windows report shader hits/misses and
miss compilation time. Run `test_shader_recompile_cache` after changing the key or recompiler contract.
Frontend windows also report texture resource uses as `textures` and callback-local duplicate decodes
avoided as `reused`, plus exact-byte cross-submit `texture_cache` hits/misses/invalidations. The persistent
cache covers guest-backed linear/tiled 2D `Unorm8` (1-4 components) and BC1-BC7 sampled textures. Its
default ceiling is one eighth of host physical memory, clamped to 1-2 GiB; use
`PROSPER_NO_TEXTURE_DECODE_CACHE=1` for an A/B or `PROSPER_TEXTURE_DECODE_CACHE_MB=<MiB>` to change the
budget. Do not infer descriptor-table
identity from shader/user-SGPR values alone:
pointed-to guest memory is mutable, and that cache experiment stalled Messenger at its loading screen.
See `FRONTEND_APP.md` for the invalidation requirement.

Immutable shader span/dispatch/interpolation analysis is separately byte-validated and bounded to
64 MiB. `PROSPER_NO_SHADER_ANALYSIS_CACHE=1` restores direct analysis for an A/B;
`PROSPER_SHADER_ANALYSIS_CACHE_MB=<MiB>` changes the bound. This cache never stores concrete user
SGPR or descriptor-table contents.

Diagnostics (env, all off by default): `PROSPER_GFXLOG` (`[gfx]`/`[agc]` PM4 decode + the `NOT satisfied`
fence log), `PROSPER_EVLOG` (`[ev]` equeue/flip/EOP), `PROSPER_SYNCLOG` (`[sync]` WaitOnAddress/Wake with
native thread id + validated guest caller, `[sync2]` cond/sema/EventFlag), `PROSPER_EXCLOG` (GC exception
raise/deliver/resume), `PROSPER_MEMLOG`, `PROSPER_VEHLOG`, and boot_trace `[memclass]`. Compare with a
known-good Linux capture when separating cross-platform frontend faults from shared renderer faults. For a
long boot, `PROSPER_SYNCLOG_DELAY_MS=N` suppresses the high-volume `[sync2]` stream until N milliseconds
after the first synchronization call, retaining normal progression before the suspected boundary.
`PROSPER_SYNCLOG_PTHREAD=0xN` restricts `[sync2]` records to one guest pthread when a full trace perturbs
the title; pair it with the app's guest-thread checkpoint to obtain the pthread id.
`PROSPER_SYNCLOG_COND_ONLY=1` further suppresses semaphore and event-flag records while retaining raw
condition waits/signals/broadcasts.
`PROSPER_SYNCLOG_SEMA_ONLY=1` instead keeps semaphore records. With a pthread filter, the last semaphore
that thread waits on becomes the focus, so matching signals from other threads are retained automatically.

## Gotchas learned (so the next agent doesn't relearn them)

- Windows resets the user `%fs` base to 0 on every kernel transition; guest TLS survives only via the
  VEH `wrfsbase` re-apply (`hle_kernel_mem.cpp`/`guest_tls.cpp`). Native FSGSBASE (`rd/wrfsbase`,
  `WaitOnAddress`, `WakeByAddress*`) works and needs `-lsynchronization` (linked into `prosper_core`).
- MinGW `longjmp` does an SEH unwind that can't cross guest/asm frames → use `__builtin_setjmp/longjmp`.
- `boot_trace` on Windows is nondeterministic-crash-prone ONLY if a diagnostic reads raw stack words —
  the sync-caller scanner is `VirtualQuery`-guarded now; keep any new stack-walk guarded.
- Windows file descriptors must use binary mode for guest assets. Text-mode reads can translate
  bytes and corrupt metadata even when the same path works on Linux (#673).
- Asynchronous GC delivery must wake a target blocked in `WaitOnAddress`; rewriting its context alone
  is insufficient because it may never return to guest code (#678).
- Dynamic-fetch constant folding must reject unreadable guest ranges on every host. The native
  renderer exercises this path; an unconditional Windows readability stub caused #688.
- `sceKernelMprotect` on Windows accepts guest memory the VA tracker (`g_maps`) does not own, but
  **only inside a fixed guest MODULE-CODE aperture** — the eboot, the PRXes, the plugin and
  runtime-PRX pools (`guest_va_in_module_code`). Those images are mapped by the exec substrate's
  `VirtualAlloc`, so they never enter `g_maps`, and a title mprotecting its own segment must succeed
  the way POSIX `mprotect` does (#2101: The Messenger abandoned AGC device init over the refusal).
  Everything else still fails: `MEM_FREE`, `MEM_RESERVE` with no tracker backing, a tracked mapping
  the tracker calls uncommitted over committed host pages, and any committed page outside every
  module aperture. **`guest_module_name(a) != "mapped/host"` is the wrong test for this** — it also
  matches `[BOOT_STUB, BOOT_STUB_END)`, which is prosper's OWN emitted import trampolines, committed
  by `install_stubs`' plain `VirtualAlloc` and therefore untracked exactly like a module image; using
  it let a guest strip execute permission from every HLE entry point (#2144 F2).
- The tracker snapshot that decides all of the above (`tracked_mapping_slices`) must be the *complete*
  coverage of the span. Returning early on the first untracked hole silently reclassified every tracked
  slice above it (#2144 F1), and **two** consumers read that vector for completeness, not one:
  `tracked_slices_back_host_reservation` in the `MEM_RESERVE` arm, and — less obviously —
  `next_slice_base` in the `MEM_COMMIT` arm, which is *derived from the vector* and collapses to
  `region_end` when it is truncated, so the untracked gap handed to the aperture test is a different
  range than it should be. Reason about both before changing that function; a safety argument covering
  only the first one is the shape of mistake that produced #2144 in the first place.
- Accepting a tracker-backed `MEM_RESERVE` region is **not** a relaxation. `win_protect`'s own
  precondition comment (added 2026-07-18 in `9e8b22b5`, well before #2117) already states it: reserved
  pages take a tracking-only protection change, matching the POSIX `PROT_NONE` reservation path. The
  `MEM_RESERVE` arm pushes nothing into `committed`, so no `VirtualProtect` runs for it at all — the
  entire effect is a success return plus `k_mprotect`'s `retrack_prot` on a range the tracker already
  owns. Truncation never implemented a narrower *policy*; it just made that arm unreachable whenever
  anything untracked preceded the reservation.
- Note that #2117 narrowed the allowance during review: its PR body still describes the withdrawn first
  cut, which accepted any committed page.
