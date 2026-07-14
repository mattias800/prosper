# Windows native port — handoff (2026-07-14)

The Windows native core boots The Messenger (`PPSA24651`) through the entire OS/runtime/asset/sync
layer and drives the GPU pipeline; it wedges just before the **first draw command buffer** on a
**GPU-fence field-decode bug**. This doc hands off that exact next step, with the evidence, a strong
root-cause lead, and the full build/run/diagnose recipe. Companion: `docs/PORTING.md` ("Windows").

## What works today (merged to master)

The guest, on Windows (MinGW, native), reproducibly:
- Boots through SELF/ELF load → multi-module link → `module_start` init → guest `%fs` TLS.
- Spawns the full Unity thread ecosystem (Job.Worker 0–12, Background Job.Worker 0–15,
  AssetGarbageCollectorHelper ×13, Loading.AsyncRead, BatchDeleteObjects, …).
- Streams assets (positioned file IO), reaches VideoOut display setup (1920×1080, 3 buffers),
  registers flip/vblank/EOP equeues, and does an initial `GpuFlip` of buffer 0.
- Compiles 36 real shaders and submits GPU command buffers; the executor decodes real PM4
  (`WriteData`/`ReleaseMem`/`WaitRegMem`) and delivers flip/vblank/EOP events.

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

## The current blocker: GPU fence fields are garbage on Windows

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

### Strong root-cause lead: stack-argument ABI in HLE handlers (args 7+)

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

### Suggested fix directions (pick one; verify against the Linux field values above)

1. **Forward the guest stack args in the stub, at a known offset.** Extend `emit_sysv_to_ms_prologue`/
   `emit_impl` so the guest's stack args 7,8 (at the guest `[rsp+8]`,`[rsp+16]` on entry to the stub) are
   copied to a fixed, handler-discoverable location, and give the `_WIN32` handlers a small accessor
   (e.g. `guest_stack_arg(n)`) that reads them from there instead of `__builtin_frame_address`. Cleanest;
   fixes all such handlers at once.
2. **Windows-specific `fp[]` offset** in each affected handler: work out the constant delta the stub
   frame adds (stub does `sub rsp,0x38` then `call` pushes 8 → the guest return address sits at a fixed
   offset above the handler frame; the guest stack args are `+8`/`+16` from there). Add
   `#ifdef _WIN32` offsets to `agc_cb_release_mem`, `agc_acb_dma_data`, and any WaitRegMem builder.
   Quicker but per-handler and brittle.
3. Confirm which handler builds the WaitRegMem `mask`/`func`/`ref` (grep `R_WAIT_REG_MEM` / the
   `WaitRegMem` builder in `hle_agc.cpp`) and check whether IT also reads stack args.

Validation: with any fix, the Windows first fence pair must read `data_sel=0x2 data=0x1 mask=0xffffffff`
(matching Linux), the `NOT satisfied` log must disappear, and `SubmitDcb #2: executed N draws ->
presented 1920x1080 frame` should appear (with BMPs in `PROSPER_FRAME_DIR`).

## Build + run + diagnose (native Windows, MinGW)

Toolchain on PATH: WinLibs MinGW-w64 UCRT `gcc/g++.exe` + Ninja; CMake; Vulkan SDK (`VULKAN_SDK` set).
Git via PowerShell; build/run via Git Bash. Configure a Vulkan-enabled build dir once:

```bash
WLB="/c/Users/matti/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin"
export PATH="$WLB:$PATH"   # + the Ninja package bin
cmake -S prosper -B prosper/build-mingw-vk -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER="$WLB/gcc.exe" -DCMAKE_CXX_COMPILER="$WLB/g++.exe" \
  -DGAME_DUMP=/c/Users/matti/repos/ps5ys/PPSA24651-app0     # VULKAN_SDK must be in env
cmake --build prosper/build-mingw-vk --target boot_trace -j8
```

Run (Git Bash — PowerShell redirection writes UTF-16). Guest-fs is default-on on Windows:

```bash
cd prosper/build-mingw-vk
PROSPER_GFXLOG=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
  PROSPER_FRAME_DIR=/some/dir ./boot_trace.exe /c/Users/matti/repos/ps5ys/PPSA24651-app0
```

Diagnostics (env, all off by default): `PROSPER_GFXLOG` (`[gfx]`/`[agc]` PM4 decode + the `NOT satisfied`
fence log), `PROSPER_EVLOG` (`[ev]` equeue/flip/EOP), `PROSPER_SYNCLOG` (`[sync]` WaitOnAddress/Wake with
tid + validated guest caller, `[sync2]` cond/sema/EventFlag), `PROSPER_MEMLOG`, `PROSPER_VEHLOG`,
boot_trace `[memclass]`. The Messenger renders on **Linux** (`build-linux`, `PROSPER_GUEST_FS=1 …`) —
use it as the oracle: `SubmitDcb #2: executed 3 draws -> presented 1920x1080 frame`.

## Gotchas learned (so the next agent doesn't relearn them)

- Windows resets the user `%fs` base to 0 on every kernel transition; guest TLS survives only via the
  VEH `wrfsbase` re-apply (`hle_kernel_mem.cpp`/`guest_tls.cpp`). Native FSGSBASE (`rd/wrfsbase`,
  `WaitOnAddress`, `WakeByAddress*`) works and needs `-lsynchronization` (linked into `prosper_core`).
- MinGW `longjmp` does an SEH unwind that can't cross guest/asm frames → use `__builtin_setjmp/longjmp`.
- `boot_trace` on Windows is nondeterministic-crash-prone ONLY if a diagnostic reads raw stack words —
  the sync-caller scanner is `VirtualQuery`-guarded now; keep any new stack-walk guarded.
- The intermittent `exit 139` on some runs is a residual worker-thread edge; runs mostly reach the
  fence stall deterministically.
