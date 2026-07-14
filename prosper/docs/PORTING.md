# Porting prosper to Windows, macOS, and Android

*Investigation, 2026-07-13. Status: proposal — nothing here is implemented beyond the existing
partial `_WIN32` compilation of the pure subsystems.*

## TL;DR

prosper has two independent porting problems, and they have different answers:

1. **The OS problem** (POSIX/Linux → Win32 / Darwin / Bionic). Small and bounded: the Linux
   coupling is concentrated in `src/host/exec_image_linux.cpp` plus a handful of HLE spots, and
   the actually-used POSIX surface is ~15 primitives. Answer: a **thin per-OS substrate**
   (`exec_image_<os>.cpp` + small shims), not Cygwin and not a full POSIX reimplementation.
2. **The CPU problem** (x86-64 guest code on ARM hosts). prosper executes the guest's x86-64
   machine code natively (`run_entry` ends in `jmp *%rax`), so on Apple Silicon and Android
   *everything* depends on an existing translation layer (we will not write a JIT). Answer:
   **pick the translator per platform, and let that choice decide how much OS porting is even
   needed** — because the mature translators (Rosetta 2, FEX, Box64) translate *whole processes*,
   running the **existing Linux x86-64 build under them** is often the entire port.

Recommended shape: **one portable core, two native OS targets, translators for all ARM**.

| Platform | CPU strategy | OS strategy | Effort | Confidence |
|---|---|---|---|---|
| Linux x86-64 | native | current code | — | exists |
| Windows x86-64 | native | new Win32 substrate (`exec_image_win.cpp`) | moderate–large | HIGH it works; two hard sub-problems |
| Windows (interim) | native | **WSL2 — works today** | zero | proven (it's the dev setup) |
| macOS (Apple Silicon + Intel) | **Rosetta 2** (build the app as x86_64 Darwin) | Darwin substrate (small delta from Linux) | small–moderate | HIGH (shadPS4 + Apple GPTK use exactly this architecture) |
| macOS post-Rosetta hedge | Rosetta-for-Linux or FEX inside a lightweight VM | **unmodified Linux build** | packaging only | MED (GPU path is the open question) |
| Android arm64 | **Box64** (later maybe FEX) | **unmodified Linux x86-64 build** in an app-private rootfs (Winlator model) | moderate packaging, no core port | MED — most speculative target |
| Windows on ARM | Prism (built-in, AVX2 since 24H2) | the Windows x64 build, unmodified | zero extra | MED |

Recommended order: **macOS first** (smallest delta, developable on the M2 laptop, and it forces
the platform-abstraction seams), **Windows native second** (biggest audience; WSL2 covers it
meanwhile), **Android last** (packaging-heavy, performance-bound, no core changes).

---

## What is actually Linux-specific (inventory)

A full sweep of the tree (2026-07-13) found the coupling is much narrower than "the core is POSIX"
suggests:

**The guest-execution substrate — the real port surface.** All of it in
`src/host/exec_image_linux.cpp` (one 144 KB file, entirely `#ifdef __linux__`), plus
`boot_program.cpp` and `guest_tls.cpp`:

- `mmap` with `MAP_FIXED` / `MAP_FIXED_NOREPLACE` at fixed guest bases (eboot `0x400000000`,
  modules up to `0x600000000` stubs; GPU-VA window `0x100000000`–`0x1000000000`; guest heap seen
  up to ~`0x1730000000`), RWX for bring-up.
- `sigaction`/`sigaltstack`/`ucontext` fault+trap handler (SIGSEGV/SIGBUS/SIGILL/SIGTRAP):
  SSE4a (#UD) emulation, lazy page commit, software watchpoints via `mprotect`+single-step
  (EFLAGS.TF), `sigsetjmp` recovery, raw `SYS_write` inside handlers.
- `perf_event_open` hardware break/watchpoints (`PROSPER_HWBP`/`PROSPER_HWWATCH`),
  `/proc/self/maps` classification.
- Guest `%fs` TLS (`PROSPER_GUEST_FS`): per-thread guest TCB, base switched with
  `rdfsbase`/`wrfsbase`; import stubs are hand-emitted x86 that swap `%fs` per HLE call.
- x86 machine-code emitters for import stubs; `int3` patching for `PROSPER_BP`.

**HLE Linuxisms (small list):**
- `hle_kernel_mem.cpp`: `memfd_create` + dual `MAP_SHARED` mappings for CPU/GPU-aliased dmem;
  `SYS_futex` for `sceKernelWaitOnAddress` (also `sync_futex.cpp`, which already has a non-Linux
  no-op fallback). Note the PS5 granularity default is already `0x4000` (16 KB) throughout.
- `hle_kernel.cpp`: glibc-isms `pthread_getattr_np`, `pthread_sigqueue`; the guest-thread
  trampoline is `#ifdef __linux__`.
- `hle_file.cpp` / `hle_kernel_time.cpp`: already have `_WIN32` branches.

**Already portable by design:**
- `src/gpu/` is Vulkan-free data translation (PM4→draw state, RDNA2→SPIR-V). No OS calls.
- The Vulkan edge (`tests/render_runner.h`, `frontends/shared/live_renderer.cpp`) is **headless
  Vulkan 1.1**, requiring only `robustBufferAccess` (+ optional `VK_EXT_image_robustness`,
  `samplerAnisotropy`). No geometry shaders, no descriptorIndexing, no BDA. Extremely portable.
- Presentation (`frontends/prosper-app`) is **SDL3** with a deliberately separate present device;
  frames cross as CPU pixels. SDL3 covers Windows/macOS/Android.
- Loader, SELF parsing, NID linking, most HLE, and the pure test suite already compile under
  `_WIN32` (the current MinGW build).

**What was *not* found (good news):** no epoll/eventfd/timerfd/shm_open/dlopen/ptrace/clone/
seccomp/arch_prctl. The POSIX surface in real use is: mmap/mprotect, memfd, futex, signals +
ucontext, perf_event (diagnostics only), pthreads, clock_gettime, open/read/dirent.

---

## Strategy question: cross-platform app, separate apps, Cygwin, or own POSIX?

**Not Cygwin/MSYS2 runtime.** Cygwin supplies the POSIX *API*, not the semantics prosper needs:
signal/`ucontext` fidelity inside a translation DLL, no `perf_event_open`, no `memfd`, emulated
`fork`-era process model, and a runtime that itself occupies address space inside our carefully
fixed guest layout. Every serious emulator on Windows (RPCS3, shadPS4, Dolphin) is native Win32.
(MinGW, which the repo already uses, is unrelated to Cygwin — it *is* native Win32.)

**Not a POSIX implementation.** We don't use POSIX; we use ~15 primitives. Reimplementing POSIX
means reimplementing the 95% nobody calls.

**Yes: one cross-platform core with a thin host-substrate layer.** `src/host/exec_image.hpp` is
already the interface and the Linux file is already named `exec_image_linux.cpp` — the seam
exists. Add `exec_image_darwin.cpp` and `exec_image_win.cpp`, and factor four tiny shims used by
both substrate and HLE:

- `host_mem`: reserve/commit/protect at fixed addresses; dual-mapped ("aliased") memory
  (Linux `memfd` / Windows `CreateFileMapping`+`MapViewOfFile3` / Darwin `shm_open` or
  `mach_make_memory_entry_64`+`mach_vm_map`).
- `host_futex`: wait/wake on address (Linux `SYS_futex` / Windows `WaitOnAddress` —
  a near-exact equivalent / Darwin `os_sync_wait_on_address`, macOS 14.4+).
- `host_fault`: install fault/trap handler with a portable register-context view
  (Linux `ucontext.gregs` / Windows VEH `CONTEXT` / Darwin `__darwin_mcontext64`).
- `host_hwbp`: hardware break/watchpoints (Linux `perf_event_open` / Windows `Dr0–Dr3` via
  `SetThreadContext` / Darwin: unavailable → fall back to the existing software-watch and `int3`
  paths; HWBP is a diagnostic, not required to run games).

The frontends stay one cross-platform SDL3 app; Android gets its own thin app shell (below).
So: **cross-platform app, not separate apps** — with the Android distribution being a packaging
of the Linux build rather than a port.

---

## Windows (x86-64): native port

Guest code runs natively — no translator involved. The substrate port is well-trodden emulator
territory (RPCS3/shadPS4 precedents): VEH for SIGSEGV/SIGILL/SIGTRAP equivalents (`CONTEXT` has
full registers and works with EFLAGS.TF single-stepping), `VirtualAlloc2`/`MapViewOfFile3` for
fixed-address reservation, placeholders and dual mappings (Win10 1803+), `WaitOnAddress` for
futex, debug registers for HWBP/HWWATCH, `VirtualQuery` instead of `/proc/self/maps`.

Two genuinely hard sub-problems:

1. **SysV ⇄ MS-x64 ABI** at every guest↔host call boundary (already flagged in
   `ROADMAP.md:297`). With clang or MinGW-GCC this is mostly *free*: mark HLE entry points
   `__attribute__((sysv_abi))` so guest code calls them directly; host→guest calls (init arrays,
   `run_entry`, callbacks) get the inverse annotation on the function-pointer types. A handful of
   variadic/trampoline cases need hand care. Full hand-written trampolines are the fallback, not
   the plan.
2. **Guest `%fs` TLS.** Windows gives user mode no reliable way to own FS base: there is no API,
   `wrfsbase` availability/persistence across context switches is undocumented, and the kernel
   validates/rewrites segment state (documented for GS/TEB; FS behavior must be assumed hostile).
   shadPS4 — same FreeBSD-style x86-64 guests — solves this on Windows by **patching/trapping
   guest FS-relative accesses** instead of owning FS. prosper already decodes instructions at
   fault sites (SSE4a path), so trap-and-fix or patch-at-load is in-house technology. Spike this
   FIRST (a 100-line probe: `wrfsbase` + spin loop + check for clobber) — if modern Win11
   preserves user FS base, the existing swap-stub design ports directly; if not, plan the
   patching route. CONFIDENCE: MED on mechanism choice, HIGH that one of them works.

**Interim (works today):** WSL2 + WSLg runs the full Linux build including the SDL3 app — that is
literally the current dev environment. "Available on Windows" can be true immediately with a
packaged WSL2 distribution while the native port proceeds.

**Windows on ARM** comes along for the ride: Prism (the built-in x86-64 emulator, AVX2 support
since 24H2) runs the native Windows x64 build unmodified.

---

## macOS: x86_64 build under Rosetta 2 (primary), VM as hedge

**Do not build arm64-native.** There is no in-process x86-64 translator on macOS (Rosetta is
process-granular; FEX targets Linux only), so an arm64 prosper could never run guest code.
Instead build the whole app **x86_64 Darwin** (`CMAKE_OSX_ARCHITECTURES=x86_64`) and let
Rosetta 2 translate everything — host and guest alike. This is precisely the architecture of
Apple's Game Porting Toolkit (x86-64 Wine under Rosetta) and of shadPS4's macOS builds, so the
class of workload is proven.

What Rosetta buys us on Apple Silicon:

- x86-64 execution including **AVX2 (macOS 15+)**; TSO memory ordering in hardware.
- **4 KB pages** inside the translated process — sidesteps the entire 16 KB Apple-Silicon page
  problem that plagues Linux-side emulators.
- Intel Macs run the same binary natively (bonus, while they last).

Darwin substrate deltas (all small; `exec_image_darwin.cpp` will look ~85% like the Linux file):

- `MAP_FIXED_NOREPLACE` → probe + `MAP_FIXED`, or `mach_vm_map` without overwrite.
- `memfd_create` dual mapping → `shm_open` or `mach_make_memory_entry_64` + double `mach_vm_map`.
- `SYS_futex` → `os_sync_wait_on_address` (14.4+); `sync_futex.cpp` already has the seam.
- `ucontext.gregs[REG_RIP]` → `uc->uc_mcontext->__ss.__rip` etc.; signals/sigaltstack same shape.
- `pthread_getattr_np` → `pthread_get_stackaddr_np`/`pthread_get_stacksize_np`;
  `pthread_sigqueue` → `pthread_kill`.
- `perf_event_open` HWBP → not available; keep software watch + `int3` paths.
- Guest `%fs`: Darwin has no fsbase API either. **Spike first:** does Rosetta execute
  `wrfsbase`/`rdfsbase`? (It plausibly does — Rosetta-for-Linux must support x86 TLS via
  `arch_prctl`, so segment-base emulation exists inside the translator.) If yes, the existing
  swap stubs work unchanged. If no, same patch/trap strategy as Windows.
- **SSE4a risk:** guest code contains Zen2 `INSERTQ`/`EXTRQ`, which Rosetta (Intel feature set)
  won't implement. Verify a SIGILL is delivered catchably; if Rosetta instead fails translation,
  pre-patch those instruction sites at load (we already decode them — `sse4a.hpp`).

GPU: **MoltenVK** (user-specified, and correct — ships universal, works in x86_64 slices, nearly
conformant Vulkan 1.4, far beyond our Vulkan 1.1 needs). One caveat: MoltenVK *advertises*
`robustBufferAccess` but does not enforce it (Metal has no equivalent), so device creation
succeeds but out-of-bounds guest buffer reads are not clamped — if artifacts appear that Linux
doesn't show, suspect this first; the recompiler can emit manual bounds checks if it ever
matters. (Longer term, Mesa's KosmicKrisp offers conformant Vulkan on Apple Silicon, but arm64 /
macOS 26+, so it only pairs with the VM path.)

**The Rosetta clock is ticking:** full Rosetta support lasts through macOS 27 (fall 2026); from
macOS 28 (fall 2027) Apple keeps only a subset "for older, unmaintained gaming titles." An
emulator running games might even fit that carve-out, but don't bet the platform on it. Hedge,
which is also the zero-port option: run the **unmodified Linux x86-64 build** in a lightweight
Linux VM — either Virtualization.framework with **Rosetta-for-Linux** (Apple-supported, fast) or
FEX/muvm in a 4 KB-page guest (fully open, slower). The open question there is GPU: Venus
paravirtualized Vulkan over MoltenVK (the krunkit stack) exists but is young; prosper's
CPU-pixel present model at least makes display trivial. Treat the VM route as the durable
fallback, not the first deliverable.

---

## Android (arm64): package the Linux build, don't port

Android is Linux under the hood, but Bionic/SELinux/app-model differences make a *native* Android
port of the substrate both painful and pointless — because the only viable CPU translators here
(**Box64**, community FEX builds) translate **Linux x86-64 processes** anyway. The proven model is
Winlator: an APK bundling the translator, a minimal x86-64 rootfs, and (on Qualcomm) the Turnip
Vulkan driver, running real Windows games via Wine+Box64 on stock unrooted phones. prosper's ask
is strictly smaller than Winlator's — no Wine, no DXVK; just one Linux x86-64 binary that speaks
Vulkan 1.1 and reads a game dump.

- **Translator: Box64 first** — Android-proven (Winlator), ships an Android build flavor, and
  supports 16 KB-page hosts (since 0.2.8, for Asahi). FEX is faster/more rigorous but officially
  "Android is not and will never be a target"; only community ports (FEXDroid/Termux) exist, and
  it hard-requires 4 KB pages. Revisit if the ecosystem shifts.
- **GPU:** the translator thunks Vulkan to the native arm64 driver (Box64's wrapped-libs
  mechanism — the same way Winlator feeds DXVK output to Turnip). Adreno+Turnip is the quality
  path; Mali/Xclipse are weak. Our tiny feature footprint (Vulkan 1.1 + robustBufferAccess) is
  about as mobile-friendly as a desktop workload gets.
- **Page size:** new 16 KB-page devices are arriving (Play requires 16 KB *app* support for
  Android 15+ targets since 2025-11; Android 16 adds a 4 KB-app compat mode). prosper is oddly
  well-positioned: the PS5's own mapping granularity is 16 KB and the HLE already defaults
  alignment to `0x4000`. Residual risk is 4 KB-grain `mprotect`/ELF-segment layout; Box64's 16 KB
  support absorbs some of it. Prefer 4 KB devices for bring-up.
- **App shell:** a thin Kotlin frontend (SurfaceView + storage access for the user's dump) that
  launches the emulator in the app's private rootfs. SDL3's Android backend can replace the
  desktop present path later.
- **Expectations:** this is the most speculative target — flagship-Snapdragon-only at first, and
  performance-bound (though the current titles are 2D indies, which is the plausible envelope
  for Box64). Do it last; it requires *no core changes*, only packaging, so nothing is lost by
  deferring.

---

## Tooling portability

The debugging toolbox (`tools/`, see `tools/AGENTS.md`) splits cleanly:

- **Pure file/Vulkan tools — portable now, no substrate needed:** `gpu_replay`, `gpu_timeline`,
  `self_dump`, `shader_histo`, `imgdump`, `spv_validate`, `niddiag`, `il2cpp/`, `re/xref.py`.
  None of them execute guest code. In particular **`gpu_replay` replays `.prgcap`/`.prgbundle`
  capsules through the full Vulkan backend without booting the guest**, and exits non-zero on
  output-hash mismatch — so an **arm64-native macOS build of `gpu_replay` against MoltenVK,
  replaying captures made on Linux**, is the cheapest possible pathfinder for the entire GPU
  stack on Apple (recompiled SPIR-V, descriptors, RTT, and the robustBufferAccess caveat),
  before any Rosetta or substrate work. This should precede the Rosetta spike.
- **Substrate-bound:** `boot_trace` (ports with `exec_image_<os>`), the `snapshot` golden-image
  guard (boots a real title — once boot works on a platform, pixel-hash regression verification
  comes along for free), and the `PROSPER_*` runtime diagnostics (all portable except the
  `perf_event`-based `PROSPER_HWBP`/`HWWATCH`; the software-watch and `int3` paths cover other
  OSes).
- **Deliberately Linux-only:** `pad_evdev` (SDL3 pads elsewhere) and the gdb-based `tools/dbg/`
  scripts.

## macOS port status (2026-07-13) — landed, in progress on branch `port/macos-core`

The macOS x86_64/Rosetta path is now real, not theoretical:

- **Substrate ported.** `src/host/posix_shim.hpp` supplies the Darwin equivalents of the ~15
  Linux/glibc primitives the emulator uses (mach `process_vm_*`, `os_sync_wait_on_address` futex,
  `shm_open` memfd, `MAP_FIXED_NOREPLACE` emulation, timed locks, real semaphore/barrier, an
  indexable mcontext register view, Mach-O global-asm). `exec_image_linux.cpp`, `boot_program.cpp`,
  `guest_tls.cpp`, `hle_kernel_mem.cpp`, and the thread/exception paths in `hle_kernel.cpp` compile
  and run on Darwin.
- **Everything green.** Full x86_64 build on an Apple M2; **67/67 ctest under Rosetta 2** (was a
  55-test pure subset before; the substrate tests — trap/boot/setjmp/stack/AGC/videoout/prot_none —
  now run too). The Rosetta spike confirmed MAP_FIXED at the guest bases, RWX self-modifying
  execution, catchable SIGILL, `os_sync_wait_on_address`, and 4 KB pages. Two spike facts shaped the
  port: **Rosetta does not implement `wrfsbase`/`rdfsbase` (SIGILL)**, and MAP_FIXED at
  `0x100000000`/`0x1000000000` is refused (Rosetta reserves the 4 GiB and 64 GiB windows) — the
  module bases (`0x4xx…`, `0x5xx…`, `0x6xx…`) map fine.
- **A game boots into guest code.** `boot_trace PPSA13579-app0` (Blasphemous 2, Unity/IL2CPP) links
  all 7 modules, resolves 3554 imports (408 cross-module), dispatches every `.init_array`, and
  executes real guest x86-64 under Rosetta — reaching the C runtime's first constructor running libc
  initialization.
- One correctness fix fell out of the port (all platforms): the unannounced-32-bit index-buffer
  fingerprint (#304) read the same bytes through `uint16_t*` and `uint32_t*` — strict-aliasing UB
  that Apple Clang 21 compiled into `ud2`. Now `memcpy` loads.

### The macOS frontier: guest `%fs` TLS

Boot stalls a few frames into libc init: a guest function does `call *rax` where `rax` resolved to
a stack address, faulting (caught cleanly now, thanks to a mandatory alt stack on Darwin). The root
cause is the predicted one — **guest initial-exec `%fs` TLS**. The Messenger needed
`PROSPER_GUEST_FS` on Linux for correct TLS (see `guest_tls.cpp`: without it, guest `%fs:`-relative
reads alias the host glibc TCB and return garbage). On macOS that gate is *force-disabled* because it
is implemented with `wrfsbase`, which Rosetta SIGILLs on — and Darwin's `%fs` base is 0 (macOS uses
`%gs` for TLS), so the guest's TLS reads resolve to garbage/stack values immediately, deeper and
earlier than on Linux.

This is exactly the hard sub-problem flagged above for both Windows and macOS. Path forward
(unchanged): find Rosetta's segment-base mechanism, or **patch/trap the guest `%fs`-relative
accesses** — the fault handler already decodes instructions at fault sites (SSE4a), so
trap-and-emulate of `%fs:` operands is in-house technology.

## Windows port status (2026-07-13) — the guest now BOOTS through module init into guest code

The foundation (below) was compile-verified via MinGW-w64; since then a **Windows host has
runtime-verified the boot**. The guest now links its modules, runs its module-init functions,
reaches the real entry point, and executes deep guest code — stopping only at the confirmed
**guest `%fs` TLS wall** (details below). `boot_trace.exe` goes from "crash at the first init fn"
to a full `RUN ENDED` report. Three runtime bugs were fixed to get here (commits on
`port/windows-core`):

- **Direct/flexible-memory HLE ported to Win32** (`hle_kernel_mem.cpp` `#else` block). The pool
  bookkeeping + VA tracker are copied verbatim from POSIX; mappings are backed by private
  `VirtualAlloc`/`VirtualProtect`/`VirtualFree`. Phys-offset aliasing is NOT preserved (Win32 view
  granularity is 64 KiB vs the guest's 16 KiB) — fine for Unity/IL2CPP (Messenger), a follow-up for
  UE4 MallocBinned3. The guest now allocates + maps its pools.
- **Host→guest SysV call trampoline** (`prosper_call_guest_sysv`). `run_guest_inits` called the guest
  init fns with a plain C call, which on Windows uses the MS x64 ABI (args in rcx/rdx) while the guest
  reads SysV (rdi/rsi) — the `module_start` got a garbage `argp` and jumped wild. The asm shim marshals
  argc→rdi/argp→rsi and preserves the MS callee-saved regs the SysV callee may clobber (rsi/rdi/xmm6-15).
- **VEH recovery uses `__builtin_setjmp`/`__builtin_longjmp`, not the CRT pair.** The CRT `longjmp`
  does a full SEH unwind (`RtlUnwindEx`) from the fault site back to `setjmp`, which cannot traverse
  the guest frame or the hand-written trampoline (no `.pdata`/`.xdata`) → `STATUS_STACK_OVERFLOW`
  instead of recovery. The `__builtin_*` pair restores rsp/rbp/rip with no unwind (matching Linux).
  Also: `run_entry` now points `NT_TIB.StackBase/StackLimit` at the switched guest stack during guest
  execution so exception dispatch on that stack doesn't spuriously report stack exhaustion.

### Guest `%fs` TLS — SOLVED (FSGSBASE + VEH re-apply)

Was: boot stopped at `eboot+0x808f35` (`mov %fs:0x0,%rax` faulting at `addr=0x0`, zero Sony imports
called) — uninitialized guest initial-exec TLS, guest `%fs` base 0 on Windows. An FS-base probe
settled the strategy: **user-mode `rdfsbase`/`wrfsbase` work** (CR4.FSGSBASE, Win10 1709+), but
**Windows resets the user FS base to 0 on every kernel transition** (a `Sleep()` zeroed it; it
restores the thread's kernel-saved base and `wrfsbase` doesn't update that copy), so "set once" is
impossible. Unlike Linux, the *host* uses `%gs` and only the *guest* uses `%fs`, so no per-HLE-call
swap is needed.

Implemented: a real Windows `guest_tls.cpp` TCB (Variant-II layout, `VirtualAlloc`-backed, self-ptr +
magic + guest canary), `wrfsbase(TP)` on each guest entry/thread start, and a VEH hook that re-applies
`wrfsbase(TP)` and retries whenever an `fs`-prefixed instruction faults with the base drifted (one
fault per kernel-transition boundary, not per access; a loop guard only re-applies when the base
actually changed). `PROSPER_NO_GUEST_FS=1` reverts to the old wall, confirming this is the enabler.
Result: boot advances far past the wall into deep guest code.

### Guest allocator + worker threads — SOLVED (lazy-commit + worker trampoline)

The deep crash after `%fs` TLS was two more Linux-parity gaps (fixed on `port/windows-boot-2`, PR #628):
- **Lazy-commit.** The guest's binned allocator writes into pages inside a range it RESERVED but never
  explicitly committed. Linux's SIGSEGV handler lazily backs these on first touch; the Windows VEH did
  not, so allocator init faulted at a reserved page (diagnosed with a `[memclass]` fault classifier that
  showed the fault address as reserved-but-uncommitted). The VEH now `VirtualAlloc(MEM_COMMIT)`s the
  64 KiB page on a tracked-reserved fault and retries.
- **Worker-thread ABI + TLS.** The Windows `pthread_create` path called the guest entry directly via
  winpthreads: MS-x64 ABI (arg in rcx, guest reads rdi) and no `guest_tls_activate_thread()`. The first
  worker got a garbage arg and no guest TCB and crashed on a null-derived deref (`mov 0x38(%rbx)`,
  rbx=0) on an unrecoverable thread. A `win_thread_trampoline` now marshals the SysV entry through
  `prosper_call_guest_sysv` and activates the worker's guest `%fs` TCB.

### Renderer wired (#655); WaitOnAddress (#663) + positioned IO (#665) FIXED; frontier: GPU EOP completion

**SOLVED — the pre-render wedge was a lost wakeup in the Windows `sceKernelWaitOnAddress`.** It had been
backed by ONE global `std::condition_variable` shared across every waited address, so a guest
`WakeByAddress(n=1)` → `notify_one()` could wake a waiter parked on a DIFFERENT address (it re-checked its
own word, found it unchanged, re-slept) while the intended waiter was never woken. That randomly lost
Unity's job/thread startup handshakes → the boot nondeterministically wedged (2 vs ~45 threads, or a
worker fault). Re-implemented on the **native Win32 futex** (`WaitOnAddress` / `WakeByAddressSingle` /
`WakeByAddressAll`, needs `-lsynchronization`), a true per-address wait exactly like the Linux
`FUTEX_WAIT` path — correct `n=1` semantics, no thundering herd. The boot is now **deterministic** and
goes far deeper: it spawns the full Unity thread ecosystem (13 `AssetGarbageCollectorHelper`,
`Job.Worker 0-12`, `Background Job.Worker 0-15`, `Loading.AsyncRead`, `BatchDeleteObjects`) and reaches
**VideoOut display setup** — `RegisterBuffers2`, "display surface: 1920x1080, 3 buffers registered",
`ConfigureOutput`, `GetOutputStatus`.

**SOLVED #2 — asset streaming (positioned file IO).** After the WaitOnAddress fix the guest reached
VideoOut setup then stalled again; root cause: Windows `sceKernelPread`/`Pwrite`/`readv`/`writev`/
`preadv`/`pwritev` all returned -1 (MinGW has no POSIX `pread`, so they were stubbed). Unity's async
asset streamer (FileCacher/CachedReader on `Loading.AsyncRead`) reads assets via **positioned reads**, so
nothing loaded. Backed them with `ReadFile`/`WriteFile` + an `OVERLAPPED` offset (atomic positioned IO,
thread-safe on a shared fd, full-read loop for the PS5 full-count contract) — #665. `f_read`/`f_write`
also loop now. (`libSceSystemService::mPpPxv5CZt4` = `sceSystemServiceGetHdrToneMapLuminance` is a benign
HDR stub returning 0, tracked in #664 — not a blocker.)

### Current frontier: the GPU EOP-completion handshake (first rendered frame)

With #663 + #665 the guest now **streams assets and drives the GPU command stream**: it builds AGC
command buffers and submits, and the executor decodes real PM4 — including `WriteData`, **`ReleaseMem`
(EOP label write)** and **`WaitRegMem` (wait-for-memory)**. It then plateaus (no `[render] frame N` yet):
the guest submits a Dcb and waits (CPU-side `WaitOnAddress`) for that submission's **GPU end-of-pipe
completion** before building the next frame. Getting the first frame is now a GPU-executor question:
confirm the Windows executor folds the submitted Dcb and DELIVERS the EOP completion (the label write +
waking the guest's waiter / the flip-completion equeue event) — the same lost-semaphore/EOP-delivery
class Linux solved in #236 ("deliver every GPU EOP completion, coalesce=false"). The EOP + vblank pump
threads are cross-platform `std::thread`s, so the machinery exists; verify it actually fires on Windows
for the guest's submitted work. Diagnostics: `PROSPER_GFXLOG` (`[gfx]`/`[agc]` PM4 decode), `PROSPER_EVLOG`
(`[ev]` equeue/flip/EOP events), `PROSPER_SYNCLOG`.

(Historical, pre-fix diagnosis retained below for context.)

The live Vulkan renderer builds + initializes on Windows (#655): the whole GPU/Vulkan translation
layer + `prosper_live_renderer` compile under MinGW (the Vulkan SDK is found via `VULKAN_SDK`), and at
runtime the live compute + submit renderers register cleanly. With the renderer wired and
`PROSPER_RENDER=1`, the guest boots into Unity engine init (allocates its pools, spawns
`AssetGarbageCollectorHelper` threads) and (before the fix above) **idle-waited** (0% CPU).

Diagnosis (via `PROSPER_SYNCLOG` WaitOnAddress logging + `PROSPER_GUEST_ARGS=-force-gfx-direct
PROSPER_RENDER=1` + gdb thread inventory). **A thread-inventory diff vs Linux localizes the stall
precisely — it is NOT the later job-dispatch loop, it is the very first GC-helper init handshake:**

- **Linux** (renders 330+ frames, ~37 000 WaitOnAddress wakes) spawns the whole Unity thread ecosystem:
  13 `AssetGarbageCollectorHelper`, `Job.Worker 0..12` + `Background Job.Worker`, `Loading.PreloadManager`,
  `Loading.AsyncRead`, `UnityEOPThread`, `GfxFlipThread`, FMOD threads, `BatchDeleteObjects`, ….
- **Windows** creates only **2 `AssetGarbageCollectorHelper` threads and then deadlocks**: exactly 3
  guest threads remain, ALL parked in `sceKernelWaitOnAddress` (`*addr==expected==0`, no timeout), ~13
  total wakes then silence. None of `Job.Worker*`/`PreloadManager`/`AsyncRead`/`GfxFlipThread` is ever
  created.

**Update — the stall is a RACE, and it wedges on a custom semaphore.** With a validated guest-caller +
thread-id sync log (the `[sync] T<tid> ... caller=0x...` fields), the stuck waits resolve to a
Unity/Sony **custom counting semaphore**: `mov $-1,%eax; lock xadd %eax,0x8(%rdi); test %eax,%eax; jle
<slow>` — decrement a job-count at `[obj+8]`, and if it went ≤0 take the `sceKernelWaitOnAddress` slow
path (sites `eboot+0x18ab088` job-pool and `eboot+0xae1463`). So workers block acquiring jobs the
producer never releases. Crucially the boot is **nondeterministic**: different runs reach very different
depths — sometimes only 2 `AssetGarbageCollectorHelper` threads then wedge, sometimes ~45 guest threads
(much of the Unity set) then wedge, and sometimes a worker hard-faults (exit 139). That variability
means a **startup race** in the sync/thread path, not a fixed missing tick. The global-`std::condition_variable`
`WaitOnAddress` is logically correct under its mutex (no lost wakeup even with the 45-thread thundering
herd), so suspicion falls on thread-startup ordering / the guest's timing assumptions under our slower
per-call sync. Linux (real per-address futex) never exhibits this and renders 330 frames.

(Historical framing, still true of the early-wedge runs:) the `AssetGarbageCollectorHelper` thread
(guest entry `eboot+0xbfbac0`) RUNS on Windows (the worker-thread ABI + `%fs` TLS fixes stopped it
crashing at `mov 0x38(%rbx)`) but on an early-wedge run never signals main ready. Next step: trace what
that helper does on Windows vs Linux — which
`WaitOnAddress`/wake or HLE call it diverges on (its own TLS-derived state, or a wake it should send to
main that never fires). The coarse Windows `WaitOnAddress` (one global `std::condition_variable`; any
wake re-checks all waiters) is correct under its mutex (no lost-wakeup) but is a candidate to revisit.
Diagnostics in place: `PROSPER_SYNCLOG` (`[sync]` WaitOnAddress + `[sync2]` cond/sema/EventFlag),
`PROSPER_MEMLOG`, `PROSPER_VEHLOG`, boot_trace `[memclass]`.

Deferred, lower-priority items (some now done): ~~honor reserve alignment > 64 KiB~~ (done, #658);
~~worker-thread stack registration for GC bounds~~ (done, #658); phys-offset aliasing in the memory HLE
(needed by UE4 MallocBinned3, not this Unity title — `CreateFileMapping`/`MapViewOfFile3`, 64 KiB
granularity); and `PROSPER_CRASHPEEK` guards.

**What is done (compiles + links, in CI):**
- **`exec_image_win.cpp`** — the Win32 sibling of `exec_image_linux.cpp` implementing the full
  `exec_image.hpp` contract: fixed-address guest mapping (`VirtualAlloc` at the guest bases), import
  stub region, a **Vectored Exception Handler** (VEH) fault handler with SSE4a `INSERTQ`/`EXTRQ`
  emulation over `CONTEXT.Xmm*`, lazy-ish recovery via a Rip-redirect to a `longjmp` trampoline,
  `run_entry` (SysV crt0 stack + `jmp`), `run_guest_inits`, the thread-stack registry, and
  `VirtualQuery`-based region classification. Linux-only diagnostics (perf_event HWBP/HWWATCH, int3
  `PROSPER_BP`, PEEK/DUMPAT) are intentionally absent.
- **The SysV⇄MS-x64 ABI boundary** is done in the emitted stub trampoline, NOT via
  `__attribute__((sysv_abi))` on handlers. The attribute route was tried and **abandoned**: on MinGW
  it conflicts with SEH-based C++ exception unwinding (`.seh_handlerdata used outside of .seh_proc
  block`) and cannot be applied to 537 STL-using handlers. Instead `emit_impl` emits a trampoline that
  converts the guest's SysV integer args (`rdi rsi rdx rcx r8 r9`) to MS x64 (`rcx rdx r8 r9`,
  `[rsp+0x20]`, `[rsp+0x28]`, +32B shadow, 16-aligned call) before calling the handler, which stays a
  plain MS-x64 C++ function. Byte encoding disassembly-verified; the register-move ordering is
  clobber-safe by construction. `PROSPER_SYSV_ABI` (dispatch.hpp) is now an empty documented marker.
- `guest_tls.cpp` Windows path (guest `%fs` TLS now IMPLEMENTED via FSGSBASE — see above),
  `boot_program.cpp` enabled on Windows, `boot_trace` built on Windows (no evdev/Vulkan), CI
  `Windows MinGW` job builds the whole boot path.

**What is left (runtime work, on a Windows host):**
1. **The Unity GC-helper init handshake — the current frontier.** The renderer is wired (#655) but the
   guest deadlocks EARLY: it creates only 2 `AssetGarbageCollectorHelper` threads (Linux creates 40+ incl.
   Job.Workers/PreloadManager/GfxFlipThread and renders 330 frames), then all 3 guest threads park in
   `sceKernelWaitOnAddress`. The helper (`eboot+0xbfbac0`) runs but never signals the main thread ready.
   Trace the helper's Windows-vs-Linux divergence (see "the Windows frontier" above) so main advances.
2. ~~Port the direct/flexible-memory HLE~~ — **DONE** (private-`VirtualAlloc`; phys-aliasing deferred,
   only needed by UE4 MallocBinned3, not this Unity title).
3. ~~Guest `%fs` TLS~~ — **DONE** (FSGSBASE + VEH re-apply).
4. ~~Guest allocator lazy-commit + worker-thread ABI/%fs TLS~~ — **DONE** (#628).
5. ~~Wire the Vulkan renderer on Windows~~ — **DONE** (#655: builds + initializes).
6. ~~Reserve alignment > 64 KiB + worker-thread stack registration~~ — **DONE** (#658).
7. **Validate/repair the guest→HLE ABI trampoline for XMM/float args.** Integer args are converted
   and now runtime-exercised through init; **XMM/float args are still not converted** (e.g. some libc
   formatters / `printf`-family) — add float-arg conversion when a title needs it.
8. **VEH recovery hardening for stack-overflow faults** — recovery resumes on the faulting thread's
   current stack; a true stack-overflow fault still needs a guard-page/dedicated-stack story (Linux
   uses `sigaltstack`). The `__builtin_longjmp` change fixed the cross-frame-unwind crash; this is the
   separate genuine-overflow case. Also: `PROSPER_CRASHPEEK` faults on Windows (the IL2CPP klass
   walker needs the same `guest_readable` guards the Linux path has).
9. **Audit the 103 `(HleFn)`-cast handlers** — almost all cast targets are `HLE()`-defined handlers
   and so route correctly through the trampoline, but confirm none are raw host functions relying on
   host-ABI arg passing.

**Reproducing the local compile loop (macOS → Windows):** `brew install mingw-w64`, then
`cmake -S prosper -B build-win -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc
-DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ -DCMAKE_SYSTEM_NAME=Windows
-DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE` and `cmake --build build-win`. A Windows agent builds
natively under MSYS2/UCRT64 exactly as the CI `Windows MinGW` job does.

## Proposed sequence

1. **Spikes (hours each, no commitment):**
   0. Build `gpu_replay` + `render_runner` arm64-native on macOS against MoltenVK and replay a
      Linux-made `.prgcap` — validates the whole GPU stack on Apple with zero substrate work
      (see *Tooling portability*).
   a. On this M2 MacBook: a 100-line x86_64 Darwin probe under Rosetta — `MAP_FIXED` at
      `0x400000000`, RWX anon map + self-modifying jump, `wrfsbase`/`rdfsbase`, catchable
      SIGILL on `INSERTQ`, `os_sync_wait_on_address`. This answers every macOS unknown at once.
   b. On Windows: the FS-base persistence probe (spin thread + `wrfsbase` + verify).
2. **Refactor seams (no behavior change on Linux):** extract `host_mem`/`host_futex`/
   `host_fault`/`host_hwbp`; replace `UNIX AND NOT APPLE` CMake gates with capability checks;
   make `hle_kernel_mem.cpp`/`hle_kernel.cpp` use the shims.
3. **macOS port:** `exec_image_darwin.cpp`, x86_64 preset, MoltenVK + SDL3 frontend. Exit
   criterion: the pure test suite + a dump-gated boot test green under Rosetta on this laptop.
4. **Windows native port:** `exec_image_win.cpp`, `sysv_abi` boundary, FS strategy per spike
   result. WSL2 remains the supported Windows path meanwhile.
5. **Android packaging:** Box64 + rootfs + Turnip APK, Kotlin shell. After macOS/Windows ship.
6. **Hedge track (background):** validate the Linux-VM-on-macOS route (Rosetta-for-Linux or
   FEX/muvm + Venus) before macOS 28 removes general Rosetta.

## Sources

- Rosetta deprecation: [Apple to Phase Out Rosetta 2 Starting With macOS 28](https://www.macrumors.com/2025/06/10/apple-to-phase-out-rosetta-2/), [AppleInsider timeline](https://appleinsider.com/articles/26/06/12/how-and-when-macos-will-finally-stop-support-for-intel-apps), [macOS 26.4 user notices](https://9to5mac.com/2026/02/16/macos-26-4-will-notify-users-of-rosetta-2-discontinuation/)
- Rosetta AVX2 (macOS 15): [Apple docs](https://developer.apple.com/documentation/apple-silicon/about-the-rosetta-translation-environment), [Stockfish AVX2-under-Rosetta issue](https://github.com/official-stockfish/Stockfish/issues/5707), [rosetta2_avx_dive](https://github.com/carsongoodwin32/rosetta2_avx_dive)
- Rosetta for Linux VMs: [Apple Virtualization docs](https://developer.apple.com/documentation/Virtualization/running-intel-binaries-in-linux-vms-with-rosetta), [implementation notes](https://blog.inoki.cc/2026/02/28/Apple-Rosetta-Linux-VM-Secret-en/)
- FEX: [fex-emu.com](https://fex-emu.com/) (AVX2 supported), [4 KB-page requirement / muvm](https://fedoraproject.org/wiki/Changes/FEX), [Android is not a target (FAQ)](https://wiki.fex-emu.com/index.php/FAQ), [FEXDroid](https://github.com/gamextra4u/FEXDroid)
- Box64: [16 KB page support in 0.2.8](https://www.phoronix.com/forums/forum/hardware/processors-memory/1466148-box64-0-2-8-released-with-support-for-16k-page-size-allowing-games-on-apple-silicon), [Winlator](https://github.com/brunodev85/winlator)
- MoltenVK: [State of Vulkan on Apple, Jan 2026 (LunarG)](https://www.lunarg.com/the-state-of-vulkan-on-apple-jan-2026/), [robustBufferAccess not enforced](https://github.com/KhronosGroup/MoltenVK/issues/2447)
- Android 16 KB pages: [Android developer guide](https://developer.android.com/guide/practices/page-sizes), [Play requirement](https://android-developers.googleblog.com/2025/05/prepare-play-apps-for-devices-with-16kb-page-size.html)
- Segment registers per OS: [merryhime's fs/gs notes](https://gist.github.com/merryhime/f22e75d5128c07d77630ca01c4272937)
