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

## Proposed sequence

1. **Spikes (hours each, no commitment):**
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
