# Architecture — `prosper`

> A PS5 (Prospero) → Windows/Linux user-space compatibility layer.
> Proton-for-PS5: **not** a CPU emulator. We run the guest's native x86-64 code
> and translate the operating system, ABI, and GPU underneath it.

## The core idea

```
        ┌──────────────────────────────────────────────────────────┐
        │  GUEST (runs natively, unmodified)                         │
        │   eboot.bin (Unity player) + *.prx (game code, libc, ...)  │
        │   x86-64 machine code, System V ABI, FreeBSD syscalls      │
        └───────────────┬───────────────────────┬──────────────────┘
                        │ Sony library calls     │ FreeBSD syscalls
                        │ (resolved via NID)      │ (SYSCALL instr / libkernel)
        ┌───────────────▼───────────────────────▼──────────────────┐
        │  PROSPER (our host process)                                │
        │  ┌────────────┐ ┌───────────────┐ ┌────────────────────┐  │
        │  │  Loader    │ │  HLE modules  │ │  Kernel HLE         │  │
        │  │ SELF/ELF   │ │ libSceAgc →   │ │ threads→host        │  │
        │  │ map+reloc  │ │  Vulkan       │ │ mem→mmap/VirtualAlloc│ │
        │  │ NID resolve│ │ VideoOut→win  │ │ sync, fs, timers    │  │
        │  └────────────┘ │ Pad→SDL       │ │ syscall trap (SUD)  │  │
        │                 │ AudioOut→host │ └────────────────────┘  │
        │                 └───────────────┘                         │
        │  ABI shim (guest SysV ⇄ host) · Shader recompiler (→SPIR-V)│
        └───────────────┬────────────────────────────────────────────┘
                        │
        ┌───────────────▼──────────────────────────────────────────┐
        │  HOST OS: Windows (Win32 + D3D12/Vulkan) / Linux (Vulkan) │
        └────────────────────────────────────────────────────────────┘
```

## Components

### 1. Loader (`src/loader`)
- Parse SELF wrapper → inner ELF; build VA→file map from `PT_LOAD` data segments.
- Reserve a guest address region in the host process; map each segment at
  `base + p_vaddr` with correct R/W/X protections (honoring `GNU_RELRO`).
- Apply relocations (`DT_RELA`, `DT_JMPREL`; types `R_X86_64_RELATIVE`,
  `GLOB_DAT`, `JUMP_SLOT`, `DTPMOD64`/`DTPOFF64` for TLS).
- Load dependent `*.prx` modules, build the module graph.
- Resolve imports: each undefined NID symbol → address of our HLE stub (below).
- Set up TLS blocks, `sce_process_param`, stack, and jump to entry.

### 2. NID resolution & HLE dispatch (`src/hle`)
- A registry maps `(libraryName, NID)` → host function pointer.
- NID = first 8 bytes of `SHA1(symbolName + <sony-suffix-salt>)`, base64. We keep a
  `nid_db` (name↔NID) generated from a known-symbol list so logs show real names.
- Unimplemented NIDs resolve to a generated **logging trap stub** that records the
  call and (initially) aborts — this is how we discover what the game needs, in order.
- Implemented modules register real handlers.

### 3. Kernel HLE (`src/hle`, with the host layer in `src/host`)
Maps FreeBSD/libkernel semantics onto the host:
- **Threads** → host threads (pthreads / Win32). Scheduling is cooperative-native.
- **Memory** → `sceKernelMapNamedFlexibleMemory`/`sceKernelReserveVirtualRange`
  onto `mmap`(`MAP_FIXED`) / `VirtualAlloc`. Guest expects a specific VA layout.
- **Sync** → mutex/cond/sema/event-flag on host primitives;
  `sync_on_address` → futex / `WaitOnAddress`.
- **File I/O** → guest `/app0/…`, `/savedata/…` mapped to host dirs.
- **Syscalls**: FreeBSD `SYSCALL` instructions inside guest code are caught via a
  handler (Linux: `SIGSYS`/seccomp-trap or libkernel interception; Windows: veh)
  and dispatched to our syscall table. Most go through `libkernel` wrappers, so
  raw `SYSCALL` interception is a fallback.

### 4. Graphics (`src/gpu`) — the hard part
- `libSceVideoOut` → create host window + swapchain (SDL3 + Vulkan; D3D12 later on Win).
- `libSceAgc`/`AgcDriver` → intercept command-buffer submission; translate PS5
  command packets and pipeline state into Vulkan command buffers.
- **Shader recompiler**: the game ships GPU shaders as **RDNA2/GCN ISA** (or an
  intermediate). Recompile to **SPIR-V**. This is a self-contained sub-project
  (decoder → SSA IR → SPIR-V emitter), the biggest single effort in the whole layer.

**Implementation status (built, all execution/pixel-verified):** the pipeline is
`pm4_decode` → `command_processor` (fold a Dcb into a `GpuState`) → `render_state` /
`resolve_pipeline_state` → `vk_translate` → a real `VkGraphicsPipeline`, plus a resource-layer
contract (`gpu_resources`). The **RDNA2→SPIR-V recompiler** (`rdna2_decode` + `rdna2_to_spirv`)
handles ~52 ALU ops + convert/compare/select/bitfield/pack and **divergent control flow** (EXEC
per-lane predication, `saveexec`/restore, forward `s_cbranch_execz`), at ~67% instruction coverage
over the game's real shaders (measured by `recompile_coverage`/`shader_histo`). Remaining: `SMEM`/
`MUBUF`/`MIMG` memory ops (need the resource-binding model) and loops (backward branches). The live
boot blocker is upstream — Unity's completion-event-driven residency pass never fires under our
headless equeue (see `docs/GRAPHICS.md`).

### 5. Peripherals (`src/io`, future — not yet a module)
- `libScePad` → SDL_GameController (DualSense passthrough where possible).
- `libSceAudioOut(2)` → SDL audio / miniaudio; `libSceAjm` → decode ATRAC9/AAC via
  ffmpeg or a dedicated ATRAC9 decoder.
- `libSceAvPlayer` → ffmpeg-backed video → texture.

## The ABI question (why host choice matters)

Guest code is **System V AMD64**. Host-side HLE functions are called *by* guest code.
- **Linux host:** host functions are already SysV → direct calls, near-zero glue.
  Guest `mmap` layout, `dlopen`-style module handling, and signals all line up.
  → **Fastest path to the first "it runs" milestone.**
- **Windows host:** host functions are Microsoft x64 (different arg registers,
  shadow space, callee-saved set). Every guest→HLE call and every HLE→guest
  callback needs a **trampoline** converting the convention. Also TLS (`%fs`/`%gs`),
  SEH, and `VirtualAlloc` granularity differ. All solvable (Wine does it), but more glue.

**Plan:** the loader, HLE registry, kernel logic, GPU translation, and shader
recompiler are all **host-agnostic C++**. Only a thin `src/host/{win,linux}` layer
(memory, threads, ABI trampolines, window) is platform-specific. We build the
agnostic core first; the first *executing* milestone is validated on whichever
host we prioritize.

## Tech stack
- **Language:** C++20 (matches the de-facto reference impls; unavoidable raw-memory/FFI surface).
- **Build:** CMake + Ninja. Toolchain: gcc/mingw or clang on Windows, gcc/clang on Linux.
- **Deps (later):** SDL3 (window/input/audio), Vulkan-Headers + volk, SPIRV-Tools, ffmpeg.
- **No external deps for M0–M2** (loader + stub framework are self-contained).
