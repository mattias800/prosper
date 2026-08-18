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
- **A `PT_LOAD` maps in full, or the module is refused (#2631).** A segment whose declared `p_filesz`
  runs past the end of the file used to be skipped *silently* while `build_image` still returned
  success, so the module mapped a zero-filled hole exactly where its bytes belonged and the loader
  said nothing; the first symptom was the guest executing a page of zeros inside a module that looked
  correctly loaded. (The sibling case — a segment that does not fit the image its own extent just
  sized — was worse than skipped: an overflowing `p_memsz` was left out of the extent computation and
  then **copied anyway**, pushing an `img.prot` record spanning the whole address space. Same
  invariant, same refusal.) Refusing, rather than mapping the surviving prefix and zero-filling the
  rest, is the format's own reading: the gABI defines `p_filesz` as the bytes of the segment *present
  in the file image*, so a short file does not contain the segment at all.
- **Ruled out — "some real dump depends on the lenient reading":** it does not. Measured across the 44
  local dumps (589 modules, 2,945 `PT_LOAD` segments — **exactly five per module**, with no spread):
  **0** whose file bytes run past EOF (none even *ends* at EOF), **0** that do not fit their own mapped
  image, **0** with `filesz > memsz`, **0** excluded from the extent by an overflowing `memsz`, and
  **2,945 of 2,945** SELF data segments whose own `file_size` equals the phdr's `p_filesz` exactly.
  Independently re-derived from the format by a second parser sharing no code with prosper (review of
  #2689): every figure identical.
- **Two traps around that measurement, both of which cost time to find.**
  1. The SELF *header's* `file_size` (container offset `0x10`) is **not** a truncation oracle, and it
     fails in **both** directions. It differs from the physical file in 517 of 589 modules: **452
     smaller** (signature/metadata follow it), **65 larger** — over-declaring by 4 bytes (25 modules),
     8 (32), 12 (6) and 64 (2) — and only **72 equal**. Hand-verified outside the scanner:
     `PPSA01826-app0/prx/akaudioinput.prx` is 10,172 bytes physically and declares `0x27c0` = 10,176.
     An oracle built on this field would flag 65 working modules as truncated and be blind on 72 more.
     *The 517 was first recorded here as "smaller in 517", which is a `!=` count reported as `<` — the
     kind of error this section exists to stop, so it is left visible rather than quietly corrected.*
  2. **"No dump does this" is true of `PT_LOAD`, not of segments generally.** The corpus does contain
     trailing-trimmed content: besides the 65 over-declaring headers, two copies of
     `sce_sys/about/right.sprx` (12,448 bytes, in `PPSA15552` and `PPSA02101`) carry a SELF data
     segment declaring 64 bytes at a file offset *equal to* EOF — zero bytes present. It keys program
     header 10, whose `p_type` is `0x6fffff00` (`SCE_COMMENT`, per `tools/self_dump`), so it is
     neither a `PT_LOAD` nor in `m.loads` and nothing reads it. Extending this refusal past `PT_LOAD`
     — the natural follow-on, and what #2686 gestures at — would reject those two modules on day one.

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

**Implementation status (built and verified in layers):** the pipeline is `pm4_decode` →
`command_processor` (fold a Dcb into a `GpuState`) → `render_state` / `resolve_pipeline_state` →
`vk_translate` → real `VkGraphicsPipeline`s, plus the shared live resource layer. The
**RDNA2→SPIR-V recompiler** (`rdna2_decode` + `rdna2_to_spirv`) handles scalar/vector ALU,
convert/compare/select/bitfield/pack, `SCC`, EXEC-mask predication and structured branches/loops;
`SMEM`, `MUBUF`, `MIMG`, LDS/barriers, exports and interpolation; and reflected runtime descriptor
validation. Runtime-selected descriptor arrays now span device capability acquisition, reflection,
SPIR-V declaration/indexing, pool/layout/write arity and pipeline-cache identity (#2458–#2475).

The remaining work is data-driven rather than one old percentage: unsupported instructions and
unprovable resource/operand paths fail visibly, and `recompile_coverage`, live reject diagnostics and
resource-bearing live/replay captures identify the next exact program counter. A raw `shader_inspect`
dump is useful for proving that an instruction packet exists, but table-dependent rejection remains
unattributable without the real resource table. GTA V is tracked by routed terminal program address +
instruction PC, not by raw structurizer line counts or adjacency in an interleaved stderr stream: the
`structured emission stopped` family is consequent, and most `backward else` reports are secondary to
later instruction, mask-domain or resource failures. The reviewed post-lift stack now includes
Wave32/64 mask and carry, scalar/vector/DPP coverage, guest coherence, zero-mip images, exact-wave
barrier phasing and compact BVH nodes. On the 2026-08-11 gameplay-entry route the exact live set is 29
recompile-empty plus 6 invalid-descriptor programs (35 unique), down from 39 in the preceding tagged
sample with no new program failures; the 3D world remains black (#2481). Counts are route- and
phase-specific, not title-wide totals. See
`docs/GRAPHICS.md`, `docs/RESOURCE_BINDING.md` and
`docs/RECOMPILER_REMAINING.md`, including each document's `## Ruled out` section, before extending
the translator.

### 5. Peripherals and frontends
- `frontends/prosper-app` provides an SDL3 window, Vulkan presentation, controller/keyboard input,
  audio and native dialogs; the headless tools share the same boot/render core.
- `libScePad` input is backed by evdev/SDL3 and can be driven by deterministic `.pad` routes for
  title progression.
- `libSceAudioOut(2)` has a host audio sink; AJM/ATRAC9 and AvPlayer/Videodec paths feed decoded
  audio/video into the guest-visible contracts. Per-title codec/layout gaps remain tracked issues,
  not a future-module placeholder.

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
