<p align="center">
  <img src="assets/prosper-logo.png" alt="prosper" width="256">
</p>

# prosper

**A user-space PlayStation 5 → PC compatibility layer.** Think *Proton/Wine, but for PS5*: prosper
runs PS5 (Prospero) game binaries on Linux (primary) and Windows (secondary) by reimplementing the
console's OS, ABI, and GPU stack on the host — **not** by emulating a CPU.

> ⚠️ **Experimental research project.** It is not a general-purpose game runner yet. Tested retail
> titles currently reach different milestones and still expose substantial compatibility gaps. See
> [Game compatibility](COMPATIBILITY.md) for title-by-title results and known blockers.

## Why no CPU emulation?

The PS5 is an x86-64 (AMD Zen 2) machine, so guest game code runs **natively** on any modern PC.
prosper is therefore shaped like Wine, not like an emulator. The engineering is:

- **Loading** Sony's `SELF`/`ELF` module format and linking multiple modules into one address space,
- **HLE** (high-level emulation) of the PS5 system libraries (`libkernel`, `libc`, `libSceAgc`,
  `libScePad`, `libSceSaveData`, `libSceNp`, the dialog/IME libraries, …) by thunking to the host's
  real facilities (libc, pthreads, `mmap`, `futex`, Vulkan, …),
- **ABI translation** where the guest's FreeBSD/SysV conventions differ from the host, and
- **Graphics translation:** PS5 **AGC** command streams → Vulkan, including a from-scratch
  **RDNA2 → SPIR-V shader recompiler** and the RDNA2 texture-tiling / block-compression decoders.

Because the PS5 and Steam Deck / AMD Linux machines share the same RDNA2 GPU family, Linux is the
natural primary target and the long-term dream is running on that hardware.

## Status

prosper boots **multiple real retail titles** from user-supplied, **unencrypted-segment** dumps. The
tested set spans multiple engines and exercises the loader, host-side OS/ABI layer, services, input,
audio, and graphics stack. See [Game compatibility](COMPATIBILITY.md) for exact per-title progress.

**Boot & runtime (host-side OS / ABI / HLE):**
- ✅ Parses `SELF`/`ELF`, builds a relocatable image, resolves Sony **NID**-hashed imports, links the
  game's modules with a global export table + per-import HLE stubs, and runs the Sony CRT + C++ global
  constructors.
- ✅ Enough `libkernel`/`libc` for real memory management (virtual + direct + flexible memory, guard
  pages), threads (pthreads, TLS, mutexes/conds, event flags, semaphores, `sync_on_address` futex),
  scheduling, time, AIO + positioned file I/O (with `/app0` path translation), and locale/ctype.
- ✅ Loads the C# metadata, spins up the full **IL2CPP GC + worker thread pool**, and boots through
  IL2CPP init and Unity's `GfxDevice` bring-up into the running frame loop.
- ✅ System services the game gates on: user/pad service, SaveData (real per-slot memory blocks),
  AvPlayer / AJM lifecycle, common dialogs + IME (with an optional real SDL3 dialog frontend), NP /
  online (honest signed-out), system-parameter (language) — implemented as faithful behaviors, not
  success-returning stubs.

**Graphics (AGC → Vulkan):**
- ✅ **AGC command frontend:** `sceAgcCreateShader` (relocates the embedded RDNA2 shader ELFs) and the
  submit path; a real submitted `Dcb` is decoded (→ PM4 packets) and folded into a GPU register-state,
  with GPU-completion (EOP) events and DMA/fence writes delivered on the guest's own timeline
  (regression-locked by tests).
- ✅ **RDNA2 → SPIR-V shader recompiler** (we recompile the GPU ISA, not emulate it): full scalar/
  vector ALU (float/int/bitwise/convert/compare/select/bitfield/pack), `SCC`, **divergent control
  flow** (EXEC-mask predication, `saveexec`/restore, `execz` if-then and loop exits), `SMEM`
  constant/descriptor loads, `MUBUF` vertex-fetch + load/store, `MIMG` `image_sample`/`_l`/`_lz`/
  `image_load`, `LDS` + barriers, `EXP` render-target/position/param exports, and `VINTRP`
  interpolation. Descriptors that spill into the **Extended User Data (EUD)** area are resolved. Every
  emitted shader is strictly `spirv-val`-gated.
- ✅ **Texture decode:** GFX10 `SW_4KB_S` / `SW_64KB_S` de-swizzle for all element sizes (1/2/4/8/16 B,
  derived from the authoritative addrlib table) plus BC1–BC7 and BC6H block decompression, honoring
  the T# format, `DST_SEL` channel swizzle, and the paired S# sampler (filter / wrap / anisotropy /
  LOD / border).
- ✅ **Frame spine → live renderer:** decoded register-state → `render_state` / `vk_translate` →
  `resolve_pipeline_state` → real `VkGraphicsPipeline`s, with topology, blend (incl. separate-alpha),
  depth, stencil, per-MRT color-write-mask, the game's real fast-clear color, and a render-to-texture
  cache for multi-pass composites — all driven from the decoded registers and pixel-verified.

**Frontend:** `prosper-app` is a windowed player (SDL3 window + Vulkan present + audio sink +
evdev/SDL3 controllers + real message/error/IME dialogs), sharing the same boot + render core as the
headless `boot_trace`.

Development is **agentic-first**: correctness is verified programmatically — **92 self-checking tests**
under `ctest` (including a headless Vulkan/llvmpipe harness that runs recompiled shaders and asserts
numeric/pixel results, and per-opcode round-trip disassembly checks), a **golden-image snapshot guard**
that boots a real title and pixel/content-asserts an exact frame, cross-platform CI (Linux +
Windows/MinGW), structured logs, and purpose-built tracing tooling — never by hand.

## Legal / ethical

- **No game files, keys, or copyrighted Sony code are included** in this repository, and none ever
  will be. You must supply your own legally-obtained dump.
- prosper only works with **unencrypted** module segments; it contains **no** console decryption
  keys and performs no circumvention of Sony's cryptography.
- Sony's library interfaces are reimplemented from published symbol/NID information, clean-room style,
  the way Wine reimplements Win32.
- This is an independent interoperability / preservation research project, not affiliated with or
  endorsed by Sony Interactive Entertainment.

## Building (Linux)

Requires a C++20 compiler, CMake, and Ninja. A Vulkan loader is needed for the graphics tests
(the CI/headless path uses the `llvmpipe` software ICD).

```sh
cd prosper
cmake -G Ninja -B build-linux
cmake --build build-linux
ctest --test-dir build-linux          # 92 self-checking tests
```

Add `-DPROSPER_APP=ON` to also build the windowed `prosper-app` frontend (fetches SDL3). Native
Windows/MinGW now boots and renders the primary title through the same live Vulkan renderer; its
current build, run, screenshot, and diagnostic recipe is maintained in
[`prosper/docs/WINDOWS_PORT_HANDOFF.md`](prosper/docs/WINDOWS_PORT_HANDOFF.md).

On Windows, the full graphics + WASAPI audio + controller/keyboard frontend is one command:

```powershell
.\prosper\scripts\run-windows.ps1 .\PPSA24651-app0
```

## Repository layout

```
prosper/
  src/self/       SELF/ELF parsing → relocatable module image
  src/loader/     multi-module linker + global export table
  src/hle/        HLE of Sony libraries (libc, libkernel, AGC/graphics, services), NID hashing
  src/host/       host execution: per-platform image mapping, ABI stubs, fault handling
  src/gpu/        AGC→Vulkan: PM4 decode, command processor, render state, vk_translate,
                  texture tiling + BC decode, and the RDNA2→SPIR-V shader recompiler
  frontends/      shared boot+render core, the windowed prosper-app, SDL3 audio/dialog, controllers
  tools/          boot_trace, self_dump, shader_histo, snapshot (golden-image guard), spv_validate, …
  tests/          unit + boot + Vulkan-execution tests (run under ctest)
  docs/           ARCHITECTURE, ROADMAP, GRAPHICS, RENDER_LOOP, VERIFICATION, and per-frontier logs
```

## License

See [LICENSE](LICENSE) if present. Game content and Sony SDK symbols are the property of their
respective owners and are not distributed here.
