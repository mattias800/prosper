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

## Screenshots

Real retail PS5 titles running under prosper on Linux — direct, unmodified captures from the frontend
(no post-processing), retained at each title's native output resolution. Each reaches the milestone noted
in [Game compatibility](COMPATIBILITY.md).

<p align="center">
  <img src="assets/screenshots/messenger.png" alt="The Messenger — first level gameplay"><br>
  <em><strong>The Messenger</strong> — first level, native 1920×1080</em>
</p>
<p align="center">
  <img src="assets/screenshots/dead-cells.png" alt="Dead Cells — Prisoners' Quarters gameplay"><br>
  <em><strong>Dead Cells</strong> — controllable Prisoners' Quarters</em>
</p>
<p align="center">
  <img src="assets/screenshots/blasphemous2.png" alt="Blasphemous 2 — first playable room"><br>
  <em><strong>Blasphemous 2</strong> — first playable room</em>
</p>
<p align="center">
  <img src="assets/screenshots/evergate.png" alt="Evergate — first tutorial room gameplay"><br>
  <em><strong>Evergate</strong> — first tutorial room</em>
</p>
<p align="center">
  <img src="assets/screenshots/gris.png" alt="GRIS — opening gameplay"><br>
  <em><strong>GRIS</strong> — opening gameplay, native 1920×1080</em>
</p>
<p align="center">
  <img src="assets/screenshots/space-adventure-cobra.png" alt="Space Adventure Cobra — tutorial combat"><br>
  <em><strong>Space Adventure Cobra — The Awakening</strong> — tutorial combat, native 1920×1080</em>
</p>
<p align="center">
  <img src="assets/screenshots/blue-prince-title.png" alt="Blue Prince — title screen"><br>
  <em><strong>Blue Prince</strong> — title screen</em>
</p>
<p align="center">
  <img src="assets/screenshots/terminator.png" alt="Terminator 2D: NO FATE — attract-mode gameplay"><br>
  <em><strong>Terminator 2D: NO FATE</strong> — attract-mode gameplay</em>
</p>
<p align="center">
  <img src="assets/screenshots/dragon-quest-vii-title.png" alt="Dragon Quest VII Reimagined — title screen"><br>
  <em><strong>Dragon Quest VII Reimagined</strong> — animated title screen, native 3840×2160</em>
</p>

> These illustrative screenshots are captures of prosper's own rendered output; the games' artwork remains
> the property of its respective owners. No game files, assets, keys, or data are redistributed in this
> repository — you must supply your own legally-obtained dump.

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
- ✅ Boots several engine families into their running frame loop and real GPU submission — **Unity /
  IL2CPP** (loads the C# metadata, spins up the full IL2CPP GC + worker thread pool, drives Unity's
  `GfxDevice` bring-up), **Unreal Engine**, and Rockstar's **RAGE** — with several titles reaching
  interactive menus or gameplay.
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

Development is **agentic-first**: correctness is verified programmatically — a broad suite of
**self-checking tests** under `ctest` (including a headless Vulkan/llvmpipe harness that runs recompiled shaders and asserts
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

## Building

### Linux

Requires a C++20 compiler, CMake, and Ninja. A Vulkan loader is needed for the graphics tests
(the CI/headless path uses the `llvmpipe` software ICD). The native AvPlayer backend uses FFmpeg
for MP4 demux/audio conversion and VA-API for hardware video decode. On Ubuntu/Debian:

```sh
sudo apt install ninja-build pkg-config libvulkan-dev libavformat-dev libavcodec-dev \
  libavutil-dev libswresample-dev libswscale-dev libva-dev
```

```sh
cd prosper
cmake -G Ninja -B build-linux
cmake --build build-linux
ctest --test-dir build-linux          # unit + boot + Vulkan-execution tests
```

Add `-DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON` to build the windowed frontend
with audio and controller support. CMake fetches SDL3 when it is not installed. Video source-open
requires a working VA-API render device by default; select a non-default node with
`PROSPER_AVP_VAAPI_DEVICE=/dev/dri/renderD129`. `PROSPER_AVP_ALLOW_SOFTWARE=1` is an explicit
diagnostic fallback, not normal playback behavior.

### Windows core

The supported Windows toolchain is 64-bit MinGW-w64 UCRT. In an MSYS2 UCRT64 shell:

```sh
pacman -S --needed git mingw-w64-ucrt-x86_64-gcc \
  mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
cmake -S prosper -B prosper/build-windows-core -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE
cmake --build prosper/build-windows-core
ctest --test-dir prosper/build-windows-core --output-on-failure
```

This builds and tests the headless core without SDL or Vulkan. GitHub Actions runs the same UCRT64
path on every push and pull request.

### Windows app

Install the Vulkan development packages in the same UCRT64 shell, then enable the app and its SDL3
backends:

```sh
pacman -S --needed mingw-w64-ucrt-x86_64-vulkan-headers \
  mingw-w64-ucrt-x86_64-vulkan-loader
cmake -S prosper -B prosper/build-windows-app -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPROSPER_APP=ON \
  -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON
cmake --build prosper/build-windows-app --target prosper-app
./prosper/build-windows-app/prosper-app.exe --test-pattern --frames 120
```

From native PowerShell, the repository launcher supports WinLibs or MSYS2 MinGW plus an installed
Vulkan SDK and performs configure, build, and launch in one command:

```powershell
.\prosper\scripts\run-windows.ps1 .\PPSA24651-app0
```

The detailed native build, screenshot, and diagnostic recipe is in
[`WINDOWS_PORT_HANDOFF.md`](prosper/docs/WINDOWS_PORT_HANDOFF.md).

## Windows download and use

Every CI run publishes a `prosper-windows-x64` artifact. A `v*` tag publishes the same
`prosper-windows-x64.zip` on the repository's [GitHub Releases page](https://github.com/mattias800/ps5ys/releases).
The archive contains
`prosper-app.exe`, a one-command PowerShell launcher, and its usage guide; it never contains games,
firmware, or keys.

After extracting the archive, launch an unpacked `app0` directory:

```powershell
./start-prosper.ps1 'D:/PS5/PPSA24651-app0'
```

There is no game-picker or settings UI yet. The launcher supplies the required guest environment,
creates local save data, and enables the Vulkan window, audio, physical controller, and keyboard
overlay. Use `./start-prosper.ps1 -TestPattern -Frames 120` to test the frontend without a game.
See [`WINDOWS_RELEASE.md`](prosper/docs/WINDOWS_RELEASE.md) for requirements, direct-executable use,
save-data selection, the complete keyboard mapping, recordings, and troubleshooting.

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

The July 2026 renderer profiling results, correctness constraints, rejected experiments, and next
architecture step are recorded in
[`RENDERER_PERFORMANCE_2026_07.md`](prosper/docs/RENDERER_PERFORMANCE_2026_07.md).

## License

See [LICENSE](LICENSE) if present. Game files, assets, keys, and Sony SDK symbols are the property of
their respective owners and are not redistributed here; the documentation screenshots are captures of
prosper's own rendered output, shown for illustration.
