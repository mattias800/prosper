# prosper

A PS5 (Prospero) → **Linux/Windows** user-space compatibility layer — *Proton for PS5*.

Not a CPU emulator: the PS5 is x86-64, so guest code runs **natively**. `prosper` reimplements the
operating system (FreeBSD-derived), the library ABI (Sony NID-linked modules), and the GPU
(AGC → Vulkan) underneath the unmodified game binary.

**Primary title:** `PPSA24651` — *The Messenger* (Unity 2022 / IL2CPP). Also exercised: `PPSA02664`
(Unity/IL2CPP) and `PPSA17942` (Unreal Engine). Their SELF segments are unencrypted, which is what
makes the project possible without console keys. Dumps are user-supplied and gitignored.

## Status
- ✅ **M0–M1 — Recon, tooling & loader.** Format cracked; SELF/ELF → relocatable image → multi-module
  link → NID import binding.
- ✅ **M2–M3 — HLE + boot.** libkernel/libc (virtual/direct/flexible memory + guard pages, threads,
  futex, AIO + file I/O, time, locale); IL2CPP GC + thread pool; system services the games gate on
  (user/pad, SaveData, AvPlayer/AJM, common dialogs + IME, NP/online, system language). Boots
  **through** IL2CPP and Unity's GfxDevice into the running frame loop.
- ✅ **M4 — Graphics (AGC → Vulkan).** AGC frontend (CreateShader + submit; PM4 decode → command
  processor → register-state fold; EOP/DMA/fence writes on the guest timeline). **RDNA2→SPIR-V
  recompiler:** full ALU + `SCC`; divergent control flow (EXEC predication, saveexec/restore, `execz`
  if-then + loop exits); `SMEM`, `MUBUF` vertex-fetch/load/store, `MIMG` sample/load, `LDS`+barriers,
  `EXP`/`VINTRP`; EUD-resident descriptor resolution; every shader `spirv-val`-gated. Texture decode:
  GFX10 `SW_4KB_S`/`SW_64KB_S` de-swizzle for all element sizes + BC1–7/BC6H, T# format + `DST_SEL`
  swizzle + paired S# sampler. Frame spine → `resolve_pipeline_state` → real `VkGraphicsPipeline`s
  with blend (incl. separate-alpha)/depth/stencil/write-mask/fast-clear + a render-to-texture cache.
- ✅ **The Messenger renders:** intro cutscene (dozens of distinct animating frames), title screen with
  readable text, working gamepad input, and progression into gameplay level loading — all asserted by
  a golden-image snapshot guard.
- 🚧 **Active frontiers:** revision-locked capture/replay of the black Messenger gameplay frame,
  strict shader/resource-interface validation, independent diagnosis of the missing save list, broader
  shader coverage, and the Unreal title's title-screen composite.

The current Messenger frontier is revision-locked capture/replay of the black gameplay frame, strict
shader/resource-interface validation, and an independent trace of the save-list failure. See
[`docs/MESSENGER_BLACK_RENDER.md`](docs/MESSENGER_BLACK_RENDER.md).

See [`docs/ROADMAP.md`](docs/ROADMAP.md), [`docs/GRAPHICS.md`](docs/GRAPHICS.md),
[`docs/RENDER_LOOP.md`](docs/RENDER_LOOP.md), and [`docs/VERIFICATION.md`](docs/VERIFICATION.md)
(the agentic-first, programmatic, no-manual-eyeballing verification strategy).

## Layout
```
prosper/
  docs/            architecture, roadmap, graphics, verification, per-frontier logs
  src/self/        SELF/ELF parsing → relocatable module image
  src/loader/      multi-module linker + global export table
  src/hle/         HLE of Sony libraries (libc, libkernel, AGC/graphics, services), NID hashing
  src/host/        host execution: image mapping, stubs, fault handling (Linux)
  src/gpu/         AGC→Vulkan: PM4 decode, command processor, render state, vk_translate,
                   texture tiling + BC decode, RDNA2→SPIR-V recompiler
  frontends/       shared boot+render core, windowed prosper-app, SDL3 audio/dialog, controllers
  tools/           self_dump, boot_trace, shader_histo, snapshot (golden-image guard), spv_validate
  tests/           unit + boot + Vulkan-execution tests (ctest)
  CMakeLists.txt
```

## Build
```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build          # 84 self-checking tests
```
Add `-DPROSPER_APP=ON` for the windowed `prosper-app` frontend (fetches SDL3). Or a tool directly:
```
g++ -O2 -std=c++20 tools/self_dump/self_dump.cpp -o self_dump
./self_dump ../PPSA24651-app0/eboot.bin [--symbols]
```

## Legal / scope
Interoperability & preservation research on legally-owned titles. `prosper` ships **no** Sony code,
firmware, or keys — it reimplements published library interfaces clean-room style. It only operates on
**already-unencrypted** dumps; it does not defeat DRM/encryption.
