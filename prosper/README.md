# prosper

A PS5 (Prospero) → **Linux/Windows** user-space compatibility layer — *Proton for PS5*.

Not a CPU emulator: the PS5 is x86-64, so guest code runs **natively**. `prosper` reimplements the
operating system (FreeBSD-derived), the library ABI (Sony NID-linked modules), and the GPU
(AGC → Vulkan) underneath the unmodified game binary.

**Primary title:** `PPSA24651` — *The Messenger* (Unity 2022 / IL2CPP). Also exercised:
`PPSA02664` (Unity/IL2CPP), `PPSA17942` (Unreal Engine), `PPSA13579` (*Blasphemous 2*), and
`PPSA15552` (*Dead Cells*). Their SELF segments are unencrypted, which is what makes the project
possible without console keys. Dumps are user-supplied and gitignored.

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
- ✅ **The Messenger renders through gameplay:** intro, title, menus, save list, and the complete first
  level render at native 1920×1080 with working scripted gamepad progression. The LUT producer,
  foreground culling, and cross-call depth/stencil lifecycle fixes are regression-covered by Vulkan
  tests and the real-game snapshot guard.
- ✅ **Graphics investigation tooling:** versioned live GPU capture/replay (`.prgcap`), per-draw/resource
  inspection and isolation, strict reflected SPIR-V/runtime descriptor validation, render-target producer
  provenance, normal screenshot capture, and a local content-metric snapshot guard. Native-speed `.prgtl`
  submit/present indexes can select one exact submit before renderer sampling and materialize immutable,
  content-deduplicated graphics/compute state plus mixed operation order for offline replay (#594).
  Offline dependency graphs resolve in-submit resource versions and identify deduplicated prior-submit
  leaves, including temporal read-before-write surfaces, without invoking Vulkan (#595).
- ✅ **Dead Cells reaches gameplay reproducibly:** a deterministic input route passes the splash and menus
  into the first playable scene. HUD and some composition render, but the world is mostly white (#566).
  Version-5 GPU captures seed temporal render targets and retain content-addressed resource versions for
  faithful offline isolation (#568/#594); one residual
  live/replay hash mismatch remains (#569). Dispatch thread counts and derived workgroup dimensions,
  compute program binding, direct type-1 buffers, and mixed graphics/compute PM4 order now execute correctly
  (#580/#576/#584).
- 🚧 **Active frontiers:** retain and replay producer-time state/bytes from Dead Cells submit 18749 for the
  two temporal 642×362 versions consumed by submit 18750 (#595/#586). Exact-submit capture, selected-submit
  graphing, and same-run producer identity are complete; recursive capsule closure is not. The stale exact Dead Cells
  snapshot baseline is tracked separately in #596. UE4's measured GPU boundary remains under its area issues.

The completed Messenger black-render investigation and reusable evidence boundary are recorded in
[`docs/MESSENGER_BLACK_RENDER.md`](docs/MESSENGER_BLACK_RENDER.md). Current work is tracked in GitHub
issues; title failures should produce a reproducible route/capture and a narrowly scoped issue rather
than a moving-revision hypothesis log.

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
  tools/           self_dump, boot_trace, shader_histo, screenshot, snapshot, gpu_timeline, gpu_replay,
                   spv_validate
  tests/           unit + boot + Vulkan-execution tests (ctest)
  CMakeLists.txt
```

## Build
```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build          # 90 self-checking tests
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
