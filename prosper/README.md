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
  leaves, including temporal read-before-write surfaces, without invoking Vulkan (#595). Capture v7 retains
  bounded, content-addressed raw RDNA2 and exact opcode/PC/state diagnostics for unrealized draws/dispatches;
  `gpu_replay` can inspect and extract the failed stage without rerunning the title (#618).
- ✅ **Dead Cells reaches gameplay reproducibly:** a deterministic input route passes the splash and menus
  into the first playable scene. The persistent-depth rejection is fixed: timeline-v5 backing hashes and
  compute writer provenance identify Unity's 32 KiB HTILE fill as the per-frame fast clear, and overlapping
  guest GPU writes now invalidate the detached Vulkan depth cache (#611). A routed live A/B restores the
  foreground/platform/HUD layers that remained black without invalidation.
  Four repeated fragment failures were uniform VCCZ-exit light-accumulation loops. The stage-specific
  structurizer now accepts that shape only when each compare operand has a proved wave-uniform reaching
  definition; varying bounds and the compute wave-mask form still reject. Current routed gameplay submits
  realize every semantic draw (#615).
  Version-7 GPU captures seed temporal render targets, retain complete depth-surface programming, and keep
  content-addressed resource versions for
  faithful offline isolation (#568/#594); one residual
  live/replay hash mismatch remains (#569). Dispatch thread counts and derived workgroup dimensions,
  compute program binding, direct type-1 buffers, and mixed graphics/compute PM4 order now execute correctly
  (#580/#576/#584).
- 🚧 **Active frontiers:** bundle v2 makes long Dead Cells history practical with exact shared-resource
  chunks, rolling semantic endpoints, final compaction, and suffix replay (#606). A fixed 1,200-submit full-state
  run reduced 122.97 GiB logical data to 301.1 MiB. A compact exact two-submit closure resolves both temporal
  642×362 edges without bounded leaves, but the first 80-draw semantic endpoint is the opening vignette rather
  than playable gameplay. #608 selected the controllable Jump tutorial by combining an exact 90-draw submit
  with the 738x420 pass at semantic draw 79..81, and tracked its exact history and first bad composition. That
  exact conjunction is now a preserved-capture recipe: current fresh routes reach sustained gameplay but no
  longer match it, so selector recalibration remains under #594.
  The faithful 883-submit closure resolves 1,764 temporal image dependencies and established the stale-depth
  failure. The draw stream has no `DB_RENDER_CONTROL` clear because hardware observes the compute-written
  HTILE metadata instead; #611 implements that missing cache boundary. #569 still tracks exact depth/stencil
  checkpoint serialization. The nondeterministic exact-frame Dead Cells
  snapshot guard is tracked separately in #596. UE4's measured GPU boundary remains under its area issues.

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
