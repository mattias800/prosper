# prosper

A PS5 (Prospero) → **Windows/Linux** user-space compatibility layer — *Proton for PS5*.

Not a CPU emulator: the PS5 is x86-64, so guest code runs **natively**. `prosper`
translates the operating system (FreeBSD-derived), the library ABI (Sony NID-linked
modules), and the GPU (AGC → Vulkan) underneath the unmodified game binary.

**Target title:** `PPSA24651` — *The Messenger* (Unity 2022 / IL2CPP), whose dump
lives in `../PPSA24651-app0`. Its SELF segments are unencrypted, which is what
makes this project possible without console keys.

## Status
- ✅ **M0 — Recon & tooling.** Format cracked, full HLE work-list extracted.
- ✅ **M1 — Loader.** SELF/ELF → relocatable image → multi-module link → NID import binding.
- ✅ **M2/M3 — HLE + boot.** libkernel/libc (memory, threads, futex, time, file I/O, locale);
  IL2CPP GC + thread pool; boots **through** IL2CPP into Unity's GfxDevice bring-up.
- 🚧 **M4/M5 — Graphics (AGC → Vulkan), active.** AGC command frontend complete (CreateShader +
  SubmitDcb, zero unimplemented `libSceAgc` calls); PM4 decode → command processor → render state →
  `resolve_pipeline_state` → a real `VkGraphicsPipeline` (topology/blend/depth/write-mask
  pixel-verified). **RDNA2→SPIR-V recompiler**: ~52 ALU ops + **divergent control flow** (EXEC
  predication, saveexec/restore, forward branches), **67% instruction coverage** over the 41 real
  game shaders (`shader_histo`); the wall is `SMEM` (needs the resource-binding model). Boot root
  cause found: the GPU-resource residency pass is **completion-event-driven** and our headless equeue
  never fires it — fix in progress is delivering a real completion on submit/flip. See
  [`docs/GRAPHICS.md`](docs/GRAPHICS.md).

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for milestones, [`docs/GRAPHICS.md`](docs/GRAPHICS.md) for
the graphics frontier, and [`docs/VERIFICATION.md`](docs/VERIFICATION.md) for the agentic-first
(programmatic, no-manual-eyeballing) verification strategy.

## Layout
```
prosper/
  docs/            architecture, roadmap, findings, graphics, verification
  src/self/        SELF/ELF parsing → relocatable module image
  src/loader/      multi-module linker + global export table
  src/hle/         HLE of Sony libraries (libc, libkernel, AGC/graphics), NID hashing
  src/host/        host execution: image mapping, stubs, fault handling (Linux)
  src/gpu/         AGC→Vulkan: PM4 decode, command processor, render state,
                   vk_translate, resource layer (gpu_resources), RDNA2→SPIR-V recompiler
  tools/           self_dump, boot_trace, shader_histo, imgdump
  tests/           unit + boot + Vulkan-execution tests (ctest)
  CMakeLists.txt
```

## Build
```
cmake -S . -B build -G Ninja
cmake --build build
```
Or the tool directly:
```
g++ -O2 -std=c++20 tools/self_dump/self_dump.cpp -o self_dump
./self_dump ../PPSA24651-app0/eboot.bin [--symbols]
```

## Legal / scope
Interoperability & preservation research on a legally-owned title. `prosper` ships
**no** Sony code, firmware, or keys — it reimplements published library interfaces.
It only operates on **already-unencrypted** dumps; it does not defeat DRM/encryption.
