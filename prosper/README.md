# prosper — source tree

This directory is the project itself: the compatibility layer, its tools, its tests and its
documentation. Start from the [repository README](../README.md) for what prosper is and why it
exists, [`../BUILDING.md`](../BUILDING.md) to build it, and [`../CLAUDE.md`](../CLAUDE.md) — the
project charter — before changing anything here.

A PS5 (Prospero) → **Linux/Windows** user-space compatibility layer, shaped like Proton rather than
like an emulator. Not a CPU emulator: the PS5 is x86-64, so guest code runs **natively**. prosper
reimplements the operating system (FreeBSD-derived), the library ABI (Sony NID-linked modules) and
the GPU (AGC → Vulkan) underneath an unmodified game binary. It operates only on dumps whose SELF
segments are already unencrypted, which is what makes the project possible without console keys.
Dumps are user-supplied and gitignored.

## Orientation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the layers fit together.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — what is planned.
- [`docs/GRAPHICS.md`](docs/GRAPHICS.md), [`docs/RESOURCE_BINDING.md`](docs/RESOURCE_BINDING.md),
  [`docs/RECOMPILER_REMAINING.md`](docs/RECOMPILER_REMAINING.md) — the AGC → Vulkan pipeline, its
  descriptor model, and the shader recompiler's remaining gaps.
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — the programmatic, no-manual-eyeballing
  verification strategy every change is held to.
- [`docs/PORTING.md`](docs/PORTING.md) — per-platform substrate notes (Linux, Windows, macOS/Rosetta).
- [`tools/AGENTS.md`](tools/AGENTS.md) — what each tool is for and which question it answers.
- [`docs/DEBUGGING_WORKFLOWS.md`](docs/DEBUGGING_WORKFLOWS.md) — question-to-tool recipes,
  capability controls, host sanitizer checks and GPU capture analysis. Start here for development.
- [`docs/VULKAN_RUNTIME.md`](docs/VULKAN_RUNTIME.md) — the Vulkan 1.4 runtime floor the live
  graphics and compute backends require, and the capability policy behind it.
- [`docs/GAME_COMPAT_ORCHESTRATION.md`](docs/GAME_COMPAT_ORCHESTRATION.md) — concurrent title work,
  and the numbered list of instrument traps: measurements that turned out to come from the
  apparatus rather than the subject.
- `docs/<TITLE>_STATUS.md` — the investigation record for one title. Read its `## Ruled out`
  section before forming a hypothesis; it lists the hypotheses already falsified and the evidence
  that settled them. Closed investigations keep theirs too, including
  [`docs/MESSENGER_BLACK_RENDER.md`](docs/MESSENGER_BLACK_RENDER.md). Current per-title state lives
  in the `tracker:game` issues, not in this tree.

Folders that hold real content carry an `AGENTS.md` describing what belongs in them. It is a map,
not documentation of the code.

## Layout

```
prosper/
  docs/            architecture, roadmap, graphics, verification, per-title and per-frontier logs
  src/self/        SELF/ELF parsing → relocatable module image
  src/loader/      multi-module linker + global export table
  src/hle/         HLE of Sony libraries (libc, libkernel, AGC/graphics, services), NID hashing
  src/host/        host execution: per-platform image mapping, ABI stubs, fault handling
  src/gpu/         AGC→Vulkan: PM4 decode, command processor, render state, vk_translate,
                   texture tiling + BC decode, RDNA2→SPIR-V recompiler
  frontends/       shared boot+render core, windowed prosper-app, SDL3 audio/dialog, controllers
  tools/           self_dump, boot_trace, screenshot, snapshot, gpu_timeline, gpu_replay,
                   spv_validate, shader_histo, re/, il2cpp/, refactor/, …
  tests/           unit + boot + Vulkan-execution tests (ctest)
  scripts/         per-title launch and input routes
  CMakeLists.txt
```

`CMakeLists.txt` globs `src/*.cpp` recursively, so a new subfolder under `src/` needs no
source-list edit. Every new file belongs under `prosper/` for that reason — CI enforces it, because
a file added at the repository root compiles nowhere while every job still reports success.

## Build

Commands, dependencies and the testing gotchas are in [`../BUILDING.md`](../BUILDING.md). The live
renderer, compute backend and windowed frontend need Vulkan 1.4 headers, loader and device. The short
version, from the repository root:

```
cmake -S prosper -B prosper/build-linux -G Ninja
cmake --build prosper/build-linux
ctest --test-dir prosper/build-linux --no-tests=error
```

## Legal / scope

Interoperability and preservation research on legally-owned titles. prosper ships **no** Sony code,
firmware or keys — it reimplements published library interfaces clean-room style. It operates only
on **already-unencrypted** dumps, performs no decryption, and refuses to link the third-party
Sony-library replacements that some dumps carry. See the repository README for the full statement.
