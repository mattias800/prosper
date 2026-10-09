<p align="center">
  <img src="assets/prosper-logo.png" alt="prosper" width="200">
</p>

# prosper

[![CI](https://img.shields.io/github/actions/workflow/status/mattias800/prosper/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/mattias800/prosper/actions/workflows/ci.yml)
[![Platform: Linux](https://img.shields.io/badge/platform-Linux-informational?style=flat-square&logo=linux)](https://github.com/mattias800/prosper)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows%2010%2F11-informational?style=flat-square&logo=windows)](https://github.com/mattias800/prosper)
[![Language: C++20](https://img.shields.io/badge/language-C%2B%2B20-informational?style=flat-square&logo=cplusplus)](https://github.com/mattias800/prosper)
[![Vulkan 1.4](https://img.shields.io/badge/Vulkan-1.4-red?style=flat-square&logo=vulkan)](https://www.vulkan.org/)

Native execution of PS5 game dumps on Linux and Windows PCs.

prosper is a user-space PS5 (Prospero) → PC compatibility layer — "Wine/Proton for PS5." The PS5
CPU is x86-64, so guest code runs natively and there is no CPU emulation. The work is loading
Sony's SELF/ELF modules and linking them into one address space, reimplementing the PS5 system
libraries on top of host facilities, translating the ABI where the guest's conventions differ, and
translating AGC GPU command streams to Vulkan, which includes an RDNA2 → SPIR-V shader recompiler.

It exists to test a method: whether a repository can be structured — charter, tooling, tests,
evidence rules — so that AI agents do reverse-engineering work of this kind over months, with a
human who sets direction and reviews results but does not write code. Preservation work has this
shape, and there is more of it than there are people doing it.

This is not a product. There is no support, no release schedule, and no claim that any title works
beyond the specific route documented for it.

---

## How it works

- **No CPU emulation:** the PS5 CPU is standard x86-64 (Zen 2). Game code runs natively at host
  speed.
- **Loader + linker (`prosper/src/self/`, `prosper/src/loader/`):** parses SELF/ELF modules into
  relocatable images and links Sony NID imports into one address space with a global export table.
- **Reimplemented system libraries (`prosper/src/hle/`):** clean-room implementations of the
  published libc, libkernel, graphics and service interfaces, plus host-side image mapping, ABI
  stubs and fault handling (`prosper/src/host/`).
- **GPU translation (`prosper/src/gpu/`):** decodes AGC/PM4 command buffers and recompiles RDNA2
  shaders to SPIR-V for a Vulkan 1.4 backend.

---

## Status — what runs

Progress is defined per title by a six-rung ladder: 1. any real graphics → 2. title screen →
3. gameplay with the scene rendering → 4. a human confirms it looks right → 5. comparison against
PS5 hardware reference → 6. a reviewed automatic snapshot guard. A title is done only at rung 6.
A game loop running over a black screen is not gameplay, and a build that compiles is not progress.

- [`COMPATIBILITY.md`](COMPATIBILITY.md) — per-title user-visible milestones (authoritative
  user-facing overview, written by hand).
- [`PROGRESS_TRACKER.md`](PROGRESS_TRACKER.md) — machine-readable rung table, generated from the
  tracker issues and kept in step by CI.
- [`BLOG.md`](BLOG.md) — progress blog, newest first: every checked-in screenshot and the story
  around the ones worth a story.
- [`prosper/docs/ROADMAP.md`](prosper/docs/ROADMAP.md) — what is planned. Per-title rungs, routes
  and blockers live in `COMPATIBILITY.md` and the `tracker:game` issues, not here.

When this page and a tracker disagree, the tracker wins.

---

## How the work is organised

- [`CLAUDE.md`](CLAUDE.md) is the project charter: scope, build and run recipes, evidence rules,
  and the recorded mistakes that have cost the most time. It is read before starting work and
  assumes no memory of previous sessions.
- Verification is programmatic
  ([`docs/process/VERIFICATION.md`](prosper/docs/process/VERIFICATION.md)): Vulkan tests that
  assert numeric and pixel results, a `spirv-val` gate on every SPIR-V emitter, and golden-image
  guards over real boots.
- Falsified hypotheses are recorded in a `## Ruled out` section in the relevant status document.
  Measurements that turned out to come from the apparatus rather than the subject are recorded as
  numbered instrument traps.
- Questions that existing tools cannot answer are answered by writing a tool, under
  [`tools/`](prosper/tools/AGENTS.md).
- GitHub issues hold work that spans sessions. Concurrent agents claim an issue before starting and
  work in separate git worktrees. Non-obvious changes get an independent review before merge.

Most of the costly failures so far have not been in generated code, but in bookkeeping and
inference. A retracted figure left standing in the charter kept being used for a day after it was
withdrawn. A review finding with a plausible `file:line` citation was repeated across many sessions
without anyone opening the file. A positive control passed on a case it could not have detected.
Each of those is now a rule in `CLAUDE.md`.

---

## Legal boundary & scope

prosper operates strictly within legal and ethical boundaries:

- **No proprietary code:** no game files, keys, firmware or Sony code are in this repository. You
  supply your own legally obtained dump.
- **No decryption:** prosper holds no console keys, performs no decryption, and operates only on
  module segments that are already unencrypted.
- **No circumvention by proxy:** it refuses to link the third-party replacements for Sony libraries
  that some dumps carry, since loading one would perform the circumvention by proxy. Entitlement
  and add-content queries are answered from the content present on disk rather than approved
  unconditionally.
- **Clean-room reimplementation:** Sony's library interfaces are reimplemented from published symbol
  and NID data, the way Wine reimplements the Win32 API.
- **Clean submissions:** issues, pull requests and CI never contain game assets, shader bytecode
  dumps, save files or recorded game footage.

The project is independent and not affiliated with Sony Interactive Entertainment.

---

## Documentation & architecture

| Document | What it is |
|---|---|
| [`BUILDING.md`](BUILDING.md) | Dependencies, build and test commands, launching a dump |
| [`prosper/README.md`](prosper/README.md) | The source tree and its architecture documents |
| [`CLAUDE.md`](CLAUDE.md) | The project charter — read before changing anything |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | What makes an external contribution reviewable |
| [`COMPATIBILITY.md`](COMPATIBILITY.md) | Per-title results |
| [`prosper/docs/architecture/ARCHITECTURE.md`](prosper/docs/architecture/ARCHITECTURE.md) | How the layers fit together |
| [`prosper/docs/gpu/GRAPHICS.md`](prosper/docs/gpu/GRAPHICS.md) | The AGC → Vulkan pipeline |
| [`prosper/docs/process/VERIFICATION.md`](prosper/docs/process/VERIFICATION.md) | The verification strategy every change is held to |
| [`prosper/docs/platforms/WINDOWS_PORT_HANDOFF.md`](prosper/docs/platforms/WINDOWS_PORT_HANDOFF.md) | Windows port state and full native recipe |
| [`prosper/docs/platforms/LINUX_RELEASE.md`](prosper/docs/platforms/LINUX_RELEASE.md) | Linux desktop release notes for users |
| [`prosper/docs/platforms/WINDOWS_RELEASE.md`](prosper/docs/platforms/WINDOWS_RELEASE.md) | Windows release notes for users |

---

## Toolchain & requirements

- **OS:** Linux (primary) and Windows 10/11 (64-bit); macOS x86_64 under Rosetta 2 exercises the
  Darwin substrate only.
- **Compiler:** C++20-capable compiler, CMake and Ninja. Windows uses 64-bit MinGW-w64 UCRT.
- **Graphics:** Vulkan 1.4 headers, loader and device for the live renderer and the app; the
  headless test path uses the `llvmpipe` software ICD, and `spirv-tools` provides the `spirv-val`
  gate.
- **Frontend:** SDL3 for the windowed app with audio and controllers (fetched by CMake when not
  installed).

Full commands per platform — see [`BUILDING.md`](BUILDING.md):

```sh
cmake -S prosper -B prosper/build-linux -G Ninja
cmake --build prosper/build-linux
ctest --test-dir prosper/build-linux --no-tests=error
```

A `v*` tag publishes desktop archives (`prosper-linux-x86_64.AppImage`,
`prosper-linux-x86_64.tar.gz`, `prosper-windows-x64.zip`), each with a launcher that supplies the
guest environment. Expect the milestone recorded in [`COMPATIBILITY.md`](COMPATIBILITY.md) for a
given title and nothing beyond it.

---

## Project structure

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
```

`CMakeLists.txt` globs `src/*.cpp` recursively, so a new subfolder under `src/` needs no
source-list edit. Every new file belongs under `prosper/`. Folders holding real content carry an
`AGENTS.md` describing what belongs in them — a map, not documentation of the code.

---

## Contributing

External contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). The one idea behind
all of the rules: **a contribution is worth what a reviewer can check.** Read `CLAUDE.md` before
anything beyond a small fix, put every new file under `prosper/`, wire new code to a real call
site, and give the command that reproduces every claim.

## License

No `LICENSE` file exists yet, so no license is granted by default. Open an issue to ask.
