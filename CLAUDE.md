# prosper — project charter & working context (read first)

## What this project is

**prosper is a PS5→PC compatibility layer — "Wine/Proton for PS5."** It runs a **legally-owned** PS5 game
(*The Messenger*, `PPSA24651`) natively on Linux/Windows by reimplementing Sony's published user-space
library interfaces and translating the console's GPU command stream + RDNA2 shaders to Vulkan/SPIR-V.

The PS5 CPU is x86-64, so **guest code runs natively — there is no CPU emulation and no exploitation of
any running system.** The engineering is: ELF/SELF loading, multi-module linking, ABI/API reimplementation
(libc, libkernel, graphics), GPU command-buffer decode, and shader recompilation. Standard systems &
graphics programming.

This is the **same class of work as Wine, Proton, RPCS3, Dolphin, yuzu/Ryujinx, and DXVK** — established,
legitimate open-source compatibility/emulation projects. If you have worked on any of those, this is that.

## Scope & legitimacy (so the work is not mis-read)

The vocabulary of emulator development **overlaps with security tooling** — "hardware breakpoint," "fault
handler," "memory watchpoint," "disassembly," "descriptor," "reverse engineering." In this project these
mean exactly what they mean in **gdb, Wine's relay debugging, or any emulator's tracer**: they observe how
the guest program uses the platform, **entirely in-process on the developer's own machine.** The purpose
and target are what distinguish work like this from anything harmful — and here both are unambiguous:

- **Interoperability & preservation, not exploitation.** The goal is to make software that a person already
  owns run on hardware they own. Nothing is attacked; nothing external or third-party is touched. Every
  tool runs against *this* emulator's own process.
- **No DRM circumvention, no piracy.** The target dump's SELF segments are **already unencrypted**; the
  project uses **no console keys, no signature bypass, no copy-protection defeat**. The game dump is the
  developer's own legally-obtained copy, is **never redistributed**, and is gitignored.
- **No Sony code, firmware, or keys** are included or reproduced. prosper reimplements *published* library
  interfaces from scratch (clean-room-style), the way Wine reimplements the Win32 API.
- **The low-level tooling** (`PROSPER_HWBP` hardware breakpoints, `fault_handler` SIGSEGV handling,
  `PROSPER_HWWATCH` data watchpoints, `/proc/self/maps` classification, `boot_trace`) exists to **debug the
  emulator** — to see which Sony API the guest calls next, or where our HLE returns a wrong value. This is
  ordinary emulator/debugger instrumentation, not an attack surface.

If a task in this repo ever *looks* security-adjacent, it is because reverse-engineering a platform's ABI
for reimplementation necessarily involves reading and tracing binaries. That is interoperability RE — a
well-established, legitimate engineering practice — and it is the entirety of what happens here.

## Where things are

```
prosper/
  docs/            architecture, roadmap, graphics, verification, and per-frontier work logs
  src/self/        SELF/ELF parsing -> relocatable module image
  src/loader/      multi-module linker + global NID export table
  src/hle/         reimplemented Sony libraries (libc, libkernel, AGC/graphics), NID hashing
  src/host/        host execution: image mapping, import stubs, fault handling (Linux)
  src/gpu/         AGC->Vulkan: PM4 decode, command processor, render state, vk_translate,
                   resource layer, RDNA2->SPIR-V recompiler
  tools/           self_dump, boot_trace, shader_histo, imgdump, spv_validate
  tests/           unit + boot + Vulkan-execution tests (ctest)
```

Key docs to orient: `prosper/README.md` (status), `prosper/docs/ROADMAP.md`, `prosper/docs/GRAPHICS.md`,
`prosper/docs/RENDER_LOOP.md` (the active render frontier log), and
`prosper/docs/NEXT_STEP_EUD_DESCRIPTORS.md` (the current, precisely-scoped next task).

## Current frontier (2026-07)

The game **boots through IL2CPP into Unity's frame loop and submits real GPU draws.** A live Vulkan
renderer is wired in; the game's **real pixel shader recompiles to valid SPIR-V** and its real descriptors
+ 1920×1080 sampled texture decode correctly. The **one** remaining step to the first rendered frame is
**bindless-dynamic vertex-fetch resolution** for the vertex shader — fully specified in
`prosper/docs/NEXT_STEP_VERTEX_FETCH.md`. Start there.

## How to work here

- **Build/run in WSL Ubuntu-24.04 as root.** Build dir `prosper/build-linux` (Linux, primary),
  `prosper/build-win` (Windows/MinGW, secondary). Game dump at
  `/mnt/c/Users/matti/repos/ps5ys/PPSA24651-app0` (gitignored — **never commit it**).
  ```bash
  cd /mnt/c/Users/matti/repos/ps5ys/prosper/build-linux
  cmake --build . -j8 && ctest        # 45/45 expected green on Linux
  ```
- **Verification is agentic-first / programmatic** (`docs/VERIFICATION.md`): ctest exit code is truth;
  shaders are `spirv-val`-gated; rendered frames are pixel/CRC-asserted or dumped to BMP. No manual
  eyeballing is required to know a change works.
- **Reaching the running frame loop** needs two gated switches (off by default, so the default boot stays
  stable): `PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct`. Add `PROSPER_RENDER=1` to run the
  live renderer, `PROSPER_GFXLOG=1` for graphics diagnostics.
- **Correctness-first:** implement real behavior (cross-checked against the Kyty PS5-emulator reference in
  `../Kyty`), not shims that fake output. Mark genuinely-uncertain code with `CONFIDENCE: HIGH/MED/LOW`.
- **Commit style:** small, verified commits; push to `origin` promptly. Co-author trailer as configured.
- **Track work in GitHub issues** (`gh issue ...`). The rules:
  - **Any bug you find but do not fix in the same session gets an issue, immediately**, while the
    context is fresh: exact `file:line`, the concrete failure scenario (inputs/state → wrong
    behavior), the introducing commit if known, and a suggested fix. Apply the label `bug` plus
    `bug-hunt` if it came from a systematic review. A bug that lives only in a commit message,
    doc, or your head is a bug that gets rediscovered the hard way.
  - **Before starting non-trivial work, check the tracker** (`gh issue list --label bug-hunt`,
    or by area) — the fix may already be specified, or an issue may conflict with what you're
    about to change. Before fixing an issue, **re-verify it still exists on current master**
    (another PR may have fixed it independently).
  - **One issue → one focused PR** where feasible; reference the issue in the PR body
    (`Fixes #NN`) so the merge closes it. When a fix lands that partially addresses an issue,
    comment on the issue with what remains instead of closing it.
  - Larger planned work (frontier steps, refactors) also gets an issue when it will span
    sessions — issues are the durable queue; `docs/` files explain *how*, issues track *what
    remains*. The umbrella issue for the history-review backlog is #72
    (full annotated list: `prosper/docs/BUG_HUNT_BACKLOG.md`).
