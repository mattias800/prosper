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
  tools/           self_dump, boot_trace, shader_histo, imgdump, spv_validate,
                   snapshot (golden-image rendering regression guard — see tools/AGENTS.md)
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

- **Work in your OWN git worktree — the main checkout is shared.** Several agents (and the human)
  run this repo concurrently, so the main working directory and its build dir are contended:
  branch-switching, staging, or `cmake --build` there collides with whatever someone else is
  mid-edit on (a `git checkout` in a dirty tree tangles the index; a shared `build-linux` races
  object files). Before starting non-trivial work, create your own worktree and stay in it:
  ```bash
  git worktree add .claude/worktrees/<your-slug> -b <your-branch>   # isolated tree + branch
  # build in a worktree-local build dir; pass -DGAME_DUMP=... (dump lives outside the worktree)
  cmake -S prosper -B prosper/build-linux -DGAME_DUMP=/mnt/c/Users/matti/repos/ps5ys/PPSA24651-app0
  ```
  Never `cd` back to the main repo root to run git or builds. If you MUST touch the main checkout,
  assume another agent is actively working there — check `git status` first and don't reset/stash/
  switch branches under them (the stash stack is shared too — see the worktree note in the environment
  preamble). Your worktree is auto-cleaned when its branch merges.
- **On a Windows host, run git through PowerShell (Windows git), not WSL.** The repo lives on the
  Windows filesystem (`C:\...` = `/mnt/c/...`), and worktrees created from Windows store a
  Windows-path gitdir link (`gitdir: C:/Users/.../.git/worktrees/<name>`). WSL's git can't resolve
  that path — it mangles it to `/mnt/c/.../C:/Users/...` and dies with `fatal: not a git repository`.
  So drive **git** (fetch/checkout/commit/push/worktree/status) from **PowerShell**, and use **WSL
  only for cmake/build/run**. Don't mix the two on one repo (it also avoids CRLF/filemode index churn).
- **Build/run in WSL Ubuntu-24.04 as root.** Build dir `prosper/build-linux` (Linux, primary),
  `prosper/build-win` (Windows/MinGW, secondary). Game dump at
  `/mnt/c/Users/matti/repos/ps5ys/PPSA24651-app0` (gitignored — **never commit it**).
  ```bash
  cd /mnt/c/Users/matti/repos/ps5ys/prosper/build-linux
  cmake --build . -j8 && ctest        # 45/45 expected green on Linux
  ```
- **Verification is agentic-first / programmatic** (`docs/VERIFICATION.md`): ctest exit code is truth;
  shaders are `spirv-val`-gated; rendered frames are pixel/CRC-asserted or dumped to BMP. No manual
  eyeballing is required to know a change works. **After any change that can affect rendered output**
  (recompiler, AGC decode, render state, detile, executor/present), run the golden-image guard
  `python3 tools/snapshot/snapshot.py check` (local-only, boots a real game and pixel-hashes an exact
  frame vs a stored baseline — see `tools/snapshot/AGENTS.md`).
- **Reaching the running frame loop** needs two gated switches (off by default, so the default boot stays
  stable): `PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct`. Add `PROSPER_RENDER=1` to run the
  live renderer, `PROSPER_GFXLOG=1` for graphics diagnostics.
- **Correctness-first:** implement real behavior (cross-checked against the Kyty PS5-emulator reference in
  `../Kyty`), not shims that fake output. Mark genuinely-uncertain code with `CONFIDENCE: HIGH/MED/LOW`.
- **Kyty is a reference, NOT an oracle — and its reliability is split by platform generation.**
  Kyty CAN run PS4 games, so the PS4-inherited surface (libkernel, pthreads, equeue, VideoOut,
  filesystem, GNM-era graphics concepts) is exercised by real titles and is solid evidence — PS5
  evolved from PS4, so those behaviors usually carry over. But Kyty's PS5-SPECIFIC surface is
  early transcription work it never runs: **AGC (the PS5's replacement for GNM) especially** —
  Gen5 Dcb builders, T#/V# Gen5 descriptor formats, PS5-only kernel calls (APR/Ampr/BatchMap).
  There, prosper's own live captures are the only exercised evidence (we boot the PS5 titles;
  Kyty does not). Trust order when sources disagree: (1) prosper's live captures/traces of the
  real guest, (2) Kyty+shadPS4 agreeing (strongest for PS4-inherited surfaces), (3) either alone
  — cite which and mark CONFIDENCE accordingly, and weight Kyty DOWN for anything Gen5/AGC.
  Never weaken a behavior that a live boot demonstrates just to match a reference.
- **PS5 3.20 firmware library reference — the definitive NID↔name database (`../PS5-3.20_Libs/`).**
  A `genstub.py`-generated dump of **all 275 PS5 3.20 system libraries**, one `libSceXxx.c` per library.
  Each file lists **every exported function AND its exact NID**: the loader lines read
  `sprx_dlsym(__handle, "<NID>", &__ptr_<funcName>)`, so each is a `<NID> ↔ <funcName>` pair. This is
  the **authoritative PS5-specific symbol map** — trust it OVER the PS4-era shadPS4 aerolib / Kyty for
  any Gen5/PS5-only surface (AGC, Ampr, Pad/UserService Gen5, etc.), and use it to see a library's
  *complete* real API surface (e.g. what functions exist that a title might call). Recipes:
  - Resolve an unknown NID → name: `grep -rn '<NID>' ../PS5-3.20_Libs/` (the matching `__ptr_<name>` names it).
  - A library's full export list: `grep -oE '\.global sce[A-Za-z0-9]+' ../PS5-3.20_Libs/libSceXxx.c | sort -u`.
  - Which library exports a symbol: `grep -rl '<funcName>' ../PS5-3.20_Libs/`.
  Caveat: it gives **names + NIDs only, no bodies** — argument layouts and behavior still come from
  prosper's live captures + Kyty/shadPS4. Gitignored sibling of the repo; never commit its contents.
- **Commit style:** small, verified commits; push to `origin` promptly. Co-author trailer as configured.
  **Do NOT add "Generated with Claude Code" attribution lines** (the 🤖 badge, "Generated with
  [Claude Code](...)" footers, session links) to PR bodies, commit messages, issue text, or
  comments — anywhere. They are noise in the project record; the co-author trailer alone is enough.
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
  - **Claiming an issue (multi-agent lock).** Several agents work this repo concurrently; claim
    before you code so two agents never fix the same issue:
    1. Check it's free: no `in-progress` label, no unexpired claim comment, and no remote fix
       branch — `gh issue view NN` + `git ls-remote origin 'refs/heads/fix/issue-NN-*'`.
    2. Claim it: add the `in-progress` label AND post a comment
       `CLAIMING: <agent/session identifier> | branch fix/issue-NN-<slug> | <UTC timestamp>`.
    3. Re-check for a race: re-read the comments right after posting; if two claims landed, the
       EARLIER timestamp wins (tie → lexicographically smaller branch name); the loser comments
       "unclaiming" , removes the label only if they added it, and picks other work.
    4. Push your `fix/issue-NN-<slug>` branch early — the remote branch doubles as the lock.
    5. Release: the merged `Fixes #NN` PR closes the issue (label goes with it). If you abandon
       or park the work, comment what state it's in and remove the `in-progress` label.
    6. Claims expire: `in-progress` with no open PR and no activity for 24 h is stale — anyone
       may re-claim after posting a takeover comment.
  - **Stay in your area lane (multi-workstream).** Separate workstreams run in parallel, each
    possibly coordinating its own sub-agents, split by title/subsystem via `area:` labels:
    `area:ue4` (PPSA17942 / Unreal Engine — IoStore/APR/Ampr, RHI, the UE allocator) and
    `area:messenger` (PPSA24651 / The Messenger, IL2CPP/Unity, and shared infra: loader, libc,
    libkernel, the recompiler, generic GPU/AGC). **Do NOT touch code or claim issues outside your
    workstream's area** — cross-lane edits collide with the other workstream's in-flight sub-agents.
    If a fix genuinely spans both (shared infra that a UE issue also needs), coordinate on the issue
    first; default to the shared-infra/Messenger lane owning shared files. Label every new issue with
    its `area:` so the lanes stay legible; when unsure which lane a file belongs to, check which
    title's boot path exercises it.
