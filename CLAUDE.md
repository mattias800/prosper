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
                   snapshot (golden-image rendering regression guard — see tools/AGENTS.md),
                   il2cpp/ (prx_to_elf.py + resolve.py: flatten a SELF -> plain ELF for Il2CppDumper
                   managed-symbol recovery), re/xref.py (cross-reference finder for unsymbolicated
                   PS5 modules: "who references address X" over calls + rip-relative refs + Sony
                   relocations, which objdump/readelf can't decode)
  tests/           unit + boot + Vulkan-execution tests (ctest)
```

Key docs to orient: `prosper/README.md` (status), `prosper/docs/ROADMAP.md`, `prosper/docs/GRAPHICS.md`,
`prosper/docs/RENDER_LOOP.md` (the historical render bring-up log), and
`prosper/docs/MESSENGER_BLACK_RENDER.md` (the current, revisioned Messenger investigation status).

## Historical frontier (superseded 2026-07-11)

The game **boots through IL2CPP into Unity's frame loop and submits real GPU draws.** A live Vulkan
renderer is wired in; the game's **real pixel shader recompiles to valid SPIR-V** and its real descriptors
+ 1920×1080 sampled texture decode correctly. The **one** remaining step to the first rendered frame is
**bindless-dynamic vertex-fetch resolution** for the vertex shader — fully specified in
`prosper/docs/NEXT_STEP_VERTEX_FETCH.md`. This paragraph is historical only.

## Current frontier (2026-07-11)

The game now **boots through IL2CPP, renders its intro/title/menu, and reaches gameplay with real GPU
draws.** The old bindless vertex-fetch frontier is complete: both shader stages recompile and dynamic
V#/T#/S# resources resolve on current master. Do **not** start from `NEXT_STEP_VERTEX_FETCH.md`; it is
retained as a historical bring-up record.

The save-game list is visible (#299 closed), and retaining color-disabled depth/stencil passes (#520) recovers
the first level's source scene. The black gameplay root cause was fixed on master by #528 (`e5fce22`):
the direct 1024x32 RGBA16F grading-LUT producer was skipped because the recompiler lacked
`V_CVT_OFF_F32_I4`, while the live renderer also treated its offscreen target as a VideoOut-sized surface.
Resource-producer history (#524) identified the exact writer; implementing the opcode and decoding
`CB_COLOR0_ATTRIB2` target dimensions (#526/#527) produces the real LUT, preserves it through the original
grading shader, and yields a visible first-level front buffer without diagnostic resource substitution.

Two later fixes complete the hardware-oracle composition and restore boot progression: #534 corrects reversed
`VkFrontFace` enum translation, recovering the foreground tree, terrain/platforms, waterline, and structures;
#541 tracks depth and stencil validity independently in persistent D32S8 surfaces, so stencil-only logo use
cannot poison the intro's reverse-Z depth plane. A clean scripted route produced 180 native 1920×1080 frames,
and the user confirmed the first-level graphics match the hardware reference. #530 and #540 are closed.

Cross-title breadth has advanced: *Dead Cells* now starts reliably after the AGC resource-name fix (#544), its
exercised NGS2 lifecycle returns initialized sizes/handles/state and silent output (#554), and a late render
window reaches the Evil Empire splash. #545 was software-render throughput, not a guest deadlock: synchronous
3840×2160 llvmpipe rendering stretches its ~13,000-submit startup into minutes.

Dead Cells now has a deterministic route through splash/menu into gameplay. HUD and partial composition render,
but the world is mostly white (#566). Version-4 `.prgcap` captures seed temporal RTT inputs (#568) and isolate the
first bad composition at draw 18; one 642x362 input has no prior color-target writer. The kernel-derived dispatch
thread/local/group contract (#580), `sceAgcCbSetShRegistersDirect`, and compute direct type-1 V# binding (#574)
now execute the real fill kernel against guest buffers before submit completion (#576). Range provenance proved
draw 19 consumes one backing, dispatch 5 fills it, then draw 31 consumes it again in one submit. Graphics spans and
compute now execute by retained PM4 order (#584), fixing that future-read. The later overbright screenshot was a
warmup artifact: the 35-second render delay skipped a 642x362 RTT producer, then a replace-copy sampled dispatch
4's raw all-`0xFF` backing and cached it indefinitely. `PROSPER_RENDER_TARGET_DIM=642x362` preserves the real
opening vignette/level geometry; #586 now tracks a practical late checkpoint with that history intact. The
residual seeded replay mismatch (#569) and animation-sensitive exact splash guard (#573) remain separate issues.
`PROSPER_PROVENANCE_DIM=WxH` reports overlapping color, compute, DMA_DATA, and WRITE_DATA writers with
submit/item/PM4 ordinals.
`PROSPER_RESOURCE_HASH_DIM=WxH` correlates raw and sampled hashes with those writers at each live draw;
`PROSPER_TARGET_STEP_HASH_DIM=WxH` plus `PROSPER_TARGET_STEP_HASH_MIN_DRAWS=N` prefix-bisects a target
pass using content metrics without writing per-draw images.
`PROSPER_DESCRIPTOR_VALIDATE=strict|poison` and `gpu_replay --validate` are landed capabilities from #515.
Do not restart the superseded
Messenger depth, vertex-fetch, geometry, palette, or tiling hypotheses without contradictory new evidence.

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
  cmake --build . -j8 && ctest        # 87/87 expected green on Linux
  ```
- **Verification is agentic-first / programmatic** (`docs/VERIFICATION.md`): ctest exit code is truth;
  shaders are `spirv-val`-gated; rendered frames are pixel/CRC-asserted or dumped to BMP. No manual
  eyeballing is required to know a change works. **After any change that can affect rendered output**
  (recompiler, AGC decode, render state, detile, executor/present), run the golden-image guard
  `python3 tools/snapshot/snapshot.py check` (local-only, boots a real game and pixel-hashes an exact
  frame vs a stored baseline — see `tools/snapshot/AGENTS.md`).
- **Build tools — don't avoid them.** This project is a long reverse-engineering effort, and getting
  progressively more complicated games running will keep demanding new instrumentation. When you hit a
  question the existing tools can't answer — "who references this address in the unsymbolicated eboot?",
  "what managed method is at this PC?", "which draw wrote this pixel?", "what does this guest struct look
  like at runtime?" — **write the tool** rather than hand-grinding it once and moving on. The payoff is
  compounding: a good tool turns hours of manual disassembly into a lookup and pays off on every future
  title. Prefer a small, documented, committed tool (in `tools/`, or a gated `PROSPER_*` diagnostic in the
  emulator) over a throwaway script, and reuse what exists first (`self_dump`, `boot_trace`, `PROSPER_HWBP`/
  `HWWATCH`/`PEEK`/`FAULTMEM`, `tools/il2cpp/prx_to_elf.py` + Il2CppDumper, `tools/re/xref.py`, the snapshot
  guard) — the RE toolbox is the force multiplier, so grow it deliberately, verify it on a known answer,
  and land it so the next agent (and the next game) inherits it.
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
