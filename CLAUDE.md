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
  src/host/        host execution: per-platform image mapping, ABI stubs, fault handling
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

Key docs to orient: `prosper/README.md` (status) and `prosper/docs/ROADMAP.md` (what is planned).
For anything title- or subsystem-specific, use the table in the next section rather than guessing.

## Where the project stands (2026-08-11)

`COMPATIBILITY.md` is authoritative for the **user-facing per-title milestones** — 39 tracked titles
at this refresh. Do not duplicate its rung counts here; read it, then open the one doc named below for
whatever you are about to touch. This section is a map, not a status report. Long-lived `tracker:game`
issues and their comments carry each active title's current development rung, route, blockers and
evidence; dated cross-title performance measurements live in their own ordinary issues (the
2026-08-02 census is #1739).

**Concurrent game work starts with `prosper/docs/GAME_COMPAT_ORCHESTRATION.md`** — lane ownership,
shared-GPU policy, the instrument-not-the-subject list, and the dated current handoff.

### Which doc to read next

| Title / area | State | Read this next |
| --- | --- | --- |
| *The Messenger* `PPSA24651` | rung 6 — complete first level, checked against PS5 hardware; `messenger-scene` guard | `docs/MESSENGER_BLACK_RENDER.md` |
| *Dead Cells* `PPSA15552` | rung 6 — full-colour Prisoners' Quarters; `dead-cells-gameplay` guard | `docs/DEAD_CELLS_STATUS.md` |
| *Blasphemous 2* `PPSA13579` | rung 6 — first playable room; `blasphemous2-gameplay` guard | `scripts/blasphemous2/README.md` |
| *Alex Kidd in Miracle World DX* `PPSA02664` | rung 6 — resolved by #1578 (generic 4 KiB mip-tail tiling); `alexkidd-gameplay` guard | `docs/PPSA02664_BLACK_WORLD.md` (historical) |
| *Blue Prince* `PPSA25009` | rung 6 — sustained Day One gameplay and the Mount Holly entrance hall; reviewed automatic gameplay snapshot guard (tracker #1808) | `docs/BLUE_PRINCE_STATUS.md` |
| *Syberia: Remastered* `PPSA30140` | rung 3 — gameplay renders; menu 3D layer and gameplay composite degraded (#1619 / #1627) | `docs/SYBERIA_STATUS.md` |
| *Dragon Quest VII Reimagined* `PPSA17942` (a.k.a. DOLL) | rung 2 — title, name entry, onboarding; composition defect and gameplay open | `docs/DRAGON_QUEST_STATUS.md` |
| *Nikoderiko* `PPSA23760` | rung 2 — 3D world dropped; **blocked on #305**, not on the recompiler | `docs/NIKODERIKO_STATUS.md` |
| *Asterix & Obelix: Babylon Mission* `PPSA30490` | rung 2 — logo movies, intro cutscene and title menu; the video-splash seek deadlock is fixed (#1949) | `docs/ASTERIX_BABYLON_STATUS.md` |
| *R-Type Delta: HD Boosted* `PPSA26414` | rung 2 — logo, full-colour opening movie, title screen and attract mode. Launch needs `tools/dropcache.py` first: the title races its own 400 ms user-event delay against prosper's asset load (#1746, a product decision, not an open investigation) | `docs/R_TYPE_DELTA_STATUS.md` |
| *The Oregon Trail* `PPSA19244` | rung 2 — title screen rendered on a default launch (60 s / 3,689 frames clean); the ordered-DMA stall is fixed (#1987) and **#1945 no longer reproduces here** (0 of 9 arms — use `PPSA21406` for that repro). The UI layer composites correctly since #1946 — its solid-block glyphs, black logo panel and flat sky were one defect: the RT0 blend state never reached the GPU | `docs/OREGON_TRAIL_STATUS.md` |
| *Little Nightmares III* `PPSA05143` | rung 2 — title screen rendered on a default launch; the render-thread stall is fixed (#1987). Most title frames arrive with red and green forced to maximum, reading as a yellow background (#2014) | `docs/LITTLE_NIGHTMARES_3_STATUS.md` |
| *ArcRunner* `PPSA21406` | rung 0 — renderer bring-up reaches real GPU submissions, then the render thread faults (#1226) | `docs/ARCRUNNER_STATUS.md` |
| *Sonic Frontiers* `PPSA03831` | rung 2 — 4K opening sequence, auto-save notice, title screen and main menu on a default launch. The four-session black-screen wall was one unregistered NID answering `SCE_OK`: `sceSaveDataTransferringMountPs4` (#2023). The menu heading draws the wrong string (#2206) | `docs/SONIC_FRONTIERS_STATUS.md` |
| *Sonic Racing: CrossWorlds* `PPSA08804` | rung 2 — a pulsed pad route reaches the complete 4K title screen and profile menu; the profile panel is black and the sequence later holds on white (#2013 / #2358 / #2360) | `docs/SONIC_CROSSWORLDS_STATUS.md` |
| *Grand Theft Auto V* `PPSA04263` | rung 3 — routed gameplay entry with real GPU draws; the HUD, radar and tutorial text render over an absent 3D world. The descriptor-array lift is complete; use the exact program-tagged terminal census in #2481 rather than the superseded aggregate CFG count | `docs/FLAT_LOAD_DESIGN.md` (historical design), tracker #1873, issue #2481 |
| *GRIS*, *Space Adventure Cobra*, *Sonic Origins* | rung 3 / rung 3 / rung 1 — Sonic's black startup loop is fixed (#1905: `sceSaveDataCreateTransactionResource` must return a positive resource id); it renders the 4K SEGA logo and then holds on white | `docs/GRIS_SONIC_COBRA_BRINGUP.md` |
| Concurrent title work | the 2026-07-31 lane allocation is historical; use live issue claims for ownership and the dated checkpoint for the current cross-lane handoff | `docs/GAME_COMPAT_ORCHESTRATION.md` |
| *Tactics Ogre: Reborn* `PPSA03839` | rung 3 — gameplay reached; HEVC movies render, sprite/HUD composition remains open | `docs/TACTICS_OGRE_STATUS.md` |
| Every other title | — | `COMPATIBILITY.md` |
| UE4 / IoStore bring-up (shared) | — | `docs/UE4_APR_IOSTORE_BRINGUP.md`, `docs/CROSS_ENGINE_UE4.md` |
| Renderer performance | July pass complete; the stop decision is recorded | `docs/RENDERER_PERFORMANCE_2026_07.md` |
| Windows port / release | native SDL3+Vulkan frontend, screenshots, release path | `docs/WINDOWS_PORT_HANDOFF.md`, `docs/WINDOWS_RELEASE.md` |
| Linux desktop release (AppImage + tarball) | published on `v*` tags; packaging is built and verified on every PR | `docs/LINUX_RELEASE.md` (users), `packaging/linux/README.md` (how, and its `## Ruled out`) |
| GPU capture / replay / timeline tooling | F9 frame grab → offline `.prgbundle` / `.prgcap` workflow, and every `PROSPER_*` graphics diagnostic | `tools/gpu_replay/README.md`, `tools/gpu_timeline/README.md`, `tools/AGENTS.md` |
| Graphics architecture, resource binding, recompiler | — | `docs/GRAPHICS.md`, `docs/RESOURCE_BINDING.md`, `docs/RECOMPILER_REMAINING.md` |
| AGC command-packet sizes (a builder's dword count is an ABI contract with the guest's own reservations) | — | `docs/AGC_PACKET_SIZES.md` |

### Before you form a hypothesis, read the `## Ruled out` section

Every `docs/*_STATUS.md` in that table — and `PPSA02664_BLACK_WORLD.md`, `MESSENGER_BLACK_RENDER.md`
and `GRIS_SONIC_COBRA_BRINGUP.md` — carries a **`## Ruled out`** section: one line per
already-falsified hypothesis, the evidence that killed it, and the issue/PR link. (Blasphemous 2's
row points at a route README, which has none.) Cross-title
falsifications live in the area doc (`docs/RESOURCE_BINDING.md`, `docs/RECOMPILER_REMAINING.md`).
These exist so nobody re-derives a dead answer at full cost — they are the most expensive knowledge
in the repository. Extend them; see the recording rule under *How to work here*.

The standing warnings that are **not** title-specific:

- **Do not restart the superseded Messenger depth, vertex-fetch, geometry, palette, or tiling
  hypotheses without contradictory new evidence.** Each is falsified with evidence in
  `docs/MESSENGER_BLACK_RENDER.md` § Ruled out. The real cause was a missing recompiler instruction
  plus loss of per-target dimensions (#526/#527, fixed by #528), with #534 and #541 behind it.
- **Do not resume Messenger-specific renderer cache work toward 60 FPS.** The July pass deliberately
  stopped at roughly 12 → 24 FPS on the first level; the remaining synchronous graphics/compute
  boundaries must be evaluated against a **3D** workload first (*Summer Sports Games* `PPSA03416` is
  the first clean candidate). `docs/RENDERER_PERFORMANCE_2026_07.md`.
- **Start Windows work from `docs/WINDOWS_PORT_HANDOFF.md`**, not the solved pre-render fence
  investigation; Windows *release* users start from `docs/WINDOWS_RELEASE.md`.
- **A historical capture hash is not a current renderer oracle**, and neither is a target extent or a
  raw draw count on its own. Addresses and operation ordinals are run-local. Re-derive the selector
  with `gpu_timeline --signatures` / `--select` before recording a new bundle.
- **`PROSPER_UD_TAIL_ALIGN` must stay off.** It exists only so the A/B that falsified the
  user-data-tail hypothesis stays reproducible (`docs/RESOURCE_BINDING.md` § Ruled out).
- **Read `GAME_COMPAT_ORCHESTRATION.md`'s instrument-not-the-subject list before believing any
  surprising measurement** — every entry on it is a phantom defect that came from the apparatus
  rather than the subject, and several cost hours. (Deliberately no count here: the list is appended
  to by concurrent lanes, so a restated total is stale as soon as it is written.) Two of its positive rules bind on any graphics investigation, not
  just orchestrated ones: **a decoded-draw census is meaningless without the render phase and a
  positive control** (draw counts vary by two orders of magnitude *within one title* — count per
  sample interval, calibrate against a title known to render the thing you claim is missing, and
  open the frames), and **prefer experiments that detect their own invalidity** over experiments that
  can fail silently. Check a diagnostic's rate limit before quoting its volume as a frequency, and
  check whether it clears `live_gpu_targets` before comparing it against a default run.

- **A positive control drawn from the same source as the null it validates tests the DISCRIMINATOR,
  never the DOMAIN.** This qualifies the rule above, and it is the harder half. "A discriminator that
  cannot show its lever moved is void" is true — but a lever that *does* move proves only that the
  machinery runs, not that the space it searches can express the case you are testing for. Worked
  example (2026-08-06, instrument trap 122): a reviewer's fuzz reported zero instances of a widening
  class. Suspecting the zero, they ran a positive control — **and it passed**: 41,206 constructions,
  17,659 narrowings, 0 widenings. It could not have done otherwise, because it drew from the same
  generator and so inherited the same broken geometry, in which the case was *structurally
  inexpressible*. Interior geometry surfaced 1,277 instances immediately.
  **So: before believing a clean zero, construct one positive instance of the class BY HAND, outside
  whatever produced the null.** A same-source control confirms the machinery fires on cases the source
  can build — which is exactly the assumption in question. This is the sharpest form of the family the
  instrument-trap list records, because unlike the silent instruments in traps 64, 116 and 121, this
  one is *loud*, and the noise is what launders the null.

### Superseded documents

`docs/NEXT_STEP_VERTEX_FETCH.md` (bindless-dynamic vertex fetch, superseded 2026-07-11) and
`docs/RENDER_LOOP.md` (the render bring-up log) are historical records. Both frontiers are complete:
both shader stages recompile and dynamic V#/T#/S# resources resolve on current master, and the
render-loop frontier is complete. Each carries its own superseded banner — do not start work from
either, and do not read `RENDER_LOOP.md`'s "Status: open" as current.

## How to work here

- **Concurrent game work starts with `prosper/docs/GAME_COMPAT_ORCHESTRATION.md`.** It defines the orchestrator and
  subagent contract, shared-GPU policy, evidence/PR cadence, and the dated current handoff for active title lanes.
  Update its current-state sections when ownership, evidence, or the exact frontier changes; keep detailed findings in
  GitHub issues so the document remains a discoverable map rather than the only record.

- **A PR that falsifies a hypothesis records it in the relevant `## Ruled out` section before merging** —
  the same way instrument traps are recorded in `GAME_COMPAT_ORCHESTRATION.md`. One line: the dead
  hypothesis, the evidence that killed it, and the issue/PR link. Title-specific → that title's status
  doc; cross-title → the area doc (`RESOURCE_BINDING.md`, `RECOMPILER_REMAINING.md`, …). A falsification
  that lives only in a PR body or an issue comment is one the next agent will re-derive at full cost.
  **A result reported only in a message to an orchestrator is not in the project record at all** — if a
  lane reports an A/B to you, land it in the doc or the issue before the session ends. Read the
  `## Ruled out` section of every doc you are about to work in **before** forming a hypothesis.

- **Work in your OWN git worktree — the main checkout is shared.** Several agents (and the human)
  run this repo concurrently, so the main working directory and its build dir are contended:
  branch-switching, staging, or `cmake --build` there collides with whatever someone else is
  mid-edit on (a `git checkout` in a dirty tree tangles the index; a shared `build-linux` races
  object files). Before starting non-trivial work, create your own worktree and stay in it:
  ```bash
  git worktree add .claude/worktrees/<your-slug> -b <your-branch>   # isolated tree + branch
  # build in a worktree-local build dir; pass -DGAME_DUMP=... (dump lives outside the worktree)
  cmake -S prosper -B prosper/build-linux -DGAME_DUMP=<DUMP_ROOT>/PPSA24651-app0
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
  `prosper/build-windows` (Windows/MinGW, secondary). Game dump at
  `<DUMP_ROOT>/PPSA24651-app0` (gitignored — **never commit it**).
  ```bash
  cd <REPO_ROOT>/prosper/build-linux
  cmake --build . -j8
  ctest --no-tests=error              # report the discovered count and exit code
  ```
- **Write run artifacts and build temporaries to the real disk, never `/tmp`.** On the Linux box `/tmp`
  is a **RAM-backed tmpfs sized at 50% of RAM, with a per-user quota shared by every concurrent agent**.
  A single `.prgcap` is 200 MB–2.7 GB, a 4K `.bmp` is 24 MB, and a `PROSPER_GFXLOG`/`PROSPER_DBG` run log
  reaches 1.5 GB, so a few sessions of ordinary capture work exhaust the quota. When that happens it does
  not just fail your write — it **eats the machine's RAM and kills the Bash tool outright for every
  agent**, because the harness stores each command's output under `$TMPDIR`. Put captures, frame dumps,
  screenshots and logs under **`$HOME`** — a gitignored worktree-local directory, or a scratch directory
  such as `~/dq-work/`. **Do not send a run's artifacts to `/var/tmp`: distrobox shares `$HOME` but NOT
  `/var/tmp`**, so anything the emulator writes there lands in the *container's* private `/var/tmp` and is
  invisible from the host (the host path shows up inside the container as `/run/host/var/tmp`). The shell
  redirect that captures a run log happens on the host, so a `/var/tmp` run directory silently ends up
  holding only the log while every screenshot and capsule goes somewhere you cannot see. `/var/tmp` is
  still fine for host-side scratch (it is real disk and `systemd-tmpfiles` ages it) — just not for output
  produced inside the container. Always build with `TMPDIR` off the tmpfs too; gcc's compile temporaries
  are large enough on their own to break a `-j8` build with
  `error writing to /tmp/ccXXXX.s: Disk quota exceeded`:
  ```bash
  mkdir -p <worktree>/build/tmpdir
  distrobox enter ps5ys -- bash -lc "cd <worktree> && TMPDIR=\$PWD/build/tmpdir cmake --build build -j6"
  ```
  **Diagnosing an exhausted `/tmp`:** every Bash call returns exit 1 with no output — including `true`
  and `echo hello`, foreground or background — which looks identical to a dead working directory. The
  tell is the **Write tool returning `EDQUOT`** for a path under `/tmp` (if Write to `/tmp` succeeds, it
  is the cwd instead). It is recoverable from inside the session: the commands still *execute*, only the
  output capture fails, so redirect stdout **and** stderr to a file on the real disk, end with `exit 0`,
  read that file, and `rm -rf` scratch that is a day or more old. Leave `/tmp/claude-*` and anything
  touched in the last few hours alone — other agents are live.
- **Verification is agentic-first / programmatic** (`docs/VERIFICATION.md`): ctest exit code is truth
  **only with `--no-tests=error`** — plain `ctest` on a build directory with nothing registered prints
  `No tests were found!!!` and **exits 0**, so "no tests ran" and "everything passed" are the same
  exit code (measured on ctest 4.4.0: plain → 0, `--no-tests=error` → 8). This is reachable without
  any mishap — a copy-pasted `--test-dir` pointing at the wrong build directory yields a green run
  that executed nothing, and an agent reporting "ctest green" has then reported a fact about an empty
  directory. `prosper/tools/vkval/vk_validation_scan.py:206` already passes the flag and its comment
  says why; nothing else in the repo does. **Pass `--no-tests=error` whenever you quote a ctest
  result**, and quote the test *count* alongside the exit code — a count is falsifiable, an exit code
  alone is not;
  every SPIR-V emitter is `spirv-val`-gated in CI (`tools/spv_validate`, §4c — one representative
  module per emitter path, not per game shader); rendered frames are asserted by pixels, hashes, or routed content
  metrics. Snapshots are a **release-time regression inventory**, not a day-to-day development or merge
  gate. Before every release, run the full local-only matrix with
  `python3 tools/snapshot/snapshot.py check` against the release candidate and review every result.
  During normal development, including long stretches of render work, authors may skip snapshots
  entirely or run only useful focused guards; the PR author decides, unless the task explicitly requires
  a run. Snapshot results do not define whether master or an individual PR is acceptable: either may
  regress a guarded title, and a correct fix may even require an intentional cross-title tradeoff. The
  release process is where detected regressions are
  fixed or explicitly accepted and documented before publishing artifacts. Record any snapshots that
  were run in the PR, but do not delay routine iteration merely to keep the whole matrix green. Gameplay
  guards inspect multiple frames in route-specific windows and use SSIM over compact luminance
  signatures from several reviewed states, looking for major collapse without rejecting subtle pixel improvements.
  Every new or materially changed baseline requires `snapshot.py verify`,
  inspection of all retained images from both runs, and a factual `review` note. See
  `tools/snapshot/AGENTS.md`.
- **Per-title bring-up ladder — gameplay is always the target.** Every game climbs the same ladder,
  and progress claims name the rung reached:
  1. **Any graphics at all** — real frames from the live renderer (a splash/logo counts; black or
     diagnostic-only output does not).
  2. **Title screen** reached and rendered.
  3. **Gameplay** reached with real GPU draws — this is the standing goal of bring-up; don't stop
     and declare success at a logo or menu.
  4. **Manual visual verification** — the user plays it and confirms by eye that it looks right.
  5. **Oracle comparison** — reference screenshots from PS5 hardware, compared against routed
     captures for visual correctness.
  6. **Automatic snapshot testing** — a deterministic route + content guard added to
     `tools/snapshot`, so regressions are caught without a human.
  Only after step 6 is a title considered **"done"**, and work moves to the next game. Earlier rungs
  are milestones worth recording (issue/`COMPATIBILITY.md`), not stopping points.
- **Reach for free vendor/open-source tooling BEFORE building a `PROSPER_*` switch.** prosper is an
  ordinary Vulkan application, so AMD's and Khronos' own tools work on it directly and need no
  per-title work. **Measured on Linux/AMD 2026-08-08 with no change to prosper's code**, stated as
  what was actually observed rather than as what the tools promise:
  - **An RGP capture succeeds** from one variable (`MESA_VK_TRACE=rgp`), and RADV reports
    `instruction timing: enabled, cache counters: enabled, queue events: enabled` on it. **Reading a
    `.rgp` needs AMD's GUI — nobody here has opened one**, so treat its per-draw timing and occupancy
    as the tool's documented capability, not as a project measurement.
  - **`radeontop` answers "is this GPU-bound?" in one run, no capture and no GUI.** *Blue Prince* at
    ~3.2 fps leaves the GPU at **4.17%** against a `vkcube` control at **56.31%** — so with #2215's
    30-45 ms/submit, its submits are long because they **wait**. That control is mandatory here: the
    tool prints "Unknown Radeon card" and its `ta`/`ee` counters are dead on this chip.
  **`docs/GPU_PROFILING_EXTERNAL.md` has the recipes and the way each tool lies** — RGP writes to
  `/tmp`, a frame ordinal silently captures the wrong regime, and the Fedora RenderDoc package ships
  no Python bindings. Build a `PROSPER_*` switch only for something the guest-facing layer knows and
  the GPU vendor cannot see.
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
  - **For a graphical bug or an FPS drop, the fastest loop is the F9 frame grab.** While a title runs in
    `prosper-app`, pressing **F9** captures the current frame — the game's real GPU commands, shaders, and
    resources, plus the renderer-owned RTTs it samples (seeded) — into a replayable `.prgbundle` + a `.bmp`
    screenshot (`PROSPER_CAPTURE_DIR`, default cwd). Then `tools/gpu_replay --bundle <file>` reproduces
    that exact frame **offline and deterministically**, and `--inspect-only` / `--draw-steps` /
    `--dump-resource` (via `--bundle-extract-submit`) dissect it draw-by-draw. This answers "which draw
    wrote this pixel / why does this frame look wrong / why is it slow" without re-booting and re-routing
    to catch the moment live — press F9 once, then iterate the fix on the frozen frame.
    It captures *rendered-frame* bugs (not CPU/logic/audio). See `tools/AGENTS.md` (interactive frame grab)
    and `tools/gpu_replay/README.md`.
  - **You do NOT need a human at the keyboard: both captures are schedulable, and an agent can run the
    whole thing headless.** This is written here because it was missed for months — the triggers live in
    `prosper-app`, which is **off by default** (`-DPROSPER_APP=ON`), so an env sweep of `src/` never sees
    them, and lanes built weaker instruments instead. Two rich diagnostics, four one-shot triggers:

    | capture | after a wall-clock delay | at a host frame |
    | --- | --- | --- |
    | **F9** replayable `.prgbundle` + `.bmp` | `PROSPER_GRAB_BUNDLE_AFTER_MS` | `PROSPER_GRAB_BUNDLE_AT_FRAME` |
    | **F8** bounded `.prperf` (5 s before + 5 s after) | `PROSPER_PERF_CAPTURE_AFTER_MS` | `PROSPER_PERF_CAPTURE_AT_FRAME` |

    ```bash
    cmake -S prosper -B build -DPROSPER_APP=ON          # the app is NOT built by default
    SDL_VIDEODRIVER=offscreen \                         # LINUX ONLY -- see below
    PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
    PROSPER_CAPTURE_DIR=$HOME/work \
    PROSPER_GRAB_BUNDLE_AFTER_MS=18000 PROSPER_PERF_CAPTURE_AFTER_MS=18000 \
        ./build/prosper-app <DUMP_ROOT>/<TITLE_ID>-app0
    ```

    **`SDL_VIDEODRIVER=offscreen` is Linux-only. Drop it on Windows** — the triggers themselves are
    platform-independent and fire correctly there with an ordinary window. SDL's offscreen driver needs
    `VK_EXT_headless_surface`, which NVIDIA's Windows Vulkan driver does not expose, so the app dies
    before writing anything:

    ```
    Installed Vulkan doesn't implement the VK_EXT_headless_surface extension
    [app] SDL_Vulkan_CreateSurface: VK_EXT_headless_surface extension is not enabled ...
    terminate called without an active exception
    ```

    That signature is recorded because it does not look like what it is: no artifacts and a
    non-graceful exit read as "the capture triggers are broken", when the triggers were never reached.

    Which axis to use: **`_AFTER_MS` aims at an event you can only describe in time** ("when the movie
    starts", "when the rate collapses"); **`_AT_FRAME` aims at an ordinal you already located** with a
    cheap `tools/screenshot` sweep or `PROSPER_CAPTURE_SCREENSHOT_AT_FRAME`, and is the reproducible one.
    Point both at the same moment to get the timing breakdown and the replayable frame together.

    Each is **one-shot and opt-in**, and a malformed value **disables its own trigger** rather than firing
    at an unintended moment — so a typo costs you a capture, never a wrong measurement. Read the
    `.prperf` with `tools/perf/performance_capture_report.py` (it reports by **time**; the record counts
    are not load), and the bundle with `tools/gpu_replay --bundle`.

    **Read the F8 report before optimising anything — and check the capture can SEE the leaf you are
    reading.** Worked example (#2215, 2026-08-07): Blue Prince at 1 fps, and the capture said
    `gpu-device` was **2.2%** of the frame while `renderer-resource` was 47%. That half stands.

    The same example used to end "and `buffer copy` was **2.4%** during the collapse". **That number was
    wrong, and this line kept publishing it for a day after it was retracted.** The capture predated
    #2243's backend sub-buckets, so the only buffer figure in it was the *frontend* materializer's,
    while the leaf two lanes were actually optimising — `res_buffer_copy_ms`, the backend memcpy at
    `tests/render_runner.h:5199` — was invisible to it. Re-captured with the sub-buckets live and the
    collapse confirmed by flip rate (1.59/s): `setup_resources 801.9ms [buffer=673.4 (copy=539.2) …]`,
    i.e. **~67%**, not 2.4%. Two independent instruments agree it is large in the collapsed regime and
    on both platforms — `PROSPER_RENDER_TIMING` gives copy/backend **15.7%** on Linux and **18.5%** on
    Windows, and `perf` puts the same leaf at 17.6% of the saturated Linux worker.

    **The lesson is not the arithmetic, it is the propagation.** The 2.4% was corrected on #2215 the
    same day it was posted; the charter was not, so the retracted figure went on steering people from
    the document they read *first*, and the other lane reports it is what sent them chasing five dead
    hypotheses. A correction that lives only in an issue thread is not a correction to the charter —
    if a number here came from your measurement and your measurement changes, the charter is part of
    the fix, not follow-up work. **A stale figure carries the charter's authority, which is exactly the
    authority it no longer deserves.**
- **Reaching the running frame loop:** `PROSPER_GUEST_ARGS=-force-gfx-direct`, plus `PROSPER_RENDER=1`
  to run the live renderer and `PROSPER_GFXLOG=1` for graphics diagnostics.
  - **`PROSPER_GUEST_FS=1` is NOT needed on Linux or Windows, and this line used to say it was.** Guest
    initial-exec TLS is **enabled by default** there; the environment variable actually read is the
    **opt-OUT** `PROSPER_NO_GUEST_FS`, kept for compatibility bisection (`src/host/guest_tls.cpp:58`,
    `:240`; `src/hle/dispatch.hpp:218`). `PROSPER_GUEST_FS` is never read as an env var on those
    platforms — setting it is harmless but does nothing, and *believing* it is required is not: it
    turns a default-on path into one people think they are enabling, so nobody checks it when a guest
    TLS problem is the actual cause. On **macOS/Rosetta** `PROSPER_GUEST_FS` does remain the opt-in for
    trap emulation, which is where the confusion came from.
  - Recorded because of how far a wrong default travelled: this line propagated into a review's wording,
    then a briefing, then a shipped code comment, then an author-verification comment — with nobody
    opening `guest_tls.cpp` at any step (#2049). **A default stated in this file is the kind of claim
    every reader inherits without checking; verify against the code before restating one.**
- **Do not reuse snapshot acceleration for interactive or performance runs.** `PROSPER_RENDER_SCALE>1`,
  `PROSPER_RENDER_EVERY>1`, and `PROSPER_RENDER_EVERY_FOR_MS` deliberately reduce resolution or skip
  graphics submits. Keep the defaults (`PROSPER_RENDER_SCALE=1`, `PROSPER_RENDER_EVERY=1`, and no timed
  sampling window) unless the run is explicitly a headless capture or sampling experiment.
- **Correctness-first:** implement real behavior from primary evidence: live captures/traces, guest
  disassembly, published platform contracts, firmware symbol data, and focused tests. Do not ship shims
  that fake output. Mark genuinely uncertain code with `CONFIDENCE: HIGH/MED/LOW`.
- **An unsupported shader/GPU operation is a FATAL gap, not an acceptable skip — support ALL of them.**
  A recompiler op the shader stage uses, a storage/texture format a dispatch writes, an AGC/PM4 packet
  the guest submits: if the guest exercises it, the goal is to implement it, not to `return {}` / skip the
  draw / skip the dispatch and move on. Silent skips drop real rendered content (a skipped LUT/exposure
  dispatch collapses the whole title composite to black; a rejected shader drops its draws) and read as
  "handled" when they are not. Reject paths still exist as a *fail-visible* backstop for genuinely
  unknown encodings (mark `CONFIDENCE: LOW`, log loudly, file an issue with the exact opcode/format), but
  treat every one you hit on a live boot as the next thing to implement. Find the exact failing op with
  `PROSPER_DBG=1` (`[recompile-reject] pc=… op=0x…` from the recompiler) or the `[compute] skip …`
  lines from the live backend, then implement it with a round-trip/execution test — do not leave it skipped.
- **Entitlement and add-content APIs answer from LOCAL INVENTORY — never blanket-approve.** When a
  title asks whether an add-on or entitlement is owned, prosper answers from the content actually
  declared and present locally (`src/hle/hle_addcontent.cpp` parses real entitlement labels and keys
  and rejects malformed or duplicated ones). That is the right design for a preservation tool, and it
  is deliberate: a compatibility layer must keep working when the licensing service it would
  otherwise consult no longer exists, so the answer has to be derivable offline.
  **What must never happen is making these calls return "owned" unconditionally to get a title past a
  check.** The distinction is not cosmetic. Faithfully reimplementing a *platform* query — what system
  software this is, whether a user is signed in — is exactly the job, the same way Wine answers
  `GetVersionEx`. Manufacturing a positive answer to an *ownership* query is not reimplementation; it
  is the emulator performing the circumvention itself, and it would make prosper a circumvention tool
  rather than a compatibility layer no matter how the code were phrased.
  This is written down because the failure mode is attractive: a title stalls at a content check, one
  `return 1` makes it progress, and the change looks like progress in a screenshot. If a title cannot
  advance because content is genuinely absent, that is the honest result — record it as the blocker
  and, if the content exists locally, wire it through the inventory. `CONFIDENCE: HIGH`.
  - **The positive half of the same rule, and it is an obligation: every feature and every piece of
    content that IS in the dump must be reachable.** Under-reporting what is present is as much a
    defect as over-reporting it would be a circumvention — it just fails in the direction that looks
    safe, so nobody files it. A stub that answers a local-inventory question with 0 silently removes
    content the user has, and the title is then behaving *correctly* on a wrong answer, which is the
    hardest kind of bug to see: the game's own UI explains the problem in the game's own words, so it
    reads as a licensing wall rather than as our defect.
  - **The test that separates the two cases is whether the answer is derivable from bytes in the
    dump.** "What SKU is this installed application?" is a *platform* query about local content —
    `param.json`'s `contentId`, `applicationCategoryType`, `applicationDrmType` all state it offline,
    so reimplement it faithfully, exactly as `GetVersionEx` is reimplemented. "Did this person buy X?"
    is not derivable from anything local, and no amount of engineering makes it so — stop there and
    record the cap. Derive; never hardcode a full/owned answer. A dump that is a free or trial SKU
    must report that SKU, and the regression test proving you derived rather than hardcoded is a
    **mutation arm where a synthesized free-SKU declaration yields the non-full answer**. Without that
    arm the change is indistinguishable from `return <full>`.
  - **Answer the same question the same way through every library that exposes it.** Sony surfaces SKU
    and add-content state through more than one library, and a divergence between them is a defect
    even when each path looks defensible alone. Worked example (#1873): `hle_service.cpp`'s
    `sceAppContentAppParamGetInt` answered `SKU_FLAG` as full while
    `sceNpEntitlementAccessGetSkuFlag` was unregistered and fell to the return-0 stub — so the answer
    depended on which library the guest happened to ask through, and *Grand Theft Auto V* asks through
    the NP one. Put the derivation in **one** place both paths call; if two paths can still disagree
    after a fix, the defect has moved rather than gone.
- **PR verification and merging.** Keep each PR description self-contained: link the issue or goal, explain the
  failure scenario and behavioral contract, summarize the approach and important invariants, identify affected
  and deliberately unaffected behavior, record risks, and list the exact build/test/snapshot commands and
  results. PRs that advance a game's visible progression must attach representative screenshots and identify
  the title, platform, and reached checkpoint; black or diagnostic-only captures are not progression evidence.
  Use direct, unmodified frontend captures and name the frontend plus the run route in each caption; output
  produced by forced guest-state diagnostics may illustrate an investigation but is not acceptance evidence.
  Run the strongest relevant local checks and wait for every applicable required CI check. Before
  merging, synchronize with the live target branch when needed, inspect the resulting diff, run `diff --check`,
  and address every known correctness concern.
  - **Exception — a documentation-only PR may be merged immediately, without waiting for CI.** If the diff
    touches nothing but `.md` files, there is no build, no test and no snapshot that CI can tell you
    anything about, so waiting on it only slows the queue. Merge it.
    **Run the docs gate locally first** — it is the one check that can actually fail on a docs diff, and
    it takes a second:
    ```bash
    python3 prosper/tools/docs/check_numbered_table.py --sequential \
        --table-header Instrument prosper/docs/GAME_COMPAT_ORCHESTRATION.md
    git ls-files '*.md' -z | xargs -0 python3 prosper/tools/docs/check_numbered_table.py
    ```
    This matters most for `GAME_COMPAT_ORCHESTRATION.md`'s numbered tables, where the gate requires rows
    to be unique, ascending **and gapless**: several lanes append to them concurrently, so a PR that was
    contiguous when written can be gapped by the time it merges (#2087 sat conflicting for exactly this).
    Confirm the diff really is `.md`-only — `git diff --name-only origin/master...HEAD | grep -v '\.md$'`
    should be empty. The exception is about the *diff*, not the intent; one stray file makes it an
    ordinary PR again.
    **This waives the CI wait only.** It does not waive independent review where review is warranted — a
    docs PR that rewrites a `## Ruled out` row, a reproduction route, or a numbered table's semantics can
    be just as wrong as code, and no CI job has ever been able to see that. It also does not waive
    `diff --check`, the trap-41 deletion check, or the session-trailer gate. An agent may merge only when the user or task explicitly
  authorizes it. Each agent owns its PR through merge and branch cleanup: before claiming or starting new work,
  merge the current PR after all gates pass, or explicitly close it with a clear rejected/superseded explanation.
  Do not accumulate floating PRs or silently hand merge ownership to another agent. The PR author owns
  verification; prefer `powershell -File prosper/tools/verify-pr.ps1 core` or,
  for renderer changes, `powershell -File prosper/tools/verify-pr.ps1 renderer -Snapshot <name>`, and post the
  exact-head results.
- **Use independent review whenever it can materially improve confidence.** Treat review as a correctness
  tool, not merely a gate for large diffs: behavioral code changes involving non-obvious judgment, unfamiliar
  code, cross-platform behavior, reverse-engineered semantics, or meaningful failure paths should normally be
  reviewed even when the patch is small and the author verification is green. Complex or high-risk changes—such
  as shader/recompiler
  control flow, memory safety or untrusted bounds, synchronization, ABI-sensitive interfaces, persistent data,
  shared renderer/executor format, size, or indexing semantics, or changes whose failure can silently corrupt
  results—require an independent code review before merge. Review is also required when the validation itself
  is easy to get wrong: for example, a regression test with another path that can produce the expected result,
  or a snapshot route or baseline update that needs judgment to confirm it exercises the intended behavior.
  Judge both implementation risk and verification risk, not merely the diff size. Skip independent review only
  for genuinely routine, mechanical, well-bounded low-risk work where another reader is unlikely to uncover a
  correctness problem; when in doubt, review. Passing author verification does not substitute for
  review where review is warranted: verification demonstrates observed behavior, while review challenges the
  contract assumptions, untested paths, and whether the regression actually proves the intended behavior.
  When uncertain, favor review for reverse-engineered API semantics and other changes where an independent
  reader can materially challenge the reasoning, even when the patch is small. The reviewer inspects the
  assumptions, code, risks, and tests, including whether each regression test would fail without the fix, but
  does not duplicate the author's builds, tests, snapshots, or CI jobs. Once the author is satisfied and the PR
  is published, run author verification and any required review; the author posts exact-head evidence and the
  reviewer posts approval on the corrected exact head. Address every blocking finding and re-review authored
  corrections, then merge only after the merge agent separately confirms author verification, reviewer approval
  where required, and green required CI.
  - **A cited claim reads as a verified one — and reviewers' findings propagate hardest.** A review
    finding carries more authority than the same sentence anywhere else: it has been *checked* by
    definition, and attaching a `file:line` makes it look checked twice. So a **wrong** finding with a
    citation is the most contagious kind of error this project produces. Worked example (#2049 → #2052):
    a reviewer stated an inverted default with a file:line, and that citation is what made it credible
    enough to travel unchallenged into a briefing, then a shipped code comment, then an author's own
    verification comment, then roughly fifteen agent briefings — with **nobody opening the cited file at
    any step**. It was `guest_tls.cpp:46`, which is the Apple-only branch; Linux is `:58`, opt-**out**,
    default **enabled**.
    **So: check a claim against the source, never against anything downstream of the thing being
    checked.** Opening the cited file and running one `grep -rn 'getenv("…")' prosper/src` is the whole
    defence, and it is two commands. This binds on reviewers most of all — when you cite a line, you are
    asserting you read it.
    - **The asymmetry that makes this worse: a finding you WANT to be true gets checked least.** A
      correction that *lowers* a claim is scrutinised, because it costs the author something. A
      correction that *raises* one is waved through, because it agrees with what the author already
      believed — and it lands in a `## Ruled out` row wearing a citation, where nobody is motivated to
      re-derive it. Worked example (2026-08-06): an author marked a VOP3P `NEG` semantic
      `CONFIDENCE: MED`, a reviewer volunteered "published support you hadn't cited", and the author
      raised it to `HIGH`. A second reviewer traced the citation: the reference says the modifier is
      *"valid for floating-point operands only"* — a **restriction**, cutting against the inference —
      and the LLVM fold came from a PR merged and reverted the same day, covering a different
      instruction family. The honest label was the original one; the author had been talked *up* from it.
    - **The multiplier is the citation itself.** A bare "I think you're being too conservative" would
      have drawn scrutiny. A published field description plus a compiler fold read as *already checked* —
      the same authority-transfer described above, now pointed at the thing the author hoped was true.
      **So trace a citation in proportion to how much you want it to be true, and treat "this supports
      raising confidence" as a trigger for verification rather than a reason to skip it.**
    - The general shape all of these share is worth naming, because it is not "someone was wrong": it is
      **a true statement reached by a route that does not establish it.** The claim survives casual
      checking precisely because it is defensible; what fails is the derivation, and nobody re-runs a
      derivation whose conclusion they already accept. Three instances landed in one day — this citation,
      a `v_med3_f16` lowering correct only for finite inputs, and a pair of test arms asserting a property
      their inputs could not exercise (#2130).
  - **How a verdict is expressed and detected — this is mechanical, and getting it wrong has merged a rejected
    PR.** The reviewer posts one or more **registered reviews** with `gh pr review <N> --comment --body '…'`,
    each stating a verdict as a literal **`APPROVED`** or **`REJECTED`** in the body. The author reads the
    comments, fixes what is raised, and merges only once a review says `APPROVED` **on the current head**.
    Two mechanical facts make this the only workable scheme here, and both are invisible until they bite:
    - **`gh pr comment` does NOT register as a review.** It creates an *issue* comment, fully visible on the PR
      page and absent from `gh pr view --json reviews`. A verdict posted that way is invisible to anyone
      checking programmatically. Several reviews were posted this way before it was noticed.
    - **GitHub refuses `--approve` on a PR opened by the same account**, and every agent here operates as the
      repository owner. So the review *state* is **always `COMMENTED`**, for approvals and rejections alike, and
      `reviewDecision` stays empty. **The GitHub review state carries no verdict in this repository. Never gate
      on it, and never treat "a review exists" as "a review approved".** A merge daemon that counted
      `reviews > 0` merged #2026 with two blocking findings outstanding; the revert is #2044. The verdict lives
      in the prose, so **a human or an agent must read it** — this specific check cannot be automated here.
    - Corollary for the merge step: **a rebase or a new push detaches every review from head.** `reviews` still
      reports rows, with a `commit_id` that is no longer on the branch. Compare each review's `commit_id`
      against the current head; if it is stale, establish by **content** (`git diff <reviewed-sha> <head>`)
      whether the approval still applies rather than assuming in either direction.
- **Unpublished desktop-app parity is not a merge requirement.** A Linux-, Windows-, or macOS-specific app
  improvement may merge without matching work in every other frontend unless the issue explicitly promises
  parity or a shared public contract requires it. Unaffected platforms must still retain existing behavior and
  pass their applicable checks; track desirable parity separately rather than blocking the focused change.
- **Evidence hierarchy and independent implementation.** Trust sources in this order: (1) prosper's
  live captures/traces of the real guest, (2) published platform contracts, firmware symbol data, and
  the guest's own disassembly, (3) agreement among independently written secondary implementations,
  then (4) a single secondary implementation as a hypothesis only. PS5-specific AGC, Gen5 descriptors,
  and PS5-only kernel calls require direct title evidence; inherited PS4 behavior still must be checked
  against the exercised guest path. External implementations are verification-only: do not copy or port
  their code, types, comments, prose, or tests. Re-derive behavior in prosper's own architecture and add
  project-owned evidence/tests. Never weaken behavior demonstrated by a live boot to match a secondary
  reference, and mark unresolved evidence with `CONFIDENCE: HIGH/MED/LOW`.
  - **Exception — vendoring permissively-licensed standalone libraries.** The "re-derive, don't copy"
    rule targets other *emulators'* reimplementations of Sony HLE interfaces (clean-room integrity). It
    does **not** forbid vendoring a general-purpose, permissively-licensed (MIT/BSD/zlib), self-contained
    library for a well-defined standalone problem — a codec, hash, math kernel — when that library is
    itself clean-room (no Sony code/keys) and re-deriving it would be disproportionate. Such a dependency
    is a deliberate project-owner decision, vendored **verbatim** under `third_party/<name>/` with its
    LICENSE and a README recording origin/version/why; prosper's own glue (parsing, wiring into the guest
    path) stays project code. First instance: `third_party/libatrac9/` (MIT, ATRAC9 decode — a large
    codec, not a Sony interface). Do not use this exception to shortcut Sony ABI/HLE work.
- **AMD RDNA 2 shader ISA reference:** ["RDNA 2" Instruction Set Architecture: Reference Guide
  (document 70648)](https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content). Consult it
  for instruction encodings, operand and condition-code semantics, wave behavior, and memory-instruction
  details when working on the shader recompiler or GPU diagnostics. Treat it as the primary published RDNA 2
  architecture reference; PS5-specific extensions, encodings, and AGC behavior still require prosper's live
  title evidence and focused tests.
- **PS5 3.20 firmware library reference — the definitive NID↔name database (`../PS5-3.20_Libs/`).**
  A `genstub.py`-generated dump of **all 275 PS5 3.20 system libraries**, one `libSceXxx.c` per library.
  Each file lists **every exported function AND its exact NID**: the loader lines read
  `sprx_dlsym(__handle, "<NID>", &__ptr_<funcName>)`, so each is a `<NID> ↔ <funcName>` pair. This is
  the **authoritative PS5-specific symbol map** — trust it over PS4-era symbol lists and secondary tables for
  any Gen5/PS5-only surface (AGC, Ampr, Pad/UserService Gen5, etc.), and use it to see a library's
  *complete* real API surface (e.g. what functions exist that a title might call). Recipes:
  - Resolve an unknown NID → name: `grep -rn '<NID>' ../PS5-3.20_Libs/` (the matching `__ptr_<name>` names it).
  - A library's full export list: `grep -oE '\.global sce[A-Za-z0-9]+' ../PS5-3.20_Libs/libSceXxx.c | sort -u`.
  - Which library exports a symbol: `grep -rl '<funcName>' ../PS5-3.20_Libs/`.
  Caveat: it gives **names + NIDs only, no bodies** — argument layouts and behavior still come from
  prosper's live captures, published contracts, and guest disassembly. Gitignored sibling of the repo;
  never commit its contents.
- **Commit style:** small, verified commits; push to `origin` promptly. Co-author trailer as configured.
  **Do NOT add "Generated with Claude Code" attribution lines** (the 🤖 badge, "Generated with
  [Claude Code](...)" footers, session links) to PR bodies, commit messages, issue text, or
  comments — anywhere. They are noise in the project record; the co-author trailer alone is enough.
  - **This one needs an explicit check, because the harness fights it.** Several agents run here with a
    default that appends a `Claude-Session: https://claude.ai/code/session_…` trailer to every commit
    message; this rule overrides that default, but the default is silent and re-applies on every commit,
    so following the rule by intention alone does not work. Measured per-commit on 2026-08-01:
    **29 of the last 100 commits on `master`** carry one.
    ```bash
    # Gate before pushing — answers "any?", expect 0. `grep -c` exits 1 on zero matches, so never
    # put it in an `&&` chain.
    git log origin/master..HEAD --format=%B | grep -c '^Claude-Session:'

    # How many COMMITS carry one — a different question, and the one to quote as a statistic.
    git log origin/master -100 --format=%H | while read h; do
        git log -1 --format=%B "$h" | grep -q '^Claude-Session:' && echo "$h"; done | wc -l
    ```
    **The two commands disagree, systematically and always in the same direction.** Over the same
    100-commit window the first returns **63** and the second **29**. `%B` is the *body*, and a
    **squash merge concatenates the bodies of every commit it absorbs**, so one squashed commit
    routinely carries several trailers — `grep -c` over `%B` counts trailers, not commits, and
    over-reports on any squash-merged history. The distribution proves the mechanism rather than
    merely fitting it: of the 29, sixteen carry one trailer, four carry two, five carry three, and
    one each carry four, five, seven and eight — which sums to exactly the 63 lines. The first form
    is still exactly right as a zero-gate; it is only wrong when read as a commit count. The gap
    widens on shorter windows, where a single large squash dominates: over the last **20** commits
    the line form reads **25** against a true **11**. All figures measured 2026-08-01.
    Keep `Co-Authored-By:`. **If the count is non-zero, amending is not enough** — the trailer is on
    commits you already made and the harness re-adds it to the fixup. `git filter-branch` is not
    available here. Two recipes that do work: `git rebase <base> --exec '<amend with a cleaned
    message>'` (the exec runs outside the harness), or rebuild the branch —
    `git checkout -b <slug>-clean <base>`, then per commit `git cherry-pick -n <sha>` followed by
    `git commit -F <message-with-the-line-stripped>`. Verify with `git diff <old-head> HEAD` over the
    paths you touched (expect empty), then `git push --force-with-lease`.
- **Rebuilding a branch onto a moved master: never take a whole file from the old branch.** The obvious
  move — `git checkout <old-branch> -- <file>` — reverts that file to its old state wherever ANOTHER
  lane has since edited it, with no conflict, no failing check, and a diff that reads as your own edit.
  It cost #1701 ten lines of documentation this way, caught only because `--stat` showed deletions in a
  file believed touched once. Re-apply your own hunks onto master's version instead; when you do take a
  file whole, **diff it against `master` and read the `-` lines**. The `Docs` CI gate does not cover
  this: it validates table structure and numbering, never prose. See instrument-trap 41.
- **A pipeline's exit status is its LAST stage's.** `cmd | tail`, `cmd | head` and `cmd | grep` all
  discard `cmd`'s failure, so `build && test | tail -3 && commit` commits through a red test. Capture
  the status (`cmd > log; rc=$?`), use `set -o pipefail`, or read `${PIPESTATUS[0]}`. Separately,
  `grep -c` **exits 1 on zero matches**, so a legitimate "none found" aborts an `&&` chain — separate it
  with `;`. Four people lost time to these in one session. See instrument-trap 40.
- **Never publish the developer machine's local paths.** This repository is public. An absolute path
  leaks the account name and the private directory layout of someone's computer, and it is never the
  information a reader needs — the *shape* of the command is. So keep absolute host paths out of commit
  messages, PR titles/bodies/comments, issue text and review comments, and out of committed files
  (docs, scripts, tests, code comments). Use placeholders instead, or a path relative to the repository:
  ```text
  <REPO_ROOT>    the checkout root          <WORKTREE>   a worktree root
  <DUMP_ROOT>    where the game dumps live  ~/           the developer's home
  ```
  So write `-DGAME_DUMP=<DUMP_ROOT>/PPSA24651-app0`, not the real path, and
  `cd <WORKTREE> && cmake --build build -j6`. Paths *inside* the repo (`prosper/src/gpu/…`) and generic
  system paths (`/usr/lib64/libc.so.6`, `/tmp`, `/var/tmp`) are fine — the rule is about the private
  part above the repository root. This is a privacy matter, not a security one: nothing here is secret,
  and it still should not be published. Sanitize before you post; if something already published slipped
  through, edit it (`gh api -X PATCH repos/OWNER/REPO/issues/comments/<id> -f body=…` edits a comment).
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
  - **Long-lived game trackers are progress indices, not bug reports.** Create one as
    `[Game tracker] <Title> (<TITLE_ID>)`, label it `tracker:game` plus the title's existing
    `game:<slug>` label, and keep it open through the compatibility ladder. Its compact body records
    the current rung and latest verified master where known, the best checked-in screenshot, what
    works, links to ordinary blocker issues, the reproduction/input/snapshot route, technical status
    docs, and the six-rung checklist. Ordinary bugs remain separate and close independently. PRs
    update a game tracker with `Refs #NN`, never `Fixes #NN`; do not give the tracker `bug` or
    `in-progress` unless work on the tracker issue itself is actually claimed.
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
  - **Labels are organizational, not access gates.** Two prefixes classify every issue:
    - **`game:<title>`** names a *game* whose specific boot path an issue exercises — `game:messenger`
      (PPSA24651 / The Messenger, IL2CPP/Unity), `game:blasphemous2` (PPSA13579), `game:ue4`
      (PPSA17942 / Unreal Engine), etc.
    - **`area:<subsystem>`** names *shared code used across titles*. `area:infra` is the shared
      substrate — loader, libc, libkernel, the recompiler, generic GPU/AGC, host mapping/ABI. Shared
      code is never bucketed under a game's label (it belongs to every title, so a single `game:` tag
      would be a category error).

    Pick `game:<title>` when the fix is specific to one game; pick `area:infra` (or another `area:`
    subsystem label) when it lives in the shared substrate. A single issue may carry both — e.g. an
    `area:infra` loader bug first observed in `game:blasphemous2`. The labels describe *what* an issue
    touches; they do **not** fence off who may work where. Any file or issue is fair game — when
    several agents run concurrently, coordinate through the claim lock and worktrees above rather than
    by carving up the codebase.
