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

Key docs to orient: `prosper/README.md` (status), `prosper/docs/ROADMAP.md`, `prosper/docs/GRAPHICS.md`,
`prosper/docs/RENDER_LOOP.md` (the historical render bring-up log), and
`prosper/docs/MESSENGER_BLACK_RENDER.md` (the revisioned Messenger investigation status). The current
Dead Cells graphics status and exact regression recipe are in `prosper/docs/DEAD_CELLS_STATUS.md`.

## Historical frontier (superseded 2026-07-11)

The game **boots through IL2CPP into Unity's frame loop and submits real GPU draws.** A live Vulkan
renderer is wired in; the game's **real pixel shader recompiles to valid SPIR-V** and its real descriptors
+ 1920×1080 sampled texture decode correctly. The **one** remaining step to the first rendered frame is
**bindless-dynamic vertex-fetch resolution** for the vertex shader — fully specified in
`prosper/docs/NEXT_STEP_VERTEX_FETCH.md`. This paragraph is historical only.

## Current frontier (2026-07-14)

The game now **boots through IL2CPP, renders its intro/title/menu, and reaches gameplay with real GPU
draws.** The old bindless vertex-fetch frontier is complete: both shader stages recompile and dynamic
V#/T#/S# resources resolve on current master. Do **not** start from `NEXT_STEP_VERTEX_FETCH.md`; it is
retained as a historical bring-up record.

The native Windows/MinGW substrate now runs the same SDL3/Vulkan frontend and normal PNG screenshot
workflow through a routed Messenger fresh save and a fully lit first-level frame (#683). The route
exposed a stale Windows `guest_readable` stub in dynamic-fetch folding; a `VirtualQuery` range guard
fixes the resulting loading-time host access violation (#688). Start Windows work from
`docs/WINDOWS_PORT_HANDOFF.md`, not the historical pre-render fence investigation.

The July native-renderer performance pass and its stop decision are documented in
`docs/RENDERER_PERFORMANCE_2026_07.md`. Exact-byte texture decode reuse, per-submit readable-range
reuse, output-copy removal, and corrected mixed-operation capture improve Messenger's first level
from roughly 12 to 24 FPS. The remaining synchronous graphics/compute boundaries must be evaluated
against a 3D workload; do not resume Messenger-specific cache work toward 60 FPS first. Windows
release users start from `docs/WINDOWS_RELEASE.md`; build/debug work stays in the port handoff.

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

*Blasphemous 2* now passes FMOD initialization (#638/#640), renders its logos/title/EULA after the corrected
AGC marker implementation (#641), and passes the post-EULA telemetry parser after `sceHttpUriParse` gained its
two-pass caller-pool contract (#642). Normal-return guest pthreads also leave host `%fs` active before glibc
thread cleanup (#644), so the route loads gameplay scenes/assets without the old host crash. One-second
poll-safe presses plus observed-state logging (#646) make the long opening reproducible under slow software
rendering, and a native 1920x1080 capture now confirms the complete first playable room: player, HUD,
foreground/background art, lighting, and interaction prompt. Use `scripts/blasphemous2/README.md` for the
validated sampled-render screenshot recipe. Screenshot manifests report source and pixel progression
separately (#648), because a newly published renderer frame can still contain byte-identical pixels. The
remaining black-world composition was stale opaque alpha: PS5 primitive type 7 is a RectList clear, but the
standard RDNA2 table labels 7 reserved and prosper fell back to PointList. The captured vertex shader maps
indices 0..3 to all four clip-space corners while the guest submits count 3. #654 maps the PS5 topology to a
Vulkan strip and invokes the fourth procedural vertex for the observed no-VB form. A 420-frame native run
then showed full-screen moving gameplay with no accumulated tutorial glyphs.

Cross-title breadth has advanced: *Dead Cells* now starts reliably after both AGC resource-registration output
queries were implemented (#544, #660). The old success stubs left stack data in the max-name and required-memory
outputs, causing intermittent multi-gigabyte stack or texture-pool allocations. Its
exercised NGS2 lifecycle returns initialized sizes/handles/state and silent output (#554), and a late render
window reaches the Evil Empire splash. #545 was software-render throughput, not a guest deadlock: synchronous
3840×2160 llvmpipe rendering stretches its ~13,000-submit startup into minutes.

Dead Cells now has a deterministic route through splash/menu into full-color gameplay.
Version-4 `.prgcap` captures seeded temporal RTT inputs (#568) historically isolated the earlier warmup artifact
at draw 18; one 642x362 input had no prior color-target writer. The kernel-derived dispatch
thread/local/group contract (#580), `sceAgcCbSetShRegistersDirect`, and compute direct type-1 V# binding (#574)
now execute the real fill kernel against guest buffers before submit completion (#576). Range provenance proved
draw 19 consumes one backing, dispatch 5 fills it, then draw 31 consumes it again in one submit. Graphics spans and
compute now execute by retained PM4 order (#584), fixing that future-read. The later overbright screenshot was a
warmup artifact: the 35-second render delay skipped a 642x362 RTT producer, then a replace-copy sampled dispatch
4's raw all-`0xFF` backing and cached it indefinitely. `PROSPER_RENDER_TARGET_DIM=642x362` preserves the real
opening vignette/level geometry; #586 established a practical late checkpoint with that history intact. The
residual seeded replay mismatch (#569) exposed persistent depth/stencil state outside color RTT seeds. The
faithful closure reuses one 642x362 depth identity, but timeline-v5 backing provenance proves compute fills its
HTILE allocation before drawing; #611 now invalidates the detached Vulkan DS cache on that guest GPU write.
The Dead Cells splash regression guard is now a run-level content check: after a three-second renderer warmup,
the richest frame in a 20-second window must contain at least 1,500 distinct colors. This replaces the historical
animation-sensitive exact-frame contract from #573/#596.

The current tooling frontier is deterministic offline capture rather than longer live-render windows.
Native-speed `.prgtl` indexes retain every submit/present boundary, and an exact-submit selector can materialize
immutable, content-deduplicated graphics/compute state plus mixed PM4 order into a version-8 `.prgcap` (#594/#569).
`gpu_replay --graph` / `--graph-json` now resolve in-submit versions and temporal read-before-write leaves (#600;
full workflow: `prosper/tools/gpu_replay/README.md`). Timeline version 6 retains the version-5 bounded same-run
target/depth history and adds compact per-draw target spans for offline scene selection. Ordered `.prgbundle`
windows now capture producer-time submits with content-defined
cross-submit deduplication and replay them through one persistent renderer (#603). Dead Cells depth 16 resolves
all 30 internal temporal edges in submits 18735..18750 while storing 2.883 GiB logical data in 166.3 MiB, but
the earliest submit still has two unseeded 642x362 leaves. Full-run aggregation places their first observed
graphics writers around submit 17,400 and records roughly 1,200-1,350 writes before the selected submit (#604).
A transparent-zero boundary A/B yields the exact unseeded hash, so zero initialization is not the missing state.
Bundle v2 (#606) now uses fault-safe bulk guest reads plus an exact shared-resource chunk dictionary: a fixed
1,200-submit full-state run folded 122.97 GiB into 301.1 MiB in 169.4 seconds. Semantic endpoints, rolling
windows, successful-only exit, final compaction, and `gpu_replay --bundle-tail` prevent timing drift and replay
holes. A compact two-submit closure resolves both 642x362 edges with no bounded leaves, but its 80-draw endpoint
is the opening vignette rather than gameplay. The preserved #608 playable bundle used exactly 90 semantic draws
and the 738x420 target at draw 79..81. Current routes use 91..94 draws, exactly 8 dispatches, and the 636x420
target at draw 77..85; timeline selection isolates sustained gameplay without depending on run-local submits.
Target extent or total draw count alone also selects cinematic/transition frames and must not be used as the
oracle. This checkpoint established the stable offline baseline used to isolate the composition defects.
The faithful playable bundle spans submits 18,165..19,047, stores 158.94 GiB logical state in 739 MiB, and
resolves all 1,764 temporal image dependencies. Its pre-#611/#615 historical hash was `5759c125812154dc`; do
not use that old absolute hash as a current renderer oracle. A two-submit color-bounded replay remains
`71b84bdfae53933c`; `gpu_replay --bundle-find-ds ADDR` scans manifest-only DS use in seconds and proves
the shared surface has no draw/register clear intent. The missing hardware boundary is now identified and
implemented (#611): timeline-v5 retains complete raw DS programming plus optional guest backing hashes/writer
provenance, which found compute program `0x401aec200` filling the exact 32 KiB HTILE allocation with
`0xfffffff0` before scene drawing. Guest GPU writes notify the live backend, and overlapping persistent Vulkan
DS entries become invalid so the next use follows the existing compare-derived clear path. A routed live A/B
restores the foreground/platform/HUD layers that remain black with invalidation disabled. The explicit
`--legacy-htile-before-stencil` switch supplies the omitted HTILE identity only for the preserved pre-v6 bundle.
Capture v8 closes the exact offline boundary (#569). Current #611-enabled full-bundle and standalone output are
both `fac9ca4cbbba8196`. An invalidation-disabled stale-depth A/B is `535256588b67a536` in both paths with
byte-identical BMPs; the self-contained capsule carries 12 RTT surfaces, one 929,616-byte 642x362 D32S8 depth
plane, effective DS lifetime/legacy settings, and the 33,177,600-byte source oracle. Standalone replay takes
about 3.3 seconds instead of roughly 24 minutes.
The four repeated Dead Cells fragment failures were canonical VCCZ-exit light-accumulation loops. #615 adds a
stage-specific proof: every VOPC input must be scalar/inline/literal or have a nearest overlapping VGPR
definition from an unmodified uniform VOP1 move/conversion. Only then can the wave-empty VCC test lower to a
per-invocation structured loop; varying bounds and compute remain fail-visible. `shader_inspect` decodes raw
`PROSPER_SHADER_DUMP` files with exact dword PCs and branch targets. Current live gameplay samples realize all
semantic draws. Capture v7 (#618) retains a failed stage fault-safely through `s_endpgm` or a 64 KiB cap,
content-deduplicates it, and records exact coverage/opcode/PC, decoded pipeline/launch state, and
resource/descriptor summaries. Start with `gpu_replay --inspect-only`; extract a raw stage with
`--dump-failed-shader FAILURE:STAGE PATH` instead of rerunning the title.

The remaining Dead Cells color defect was fixed by #626. Its world shaders emit color exports in descending
MRT3..MRT0 order; the single-attachment recompiler incorrectly used the first export, presenting a grayscale
G-buffer plane as color0. Selecting MRT0 restores the full-color Prisoners' Quarters composition. The same PR
also exposes the directly placed destination V# for a format-copy compute shader, taking the current checkpoint
from seven to eight realized dispatches. #566 is closed. Use `prosper/docs/DEAD_CELLS_STATUS.md` for the current
route and regression workflow rather than restarting the completed composition localization.

The later giant translucent gameplay surface was native-format loss (#773). Draw 23 sampled a 642x362
`Float16x4` lighting target that the backend had rendered and reuploaded as RGBA8, clamping its HDR data.
Renderer-owned targets now preserve `VK_FORMAT_R16G16B16A16_SFLOAT` through attachments, readback/seed bytes,
sampled images, and persistent target/texture/pipeline cache identities. Capture v13 tags RTT seeds as `rgba8`
or `rgba16f` and reads v1..v12 artifacts. The current semantic selector is documented in
`prosper/docs/DEAD_CELLS_STATUS.md`; do not reuse the historical 738x420 predicate for new captures.

The first FP16 live run then exposed stale temporal history (#780), not another shader defect. Compute operation
19 resets the same lighting backing to RGBA16F `(0,0,0,1)` before draws 17..22, but guest-write notification
invalidated only the persistent Vulkan target; the frontend immediately uploaded the previous frame's CPU RTT
copy as a seed and brightness fed back toward white/yellow. Guest GPU writes now discard overlapping CPU RTT
entries using their native byte width as well as invalidating GPU targets. Live user validation reports stable,
artifact-free composition apart from separately tracked window-light banding (#781). The corrected 77-submit
source and standalone capsule are byte-identical at `13b4ccdfa15b1f4d`.
The #781 investigation localizes that residual pattern to the additive window-light visibility pass and rejects
depth, history-alpha, simple sampler-filter, and perspective-interpolation explanations. It remains open and
deprioritized; see `docs/DEAD_CELLS_STATUS.md` before repeating live experiments. Successful raw RDNA2/SPIR-V
pairs can now be captured with `PROSPER_SHADER_DUMP_SUCCESS=DIR` for offline inspection.

`--bundle-final-capsule` snapshots both color RTT state and exact valid planes from persistent Vulkan
depth/stencil images into capture v8 (#569). Capture v8 reads v1-v7 artifacts; pre-v7 failed-operation diagnostics
report unavailable and pre-v8 captures contain no invented DS seeds. Timeline v6 reads timeline v1-v5
artifacts. Addresses and operation ordinals are run-local. Use
`gpu_timeline FILE --signatures DRAWS DISPATCHES` to discover target spans and `--select` to validate the exact
live-capture predicate before recording another detailed bundle. The Dead Cells splash guard deliberately
uses `min_colors=1500` rather than an exact hash: unchanged builds select multiple valid animation states with
1,650-1,698 distinct colors, while observed partial transitions contain only about 325-339.
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
  cmake --build . -j8 && ctest        # 99/99 expected green on Linux
  ```
- **Verification is agentic-first / programmatic** (`docs/VERIFICATION.md`): ctest exit code is truth;
  shaders are `spirv-val`-gated; rendered frames are asserted by pixels, hashes, or routed content
  metrics. **After any change that can affect rendered output** (recompiler, AGC decode, render state,
  detile, executor/present), run `python3 tools/snapshot/snapshot.py check` (local-only). Gameplay
  guards inspect multiple frames in route-specific windows and use SSIM over compact luminance
  signatures from several reviewed states, looking for major collapse without rejecting subtle pixel improvements.
  Every new or materially changed baseline requires `snapshot.py verify`,
  inspection of all retained images from both runs, and a factual `review` note. See
  `tools/snapshot/AGENTS.md`.
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
- **Correctness-first:** implement real behavior from primary evidence: live captures/traces, guest
  disassembly, published platform contracts, firmware symbol data, and focused tests. Do not ship shims
  that fake output. Mark genuinely uncertain code with `CONFIDENCE: HIGH/MED/LOW`.
- **Independent review is a mandatory merge gate for every PR containing agent work.** This applies whenever
  the current PR head contains changes designed, debugged, authored, co-authored, implemented, or materially
  modified by an agent, regardless of who opened the branch or PR. An agent taking over an existing PR must
  arrange the same review before merging. The implementing agent must never merge immediately after coding,
  even when its tests pass. Once the implementation and author-side verification are ready, push the branch,
  open or update the PR, and spawn a fresh, independent code-review sub-agent. Correctness is priority 1;
  style and convenience never outweigh behavioral correctness.
  - Before requesting review, make the PR description self-contained: link the issue/goal; explain the failure
    scenario and violated contract; describe the approach and important invariants; identify affected and
    deliberately unaffected behavior; list risks, uncertainty, and compatibility concerns; and give exact
    author-side build/test/snapshot commands and results. A reviewer should not need private chat context.
  - Reviewer independence covers participation in producing the entire implementation under review: the
    reviewer must not have privately designed/debugged the solution with the author, authored or edited the
    patch, committed any part of it, or otherwise materially co-produced it. Spawn it with fresh context,
    without the author's private implementation/debugging transcript; give it only the PR number and the neutral
    brief below. The reviewer uses its own worktree and must not modify the PR branch or implement fixes. Normal
    review work—identifying a failure, stating the required behavioral contract, suggesting possible remediation,
    requesting regression coverage, and evaluating the author's fix or rebuttal—does not itself make the
    reviewer a contributor. The author remains responsible for designing and implementing the patch. If the
    reviewer supplies or edits the implementation or materially co-designs the final patch, it is disqualified
    from approving that head and a different independent reviewer must perform the complete review. The author
    may identify known risks and required checks on the PR, but not an expected conclusion.
  - Direct author/reviewer communication is limited to coordination such as `review requested` and
    `review posted`. Put all substantive context, questions, findings, replies, verification evidence, finding
    dispositions, and verdicts on the PR as comments so the complete decision trail is durable and traceable.
    Sign reviewer comments with one stable reviewer name.
  - The reviewer resolves the live base repository and full target ref from the PR, fetches that target ref
    directly, and records its identity and full SHA. Do not trust cached PR `base.sha`/`baseRefOid` metadata or a
    synthetic PR merge ref as the authoritative target tip. Resolve and fetch the head repository/ref directly
    too. Then read this file, every applicable `AGENTS.md`, the PR description and linked issue, the relevant
    architecture/status docs, the complete base-to-head diff, all PR discussion, and the affected callers and
    data paths. Inspect the prospective integration with that exact base and record its resulting tree SHA rather
    than reviewing changed lines in isolation.
  - The review must be adversarial and evidence-based. Check assumptions, invariants, ABI/API contracts,
    ownership and lifetimes, concurrency and ordering, bounds/overflow, error and cleanup paths, malformed
    inputs, platform differences, compatibility with old captures/state, and whether diagnostics themselves
    are race-safe and truthful. Inspect tests for meaningful failure coverage; passing tests are evidence, not
    proof. For renderer/recompiler/detile/present changes, the mandatory snapshot gate remains part of review.
  - The reviewer runs the strongest relevant focused builds/tests and required integration or snapshot checks.
    Every applicable required check must be present, completed, and successful before merge. Only an explicit,
    task-specific maintainer instruction may waive waiting for named CI checks (for example, on a docs-only
    change), and the authorization and exact skipped checks must be recorded on the PR. Such an exception never
    waives mandatory local, integration, or snapshot verification. If any other required check cannot run, the
    reviewer posts the exact blocker and does not approve. It posts every finding on the PR with severity, exact
    location, a concrete failing scenario, the violated contract, and the expected fix or regression test. New
    regressions in the PR are blockers; pre-existing or out-of-scope bugs follow the issue-tracking rules below.
  - The author addresses every finding on the PR by either pushing a fix with appropriate regression coverage
    or posting a concrete, evidence-based rebuttal. Correctness outranks reviewer authority: never implement a
    harmful request merely to obtain approval. The reviewer must explicitly accept the fix or withdraw/accept
    the rebuttal; otherwise the finding remains open and approval is withheld. The same reviewer then re-reads
    the complete updated diff and re-runs affected verification. If unavailable, a replacement independent
    reviewer must read the whole review record and perform the same full review.
  - Continue until the reviewer explicitly posts either
    `APPROVED FOR MERGE <head-repo>:<head-ref>@<head-SHA> into <base-repo>:<base-ref>@<base-SHA> with tree <tree-SHA>`
    or `NOT APPROVED`, with remaining concerns. Approval covers the PR's reviewed authored change. Any later
    authored change to code, tests, workflows, fixtures, assets, generated files, configuration, docs, or
    metadata invalidates approval and requires review of the updated change. Re-review is proportional to the
    delta; do not repeat unrelated inspection or verification when the impact is clearly bounded.
  - **Synchronizing an approved PR with its target branch does not invalidate approval and does not require an
    external reviewer.** This includes rebasing onto the target, merging the target into the PR, or integrating
    a target tip that moved after approval. The PR author owns this synchronization. Fetch both refs directly,
    resolve conflicts, inspect the target delta and the refreshed PR diff, recompute the prospective integration
    tree, run `diff --check`, and run any tests that the synchronization or conflict resolution can affect.
    Record a `BASE SYNC VALIDATION` PR comment with the old/new base and head SHAs, changed/conflicting paths,
    resulting tree SHA, and checks run. Existing review and CI evidence may be reused; do not rerun unrelated
    suites or snapshots merely because the target advanced. If synchronization reveals a real bug, fix it and
    obtain review for that authored fix, but the synchronization itself never needs reviewer re-approval.
  - The final merge must atomically preserve every reviewed input. Prefer a repository merge queue/ruleset that
    verifies the reviewed authored change and the author-validated current target/integration. Otherwise, only a
    same-repository PR may use the direct-push fallback below; cross-repository PRs stop for a capable server-side
    mechanism or maintainer
    direction. Once approval, any required base-sync validation, and separate merge authorization exist, post the
    complete transaction tuple as a `MERGE TRANSACTION` PR comment. That freezes the validated target for this
    transaction: agents must not push/retarget it, a later PR UI retarget cannot redirect the hard-coded
    destination, and only an explicit
    maintainer cancellation revokes the authorized transaction. Abort if a head/target change is observed before
    the push; repeat author-side base-sync validation for target synchronization, or obtain review for new
    authored changes, as applicable.
  - For the same-repository fallback, create a merge commit whose first parent is the validated transaction base
    SHA, second parent is the current PR head SHA, and tree is the validated integration tree; verify all three
    locally. In one server-side atomic push, advance both the exact target ref and exact PR head ref to that merge
    commit with explicit expected-old-OID leases for the validated base and current head, for example:
    `git push --atomic --force-with-lease=<base-ref>:<base-SHA> --force-with-lease=<head-ref>:<head-SHA> origin`
    `<merge-commit>:<base-ref> <merge-commit>:<head-ref>`. The merge object fixes the reviewed content and target;
    the two leases make concurrent base or head movement reject the entire transaction. The head's atomic advance
    to the merge commit is part of the merge and does not invalidate approval. If either lease fails or
    the server cannot perform the atomic push, never override/retry it or fall back to a plain merge: fetch the
    live state and repeat base-sync validation or obtain review for new authored changes, as applicable. A normal
    `gh pr merge` head-only guard is not
    sufficient for this multi-agent repository.
  - Never merge with an unresolved correctness finding; an applicable required check that is absent, pending,
    or unsuccessful; an unrecorded/overbroad verification exception; unreviewed authored PR content; an
    author-unvalidated live base/integration tree; or a merge mechanism that cannot atomically preserve the
    transaction.
    Reviewer approval satisfies the quality gate but is not permission to merge: the agent may merge only when
    the user/task separately authorizes it.

  Use this minimum review brief when spawning the sub-agent:
  ```text
  Independently review PR #<N> under the mandatory review policy in CLAUDE.md. Begin with fresh context; do not
  modify the PR branch or implement fixes. Treat the PR description and comments as the complete coordination
  record. Resolve the live head and base repository/refs, fetch both directly, and record their full SHAs; do not
  rely on cached PR base metadata or a synthetic merge ref. Inspect the full diff, affected callers, invariants,
  and prospective integration, and record its tree SHA. Run the strongest applicable verification, including
  mandatory snapshot checks. Post every finding, question, test result, disposition, and final verdict on the
  PR, signed with a stable reviewer name; include severity, exact location, concrete failure scenario, required
  fix, and evidence. Do not approve while any correctness concern or required check remains. After any authored
  update other than target-branch synchronization, review the complete new change. Post either APPROVED FOR MERGE
  <head-repo>:<head-ref>@<head-SHA> into
  <base-repo>:<base-ref>@<base-SHA> with tree <tree-SHA> or NOT APPROVED. Directly message the author only to say
  that the posted review is complete.
  ```
- **Evidence hierarchy and independent implementation.** Trust sources in this order: (1) prosper's
  live captures/traces of the real guest, (2) published platform contracts, firmware symbol data, and
  the guest's own disassembly, (3) agreement among independently written secondary implementations,
  then (4) a single secondary implementation as a hypothesis only. PS5-specific AGC, Gen5 descriptors,
  and PS5-only kernel calls require direct title evidence; inherited PS4 behavior still must be checked
  against the exercised guest path. External implementations are verification-only: do not copy or port
  their code, types, comments, prose, or tests. Re-derive behavior in prosper's own architecture and add
  project-owned evidence/tests. Never weaken behavior demonstrated by a live boot to match a secondary
  reference, and mark unresolved evidence with `CONFIDENCE: HIGH/MED/LOW`.
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
