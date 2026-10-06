# Roadmap — `prosper`

What is planned and where the project is heading, as of **2026-10-02**. This is a map, not a status
report: per-title rungs, routes and blockers live in `COMPATIBILITY.md` and the `tracker:game`
issues, and `CLAUDE.md` § *Where the project stands* names the doc to read for each title or area.
The original M0-M7 milestone plan and the July 2026 status log are archived in
`archive/ROADMAP_HISTORY.md` and are not current.

Honest framing: a PS5→PC layer is a multi-year, reference-implementation-scale effort. It is
tractable because guest code is x86-64 and runs natively; the cost is the library surface, the AGC
command stream and the RDNA2 shader recompiler.

## How progress is measured

Every title climbs one ladder, and claims name the rung reached:

1. any real graphics → 2. title screen → 3. gameplay with the scene rendering → 4. a human confirms it
looks right → 5. comparison against PS5 hardware reference → 6. a reviewed automatic snapshot guard
(`tools/snapshot`). A title is done only at rung 6.

Verification is programmatic first (`docs/process/VERIFICATION.md`): quote the `ctest --no-tests=error`
count with the exit code, and assert rendered frames by pixels, hashes or routed content metrics, never
by "a window opens".

## Where things stand

The foundational stack — loader, libc/libkernel/HLE, AGC decode, the Vulkan executor, the RDNA2→SPIR-V
recompiler, input, audio, video decode, save data — exists and is exercised daily. Several titles reach
rung 6 (*The Messenger*, *Dead Cells*, *Blasphemous 2*, *Alex Kidd in Miracle World DX*, *Blue Prince*,
*GRIS*, *Space Adventure Cobra*); the rest of the corpus sits between rungs 0 and 3. Read
`COMPATIBILITY.md` for the current table rather than a count copied here.

## Active directions

Ordered by how many titles each one unblocks.

1. **Renderer throughput.** The steady-state invariants in `CLAUDE.md` (P1-P6) set the direction; prosper
   still violates P1 (the main render submit waits on its fence, staged fix #3948) and P5 (full `memcmp`
   revalidation of resident buffers once write-watch disables itself, #3155). *GTA V*'s frontier is
   framerate, not the picture. Method: `docs/performance/`.
2. **Submit-race on SDK < 13 titles.** The post-submit completion-visibility contract is armed only for
   SDK ≥ 13, so *ArcRunner* and *Crisis Core* lose their command-chunk recycler mid-fold (#2217, #2219).
   The fix regresses *Sonic Frontiers* (#2223), so it waits on a cross-title snapshot pass (#2220).
3. **Resource binding and the recompiler.** The user-data-window mismatch that drops *Nikoderiko*'s 3D
   world (#1607); the white clipped band in *Sonic Frontiers*' post-process chain (#2790); remaining
   recompiler gaps (`docs/gpu/RECOMPILER_REMAINING.md`). An unsupported GPU operation is a fatal gap to
   implement, not a skip.
4. **Per-title bring-up.** Titles at rung 2-3 need their next-rung blocker closed; the never-booted
   titles need a first boot (`docs/games/NEVER_BOOTED_SURVEY_2026_08.md`). Shared engine work for Unreal
   and Unity lives in `docs/engines/`.
5. **Release and platforms.** Linux AppImage/tarball and the Windows port ship on `v*` tags
   (`docs/platforms/`). The pre-release full snapshot matrix is the regression gate.
6. **Architecture health.** The ratchets in `tools/ci/check_arch_ratchet.py` only go down (title ids in
   shared code, raw `getenv`, blocking GPU syncs, 5,000-line files). Target tree and file splits:
   `docs/architecture/`.

## Architecture sequence (proposed)

The architecture work is recorded as decision records in `docs/adr/`. None of the ADRs sequenced here
is accepted yet, so this is a **proposed** order for the owner to confirm or change, and no step
starts before its ADR is accepted. It is chosen so each step lands behind the safety net the previous
one built. The binding rules each step serves are in `docs/spec/`.

| order | step | ADRs | why here |
| ---: | --- | --- | --- |
| 1 | Move the Vulkan backend out of `tests/` | 0004 | every renderer change after this lands in the backend's permanent home |
| 2 | Split guest semantics into `guest/`; one CMake target per clean layer | 0003, 0008 | gives engines, the memory tracker and the sync model a home, and makes layering a compile error |
| 3 | Replay gate for GPU-path changes; sync validation in CI (blocked on #3255: a hazard only the CI runner's layers report) | 0005 | the regression net for every restructure below |
| 4 | Pipelined submission (Stage 2 first), off-thread pipeline compilation | 0009, 0014 | the largest measured cost (`PERF-P1`, `PERF-P6`, `PERF-P3`) |
| 5 | Page-owned memory tracking, canonical resource identity, one sync model | 0010, 0013 | `PERF-P5`, and the fiber-migration defect class |
| 6 | Typed GPU command representation; SSA IR in a pure recompiler | 0011, 0006, 0012 | separable passes instead of edits to thousand-line bodies |
| 7 | Declarative HLE tables, typed guest pointers, per-library conformance suites, coverage report | 0015, 0018, 0020, 0007 | the library surface becomes listable, traceable and contract-tested |
| 8 | Log channels and typed configuration; evidence runs record their switches | 0017, 0023 | the switch surface shrinks to what is genuinely temporary |
| 9 | Fuzzing, bug-report bundle, no host exceptions into guest frames, title-code registry, release perf gate | 0019, 0021, 0022, 0024, 0016 | hardening that can run alongside any step above |

Profile-guided and AI-assisted optimisation stay deferred until steps 3, 6 and the release perf gate
exist (`docs/spec/performance.md`).

## Release criteria (open)

The ladder defines when a *title* is done; nothing yet defines what a prosper *release* promises. These
are the questions a release definition answers, left for the project owner to decide rather than
filled in here:

- which titles, at which rung, a release must keep (today: the snapshot matrix is the pre-release gate);
- the supported hosts, GPUs and drivers, and what "supported" commits to;
- the frame-rate tier per workload class, measured as `PERF-G1` describes;
- what a user report must contain (`OPS-1`) and how a regression found after release is handled.

## Standing rules that shape the plan

- Entitlement and add-content APIs answer from local inventory, never "owned" unconditionally.
- No DRM circumvention and no third-party bypass modules are loaded
  (`src/host/image/module_path_policy.hpp`).
- Reach for free vendor tooling (RGP, RenderDoc, `radeontop`) and the F8/F9 captures before adding a
  `PROSPER_*` switch; a guest-behaviour selector needs an issue that settles the default.

## Reality checkpoints

- A new title's first boot tells us which library surface it needs; re-rank direction 4 from that.
- If throughput work stalls on a synchronous boundary, evaluate it against a **3D** workload before a
  2D one (the July pass deliberately stopped at the first 2D level).
- Refresh this file when a direction completes or a new cross-title blocker appears; delete finished
  items instead of striking them through.
