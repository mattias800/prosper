# The Plucky Squire (`PPSA15319`) — status

Unreal Engine 4. Tracker: [#1882](https://github.com/mattias800/prosper/issues/1882).
Bring-up record: [#1390](https://github.com/mattias800/prosper/issues/1390).

**Read `## Ruled out` before forming a hypothesis.**

## Current state (2026-08-19, master `2703a6c3`)

Rung 2 on the ladder, with the **real 3D world now rendering**: the checked-in route reaches title,
save-slot and play-style menus, streams `MainLevel` + `Desk_C01` + `Desk_C01_Lighting`, and then plays
the authored chapter-one intro cutscene (`cam_cutscene_c01_intro`) with real GPU draws at 3840x2160.

[![The Plucky Squire — chapter-one intro](https://raw.githubusercontent.com/mattias800/prosper/master/assets/screenshots/plucky-squire-chapter1-intro.png)](https://github.com/mattias800/prosper/blob/master/assets/screenshots/plucky-squire-chapter1-intro.png)

*`tools/screenshot` (headless frontend), unmodified 3840x2160 capture downscaled for the repository,
checked-in `scripts/plucky-squire/reach-first-gameplay.pad` route, t = 1080 s.*

**This is not rung 3.** A cutscene is not gameplay.

**The frontier was re-measured on 2026-08-21 (master `9dcf807f`) and it is not what this section
originally guessed.** "The route stops driving input at 525 s" is true and irrelevant: the guest reads
the input it is given and the cutscene is not waiting on a button. The cutscene is not waiting at all —
it is *advancing about 300x too slowly to finish*, because the guest runs at ~0.19 flips/s once the 3D
world is up and prosper's guest clock advances in-game time **per flip**. See
[**## The wall is guest THROUGHPUT**](#the-wall-is-guest-throughput-and-the-reason-is-the-guest-clock-2026-08-21-master-9dcf807f)
below and [#2839](https://github.com/mattias800/prosper/issues/2839); raising the flip rate alone
carries the guest past the intro to the `Book_MAIN` storybook camera with no other change.

## Route and timing

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_RENDER_SCALE=1 PROSPER_RENDER_EVERY=1 PROSPER_NO_FRAME_DUMPS=1 \
PROSPER_PAD_SCRIPT=@scripts/plucky-squire/reach-first-gameplay.pad PROSPER_PAD_SCRIPT_LOG=1 \
PROSPER_SAVE0=<RUN>/save0 PROSPER_SAVEDATA_DIR=<RUN>/savedata \
  ./build/screenshot <DUMP_ROOT>/PPSA15319-app0 --seconds 10 --count 120 --out <RUN>/shots --timeout 1400
```

**Use isolated save roots for a first-boot arm.** Since #2734 both roots are namespaced by title id
(`<root>/<TITLE_ID>/…`, `prosper/docs/SAVE_DATA_LAYOUT.md`), so two UE4 titles picking the same
guest-chosen directory name no longer collide. Setting the roots still matters here for a different
reason: it makes the arm start from an *empty* save state rather than from whatever an earlier run of
**this** title left. `PROSPER_SAVEDATA_DIR` is the separate SaveDataMemory root; the launcher
`scripts/plucky-squire/run-first-gameplay.sh` sets both.

Observed phase timings on one reference machine (they vary by well over a minute between hosts, so
bound generously — a 420 s run is too short and reads as "never leaves loading"):

| t | phase |
| --- | --- |
| 0-20 s | logos |
| 30-150 s | title, `SAVE FILES`, `PLAY STYLE` menus, all legible and full colour |
| ~160 s | white fade, then black + the game's own loading-book glyph |
| ~203 s | `MainLevel` loaded (1.3 s), `Desk_C01` + `Desk_C01_Lighting` streamed, `FinishDeskLevelLoad` |
| ~434 s | `Finished Streaming All Minigames`, `SetTargetCamera(cam_cutscene_c01_intro)`, `LCD_MUS_C01_INTRO_FULL` queued |
| 434-1200 s | intro cutscene; real world geometry in roughly a quarter of 10 s samples, black in the rest |

Renderer throughput during the cutscene is about **5 fps** at native 4K.

## Reproducibility

The route reaches the intro cutscene **2 of 2** on the checked-in `.pad` file, same binary, same dump,
fresh private save roots each time. Both runs completed 120/120 samples with `guest=running status=ok`
and zero `VK_ERROR` / device-lost / worker faults, and world geometry was confirmed by **opening the
frames**, not by aggregate metrics:

| run | `FinishDeskLevelLoad` | `cam_cutscene_c01_intro` | world content |
| --- | --- | --- | --- |
| A | ~203 s | ~434 s | yes, frames 46-119 intermittently |
| B | ~150 s | ~370 s | yes, frames 47-119 intermittently |

Aggregate metrics cannot establish this and must not be quoted as if they could: an animating menu is
exactly as `pixel-distinct` as a rendered level. The load-bearing check is the **scene sequence** —
`SetTargetCamera(cam_cutscene_c01_intro)` in the guest log, plus an opened frame showing bedroom
geometry.

A third run using a locally-extended multi-button route **plus** `PROSPER_SHADER_DUMP` and
`PROSPER_GPU_TIMELINE` ran roughly 3x slower (`FinishDeskLevelLoad` at ~670 s instead of ~200 s) and
aborted at ~1016 s, so it never reached the cutscene. Instrument overhead, not a route difference, is
the likely cause; either way, do not use a shader-dump run to judge progression.

## Input

The title opens **exactly one pad handle** — `[pad] OPEN userId=1 type=0 index=0 -> handle=1` under
`PROSPER_PADLOG=1`, on a 60 s probe. It is therefore **not** exposed to the shared suspect that
`poll_controller` ignores its handle argument (`src/hle/input/hle_pad.cpp:415`), which can make two handles
mirror one controller. Recorded as a negative so the next lane does not re-probe it.

Input is delivered and observed: `[pad-script]` lines carry the guest's own advancing pad-read index
(`read=`), which tracks the frame counter across all 49 transitions of the route. So anywhere this
title fails to advance, the correct statement is "the guest read the input and did not act on it", not
"the input was not delivered".

### The title's own action map, read out of its own save

The game writes `/savedata0/<TITLE_ID>/InputSettingsKeyMappings/ue4savegame.dpx.sav`, a UE4 `GVAS`
blob whose `/Script/Storybook.UserInputKeyMappings` array is the complete
`ActionName` -> `KeyName` table. It is plain enough to read with `strings`, and it settles button
questions without guessing:

| action | button |
| --- | --- |
| `UI_Select`, `Jump` | **Cross** (`Gamepad_FaceButton_Bottom`) |
| `Pause` | **OPTIONS** (`Gamepad_Special_Right`) |
| `UI_Cancel`, `Roll` | Circle (`Gamepad_FaceButton_Right`) |
| `Action`, `UI_GoToGallery2` | Triangle (`Gamepad_FaceButton_Top`) |
| `Attack`, `UI_GoToGallery1` | Square (`Gamepad_FaceButton_Left`) |
| `Tool_1` / `Tool_2` | D-pad left / right |

**There is no `Skip` action at all**, so a cutscene skip — if the title has one — goes through the
`Pause` menu rather than a dedicated button. The Cross-only route is therefore correct for the menus
(Cross *is* `UI_Select`), and the reason it does not clear the cutscene is not a mis-mapped button.

### Wall-clock anchors cannot serve this title; use pad-read anchors

`scripts/plucky-squire/reach-gameplay-reads.pad` is the same ladder written on the guest's own axis
(`p<start>-<end>:button`, already parsed and unit-tested — `tests/input/test_pad.cpp:573`). With a 147x
tick-rate spread between the menus and the cutscene, one wall-clock cadence cannot serve both ends: the
5 s pulses that pace the menus are worth ~125 guest polls at the title screen and **one or two** once
the 3D world is up, while a gap that feels brisk in a menu is over ten minutes of cutscene. Read
anchors are invariant to that, and to the sampling cadence — the same file drives a default run and a
`PROSPER_RENDER_EVERY` run without retiming. Verified to navigate logos, `SAVE FILES` and `PLAY STYLE`
through to `FinishDeskLevelLoad`.

## The wall is guest THROUGHPUT, and the reason is the guest clock (2026-08-21, master `9dcf807f`)

The chapter-one intro cutscene is **not stuck** — it advances, roughly 300x too slowly to finish. What
makes that fatal rather than merely slow is prosper's own guest-clock contract, so the two have to be
read together.

**1. The guest's tick rate collapses when the 3D world comes up.** `PROSPER_PAD_SCRIPT_LOG=1` carries
the guest's own pad-read counter, and the UE4 log prefix carries `GFrameCounter`; the two agree, which
is what makes this a guest-side measurement rather than a renderer one. The table counts **guest
polls**; the flip rate that drives the clock below is the same quantity to within the tick-to-flip
ratio (UE polls the pad once per game tick and flips once per rendered frame), so the two are used
interchangeably here and the small difference is inside the model's error bar:

| phase | guest polls/s |
| --- | --- |
| logos, title, `SAVE FILES`, `PLAY STYLE` | **~25** |
| `Desk_C01` streaming | ~4-5 |
| desk level up, pre-cutscene | ~2.1 |
| chapter-one intro cutscene | **~0.17-0.20** |

A **147x** spread, reproduced on four independent runs (0.198, 0.197, 0.198, 0.218 polls/s).

**2. In-game time advances per FLIP, not per second.** `execute_submit_work` wraps its GPU work in
`HostGpuClockScope(clock_budget_ns)` with a budget of **one refresh interval per flip**, and
`guest_clock_host_gpu_end` (`src/hle/kernel/hle_kernel_time.cpp:208`) accumulates everything past that
budget into `total_excess_ns`, which the guest clock subtracts. That is deliberate and right — a real
console does not charge shader compilation and resource conversion to the guest's next frame delta —
but it means the game's clock advances **~16.7 ms per flip regardless of how long the flip took**.

Checked numerically against the title's own logic clock rather than assumed: over the 221 s
pre-cutscene window the guest flipped ~2.3/s, predicting 221 x 2.3 x 16.7 ms = **8.5 s** of game time;
the game's `LOGIC:` stamps moved 3.577 -> 11.222, i.e. **7.6 s**.

**Multiply the two and the wall follows.** At 0.19 flips/s the cutscene's clock runs at
0.19 x 16.7 ms = **~3 ms of game time per second of wall clock, ~0.3% of real time**. The intro is
~60 s of game time, so a default-cadence run needs **hours** to finish it; the 1,200 s run that
"never reached gameplay" bought a few seconds of it. Treat the model as an order-of-magnitude
account, not a stopwatch — it over-predicted the one interval it was checked against by ~11% (8.5 s
predicted, 7.6 s measured), because not every guest tick is a flip and the intro's exact length is
not measured. The conclusion survives any correction that size: nothing near 1,200 s is enough.

So this title's rung-3 blocker is a renderer-throughput problem wearing a progression problem's
clothes. Nothing about the cutscene logic, the route, or the two absent fog programs is implicated.

### Where the time goes

`PROSPER_RENDER_TIMING=1`, cutscene windows, per **draw submit** (~24 of them per guest flip):

```
total=189-341 ms  build_resources=127-229 ms  backend=62-203 ms
  build_resources: textures=657 bindings/submit, reused=434, 126-227 ms   <- dominant
  backend:         pipeline=up to 133 ms, resources texture upload=47 ms
```

Not GPU-bound: `radeontop` reads **5-16%** during the cutscene, against the `vkcube` control of
**56.31%** recorded in `CLAUDE.md` for this box. A gdb stack sample (10 samples) puts the guest's
`RenderThread 1` in `sched_yield` via `prosper::k_pthread_yield` **10 of 10 times** while prosper's
single `AgcSubmissionTh` does BC decode, `float_to_half`, `memmove` and `amdgpu_bo_alloc` inside
`build_resources`. This is the same frontier as [#1177](https://github.com/mattias800/prosper/issues/1177)
(Bendy, CPU detile dominates) and the Blue Prince `setup_resources` result in `CLAUDE.md`.

### What that unlocks

With the sampling cadence genuinely engaged (`PROSPER_RENDER_EVERY=16`, verified by the new
`[render-cadence]` line — see *Instrument notes*), the cutscene runs at **2.71 polls/s, a 13.7x
speedup**, and the game's logic clock advances past the intro: `SetTargetCamera()` leaves
`cam_cutscene_c01_intro` for **`Book_MAIN`**, the storybook camera, at `LOGIC 0:30.8`. That is half
the success condition this document set on 2026-08-19 ("a `SetTargetCamera()` to a non-cutscene
camera **plus a frame showing** the C01 storybook page or a controllable character").

**The other half is not met, so this is not rung 3.** Two things block it, and both are recorded
rather than worked around:

- **Accelerated frames are not progression evidence, and cannot be made into it by timing.** At
  `every=16`/`32` most passes never render and the composite is black — the skipped submits are
  different *passes* of one UE4 deferred frame, not repeats of it. Handing the cadence back with
  `PROSPER_RENDER_EVERY_FOR_MS=700000` while the storybook camera was live produced frames that are
  **solid black or solid white**, not a page.
- **Whether that white is the accelerator's residue or a real composite defect is not separable on
  this route**, because the only arm that could tell them apart — a default-cadence run that reaches
  the same state — is the ~6-hour run the wall above forbids. Do not record it as either.

  A trap worth inheriting: a solid-white 4K PNG is **166 KB**, so file size read as "real content
  appeared" exactly when it had not. Open the frames.

- The run then dies on a **deterministic** guest `SIGSEGV at addr=0x8, rip=image+0x16460f7`,
  reproduced 2 of 2 at two different cadences — [#2841](https://github.com/mattias800/prosper/issues/2841),
  the next blocker behind this one.

## Open defects

- [#2741](https://github.com/mattias800/prosper/issues/2741) — two UE4 volumetric-fog compute programs
  never execute (`0x3015ab0000` 0/72, `0x3015fd0000` 0/6). Exact rejects: an entry `s_mov_b32 s14, m0`
  and an `s_cselect_b32 vcc_lo, s37, s36` where LLVM has recycled both VCC words as scalar data.
  **The VCC half is fixed** — a live env-gated A/B moved `0x3015fd0000` from `executed=0 skipped=6` to
  `executed=6 skipped=0` with nothing rejecting behind it, and the widened predicate has landed. The
  **M0** half (`0x3015ab0000`, pc151, this title's *main-view* 240x135x64 volume) is untouched and
  still blocked on the narrower liveness-proved form, so this title's main froxel pass remains absent
  and **no image change is claimed here**.
- [#1581](https://github.com/mattias800/prosper/issues/1581) — the rare self-recovering descriptor
  transient; measured here at 1 skip in 46,667 and 2 in 4,488.
- The world renders very dark with large fully-black regions, and one sampled frame shows blown-out
  white exteriors against a crushed interior. Not yet attributed.

## Ruled out

- **"The pipeline cache is thrashing, and sizing it is the throughput lever."** The premise is true and
  the conclusion is false — recorded together because the premise is seductive on its own. On a default
  run the backend pipeline cache sits pinned at `entries=4096` (its cap) for the whole cutscene with
  `misses ~= evictions` in every window (1 to 30 per submit, ~8.8 average, up to 133 ms/submit), which
  is textbook capacity thrash. **`PROSPER_PIPELINE_CACHE_ENTRIES=16384` removes it completely** —
  `evictions=0.0`, entries climbing 13,990 -> 15,005 — and the cutscene tick rate moves **0.198 ->
  0.218 polls/s**, i.e. nothing beyond run-to-run spread. The misses were never capacity re-creations:
  they are *first-time* pipeline creations, and this title had built >15,000 distinct pipelines and was
  still climbing. Sizing the cache is worth doing for its own sake; it is not the throughput fix
  (2026-08-21, [#2839](https://github.com/mattias800/prosper/issues/2839)).
- **"`PROSPER_RENDER_EVERY` is inert on this title because `ordered_dma_requires_render` overrides
  every submit."** **Void, and it was published as a conclusion before it was checked** — the lane's own
  launcher re-exported `PROSPER_RENDER_EVERY=1` after the caller set 16, so the "accelerated" arm was a
  cadence-1 run and its 3.92 present/s against a 3.17 baseline measured only run-to-run spread. With
  the variable actually reaching the process the cadence works: `skips_wanted=23039 dma_forced=128`,
  i.e. the DMA override fires on **0.6%** of requested skips, not 100%. The `[render-cadence]` line
  added in #2837 exists because of this specific mistake and is what caught it — **quote
  `requested_every=` from the run, never from the command you believe you typed.**
- **"#2741's absent fog programs are why the world is dark / why gameplay is unreachable."** Not
  reached by this lane's evidence either way for the darkness, but **excluded for the progression
  question**: the guest advances to the storybook camera with both programs still absent, purely by
  raising the flip rate. Whatever the two fog rejects cost, they do not gate the cutscene.
- **"#2741 is this title's bug."** Falsified 2026-08-19 by a cross-title census on `2703a6c3`
  (#2747). *Little Nightmares III* rejects `0x30114c0000` at the **byte-identical** dword `be8e037c`
  (`s_mov_b32 s14, m0`) with the **byte-identical** dispatch `groups=30x17x64 local=8x8
  threads=240x136` and the same 16,588,800-byte volume as `0x3015ab0000` here; *Dragon Quest VII*
  (`0x3017400000`) and *The Pathless* (`0x200ea80000`, `0x200ead0000`) each lose a froxel program to
  `s_cselect_b32 vcc_lo, sX, sY`, and *The Pathless*'s second volume is byte-for-byte the same
  131,072 B as `0x3015fd0000` here. Four titles, one stock UE shader pair, one recompiler gap.
- **"The route's dispatch geometry is fixed at `groups=30x17x64 / threads=240x136`."** Not falsified,
  but **not stable either** — an independent 1,152 s arm of the same route on the same binary reports
  `groups=27x15x64 local=8x8 threads=216x120` for `0x3015ab0000`, i.e. a 3456x1920 view. UE dynamic
  resolution moves the froxel grid's XY between runs; `gz=64` and the 16,588,800 / 131,072 bindings do
  not move. Do not use the XY extent as a run-to-run selector. #2747.
- **"The three compute programs of #1554 are still skipped."** Falsified 2026-08-19 on `2703a6c3`:
  `PROSPER_COMPUTE_PROGRAM_CENSUS=1` over 131,072 dispatch decisions across 50 programs shows
  `0x3017d90000`, `0x3017450000` and `0x3017460000` with **no skip rows at all** — the census prints a
  row only for a program that skipped at least once. Closed by #1561 / #1564 / #1572.
- **"The route never leaves the loading phase / level streaming is the blocker."** Falsified
  2026-08-19: `MainLevel` loads in **1.27 s** and `AStorybookGameModeBase::FinishDeskLevelLoad` fires at
  ~203 s, while the screen stays black until ~434 s. The black interval is the game's own pre-cutscene
  state, not a stalled load. The earlier 420 s bounded run that "never left loading" simply ended before
  the cutscene started (#1390, 2026-08-02).
- **"The title is blocked on the pre-SDK-13 AGC contract"** — the shared UE4 wall behind Oregon Trail's
  unblended UI and the ArcRunner/Crisis Core submit race. Falsified: this title logs
  `[agc] register defaults requested for SDK version 13`, so it is not on the pre-13 path at all.
- **"An untracked `VCC_LO` read is a usable mutation arm for a scalar-data reject."** Void, not
  falsified — it passes on both sides of the change, because when the native subgroup equals the guest
  wave the operand resolves through `native_wave_ballot_half` before reaching the reject. Use `TTMP0`
  (operand 108), which has no such alternative path (#2741).
- **"Reading an unwritten M0 as 0 is a safe generic fix."** Rejected 2026-08-19, deliberately: it
  reverts #134, whose guard is `tests/gpu/recompiler/test_rdna2_to_spirv.cpp:9483`
  (`"kernel X2 (m0 read as ALU data) is REJECTED"`). A silent 0 cannot be distinguished from a *decoder*
  gap that lost an M0 write. The narrower liveness-proved form remains open in #2741.

## Instrument notes

- **A requested `PROSPER_RENDER_EVERY` is not a cadence you actually got.** `PROSPER_RENDER_CADENCE_LOG=1`
  prints `[render-cadence] draw_submits=… requested_every=… skips_wanted=… dma_forced=… (…% of
  requested skips)`, and an *un*gated one-shot `WARNING` fires when a retained DMA copy overrode every
  skip the cadence asked for. Read `requested_every=` before believing any accelerated run: it reports
  the value the process actually holds, which is how this lane discovered its own launcher had been
  overwriting it (#2837). Measured here: `every=16` -> `dma_forced` **0.6%** of requested skips.
- **`PROSPER_RENDER_EVERY` accelerates the guest and destroys the picture, in that order.** At
  `every=16` the cutscene runs 13.7x faster and most sampled frames are **black**, because the skipped
  submits are different *passes* of one UE4 deferred frame rather than repeats of it. Hand the cadence
  back with `PROSPER_RENDER_EVERY_FOR_MS` well before any checkpoint you intend to photograph, and
  budget for the fact that the tick rate returns to ~0.2/s the moment you do.
- **The guest's own pad-read counter is the cheapest throughput instrument this title has.**
  `PROSPER_PAD_SCRIPT_LOG=1` prints `read=` on every scripted transition; differencing it gives guest
  polls/s with no renderer involvement. prosper's `[shot] (frame N, …)` counter is **not** this — it is
  `present_frame_seq`, one per *draw submit*, and this title issues ~24 of those per guest flip, so the
  two differ by more than an order of magnitude and only the pad-read one answers "is the game
  advancing?".
- `[compute] skip unsupported program` prints **once per program address**, so a run showing four skip
  lines says nothing about how many dispatches were lost. Always pair it with
  `PROSPER_COMPUTE_PROGRAM_CENSUS=1`, which prints executed/skipped ratios and dispatch grids.
- `PROSPER_DBG=1` desyncs the pad route badly enough that a long route never reaches the phase being
  diagnosed. Use `PROSPER_SHADER_DUMP=<dir>` on the live route to retain the raw stages, then replay the
  reject offline with `PROSPER_DBG=1 shader_inspect <dump> --stage compute`. Its
  `status=undetermined-no-resource-table` verdict is a tool limitation, not evidence — but its
  `[scalar-data-reject]` lines reproduce the live cause exactly.
- A run configured with `PROSPER_SHADER_DUMP` **and** `PROSPER_GPU_TIMELINE` aborted with a glibc
  `corrupted double-linked list` at ~1016 s, immediately after repeated `[render] dumped SPIR-V` lines.
  The same route without those two variables completed 120/120 samples with `guest=running status=ok`.
  Treat long shader-dump runs as bounded.
