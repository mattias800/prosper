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

**This is not rung 3.** A cutscene is not gameplay. The frontier is what happens after the intro
sequence: the route as checked in stops driving input at 525 s, and no sample in a 1200 s run showed
the storybook page or a controllable character.

## Route and timing

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_RENDER_SCALE=1 PROSPER_RENDER_EVERY=1 PROSPER_NO_FRAME_DUMPS=1 \
PROSPER_PAD_SCRIPT=@scripts/plucky-squire/reach-first-gameplay.pad PROSPER_PAD_SCRIPT_LOG=1 \
PROSPER_SAVE0=<RUN>/save0 PROSPER_SAVEDATA_DIR=<RUN>/savedata \
  ./build/screenshot <DUMP_ROOT>/PPSA15319-app0 --seconds 10 --count 120 --out <RUN>/shots --timeout 1400
```

**Use isolated save roots.** `PROSPER_SAVE0` defaults to a single flat `/tmp/prosper-savedata0` shared
by every title (#2734), and the per-save host directory is that root plus the *guest-chosen* directory
name — so two UE4 titles picking a generic name collide. `PROSPER_SAVEDATA_DIR` is the separate
SaveDataMemory root; the launcher `scripts/plucky-squire/run-first-gameplay.sh` sets both.

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

## Open defects

- [#2741](https://github.com/mattias800/prosper/issues/2741) — two UE4 volumetric-fog compute programs
  never execute (`0x3015ab0000` 0/72, `0x3015fd0000` 0/6). Exact rejects: an entry `s_mov_b32 s14, m0`
  and an `s_cselect_b32 vcc_lo, s37, s36` where LLVM has recycled both VCC words as scalar data.
- [#1581](https://github.com/mattias800/prosper/issues/1581) — the rare self-recovering descriptor
  transient; measured here at 1 skip in 46,667 and 2 in 4,488.
- The world renders very dark with large fully-black regions, and one sampled frame shows blown-out
  white exteriors against a crushed interior. Not yet attributed.

## Ruled out

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
