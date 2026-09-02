# R-Type Delta: HD Boosted (`PPSA26414`) — input routes

Routes for the `tools/screenshot` frontend, driven through `PROSPER_PAD_SCRIPT`. Every anchor in
every file here is **seconds from the first pad poll** (`prosper/src/input/pad.cpp` — a bare number is
seconds, `f` is guest flips, `p` is pad reads).

| file | what it is |
| --- | --- |
| `reach-gameplay.pad` | the route. OPTIONS clears the title screen, then Cross walks the pilot-registration flow, the ship / Force-device select and into stage 1. Run **twice** back to back on `a37ef264`; both runs reached stage 1 (45 and 46 distinct frames over the samples at t ≥ 30 s, peak 2,073,478 / 2,073,475 non-black pixels, five scene modules ending in `st1r9_Release.prx`). Both validation runs are on `a37ef264` and **not** on master, because at the time master drew none of it (#2783). That regression is now fixed by #2799, and its A/B re-ran this route on the repaired build (46 of 46 distinct late CRCs, zero present-source mismatches), so the route validates on master too. The checked-in `assets/screenshots/rtype-delta-stage1-gameplay.webp` is from the `explore-v2.pad` arm on `7cc74ef5`, not from these two runs — its commit caption names that head, and both heads are simply pre-#2783 builds. |
| `neutral.pad` | the matched **neutral-input control**. Delivers nothing inside any bounded run, but keeps a route configured so the arm differs from `reach-gameplay.pad` only in the buttons. |
| `explore-v1.pad`, `explore-v2.pad` | the two probes the route was derived from. Both were **run**, and each header records what its run measured. |

## Launching

This title must have its dump **evicted from the host page cache** before every launch or it loses its own
startup race (#1746 — a product decision, not an open investigation; `docs/R_TYPE_DELTA_STATUS.md`
§ *Host page-cache state decides the race*):

```bash
python3 prosper/tools/dropcache.py <DUMP_ROOT>/PPSA26414-app0        # prints resident MB before -> after
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
  PROSPER_SAVE0=<a fresh, per-title, per-run directory under $HOME> \
  PROSPER_PAD_SCRIPT=@prosper/scripts/rtype-delta-PPSA26414/reach-gameplay.pad \
  PROSPER_MODLOG=1 PROSPER_PAD_SCRIPT_LOG=1 \
  ./build/screenshot <DUMP_ROOT>/PPSA26414-app0 --seconds 2 --count 90 --out <dir>
```

Eviction is one-shot — the run re-warms the cache, so repeat it before every launch — and it is not a
guaranteed win. Measured on 2026-08-20, on a box shared with one other agent lane (1-minute load
average 3.9-6.6 across the arms): with eviction alone, **1 launch in 6 won the race** (one arm lost
once then won; the next lost 4 consecutively). With the launch *additionally* throttled by 12 busy
loops pinned to the same 4 logical CPUs as prosper for the first 12 seconds, **4 of 4 won it on the
first attempt**. That throttle is the lever
`R_TYPE_DELTA_STATUS.md` § *The race is decided by host single-thread CPU speed* documents with an
executed positive arm. That throttle changes the **host** only: unmodified binary, unmodified guest,
no patched bytes, no capped guest sleep. A lost launch is unambiguous — it faults at `eboot+0x24055`
and `tools/screenshot` exits non-zero with `status=GUEST-FAULT` (#2007) — so scripted retry is safe.

**`PROSPER_SAVE0` must be fresh.** The route is save-state dependent: with no pilot data the title
diverts the first `START GAME` into `REGISTER PILOT`, which is what the Cross presses are timed
against. `PROSPER_SAVE0` is a root since #2734 and saves land in `<root>/PPSA26414/`.

## Reading a run: the title tells you where it is, twice

**Module loads.** The shell loads one PRX per scene from `/app0/prx`, and `PROSPER_MODLOG=1` names
each one as it starts. This is the progress oracle: `title_Release.prx` → `loadsel_Release.prx` →
`select_Release.prx` → `loads1_Release.prx` → `st1r9_Release.prx` (stage 1, R-9).

**A description line.** The title draws a one-line English description of the highlighted menu entry
at the bottom centre of the frame — `Start Game`, `Select Slot to Create`, `Enter Name`. It survives
even when the rest of the menu does not draw, so it reads a run's menu state straight off the frames.

## Attract mode loads stage PRXs too — the discriminator is `loadsel` + `loads1` + save data

**A stage PRX load is NOT evidence of player-driven play on this title.** Measured 2026-08-20 with
`neutral.pad` (same command, same build, no input at all): the attract loop loaded
`title_Release.prx`, `select_Release.prx`, `st1r9_Release.prx`, `st2_Release.prx` and
`st5_Release.prx` — so the scene modules a real playthrough loads are loaded with the pad untouched.
(That arm's own frames could not show what those scenes looked like, because #2783 had frozen the
composite by then; that the Force-device scene is part of the attract loop is corroborated
independently by `assets/screenshots/rtype-delta-force-select.webp`, captured on a **default,
unrouted** run.)

What the neutral arm never did: load **`loadsel_Release.prx`** or **`loads1_Release.prx`** — the
title's own two loading-screen modules, which only the menu → select → stage path goes through — and
never wrote **`<PROSPER_SAVE0>/PPSA26414/SaveData.dat`**. All three appear in every routed run.
Use them, not the stage module and not the look of the frame.

## Ruled out

| Hypothesis | Verdict and evidence |
| --- | --- |
| The title screen's `PRESS` prompt names Cross, as most titles' do | **Falsified by the rendered prompt itself.** Its glyph is the PS5 **OPTIONS** icon — a filled vertical pill with three horizontal lines above it (`assets/screenshots/rtype-delta-title.webp`, region 1060,740–1180,870 at 4×). *Limit of this row:* no Cross-only arm was run, so this falsifies what the prompt **names**, not what the title would accept. What is measured about behaviour is that a route containing OPTIONS clears the title by t≈12 s, and that a no-input arm never clears it at all. |
| OPTIONS also confirms inside the menus, so it is safe to keep pressing it | **Falsified by a two-arm A/B.** `explore-v1.pad` (OPTIONS and Cross alternating, 5 s apart) never advanced past the first two menu states — it oscillated between `Start Game` and `Select Slot to Create` with a 15 s period, three times its own 5 s cycle. `explore-v2.pad` is the same file with OPTIONS removed after t=7 s and it walks the whole flow into stage 1. So OPTIONS **undoes** menu progress; whether it is *back* or *cancel* specifically is not distinguished here, and does not need to be — it must stop after the title either way. |
| Configuring `PROSPER_PAD_SCRIPT` is what makes a routed run behave differently from an unrouted one, because the route reports a connected controller | **Falsified.** prosper's default pad backend already reports **one connected controller with neutral input** when no route is set (`src/hle/input/hle_pad.cpp`, `poll_controller`). A route changes the buttons, not the connection — so `neutral.pad` is a matched control and an unrouted run is not a different controller state. |
| The name-entry screen needs a non-Cross "decide" button | **Falsified.** Cross alone completes it: it types `A` into the grid repeatedly and the flow returns to `Start Game` with `SaveData.dat` written. The eboot imports no `libSceImeDialog`, so the grid is the title's own. |
