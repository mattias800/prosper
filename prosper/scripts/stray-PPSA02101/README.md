# Stray — `PPSA02101`

`reach-first-map.pad` drives the boot past the **brightness-calibration screen** to the first map
load.

`reach-title-flip.pad` was landed as "reaches the title screen and holds there". **Measured 2026-08-30,
that is wrong**: over a 320 s run all 16 captured frames are the brightness-calibration screen
(`max_nonblack` 0.1140). Its single flip-anchored Cross does not accept. Treat it as a deterministic
way to *hold calibration*, which is still useful — that screen renders correctly and is a stable
oracle — but not as a title-screen route.

**`reach-title-hold.pad` is the title route**, and it needs `PROSPER_NULL_PAGE=1`. Without that the
run either faults with a `[nullpage]` report or survives and renders pure black after Accept. With it
the route lands on the title screen (`max_nonblack` 0.0069, held) in roughly **one run in three**, so
repeat any measurement rather than trusting one arm.

```bash
PROSPER_NULL_PAGE=1 \
PROSPER_PAD_SCRIPT=@prosper/scripts/stray-PPSA02101/reach-title-hold.pad \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA02101-app0 \
    --seconds 6 --count 30 --warmup-seconds 140 --timeout 360 \
    --allow-guest-fault --no-stop-after-guest-fault --out <OUTDIR>
```

Frames 09-28 settle at 0.0069 when it lands. It presses Cross twice (t=150 and t=160): the first
accepts calibration, and the second is the margin for a slow boot — on a fast one it lands on the
title menu's already-selected `START GAME`, which is why a window that runs far past t=160 can catch
a load instead of the title.

## Why the CALIBRATION route is anchored on flips, not seconds

This section is about `reach-title-flip.pad`, which despite its name reaches **calibration**, not the
title. The title route is the wall-clock one above; this heading used to claim the opposite and the
export below used to present this file as the title route, both of which contradicted the top of this
same README.

The flip anchor is the right shape *for calibration* because that target is a **steady state** rather
than a moment: a no-input boot reaches calibration at `present_count` ≈ 1764 and then holds it
unchanged (identical nonblack 0.1070 from present 1838 through 3205), because that screen waits for
input. Any anchor after it arrives lands on it however slow the boot was — which is what makes it a
stable oracle.

An earlier attempt to read wall-clock timing as unreliable rested on an **inverted number map**:
~0.11 was taken for the title screen and ~0.006 for a failure, when it is the other way round
(`docs/STRAY_STATUS.md` § Ruled out). Two of the three samples read as failures were the successes —
the third was a black frame, which is a failure under either map. That is why
the wall-clock `reach-title-hold.pad` above is the title route despite this section once arguing no
wall-clock route could work.

Note the unreconciled figure recorded in `reach-title-flip.pad`'s own header: this file's runs
settle at 0.1140 while an earlier set reported 0.1097 for what should be the same screen. Both are
calibration; the gap is not explained.

```bash
# calibration hold -- NOT the title route
export PROSPER_PAD_SCRIPT=@prosper/scripts/stray-PPSA02101/reach-title-flip.pad
```

```bash
export PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct
export PROSPER_PAD_SCRIPT=@prosper/scripts/stray-PPSA02101/reach-first-map.pad
export PROSPER_FILELOG=1          # the progress oracle -- see below
./build-linux/screenshot <DUMP_ROOT>/PPSA02101-app0 --seconds 1.5 --count 280 --out <OUTDIR>
```

## This is the cross-only one

The calibration screen draws `△ Defaults` / `✕ Accept`, and unlike the other two routes landed
alongside it, **the prompt is the whole story**: CROSS accepts, and CROSS carries the main menu
behind it as well. That is worth stating rather than assuming — of the three titles measured in
this batch, this is the only one where the on-screen glyph was sufficient. BALAN names CROSS on a
screen CROSS cannot leave, and Unbound's title screen names CROSS while the screen immediately
after it names SQUARE.

## Reading a run

**Use the file log, not the pixels** — this title loses most of its presents to
[#2932](https://github.com/mattias800/prosper/issues/2932).

| route | assets reached |
| --- | --- |
| default, no input, 150 s | `/app0/hk_project/config/tags` — `hk_project_mainstart` appears on 0 log lines |
| this route, 415 s | **`/app0/hk_project_mainstart` at t ≈ 36.5 s** |

## What this route does not do

It does not produce a rendered world. After the map load the frames are **letterboxed** — black
bars top and bottom, the game's own cinematic framing — with the picture between them a flat
single-colour fill. The guest stays alive and keeps flipping throughout. That is #2932, not the
route, and it is why the title is still rung 2. Phase 3 is written for the world that map load
implies; it has not yet had one to move through.
