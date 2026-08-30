# Stray — `PPSA02101`

`reach-first-map.pad` drives the boot past the **brightness-calibration screen** to the first map
load.

`reach-title-flip.pad` stops one screen earlier — it reaches the **title screen** and holds there —
and it is the one to use for anything that has to be reproducible.

## Why the title route is anchored on flips, not seconds

A wall-clock route cannot hit this title reliably: an earlier 150 s/160 s version fired before
calibration on slower boots and reached the title only sometimes (`max_nonblack` 0.1095 once, then
0.0063 / 0.0000 / 0.0063 on three unmodified reruns — a single lucky sample adopted as a baseline is
how that hour was lost). The flip anchor works because the target is a **steady state** rather than a
moment: a no-input boot reaches calibration at `present_count` ≈ 1764 and then holds it unchanged
(identical nonblack 0.1070 from present 1838 through 3205), because that screen waits for input. Any
anchor after it arrives lands on it however slow the boot was.

```bash
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
