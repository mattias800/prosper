# BALAN WONDERWORLD — `PPSA02058`

`reach-title-and-prologue.pad` drives the first-boot flow past the **language-select menu** to the
title screen, the main menu and the opening story cutscene.

```bash
export PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct
export PROSPER_PAD_SCRIPT=@prosper/scripts/balan-PPSA02058/reach-title-and-prologue.pad
export PROSPER_FILELOG=1          # the progress oracle -- see below
./build-linux/screenshot <DUMP_ROOT>/PPSA02058-app0 --seconds 1.5 --count 260 --out <OUTDIR>
```

## What the route is for

The screen the title stops on names CROSS on its own prompt bar, and **CROSS alone never leaves
it**. Pressing it raises a modal — *"Are you sure you want to change the game language to
English?"* — and that modal, not the grid, is the gate. It is answered by **DOWN then CROSS**.

Measured on master `6cf7e075`: 109 CROSS presses over 330 s produced no asset load past
`game.locres`, identical to a default no-input run. A 10-arm probe (one 30 s window per candidate
button) put the breakout in the `down` window after `left` and `right` had both failed; re-run with
CROSS+DOWN alone, `/app0/title` opens at **t = 15.5 s**.

`DOWN` is deliberately confined to phase 1. On the main menu that follows it walks the cursor into
*Credits*, and the route keeps no button after t = 37 that could commit such a selection.

## Reading a run

**Use the file log, not the pixels.** This title interleaves its good frames with the
[#2932](https://github.com/mattias800/prosper/issues/2932) wrong composite — a flat white 4K clear
— so any single sample is unreliable in both directions. `open '/app0/title'` is unambiguous and
is absent from every default run measured.

| marker | meaning |
| --- | --- |
| `open '/app0/…/game.locres'` | still on the language grid |
| `open '/app0/title'` | the modal has been answered — the gate this route exists to open |

The **title screen and main menu are engine-rendered** — they appear from t ≈ 22 s. The
**opening cutscene is a decoded 4K H.264 movie**: at t ≈ 125.6 s two access-unit decoders open
(VA-API), first picture `3840x2160 NV12`, and 3072 access units / 3070 pictures follow. Every rich
frame in the run postdates that line. Reproduced on two independent runs.

> An earlier revision of this file claimed the opposite — "rendered in engine, no video decode of
> any kind" — on the strength of a grep for `[videodec2]` that found nothing. **The tag prosper
> actually prints is `[vdec2]`.** The zero was the grep's, not the title's. Search for the tag the
> code emits, and check a positive control before believing a clean zero.
