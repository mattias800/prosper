# Unbound: Worlds Apart — `PPSA03274`

`reach-first-level.pad` drives the boot past the **4K title screen** and the **intro cinematic** to
the first level's map load.

```bash
export PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct
export PROSPER_PAD_SCRIPT=@prosper/scripts/unbound-PPSA03274/reach-first-level.pad
export PROSPER_FILELOG=1          # the progress oracle -- see below
./build-linux/screenshot <DUMP_ROOT>/PPSA03274-app0 --seconds 1.5 --count 230 --out <OUTDIR>
```

## Two screens, two different buttons

The title screen draws a lone `✕` glyph and CROSS is right for it. The **intro cinematic that
follows draws its own prompt, and it reads `Press ▢ to skip` — SQUARE.** A cross-only ladder (110
presses, 330 s) reached the cinematic and then sat on it for the remaining 180 s with that SQUARE
prompt still on screen. Adding SQUARE at t = 78 skips it and the first level loads immediately
after.

The skip path is visible in the log as the guest's own `sceAvPlayerSetTrickSpeed`
(NID `av8Z++94rs0`), which is currently unregistered and answers 0.

## Reading a run

**Use the file log, not the pixels** — this title loses most of its presents to
[#2932](https://github.com/mattias800/prosper/issues/2932). The oracle has a negative control,
which is what makes it worth quoting:

| route | assets reached |
| --- | --- |
| default, no input, 120 s | `movies/screen_village_logo.mp4` — and nothing after it |
| this route, 330 s | `movies/mainmenubg.mp4` t~3.5 → `movies/intro_01.mp4` t~7.0 → **`/app0/normalvillage` t~78.0** |

`normalvillage` is the first level. It reproduced at t = 78.01 s on two independent runs and is
absent from every default run measured.

## What this route does not do

It does not produce a rendered level. After the map load the guest stays alive and the renderer
keeps producing frames (flips 3146 → 7812, rendered frames 3258 → 12811 over the following 250 s),
but **175 of the 178 samples after the load are a flat two-tone clear** with exactly half the
pixels non-black. That is #2932, not the route — which is why the title is still rung 2. Phase 3
is written for the world that map load implies; it has not yet had one to move through.
