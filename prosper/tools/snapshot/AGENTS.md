# AGENTS.md - rendering snapshot tests

Real-game rendering regression guard. Run this whenever a change can affect
rendered output: RDNA2-to-SPIR-V recompilation, AGC/PM4 decoding, render state,
texture detiling, or executor/present behavior.

The current Messenger guard records the richest frame's distinct-color count
across the boot. This catches a render collapsed to a clear while tolerating the
title's run-to-run frame variance.

## Run It

```bash
# From prosper/, with build-linux/boot_trace built for the change.
python3 tools/snapshot/snapshot.py check
python3 tools/snapshot/snapshot.py check messenger-scene
```

On failure, the screenshot and boot log are written to
`tools/snapshot/failures/<name>.{bmp,log}`. Convert a BMP for inspection with:

```bash
python3 -c 'from PIL import Image; Image.open("x.bmp").save("x.png")'
```

## Non-Negotiables

- Local only, never CI. Game dumps and captured imagery must not be committed.
- Only thresholds or pixel hashes live in `snapshots.json`.
- Do not lower `min_colors` merely to make a failing check pass. Investigate the
  regression, or explain an intentional baseline change in the PR.

## Guard Modes

- `min_colors`: run the full configured timeout, inspect every dumped frame,
  and require the richest frame to meet the threshold. `messenger-scene` uses
  this because exact frame hashes are not stable across threaded boots.
- `frame` plus `hash`: target one draw-submit frame. Use `verify <name>` before
  trusting a new exact baseline, then `update <name>` after an intentional pixel
  change.

`RENDER_SCALE` shrinks the framebuffer. Keep `scale` fixed for either mode.

## Adding A Snapshot

Add `name`, `dump`, `scale`, `timeout`, optional `env`, and either:

- a justified `min_colors` threshold for a run-level content guard; or
- `frame` plus an exact `hash`, established with `verify` then `update`.

## Environment

- `PROSPER_GAME_ROOT`: directory holding `*-app0` dumps; defaults to the repo.
- `PROSPER_BOOT_TRACE`: boot_trace path; defaults to `build-linux/boot_trace`.
