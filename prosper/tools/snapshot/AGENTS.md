# AGENTS.md — rendering snapshot tests

Golden-image regression guard for game rendering. **Run this whenever you touch
anything that can change rendered output** (the RDNA2→SPIR-V recompiler, AGC/PM4
decode, render state, texture detile, the executor/present path) — before AND
after your change. A blued title or a mis-decoded texture flips a stored pixel
hash, so you catch the regression instead of shipping it.

This exists because two recompiler fixes for *other* titles silently blued The
Messenger to a full-screen clear (PR #227). Nothing had guarded a real game's
output. Now something does.

## Run it

```bash
# from prosper/ , with build-linux/boot_trace built for your change
python3 tools/snapshot/snapshot.py check            # all snapshots; EXIT 1 on any diff
python3 tools/snapshot/snapshot.py check messenger-title
python3 tools/snapshot/snapshot.py update [name...]  # re-baseline AFTER an intended output change
python3 tools/snapshot/snapshot.py verify [name...]  # capture twice; is the frame deterministic?
python3 tools/snapshot/snapshot.py list
```

`check` failing → the offending screenshot + boot log are saved to
`failures/<name>.{bmp,log}`. Eyeball the BMP
(`python3 -c 'from PIL import Image; Image.open("failures/x.bmp").save("x.png")'`).
If your change **intentionally** alters output, re-run `update` and commit the new
hash in `snapshots.json` — with a note in your PR on why the pixels moved.

## Non-negotiables

- **Local only, NEVER CI.** Game dumps are gitignored and must never be committed.
  Only the pixel **hashes** live in the repo (`snapshots.json`) — a hash is a
  checksum, not game imagery (same rule as [[pr-screenshots]]: no game imagery in
  mainline).
- **`verify` before you trust a new snapshot.** A game boot has threads; if the
  target frame lands in a transient (loading→title) it won't be deterministic.
  Pick F in a **static** window (a title/menu loop re-renders the same composite,
  so its hash is stable). Only F ≤ 59 (and multiples of 10) are dumped.
- **Don't `update` to make a red `check` green.** A `check` failure is either a
  real regression (fix it) or an intended change (re-baseline *and explain it*).
  Silently re-baselining a regression defeats the whole point.

## How a frame is targeted

`RENDER_EVERY=1` renders every draw-carrying submit, so `frame_<F>.bmp` is the
F-th draw submit's render. `RENDER_SCALE` shrinks the framebuffer (faster; the
hash is per-scale, so keep `scale` fixed per snapshot). The tool runs a
uniquely-named copy of `boot_trace` so a concurrent agent's `pkill -x boot_trace`
can't kill your capture.

## Adding a snapshot (e.g. another title / a specific screen)

1. Add an entry to `snapshots.json`: `name`, `dump` (the `*-app0` dir under
   `PROSPER_GAME_ROOT`), `frame`, `scale`, optional per-title `env` (e.g. a title
   that needs extra guest args). Leave `hash` empty.
2. `verify <name>` — must print DETERMINISTIC. If not, move `frame` later / to a
   more static screen.
3. `update <name>` — stores the baseline. Commit `snapshots.json`.

## Env

- `PROSPER_GAME_ROOT`  — dir holding the `*-app0` dumps (default `/mnt/c/Users/matti/repos/ps5ys`)
- `PROSPER_BOOT_TRACE` — boot_trace path (default `<prosper>/build-linux/boot_trace`)
