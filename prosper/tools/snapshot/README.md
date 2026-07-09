# Rendering snapshot tests (golden-image, run locally)

Catches rendering regressions in real games: a change that blanks a title, breaks
a shader recompile, or mis-decodes a texture flips a stored pixel hash. Any agent
can run it before/after a change.

**Local only — never CI.** The game dumps are gitignored and must not be
committed, so this can't run in CI. Only the small pixel **hashes** live in the
repo (`snapshots.json`) — a hash is a checksum, not game imagery.

## Run

```bash
# from prosper/ (build boot_trace first: build-linux/boot_trace)
python3 tools/snapshot/snapshot.py check           # all snapshots; exit 1 on any diff
python3 tools/snapshot/snapshot.py check messenger-title
python3 tools/snapshot/snapshot.py update           # (re)capture baselines after an INTENDED change
python3 tools/snapshot/snapshot.py verify           # capture twice; confirm a frame is deterministic
python3 tools/snapshot/snapshot.py list
```

On a mismatch, `check` writes the offending screenshot + boot log to
`tools/snapshot/failures/<name>.{bmp,log}` and returns non-zero. Convert the BMP
to PNG to eyeball it (`python3 -c 'from PIL import Image; Image.open("x.bmp").save("x.png")'`).

## How it targets a frame

`RENDER_EVERY=1` renders every draw-carrying submit, so `frame_<F>.bmp` is the
F-th draw submit's render. Pick **F in a stable-content window** — a static
title/menu loop re-renders the same composite each submit, so its hash is stable.
Always `verify` a new snapshot's frame before trusting its baseline; if it's
NON-DETERMINISTIC, move F later (or to a more static screen). Only F ≤ 59 (or
multiples of 10) are dumped by the renderer.

## Adding a snapshot

Add an entry to `snapshots.json` (`name`, `dump` = the `*-app0` dir under
`PROSPER_GAME_ROOT`, `frame`, `scale`, optional per-title `env`), then
`verify` it, then `update` to store the baseline hash.

## Env

- `PROSPER_GAME_ROOT`  — dir holding the `*-app0` dumps (default `/mnt/c/Users/matti/repos/ps5ys`)
- `PROSPER_BOOT_TRACE` — boot_trace path (default `<prosper>/build-linux/boot_trace`)
