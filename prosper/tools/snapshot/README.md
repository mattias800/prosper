# Rendering Regression Snapshots

This local tool catches real-game rendering regressions. A snapshot can use an
exact frame hash, or a run-level content metric for titles whose threaded boots
do not land on a deterministic frame.

Game dumps and screenshots are local and gitignored, so this does not run in
CI. Only hashes or content thresholds live in `snapshots.json`.

## Run

```bash
# From prosper/, after building build-linux/boot_trace.
python3 tools/snapshot/snapshot.py check
python3 tools/snapshot/snapshot.py check messenger-scene
```

On failure, `check` writes the screenshot and boot log to
`tools/snapshot/failures/<name>.{bmp,log}` and exits nonzero.

## Guard Modes

- `min_colors`: run for the configured timeout and require the richest rendered
  frame to reach a distinct-color threshold. The current `messenger-scene`
  guard uses this mode and tolerates frame timing variance.
- `frame` plus `hash`: compare one targeted draw-submit frame exactly. Run
  `verify <name>` before establishing its baseline with `update <name>`.

## Adding A Snapshot

Add an entry to `snapshots.json` with `name`, `dump`, `scale`, `timeout`, optional
per-title `env`, and either a justified `min_colors` threshold or a verified
`frame`/`hash` baseline.

## Environment

- `PROSPER_GAME_ROOT`: directory holding `*-app0` dumps; defaults to the repo.
- `PROSPER_BOOT_TRACE`: boot_trace path; defaults to `build-linux/boot_trace`.
