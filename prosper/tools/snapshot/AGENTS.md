# AGENTS.md - rendering snapshot tests

This is the local real-game rendering regression inventory. Run and review the
full matrix before every release. It is not a day-to-day development or merge
gate: a PR author may skip snapshots entirely during long iterations or run only
useful focused guards, unless the task explicitly requires a run. Snapshot results
do not define whether master or a PR is acceptable; either may contain a detected
regression, including an intentional cross-title
tradeoff from an otherwise correct fix. Release review decides whether to fix the
regression or explicitly accept and document it before publishing artifacts.

```bash
# From prosper/, with build-linux/boot_trace built from the change.
python3 tools/snapshot/snapshot.py check
python3 tools/snapshot/snapshot.py check messenger-scene
python3 tools/snapshot/snapshot.py check dead-cells-splash
python3 tools/snapshot/snapshot.py check blasphemous2-gameplay
```

## Contract

- Local only. Game dumps and captured imagery must never be committed.
- Gameplay guards are deliberately coarse. Subtle pixel changes may be valid or
  improvements; use tolerant average/difference hashes to detect major collapse,
  missing layers, lost progression, and wrong dimensions without freezing every pixel.
- A content guard examines all frames in a gameplay-only evidence window and
  requires multiple qualifying frames. Do not treat one richest frame, a logo,
  a menu, or a static screen as proof of gameplay.
- `check` is fully automated. On failure, inspect the representative PNGs and
  log under `tools/snapshot/failures/` before changing code or thresholds.
- Do not lower a threshold merely to pass. Explain intentional contract changes
  and repeat the baseline-review workflow below.

## New Or Changed Baselines

Baseline evidence requires visual review even though routine regression runs do
not. This prevents checksums or thresholds from blessing black output or the
wrong scene.

1. Set `review` to `pending` and run `snapshot.py verify NAME`.
2. Inspect every image retained from both independent runs in
   `tools/snapshot/review/NAME/`. Confirm the intended gameplay state, expected
   layers, and progression across multiple timestamps.
   Use the adjacent `runN-evidence.json` when a bad frame needs correlation
   with its flip count, front-buffer index, or renderer publication.
3. For content mode, choose a conservative `min_colors`, require at least two
   qualifying frames, and add `min_pixel_changes` for moving routes.
4. Run `snapshot.py update NAME --reviewed` only after every image is accepted.
   Content mode records reviewed luminance references for SSIM plus a
   conservative non-black coverage floor; exact mode records the identical
   pixel hash. Never use exact hashes for threaded gameplay merely because one
   run happened to be stable.
5. Run `snapshot.py check NAME` after approval.

See `README.md` for every manifest field and the current title matrix.

## Environment

- `PROSPER_GAME_ROOT`: directory holding `*-app0` dumps; defaults to the repo.
- `PROSPER_BOOT_TRACE`: `boot_trace` path; defaults to `build-linux/boot_trace`.
- `PROSPER_SCREENSHOT`: presented-capture frontend; defaults to
  `build-linux/screenshot`.
