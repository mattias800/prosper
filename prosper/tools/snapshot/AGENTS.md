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

1. Write the whole candidate entry, **including `_note`**, and set `review` to
   `pending`. Then run `snapshot.py verify NAME`.
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
5. Replace the generic `review` string that `update` wrote with the factual note
   describing what was actually inspected.
6. Run `snapshot.py check NAME` after approval.

See `README.md` for every manifest field and the current title matrix.

### Three ordering traps that silently produce a bad baseline

Each of these leaves a guard that *looks* adopted. None of them fails loudly, so
follow the order above rather than the obvious one.

- **The factual `review` note must be written after `update --reviewed`, not
  before.** `approve_content_candidate` unconditionally overwrites `review` with
  a generic "Approved N composited images ... visually confirmed" string. A note
  written before approval is destroyed by the approval itself, and the guard then
  carries boilerplate that records no evidence while still satisfying `check`'s
  "not pending" test.
- **`_note` must be set before `verify`, because it is part of the candidate
  fingerprint.** `entry_fingerprint` excludes only `hash`, `dims`, `review`,
  `structural_references`, `perceptual_references`, and `min_nonblack_ratio`;
  every other key, `_note` included, is hashed. Adding or editing `_note` after
  `verify` makes `update` refuse with "snapshot configuration changed after
  verify", costing a full two-capture rerun.
- **`min_content_match_ratio` applies to every frame in the window, not to the
  qualifying ones.** The requirement is
  `max(min_structural_matches, ceil(total_window_frames * ratio))`, so at the
  0.75 default, three quarters of *all* window frames must clear both SSIM and
  non-black coverage. A window that merely contains good gameplay but also
  straddles a load, a fade, or a transition will fail on a perfectly healthy run.
  Place the window entirely inside the settled state and confirm the margin with
  `profile_route.py` instead of assuming it.

## Environment

- `PROSPER_GAME_ROOT`: directory holding `*-app0` dumps; defaults to the repo.
- `PROSPER_BOOT_TRACE`: `boot_trace` path; defaults to `build-linux/boot_trace`.
- `PROSPER_SCREENSHOT`: presented-capture frontend; defaults to
  `build-linux/screenshot`.
