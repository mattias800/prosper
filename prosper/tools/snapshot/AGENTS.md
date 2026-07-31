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

### Why the content contract is conjoined, measured on real titles

"Colour count must never be the contract on its own" is not a precaution; it is
a measured result. Two profiled titles settle it by demonstration, and they
matter as a **pair**: in each one, a different metric goes blind, so neither can
be the contract alone.

| title | colour count | non-black coverage |
|---|---|---|
| **Rugrats** (`PPSA23396`) | separates cleanly — menu 56,090 vs gameplay 17,645 | **blind** — title card fully opaque at 1,551 colours |
| **Greak** (`PPSA02849`) | **blind** — 87-colour gap (0.27%) between cinematic peak 32,153 and gameplay floor 32,240 | separates cleanly — letterboxed 0.6667 vs gameplay 1.0000 |

Whichever metric you were about to trust on its own, one of these two titles is
the counter-example. That is the whole argument for conjoining `min_colors`,
`min_nonblack_ratio`, SSIM, `dims` and `min_pixel_changes`.

Profiling the Rugrats route once per second across a whole boot produced both
halves of the argument from one run, and they fail in **opposite** directions:

- **Colour count alone prefers a menu to the game.** The two richest frames of
  the entire run are the GAME MODE selector at 56,071-56,090 distinct colours.
  The best actual gameplay frame reaches 17,645. A colour-only guard would rank
  a menu **3.2x above every frame of the scene it exists to protect**, so it
  would pass a build whose gameplay had collapsed as long as the menu still drew.
- **Coverage alone accepts a nearly colourless screen.** The "BABIES IN
  GAMELAND" level-title card and its fade are **fully opaque — a 1.0 non-black
  ratio — at only 1,551-3,448 colours**. 54 frames of the run reach 0.999
  coverage with as few as 1,551 colours, so a coverage-only guard treats a
  title card as a rendered level.

Greak then supplied the inverse case. Its route passes through a **letterboxed**
level-intro cinematic immediately before gameplay, and that cinematic peaks at
32,153 colours while the gameplay floor is 32,240 — a gap of **87 colours, or
0.27%**. No usable `min_colors` separates them. What does separate them is
coverage: the letterbox bars hold the cinematic at 0.6667 while every gameplay
frame measures exactly 1.0000.

The practical rule: let SSIM decide *scene identity*, and keep `min_colors` as a
gross-collapse floor rather than tuning it to separate two valid scenes. For
Rugrats the tempting floor is ~16,000 — just under the 17,259 gameplay minimum
and just over the 15,030 menu ceiling — but that discrimination is redundant
with SSIM while making the guard fragile against a slightly dimmer healthy
frame. 12,000 was chosen instead: comfortably below the observed gameplay range
and far above every observed failure state (black at 1 colour, logos at most
3,830, title card at most 3,448).

The corollary is that **when the usual discriminator goes blind, the other
threshold has to carry the load, and should be set for that job rather than from
the generic formula**. `greak-gameplay` therefore sets `min_nonblack_ratio` to
0.9 instead of the derived 0.5 (half the lowest reviewed coverage): colour count
provably cannot catch a window drifting into that cinematic, so coverage must,
and the derived 0.5 would have admitted the 0.6667 letterbox. Raising a
threshold this way needs the same evidence as lowering one — here, every
reviewed gameplay frame measured exactly 1.0000 with zero variance across
independent runs, leaving a 10% margin.

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
