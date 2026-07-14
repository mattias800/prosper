# Rendering Regression Snapshots

This local tool boots real game dumps and catches major rendering failures. It is
deliberately not a pixel-perfection gate for timing-sensitive gameplay: small
pixel differences may be valid or improved output. Content guards instead check
several frames from a routed gameplay window for scene richness, structural
likeness to reviewed references, stable dimensions, non-black coverage, and optional visible progression.

Game dumps and captured images are local and gitignored, so this cannot run in
CI. Only reviewed 16x9 luminance signatures, diagnostic hashes, coarse content
thresholds, routes, and review notes live in `snapshots.json`.

## Run

Build `build-linux/boot_trace`, then run from `prosper/`:

```bash
# Run the whole real-game matrix, or one affected title.
python3 tools/snapshot/snapshot.py check
python3 tools/snapshot/snapshot.py check messenger-scene
python3 tools/snapshot/snapshot.py check dead-cells-gameplay
python3 tools/snapshot/snapshot.py check blasphemous2-gameplay
```

Run the relevant checks after changes to shader recompilation, AGC/PM4 decode,
render state, texture decode/detiling, executor behavior, or presentation.
`check` is automated and nonzero on failure. A content failure retains several
representative presented PNGs, the boot log, and structured per-frame evidence
in `tools/snapshot/failures/`.

## Guard Modes

### Routed content guard

This is the default for gameplay and other timing-sensitive scenes. A manifest
entry can define:

- `pad_script`: input route relative to `prosper/`.
- `savedata_policy`: `fresh` for an isolated temporary save, or `preserve`.
- `capture_after_seconds` / `capture_before_seconds`: evidence window. Put this
  after the route reaches the state being protected; logos and menus are not
  gameplay evidence.
- `min_colors`: coarse per-frame richness floor.
- `min_qualifying_frames`: number of frames that must meet that floor; defaults
  to two, so one lucky frame cannot pass the guard.
- `min_pixel_changes`: required changes between consecutive captured frames.
  Use this when movement or animation is part of the route.
- `structural_references`: 16x9 luminance signatures generated only from
  visually approved evidence. Average/difference hashes are retained beside
  them for diagnostics, but do not decide pass/fail.
- `min_structural_similarity`: minimum SSIM against the closest reviewed
  signature; `0.85` is the conservative default. SSIM preserves broad spatial
  structure while ignoring exact raster detail.
- `min_structural_matches`: number of current frames that must resemble a
  reviewed reference.
- `min_content_match_ratio`: fraction of all evidence-window frames that must
  meet both structural and non-black contracts; defaults to `0.75`. This keeps
  a few good frames from hiding an intermittently broken renderer.
- `min_nonblack_ratio`: conservative coverage floor generated during baseline
  approval at half the lowest reviewed coverage. It rejects blank output
  without requiring a normally dark game to be bright.
- `min_nonblack_matches`: number of frames that must meet that coverage floor;
  defaults to `min_qualifying_frames`.
- `dims`: expected scaled framebuffer dimensions.
- `review`: factual record of the evidence that a person inspected when the
  guard was introduced or materially changed.

The checker measures every complete frame in the evidence window. It does not
compare exact pixels. It computes the established Structural Similarity Index
(SSIM) over a downsampled luminance grid, so it tolerates subtle raster changes
while still detecting black output, missing scene layers that alter broad
layout, a route that no longer progresses, or wrong framebuffer dimensions.
CRC32 and perceptual hashes identify samples for diagnostics but are not the
content guard's pass/fail oracle.

### Exact frame guard

An entry with `frame` and `hash` compares one targeted draw-submit frame. Use
this only when two independent captures prove that the frame is deterministic.
Exact mode is appropriate for stable synthetic or static checkpoints, not
threaded gameplay.

## Baseline Review Is Mandatory

A checksum or color threshold is not trustworthy merely because it passes. A
logo, menu, or detailed error screen can have many colors; an exact hash can
faithfully preserve a black frame. Every new or changed baseline must be tied to
inspected image evidence:

1. Add or change the candidate entry with `"review": "pending"`.
2. Run `python3 tools/snapshot/snapshot.py verify NAME`. It performs two
   independent full captures and retains eight temporally spread images per run under
   `tools/snapshot/review/NAME/`. Each run also retains an evidence JSON file
   with every sample's flip count, renderer sequence, front-buffer index, CRC,
   coverage, and structural signature.
3. Inspect every retained image from both runs. Confirm the intended scene,
   expected world layers/HUD/player, and progression across multiple moments.
4. For a content guard, confirm the threshold has a conservative margin below
   the observed valid range. Never lower a threshold just to make a failure pass.
5. After inspecting every image, run
   `python3 tools/snapshot/snapshot.py update NAME --reviewed`. Content mode
   records the reviewed structural references and a conservative non-black
   floor; exact mode records the verified pixel hash. `update` refuses stale,
   incomplete, or unverified candidates.
6. Run `check NAME` once more. Commit only the manifest and route, never the
   local review/failure images or game data.

Repeat this review when the route, evidence window, expected scene, dimensions,
hash, or threshold changes materially. Ordinary `check` runs need no manual
inspection unless they fail.

## Current Matrix

- `messenger-scene`: broad content guard for a timing-sensitive rendered scene.
- `dead-cells-gameplay`: provisional full-render route toward the controllable
  Jump tutorial. Baseline approval is blocked on repeatably clearing level loading.
- `blasphemous2-gameplay`: native-resolution fresh-save routed guard for
  multiple moving frames in the first playable room.

## Environment

- `PROSPER_GAME_ROOT`: directory holding `*-app0` dumps; defaults to the repo.
- `PROSPER_BOOT_TRACE`: `boot_trace` path; defaults to `build-linux/boot_trace`.
- `PROSPER_SCREENSHOT`: presented-capture frontend; defaults to
  `build-linux/screenshot`.
