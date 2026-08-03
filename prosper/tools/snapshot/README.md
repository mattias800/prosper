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

`check` and `verify` hold a cooperative lock in the clone's Git common directory,
so snapshot runs from sibling worktrees wait instead of contending for the GPU
and shifting long-boot capture windows. The waiter prints the owning PID and
command. Set `PROSPER_SNAPSHOT_LOCK` to share a lock across separate clones, or
set `PROSPER_SNAPSHOT_NO_LOCK=1` only when concurrent captures are intentional.

Snapshot rendering uses the shared renderer's hardware-first Vulkan selection: discrete, integrated,
and virtual GPUs are preferred over CPU devices, with llvmpipe used only when no usable GPU is exposed.
The selected device is printed in each retained run log.

Rendered captures follow `prosper-app`'s opaque-swapchain contract: the screenshot
export boundary normalizes alpha to 255 before writing the PNG, CRC, colour and
coverage metrics, perceptual signatures, or stale-pixel classification. A render
target's alpha channel is not desktop transparency. Raw scanout captures remain
byte-for-byte guest evidence, including alpha.

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
- `min_consecutive_content_matches`: alternative for a phase-variable checkpoint
  inside a broad boot profile. It requires this many adjacent frames to jointly
  satisfy colour, SSIM, non-black coverage, and dimensions, so scattered matches
  from an intro or attract loop cannot add up to a pass. It is mutually exclusive
  with `min_content_match_ratio`; all existing ratio guards retain their original
  semantics. This mode needs existing reviewed `structural_references` to seed
  plateau identification. `verify` first finds each run's plateau against that
  seed, then requires A-plateau-only references to pass run B and B-plateau-only
  references to pass run A before it can produce an adoptable candidate. Saved
  review and candidate references come only from those identified plateaus. When
  `min_pixel_changes` is set, those changes must occur inside the identified
  plateau; animation in an unrelated intro cannot hide a frozen checkpoint.
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

- `messenger-scene`: reviewed first-level gameplay guard with a late recovery
  sequence for cold full-render processes that miss early menu pad polls.
- `evergate-title`: reviewed animated title-screen guard using a neutral
  connected-controller route to retain the complete logo, prompt, and scene composition.
- `blue-prince-title`: reviewed fresh-save title-screen-only guard; it does not
  navigate into New Game or claim Blue Prince gameplay coverage.
- `dead-cells-gameplay`: reviewed full-render guard for the controllable
  Prisoners' Quarters Primary weapon tutorial.
- `blasphemous2-gameplay`: reviewed native-resolution fresh-save guard for
  multiple moving frames in the first playable area.
- `terminator-boot`: reviewed fresh-save guard for Terminator 2D's boot intro
  and settled main menu.
- `alexkidd-gameplay`: reviewed fresh-save guard for Alex Kidd in Miracle World
  DX's first level. Its window opens after the route's last pad press, so it
  samples a settled scene whose only motion is cloud drift and enemy animation.
- `gris-gameplay`: reviewed fresh-save guard for GRIS's opening ink-ground scene.
  Both numeric floors are blind on this title at once — its title screen is 17x
  richer in colour than its gameplay, and non-black coverage is exactly 1.0000 for
  logos, title, intro and gameplay alike — so SSIM does all the discrimination.
- `cobra-gameplay`: reviewed fresh-save guard for Space Adventure Cobra's opening
  desert tutorial combat. Its coverage floor is deliberately low: gameplay coverage
  (0.82-0.90) is *lower* than its menus (0.99-1.00), because the bottom of the frame
  is the dark underside of a walkway, so a high floor would reject the guarded state.
- `worms-armageddon-gameplay`: reviewed fresh-save guard for a fixed-map Training
  level. It uses `reach-training-gameplay.pad` rather than the published Quickstart
  route because Quickstart regenerates its terrain every run; see that entry's
  `_note` and trap 36 in `docs/GAME_COMPAT_ORCHESTRATION.md`.

## Choosing A Window And Thresholds

A candidate entry cannot report the range its thresholds should sit below:
`verify` only accepts or rejects the numbers it is given, and a rejected
candidate saves no evidence to learn from. `profile_route.py` runs one capture
with a permissive candidate and prints every sample, so the evidence window and
`min_colors` come from the title's measured behaviour:

```bash
python3 tools/snapshot/profile_route.py '{"name":"probe","dump":"PPSA02664-app0",
  "scale":4,"timeout":150,"sample_seconds":1,"capture_after_seconds":0,
  "capture_before_seconds":150,"pad_script":"scripts/ppsa02664/reach-first-gameplay.pad",
  "savedata_policy":"fresh","env":{},"min_colors":1}' ~/route-profile
```

Prefer a window that begins well after the guarded state is reached and, where
the route allows, after its last input: a settled scene keeps SSIM comfortably
above the threshold across animation phases, whereas a window straddling a load
or transition makes the guard fail on healthy frames. Confirm the margin rather
than assuming it — for `alexkidd-gameplay` the worst of 100 reviewed frames
scores 0.93 against the reviewed references, against a 0.85 floor.

For a checkpoint whose absolute wall-clock phase moves but whose intended state
holds, profile the whole bounded boot and use `min_consecutive_content_matches`
rather than narrowing the window until it happens to fit one run. The required
run must have real margin in both independent profiles. Explicit intro/menu/
attract negative controls still have to be scored offline; a consecutive count
does not identify the intended scene by itself.

## Environment

- `PROSPER_GAME_ROOT`: directory holding `*-app0` dumps; defaults to the repo.
- `PROSPER_BOOT_TRACE`: `boot_trace` path; defaults to `build-linux/boot_trace`.
- `PROSPER_SCREENSHOT`: presented-capture frontend; defaults to
  `build-linux/screenshot`.
- `PROSPER_SNAPSHOT_LOCK`: lockfile override, useful for coordinating separate clones.
- `PROSPER_SNAPSHOT_NO_LOCK=1`: opt out of capture serialization explicitly.
