# Dead Cells graphics status and regression workflow

Last updated: 2026-07-15

The operational route and selector reference is
[`../scripts/dead-cells/README.md`](../scripts/dead-cells/README.md). The gameplay-composition bug
[#566](https://github.com/mattias800/ps5ys/issues/566) was fixed by #626; this document retains the exact
checkpoint recipe as a regression and future-investigation workflow.
The current full-render progression investigation and its behavior-equivalence rows are documented in
[`DEAD_CELLS_PROGRESSION_MATRIX.md`](DEAD_CELLS_PROGRESSION_MATRIX.md) and tracked by
[#723](https://github.com/mattias800/ps5ys/issues/723). The former native-Windows render-disabled divergence is
tracked by [#768](https://github.com/mattias800/ps5ys/issues/768).

## Current state

*Dead Cells* boots, passes the splash and menus, and loads `PrisonStart`. WSL/Linux native-speed and sampled-graphics
routes reach the controllable Jump tutorial and have supplied full-color gameplay captures. Native Windows now also
reaches the sustained 92-94 draw / 8-dispatch gameplay state in the no-GPU-work progression control. #768's former
356-390-draw loop was not a renderer stall: `sceSaveDataDialogUpdateStatus` was unimplemented and returned `NONE`
more than a thousand times while the game waited for `FINISHED`. The separate full synchronous-render throughput
and progression gap remains #723. After renderer PRs #767 and #769, the native-Windows compute-enabled,
render-disabled control also reaches the sustained gameplay signature; its earlier first-dispatch process exit is
not reproducible on current `master`.

The scene renders in full color, including geometry, lighting, smoke, silhouettes, the player, terrain, effects,
the tutorial prompt, and the HUD. The final grayscale-world root cause was the fragment recompiler selecting the
first color export from shaders that emit MRT3..MRT0; MRT0 now feeds the backend's single color attachment (#626).
The same change set recovers a separately dropped format-copy dispatch by resolving its directly placed
destination buffer descriptor at s4.

Startup and progression are stable after implementing both AGC resource-registration output queries. The former
success-only `QueryResourceRegistrationUserMemoryRequirements` stub left Dead Cells' stack value untouched and
occasionally requested a multi-gigabyte texture-pool allocation (#660). A native-speed fresh-save matrix improved
from 7/10 gameplay matches, two texture-memory crashes, and one timeout to 20/20 gameplay matches with no crash or
timeout. The issue's 120-second live-render screenshot route also completes without the former lavapipe fault.

The original mostly-white repeated-block image is not the current bug. It was caused by beginning diagnostic
rendering after a required 642x362 temporal render-target producer had already run (#586). Do not use a
35-second sparse-render screenshot as a color oracle.

The important fixes already on `master` are:

- mixed graphics/compute execution follows PM4 order (#584);
- compute writes to a depth surface's HTILE allocation invalidate the detached Vulkan depth cache (#611);
- four uniform VCCZ-exit fragment-lighting loops recompile through a narrowly proved structured form (#615);
- failed operations retain bounded raw shaders and exact rejection diagnostics in capture v7 (#618);
- capture v8 checkpoints complete color RTT state and persistent depth/stencil planes, including a source-output
  hash oracle (#569);
- directly placed compute buffer destinations resolve, so the current scene realizes all eight dispatches (#626);
- MRT0, rather than the first-emitted non-color G-buffer plane, supplies the visible fragment output (#626);
- AGC resource-registration memory requirements always initialize their output instead of leaking stack data into
  the game's texture-pool allocation size (#660).

The producer-complete post-#626 checkpoint realizes every semantic draw and all eight dispatches. Descriptor
validation has not identified a missing or undersized binding in the exercised frame.

## Evidence boundary

The preserved #608 bundle is useful historical evidence, but it is not a current live-frame oracle. It contains
883 submits (`18165..19047`), resolves all 1,764 temporal image dependencies, and was captured before #615. Its
final submit has four unrealized draws and one unrealized dispatch. Capture-v8 migration represents those five
holes explicitly as `Unknown`; it cannot recover work that was absent from the old artifact.

On the current renderer, that bundle and its exported v8 capsule both produce `fac9ca4cbbba8196`. With depth
invalidation deliberately disabled, both produce `535256588b67a536`. This exact source/capsule equality proves
the checkpoint implementation, but the image still reflects the old capture's missing operations. Do not use it
to conclude which pass is wrong in a post-#615 live submit.

A fresh producer-complete current bundle is the required starting point for any new regression or deeper-scene
investigation. Title-derived `.prgtl`,
`.prgcap`, `.prgbundle`, shader, and image artifacts are local and gitignored; none belong in a commit.

## Recreate the regression checkpoint

Build in WSL from the repository root. Adjust the dump path if needed:

```bash
cd prosper
cmake -S . -B build-linux \
  -DGAME_DUMP=/mnt/c/Users/matti/repos/ps5ys/PPSA15552-app0
cmake --build build-linux -j8 --target boot_trace gpu_timeline gpu_replay
```

Capture a rolling producer-time bundle. Let `boot_trace` exit through
`PROSPER_GPU_TIMELINE_EXIT_AFTER_CAPTURE`; a wall-clock timeout can stop after level parsing but before the
semantic endpoint is installed.

```bash
mkdir -p /tmp/dead-cells-current-save

PROSPER_CAPTURE_TITLE=PPSA15552 \
PROSPER_SAVEDATA_DIR=/tmp/dead-cells-current-save \
PROSPER_PAD_SCRIPT=@scripts/dead-cells/reach-first-gameplay-capture.pad \
PROSPER_GPU_TIMELINE=/tmp/dead-cells-current.prgtl \
PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT=1 \
PROSPER_GPU_TIMELINE_CAPTURE=/tmp/dead-cells-current.prgcap \
PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE=/tmp/dead-cells-current.prgbundle \
PROSPER_GPU_TIMELINE_CAPTURE_DEPTH=1000 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_UNIQUE_MB=1024 \
PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM=642x362 \
PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM=738x420 \
PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DRAW_INDEX=80:82 \
PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS=91 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_DRAWS=93 \
PROSPER_GPU_TIMELINE_CAPTURE_MIN_DISPATCHES=8 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_DISPATCHES=8 \
PROSPER_GPU_TIMELINE_EXIT_AFTER_CAPTURE=1 \
  ./build-linux/boot_trace \
  /mnt/c/Users/matti/repos/ps5ys/PPSA15552-app0 \
  > /tmp/dead-cells-current.log 2>&1
```

Confirm the timeline selected the intended scene:

```bash
./build-linux/gpu_timeline /tmp/dead-cells-current.prgtl \
  --select 738x420 80:82 91:93 8
```

Replay the complete source once and export the exact final checkpoint:

```bash
./build-linux/gpu_replay \
  --bundle /tmp/dead-cells-current.prgbundle \
  --bundle-final-capsule /tmp/dead-cells-current-v8.prgcap \
  /tmp/dead-cells-current-source.bmp

./build-linux/gpu_replay \
  /tmp/dead-cells-current-v8.prgcap \
  /tmp/dead-cells-current-standalone.bmp

cmp /tmp/dead-cells-current-source.bmp \
    /tmp/dead-cells-current-standalone.bmp
```

Do not use a checkpoint as an oracle until `cmp` succeeds and standalone replay reports that its embedded oracle
passed. Also require `gpu_replay --inspect-only` to report zero failed operations for the current endpoint.

## What remains

1. Keep the deterministic gameplay route and source/capsule equality check green as shared GPU work lands.
2. Extend routed playability and checkpoint coverage beyond the first tutorial scene and into later rooms.
3. Obtain a pixel-aligned hardware capture when exact visual comparison is needed; the #566 reference is from the
   preceding opening vignette and remains a qualitative rather than pixel-exact oracle.
4. For any new visual regression, use semantic operation prefixes to name the first divergent pass before changing
   the renderer. Record its draw/dispatch source, PM4 order, target, shader and resource hashes, and fixed-function
   state, then add a synthetic contract test for the generic fix.

There is no known remaining #566 composition defect in the exercised checkpoint. Generic GPU limitations and
tooling follow-ups should be tracked as separate issues rather than reopening the completed grayscale investigation.

## Tooling note

`--bundle-final-capsule` is currently the strict path: it snapshots the complete live RTT and depth/stencil cache
at final-submit entry and embeds the source output oracle. A standalone capsule selected directly by the timeline
is useful for state inspection, but it does not by itself prove complete renderer history or source-image
equality. Prefer improving the capture/replay diagnostics when an investigation would otherwise require repeated
full-title runs.
