# Dead Cells graphics status

Last updated: 2026-07-13

This is the handoff for the remaining *Dead Cells* gameplay-composition work. The operational route and
selector reference is [`../scripts/dead-cells/README.md`](../scripts/dead-cells/README.md). The active bug is
[#566](https://github.com/mattias800/ps5ys/issues/566).

## Current state

*Dead Cells* boots, passes the splash and menus, loads `PrisonStart`, and reaches the controllable Jump tutorial.
Scene geometry, smoke, silhouettes, the player, the tutorial prompt, and the HUD render. The remaining output is
visibly wrong compared with the game: the world composition is largely grayscale and overexposed instead of the
colored, lit scene.

The original mostly-white repeated-block image is not the current bug. It was caused by beginning diagnostic
rendering after a required 642x362 temporal render-target producer had already run (#586). Do not use a
35-second sparse-render screenshot as a color oracle.

The important fixes already on `master` are:

- mixed graphics/compute execution follows PM4 order (#584);
- compute writes to a depth surface's HTILE allocation invalidate the detached Vulkan depth cache (#611);
- four uniform VCCZ-exit fragment-lighting loops recompile through a narrowly proved structured form (#615);
- failed operations retain bounded raw shaders and exact rejection diagnostics in capture v7 (#618);
- capture v8 checkpoints complete color RTT state and persistent depth/stencil planes, including a source-output
  hash oracle (#569).

Current live gameplay submits after #615 realize all semantic draws. Descriptor validation has not identified a
missing or undersized binding in the exercised frame.

## Evidence boundary

The preserved #608 bundle is useful historical evidence, but it is not a current live-frame oracle. It contains
883 submits (`18165..19047`), resolves all 1,764 temporal image dependencies, and was captured before #615. Its
final submit has four unrealized draws and one unrealized dispatch. Capture-v8 migration represents those five
holes explicitly as `Unknown`; it cannot recover work that was absent from the old artifact.

On the current renderer, that bundle and its exported v8 capsule both produce `fac9ca4cbbba8196`. With depth
invalidation deliberately disabled, both produce `535256588b67a536`. This exact source/capsule equality proves
the checkpoint implementation, but the image still reflects the old capture's missing operations. Do not use it
to conclude which pass is wrong in a post-#615 live submit.

A fresh producer-complete current bundle is therefore the required starting point. Title-derived `.prgtl`,
`.prgcap`, `.prgbundle`, shader, and image artifacts are local and gitignored; none belong in a commit.

## Recreate the current checkpoint

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

Do not begin isolation until `cmp` succeeds and standalone replay reports that its embedded oracle passed. Also
require `gpu_replay --inspect-only` to report zero failed operations for the current endpoint.

## What remains

1. Produce the fresh current bundle and hash-checked v8 capsule above. This removes the five legacy operation
   holes from the debugging baseline.
2. Obtain a hardware/reference capture of the same controllable frame and route state. The image currently
   attached to #566 is from the preceding opening vignette, so it is a qualitative color reference rather than
   a pixel-aligned oracle.
3. Use `gpu_replay --through-operation N --allow-mismatch` to locate the operation where the 642x362 scene first
   loses the expected material/color information. Preserve mixed graphics/compute order; `--draw` is not a
   substitute for a semantic prefix.
4. For the first suspect operation, record its draw/dispatch source, PM4 order, target, shader hashes, resource
   hashes, blend/depth state, and before/after output. Dump only the relevant shader/resource inputs.
5. Implement the missing generic GPU contract, add a synthetic regression, run all CTests and both Messenger and
   Dead Cells snapshot guards, then repeat the source/capsule equality check.

The likely fault domain is now a scene color/light composition contract, but shader execution, resource decode,
format conversion, and blending remain hypotheses until the current producer-complete prefix identifies the
first bad pass.

## Tooling note

`--bundle-final-capsule` is currently the strict path: it snapshots the complete live RTT and depth/stencil cache
at final-submit entry and embeds the source output oracle. A standalone capsule selected directly by the timeline
is useful for state inspection, but it does not by itself prove complete renderer history or source-image
equality. Prefer improving the capture/replay diagnostics when an investigation would otherwise require repeated
full-title runs.
