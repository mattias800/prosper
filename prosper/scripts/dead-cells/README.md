# Dead Cells Routes

Reusable `PROSPER_PAD_SCRIPT` routes for Dead Cells (`PPSA15552`). Run tools
from `prosper/` so the relative paths below resolve:

```bash
PROSPER_PAD_SCRIPT=@scripts/dead-cells/reach-first-gameplay.pad \
  ./build-linux/screenshot /path/PPSA15552-app0 \
  --warmup-seconds 35 --seconds 1 --count 5 --timeout 65 --out shots
```

## Routes

- `reach-first-gameplay.pad`: selects Play/slot 1, starts `PrisonStart`, and
  holds Circle through the skippable opening. It was verified from fresh save
  roots on current master and reaches the controllable Jump tutorial. The
  route is wall-clock anchored because Dead Cells submits tens of thousands of
  loading flips before the menu.
- `reach-first-gameplay-full-render.pad`: performs the same route with input
  delayed until the title appears when every submit is rendered. Use this for
  presented-image regression investigation; it does not depend on a renderer
  warmup. Circle remains held through 240 seconds because synchronous rendering
  has made `PARSEALL` take from about 60 to more than 160 seconds across measured
  builds. The route now survives that throughput variance, but the gameplay
  snapshot baseline remains pending until its retained frames are reviewed.
- `reach-first-gameplay-capture.pad`: uses the same menu input but holds Circle from
  28 through 300 seconds. Use it when synchronous timeline/bundle capture begins during
  level loading; the ordinary six-second hold can expire while capture stalls GPU progress.

The long renderer warmup makes this route practical under llvmpipe, but it can
skip temporal GPU producers. The former post-warmup fullscreen-white image was
caused by that skip and is not normal renderer output (#586). For graphics
investigation, add `PROSPER_RENDER_TARGET_DIM=642x362` to preserve the level's
RTT chain; this is much slower and currently remains in the opening vignette at
the 35-second checkpoint. Do not use the warmup-based fast route as a visual
golden guard. The snapshot matrix uses the full-render route and requires
inspected multi-frame evidence.

## Playable Semantic Checkpoints

### Historical #608 bundle

Do not select the first 738x420 target: it also occurs in the skippable opening. The preserved #608
Jump-tutorial capture used all of these predicates:

```bash
PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM=738x420 \
PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DRAW_INDEX=79:81 \
PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS=90 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_DRAWS=90
```

The draw index is zero-based in the raw semantic sequence. Those #608 runs selected different submit numbers
but replayed the same controllable scene with 86 realized draws, seven realized dispatches, the Jump prompt,
and the full HUD. Nearby cinematic and transition captures were outside this conjunction.

Treat that conjunction as a preserved-capture recipe only.

### Current timeline-v6 checkpoint (#594)

Two independent 90-second native-speed routes selected sustained gameplay from about 29.5 seconds onward with
this conjunction, despite different submit ordinals:

```bash
PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM=738x420 \
PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DRAW_INDEX=80:82 \
PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS=91 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_DRAWS=93 \
PROSPER_GPU_TIMELINE_CAPTURE_MIN_DISPATCHES=8 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_DISPATCHES=8
```

Timeline v6 retains compact target spans, so calibrate and validate the endpoint offline before starting a
large capsule or bundle:

```bash
./build-linux/gpu_timeline run.prgtl --signatures 91:93 8
./build-linux/gpu_timeline run.prgtl --select 738x420 80:82 91:93 8
```

The adjacent target indices 79 and 83, exact 90-draw count, and dispatch counts 7 and 9 all produced zero
matches. The live selector reproduced the offline result and installed a gameplay capsule at its first match.

The faithful playable reference on #608 retained 883 submits; its `5759c125812154dc` hash is historical to the
pre-#611/#615 renderer. A two-submit color-bounded replay still renders `71b84bdfae53933c`. Use
`gpu_replay --bundle-ds-summary` to scan every depth/stencil identity and raw programming variant without
reconstructing resource payloads or invoking Vulkan. Timeline-v5 `--depth-summary 642x362`, enabled during
capture with `PROSPER_GPU_TIMELINE_DEPTH_HASH_DIM=642x362`, found that the stable surface backing changed via
compute: program `0x401aec200` fills the 32 KiB HTILE block with `0xfffffff0` before the scene draws. #611 now
invalidates overlapping persistent Vulkan depth/stencil images on guest GPU writes, restoring the rejected
layers. Dead Cells' four repeated fragment failures were uniform VCCZ-exit light loops; #615 adds a narrowly
proved fragment/vertex structurization path and current gameplay submits realize every semantic draw. Use
`shader_inspect` for raw `PROSPER_SHADER_DUMP` binaries; capture v7 retains bounded raw stages and exact
failure diagnostics for unrealized operations (#618). Capture v8 additionally snapshots exact persistent
Vulkan depth/stencil planes at final-submit entry; export it with `--bundle-final-capsule`, then require its
standalone hash to equal the source bundle before using fast operation-prefix experiments (#569). The
current invalidation-disabled stale-depth A/B is `535256588b67a536` in both the 883-submit source and a
self-contained v8 capsule, with byte-identical BMPs and a 3.3-second standalone replay. The
35-second screenshot warmup still skips color RTT
history and therefore remains an overbright progression diagnostic, not a color-correct gameplay oracle.
