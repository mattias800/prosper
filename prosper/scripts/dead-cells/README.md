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
- `reach-first-gameplay-capture.pad`: uses the same menu input but holds Circle from
  28 through 300 seconds. Use it when synchronous timeline/bundle capture begins during
  level loading; the ordinary six-second hold can expire while capture stalls GPU progress.

The long renderer warmup makes this route practical under llvmpipe, but it can
skip temporal GPU producers. The former post-warmup fullscreen-white image was
caused by that skip and is not normal renderer output (#586). For graphics
investigation, add `PROSPER_RENDER_TARGET_DIM=642x362` to preserve the level's
RTT chain; this is much slower and currently remains in the opening vignette at
the 35-second checkpoint. Do not use the fast route as a visual golden guard.

## Playable semantic checkpoint

Do not select the first 738x420 target: it also occurs in the skippable opening. The validated Jump-tutorial
endpoint combines all of these predicates:

```bash
PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM=738x420 \
PROSPER_GPU_TIMELINE_CAPTURE_TARGET_DRAW_INDEX=79:81 \
PROSPER_GPU_TIMELINE_CAPTURE_MIN_DRAWS=90 \
PROSPER_GPU_TIMELINE_CAPTURE_MAX_DRAWS=90
```

The draw index is zero-based in the raw semantic sequence. Fresh native-speed runs selected different submit
numbers but replayed the same controllable scene with 86 realized draws, seven realized dispatches, the Jump
prompt, and the full HUD. Nearby cinematic and transition captures are explicitly outside this conjunction.

The faithful playable reference on #608 retained 883 submits and renders hash
`5759c125812154dc`. A final-submit replay with fresh depth renders `71b84bdfae53933c` instead. Use
`gpu_replay --bundle-ds-summary` to scan every depth/stencil identity and raw programming variant without
reconstructing resource payloads or invoking Vulkan. Timeline-v5 `--depth-summary 642x362`, enabled during
capture with `PROSPER_GPU_TIMELINE_DEPTH_HASH_DIM=642x362`, found that the stable surface backing changed via
compute: program `0x401aec200` fills the 32 KiB HTILE block with `0xfffffff0` before the scene draws. #611 now
invalidates overlapping persistent Vulkan depth/stencil images on guest GPU writes, restoring the rejected
layers. The 35-second screenshot warmup still skips color RTT history and therefore remains an overbright
progression diagnostic, not a color-correct gameplay oracle.
