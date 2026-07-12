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
