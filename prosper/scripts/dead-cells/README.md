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

The long renderer warmup makes this route practical under llvmpipe, but it can
skip one-time GPU producers. The current post-warmup gameplay image is tracked
in issues #566/#586 and must not be used as a golden visual checkpoint until
the warmup artifact has been separated from genuine renderer behavior.
