# The Messenger Routes

Reusable `PROSPER_PAD_SCRIPT` routes for The Messenger (`PPSA24651`). Run tools
from `prosper/` so the relative paths below resolve:

```bash
PROSPER_PAD_SCRIPT=@scripts/messenger/reach-intro-story.pad \
  ./build-linux/screenshot /path/PPSA24651-app0 --seconds 1 --count 10 --out shots
```

## Routes

- `reach-intro-story.pad`: presses Options at flips 10-14 after the first pad
  poll. The interval was produced by `PROSPER_PAD_RECORD` and verified twice on
  current master at `PROSPER_RENDER_SCALE=4`; both runs reached the opening map
  narration. The exact narration phase still drifts by a few rendered frames,
  so this is a coarse state route, not an exact visual checkpoint.

Create another route with `prosper-app --record <path>` or
`PROSPER_PAD_RECORD=<path>`, replay it with `PROSPER_PAD_SCRIPT=@path`, and only
commit it after repeated current-master runs reach the state named by the file.
