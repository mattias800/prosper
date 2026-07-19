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
- `reach-first-level.pad`: the wall-time route used for the final fresh-save
  validation on issue #522. It crosses the title, story choices, and dialogue
  into the first playable level. A late Up/Cross-only recovery sequence lets a
  cold full-render process complete name/menu prompts even when a long renderer
  stall skips the early input windows; those inputs are harmless once gameplay
  has begun. Use one-second screenshot sampling and allow at least 320 seconds
  for the reviewed gameplay evidence window.
- `reach-first-level-windows.pad`: native Windows full-render variant. Explicit
  1.5-second input windows survive sparse pad polling and its delayed tail waits
  for save-slot and name-entry transitions. The validated native run reached a
  fully lit first-level frame at 360 seconds. Use a fresh `PROSPER_SAVEDATA_DIR`
  and keep `PROSPER_PAD_SCRIPT_LOG=1` enabled.

Create another route with `prosper-app --record <path>` or
`PROSPER_PAD_RECORD=<path>`. Add `--record-axis pad-read` (or set
`PROSPER_PAD_RECORD_AXIS=pad-read`) when pad polling is a more stable clock than presentation. Replay
it with `PROSPER_PAD_SCRIPT=@path`, and only
commit it after repeated current-master runs reach the state named by the file.
