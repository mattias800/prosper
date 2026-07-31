# Dragon Quest VII Reimagined routes

Reusable `PROSPER_PAD_SCRIPT` routes for Dragon Quest VII Reimagined
(`PPSA17942`). Run tools from `prosper/` so the relative route path resolves.

## Title screen

`reach-title-screen.pad` sends six ordinary Cross pulses three seconds apart,
then a seventh at 30 seconds to leave the post-logo authored-black state. It
returns to neutral so the title can idle. This exact sequence was revalidated
from genuinely isolated save roots on current master: the native 3840x2160
frontend reached the animated title at about 34 seconds.

Without input, the startup flow can remain behind an authored opaque-black
Slate background. That state is not evidence that the scene underneath failed
to render, so use this route for title-screen graphics work.

Capture one direct frontend PNG per second with the recipe below. `<EVIDENCE_ROOT>` means a unique
directory created under `$HOME`; substitute its absolute path before running the command so the
screenshots do not land in the worktree or `/tmp`.

```bash
PROSPER_GUEST_FS=1 \
PROSPER_NULL_PAGE=1 \
PROSPER_GUEST_ARGS= \
PROSPER_SAVE0=<FRESH_SAVE_ROOT>/save0 \
PROSPER_SAVEDATA_DIR=<FRESH_SAVE_ROOT>/savedata \
PROSPER_PAD_SCRIPT=@scripts/dragon-quest-vii/reach-title-screen.pad \
PROSPER_PAD_SCRIPT_LOG=1 \
./build-linux/screenshot <DUMP_ROOT>/PPSA17942-app0 \
  --seconds 1 --count 40 --timeout 300 --require-composited-frame \
  --out <EVIDENCE_ROOT>/shots
```

Keep `PROSPER_PAD_SCRIPT_LOG=1`: it proves the game observed every press and
neutral edge. Point entries intentionally use the standard scripted-input hold;
no title-specific parser behavior or renderer override is involved.

Set both save roots for a genuinely fresh run. Dragon Quest VII stores its normal
files through the mounted `/savedata0` backend controlled by `PROSPER_SAVE0`;
`PROSPER_SAVEDATA_DIR` covers the separate SaveDataMemory API and does not isolate
those mounted files by itself.

A later exact-master replay corrected #1553's temporal/flicker interpretation.
Cross at 55 seconds had already entered and highlighted `1: Unused` with its
normal slot prompt. Circle at 140 seconds canceled to the adventure-log list,
and Circle at 270 seconds canceled again to the title. Start/Options at 330
seconds and Circle at 350 seconds did not advance the title. Cross is confirm
and Circle is cancel in this flow; #1553 incorrectly attributed the already-
visible post-Cross prompt to the later Circle press. The startup/title route
above is unchanged.

A subsequent current-master run validated the next control using only Cross.
After the checked-in title sequence, Cross at 35 seconds left the title and
Cross at 55 seconds opened the slot prompt with `1: Unused` highlighted. The
route then paused on that visible prompt before one Cross at 200 seconds; the
player-name keyboard was visible by 204 seconds and had a clean direct frontend
frame at 222.3 seconds. The long
pause is an evidence-gathering aid, not a claim that the guest requires that
delay. No character was entered, no normal game-save artifact was created, and
gameplay remains unvalidated. The representative unmodified capture is
[`../../../assets/screenshots/dragon-quest-vii-name-entry.png`](../../../assets/screenshots/dragon-quest-vii-name-entry.png).

This name-entry state ran at roughly 5.5 rendered FPS during a shared-GPU run,
with intermittent white/blue washed frames around otherwise coherent keyboard
frames. Treat that as an open rendering/performance limitation. Validate the
remaining name-entry presses before committing a later checkpoint route.

The title is animated. Some retained frames currently show a dark/purple
background behind the stable logo while adjacent frames show the expected sky
and ocean. Treat that as a separate flicker/animation investigation; do not
suppress the final opaque-black Slate background draw, which is authored state.
