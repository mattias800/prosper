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

## Opening chapter — `reach-gameplay.pad`

`reach-gameplay.pad` continues past the title: new adventure log, save slot, name entry, the four
first-run System Settings screens, and into the opening chapter in Estard. Run it with the same
recipe as the title route.

**Read the control model before editing it — three of these are not guessable from the screen.**

- **`OPTIONS` is the jump-to-OK shortcut on the player-name keyboard.** The cursor starts on `A`,
  cross types the highlighted letter, and cross alone never leaves the letter grid. One `options`
  press moves the cursor straight to the on-screen `OK` cell. (`options` does nothing on any other
  screen in this flow, so a stray one is harmless.)
- **Cross on `OK` with an empty field raises a "No name has been entered." modal**, which then eats
  the next press. Type at least one character first.
- **Cross on `Back` cancels the whole name entry** back to the slot list. `Back` sits directly above
  `OK` in the same column and the column wraps vertically, so an extra `down` lands on it.
- **The two System Settings menus have different row counts.** 3/4 (camera) has four rows — vertical
  axis, horizontal axis, revert, confirm — so three `down`s reach Confirm. 4/4 (brightness) has
  three, so two do; a third wraps back to the bar. Cross on a value row only toggles that value, so
  a route that only presses cross sits on these two screens forever.

**Wall-clock anchors are not load-robust on this title, and this route uses them anyway.** Two runs
of the same binary and the same route reached the title screen at **34 s** and at **76 s**, and the
whole sequence shifts with it — a second instance of #2764. Every screen in the flow waits
indefinitely for input, so a late press still advances; what breaks is a press that lands one screen
early. Drive exploratory work with `PROSPER_PAD_SCRIPT_RELOAD=1` and watch the samples rather than
trusting the timings, and re-anchor the route if you extend it.

Two independent runs reach Estard, write `GameSaveData000.dat` and render the world. Neither
demonstrates free player control — see `docs/DRAGON_QUEST_STATUS.md` for exactly what is and is not
established, and for the state of the composite in that phase.

## Free field control — `reach-field-control.pad`

Reaches free field control in Pilchard Bay at t~600 s and holds it to the end of the run. Use this
one for gameplay work; `reach-gameplay.pad` stops inside the opening chapter's script.

The only difference between them is **confirm spacing**: 2 s instead of 15 s. Measured on one
binary, quiet box, world phase only — 41 confirms gave 1 frame with the field HUD, 447 gave 145.
The chapter script is simply long, and every screen in it waits indefinitely for confirm, so a
press that lands early is absorbed rather than lost. That makes this route unusually robust to the
wall-clock drift recorded in #2764.

```bash
PROSPER_NULL_PAGE=1 \
PROSPER_GUEST_ARGS= \
PROSPER_RENDER=1 \
PROSPER_SAVE0=<FRESH_SAVE_ROOT>/save0 \
PROSPER_SAVEDATA_DIR=<FRESH_SAVE_ROOT>/savedata \
PROSPER_PAD_SCRIPT=@scripts/dragon-quest-vii/reach-field-control.pad \
PROSPER_PAD_SCRIPT_LOG=1 \
./build-linux/screenshot <DUMP_ROOT>/PPSA17942-app0 \
  --seconds 4 --count 310 --timeout 1350 --out <EVIDENCE_ROOT>/shots
```

**Confirming you reached it, without trusting the clock.** The field HUD is the positive marker:
the circular minimap at bottom-left and the party block at bottom-right, neither of which the game
draws during a cutscene. Cinematic bars are the negative one
(`tools/frameclass/letterbox.py`). Require both — a collapsed white present has no bars either, and
a colourful cutscene frame passes a HUD-colour test on its own.
