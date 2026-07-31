# Dragon Quest VII Reimagined routes

Reusable `PROSPER_PAD_SCRIPT` routes for Dragon Quest VII Reimagined
(`PPSA17942`). Run tools from `prosper/` so the relative route path resolves.

## Title screen

`reach-title-screen.pad` sends six ordinary Cross pulses, three seconds apart,
to skip the startup logos and movie, then returns to neutral so the title can
idle. This exact sequence was revalidated from a fresh save on current master:
the native 3840x2160 frontend reached the animated title at about 28 seconds.

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
PROSPER_SAVEDATA_DIR=<FRESH_SAVE_ROOT> \
PROSPER_PAD_SCRIPT=@scripts/dragon-quest-vii/reach-title-screen.pad \
PROSPER_PAD_SCRIPT_LOG=1 \
./build-linux/screenshot <DUMP_ROOT>/PPSA17942-app0 \
  --seconds 1 --count 40 --timeout 300 --require-composited-frame \
  --out <EVIDENCE_ROOT>/shots
```

Keep `PROSPER_PAD_SCRIPT_LOG=1`: it proves the game observed every press and
neutral edge. Point entries intentionally use the standard scripted-input hold;
no title-specific parser behavior or renderer override is involved.

The title is animated. Some retained frames currently show a dark/purple
background behind the stable logo while adjacent frames show the expected sky
and ocean. Treat that as a separate flicker/animation investigation; do not
suppress the final opaque-black Slate background draw, which is authored state.
