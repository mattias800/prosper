# Kena: Bridge of Spirits routes

Reusable `PROSPER_PAD_SCRIPT` routes for *Kena: Bridge of Spirits* (`PPSA01802`, Unreal Engine 4). Run tools
from `prosper/` so the relative route path resolves.

## `reach-first-gameplay-prompt.pad`

Main menu → New Game → difficulty → brightness calibration → first level load → intro narration → the first
gameplay prompt. The anchors are **pad reads**, not seconds: the title polls the pad about once a second at its
menus, so wall-clock pulses mostly fall between reads. The file header records what each press does and when
each screen appears.

No input is needed to reach the main menu itself.

```bash
PROSPER_NULL_PAGE=1 PROSPER_GUEST_ARGS= \
PROSPER_SAVE0=<FRESH>/save0 PROSPER_SAVEDATA_DIR=<FRESH>/savedata \
PROSPER_PAD_SCRIPT=@scripts/kena/reach-first-gameplay-prompt.pad PROSPER_PAD_SCRIPT_LOG=1 \
  ./build-mingw-app/screenshot.exe <DUMP_ROOT>/PPSA01802-app0 \
  --seconds 10 --count 54 --timeout 560 --out <OUT>
```

Keep `PROSPER_PAD_SCRIPT_LOG=1`: its `[pad-script] ... read=N buttons=cross` lines are the proof each press was
read by the guest. Use fresh save roots for every run.

On a 32-wide device (NVIDIA) the gameplay world behind the prompt is black; that is the renderer's wave64 gap
(#2147), not the route.
