# Blasphemous 2 Routes

Reusable `PROSPER_PAD_SCRIPT` routes for Blasphemous 2 (`PPSA13579`). Run tools
from `prosper/` so relative script paths resolve.

## Current Route

`reach-first-gameplay.pad` is an exploratory Cross route anchored to the
first pad poll. It passes the studio logos, title, and EULA. With the local URI
parser from #642, it also passes the title's telemetry calls and loads gameplay
scenes and assets. The normal-return pthread/TLS cleanup crash is fixed by #644,
and native-resolution captures render the opening cinematic and first gameplay. Its one-second holds
are intentionally longer than normal taps: synchronous native-resolution rendering
can reduce this title to two or three pad polls per second, so short wall-clock
windows are not reliable. Set `PROSPER_PAD_SCRIPT_LOG=1` to record the scripted
states that were actually observed by the game.

Capture one full-resolution PNG per second with:

```bash
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_RENDER_EVERY=30 \
PROSPER_PAD_SCRIPT=@scripts/blasphemous2/reach-first-gameplay.pad \
./build-linux/screenshot /path/PPSA13579-app0 \
  --seconds 1 --count 360 --timeout 450 --out shots
```

At full resolution, rendering every draw-carrying submit reduces this route to
roughly two or three game frames per second. The validated Blasphemous 2 recipe
samples every 30th complete submit while still writing one normal 1920x1080 PNG
per second; this reaches the first playable room in about five minutes. The
manifest records the renderer policy and reports source and pixel progression
separately.

The title's optional `libfmodstudio.prx` and `libfmod.prx` must be prelinked as
described by #638/#640. A run that stops before the first studio logo has not
reached the #641 marker fix; a run that faults at guest `eboot+0x11f79d0` has not
picked up the #642 URI parser.
