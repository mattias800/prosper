# Blasphemous 2 Routes

Reusable `PROSPER_PAD_SCRIPT` routes for Blasphemous 2 (`PPSA13579`). Run tools
from `prosper/` so relative script paths resolve.

## Current Route

`reach-first-gameplay.pad` is an exploratory Cross-tap route anchored to the
first pad poll. It passes the studio logos, title, and EULA. With the local URI
parser from #642, it also passes the title's telemetry calls and loads gameplay
scenes and assets. The normal-return pthread/TLS cleanup crash is fixed by #644,
and native-resolution captures render the opening cinematic. A gameplay screenshot
is not yet confirmed because this exploratory route's 250 ms input windows can be
missed by the slower synchronous renderer; #646 tracks the poll-safe route/tooling fix.

Capture one full-resolution PNG per second with:

```bash
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_PAD_SCRIPT=@scripts/blasphemous2/reach-first-gameplay.pad \
./build-linux/screenshot /path/PPSA13579-app0 \
  --seconds 1 --count 120 --timeout 180 --out shots
```

The title's optional `libfmodstudio.prx` and `libfmod.prx` must be prelinked as
described by #638/#640. A run that stops before the first studio logo has not
reached the #641 marker fix; a run that faults at guest `eboot+0x11f79d0` has not
picked up the #642 URI parser.
