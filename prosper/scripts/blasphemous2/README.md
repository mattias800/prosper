# Blasphemous 2 Routes

Reusable `PROSPER_PAD_SCRIPT` routes for Blasphemous 2 (`PPSA13579`). Run tools
from `prosper/` so relative script paths resolve.

## Current Route

`reach-first-gameplay.pad` is an exploratory Cross-tap route anchored to the
first pad poll. On #641 stacked over #640 it passes the studio logos, title, and
EULA, then reaches the offline HTTP boundary tracked by #642. It does not yet
reach gameplay; tighten the taps only after that service boundary is fixed.

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
reached the #641/#642 frontier.
