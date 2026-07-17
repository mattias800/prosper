# Astro Bot routes

Reusable `PROSPER_PAD_SCRIPT` routes for Astro Bot (`PPSA21564`). Run tools from
`prosper/` so the relative paths below resolve.

## First-level headless checkpoint

`reach-first-level.pad` advances a fresh boot through `ps_logo`,
`title_controller_ship`, and `worldmap`, then selects the opening level. The
checkpoint is reached when the guest reports both:

```text
GAME: Level has started: intro_next
PLAY: SubLevelLocator sublevel_locator_hub [hub_crashsite_tutorial], state changed 1->4
```

On a host without a usable VA-API device, explicitly enable the bounded synthetic
video fallback for control-flow testing:

```sh
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_NO_COMPUTE=1 \
PROSPER_AVP_SYNTH_FRAMES=120 \
PROSPER_PAD_SCRIPT=@scripts/astrobot/reach-first-level.pad \
PROSPER_PAD_SCRIPT_LOG=1 \
  timeout 180 ./build-linux/boot_trace /path/to/PPSA21564-app0
```

Synthetic video is not video-decode acceptance evidence. On a VA-API-capable
Linux host, omit `PROSPER_AVP_SYNTH_FRAMES`; on Windows, the Media Foundation
backend decodes the real H.264 stream. Keep `PROSPER_PAD_SCRIPT_LOG=1` enabled
when validating a new frontend because the flip and wall-time recovery windows
can overlap.
