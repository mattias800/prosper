# Evergate progression route

`reach-main-menu.pad` keeps a synthetic controller connected but neutral so the title screen remains
visible. `reach-first-gameplay.pad` repeatedly taps Cross through the opening logos, title screen,
and first save slot. Its timing is anchored to the first pad poll and leaves ten seconds for native
Windows initialization before sending twelve Cross pulses.

## Headless screenshot acceptance

From `prosper/`, with a fresh game-data directory:

```sh
PROSPER_PAD_SCRIPT=@scripts/evergate/reach-first-gameplay.pad \
PROSPER_RENDER_SCALE=4 \
  ./build-linux-app/screenshot /path/to/PPSA01885-app0 \
  --render-every 500 --render-every-for-seconds 90 \
  --seconds 10 --count 12 --out evergate-shots
```

Use `reach-main-menu.pad` to retain the Evergate logo and `Press any button` prompt. Successful
gameplay shows the masked character in the first tutorial room; the clearest early checkpoint is
the `HOLD X TO JUMP HIGHER` prompt. The luminous orb over white terrain is part of the opening ident
and is not gameplay. `--render-every` only accelerates the opening; normal per-submit rendering
resumes after 90 seconds.

## Realtime harness

```sh
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_RENDER_EVERY=500 \
PROSPER_RENDER_EVERY_FOR_MS=90000 \
PROSPER_RENDER_SCALE=4 \
PROSPER_PAD_SCRIPT=@scripts/evergate/reach-first-gameplay.pad \
  ./build-linux-app/prosper-app --dump /path/to/PPSA01885-app0
```

Use the equivalent environment variables with `prosper-app.exe` on Windows. The scripted input is
composed with the normal SDL keyboard/controller backend, so the window remains interactive after
the route reaches gameplay. Set `PROSPER_APP_DUMP_FRAMES=1` and `PROSPER_FRAME_DIR=/path/to/frames`
to retain the exact composited frames consumed by the realtime harness.
