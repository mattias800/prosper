# Evergate progression route

`reach-main-menu.pad` keeps a synthetic controller connected but neutral so the title screen remains
visible. `reach-first-gameplay.pad` repeatedly taps Cross through the opening logos, title screen,
and first save slot. Its wall-time windows are intended for the accelerated screenshot command below,
where hundreds of render submits are deliberately skipped.

`reach-first-gameplay-realtime.pad` expresses the same twelve Cross/neutral edges on the guest's pad-read
axis. Use it for native-resolution interactive and performance runs. A synchronous render can take longer
than a wall-time pulse, causing two pulses to merge or a complete press/release to pass between polls; the
read-anchored route cannot lose those edges. Do not use it with snapshot acceleration: skipped render submits
let the guest consume the early read ranges before wall-time-driven logos finish.

The realtime route was validated from a fresh save with the native Windows frontend rendering every submit
at 1920x1080: all twelve Cross/neutral edges appeared in `PROSPER_PAD_SCRIPT_LOG`, and retained captures
progressed from save-slot selection through the transition into the opening scene.

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

The reviewed title-screen regression guard uses the neutral route:

```sh
python3 tools/snapshot/snapshot.py check evergate-title
```

## Realtime harness

Realtime and performance runs must render every submit at the guest's native resolution. The
`--render-every 500` and `PROSPER_RENDER_SCALE=4` settings above are snapshot-only acceleration:
carrying them into `prosper-app` skips visible updates during the opening and renders Evergate's
1920x1080 output at 480x270 before scaling it to the window.

```sh
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_RENDER_EVERY=1 \
PROSPER_RENDER_SCALE=1 \
PROSPER_PAD_SCRIPT=@scripts/evergate/reach-first-gameplay-realtime.pad \
  ./build-linux-app/prosper-app --dump /path/to/PPSA01885-app0
```

Use the equivalent environment variables with `prosper-app.exe` on Windows. The scripted input is
composed with the normal SDL keyboard/controller backend, so the window remains interactive after
the route reaches gameplay. Ensure `PROSPER_RENDER_EVERY_FOR_MS` is unset if the process inherits a
shell used for snapshot capture. Set `PROSPER_APP_DUMP_FRAMES=1` and
`PROSPER_FRAME_DIR=/path/to/frames` to retain the exact composited frames consumed by the realtime
harness.
