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

### Route pacing is frontend-sensitive

`reach-first-level.pad` mixes wall-time pulses (60-96 s) with flip-anchored pulses at `f3800`+. Under
`boot_trace` the wall-time pulses are all spent by 96 s while the run is only near frame 1100, so
`f3800` is never reached and the guest stalls at `title_controller_ship`. That route is calibrated for
a frontend that flips far more slowly per wall-second than headless `boot_trace` does.

To reach the **world map** under `boot_trace`, use a flip-anchored route in the f730-960 window
instead; that is the pacing the retained world-map captures were produced with. Check
`PROSPER_PAD_SCRIPT_LOG=1` output against the observed frame counter before assuming a route applies
to a new frontend.

`boot_trace` also writes a 3840x2160 BMP per sampled frame by default (`dump=1`), which reaches
gigabytes over a multi-minute route. Set `PROSPER_NO_FRAME_DUMPS=1` for capture runs that only need
the `.prgbundle`.

## Captures

These are unmodified frontend captures from the documented routes. The incomplete rendering is
visible in the images and is not presented as renderer-completeness evidence.

Linux app, real decoded opening movie:

![Astro Bot Sony Presents on the Linux app](../../docs/screenshots/issue-825-astrobot-linux-sony-presents.png)

Windows app, Media Foundation/DXVA opening movie and title screen:

![Astro Bot Sony Presents on the Windows app](../../docs/screenshots/issue-825-astrobot-windows-sony-presents.png)

![Astro Bot title screen on the Windows app](../../docs/screenshots/issue-825-astrobot-windows-title.png)

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

## Native Windows

Use `reach-first-level-windows.pad` for native Windows hardware-decode runs.
Without full-resolution frame dumping, the unskipped reference DXVA path reaches
`title_controller_ship` between about 167 and 273 seconds depending on decoder
throughput, then finishes loading `worldmap` about 13-17 seconds later. The route waits until 80 seconds—after the
hardware decoder has delivered its first frame—then repeats released
Cross/Cross/Options pulses across the logo/title transition. A second recovery
sequence covers the late unskipped-movie window because native Windows pad
polling becomes sparse once the world-map load is idle. Long Cross holds after
720 seconds cover screenshot runs where writing 3840x2160 frame dumps reduces
guest pad polling to roughly 1-2 Hz. The route also retains the validated
flip-anchored Linux sequence:

```powershell
$env:PROSPER_GUEST_FS = '1'
$env:PROSPER_GUEST_ARGS = '-force-gfx-direct'
$env:PROSPER_NO_COMPUTE = '1'
$env:PROSPER_PAD_SCRIPT = '@C:/path/to/prosper/scripts/astrobot/reach-first-level-windows.pad'
$env:PROSPER_PAD_SCRIPT_LOG = '1'
prosper/build-mingw-app/boot_trace.exe C:\path\to\PPSA21564-app0
```

Do not set `PROSPER_AVP_SYNTH_FRAMES` for Windows acceptance. The log must report
`decoder=hardware DXVA MFT` before the title/first-level state markers. The
earlier 60-second mixed-clock route exposed the backend-frame lifetime bug
tracked in #855; this Windows route begins later so the log retains direct DXVA
evidence before it skips the remainder of the movie.
