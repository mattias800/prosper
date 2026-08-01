# Audio (`libSceAudioOut`)

prosper implements the PS5 `sceAudioOut*` API as a **headless, backend-agnostic core** with
**pluggable frontends**. The HLE never depends on any audio library — it decodes the guest's calls
into a port lifecycle plus interleaved PCM "grains" and forwards them to an installed `AudioSink`.
This keeps `prosper_core` dependency-free and unit-testable, and lets any frontend (SDL3, a file
recorder, a network sink, …) be swapped in at runtime.

## Layers

| Layer | File | In `prosper_core`? |
|---|---|---|
| Backend interface | `src/hle/audio.hpp` (`AudioSink`, `AudioPortInfo`, format decode) | yes (header) |
| HLE + default sink | `src/hle/hle_audio.cpp` | yes |
| SDL3 frontend (optional) | `frontends/audio_sdl3/` | **no** (separate target) |

### HLE coverage (`hle_audio.cpp`)
`sceAudioOutInit`, `sceAudioOutOpen`, `sceAudioOutOutput`, `sceAudioOutOutputs`,
`sceAudioOutSetVolume`, `sceAudioOutClose`, `sceAudioOutGetPortState`. A 16-port table tracks each
open port's `{freq, channels, format, grain}`. `sceAudioOutOpen`'s `param` is decoded into channels
+ sample format (`SceAudioOutParamFormat` 0–7: S16/F32 × mono/stereo/8ch). PCM pointers are read
directly (guest memory is 1:1-mapped), so grains are forwarded to the sink with **zero copies** in
the HLE.

### Default backend — silent, real-time paced
With no frontend installed, `prosper_core` uses a built-in `RealtimeSilentSink`. On real hardware,
`sceAudioOutOutput` **blocks until the audio ring has room**, which is what paces a game's audio
thread. The default sink reproduces that pacing by sleeping each grain's wall-clock duration
(`grain / freq` seconds), so the guest advances at the correct speed with **no device attached** —
correct timing, no sound. This is the right default for headless bring-up.

### Installing a frontend
```cpp
#include "hle/audio.hpp"
prosper::audio_set_sink(&my_sink);   // AudioSink*; pass nullptr to restore the silent default
```

## SDL3 frontend (optional)

Builds a real audio backend that plays the guest's output on the host's default device.

```sh
cmake -S prosper -B build -DPROSPER_AUDIO_SDL3=ON   # uses system SDL3 if present, else fetches release-3.2.30
```
- Uses a system SDL3 (`find_package(SDL3)`) if available, otherwise fetches and builds an
  **audio-only** SDL3 (video/render/joystick/etc. disabled — no X11/Wayland needed).
- One `SDL_AudioStream` per PS5 port; `output()` pushes PCM and blocks while the device queue is full
  (matching hardware pacing). Per-channel volumes are mapped to the stream gain.
- When enabled, `boot_trace` calls `install_sdl3_audio_sink()` at startup automatically.

## Diagnosing a silent title — `PROSPER_AUDIO_FLOW=1`

"No audio" has three mutually exclusive causes that are **indistinguishable from outside** and need
completely different fixes:

1. the guest never submits PCM — the defect is upstream (voice/mixer setup, or an HLE return that
   makes the title's audio engine give up);
2. the guest submits PCM but it is all zero — the defect is in the title's own mixing/decode;
3. the guest submits real PCM and prosper loses it — the defect is ours (routing, sink, volume).

`PROSPER_AUDIO_FLOW=1` reports **unconditionally**, once per second per AudioOut2 context, from both
`ContextAdvance` and `ContextPush`, so each branch has its own signature:

```text
[audio-flow] ctx0 created: grain=256 queue_depth=2 (host sink port 17)
[audio-flow] port1 created: type=0x0 data_format=0xc00 flags=0x0 (MAIN -> mixed to host)
[audio-flow] ctx0 advance=188 push=188 (with-pcm=0 silent-paced=188 not-ready=0) sink=never-opened \
             port=17 grains=0 bytes=0 bed-peak=0.00000 bed-rms=0.00000 bed-nonzero=0/96256 \
             bed-nan=0 | BED LIFE: nonzero=0/3944448 peak=0.00000 rms=0.00000 nan=0 \
             | lifetime: ctx=2 ports=2 pcm-published=15039
[audio-flow]   port1 type=0x0 fmt=0xc00 reads=188 mixed=0 frames=48128 bytes=2310144 \
               peak=0.15349 rms=0.01289 nonzero=86528/577536 nan=0 no-pcm=0 skip-fmt=188 \
               skip-not-main=0 short=0 \
               | LIFE: nonzero=1782048/18456576 peak=0.38976 rms=0.01033 nan=0
```

That is a real *Dragon Quest VII* capture, and the port line alone diagnoses the title: the guest is
handing us 2,310,144 bytes per second of genuine audio (`LIFE:` run peak **0.38976**, 1.78 M non-zero
samples, `nan=0` ruling out a NaN artifact), and `mixed=0` with `skip-fmt=188` says prosper discards
every grain of it because the port declares 12 channels. Its context correspondingly shows
`sink=never-opened` and `BED LIFE: nonzero=0/3944448` — nothing from it ever reaches the device.

Note how little the interval columns would tell you on their own: this line's `peak=0.15349` happens
to be a quiet moment, and other intervals in the same run read `nonzero=0/577536` outright. The
verdict is in `LIFE:`.

| Observation | Meaning |
|---|---|
| no `[audio-flow]` lines at all | the guest never created an AudioOut2 context |
| `ctx…created` but no interval lines | a context exists, but the pump loop never runs |
| `advance=N push=0` | the pump runs but never submits — **case 1** |
| `push=N`, **`BED LIFE: nonzero=0/M`** | real submissions carrying exact silence — **case 2** |
| `push=N`, **`BED LIFE: nonzero>0`** | signal reaches the host sink — **case 3**, look at the sink/volume |
| any port with `LIFE: nonzero>0` but `mixed=0` | **prosper is discarding real audio** — read `skip-fmt` / `skip-not-main` to see why |

The last row is the one that matters most and is invisible in the bed alone: a port can carry the
title's entire soundtrack and never reach the mix. That is what *Dragon Quest VII* does (#1692).

Read the counters, not just presence:

- **Reached-ness** (`lifetime: ctx= ports= pcm-published=`) makes a null reading a *finding*: zero
  submissions means something different when the guest never built an audio graph
  (`ports=0`), built one but never handed over a buffer (`pcm-published=0`), or wired everything up
  and still never pushed.
- **`nonzero=N/M` is the measure that decides silent vs quiet.** Peak is raised by a single stray
  sample and RMS needs a threshold that will call some genuinely-mixed quiet passage silent; only an
  exact `nonzero=0/M` proves the guest submitted a cleared buffer. The rule lives in
  `AudioSignalStats` (`src/hle/audio.hpp`) and is unit-tested against both alternatives.
- **Judge a port by its `LIFE:` totals, never by one interval.** Per-interval counters reset every
  second and real playback has gaps, so any single line can read `nonzero=0/M` on a port carrying
  strong signal over the run. The never-reset totals are what make one line sufficient — reading
  only the last interval inverted this investigation's first conclusion (instrument-trap 39).
- **`BED LIFE:` and a port's `LIFE:` do not share a denominator.** They start at different moments —
  the context's lifetime begins at ContextCreate, each port's at PortCreate — so the two adjacent
  `nonzero=N/M` values on consecutive lines are *not* directly comparable, and in the example above
  they cover roughly 41 and 32 seconds respectively. Compare each against itself over time, or
  against its own `samples`; never read one as a fraction of the other.
- **`nan=N` names the one remaining input class.** NaN is neither silence nor signal: the sink
  clamps it to zero, so a NaN-producing guest is audibly silent while its buffers are "full". The
  probe counts NaN separately on both port and bed lines and then treats it as the silence it
  becomes, so `nonzero>0 nan>0` is a decode bug, not a working mix.
- **Per-port attribution** locates loss the mixed bed hides: `skip-not-main` (a port with signal
  discarded for not being MAIN), `skip-fmt` (unsupported layout), `no-pcm` (no buffer published),
  `short` (a partial/faulting guest read).
- A port **rejected for its format is still read and measured** under this probe, because a rejected
  port is exactly where lost audio would hide — "nothing there" must be a measurement, not an
  assumption. Rejection still discards it from the mix; only the diagnostic reads it. Treat such a
  reading as slightly lower confidence than an accepted port's: the read length is derived from the
  context grain and a layout prosper does not support, so an over-read past the guest buffer would
  be attributed to that port. It is fault-safe and cannot invent silence, but it could in principle
  inflate `nonzero` — confirm a surprising result against the producing code before acting on it.
- **`peak=inf` / `rms=inf` is a correct reading, not a probe bug.** Infinities are deliberately not
  filtered the way NaN is: the output path clamps `+Inf` to `+1.0`, so an Inf-producing guest is
  genuinely, loudly audible and belongs in the signal statistics. Because `LIFE:` never resets, one
  Inf pins the run peak and rms for the rest of the session — read the per-interval columns to see
  where it came from, and treat it as a guest-side mixing defect worth its own investigation.

Related, narrower probes: `PROSPER_AUDIOLOG=1` (legacy `sceAudioOut` path: per-port calls, bytes,
peak and RMS), `PROSPER_AUDIO_DUMP=PATH` (raw `PATH.portN.raw` PCM for offline inspection),
`PROSPER_AUDIO2LOG=1` (full AudioOut2 call trace with hexdumps), `PROSPER_AUDIO2_PROBE=1` (per-port
PCM sampling, signal-bearing ports only) and `PROSPER_AUDIO2_CONTROL_PROBE=1` (control surface).
Note that the first two fall silent in cases 1 **and** 2, which is why `PROSPER_AUDIO_FLOW` exists.

## Tests (`ctest`)
- **`audio_hle`** (`tests/test_audio.cpp`) — always built. Drives the real HLE entrypoints through the
  dispatch table with a capturing sink: format decode (all 8 formats + unknown), port open/close,
  byte-exact PCM forwarding, `Output`/`Outputs`, volume, port state, exhaustion, and error paths.
- **`audio_sdl3`** (`frontends/audio_sdl3/test_audio_sdl3.cpp`) — built with `-DPROSPER_AUDIO_SDL3=ON`.
  End-to-end through the real SDL3 backend under the SDL **dummy** audio driver, so it runs on
  headless CI with no sound hardware.

`test_audio` also covers `AudioSignalStats`, the silent-vs-quiet decision rule behind
`PROSPER_AUDIO_FLOW`, including the quiet-but-real case that peak and RMS alone cannot separate
from silence.
