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
[audio-flow] ctx1 created: grain=256 queue_depth=2 (host sink port 18)
[audio-flow] port2 created: type=0x0 data_format=0x800 flags=0x0 (MAIN -> mixed to host)
[audio-flow] ctx1 advance=188 push=188 (with-pcm=188 silent-paced=0 not-ready=0) sink=open port=18 \
             grains=188 bytes=385024 bed-peak=0.00000 bed-rms=0.00000 bed-nonzero=0/96256 \
             | lifetime: ctx=2 ports=2 pcm-published=2228
[audio-flow]   port2 type=0x0 fmt=0x800 reads=169 mixed=169 frames=43264 bytes=1384448 \
               peak=0.00000 rms=0.00000 nonzero=0/346112 no-pcm=0 skip-fmt=0 skip-not-main=0 short=0
```

| Observation | Meaning |
|---|---|
| no `[audio-flow]` lines at all | the guest never created an AudioOut2 context |
| `ctx…created` but no interval lines | a context exists, but the pump loop never runs |
| `advance=N push=0` | the pump runs but never submits — **case 1** |
| `push=N` with `bed-nonzero=0/…` | real submissions carrying exact silence — **case 2** |
| `push=N` with `bed-nonzero>0` | signal reaches the host sink — **case 3**, look at the sink/volume |

Read the counters, not just presence:

- **Reached-ness** (`lifetime: ctx= ports= pcm-published=`) makes a null reading a *finding*: zero
  submissions means something different when the guest never built an audio graph
  (`ports=0`), built one but never handed over a buffer (`pcm-published=0`), or wired everything up
  and still never pushed.
- **`nonzero=N/M` is the measure that decides silent vs quiet.** Peak and RMS both print `0.00000`
  for a correctly-mixed quiet passage; only an exact `nonzero=0/M` proves the guest submitted a
  cleared buffer. The rule lives in `AudioSignalStats` (`src/hle/audio.hpp`) and is unit-tested.
- **Per-port attribution** locates loss the mixed bed hides: `skip-not-main` (a port with signal
  discarded for not being MAIN), `skip-fmt` (unsupported layout), `no-pcm` (no buffer published),
  `short` (a partial/faulting guest read).
- A port **rejected for its format is still read and measured** under this probe, because a rejected
  port is exactly where lost audio would hide — "nothing there" must be a measurement, not an
  assumption. Rejection still discards it from the mix; only the diagnostic reads it.

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
