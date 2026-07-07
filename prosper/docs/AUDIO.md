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

## Tests (`ctest`)
- **`audio_hle`** (`tests/test_audio.cpp`) — always built. Drives the real HLE entrypoints through the
  dispatch table with a capturing sink: format decode (all 8 formats + unknown), port open/close,
  byte-exact PCM forwarding, `Output`/`Outputs`, volume, port state, exhaustion, and error paths.
- **`audio_sdl3`** (`frontends/audio_sdl3/test_audio_sdl3.cpp`) — built with `-DPROSPER_AUDIO_SDL3=ON`.
  End-to-end through the real SDL3 backend under the SDL **dummy** audio driver, so it runs on
  headless CI with no sound hardware.
