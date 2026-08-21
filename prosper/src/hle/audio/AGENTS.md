# `src/hle/audio` — the guest-facing audio libraries

Everything a PS5 title calls to make sound, reimplemented against prosper's own backends. The
folder's boundary is the **guest ABI**: the byte layouts, handle lifecycles and error conventions
Sony's libraries present. Actual codec work and actual sound output live outside it — under
`frontends/` — and reach this code through two narrow seams. Keep it that way: `prosper_core` must
stay buildable with no FFmpeg, no SDL and no OS media dependency.

## The libraries, and why they are not one file

Four separate guest libraries land here, and they are genuinely separate surfaces even though a
title may use several at once:

- **`hle_audio.cpp`** — `libSceAudioOut` / `AudioOut2` / `AudioIn` (port lifecycle and interleaved
  PCM grains), plus `libSceNgs2` (the voice/rack mixer) and `libSceAjm`. It is the oldest and
  largest file here and it has accumulated more than one responsibility; a split is welcome, but it
  is a hot file that several lanes edit, so do it as its own commit with `tools/refactor/`.
- **`hle_audiodec.cpp`** — `libSceAudiodec`. A *direct* create/decode/delete decoder API, and the
  one Unreal Engine's **Electra** media player uses for movie audio. Do not confuse it with AJM:
  AJM is a batched job queue with a sideband result block, `libSceAudiodec` is four arguments and a
  control struct. They share a codec backend and nothing else.
- **`atrac9_decode.cpp`** — glue for the vendored `third_party/libatrac9`. ATRAC9 is the one codec
  the core decodes itself, because it is a Sony format with no host implementation to defer to.

## The two seams out of the core

- **`ajm_decoder.hpp`** — compressed-audio decode. `DecoderBackend::create_configured()` hands back a
  stateful `StreamDecoder` that eats a byte span and emits interleaved S16. `frontends/audio_ffmpeg`
  implements MP3 and AAC; a build without it simply has no backend and every library here refuses
  loudly rather than pretending. A codec added for one guest library is immediately available to
  the others.
- **`audio.hpp`** — output. `AudioSink` is installed by a host frontend; the core's default is a
  paced silent sink, which is what makes headless runs deterministic.

`audio.hpp` also declares the fault-safe guest-memory copies (`audio_read_bytes` /
`audio_store_bytes`). Use them rather than touching guest pointers directly: they return `false` for
an unmapped range instead of raising, which is the only reason an entry point here can answer a bad
guest pointer with an error rather than killing the process.

## The failure mode this folder keeps producing

Twice now a title has been killed by an audio library answering `SCE_OK` and writing **nothing**.
The dispatcher's default for an unregistered NID is `return 0`, which is success for a
status-returning contract, so the guest proceeds on an out-parameter that was never filled — and
audio out-parameters are overwhelmingly *divisors* (sample rate, channel count, frame size). The
crash then lands in the guest, a long way from the cause, and looks nothing like a missing library.

So: when you register an entry point here, register what it **writes**, not just what it returns.
If you cannot fill the out-parameters faithfully, returning an error is the honest answer and a
`return 0` is not. `hle_audiodec.cpp`'s header comment works a full example through, from the
guest's `div` instruction back to the unwritten field.
