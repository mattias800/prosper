// audio_sdl3.hpp — optional SDL3 audio frontend for the headless sceAudioOut HLE.
//
// Built only when -DPROSPER_AUDIO_SDL3=ON. Installs an SDL3-backed AudioSink so the guest's
// audio actually plays out of the host's default device. The core HLE (src/hle/hle_audio.cpp)
// has no knowledge of SDL — this frontend lives outside prosper_core and plugs in at runtime.
#pragma once

namespace prosper {

// Initialise SDL audio and install the SDL3 AudioSink as the active backend. Idempotent.
// Returns true on success; false (and leaves the default silent sink active) if SDL audio
// could not be initialised. Safe to call once at startup after register_builtin_hle().
bool install_sdl3_audio_sink();

// Pause/resume every active stream without discarding its queued samples. The guest output side
// also waits on the shared lifecycle gate, so no additional audio accumulates during a host pause.
// No-op when the SDL3 sink is not installed.
void set_sdl3_audio_paused(bool paused);

// Set the playback gain applied to every stream, as a linear multiplier (1.0 = unchanged).
// Applied via SDL_SetAudioStreamGain, so no sample data is copied or clipped by prosper.
// Takes effect immediately on open streams and is remembered for streams opened later.
//
// prosper-app defaults this to 1.0 -- the title's own mix, unedited, which is the only output a
// compatibility layer can call correct. Attenuate a bring-up run with `--volume N` rather than
// changing the default; see kDefaultVolumePercent in prosper-app/main.cpp.
void set_sdl3_audio_gain(float gain);

// Uninstall the SDL3 sink (restores the default), closing any open streams and SDL audio.
void shutdown_sdl3_audio_sink();

} // namespace prosper
