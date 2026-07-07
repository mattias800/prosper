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

// Uninstall the SDL3 sink (restores the default), closing any open streams and SDL audio.
void shutdown_sdl3_audio_sink();

} // namespace prosper
