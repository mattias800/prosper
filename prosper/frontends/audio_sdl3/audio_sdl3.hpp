// audio_sdl3.hpp — optional SDL3 audio frontend for the headless sceAudioOut HLE.
//
// Built only when -DPROSPER_AUDIO_SDL3=ON. Installs an SDL3-backed AudioSink so the guest's
// audio actually plays out of the host's default device. The core HLE (src/hle/audio/hle_audio.cpp)
// has no knowledge of SDL — this frontend lives outside prosper_core and plugs in at runtime.
#pragma once
#include <array>
#include <cstdint>

namespace prosper {

enum class AudioDemandPhase : unsigned { Startup, Active, Paused };
struct AudioDemandCounts {
    uint64_t calls = 0, requested_bytes = 0, shortfall_calls = 0, additional_bytes = 0;
    uint64_t first_ns = 0, last_ns = 0;
};
struct Sdl3AudioDemandSnapshot {
    uint64_t generation = 0; // changes when this port is reopened
    std::array<AudioDemandCounts, 3> phases{};
};

// Optional PROSPER_AUDIO_DEMAND=1 diagnostic. Counts SDL's input-format demand at
// consumption, not hardware underruns; resampling can overestimate additional bytes.
// False means unavailable/closed, never a measurement of zero shortfalls.
bool sdl3_audio_demand_snapshot(int port, Sdl3AudioDemandSnapshot& snapshot);

// Initialise SDL audio and install the SDL3 AudioSink as the active backend. Idempotent.
// Returns true on success; false (and leaves the default silent sink active) if SDL audio
// could not be initialised. Safe to call once at startup after register_builtin_hle().
bool install_sdl3_audio_sink();

// Pause/resume every active stream without discarding its queued samples. The guest output side
// also waits on the shared lifecycle gate, so no additional audio accumulates during a host pause.
// No-op when the SDL3 sink is not installed.
void set_sdl3_audio_paused(bool paused);

// Set the HOST's playback gain -- an amplifier after the guest's own mixer, as a linear multiplier
// (1.0 = unchanged). Applied via SDL_SetAudioStreamGain on the logical DEVICE each port feeds, so
// no sample data is copied or clipped by prosper and the guest's per-port channel gain (which it
// sets through sceAudioOutSetVolume) is left entirely alone. Deliberately a different knob: the two
// used to write the same one, so the guest silently undid `--volume` seconds into every boot
// (#3489). Takes effect immediately on open streams and is remembered for streams opened later.
//
// prosper-app defaults this to 1.0 -- the title's own mix, unedited, which is the only output a
// compatibility layer can call correct. Attenuate a bring-up run with `--volume N` rather than
// changing the default; see kDefaultVolumePercent in prosper-app/main.cpp.
void set_sdl3_audio_gain(float gain);

// Read back what SDL actually holds for a port: `channel` is the guest's own per-stream gain and
// `amplifier` the host's device gain. False when the port has no stream. For tests -- the whole
// point of the split is that the guest cannot move the host's setting, and an assertion has to be
// able to observe BOTH to say so.
bool sdl3_audio_port_gains_for_test(int port, float* channel, float* amplifier);

// Uninstall the SDL3 sink (restores the default), joining diagnostics and closing streams.
// Call after guest audio producers stop, BEFORE SDL_Quit/SDL_QuitSubSystem(AUDIO).
// Concurrent external SDL teardown and quit/reinit with retained streams are unsupported.
void shutdown_sdl3_audio_sink();

} // namespace prosper
