// audio.hpp — backend-agnostic audio sink for the sceAudioOut HLE.
//
// The audio HLE (hle_audio.cpp) decodes the PS5 sceAudioOut* calls into port lifecycle
// events + interleaved PCM "grains" and forwards them to a pluggable AudioSink. The core
// is HEADLESS: prosper_core ships a silent, real-time-paced default sink (no dependencies),
// so the guest's audio thread is paced correctly even with no audio device. A concrete
// frontend (e.g. SDL3, see frontends/audio_sdl3/) lives OUTSIDE prosper_core and installs
// itself via audio_set_sink(), keeping prosper_core dependency-free and unit-testable.
#pragma once
#include <cstdint>

namespace prosper {

// Sample format, decoded from sceAudioOutOpen's `param`.
enum class AudioFmt : int { S16 = 0, F32 = 1 };

// Fully-decoded parameters of an open audio port.
struct AudioPortInfo {
    int      freq     = 48000;          // sample rate (Hz)
    int      channels = 2;              // 1 (mono), 2 (stereo), 8 (7.1)
    AudioFmt fmt      = AudioFmt::S16;  // interleaved sample format
    int      grain    = 256;            // frames delivered per sceAudioOutOutput call (the open `len`)
};

inline int audio_bytes_per_sample(AudioFmt f) { return f == AudioFmt::F32 ? 4 : 2; }
inline int audio_frame_bytes(const AudioPortInfo& p) { return p.channels * audio_bytes_per_sample(p.fmt); }
inline int audio_grain_bytes(const AudioPortInfo& p) { return p.grain * audio_frame_bytes(p); }

// Decode a sceAudioOutOpen `param` word into channels + sample format. The low byte is the
// SceAudioOutParamFormat enum; higher bits are attributes (ignored here). Unknown -> S16 stereo.
void audio_decode_format(uint32_t param, int& channels, AudioFmt& fmt);

// Sony's volume array is channel-indexed even when `mask` is sparse: bit i selects vols[i], not
// the next compacted element. Shared helpers keep the HLE cache and concrete sinks on that contract.
void audio_apply_channel_volumes(int dst[8], uint32_t mask, const int* vols);
int audio_peak_channel_volume(uint32_t mask, const int* vols);

// Pluggable audio backend. All calls arrive on the guest's audio thread; output() MAY block to
// pace it (as real hardware does — sceAudioOutOutput blocks until the ring has room).
struct AudioSink {
    virtual ~AudioSink() = default;
    // Open (or reconfigure) a port. Return true on success. Called before any output().
    virtual bool open(int port, const AudioPortInfo& info) { (void)port; (void)info; return true; }
    // Deliver one grain (info.grain frames) of interleaved PCM. `pcm` is host-readable, non-owning.
    virtual void output(int port, const void* pcm, int frames) = 0;
    // Per-channel volume in [0, 32768] (SCE_AUDIO_VOLUME_0DB). `mask` selects which channels;
    // `vols` is a host-readable eight-element channel-indexed array (bit i applies vols[i]).
    virtual void set_volume(int port, uint32_t mask, const int* vols) { (void)port; (void)mask; (void)vols; }
    virtual void close(int port) { (void)port; }
};

// Install a backend. Non-owning; pass nullptr to restore the built-in silent/real-time sink.
void audio_set_sink(AudioSink* sink);
AudioSink* audio_sink();

// Close all ports and restore the default sink. Intended for tests.
void audio_reset();

} // namespace prosper
