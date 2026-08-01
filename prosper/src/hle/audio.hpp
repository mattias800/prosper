// audio.hpp — backend-agnostic audio sink for the sceAudioOut HLE.
//
// The audio HLE (hle_audio.cpp) decodes the PS5 sceAudioOut* calls into port lifecycle
// events + interleaved PCM "grains" and forwards them to a pluggable AudioSink. The core
// is HEADLESS: prosper_core ships a silent, real-time-paced default sink (no dependencies),
// so the guest's audio thread is paced correctly even with no audio device. A concrete
// frontend (e.g. SDL3, see frontends/audio_sdl3/) lives OUTSIDE prosper_core and installs
// itself via audio_set_sink(), keeping prosper_core dependency-free and unit-testable.
#pragma once
#include <cmath>
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

// Reserve one AudioOut2 device-queue slot. Kept as a pure transition so queue accounting can be
// verified independently of the wall-clock drain performed by sceAudioOut2ContextGetQueueLevel.
bool audio2_reserve_queue_slot(uint32_t& queued, uint32_t queue_depth);

// --- Multichannel MAIN bed -> stereo host sink -------------------------------------------------
// prosper's host sink is stereo, so every MAIN port wider than two channels has to be folded down.
// The fold is expressed as one gain pair per SOURCE channel rather than inline in the mix loop:
// it is a pure function of the layout, so it is unit-testable without a device or a guest, and a
// channel prosper cannot place is then representable as {0, 0} instead of being silently smeared
// into the bed at a guessed position.
struct AudioStereoGain { float left = 0.0f, right = 0.0f; };

// Fill out[0..channels) with the stereo fold-down gains for a `channels`-wide MAIN bed and return
// how many of those channels prosper could NOT place. A non-zero return is a fail-visible gap, not
// a rounding detail: those channels contribute nothing and the caller must report them.
// `channels` above kAudioMaxBedChannels, or an out_capacity too small, places nothing and returns
// `channels`.
//
// A channel given a zero gain on a side must NOT be read on that side: `Inf * 0.0f` and
// `NaN * 0.0f` are both NaN, and the output path clamps NaN to zero, so multiplying every channel
// by both gains lets one non-finite sample destroy a side whose own content is fine.
//
// CONFIDENCE: HIGH that the bed's first two channels are the front pair; **MED for which of a pair
// is the LEFT one** — prosper's live measurement groups channels by side but cannot orient the
// groups (correlation is symmetric under a swap), so orientation rests on the index-0 = FrontLeft
// convention rather than on evidence. See hle_audio.cpp for the measurement and the experiment that
// would settle it. MED for the 3..8 placements; 9..16 places only its first eight.
constexpr unsigned kAudioMaxBedChannels = 16;
unsigned audio_stereo_downmix(unsigned channels, AudioStereoGain* out, unsigned out_capacity);

// Non-silence measure for submitted PCM (see PROSPER_AUDIO_FLOW in hle_audio.cpp). Diagnosing a
// silent title requires separating "the guest submitted nothing", "the guest submitted zeros" and
// "the guest submitted real signal we then lost", and no single statistic does that:
//   peak    catches a lone transient but says nothing about how much signal is present — one stray
//           sample in an otherwise empty buffer reports the same peak as a full mix;
//   rms()   separates an audible mix from a denormal noise floor, but is a threshold judgement: any
//           cutoff you pick calls some genuinely-mixed quiet passage "silent";
//   nonzero counts samples differing from exactly zero. It is the only EXACT member of the three,
//           and the only one that separates the two cases the caller must not confuse: a buffer the
//           guest cleared has nonzero==0 exactly, whereas real programme material — however quiet —
//           has nonzero close to samples. Note it is exactness, not print precision, that matters:
//           a level low enough to print as 0.00000 can still be unambiguously non-silent here.
// Kept as a pure accumulator so the decision rule the diagnostic rests on is unit-testable
// without a device, a guest, or a live boot.
struct AudioSignalStats {
    double   peak    = 0.0;
    double   sumsq   = 0.0;
    uint64_t samples = 0;
    uint64_t nonzero = 0;

    void add(float v) {
        const double d = (double)v;
        const double a = d < 0 ? -d : d;
        if (a > peak) peak = a;
        sumsq += d * d;
        ++samples;
        if (v != 0.0f) ++nonzero;
    }
    double rms() const { return samples ? std::sqrt(sumsq / (double)samples) : 0.0; }
    // True only for a buffer that is exactly zero everywhere — the signature of a guest that
    // submitted a cleared buffer, as opposed to one that mixed something quiet.
    bool silent() const { return nonzero == 0; }
    void reset() { peak = 0.0; sumsq = 0.0; samples = 0; nonzero = 0; }
};

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
