// audio_sink_sdl3.cpp — SDL3-backed AudioSink for the sceAudioOut HLE (optional frontend).
//
// Enabled with -DPROSPER_AUDIO_SDL3=ON. Bridges the headless AudioSink interface (audio.hpp)
// to SDL3's audio-stream API: one SDL_AudioStream per PS5 audio port, fed the guest's PCM
// grains. output() blocks while the device's queue is full, reproducing the pacing that
// sceAudioOutOutput has on real hardware (it blocks until the audio ring has room).
#include "audio_sdl3.hpp"
#include "../../src/hle/audio.hpp"

#include <SDL3/SDL.h>

#include <array>
#include <mutex>
#include <thread>

namespace prosper {
namespace {

constexpr int kMaxPorts = 16;

SDL_AudioFormat to_sdl_format(AudioFmt f) { return f == AudioFmt::F32 ? SDL_AUDIO_F32 : SDL_AUDIO_S16; }

class Sdl3AudioSink : public AudioSink {
public:
    bool init() {
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            SDL_Log("prosper-audio: SDL_InitSubSystem(AUDIO) failed: %s", SDL_GetError());
            return false;
        }
        return true;
    }

    void quit() {
        for (int i = 0; i < kMaxPorts; i++) close(i + 1);
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
    }

    bool open(int port, const AudioPortInfo& info) override {
        if (port < 1 || port > kMaxPorts) return false;
        std::lock_guard<std::mutex> lk(mx_);
        Slot& s = slots_[port - 1];
        if (s.stream) { SDL_DestroyAudioStream(s.stream); s.stream = nullptr; }
        SDL_AudioSpec spec{};
        spec.format   = to_sdl_format(info.fmt);
        spec.channels = info.channels;
        spec.freq     = info.freq;
        // Bind straight to the default playback device; NULL callback => we push data ourselves.
        s.stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
        if (!s.stream) { SDL_Log("prosper-audio: OpenAudioDeviceStream failed: %s", SDL_GetError()); return false; }
        s.frame_bytes = audio_frame_bytes(info);
        s.grain_bytes = audio_grain_bytes(info);
        // Queue cap for output()'s pacing: enough to cover TWO device periods plus two grains of
        // jitter margin, floored at the historical 4 grains. A fixed 4-grain cap (~21 ms at the
        // common 256-frame grain) left ZERO headroom against PipeWire's ~21 ms pull quantum, so any
        // scheduling jitter starved the device once per quantum and playback came out garbled —
        // while WASAPI's ~10 ms periods and PulseAudio's deep server buffering masked the same math
        // on Windows/WSL. Sizing from the real device period keeps latency minimal per platform
        // instead of hard-coding one backend's numbers.
        int device_frames = 0;
        SDL_AudioSpec dev_spec{};
        if (SDL_AudioDeviceID dev = SDL_GetAudioStreamDevice(s.stream))
            SDL_GetAudioDeviceFormat(dev, &dev_spec, &device_frames);
        // device_frames is in DEVICE-rate frames; rescale to the port's rate before mixing units.
        int period_port_frames = (device_frames > 0 && dev_spec.freq > 0)
            ? (int)((int64_t)device_frames * info.freq / dev_spec.freq) : 0;
        int cap = 4 * s.grain_bytes;
        if (period_port_frames > 0) {
            int period_bytes = period_port_frames * s.frame_bytes;
            if (2 * period_bytes + 2 * s.grain_bytes > cap) cap = 2 * period_bytes + 2 * s.grain_bytes;
        }
        s.queue_cap = cap;
        SDL_ResumeAudioStreamDevice(s.stream);
        return true;
    }

    void output(int port, const void* pcm, int frames) override {
        if (port < 1 || port > kMaxPorts) return;
        SDL_AudioStream* stream; int frame_bytes, queue_cap;
        { std::lock_guard<std::mutex> lk(mx_); Slot& s = slots_[port - 1];
          stream = s.stream; frame_bytes = s.frame_bytes; queue_cap = s.queue_cap; }
        if (!stream) return;
        int bytes = frames * frame_bytes;
        SDL_PutAudioStreamData(stream, pcm, bytes);
        // Pace like real hardware: block while the queue is ahead of the device-derived cap
        // (computed in open() — two device periods + jitter margin, min 4 grains).
        const int cap = queue_cap > 0 ? queue_cap : bytes * 4;
        for (int spins = 0; SDL_GetAudioStreamQueued(stream) > cap && spins < 1000; spins++)
            SDL_Delay(1);
    }

    void set_volume(int port, uint32_t mask, const int* vols) override {
        if (port < 1 || port > kMaxPorts || !vols) return;
        // Approximate the PS5 per-channel volumes as a single stream gain (0..1) from the loudest set channel.
        int maxv = audio_peak_channel_volume(mask, vols);
        float gain = maxv / 32768.0f;                     // SCE_AUDIO_VOLUME_0DB == 32768
        std::lock_guard<std::mutex> lk(mx_);
        if (SDL_AudioStream* st = slots_[port - 1].stream) SDL_SetAudioStreamGain(st, gain);
    }

    void close(int port) override {
        if (port < 1 || port > kMaxPorts) return;
        std::lock_guard<std::mutex> lk(mx_);
        Slot& s = slots_[port - 1];
        if (s.stream) { SDL_DestroyAudioStream(s.stream); s.stream = nullptr; }
        s.frame_bytes = s.grain_bytes = 0;
    }

private:
    struct Slot { SDL_AudioStream* stream = nullptr; int frame_bytes = 0; int grain_bytes = 0;
                  int queue_cap = 0; };   // pacing cap in bytes, derived from the device period in open()
    std::mutex mx_;
    std::array<Slot, kMaxPorts> slots_{};
};

Sdl3AudioSink g_sink;
bool g_installed = false;

} // namespace

bool install_sdl3_audio_sink() {
    if (g_installed) return true;
    if (!g_sink.init()) return false;
    audio_set_sink(&g_sink);
    g_installed = true;
    SDL_Log("prosper-audio: SDL3 audio backend installed");
    return true;
}

void shutdown_sdl3_audio_sink() {
    if (!g_installed) return;
    audio_set_sink(nullptr);   // restore the default silent sink
    g_sink.quit();
    g_installed = false;
}

} // namespace prosper
