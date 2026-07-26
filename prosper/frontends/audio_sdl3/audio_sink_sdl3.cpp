// audio_sink_sdl3.cpp — SDL3-backed AudioSink for the sceAudioOut HLE (optional frontend).
//
// Enabled with -DPROSPER_AUDIO_SDL3=ON. Bridges the headless AudioSink interface (audio.hpp)
// to SDL3's audio-stream API: one SDL_AudioStream per PS5 audio port, fed the guest's PCM
// grains. output() blocks while the device's queue is full, reproducing the pacing that
// sceAudioOutOutput has on real hardware (it blocks until the audio ring has room).
#include "audio_sdl3.hpp"
#include "../../src/hle/audio.hpp"
#include "../../src/host/lifecycle.hpp"

#include <SDL3/SDL.h>

#include <array>
#include <chrono>
#include <mutex>
#include <thread>

namespace prosper {
namespace {

// 16 public sceAudioOut ports plus four host-only streams for concurrent AudioOut2 contexts.
// SDL mixes the bound streams into the same playback device while each retains its own pacing
// clock; this mirrors independent hardware contexts without serializing their sample timelines.
constexpr int kMaxPorts = 20;

SDL_AudioFormat to_sdl_format(AudioFmt f) { return f == AudioFmt::F32 ? SDL_AUDIO_F32 : SDL_AUDIO_S16; }

class Sdl3AudioSink : public AudioSink {
public:
    bool init() {
        paused_ = false;
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
        s.freq = info.freq;
        s.put_failed = false;
        s.next = {};   // (re)start the per-grain pacing clock on the first output()
        const bool stream_ready = paused_ ? SDL_PauseAudioStreamDevice(s.stream)
                                          : SDL_ResumeAudioStreamDevice(s.stream);
        if (!stream_ready) {
            SDL_Log("prosper-audio: %sAudioStreamDevice failed: %s",
                    paused_ ? "Pause" : "Resume", SDL_GetError());
            SDL_DestroyAudioStream(s.stream);
            s.stream = nullptr;
            return false;
        }
        const SDL_AudioDeviceID device = SDL_GetAudioStreamDevice(s.stream);
        SDL_Log("prosper-audio: opened port %d on %s (%s), %d Hz/%d channel%s",
                port, SDL_GetAudioDeviceName(device) ? SDL_GetAudioDeviceName(device) : "default device",
                SDL_GetCurrentAudioDriver() ? SDL_GetCurrentAudioDriver() : "unknown driver",
                info.freq, info.channels, info.channels == 1 ? "" : "s");
        return true;
    }

    void output(int port, const void* pcm, int frames) override {
        if (port < 1 || port > kMaxPorts) return;
        // Block before touching the SDL queue. Pausing the device preserves samples that were
        // already accepted; this gate prevents guest audio threads from filling it while paused.
        if (!prosper_wait_while_paused()) return;
        int freq = 0;
        std::chrono::steady_clock::time_point target{};
        {
            std::lock_guard<std::mutex> lk(mx_);
            Slot& s = slots_[port - 1];
            if (!s.stream) return;
            freq = s.freq;
            if (freq > 0) {
                // Pace EACH call one grain of wall-clock time apart (real sceAudioOutOutput blocks
                // until the hardware ring frees one grain — evenly spaced, never bursty). The old
                // cap-based pacing let the guest burst a whole device quantum (4+ grains) and then
                // stall: FMOD signals its mixer's condvar once per submitted grain, condvar signals
                // COALESCE within a burst, so the mixer woke once per ~21 ms instead of once per
                // grain and mixed at 1/4 real time — every DSP block audibly replayed ~4x (#1016).
                // Same resync-if-behind model as the core's RealtimeSilentSink.
                auto dur = std::chrono::nanoseconds((int64_t)frames * 1000000000LL / freq);
                auto now = std::chrono::steady_clock::now();
                if (s.next.time_since_epoch().count() == 0 || s.next < now - dur * 4) s.next = now;
                s.next += dur;
                target = s.next;
            }
            // Keep the stream protected until SDL has consumed this call. sceAudioOutOutput and
            // sceAudioOutClose may run on different guest audio threads; copying s.stream out of
            // the lock let close()/open()/quit() destroy it immediately before this call (#855).
            // The pacing sleep remains outside the lock, so lifecycle calls wait only for SDL's
            // synchronous queue copy rather than for an entire audio grain.
            if (!SDL_PutAudioStreamData(s.stream, pcm, frames * s.frame_bytes) && !s.put_failed) {
                SDL_Log("prosper-audio: PutAudioStreamData failed on port %d: %s", port, SDL_GetError());
                s.put_failed = true;
            }
        }
        if (freq > 0) std::this_thread::sleep_until(target);
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

    void set_paused(bool paused) {
        std::lock_guard<std::mutex> lk(mx_);
        if (paused_ == paused) return;
        paused_ = paused;
        int active_streams = 0;
        for (Slot& s : slots_) {
            if (!s.stream) continue;
            ++active_streams;
            const bool ok = paused ? SDL_PauseAudioStreamDevice(s.stream)
                                   : SDL_ResumeAudioStreamDevice(s.stream);
            if (!ok)
                SDL_Log("prosper-audio: %sAudioStreamDevice failed: %s",
                        paused ? "Pause" : "Resume", SDL_GetError());
            if (!paused) s.next = {};   // resume pacing from now; never catch up in a burst
        }
        SDL_Log("prosper-audio: %s %d active stream%s", paused ? "paused" : "resumed",
                active_streams, active_streams == 1 ? "" : "s");
    }

private:
    struct Slot { SDL_AudioStream* stream = nullptr; int frame_bytes = 0; int grain_bytes = 0;
                  bool put_failed = false;
                  int freq = 0;                                     // port sample rate for pacing
                  std::chrono::steady_clock::time_point next{}; };  // per-grain pacing deadline
    std::mutex mx_;
    std::array<Slot, kMaxPorts> slots_{};
    bool paused_ = false;
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

void set_sdl3_audio_paused(bool paused) {
    if (g_installed) g_sink.set_paused(paused);
}

void shutdown_sdl3_audio_sink() {
    if (!g_installed) return;
    audio_set_sink(nullptr);   // restore the default silent sink
    g_sink.quit();
    g_installed = false;
}

} // namespace prosper
