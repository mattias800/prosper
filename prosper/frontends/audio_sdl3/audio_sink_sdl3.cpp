// audio_sink_sdl3.cpp — SDL3-backed AudioSink for the sceAudioOut HLE (optional frontend).
//
// Enabled with -DPROSPER_AUDIO_SDL3=ON. Bridges the headless AudioSink interface (audio.hpp)
// to SDL3's audio-stream API: one SDL_AudioStream per PS5 audio port, fed the guest's PCM
// grains. output() blocks while the device's queue is full, reproducing the pacing that
// sceAudioOutOutput has on real hardware (it blocks until the audio ring has room).
#include "audio_sdl3.hpp"
#include "hle/audio/audio.hpp"
#include "host/platform/lifecycle.hpp"
#include "host/platform/precise_sleep.hpp"   // grain pacing must not inherit the Win32 tick (#1765)

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
        SDL_SetAudioStreamGain(s.stream, gain_);
        // FILL-THEN-START: the device opens PAUSED. The guest's mixer delivers one grain per
        // mix period (~16 ms) with tick-quantized wake jitter; a device that starts on the
        // first grain holds ~1 grain of cushion and underruns on every jitter. output()
        // resumes the device only once ~8 grains (~130 ms) are queued, and that cushion then
        // absorbs the mixer's jitter for the rest of the session (#2985, Blasphemous 2 FMVs).
        const bool stream_ready = SDL_PauseAudioStreamDevice(s.stream);
        s.device_paused = true;
        if (!stream_ready) {
            SDL_Log("prosper-audio: PauseAudioStreamDevice failed: %s", SDL_GetError());
            SDL_DestroyAudioStream(s.stream);
            s.stream = nullptr;
            return false;
        }
        const SDL_AudioDeviceID device = SDL_GetAudioStreamDevice(s.stream);
        SDL_Log("prosper-audio: opened port %d on %s (%s), %d Hz/%d channel%s (fill-then-start)",
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
        {
            std::lock_guard<std::mutex> lk(slots_[port - 1].mx);
            Slot& s = slots_[port - 1];
            if (!s.stream) return;
            // PROSPER_AUDIO_DEBUG=1: per-call arrival cadence + queue level, for underrun
            // forensics -- distinguishes "the guest delivered late" from "the device drained
            // early" (user report: micro-stutters identical before/after every pacing fix).
            static const bool audio_dbg = getenv("PROSPER_AUDIO_DEBUG") != nullptr;
            const int avail_before = SDL_GetAudioStreamAvailable(s.stream);
            if (audio_dbg) {
                static thread_local std::chrono::steady_clock::time_point last{};
                const auto now = std::chrono::steady_clock::now();
                const double gap = last.time_since_epoch().count() == 0
                    ? 0.0
                    : std::chrono::duration<double, std::milli>(now - last).count();
                last = now;
                std::fprintf(stderr,
                             "[audio-dbg] port=%d gap=%.2fms frames=%d queued_before=%d"
                             " (grain=%d)\n",
                             port, gap, frames, avail_before, s.grain_bytes);
            }
            const int cushion_bytes = s.grain_bytes * 8;
            int spins = 0;
            while (SDL_GetAudioStreamAvailable(s.stream) > cushion_bytes && spins++ < 100)
                prosper::host::sleep_until_steady_ns(
                    (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count() +
                    2000000ull);
            if (!SDL_PutAudioStreamData(s.stream, pcm, frames * s.frame_bytes) && !s.put_failed) {
                SDL_Log("prosper-audio: PutAudioStreamData failed on port %d: %s", port, SDL_GetError());
                s.put_failed = true;
            }
            // Fill-then-start: hold the device paused until the cushion is deep enough to
            // absorb the mixer's tick-quantized wake jitter (~8 grains ≈ 130 ms), then let it
            // run. From here on the cushion absorbs every late mixer wake.
            if (s.device_paused &&
                SDL_GetAudioStreamAvailable(s.stream) >= s.grain_bytes * 8) {
                if (SDL_ResumeAudioStreamDevice(s.stream)) {
                    s.device_paused = false;
                }
            }
            // Clock-drift compensation (#2985): the guest's audio clock (budgeted from the
            // flip timeline) and the WASAPI device clock are independent crystals; the ~0.8%
            // deficit drained every cushion at a constant rate no matter how deep. Nudge the
            // stream's frequency ratio toward the level target -- a <1% pitch shift, inaudible,
            // and the cushion holds for the whole session.
            {
                const int level = SDL_GetAudioStreamAvailable(s.stream);
                const int target_level = s.grain_bytes * 6;
                s.freq_ratio += 0.000001 * (double)(level - target_level);
                if (s.freq_ratio < 0.97) s.freq_ratio = 0.97;
                if (s.freq_ratio > 1.03) s.freq_ratio = 1.03;
                SDL_SetAudioStreamFrequencyRatio(s.stream, (float)s.freq_ratio);
            }
        }
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

    // Applies to open streams immediately AND is remembered for later opens, so it works

    // whether it is set before or after the guest creates its ports.

    void set_gain(float g) {
        std::lock_guard<std::mutex> lk(mx_);
        gain_ = g;
        for (auto& s : slots_) if (s.stream) SDL_SetAudioStreamGain(s.stream, gain_);
    }

    void set_paused(bool paused) {
        std::lock_guard<std::mutex> lk(mx_);
        if (paused_ == paused) return;
        paused_ = paused;
        int active_streams = 0;
        for (Slot& s : slots_) {
            if (!s.stream) continue;
            ++active_streams;
            if (paused) {
                SDL_PauseAudioStreamDevice(s.stream);
                s.device_paused = true;
            } else {
                // Resume goes through the fill-then-start path in output(): the cushion
                // refills before the device runs, so a stale queue is never replayed.
                s.device_paused = true;
                s.next = {};   // resume pacing from now; never catch up in a burst
            }
        }
        SDL_Log("prosper-audio: %s %d active stream%s", paused ? "paused" : "resumed",
                active_streams, active_streams == 1 ? "" : "s");
    }

    // PROSPER_AUDIO_QUEUE_TRACE=1: spawn the 1 ms queue-level sampler (queue_trace_loop).
    void start_queue_trace_if_requested() {
        if (!getenv("PROSPER_AUDIO_QUEUE_TRACE")) return;
        g_trace_running.store(true, std::memory_order_relaxed);
        std::thread(&Sdl3AudioSink::queue_trace_loop, this).detach();
        SDL_Log("prosper-audio: queue-level trace started");
    }

private:
    struct Slot { SDL_AudioStream* stream = nullptr; int frame_bytes = 0; int grain_bytes = 0;
                  bool put_failed = false;
                  int freq = 0;                                     // port sample rate for pacing
                  std::chrono::steady_clock::time_point next{};     // per-grain pacing deadline
                  double freq_ratio = 1.0;                          // clock-drift compensation
                  // Per-port lock: the queue-depth wait in output() can hold this for a couple
                  // of device drain periods; a GLOBAL lock made port 18's puts wait for port
                  // 17's pacing and micro-stuttered every other audio source (#2985 follow-up).
                  std::mutex mx;
                  bool device_paused = true; };   // fill-then-start: paused until the cushion fills
    std::mutex mx_;   // guards gain_/paused_ and the multi-slot set_gain/set_paused walks
    std::array<Slot, kMaxPorts> slots_{};
    bool paused_ = false;
    float gain_ = 1.0f;   // linear playback gain, applied via SDL_SetAudioStreamGain

    // PROSPER_AUDIO_QUEUE_TRACE=1: a 1 ms sampler of every port's queued-byte level. The
    // timeline is the direct evidence for an underrun hunt: the drain slope names the device
    // consumption, the refill bursts name the mixer's delivery pattern, and the zero
    // crossings are the audible underruns -- no ear test required.
    void queue_trace_loop() {
        const auto t0 = std::chrono::steady_clock::now();
        while (g_trace_running.load(std::memory_order_relaxed)) {
            const double t = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            for (int i = 0; i < kMaxPorts; i++) {
                std::lock_guard<std::mutex> lk(slots_[i].mx);
                if (!slots_[i].stream) continue;
                std::fprintf(stderr, "[audio-queue] t=%.3f port=%d avail=%d\n",
                             t, i + 1, SDL_GetAudioStreamAvailable(slots_[i].stream));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    std::atomic<bool> g_trace_running{false};
};

Sdl3AudioSink g_sink;
bool g_installed = false;

} // namespace

bool install_sdl3_audio_sink() {
    if (g_installed) return true;
    if (!g_sink.init()) return false;
    audio_set_sink(&g_sink);
    g_installed = true;
    g_sink.start_queue_trace_if_requested();
    SDL_Log("prosper-audio: SDL3 audio backend installed");
    return true;
}

void set_sdl3_audio_gain(float gain) {
    if (gain < 0.0f) gain = 0.0f;
    g_sink.set_gain(gain);   // safe before install: the value is remembered and applied on open
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
