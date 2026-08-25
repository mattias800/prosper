// audio_sink_sdl3.cpp — SDL3-backed AudioSink for the sceAudioOut HLE (optional frontend).
//
// Enabled with -DPROSPER_AUDIO_SDL3=ON. Bridges the headless AudioSink interface (audio.hpp)
// to SDL3's audio-stream API: one SDL_AudioStream per PS5 audio port, fed the guest's PCM
// grains. FILL-THEN-START: the stream is created UNBOUND (no device attached) and the
// playback device is attached only once ~8 grains are buffered -- the guest's mixer delivers
// on its own cadence with tick-quantized jitter, and a device bound from the first grain
// would drain the buffer to zero between deliveries and underrun continuously
// (Blasphemous 2 PPSA13579 on Windows/RTX 4090, #2985).
#include "audio_sdl3.hpp"
#include "hle/audio/audio.hpp"
#include "host/platform/lifecycle.hpp"
#include "host/platform/precise_sleep.hpp"   // waits must not inherit the Win32 tick (#1765)

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

    // FILL-THEN-START: the stream is created UNBOUND and the playback device is attached
    // lazily by output() once ~8 grains (~130 ms at the title's grain size) are buffered.
    // The unbound stream buffers without draining, so the cushion actually builds.
    bool open(int port, const AudioPortInfo& info) override {
        if (port < 1 || port > kMaxPorts) return false;
        std::lock_guard<std::mutex> lk(slots_[port - 1].mx);
        Slot& s = slots_[port - 1];
        if (s.stream) { SDL_DestroyAudioStream(s.stream); s.stream = nullptr; }
        s.devid = 0; s.bound = false;
        SDL_AudioSpec spec{};
        spec.format   = to_sdl_format(info.fmt);
        spec.channels = info.channels;
        spec.freq     = info.freq;
        s.stream = SDL_CreateAudioStream(&spec, &spec);
        if (!s.stream) { SDL_Log("prosper-audio: CreateAudioStream failed: %s", SDL_GetError()); return false; }
        SDL_SetAudioStreamFormat(s.stream, &spec, &spec);
        s.frame_bytes = audio_frame_bytes(info);
        s.grain_bytes = audio_grain_bytes(info);
        s.freq = info.freq;
        s.put_failed = false;
        s.next = {};   // (re)start the per-grain pacing clock on the first output()
        SDL_SetAudioStreamGain(s.stream, gain_);
        SDL_Log("prosper-audio: opened port %d (unbound, fill-then-start), %d Hz/%d channel%s",
                port, info.freq, info.channels, info.channels == 1 ? "" : "s");
        return true;
    }

    void output(int port, const void* pcm, int frames) override {
        if (port < 1 || port > kMaxPorts) return;
        // Block before touching the SDL queue. Pausing the device preserves samples that were
        // already accepted; this gate prevents guest audio threads from filling it while paused.
        if (!prosper_wait_while_paused()) return;
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
        // Queue-depth cap: hold ~24 grains (~125 ms) buffered. The mixer's three threads
        // wake in lockstep on the same boundary and deliver 3 grains at once, then go
        // silent for a full period -- a shallow FIFO empties between the clusters and the
        // device starves (the measured micro-stutters). A deep FIFO bridges the cluster
        // gaps: the device pulls smoothly while the mixer refills it each period.
        while (SDL_GetAudioStreamAvailable(s.stream) > s.grain_bytes * 24)
            prosper::host::sleep_until_steady_ns(
                (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count() +
                2000000ull);
        const int put_bytes = frames * s.frame_bytes;
        if (!SDL_PutAudioStreamData(s.stream, pcm, put_bytes) && !s.put_failed) {
            SDL_Log("prosper-audio: PutAudioStreamData failed on port %d: %s", port, SDL_GetError());
            s.put_failed = true;
        }
        // Fill-then-start: attach the playback device once ~8 grains (~130 ms at the title's
        // grain size) are buffered. The unbound stream buffers without draining, so the
        // cushion builds; from here on it absorbs every late mixer wake.
        if (!s.bound && SDL_GetAudioStreamAvailable(s.stream) >= s.grain_bytes * 8) {
            SDL_AudioSpec spec{};
            spec.format = to_sdl_format(AudioFmt::F32);   // replaced below with the real fmt
            SDL_AudioSpec unused{};
            (void)unused; (void)spec;
            s.devid = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, nullptr);
            if (!s.devid) {
                SDL_Log("prosper-audio: OpenAudioDevice failed: %s", SDL_GetError());
                return;
            }
            if (!SDL_BindAudioStream(s.devid, s.stream)) {
                SDL_Log("prosper-audio: BindAudioStream failed: %s", SDL_GetError());
                SDL_CloseAudioDevice(s.devid); s.devid = 0;
                return;
            }
            s.bound = true;
            SDL_Log("prosper-audio: port %d cushion filled (%d bytes) -- device attached",
                    port, SDL_GetAudioStreamAvailable(s.stream));
        }
        // Clock-drift compensation by frame duplication (#2985). The guest's budgeted
        // audio clock and the WASAPI device crystal drift ~0.8% against each other; the
        // queue's natural equilibrium would be 0-1 grain, so the device starves between
        // deliveries no matter how deep the cushion starts. SDL's frequency ratio is
        // ignored for device-bound streams at matching frequencies (measured: a 0.5x
        // ratio did not change the drain). Bang-bang instead: when the level drops below
        // the target, deliver the grain PLUS its last frame again (+1 frame ≈ +0.8% at
        // 1 frame per ~125 grains -- inaudible); above it, plain.
        const int level_after = SDL_GetAudioStreamAvailable(s.stream);
        const int target_bytes = s.grain_bytes * 4;
        if (s.bound && level_after < target_bytes) {
            // The queue drained below the target: the guest's audio clock runs ~0.8%
            // slow vs the device crystal. Bridge by repeating the grain's last frame
            // until the level reaches the target (bounded); the next delivery
            // re-checks, so this self-regulates around the target. A repeated 1/48k s
            // sample ~1.5 times per grain period is inaudible; the underruns are not.
            int level = level_after;
            const uint8_t* last_frame =
                (const uint8_t*)pcm + (frames - 1) * s.frame_bytes;
            for (int dup = 0; dup < 4 && level < target_bytes; dup++) {
                SDL_PutAudioStreamData(s.stream, last_frame, s.frame_bytes);
                level += s.frame_bytes;
            }
        }
    }

    void set_volume(int port, uint32_t mask, const int* vols) override {
        if (port < 1 || port > kMaxPorts || !vols) return;
        int maxv = audio_peak_channel_volume(mask, vols);
        float gain = maxv / 32768.0f;                     // SCE_AUDIO_VOLUME_0DB == 32768
        std::lock_guard<std::mutex> lk(slots_[port - 1].mx);
        if (SDL_AudioStream* st = slots_[port - 1].stream) SDL_SetAudioStreamGain(st, gain);
    }

    void close(int port) override {
        if (port < 1 || port > kMaxPorts) return;
        std::lock_guard<std::mutex> lk(slots_[port - 1].mx);
        Slot& s = slots_[port - 1];
        if (s.stream) {
            if (s.bound) SDL_UnbindAudioStream(s.stream);
            SDL_DestroyAudioStream(s.stream); s.stream = nullptr;
        }
        if (s.devid) { SDL_CloseAudioDevice(s.devid); s.devid = 0; }
        s.frame_bytes = s.grain_bytes = 0; s.bound = false;
    }

    // Applies to open streams immediately AND is remembered for later opens, so it works

    // whether it is set before or after the guest creates its ports.

    void set_gain(float g) {
        gain_ = g;
        for (auto& slot : slots_) {
            std::lock_guard<std::mutex> lk(slot.mx);
            if (slot.stream) SDL_SetAudioStreamGain(slot.stream, gain_);
        }
    }

    void set_paused(bool paused) {
        // The app-level pause is enforced by output()'s prosper_wait_while_paused gate;
        // paused_ only records the state. The device (if bound) keeps playing what is
        // queued and then idles -- the gate stops new grains, so no stale queue replays.
        paused_ = paused;
    }

    // PROSPER_AUDIO_QUEUE_TRACE=1: spawn the 1 ms queue-level sampler (queue_trace_loop).
    // CAVEAT (measured): the sampler's per-port mutex contends with output() and slows the
    // delivery ~30% -- for rate measurements prefer PROSPER_AUDIO_DEBUG instead.
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
                  SDL_AudioDeviceID devid = 0;
                  bool bound = false; };   // fill-then-start: unbound until the cushion fills
    std::array<Slot, kMaxPorts> slots_{};
    bool paused_ = false;
    float gain_ = 1.0f;   // linear playback gain, applied via SDL_SetAudioStreamGain

    // PROSPER_AUDIO_QUEUE_TRACE=1: a 1 ms sampler of every port's queued-byte level. The
    // timeline is the direct evidence for an underrun hunt: the drain slope names the device
    // consumption, the refill bursts name the mixer's delivery pattern, and the zero
    // crossings are the audible underruns -- no ear test required.
    // CAVEAT (measured): the sampler's per-port mutex contends with output() and slows the
    // delivery ~30% -- for rate measurements prefer PROSPER_AUDIO_DEBUG instead.
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
