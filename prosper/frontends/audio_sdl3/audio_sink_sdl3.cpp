// audio_sink_sdl3.cpp — SDL3-backed AudioSink for the sceAudioOut HLE (optional frontend).
//
// Enabled with -DPROSPER_AUDIO_SDL3=ON. Bridges the headless AudioSink interface (audio.hpp)
// to SDL3's audio-stream API: one SDL_AudioStream per PS5 audio port, fed the guest's PCM
// grains. output() blocks while the device's queue is full, reproducing the pacing that
// sceAudioOutOutput has on real hardware (it blocks until the audio ring has room).
#include "audio_sdl3.hpp"
#include "hle/audio/audio.hpp"
#include "host/platform/lifecycle.hpp"
#include "host/platform/precise_sleep.hpp"   // the grain pacer must not quantize to the winpthreads tick (#3016)

#include <SDL3/SDL.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>   // getenv, for the opt-in gates below (transitive via libstdc++, not via libc++)
#include <mutex>
#include <thread>
#include <string>

namespace prosper {
namespace {

// Both off by default: reading the device queue at every handoff is cheap, but an instrument that
// runs unasked is how a measurement pass ends up measuring itself (#2113).
bool legacy_pacer() {
    static const bool on = getenv("PROSPER_GUEST_SLEEP_LEGACY") != nullptr;
    return on;
}
// PROSPER_AUDIO_QUEUE_TIMELINE[=<ms>]: sampling interval for the queue-level timeline, 0 = off.
// Its OWN variable rather than folded into PROSPER_AUDIO_QUEUE_TRACE, which already means the
// per-second min/max summary (#3016): the two differ by three orders of magnitude in output
// volume -- roughly 1,000 lines per second per port here against one -- and overloading one
// switch would make a cheap summary and a firehose indistinguishable at the call site.
int queue_timeline_ms() {
    static const int ms = [] {
        const char* e = getenv("PROSPER_AUDIO_QUEUE_TIMELINE");
        if (!e || !*e) return 0;
        // A malformed or zero value DISABLES, per the convention CLAUDE.md states for these
        // triggers: a typo must cost you the measurement, never fire one you did not ask for.
        // The previous form returned 1 for anything atoi could not parse -- so `=0`, documented
        // as off, armed the 1 ms firehose, failing in the loud direction.
        char* end = nullptr;
        const long v = strtol(e, &end, 10);
        if (!end || *end || v <= 0 || v > 60000) {
            SDL_Log("prosper-audio: PROSPER_AUDIO_QUEUE_TIMELINE=%s is not a positive"
                    " millisecond count; timeline disabled", e);
            return 0;
        }
        return (int)v;
    }();
    return ms;
}

bool audio_debug() {
    static const bool on = getenv("PROSPER_AUDIO_DEBUG") != nullptr;
    return on;
}
bool queue_trace() {
    static const bool on = getenv("PROSPER_AUDIO_QUEUE_TRACE") != nullptr;
    return on;
}


// 16 public sceAudioOut ports plus four host-only streams for concurrent AudioOut2 contexts.
// SDL mixes the bound streams into the same playback device while each retains its own pacing
// clock; this mirrors independent hardware contexts without serializing their sample timelines.
// 28 covers the 16 public sceAudioOut ports plus the four host-only AudioOut2 context ports
// (kA2SinkPortBase = kMaxPorts+1 = 17..20 in hle_audio.cpp) with headroom. NOTE: the previous
// cap of 20 already accepted ports 17..20, so it did NOT discard GRIS's mix — that claim was
// wrong; the actual cause of GRIS's silence was the DecodeSplit job rejection (#2981).
constexpr int kMaxPorts = 28;

SDL_AudioFormat to_sdl_format(AudioFmt f) { return f == AudioFmt::F32 ? SDL_AUDIO_F32 : SDL_AUDIO_S16; }

// PROSPER_AUDIO_DUMP_WAV=<path.wav>: dump every port's mixed PCM to per-port WAV
// files. Distinct from the core's PROSPER_AUDIO_DUMP (<prefix>.portN.raw) so both
// can coexist without double-dumping one variable.
// (<path> with the port number inserted before the extension). Measures what the guest
// actually mixed — immune to host volume/mute — so "is audio playing" is a measurement,
// not an impression (#2981).
class WavDump {
public:
    bool open(const char* path, int freq, int channels, bool f32) {
        if (!path || !*path) return false;
        finalize();   // a reopen mid-run finalizes the earlier capture instead of leaking
        file_ = fopen(path, "wb");
        if (!file_) return false;
        const uint16_t fmt_tag = f32 ? 3 : 1;
        const uint16_t block = static_cast<uint16_t>(channels * (f32 ? 4 : 2));
        const uint32_t byte_rate = static_cast<uint32_t>(freq) * block;
        auto w16 = [&](uint16_t v) { fwrite(&v, 1, 2, file_); };
        auto w32 = [&](uint32_t v) { fwrite(&v, 1, 4, file_); };
        fwrite("RIFF", 1, 4, file_); w32(0); fwrite("WAVE", 1, 4, file_);
        fwrite("fmt ", 1, 4, file_); w32(16); w16(fmt_tag);
        w16(static_cast<uint16_t>(channels));
        w32(static_cast<uint32_t>(freq)); w32(byte_rate); w16(block); w16(f32 ? 32 : 16);
        fwrite("data", 1, 4, file_); w32(0);
        data_size_pos_ = 40;   // "data" chunk size field: 36..39 is the chunk id, 40..43 the size
        return true;
    }
    void write(const void* pcm, int frames, int channels, bool f32) {
        if (!file_) return;
        const size_t bytes = static_cast<size_t>(frames) * channels * (f32 ? 4 : 2);
        fwrite(pcm, 1, bytes, file_);
        data_bytes_ += static_cast<uint32_t>(bytes);
    }
    void finalize() {
        if (!file_) return;
        fseek(file_, 4, SEEK_SET);
        uint32_t riff = 36 + data_bytes_; fwrite(&riff, 1, 4, file_);
        fseek(file_, data_size_pos_, SEEK_SET); fwrite(&data_bytes_, 1, 4, file_);
        fclose(file_); file_ = nullptr;
    }
    bool active() const { return file_ != nullptr; }
    ~WavDump() { finalize(); }
    WavDump() = default;                // the deleted copy suppresses the implicit default
    WavDump(const WavDump&) = delete;
    WavDump& operator=(const WavDump&) = delete;
private:
    FILE* file_ = nullptr;
    long data_size_pos_ = 0;
    uint32_t data_bytes_ = 0;
};

class Sdl3AudioSink : public AudioSink {
public:
    bool init() {
        paused_ = false;
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            SDL_Log("prosper-audio: SDL_InitSubSystem(AUDIO) failed: %s", SDL_GetError());
            return false;
        }
        start_queue_timeline();   // no-op unless PROSPER_AUDIO_QUEUE_TIMELINE is set
        return true;
    }

    void quit() {
        // Stopped and JOINED first, before any stream is destroyed. Clearing a flag was not
        // enough: it only asks, and the close() below would then race a sampler that had already
        // entered its critical section. The join makes the ordering a fact rather than a hope.
        stop_queue_timeline();
        for (auto& s : slots_) s.dump.finalize();
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
        if (const char* dump = getenv("PROSPER_AUDIO_DUMP_WAV"); dump && *dump) {
            std::string path(dump);
            const auto dot = path.rfind('.');
            const std::string port_tag = std::to_string(port);
            path = dot == std::string::npos ? path + "." + port_tag + ".wav"
                                            : path.substr(0, dot) + "." + port_tag + path.substr(dot);
            s.channels = info.channels;
            s.f32 = info.fmt == AudioFmt::F32;
            if (s.dump.open(path.c_str(), info.freq, info.channels, s.f32))
                SDL_Log("prosper-audio: dumping port %d PCM to %s", port, path.c_str());
        }
        const char* dev_name = SDL_GetAudioDeviceName(device);
        const char* drv_name = SDL_GetCurrentAudioDriver();
        SDL_Log("prosper-audio: opened port %d on %s (%s), %d Hz/%d channel%s",
                port, dev_name ? dev_name : "default device",
                drv_name ? drv_name : "unknown driver",
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
            // PROSPER_AUDIO_DEBUG=1: one line per guest grain delivery -- arrival gap, frame
            // count and the queue level BEFORE the put. This is the emitter that
            // tools/perf/audio_delivery_report.py consumes; that tool landed in #3034 without
            // it, so until now master carried a parser with no producer and the tool could not
            // be run at all (its self-test passes on synthetic records, so nothing went red).
            // Recovered from #2984 e389440a, diagnostic half only -- that commit also changed
            // the device open to fill-then-start, which is a behavioural change and belongs
            // with the pacer decision on #3033, not with an instrument.
            if (audio_debug()) {
                const auto now = std::chrono::steady_clock::now();
                const double gap = s.dbg_last.time_since_epoch().count() == 0
                    ? 0.0
                    : std::chrono::duration<double, std::milli>(now - s.dbg_last).count();
                s.dbg_last = now;
                SDL_Log("[audio-dbg] port=%d gap=%.2fms frames=%d queued_before=%d (grain=%d)",
                        port, gap, frames, SDL_GetAudioStreamQueued(s.stream), s.grain_bytes);
            }
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
            s.dump.write(pcm, frames, s.channels, s.f32);
            // PROSPER_AUDIO_QUEUE_TRACE=1: the device queue depth AT each grain handoff, reported
            // as a per-second MINIMUM rather than a mean. A mean cannot see this defect -- #3016
            // averaged 100% of real time while the queue emptied in every gap -- so the minimum is
            // the whole point, together with the count of handoffs that found less than one grain
            // buffered, which is the moment a real device would underrun.
            if (queue_trace()) {
                const int queued = SDL_GetAudioStreamQueued(s.stream);
                if (queued >= 0) {
                    if (s.q_min < 0 || queued < s.q_min) s.q_min = queued;
                    if (queued > s.q_max) s.q_max = queued;
                    if (s.grain_bytes > 0 && queued < s.grain_bytes) s.q_starved++;
                    s.q_calls++;
                    const auto now2 = std::chrono::steady_clock::now();
                    if (s.q_last.time_since_epoch().count() == 0) s.q_last = now2;
                    if (now2 - s.q_last >= std::chrono::seconds(1)) {
                        SDL_Log("prosper-audio: port %d queue min=%d max=%d bytes (grain=%d) "
                                "handoffs=%llu below-one-grain=%llu",
                                port, s.q_min, s.q_max, s.grain_bytes,
                                (unsigned long long)s.q_calls, (unsigned long long)s.q_starved);
                        s.q_min = -1; s.q_max = 0; s.q_calls = 0; s.q_starved = 0; s.q_last = now2;
                    }
                }
            }

        }
        // Pace the guest thread to real time. This sleep IS the Heaps unqueueBuffer signal
        // (#2978): by the time it returns, the previous grain has been consumed by the audio
        // device, giving Heaps the buffer-completion timing it needs to unqueue SFX buffers
        // instead of re-triggering them. The sleep must be outside the lock so other audio
        // operations (open/close/volume) can proceed while this grain plays.
        if (freq > 0) {
            // NOT std::this_thread::sleep_until (#3016). On Windows that resolves on the
            // winpthreads master tick -- measured 15.6 ms for a 5.33 ms request, and unaffected by
            // timeBeginPeriod -- while one 256-frame grain at 48 kHz IS 5.33 ms. Because `s.next`
            // accumulates an ABSOLUTE target, an overshoot leaves the next few targets already in
            // the past, so those calls returned instantly: delivery clumped into bursts of ~3
            // grains every ~15.6 ms. The one-second AVERAGE stayed at ~100% of real time, which is
            // why this hid from a delivery-rate check, while the device queue drained inside every
            // gap -- continuous small underruns in every title, Windows only.
            //
            // sleep_until_steady_ns uses a high-resolution waitable timer, so the grain boundary is
            // honoured and the pacing stays evenly spaced, which is the property the comment above
            // this function has always depended on.
            // PROSPER_GUEST_SLEEP_LEGACY=1 restores the pre-fix pacer so the A/B stays
            // reproducible, the same way PROSPER_UD_TAIL_ALIGN does. Do not set it to fix anything.
            if (legacy_pacer()) {
                std::this_thread::sleep_until(target);
            } else {
                prosper::host::sleep_until_steady_ns(
                    (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        target.time_since_epoch()).count());
            }

        }

    }

    // PROSPER_AUDIO_QUEUE_TIMELINE: a time-based sampler of every open port's queue level.
    //
    // This is the instrument #3016 records as the missing one. The per-second summary there
    // samples AT each grain handoff, so a queue that drains to zero BETWEEN handoffs is
    // invisible to it -- which is why `below-one-grain=0` appeared in both arms of that A/B and
    // could not settle it. Sampling on a clock instead makes the drain slope, the refill bursts
    // and the zero crossings all directly visible, so an underrun can be established without
    // an ear in the room.
    //
    // Paced with host::sleep_until_steady_ns on an ABSOLUTE grid, and that is not incidental:
    // as recovered from #2984 this loop used std::this_thread::sleep_for(1ms), which on Windows
    // resolves on the winpthreads tick (#3013) -- so the '1 ms sampler' would have sampled at
    // ~15.6 ms, coarser than the grain period it exists to resolve, while its own log said 1 ms.
    // An instrument subject to the defect it measures is worse than no instrument. Absolute
    // rather than relative so the interval cannot drift under load.
    //
    // Values are snapshotted under the lock and printed outside it: at 1 kHz, holding the audio
    // mutex across an fprintf would put the sampler's own I/O into the path it is measuring.
    void queue_timeline_loop(int interval_ms) {
        const uint64_t interval_ns = (uint64_t)interval_ms * 1000000ull;
        const auto     t0 = std::chrono::steady_clock::now();
        uint64_t deadline = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                t0.time_since_epoch()).count();
        // queued, not available: SDL_GetAudioStreamAvailable reports DEVICE-format bytes while
        // grain_bytes, the [audio-dbg] field name and the #3016 sibling meter are all guest
        // format. Mixing them is benign on an F32 title by luck and silently 2x out on an S16 one.
        struct Sample { int port; int queued; int grain; };
        std::array<Sample, kMaxPorts> samples{};
        while (timeline_running_.load(std::memory_order_relaxed)) {
            int n = 0;
            {
                std::lock_guard<std::mutex> lk(mx_);
                for (int i = 0; i < kMaxPorts; i++) {
                    if (!slots_[i].stream) continue;
                    const int q = SDL_GetAudioStreamQueued(slots_[i].stream);
                    if (q < 0) continue;          // an errored stream reports -1; do not log it as a level
                    samples[n++] = { i + 1, q, slots_[i].grain_bytes };
                }
            }
            const uint64_t t_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
            // Microseconds, and as an integer. %.3f of a SECONDS value quantizes the timestamp to
            // exactly the sampling interval, so a 1 ms sampler could only ever print gaps of
            // 1.000/2.000/3.000 -- the honesty figures quoted for this instrument were partly an
            // artifact of its own format string.
            for (int i = 0; i < n; i++)
                SDL_Log("[audio-queue] t_us=%llu port=%d queued=%d grain=%d",
                        (unsigned long long)t_us, samples[i].port, samples[i].queued,
                        samples[i].grain);
            // Advance the grid, but RESYNC if a pass overran it. Without this, one slow pass
            // leaves every subsequent deadline in the past, sleep_until_steady_ns returns
            // immediately by contract, and the loop becomes a busy spin hammering mx_ at full
            // core speed -- while still logging as though it were sampling on schedule. Overruns
            // are expected rather than hypothetical here: SDL_Log on Windows writes through
            // OutputDebugString and a console handle under a global lock, a thousand times a
            // second.
            const uint64_t now_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            deadline += interval_ns;
            if (deadline <= now_ns) deadline = now_ns + interval_ns;
            prosper::host::sleep_until_steady_ns(deadline);
        }
    }

    void start_queue_timeline() {
        const int ms = queue_timeline_ms();
        if (ms <= 0 || timeline_running_.exchange(true)) return;
        timeline_thread_ = std::thread(&Sdl3AudioSink::queue_timeline_loop, this, ms);
        SDL_Log("prosper-audio: queue-level timeline started at %d ms", ms);
    }

    // JOINED, not detached, and joined from both places that can actually run.
    //
    // The first version cleared a flag at the top of quit() and called that sufficient. Review
    // found the argument was about dead code: shutdown_sdl3_audio_sink() -- the only caller of
    // quit() -- is invoked ONLY from test_audio_sdl3.cpp. prosper-app and boot_trace install the
    // sink and never shut it down, so on neither path where this instrument is used did the flag
    // ever get cleared.
    //
    // What that leaves per path: prosper-app ends in std::_Exit, which runs no destructors at all,
    // so a sampler alive at exit is harmless by construction. boot_trace returns from main and
    // drops into static destruction of mx_ and slots_ WITH the sampler live -- that one was a real
    // use-after-free, and the destructor below is what closes it. A detached thread cannot be made
    // safe here by any flag, because the flag only says "please stop" and static destruction does
    // not wait to be asked.
    void stop_queue_timeline() {
        timeline_running_.store(false, std::memory_order_relaxed);
        if (timeline_thread_.joinable()) timeline_thread_.join();
    }

    ~Sdl3AudioSink() override { stop_queue_timeline(); }

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
        s.dump.finalize();   // a closed port's capture is complete
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
                  std::chrono::steady_clock::time_point next{};     // per-grain pacing deadline
                  int channels = 2; bool f32 = false;               // dump format (#2981)
                  WavDump dump;                                     // PROSPER_AUDIO_DUMP_WAV
                  // PROSPER_AUDIO_DEBUG: previous arrival, PER PORT. Deliberately not
                  // thread_local, which is how this arrived from #2984: a title whose mixer
                  // interleaves three threads onto one port records three independent gaps,
                  // each ~3x the port's real arrival interval -- and audio_delivery_report.py
                  // aggregates BY PORT, so its headline cadence would have been ~3x wrong on
                  // exactly the title the instrument was built for. Guarded by mx_ like the
                  // rest of the slot.
                  std::chrono::steady_clock::time_point dbg_last{};
                  // PROSPER_AUDIO_QUEUE_TRACE accounting, reset each reporting second (#3016).
                  int q_min = -1, q_max = 0;
                  unsigned long long q_calls = 0, q_starved = 0;
                  std::chrono::steady_clock::time_point q_last{};

                  // A defaulted default ctor keeps slots_{} value-initialization off the
                  // copy-construct path, which WavDump's deleted copy would reject.
                  Slot() = default;
    };
    std::mutex mx_;
    std::array<Slot, kMaxPorts> slots_{};
    // Set once when the timeline starts; cleared on quit() so the detached sampler exits rather
    // than outliving the streams it reads.
    std::atomic<bool> timeline_running_{false};
    std::thread       timeline_thread_;
    bool paused_ = false;
    float gain_ = 1.0f;   // linear playback gain, applied via SDL_SetAudioStreamGain
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
