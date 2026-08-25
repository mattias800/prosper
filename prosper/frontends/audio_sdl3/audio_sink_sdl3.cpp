// audio_sink_sdl3.cpp — SDL3-backed AudioSink for the sceAudioOut HLE (optional frontend).
//
// Enabled with -DPROSPER_AUDIO_SDL3=ON. Bridges the headless AudioSink interface (audio.hpp)
// to SDL3's audio-stream API: one SDL_AudioStream per PS5 audio port, fed the guest's PCM
// grains. output() blocks while the device's queue is full, reproducing the pacing that
// sceAudioOutOutput has on real hardware (it blocks until the audio ring has room).
#include "audio_sdl3.hpp"
#include "hle/audio/audio.hpp"
#include "host/platform/lifecycle.hpp"

#include <SDL3/SDL.h>

#include <array>
#include <chrono>
#include <mutex>
#include <thread>
#include <string>

namespace prosper {
namespace {

// 16 public sceAudioOut ports plus four host-only streams for concurrent AudioOut2 contexts.
// SDL mixes the bound streams into the same playback device while each retains its own pacing
// clock; this mirrors independent hardware contexts without serializing their sample timelines.
// 20 covers the 16 public sceAudioOut ports plus the four host-only AudioOut2 context ports
// (kA2SinkPortBase = 21..24 in hle_audio.cpp). The old cap of 20 made every AudioOut2 open fail
// and fall back to silent pacing — GRIS's Wwise mix was correct and discarded at the sink (#2981).
constexpr int kMaxPorts = 28;

SDL_AudioFormat to_sdl_format(AudioFmt f) { return f == AudioFmt::F32 ? SDL_AUDIO_F32 : SDL_AUDIO_S16; }

// PROSPER_AUDIO_DUMP=<path.wav>: dump every port's mixed PCM to per-port WAV files
// (<path> with the port number inserted before the extension). Measures what the guest
// actually mixed — immune to host volume/mute — so "is audio playing" is a measurement,
// not an impression (#2981).
class WavDump {
public:
    bool open(const char* path, int freq, int channels, bool f32) {
        if (!path || !*path) return false;
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
        data_size_pos_ = 44;
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
        return true;
    }

    void quit() {
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
        if (const char* dump = getenv("PROSPER_AUDIO_DUMP"); dump && *dump) {
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
        }
        // Pace the guest thread to real time. This sleep IS the Heaps unqueueBuffer signal
        // (#2978): by the time it returns, the previous grain has been consumed by the audio
        // device, giving Heaps the buffer-completion timing it needs to unqueue SFX buffers
        // instead of re-triggering them. The sleep must be outside the lock so other audio
        // operations (open/close/volume) can proceed while this grain plays.
        if (freq > 0) std::this_thread::sleep_until(target);
        // Consumption confirmation (#2978): after the pacing sleep, the grain has had its
        // full duration to play. A brief bounded wait here gives Heaps' hxd.snd.Manager the
        // "buffer consumed" signal its unqueueBuffer needs — without this, short SFX loop
        // forever because Heaps can't tell a played buffer from a pending one. The wait is
        // OUTSIDE the lock and bounded, so it cannot block other audio operations or cause
        // underruns (the grain duration has already elapsed during the pacing sleep).
        {
            std::lock_guard<std::mutex> lk(mx_);
            if (SDL_AudioStream* st = slots_[port - 1].stream) {
                auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(2);
                while (SDL_GetAudioStreamAvailable(st) > 0 &&
                       std::chrono::steady_clock::now() < deadline)
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
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
                  WavDump dump; };                                  // PROSPER_AUDIO_DUMP (#2981)
    std::mutex mx_;
    std::array<Slot, kMaxPorts> slots_{};
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
