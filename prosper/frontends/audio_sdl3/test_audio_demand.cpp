#include "audio_demand.hpp"
#include "hle/audio/audio.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace prosper;
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)


// A real device callback remains inside SDL's stream/device synchronization while Destroy
// starts. The fixture gate is test-owned; it never calls the sink or logs from the callback.
static int lifecycle_destroy_race(bool pre_destroy_control) {
    CHECK(SDL_InitSubSystem(SDL_INIT_AUDIO));
    const SDL_AudioSpec spec{SDL_AUDIO_F32, 2, 48000};
    SDL_AudioStream* stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
    CHECK(stream);
    if (!stream) return 1;
    struct Gate {
        AudioDemandMeter meter;
        std::mutex mutex;
        std::condition_variable condition;
        bool entered = false, release = false, timed_out = false;
        std::atomic<uint64_t> completed{0};
    } gate;
    gate.meter.reset(AudioDemandPhase::Active);
    const auto callback = [](void* data, SDL_AudioStream* current, int additional, int total) {
        auto& g = *static_cast<Gate*>(data);
        {
            std::unique_lock lock(g.mutex);
            g.entered = true;
            g.condition.notify_all();
            if (!g.condition.wait_for(lock, std::chrono::seconds(3), [&] { return g.release; }))
                g.timed_out = true;
        }
        AudioDemandMeter::consume(&g.meter, current, additional, total);
        g.completed.fetch_add(1, std::memory_order_release);
    };
    CHECK(SDL_SetAudioStreamGetCallback(stream, callback, &gate));
    Sdl3AudioDemandSnapshot prior{}, final{};
    CHECK(gate.meter.snapshot(stream, prior)); // device starts paused; no callback yet
    CHECK(prior.phases[1].calls == 0);
    CHECK(SDL_ResumeAudioStreamDevice(stream));
    bool entered = false;
    {
        std::unique_lock lock(gate.mutex);
        entered = gate.condition.wait_for(lock, std::chrono::seconds(2), [&] { return gate.entered; });
    }
    CHECK(entered);
    std::atomic<bool> destroying{false}, destroyed{false};
    std::thread destroyer([&] {
        destroying.store(true, std::memory_order_release);
        SDL_DestroyAudioStream(stream);
        gate.meter.snapshot_quiesced(final);
        destroyed.store(true, std::memory_order_release);
    });
    while (!destroying.load(std::memory_order_acquire)) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (entered) CHECK(!destroyed.load(std::memory_order_acquire));
    {
        std::lock_guard lock(gate.mutex);
        gate.release = true;
        gate.condition.notify_all();
    }
    destroyer.join();
    CHECK(!gate.timed_out);
    CHECK(gate.completed.load() > 0);
    const auto& observed = pre_destroy_control ? prior : final;
    CHECK(observed.phases[1].calls == gate.completed.load());
    CHECK(observed.phases[1].shortfall_calls > 0 && observed.phases[1].additional_bytes > 0);
    CHECK(final.phases[1].last_ns >= final.phases[1].first_ns && final.phases[1].first_ns > 0);
    CHECK(prior.phases[1].calls < final.phases[1].calls); // pre-destroy observation really omitted a callback
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    return failures ? 1 : 0;
}

struct LifecycleLogs {
    std::mutex mutex;
    std::vector<std::string> lines;
    static void SDLCALL output(void* data, int, SDL_LogPriority, const char* text) {
        auto& self = *static_cast<LifecycleLogs*>(data);
        std::lock_guard lock(self.mutex);
        self.lines.emplace_back(text);
    }
};
static uint64_t lifecycle_field(const std::string& line, const char* name) {
    const std::string needle = std::string(" ") + name + "=";
    const auto offset = line.find(needle);
    CHECK(offset != std::string::npos);
    return offset == std::string::npos ? 0 : std::strtoull(line.c_str() + offset + needle.size(), nullptr, 0);
}
static uint64_t lifecycle_now() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static int lifecycle_actual_sink() {
    // The invalid-Put control must not pass a null pointer to an unrelated WAV dump.
#ifdef _WIN32
    _putenv_s("PROSPER_AUDIO_DUMP_WAV", "");
#else
    unsetenv("PROSPER_AUDIO_DUMP_WAV");
#endif
    LifecycleLogs logs;
    SDL_LogOutputFunction previous = nullptr;
    void* previous_data = nullptr;
    SDL_GetLogOutputFunction(&previous, &previous_data);
    SDL_SetLogOutputFunction(LifecycleLogs::output, &logs);
    CHECK(install_sdl3_audio_sink());
    // Pause only this test's device to make the final demand sets deterministic; the core's
    // producer pause gate remains open, so real SDL_PutAudioStreamData is still exercised.
    set_sdl3_audio_paused(true);
    auto* sink = audio_sink();
    AudioPortInfo info;
    info.grain = 1;
    int16_t pcm[2] = {123, -321};
    CHECK(sink->open(27, info));
    CHECK(sink->open(28, info));
    Sdl3AudioDemandSnapshot a{}, b{};
    CHECK(sdl3_audio_demand_snapshot(27, a));
    CHECK(sdl3_audio_demand_snapshot(28, b));
    sink->output(27, pcm, 1);
    const uint64_t before_second = lifecycle_now();
    sink->output(27, pcm, 1);
    const uint64_t after_second = lifecycle_now();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const uint64_t failure_begin = lifecycle_now();
    sink->output(27, nullptr, 1); // SDL rejects nonzero length with a null data pointer
    sink->output(28, pcm, 1);
    CHECK(sink->open(27, info)); // closes generation a while preserving its observation
    Sdl3AudioDemandSnapshot reopened{};
    CHECK(sdl3_audio_demand_snapshot(27, reopened));
    CHECK(reopened.generation > a.generation);
    sink->output(27, pcm, 1);
    sink->close(27);
    sink->close(27); // no second final record for an already-closed generation
    sink->close(28);
    shutdown_sdl3_audio_sink();
    SDL_SetLogOutputFunction(previous, previous_data);
    const char* gate = std::getenv("PROSPER_AUDIO_LIFECYCLE");
    const bool enabled = gate && std::strcmp(gate, "1") == 0;
    std::vector<std::string> closed, final;
    for (const auto& line : logs.lines) {
        if (line.find("[audio-lifecycle-sink] event=close ") == 0 ||
            line.find("[audio-lifecycle-sink] event=reopen ") == 0) closed.push_back(line);
        if (line.find("[audio-demand-final] ") == 0) final.push_back(line);
    }
    if (!enabled) {
        CHECK(closed.empty() && final.empty());
        return failures ? 1 : 0;
    }
    CHECK(closed.size() == 3 && final.size() == 9);
    bool old_seen = false, new_seen = false, peer_seen = false;
    for (const auto& line : closed) {
        const auto port = lifecycle_field(line, "port"), generation = lifecycle_field(line, "generation");
        const auto puts = lifecycle_field(line, "puts");
        CHECK(lifecycle_field(line, "demand_available") == 1);
        CHECK(lifecycle_field(line, "last_put_frames") == 1);
        CHECK(lifecycle_field(line, "last_put_ns") <= lifecycle_field(line, "begin_ns"));
        CHECK(lifecycle_field(line, "begin_ns") <= lifecycle_field(line, "destroy_return_ns"));
        CHECK(lifecycle_field(line, "destroy_return_ns") <= lifecycle_field(line, "anchor_before_ns"));
        CHECK(lifecycle_field(line, "anchor_before_ns") <= lifecycle_field(line, "anchor_after_ns"));
        CHECK(lifecycle_field(line, "sdl_ticks_ns") > 0);
        if (port == 27 && generation == a.generation) {
            old_seen = true;
            CHECK(line.find("event=reopen ") != std::string::npos && puts == 2);
            const auto last = lifecycle_field(line, "last_put_ns");
            CHECK(before_second <= last && last <= after_second && last < failure_begin);
        } else if (port == 27 && generation == reopened.generation) {
            new_seen = true; CHECK(puts == 1);
        } else if (port == 28 && generation == b.generation) {
            peer_seen = true; CHECK(puts == 1);
        } else CHECK(false);
        unsigned phase_count = 0;
        for (const auto& row : final) {
            if (lifecycle_field(row, "port") != port || lifecycle_field(row, "generation") != generation) continue;
            ++phase_count;
            CHECK(lifecycle_field(row, "calls") == 0 && lifecycle_field(row, "shortfall_calls") == 0);
            CHECK(lifecycle_field(row, "first_ns") == 0 && lifecycle_field(row, "last_ns") == 0);
        }
        CHECK(phase_count == 3);
    }
    CHECK(old_seen && new_seen && peer_seen);
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "lifecycle-race") == 0 ||
                      std::strcmp(argv[1], "lifecycle-race-precontrol") == 0))
        return lifecycle_destroy_race(std::strcmp(argv[1], "lifecycle-race-precontrol") == 0);
    if (argc == 2 && std::strcmp(argv[1], "lifecycle-sink") == 0)
        return lifecycle_actual_sink();
    // Separate processes exercise static destruction, which cannot be tested by
    // explicitly shutting down the same sink earlier in the ordinary test.
    if (argc == 2) {
        CHECK(install_sdl3_audio_sink());
        CHECK(audio_sink()->open(1, AudioPortInfo{}));
        if (std::strcmp(argv[1], "external-quit") == 0) SDL_Quit();
        else if (std::strcmp(argv[1], "ordered-quit") == 0) {
            shutdown_sdl3_audio_sink();
            SDL_Quit();
        } else CHECK(std::strcmp(argv[1], "return-open") == 0);
        return failures ? 1 : 0;
    }
    if (!std::getenv("PROSPER_AUDIO_DEMAND")) {
#ifdef _WIN32
        _putenv_s("PROSPER_AUDIO_DEMAND", "1");
#else
        setenv("PROSPER_AUDIO_DEMAND", "1", 1);
#endif
    }
    CHECK(install_sdl3_audio_sink());
    const SDL_AudioSpec spec{SDL_AUDIO_F32, 2, 48000};
    SDL_AudioStream* a = SDL_CreateAudioStream(&spec, &spec);
    SDL_AudioStream* b = SDL_CreateAudioStream(&spec, &spec);
    CHECK(a && b);
    AudioDemandMeter ma, mb;
    if (a && b) {
        ma.reset(AudioDemandPhase::Active); mb.reset(AudioDemandPhase::Active);
        CHECK(SDL_SetAudioStreamGetCallback(a, AudioDemandMeter::consume, &ma));
        CHECK(SDL_SetAudioStreamGetCallback(b, AudioDemandMeter::consume, &mb));
        float pcm[16], out[16]{};
        for (unsigned i = 0; i < 16; ++i) pcm[i] = float(i) / 32;
        CHECK(SDL_PutAudioStreamData(a, pcm, sizeof(pcm)));
        CHECK(SDL_PutAudioStreamData(b, pcm, sizeof(pcm)));
        CHECK(SDL_GetAudioStreamData(a, out, sizeof(out)) == sizeof(out));
        CHECK(std::memcmp(pcm, out, sizeof(out)) == 0);
        CHECK(SDL_GetAudioStreamQueued(a) == 0);
        Sdl3AudioDemandSnapshot sa{}, sb{};
        CHECK(ma.snapshot(a, sa));
        // Empty input after a successful pull is not a shortage: its full PCM is
        // now in the consumer's output buffer and can still be playing downstream.
        CHECK(sa.phases[1].calls == 1 && sa.phases[1].requested_bytes == sizeof(out));
        CHECK(sa.phases[1].shortfall_calls == 0 && sa.phases[1].additional_bytes == 0);
        CHECK(SDL_GetAudioStreamData(a, out, sizeof(out)) == 0); // deliberate producer gap
        CHECK(ma.snapshot(a, sa));
        CHECK(sa.phases[1].calls == 2 && sa.phases[1].shortfall_calls == 1);
        CHECK(sa.phases[1].additional_bytes == sizeof(out));
        CHECK(sa.phases[1].first_ns > 0 && sa.phases[1].last_ns >= sa.phases[1].first_ns);
        CHECK(SDL_GetAudioStreamData(b, out, sizeof(out)) == sizeof(out));
        CHECK(mb.snapshot(b, sb));
        CHECK(sb.phases[1].calls == 1 && sb.phases[1].additional_bytes == 0);
        CHECK(std::memcmp(pcm, out, sizeof(out)) == 0); // independent stream retained its PCM
        ma.set_phase(AudioDemandPhase::Paused);
        CHECK(SDL_GetAudioStreamData(a, out, sizeof(out)) == 0);
        CHECK(ma.snapshot(a, sa));
        CHECK(sa.phases[2].shortfall_calls == 1 && sa.phases[1].shortfall_calls == 1);
    }
    SDL_DestroyAudioStream(a); SDL_DestroyAudioStream(b); // before callback userdata dies

    // The meter must also be wired into the actual sink, not only a test stream.
    auto* sink = audio_sink();
    AudioPortInfo info;
    CHECK(sink->open(1, info));
    Sdl3AudioDemandSnapshot before{}, after{};
    const bool wired = sdl3_audio_demand_snapshot(1, before);
    CHECK(wired);
    if (wired) {
        auto wait_for = [&](auto predicate) {
            const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            do {
                CHECK(sdl3_audio_demand_snapshot(1, after));
                if (predicate(after)) return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } while (std::chrono::steady_clock::now() < end);
            return false;
        };
        CHECK(wait_for([](const auto& s) { return s.phases[0].calls > 0; }));
        int16_t pcm[256 * 2]{};
        sink->output(1, pcm, 256); // intentional silent PCM is still a supplied grain
        CHECK(wait_for([](const auto& s) { return s.phases[1].shortfall_calls > 0; }));
        set_sdl3_audio_paused(true);
        CHECK(sdl3_audio_demand_snapshot(1, before));
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        CHECK(sdl3_audio_demand_snapshot(1, after));
        CHECK(after.phases[1].calls == before.phases[1].calls);
        sink->close(1);
        CHECK(!sdl3_audio_demand_snapshot(1, after));
        CHECK(sink->open(1, info)); // still paused: no background reads can race the reset
        CHECK(sdl3_audio_demand_snapshot(1, after));
        CHECK(after.generation > before.generation);
        for (const auto& phase : after.phases) CHECK(phase.calls == 0);
        set_sdl3_audio_paused(false);
    }
    sink->close(1);
    shutdown_sdl3_audio_sink();
    CHECK(!sdl3_audio_demand_snapshot(1, after));
    return failures ? 1 : 0;
}
