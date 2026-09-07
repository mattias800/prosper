#include "audio_demand.hpp"
#include "hle/audio/audio.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace prosper;
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

int main(int argc, char** argv) {
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
