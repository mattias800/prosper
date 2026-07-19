// test_audio_sdl3 — end-to-end test of the SDL3 audio frontend, headless under the SDL "dummy"
// audio driver (set by the ctest env). Installs the SDL3 sink, drives the real sceAudioOut HLE
// through the dispatch table, and asserts a full port lifecycle flows through SDL with no device.
#include "audio_sdl3.hpp"
#include "../../src/hle/dispatch.hpp"
#include "../../src/hle/nid.hpp"
#include "../../src/hle/audio.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("  [FAIL] %s:%d  %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static int64_t call(const char* n, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                    uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0) {
    HleFn f = Hle::lookup(nid_hash(n));
    if (!f) { printf("  [FAIL] not registered: %s\n", n); fails++; return -999; }
    return (int64_t)f(a0, a1, a2, a3, a4, a5);
}
static uint64_t PTR(const void* p) { return (uint64_t)(uintptr_t)p; }

int main() {
    printf("== test_audio_sdl3 ==\n");
    register_builtin_hle();
    // Force the dummy driver so this runs on headless CI with no sound hardware.
    if (!getenv("SDL_AUDIO_DRIVER")) {
#ifdef _WIN32
        _putenv_s("SDL_AUDIO_DRIVER", "dummy");
#else
        setenv("SDL_AUDIO_DRIVER", "dummy", 1);
#endif
    }

    bool ok = install_sdl3_audio_sink();
    CHECK(ok);
    if (ok) {
        CHECK(audio_sink() != nullptr);

        int64_t h = call("sceAudioOutOpen", 1, 0, 0, 256, 48000, 1 /*S16 stereo*/);
        CHECK(h >= 1);

        std::vector<int16_t> pcm(256 * 2);
        for (size_t i = 0; i < pcm.size(); i++) pcm[i] = (int16_t)(i * 131 - 4000);
        CHECK(call("sceAudioOutOutput", (uint64_t)h, PTR(pcm.data())) == 256);

        int vols[2] = { 20000, 20000 };
        CHECK(call("sceAudioOutSetVolume", (uint64_t)h, 0x3, PTR(vols)) == 0);
        CHECK(call("sceAudioOutClose", (uint64_t)h) == 0);

        // Output and lifecycle calls can arrive from different guest audio threads. Exercise the
        // exact #855 boundary repeatedly: close/reopen must not destroy an SDL_AudioStream while
        // output() is queueing into it. Completion without a host fault is the lifetime contract;
        // every dummy-device reopen must also succeed.
        AudioPortInfo race_info;
        race_info.grain = 1;
        AudioSink* sink = audio_sink();
        CHECK(sink->open(1, race_info));
        std::atomic<bool> start{false};
        std::atomic<int> outputs{0};
        int16_t race_pcm[2] = {123, -123};
        std::thread output_thread([&] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int i = 0; i < 100; ++i) {
                sink->output(1, race_pcm, 1);
                outputs.fetch_add(1, std::memory_order_relaxed);
            }
        });
        start.store(true, std::memory_order_release);
        while (outputs.load(std::memory_order_acquire) == 0) std::this_thread::yield();
        bool all_reopens_succeeded = true;
        for (int i = 0; i < 50; ++i) {
            sink->close(1);
            all_reopens_succeeded &= sink->open(1, race_info);
        }
        output_thread.join();
        CHECK(all_reopens_succeeded);
        CHECK(outputs.load(std::memory_order_relaxed) == 100);
        sink->close(1);

        shutdown_sdl3_audio_sink();
        CHECK(audio_sink() != nullptr);   // default silent sink restored
    }

    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS: SDL3 audio frontend end-to-end (dummy driver) ==\n");
    return 0;
}
