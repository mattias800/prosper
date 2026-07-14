// test_audio_sdl3 — end-to-end test of the SDL3 audio frontend, headless under the SDL "dummy"
// audio driver (set by the ctest env). Installs the SDL3 sink, drives the real sceAudioOut HLE
// through the dispatch table, and asserts a full port lifecycle flows through SDL with no device.
#include "audio_sdl3.hpp"
#include "../../src/hle/dispatch.hpp"
#include "../../src/hle/nid.hpp"
#include "../../src/hle/audio.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

        shutdown_sdl3_audio_sink();
        CHECK(audio_sink() != nullptr);   // default silent sink restored
    }

    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS: SDL3 audio frontend end-to-end (dummy driver) ==\n");
    return 0;
}
