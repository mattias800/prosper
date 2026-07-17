// test_avplayer (#324) — libSceAvPlayer lifecycle. prosper doesn't decode video, but the two init
// entry points have DIFFERENT ABIs and must be distinguished, or the guest's video wrapper misreads the
// result. sceAvPlayerInit RETURNS the handle; sceAvPlayerInitEx returns an int32 error code (0 = success)
// and WRITES the handle to its out-param. Registering InitEx to the return-the-handle handler made
// PPSA02664's PS5VideoPlayback wrapper read a non-zero handle as an error code and abort the intro video
// ("[PS5VideoPlayback] ERROR: sceAvPlayerInitEx() failed"). Astro Bot also pulls frames without ever
// polling IsActive, so synthetic EOF must be driven by GetVideoData[Ex] and fire STOP outside the
// player mutex. This locks both contracts.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

struct AvpInitData {
    struct { void* obj; void* allocate; void* deallocate; void* allocate_texture; void* deallocate_texture; } memory{};
    struct { void* obj; void* open; void* close; void* read_offset; void* size; } file{};
    struct { void* obj; void* event_callback; } event{};
    uint32_t debug_level = 0, base_priority = 0; int32_t num_fb = 0; uint8_t auto_start = 0, reserved[3]{};
    const char* default_language = nullptr;
};
struct AvpFrameInfoEx {
    void* data = nullptr; uint8_t reserved[4]{}; uint64_t timestamp = 0; uint8_t details[80]{};
};
static_assert(offsetof(AvpFrameInfoEx, details) == 24, "AvPlayerFrameInfoEx ABI");

static uint32_t events[8]{};
static int event_count = 0;
#ifdef _WIN32
extern "C" void on_avplayer_event_host(uint32_t event) {
    if (event_count < 8) events[event_count++] = event;
}
extern "C" void on_avplayer_event();
// Production callbacks are guest SysV functions even in a Windows build.  Keep the test honest by
// accepting event in SysV RSI, then bridge the one value into a normal Microsoft-x64 C++ helper.
__asm__(
    ".text\n"
    ".globl on_avplayer_event\n"
    "on_avplayer_event:\n"
    "  movl %esi, %ecx\n"
    "  subq $40, %rsp\n"
    "  callq on_avplayer_event_host\n"
    "  addq $40, %rsp\n"
    "  retq\n");
#else
static void PROSPER_SYSV_ABI on_avplayer_event(void*, uint32_t event, int32_t, void*) {
    if (event_count < 8) events[event_count++] = event;
}
#endif

int main() {
    printf("== test_avplayer ==\n");
    register_builtin_hle();

    HleFn init   = Hle::lookup(nid_hash("sceAvPlayerInit"));
    HleFn initex = Hle::lookup(nid_hash("sceAvPlayerInitEx"));
    HleFn active = Hle::lookup(nid_hash("sceAvPlayerIsActive"));
    HleFn add    = Hle::lookup(nid_hash("sceAvPlayerAddSource"));
    HleFn start  = Hle::lookup(nid_hash("sceAvPlayerStart"));
    HleFn video  = Hle::lookup(nid_hash("sceAvPlayerGetVideoDataEx"));
    HleFn audio  = Hle::lookup(nid_hash("sceAvPlayerGetAudioData"));
    HleFn streams = Hle::lookup("hdTyRzCXQeQ");
    HleFn infoex  = Hle::lookup("ctTAcF5DiKQ");
    HleFn pause   = Hle::lookup("9y5v+fGN4Wk");
    HleFn resume  = Hle::lookup("w5moABNwnRY");
    CHECK(init && initex && active && add && start && video && audio && streams && infoex && pause && resume,
          "AvPlayer lifecycle and stream functions registered");
    if (!(init && initex && active && add && start && video && audio && streams && infoex && pause && resume)) {
        printf("== FAIL ==\n"); return 1;
    }

    // sceAvPlayerInit(data) -> handle (the RETURN value is the handle; non-NULL so the game proceeds).
    uint64_t h1 = init(0, 0, 0, 0, 0, 0);
    CHECK(h1 != 0, "sceAvPlayerInit returns a non-NULL handle");

    // sceAvPlayerInitEx(data, out) -> 0 (success error-code), handle written to *out. It must NOT return
    // the handle as its result — that is the regression this test guards.
    uint64_t out = 0xDEAD;
    uint64_t rc = initex(0, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0);
    CHECK(rc == 0, "sceAvPlayerInitEx returns 0 (success error code), NOT the handle");
    CHECK(out != 0 && out != 0xDEAD, "sceAvPlayerInitEx WROTE a valid non-NULL handle to the out-param");

    // IsActive must report "not active" so the game's while(IsActive) playback loop ends and it advances.
    CHECK(active(out, 0, 0, 0, 0, 0) == 0, "sceAvPlayerIsActive -> 0 (finished, proceed)");

    // One delivered synthetic frame, then EOF on the next pull. Astro Bot follows this path and does
    // not call IsActive. The STOP callback re-enters no HLE here, but the production callback may do
    // so, therefore GetVideoDataEx must release the player mutex before firing it.
#ifdef _WIN32
    _putenv_s("PROSPER_AVP_SYNTH_FRAMES", "1");
#else
    setenv("PROSPER_AVP_SYNTH_FRAMES", "1", 1);
#endif
    AvpInitData data{};
    data.event.event_callback = (void*)&on_avplayer_event;
    uint64_t h2 = init((uint64_t)(uintptr_t)&data, 0, 0, 0, 0, 0);
    const char source[] = "/app0/test.mp4";
    CHECK(add(h2, (uint64_t)(uintptr_t)source, 0, 0, 0, 0) == 0, "sceAvPlayerAddSource succeeds");
    CHECK(streams(h2, 0, 0, 0, 0, 0) == 2, "synthetic MP4 source enumerates video and audio streams");
    struct StreamInfoEx { uint64_t size; uint32_t type; uint8_t reserved[4]; uint8_t details[80]; uint64_t duration; } info{};
    info.size = sizeof(info);
    CHECK(infoex(h2, 0, (uint64_t)(uintptr_t)&info, 0, 0, 0) == 0 && info.size == sizeof(info) &&
          info.type == 1 && info.duration == 33, "synthetic stream metadata is valid and size-preserving");
    info = {}; info.size = sizeof(info);
    CHECK(infoex(h2, 1, (uint64_t)(uintptr_t)&info, 0, 0, 0) == 0 && info.type == 2,
          "synthetic audio stream metadata is enumerated");
    CHECK(start(h2, 0, 0, 0, 0, 0) == 0, "sceAvPlayerStart succeeds");
    CHECK(event_count == 2 && events[0] == 2 && events[1] == 3, "READY and PLAY callbacks fire");

    AvpFrameInfoEx frame{};
    struct AudioFrameInfo { void* data; uint8_t reserved[4]; uint64_t timestamp; uint8_t details[16]; } audio_frame{};
    CHECK(audio(h2, (uint64_t)(uintptr_t)&audio_frame, 0, 0, 0, 0) == 1 && audio_frame.data,
          "synthetic audio pull returns stable silent PCM");
    CHECK(video(h2, (uint64_t)(uintptr_t)&frame, 0, 0, 0, 0) == 1, "first synthetic video pull returns a frame");
    CHECK(frame.data != nullptr, "synthetic frame has stable storage");
    uint32_t width = 0, height = 0, pitch = 0; double fps = 0.0;
    memcpy(&width, frame.details + 0, sizeof(width));
    memcpy(&height, frame.details + 4, sizeof(height));
    memcpy(&pitch, frame.details + 36, sizeof(pitch));
    memcpy(&fps, frame.details + 48, sizeof(fps));
    CHECK(width == 1920 && height == 1080 && pitch == 1920 && fps == 30.0,
          "AvPlayerVideoEx uses the published 80-byte layout");
    CHECK(pause(h2, 0, 0, 0, 0, 0) == 0 && event_count == 3 && events[2] == 4,
          "Pause reports success and fires PAUSE without deadlocking the callback");
    CHECK(video(h2, (uint64_t)(uintptr_t)&frame, 0, 0, 0, 0) == 0 && event_count == 3,
          "paused video pull returns no frame without consuming EOF");
    CHECK(resume(h2, 0, 0, 0, 0, 0) == 0 && event_count == 4 && events[3] == 3,
          "Resume restarts delivery and fires PLAY");
    CHECK(video(h2, (uint64_t)(uintptr_t)&frame, 0, 0, 0, 0) == 0, "resumed synthetic video pull reports EOF");
    CHECK(event_count == 5 && events[4] == 1, "EOF fires one STOP callback");
    CHECK(video(h2, (uint64_t)(uintptr_t)&frame, 0, 0, 0, 0) == 0 && event_count == 5,
          "EOF remains stopped without duplicate callbacks");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
