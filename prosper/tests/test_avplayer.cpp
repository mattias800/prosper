// test_avplayer (#324/#841) — libSceAvPlayer lifecycle and native-backend handoff. The two init
// entry points have DIFFERENT ABIs and must be distinguished, or the guest's video wrapper misreads the
// result. sceAvPlayerInit RETURNS the handle; sceAvPlayerInitEx returns an int32 error code (0 = success)
// and WRITES the handle to its out-param. Registering InitEx to the return-the-handle handler made
// PPSA02664's PS5VideoPlayback wrapper read a non-zero handle as an error code and abort the intro video
// ("[PS5VideoPlayback] ERROR: sceAvPlayerInitEx() failed"). Astro Bot also pulls frames without ever
// polling IsActive, so synthetic EOF must be driven by GetVideoData[Ex] and fire STOP outside the
// player mutex. This locks both contracts.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/video_backend.hpp"
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
struct AvpStreamInfoEx {
    uint64_t size = 0; uint32_t type = 0; uint8_t reserved[4]{};
    uint8_t details[80]{}; uint64_t duration = 0;
};
static_assert(offsetof(AvpFrameInfoEx, details) == 24, "AvPlayerFrameInfoEx ABI");

class FakeVideoBackend final : public video::VideoBackend {
public:
    bool fail_open = false;
    bool delivered = false;
    bool audio_delivered = false;
    int close_count = 0;
    std::string opened_path;
    std::vector<uint8_t> nv12 = std::vector<uint8_t>(640 * 360 * 3 / 2, 0x40);
    std::vector<int16_t> pcm = std::vector<int16_t>(2 * 128, 0x20);

    int open(const std::string& host_path) override {
        opened_path = host_path; delivered = false; audio_delivered = false;
        return fail_open ? -1 : 17;
    }
    bool info(int id, video::StreamInfo& out) override {
        if (id != 17) return false;
        out.width = 640; out.height = 360; out.fps = 60.0f;
        out.has_audio = true; out.audio_channels = 2; out.audio_rate = 48'000;
        out.duration_us = 2'000'000;
        return true;
    }
    bool next_video(int id, video::VideoFrame& out) override {
        if (id != 17 || delivered) return false;
        out.y = nv12.data(); out.uv = nv12.data() + 640 * 360;
        out.width = 640; out.height = 360; out.y_stride = 640; out.uv_stride = 640;
        out.pts_us = 16'667; delivered = true;
        return true;
    }
    bool next_audio(int id, video::AudioFrame& out) override {
        if (id != 17 || audio_delivered) return false;
        out.pcm = pcm.data(); out.channels = 2; out.samples = 128; out.sample_rate = 48'000;
        out.pts_us = 0; audio_delivered = true;
        return true;
    }
    // Video completion must not wait for an audio packet that the guest never requests.
    bool eof(int id) override { return id != 17 || delivered; }
    void close(int id) override { if (id == 17) ++close_count; }
};

static void set_synthetic_frames(const char* value) {
#ifdef _WIN32
    _putenv_s("PROSPER_AVP_SYNTH_FRAMES", value ? value : "");
#else
    if (value) setenv("PROSPER_AVP_SYNTH_FRAMES", value, 1);
    else unsetenv("PROSPER_AVP_SYNTH_FRAMES");
#endif
}

static uint32_t events[8]{};
static int event_count = 0;
static int texture_alloc_count = 0, texture_free_count = 0;
static uint32_t texture_last_align = 0, texture_last_size = 0;
static void* texture_expected_object = nullptr;
static std::vector<void*> texture_allocations;

static void* texture_allocate_host(void* object, uint32_t align, uint32_t size) {
    if (object != texture_expected_object) return nullptr;
    void* result = malloc(size);
    if (!result) return nullptr;
    memset(result, 0xcc, size);
    texture_alloc_count++;
    texture_last_align = align;
    texture_last_size = size;
    texture_allocations.push_back(result);
    return result;
}
static void texture_deallocate_host(void* object, void* allocation) {
    if (object != texture_expected_object || !allocation) return;
    texture_free_count++;
    free(allocation);
}

#ifdef _WIN32
extern "C" void on_avplayer_event_host(uint32_t event) {
    if (event_count < 8) events[event_count++] = event;
}
extern "C" void on_avplayer_event();
extern "C" void* on_avplayer_texture_allocate_host(void* object, uint32_t align, uint32_t size) {
    return texture_allocate_host(object, align, size);
}
extern "C" void on_avplayer_texture_deallocate_host(void* object, void* allocation) {
    texture_deallocate_host(object, allocation);
}
extern "C" void on_avplayer_texture_allocate();
extern "C" void on_avplayer_texture_deallocate();
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
// Texture allocation arrives as SysV (RDI object, RSI alignment, RDX size) and is forwarded to the
// Windows host helper (RCX, RDX, R8).  Deallocation similarly forwards RDI/RSI to RCX/RDX.
__asm__(
    ".text\n"
    ".globl on_avplayer_texture_allocate\n"
    "on_avplayer_texture_allocate:\n"
    "  movq %rdx, %r8\n"
    "  movl %esi, %edx\n"
    "  movq %rdi, %rcx\n"
    "  subq $40, %rsp\n"
    "  callq on_avplayer_texture_allocate_host\n"
    "  addq $40, %rsp\n"
    "  retq\n"
    ".globl on_avplayer_texture_deallocate\n"
    "on_avplayer_texture_deallocate:\n"
    "  movq %rsi, %rdx\n"
    "  movq %rdi, %rcx\n"
    "  subq $40, %rsp\n"
    "  callq on_avplayer_texture_deallocate_host\n"
    "  addq $40, %rsp\n"
    "  retq\n");
#else
static void PROSPER_SYSV_ABI on_avplayer_event(void*, uint32_t event, int32_t, void*) {
    if (event_count < 8) events[event_count++] = event;
}
static void* PROSPER_SYSV_ABI on_avplayer_texture_allocate(void* object, uint32_t align,
                                                           uint32_t size) {
    return texture_allocate_host(object, align, size);
}
static void PROSPER_SYSV_ABI on_avplayer_texture_deallocate(void* object, void* allocation) {
    texture_deallocate_host(object, allocation);
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
    HleFn close   = Hle::lookup(nid_hash("sceAvPlayerClose"));
    CHECK(init && initex && active && add && start && video && audio && streams && infoex && pause && resume && close,
          "AvPlayer lifecycle and stream functions registered");
    if (!(init && initex && active && add && start && video && audio && streams && infoex && pause && resume && close)) {
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

    // An unopenable native source must be SKIPPED GRACEFULLY, not left hanging (#1105): AddSource sets
    // up an empty source and drives the normal lifecycle to an immediate EOF instead of returning a bare
    // error with no event. The same backend then proves /app0 translation and real stream metadata/frame
    // handoff.
    set_synthetic_frames(nullptr);
    set_app0_root("C:/prosper-test-app0");
    CHECK(resolve_guest_path("movie.mp4") == "C:/prosper-test-app0/movie.mp4",
          "relative media paths resolve beneath the app0 host root");
    CHECK(resolve_guest_path("/app0/movie.mp4") == "C:/prosper-test-app0/movie.mp4",
          "absolute /app0 media paths retain the shared mount translation");
    FakeVideoBackend fake;
    prosper::video::set_backend(&fake);
    AvpInitData data{};
    data.event.event_callback = (void*)&on_avplayer_event;
    event_count = 0;
    fake.fail_open = true;
    uint64_t failed_handle = init((uint64_t)(uintptr_t)&data, 0, 0, 0, 0, 0);
    const char missing_source[] = "/app0/missing.mp4";
    CHECK(add(failed_handle, (uint64_t)(uintptr_t)missing_source, 0, 0, 0, 0) == 0,
          "unopenable source is skipped gracefully: AddSource succeeds instead of hanging (#1105)");
    CHECK(event_count == 1 && events[0] == 2,
          "AddSource on an unopenable source fires READY (the source lifecycle begins)");
    CHECK(streams(failed_handle, 0, 0, 0, 0, 0) == 1,
          "the skipped source presents as a single empty video stream (immediate EOF, no frames)");
    CHECK(start(failed_handle, 0, 0, 0, 0, 0) == 0 && event_count == 2 && events[1] == 3,
          "Start on the skipped source fires PLAY (this player is not auto_start)");
    CHECK(active(failed_handle, 0, 0, 0, 0, 0) == 0 && event_count == 3 && events[2] == 1,
          "the first IsActive poll fires STOP (immediate EOF) so the while(IsActive) loop ends");
    close(failed_handle, 0, 0, 0, 0, 0);

    // Same graceful skip on the auto_start branch (the shape PPSA02664 actually uses): AddSource fires
    // READY and PLAY itself, then the first IsActive poll fires STOP.
    AvpInitData auto_data = data;
    auto_data.auto_start = 1;
    event_count = 0;
    uint64_t auto_failed = init((uint64_t)(uintptr_t)&auto_data, 0, 0, 0, 0, 0);
    CHECK(add(auto_failed, (uint64_t)(uintptr_t)missing_source, 0, 0, 0, 0) == 0 &&
              event_count == 2 && events[0] == 2 && events[1] == 3,
          "auto_start unopenable source: AddSource fires READY then PLAY (graceful skip begins)");
    CHECK(active(auto_failed, 0, 0, 0, 0, 0) == 0 && event_count == 3 && events[2] == 1,
          "auto_start skipped source reaches STOP on the first IsActive poll");
    close(auto_failed, 0, 0, 0, 0, 0);

    fake.fail_open = false;
    event_count = 0;
    texture_alloc_count = texture_free_count = 0;
    texture_allocations.clear();
    int texture_object = 0x1234;
    texture_expected_object = &texture_object;
    AvpInitData texture_data = data;
    texture_data.memory.obj = texture_expected_object;
    texture_data.memory.allocate_texture = (void*)&on_avplayer_texture_allocate;
    texture_data.memory.deallocate_texture = (void*)&on_avplayer_texture_deallocate;
    texture_data.num_fb = 3;
    uint64_t native_handle = init((uint64_t)(uintptr_t)&texture_data, 0, 0, 0, 0, 0);
    const char native_source[] = "movie.mp4";
    CHECK(add(native_handle, (uint64_t)(uintptr_t)native_source, 0, 0, 0, 0) == 0,
          "native source opens through the registered backend");
    CHECK(texture_alloc_count == 3 && texture_last_align == 0x100 &&
              texture_last_size == fake.nv12.size(),
          "native source allocates the requested guest NV12 texture ring");
    CHECK(fake.opened_path == "C:/prosper-test-app0/movie.mp4",
          "native backend receives the resolved host path, not the raw relative guest path");
    CHECK(streams(native_handle, 0, 0, 0, 0, 0) == 2,
          "native audio-bearing source enumerates video and audio streams");
    AvpStreamInfoEx native_info{}; native_info.size = sizeof(native_info);
    CHECK(infoex(native_handle, 0, (uint64_t)(uintptr_t)&native_info, 0, 0, 0) == 0 &&
              native_info.type == 1 && native_info.duration == 2000,
          "native stream metadata and duration come from the backend");
    native_info = {}; native_info.size = sizeof(native_info);
    CHECK(infoex(native_handle, 1, (uint64_t)(uintptr_t)&native_info, 0, 0, 0) == 0 &&
              native_info.type == 2,
          "native audio stream metadata comes from the backend");
    CHECK(start(native_handle, 0, 0, 0, 0, 0) == 0 && event_count == 2 &&
              events[0] == 2 && events[1] == 3,
          "native playback fires READY then PLAY");
    AvpFrameInfoEx native_frame{};
    CHECK(video(native_handle, (uint64_t)(uintptr_t)&native_frame, 0, 0, 0, 0) == 1 &&
              native_frame.data && native_frame.data != fake.nv12.data() &&
              std::find(texture_allocations.begin(), texture_allocations.end(), native_frame.data) !=
                  texture_allocations.end() &&
              native_frame.timestamp == 16 &&
              memcmp(native_frame.data, fake.nv12.data(), fake.nv12.size()) == 0,
          "native decoded NV12 frame is copied into caller-owned guest texture staging");
    fake.nv12[0] = 0x7f;
    CHECK(static_cast<const uint8_t*>(native_frame.data)[0] == 0x40,
          "guest frame storage remains independent when the backend recycles its packet");
    uint32_t native_width = 0, native_height = 0; double native_fps = 0.0;
    memcpy(&native_width, native_frame.details + 0, sizeof(native_width));
    memcpy(&native_height, native_frame.details + 4, sizeof(native_height));
    memcpy(&native_fps, native_frame.details + 48, sizeof(native_fps));
    CHECK(native_width == 640 && native_height == 360 && native_fps == 60.0,
          "native frame details preserve backend dimensions and frame rate");
    CHECK(video(native_handle, (uint64_t)(uintptr_t)&native_frame, 0, 0, 0, 0) == 0 &&
              event_count == 3 && events[2] == 1 && !fake.audio_delivered,
          "video-only consumption of an audio-bearing source reaches EOF and fires one STOP");
    CHECK(active(native_handle, 0, 0, 0, 0, 0) == 0 &&
              video(native_handle, (uint64_t)(uintptr_t)&native_frame, 0, 0, 0, 0) == 0 &&
              event_count == 3,
          "completed audio-bearing playback stays inactive without repeating STOP");
    close(native_handle, 0, 0, 0, 0, 0);
    CHECK(fake.close_count == 1 && texture_free_count == 3,
          "Close releases the native decode session and every guest texture");
    prosper::video::set_backend(nullptr);

    // One delivered synthetic frame, then EOF on the next pull. Astro Bot follows this path and does
    // not call IsActive. The STOP callback re-enters no HLE here, but the production callback may do
    // so, therefore GetVideoDataEx must release the player mutex before firing it.
    set_synthetic_frames("1");
    event_count = 0;
    uint64_t h2 = init((uint64_t)(uintptr_t)&data, 0, 0, 0, 0, 0);
    const char source[] = "/app0/test.mp4";
    CHECK(add(h2, (uint64_t)(uintptr_t)source, 0, 0, 0, 0) == 0, "sceAvPlayerAddSource succeeds");
    CHECK(streams(h2, 0, 0, 0, 0, 0) == 2, "synthetic MP4 source enumerates video and audio streams");
    AvpStreamInfoEx info{};
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
