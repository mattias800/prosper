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
#include "../src/gpu/guest_texture_layout.hpp"
#include "../src/gpu/tile.hpp"
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
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
struct AvpFrameInfo {
    void* data = nullptr; uint8_t reserved[4]{}; uint64_t timestamp = 0; uint32_t details[4]{};
};
struct AvpStreamInfoEx {
    uint64_t size = 0; uint32_t type = 0; uint8_t reserved[4]{};
    uint8_t details[80]{}; uint64_t duration = 0;
};
static_assert(offsetof(AvpFrameInfoEx, details) == 24, "AvPlayerFrameInfoEx ABI");

class FakeVideoBackend final : public video::VideoBackend {
public:
    // R-Type Delta's 1920x1080 movie is the discriminating case: PS5 AvPlayer exposes a
    // 2048-byte physical pitch and crops the 128 padded pixels from the valid image extent.
    static constexpr uint32_t kWidth = 1920;
    static constexpr uint32_t kHeight = 1080;
    static constexpr uint32_t kStride = 2048;
    static constexpr size_t kUvRows = (kHeight + 1u) / 2u;
    // prosper RE-pitches rather than propagating the decoder's stride, so what it publishes is its
    // OWN 256-byte-aligned pitch, not kStride. The two coincide here; asserting against this constant
    // keeps the checks true for the right reason if the fake backend's stride ever changes.
    static constexpr uint32_t kPitch = (kWidth + 255u) & ~255u;
    bool fail_open = false;
    bool delivered = false;
    bool audio_delivered = false;
    int close_count = 0;
    std::string opened_path;
    std::vector<uint8_t> nv12 =
        std::vector<uint8_t>(static_cast<size_t>(kStride) * (kHeight + kUvRows), 0xa5);
    std::vector<int16_t> pcm = std::vector<int16_t>(2 * 128, 0x20);

    FakeVideoBackend() {
        for (uint32_t row = 0; row < kHeight; ++row)
            std::fill_n(nv12.data() + static_cast<size_t>(row) * kStride, kWidth,
                        static_cast<uint8_t>(0x40u + (row & 0x0fu)));
        uint8_t* uv = nv12.data() + static_cast<size_t>(kStride) * kHeight;
        for (size_t row = 0; row < kUvRows; ++row)
            std::fill_n(uv + row * kStride, kWidth,
                        static_cast<uint8_t>(0x80u + (row & 0x0fu)));
    }

    int open(const std::string& host_path) override {
        opened_path = host_path; delivered = false; audio_delivered = false;
        return fail_open ? -1 : 17;
    }
    bool info(int id, video::StreamInfo& out) override {
        if (id != 17) return false;
        out.width = kWidth; out.height = kHeight; out.fps = 60.0f;
        out.has_audio = true; out.audio_channels = 2; out.audio_rate = 48'000;
        out.duration_us = 2'000'000;
        return true;
    }
    bool next_video(int id, video::VideoFrame& out) override {
        if (id != 17 || delivered) return false;
        out.y = nv12.data();
        out.uv = nv12.data() + static_cast<size_t>(kStride) * kHeight;
        out.width = kWidth; out.height = kHeight; out.y_stride = kStride; out.uv_stride = kStride;
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

// A 256-ALIGNED width, which every other case in this file lacks (#2051). FakeVideoBackend is 1920,
// so `pitch != width` and `crop_right == 128` in every existing assertion -- and the reviewer who
// found this put the consequence precisely: a refactor that unconditionally padded, or that set
// crop_right to a constant 128, would pass the entire file.
//
// Both are plausible edits. The first is what "simplify the pitch logic" looks like; the second is
// what a copy-paste of the 1920 case looks like. Both would silently break every title whose movie
// is 1280x720 or 3840x2160 -- which is the majority of real content, and exactly the set #2032
// established its own change is a bit-for-bit no-op over.
//
// 1280 % 256 == 0, so the contract here is the OPPOSITE of the padded one: width == pitch == visible
// and crop_right == 0, with no padding bytes existing at all.
class AlignedVideoBackend final : public video::VideoBackend {
public:
    static constexpr uint32_t kWidth = 1280;          // 1280 % 256 == 0
    static constexpr uint32_t kHeight = 720;
    static constexpr uint32_t kStride = kWidth;       // no padding: the decoder's stride IS the width
    static constexpr size_t kUvRows = (kHeight + 1u) / 2u;
    static constexpr uint32_t kPitch = (kWidth + 255u) & ~255u;   // == kWidth, and the arm proves it
    static_assert(kPitch == kWidth, "1280 must be 256-aligned for this to be the aligned case");
    bool delivered = false;
    std::vector<uint8_t> nv12 =
        std::vector<uint8_t>(static_cast<size_t>(kStride) * (kHeight + kUvRows), 0x5a);

    int open(const std::string&) override { delivered = false; return 21; }
    bool info(int id, video::StreamInfo& out) override {
        if (id != 21) return false;
        out.width = kWidth; out.height = kHeight; out.fps = 60.0f;
        out.has_audio = false; out.duration_us = 2'000'000;
        return true;
    }
    bool next_video(int id, video::VideoFrame& out) override {
        if (id != 21 || delivered) return false;
        out.y = nv12.data();
        out.uv = nv12.data() + static_cast<size_t>(kStride) * kHeight;
        out.width = kWidth; out.height = kHeight; out.y_stride = kStride; out.uv_stride = kStride;
        out.pts_us = 16'667; delivered = true;
        return true;
    }
    bool next_audio(int, video::AudioFrame&) override { return false; }
    bool eof(int id) override { return id != 21 || delivered; }
    void close(int) override {}
};

// A backend with a real timeline, so a seek can be observed as a CHANGE OF POSITION rather than as
// "the call returned zero" (#1949). Frame i sits at i * kFrameUs; a seek lands on the first frame at
// or after the requested time, exactly as an accurate container seek does.
class SeekingVideoBackend final : public video::VideoBackend {
public:
    static constexpr uint32_t kWidth = 64;
    static constexpr uint32_t kHeight = 64;
    static constexpr uint64_t kFrameUs = 16'667;
    static constexpr uint64_t kFrames = 300;
    bool support_seek = true;
    int seek_calls = 0;
    uint64_t last_seek_us = 0;
    uint64_t cursor = 0;
    std::vector<uint8_t> nv12 =
        std::vector<uint8_t>(static_cast<size_t>(kWidth) * (kHeight + kHeight / 2), 0x30);

    int open(const std::string&) override { cursor = 0; return 21; }
    bool info(int id, video::StreamInfo& out) override {
        if (id != 21) return false;
        out.width = kWidth; out.height = kHeight; out.fps = 60.0f;
        out.has_audio = false; out.duration_us = kFrames * kFrameUs;
        return true;
    }
    bool next_video(int id, video::VideoFrame& out) override {
        if (id != 21 || cursor >= kFrames) return false;
        out.y = nv12.data();
        out.uv = nv12.data() + static_cast<size_t>(kWidth) * kHeight;
        out.width = kWidth; out.height = kHeight; out.y_stride = kWidth; out.uv_stride = kWidth;
        out.pts_us = cursor * kFrameUs;
        ++cursor;
        return true;
    }
    bool next_audio(int, video::AudioFrame&) override { return false; }
    bool eof(int id) override { return id != 21 || cursor >= kFrames; }
    bool seek(int id, uint64_t position_us) override {
        if (id != 21 || !support_seek) return false;
        ++seek_calls;
        last_seek_us = position_us;
        cursor = (position_us + kFrameUs - 1) / kFrameUs;   // first frame at or after the request
        return true;
    }
    void close(int) override {}
};

// #1955 — a backend that only accepts caller-supplied bytes. host `open()` returning -1 while
// recording the call makes "prosper fell back to the host path" a visible failure rather than a
// silent one.
class MemorySourceBackend final : public video::VideoBackend {
public:
    std::vector<uint8_t> received;
    std::string received_name;
    int host_open_calls = 0;
    int memory_open_calls = 0;

    int open(const std::string&) override { ++host_open_calls; return -1; }
    int open_memory(const std::string& debug_name, const uint8_t* data, size_t bytes) override {
        ++memory_open_calls;
        received_name = debug_name;
        received.assign(data, data + bytes);
        return 31;
    }
    bool info(int id, video::StreamInfo& out) override {
        if (id != 31) return false;
        out.width = 64; out.height = 64; out.fps = 30.0f; out.duration_us = 1'000'000;
        return true;
    }
    bool next_video(int, video::VideoFrame&) override { return false; }
    bool next_audio(int, video::AudioFrame&) override { return false; }
    bool eof(int) override { return true; }
    void close(int) override {}
};

// #1973 — a source NOBODY CONSUMES. prosper's host decode workers fill a small bounded queue and
// block while it is full, so a guest that starts a movie and never pulls a frame leaves the queue
// full forever: frames are always available to anyone who asks, and the decoder can never finish.
// eof() therefore stays false for as long as the session lives, which is exactly the state
// PPSA27624 sat in (14,843 sceAvPlayerIsActive polls, zero GetVideoDataEx calls, no STOP).
class UnconsumedVideoBackend final : public video::VideoBackend {
public:
    static constexpr uint32_t kWidth = 64;
    static constexpr uint32_t kHeight = 64;
    static constexpr uint64_t kFrameUs = 16'667;
    // Deliberately short, so the guard exercises a real wall-clock timeline in under a second.
    uint64_t duration_us = 300'000;
    uint64_t cursor = 0;
    std::vector<uint8_t> nv12 =
        std::vector<uint8_t>(static_cast<size_t>(kWidth) * (kHeight + kHeight / 2), 0x20);

    int open(const std::string&) override { cursor = 0; return 51; }
    bool info(int id, video::StreamInfo& out) override {
        if (id != 51) return false;
        out.width = kWidth; out.height = kHeight; out.fps = 60.0f;
        out.has_audio = false; out.duration_us = duration_us;
        return true;
    }
    bool next_video(int id, video::VideoFrame& out) override {
        if (id != 51) return false;
        out.y = nv12.data();
        out.uv = nv12.data() + static_cast<size_t>(kWidth) * kHeight;
        out.width = kWidth; out.height = kHeight; out.y_stride = kWidth; out.uv_stride = kWidth;
        out.pts_us = cursor * kFrameUs;
        ++cursor;
        return true;
    }
    bool next_audio(int, video::AudioFrame&) override { return false; }
    bool eof(int) override { return false; }
    void close(int) override {}
};

#ifndef _WIN32
// A guest file-replacement table over a synthetic container: the "media" is a byte range that does
// NOT start at offset 0, which is exactly the case prosper cannot express any other way. Windows
// production calls these through prosper_call_guest_sysv4; the in-memory backend is Linux-only, so
// this fixture stays POSIX rather than growing four more ABI bridge stubs.
static std::vector<uint8_t> guest_container;
static constexpr size_t kGuestMediaOffset = 4096;
static constexpr size_t kGuestMediaBytes = 5000;
static int guest_file_open_calls = 0, guest_file_close_calls = 0;
static void* guest_file_expected_object = nullptr;
static bool guest_file_is_open = false;

static int32_t PROSPER_SYSV_ABI on_avplayer_file_open(void* object, const char* path) {
    if (object != guest_file_expected_object || !path) return -1;
    guest_file_open_calls++;
    guest_file_is_open = true;
    return 5;
}
static int32_t PROSPER_SYSV_ABI on_avplayer_file_close(void* object) {
    if (object != guest_file_expected_object) return -1;
    guest_file_close_calls++;
    guest_file_is_open = false;
    return 0;
}
static uint64_t PROSPER_SYSV_ABI on_avplayer_file_size(void* object) {
    if (object != guest_file_expected_object || !guest_file_is_open) return 0;
    return kGuestMediaBytes;
}
static int32_t PROSPER_SYSV_ABI on_avplayer_file_read(void* object, uint8_t* buffer,
                                                      uint64_t position, uint32_t length) {
    if (object != guest_file_expected_object || !guest_file_is_open || !buffer) return -1;
    if (position >= kGuestMediaBytes) return 0;
    const uint64_t available = kGuestMediaBytes - position;
    const uint32_t take = static_cast<uint32_t>(std::min<uint64_t>(length, available));
    memcpy(buffer, guest_container.data() + kGuestMediaOffset + position, take);
    return static_cast<int32_t>(take);
}
#endif

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
    HleFn video_basic = Hle::lookup(nid_hash("sceAvPlayerGetVideoData"));
    HleFn video  = Hle::lookup(nid_hash("sceAvPlayerGetVideoDataEx"));
    HleFn audio  = Hle::lookup(nid_hash("sceAvPlayerGetAudioData"));
    HleFn streams = Hle::lookup("hdTyRzCXQeQ");
    HleFn infoex  = Hle::lookup("ctTAcF5DiKQ");
    HleFn pause   = Hle::lookup("9y5v+fGN4Wk");
    HleFn resume  = Hle::lookup("w5moABNwnRY");
    HleFn jump    = Hle::lookup("XC9wM+xULz8");   // sceAvPlayerJumpToTime (#1949)
    HleFn current = Hle::lookup("wwM99gjFf1Y");   // sceAvPlayerCurrentTime
    HleFn close   = Hle::lookup(nid_hash("sceAvPlayerClose"));
    CHECK(init && initex && active && add && start && video_basic && video && audio && streams &&
              infoex && pause && resume && close,
          "AvPlayer lifecycle and stream functions registered");
    if (!(init && initex && active && add && start && video_basic && video && audio && streams &&
          infoex && pause && resume && close)) {
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
    constexpr size_t native_y_bytes =
        static_cast<size_t>(FakeVideoBackend::kStride) * FakeVideoBackend::kHeight;
    constexpr size_t native_frame_bytes = native_y_bytes +
        static_cast<size_t>(FakeVideoBackend::kStride) * FakeVideoBackend::kUvRows;
    CHECK(texture_alloc_count == 3 && texture_last_align == 0x100 &&
              texture_last_size == native_frame_bytes,
          "native source allocates the requested pitch-preserving guest NV12 texture ring");
    CHECK(fake.opened_path == "C:/prosper-test-app0/movie.mp4",
          "native backend receives the resolved host path, not the raw relative guest path");
    CHECK(streams(native_handle, 0, 0, 0, 0, 0) == 2,
          "native audio-bearing source enumerates video and audio streams");
    AvpStreamInfoEx native_info{}; native_info.size = sizeof(native_info);
    uint32_t native_info_width = 0, native_info_crop_left = 0, native_info_crop_right = 0;
    uint32_t native_info_crop_top = 0, native_info_crop_bottom = 0, native_info_pitch = 0;
    CHECK(infoex(native_handle, 0, (uint64_t)(uintptr_t)&native_info, 0, 0, 0) == 0 &&
              native_info.type == 1 && native_info.duration == 2000,
          "native stream metadata and duration come from the backend");
    memcpy(&native_info_width, native_info.details + 0, sizeof(native_info_width));
    memcpy(&native_info_crop_left, native_info.details + 20, sizeof(native_info_crop_left));
    memcpy(&native_info_crop_right, native_info.details + 24, sizeof(native_info_crop_right));
    memcpy(&native_info_crop_top, native_info.details + 28, sizeof(native_info_crop_top));
    memcpy(&native_info_crop_bottom, native_info.details + 32, sizeof(native_info_crop_bottom));
    memcpy(&native_info_pitch, native_info.details + 36, sizeof(native_info_pitch));
    CHECK(native_info_width == FakeVideoBackend::kPitch &&
              native_info_pitch == FakeVideoBackend::kPitch &&
              native_info_crop_left == 0 && native_info_crop_right == 128 &&
              native_info_crop_top == 0 && native_info_crop_bottom == 0,
          "native stream metadata separates the 2048 physical pitch from 1920 visible pixels");
    CHECK(native_info_width - native_info_crop_left - native_info_crop_right ==
              FakeVideoBackend::kWidth &&
          native_info_pitch - native_info_crop_left - native_info_crop_right ==
              FakeVideoBackend::kWidth,
          "GetStreamInfoEx: both the width- and pitch-based crop spellings yield 1920 visible");
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
              native_frame.timestamp == 16,
          "native decoded NV12 frame is copied into caller-owned guest texture staging");
    const auto* staged = static_cast<const uint8_t*>(native_frame.data);
    const uint64_t staged_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(staged));
    CHECK(staged[0] == 0x40 && staged[FakeVideoBackend::kWidth - 1] == 0x40 &&
              staged[FakeVideoBackend::kStride] == 0x41 &&
              staged[native_y_bytes] == 0x80 &&
              staged[native_y_bytes + FakeVideoBackend::kStride] == 0x81,
          "1920 visible bytes are copied row-by-row into the padded pitch");
    // #2011: the padding is limited-range BLACK, not zero. `Y=0, U=V=0` converts to roughly
    // (0, 136, 0) — mid green — so with `width` published as the coded extent a crop-ignoring
    // consumer would draw a 128-column green stripe down every movie frame. Assert the exact
    // values, on the first and last padded column of a padded row of each plane: a revert to
    // memset(..., 0, ...) turns this red, and so does padding left uninitialised.
    bool luma_pad_black = true, chroma_pad_black = true;
    for (uint32_t col = FakeVideoBackend::kWidth; col < FakeVideoBackend::kPitch; ++col) {
        if (staged[col] != 0x10) luma_pad_black = false;
        if (staged[static_cast<size_t>(FakeVideoBackend::kPitch) + col] != 0x10)
            luma_pad_black = false;
        if (staged[native_y_bytes + col] != 0x80) chroma_pad_black = false;
        if (staged[native_y_bytes + FakeVideoBackend::kPitch + col] != 0x80)
            chroma_pad_black = false;
    }
    CHECK(luma_pad_black && chroma_pad_black,
          "the 128 padded columns are limited-range black (Y=0x10, U=V=0x80), never zero/green");
    CHECK(gpu::guest_linear_texture_row_pitch(staged_address, FakeVideoBackend::kStride) ==
              FakeVideoBackend::kStride &&
          gpu::guest_linear_texture_row_pitch(staged_address, FakeVideoBackend::kWidth) ==
              FakeVideoBackend::kStride &&
          gpu::guest_linear_texture_row_pitch(
              staged_address + native_y_bytes,
              FakeVideoBackend::kWidth) == FakeVideoBackend::kStride,
          "native luma and chroma addresses expose their exact physical pitch for visible rows");
    fake.nv12[0] = 0x7f;
    CHECK(static_cast<const uint8_t*>(native_frame.data)[0] == 0x40,
          "guest frame storage remains independent when the backend recycles its packet");
    uint32_t native_width = 0, native_height = 0, native_crop_left = 0, native_crop_right = 0;
    uint32_t native_crop_top = 0, native_crop_bottom = 0, native_pitch = 0;
    double native_fps = 0.0;
    memcpy(&native_width, native_frame.details + 0, sizeof(native_width));
    memcpy(&native_height, native_frame.details + 4, sizeof(native_height));
    memcpy(&native_crop_left, native_frame.details + 20, sizeof(native_crop_left));
    memcpy(&native_crop_right, native_frame.details + 24, sizeof(native_crop_right));
    memcpy(&native_crop_top, native_frame.details + 28, sizeof(native_crop_top));
    memcpy(&native_crop_bottom, native_frame.details + 32, sizeof(native_crop_bottom));
    memcpy(&native_pitch, native_frame.details + 36, sizeof(native_pitch));
    memcpy(&native_fps, native_frame.details + 48, sizeof(native_fps));
    CHECK(native_width == FakeVideoBackend::kPitch && native_height == FakeVideoBackend::kHeight &&
              native_pitch == FakeVideoBackend::kPitch && native_crop_left == 0 &&
              native_crop_right == 128 && native_crop_top == 0 && native_crop_bottom == 0 &&
              native_fps == 60.0,
          "native frame details publish the coded extent, physical pitch, crop, and frame rate");
    CHECK(gpu::linear_sampled_surface_bytes(native_pitch, native_height, 1) ==
              static_cast<size_t>(native_pitch) * native_height,
          "R-Type movie luma descriptor footprint matches the published pitch extent");
    CHECK(native_pitch - native_crop_left - native_crop_right == FakeVideoBackend::kWidth,
          "GRIS crop offsets hide the padded tail instead of exposing a right-edge strip");
    // #2011: ArcRunner (PPSA21406) spells the same question off `width` rather than `pitch`. While
    // width published the VISIBLE 1920 alongside crop_right=128 the two spellings disagreed, and the
    // title rendered every movie frame into a 1792x1080 surface — a 128-column slice lost. Both
    // spellings must land on the same visible extent.
    CHECK(native_width - native_crop_left - native_crop_right == FakeVideoBackend::kWidth,
          "ArcRunner crop spelling off width yields the same 1920 visible extent as off pitch");
    float native_aspect = 0.0f;
    memcpy(&native_aspect, native_frame.details + 8, sizeof(native_aspect));
    CHECK(native_aspect == static_cast<float>(FakeVideoBackend::kWidth) /
                           static_cast<float>(FakeVideoBackend::kHeight),
          "aspect stays the VISIBLE display ratio and is not widened by the coded pitch");
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
    CHECK(gpu::guest_linear_texture_row_pitch(staged_address, FakeVideoBackend::kStride) == 0,
          "Close removes the released AvPlayer texture layout provenance");

    // The basic API has no pitch/crop fields. Keep its historical tight fallback contract while
    // GetVideoDataEx exposes the PS5 physical layout above.
    uint64_t basic_handle = init((uint64_t)(uintptr_t)&data, 0, 0, 0, 0, 0);
    CHECK(add(basic_handle, (uint64_t)(uintptr_t)native_source, 0, 0, 0, 0) == 0 &&
              start(basic_handle, 0, 0, 0, 0, 0) == 0,
          "basic AvPlayer source starts through the same padded-stride backend");
    AvpFrameInfo basic_frame{};
    CHECK(video_basic(basic_handle, (uint64_t)(uintptr_t)&basic_frame, 0, 0, 0, 0) == 1 &&
              basic_frame.data && basic_frame.details[0] == FakeVideoBackend::kWidth &&
              basic_frame.details[1] == FakeVideoBackend::kHeight,
          "basic AvPlayer frame preserves its width/height-only ABI");
    const uint64_t basic_address =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(basic_frame.data));
    CHECK(gpu::guest_linear_texture_row_pitch(basic_address, FakeVideoBackend::kWidth) ==
              FakeVideoBackend::kWidth,
          "basic AvPlayer storage remains tight because that ABI cannot publish padding crops");
    close(basic_handle, 0, 0, 0, 0, 0);
    CHECK(gpu::guest_linear_texture_row_pitch(basic_address, FakeVideoBackend::kWidth) == 0,
          "Close removes the basic AvPlayer fallback layout provenance");

    // ---- sceAvPlayerJumpToTime (#1949) ---------------------------------------------------------
    // Registering the NID is only half the fix. The guest reads a zero return as a COMPLETED seek:
    // PPSA30490 pauses the player, jumps, then pulls frames until the reported timestamp moves
    // before it will call sceAvPlayerResume. So all of these must hold together — a backend that
    // cannot seek produces a real error instead of a silent success, a backend that can is really
    // repositioned, and the paused player publishes the post-seek frame exactly once.
    CHECK(jump != nullptr && current != nullptr,
          "sceAvPlayerJumpToTime (XC9wM+xULz8) and sceAvPlayerCurrentTime are registered");
    if (jump && current) {
        AvpInitData seek_data{};
        seek_data.event.event_callback = (void*)&on_avplayer_event;

        // FakeVideoBackend does not override the defaulted VideoBackend::seek, i.e. it is a backend
        // with no seek support. That must reach the guest as an error, never as SCE_OK.
        prosper::video::set_backend(&fake);
        uint64_t unseekable = init((uint64_t)(uintptr_t)&seek_data, 0, 0, 0, 0, 0);
        CHECK(add(unseekable, (uint64_t)(uintptr_t)native_source, 0, 0, 0, 0) == 0 &&
                  start(unseekable, 0, 0, 0, 0, 0) == 0,
              "seek probe source starts on the backend that cannot seek");
        CHECK(jump(unseekable, 1200, 0, 0, 0, 0) != 0,
              "sceAvPlayerJumpToTime FAILS when the backend cannot seek (never an implicit SCE_OK)");
        close(unseekable, 0, 0, 0, 0, 0);
        CHECK(jump(0xbadf00d, 1200, 0, 0, 0, 0) != 0,
              "sceAvPlayerJumpToTime FAILS for an unknown player handle");

        SeekingVideoBackend seeker;
        prosper::video::set_backend(&seeker);
        event_count = 0;
        uint64_t seek_handle = init((uint64_t)(uintptr_t)&seek_data, 0, 0, 0, 0, 0);
        CHECK(add(seek_handle, (uint64_t)(uintptr_t)native_source, 0, 0, 0, 0) == 0 &&
                  start(seek_handle, 0, 0, 0, 0, 0) == 0,
              "seekable source starts through the registered backend");
        AvpFrameInfoEx seek_frame{};
        CHECK(video(seek_handle, (uint64_t)(uintptr_t)&seek_frame, 0, 0, 0, 0) == 1 &&
                  seek_frame.timestamp == 0,
              "playback before the seek delivers the first frame at timestamp 0");

        // The guest's own order: pause, then jump. While paused and un-sought, delivery stays shut.
        CHECK(pause(seek_handle, 0, 0, 0, 0, 0) == 0,
              "the player pauses before seeking, as the guest does");
        AvpFrameInfoEx paused_frame{};
        CHECK(video(seek_handle, (uint64_t)(uintptr_t)&paused_frame, 0, 0, 0, 0) == 0,
              "a paused player with no pending seek still delivers nothing");

        CHECK(jump(seek_handle, 2000, 0, 0, 0, 0) == 0,
              "sceAvPlayerJumpToTime succeeds on a backend that can seek");
        CHECK(seeker.seek_calls == 1 && seeker.last_seek_us == 2'000'000,
              "the millisecond guest target reaches the backend as the same microsecond position");
        CHECK(current(seek_handle, 0, 0, 0, 0, 0) == 2000,
              "sceAvPlayerCurrentTime reports the new position before any frame is pulled");

        AvpFrameInfoEx sought_frame{};
        CHECK(video(seek_handle, (uint64_t)(uintptr_t)&sought_frame, 0, 0, 0, 0) == 1,
              "the PAUSED player publishes the post-seek frame (what unblocks the guest's wait)");
        CHECK(sought_frame.timestamp >= 2000 && sought_frame.timestamp < 2020,
              "the delivered post-seek frame is the one at the requested time, not the old position");
        AvpFrameInfoEx after_frame{};
        CHECK(video(seek_handle, (uint64_t)(uintptr_t)&after_frame, 0, 0, 0, 0) == 0,
              "the post-seek delivery window closes after one frame, restoring the paused contract");
        CHECK(resume(seek_handle, 0, 0, 0, 0, 0) == 0 &&
                  video(seek_handle, (uint64_t)(uintptr_t)&after_frame, 0, 0, 0, 0) == 1 &&
                  after_frame.timestamp > sought_frame.timestamp,
              "playback continues forward from the seek target once the guest resumes");

        seeker.support_seek = false;
        CHECK(jump(seek_handle, 500, 0, 0, 0, 0) != 0 && seeker.cursor > 0,
              "a backend that refuses the seek yields an error and leaves the position alone");
        close(seek_handle, 0, 0, 0, 0, 0);
    }

    // ---- the media clock (#1973) ---------------------------------------------------------------
    // Playback used to end only when the DECODER finished and its queue drained. That signal is
    // unreachable for a source nobody pulls from, because prosper's decode workers block on a full
    // queue — so PPSA27624 polled sceAvPlayerIsActive forever on a movie that could not end. A real
    // player is clocked: the media plays out whether or not the application collects frames.
    //
    // The two arms below are the whole contract. The clock must end an UNCONSUMED source at its own
    // media duration, and it must never touch a source something is still pulling from — every
    // title that plays video today (PPSA02664, PPSA02849, PPSA30490) is in the second arm.
    {
        using clock = std::chrono::steady_clock;
        constexpr uint64_t kDurationMs = 300;   // UnconsumedVideoBackend::duration_us
        UnconsumedVideoBackend stalled;
        prosper::video::set_backend(&stalled);
        AvpInitData clock_data{};
        clock_data.event.event_callback = (void*)&on_avplayer_event;

        event_count = 0;
        uint64_t unconsumed = init((uint64_t)(uintptr_t)&clock_data, 0, 0, 0, 0, 0);
        CHECK(add(unconsumed, (uint64_t)(uintptr_t)native_source, 0, 0, 0, 0) == 0 &&
                  start(unconsumed, 0, 0, 0, 0, 0) == 0,
              "an unconsumed source starts on a backend whose decoder can never finish");
        const auto unconsumed_start = clock::now();
        CHECK(active(unconsumed, 0, 0, 0, 0, 0) == 1,
              "an unconsumed player is still active while its media is still playing");
        uint64_t played_out_after_ms = 0;
        bool played_out = false;
        while (clock::now() - unconsumed_start < std::chrono::seconds(10)) {
            if (active(unconsumed, 0, 0, 0, 0, 0) == 0) {
                played_out = true;
                played_out_after_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                    clock::now() - unconsumed_start).count();
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(played_out,
              "a source no consumer ever pulls from still reaches the end of its media (#1973)");
        CHECK(played_out && played_out_after_ms >= kDurationMs,
              "the media clock never ends a movie before the media itself would have ended");
        CHECK(event_count == 3 && events[0] == 2 && events[1] == 3 && events[2] == 1,
              "playing out on the media clock fires exactly one STOP, after READY and PLAY");
        CHECK(active(unconsumed, 0, 0, 0, 0, 0) == 0 && event_count == 3,
              "a played-out player stays inactive without repeating STOP");
        close(unconsumed, 0, 0, 0, 0, 0);

        // THE ARM THAT PROTECTS THE CORPUS. Same never-finishing backend, same short duration — the
        // only difference is that this player is being pulled from. It must stay active for as long
        // as it is, however far past the media duration that runs: a title whose movie plays back
        // slower than real time (prosper renders below 60 FPS routinely) must not be cut short.
        event_count = 0;
        uint64_t consumer = init((uint64_t)(uintptr_t)&clock_data, 0, 0, 0, 0, 0);
        CHECK(add(consumer, (uint64_t)(uintptr_t)native_source, 0, 0, 0, 0) == 0 &&
                  start(consumer, 0, 0, 0, 0, 0) == 0,
              "a consumed source starts on the same never-finishing backend");
        bool consumer_stayed_active = true;
        unsigned consumer_frames = 0;
        const auto consumer_start = clock::now();
        // Four times the clock's own deadline (duration + its guard band), so an over-eager clock
        // has ample room to fire.
        while (clock::now() - consumer_start < std::chrono::milliseconds(4 * (kDurationMs + 250))) {
            AvpFrameInfoEx pulled{};
            if (video(consumer, (uint64_t)(uintptr_t)&pulled, 0, 0, 0, 0) == 1) ++consumer_frames;
            if (active(consumer, 0, 0, 0, 0, 0) == 0) { consumer_stayed_active = false; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(consumer_stayed_active && consumer_frames > 0,
              "a player that keeps pulling frames is never ended by the media clock");
        CHECK(event_count == 2,
              "a consumed player fires no STOP while something is still asking it for frames");
        close(consumer, 0, 0, 0, 0, 0);

        // A pause is not an absent consumer. The guest holds the player paused (PPSA30490 does this
        // around a seek); nothing is presented and nothing should be pulled, so resuming must
        // restart the clock rather than find the whole media duration already spent.
        event_count = 0;
        uint64_t held = init((uint64_t)(uintptr_t)&clock_data, 0, 0, 0, 0, 0);
        CHECK(add(held, (uint64_t)(uintptr_t)native_source, 0, 0, 0, 0) == 0 &&
                  start(held, 0, 0, 0, 0, 0) == 0 && pause(held, 0, 0, 0, 0, 0) == 0,
              "a paused source starts and pauses on the never-finishing backend");
        bool paused_stayed_active = true;
        const auto pause_start = clock::now();
        while (clock::now() - pause_start < std::chrono::milliseconds(2 * (kDurationMs + 250))) {
            if (active(held, 0, 0, 0, 0, 0) == 0) { paused_stayed_active = false; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(paused_stayed_active && event_count == 3 && events[2] == 4,
              "a paused player stays active however long it is held, and fires no STOP");
        CHECK(resume(held, 0, 0, 0, 0, 0) == 0 && active(held, 0, 0, 0, 0, 0) == 1,
              "resuming after a long pause restarts the media clock instead of expiring it");
        close(held, 0, 0, 0, 0, 0);

        // No duration, no clock. A container that does not say how long it is gives prosper nothing
        // to reason about, and inventing an end is exactly the lie #1949 was about.
        stalled.duration_us = 0;
        event_count = 0;
        uint64_t timeless = init((uint64_t)(uintptr_t)&clock_data, 0, 0, 0, 0, 0);
        CHECK(add(timeless, (uint64_t)(uintptr_t)native_source, 0, 0, 0, 0) == 0 &&
                  start(timeless, 0, 0, 0, 0, 0) == 0,
              "a source with no reported duration starts");
        bool timeless_stayed_active = true;
        const auto timeless_start = clock::now();
        while (clock::now() - timeless_start < std::chrono::milliseconds(4 * (kDurationMs + 250))) {
            if (active(timeless, 0, 0, 0, 0, 0) == 0) { timeless_stayed_active = false; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(timeless_stayed_active && event_count == 2,
              "a source whose duration is unknown is never clocked out, and fires no STOP");
        close(timeless, 0, 0, 0, 0, 0);
        prosper::video::set_backend(nullptr);
    }

    // ---- the 256-ALIGNED width, which nothing else in this file covers (#2051) -----------------
    // Every assertion above runs against FakeVideoBackend at 1920, so pitch != width and
    // crop_right == 128 everywhere. This is the other half of the contract: at 1280 the padding
    // does not exist, so width == pitch == visible and crop_right == 0.
    //
    // What it kills is specific, and both edits are plausible: a refactor that pads
    // UNCONDITIONALLY, and one that hardcodes crop_right = 128. Either passes the entire rest of
    // this file and breaks every 1280x720 or 3840x2160 movie -- the majority of real content, and
    // exactly the set over which #2032 established its change is a bit-for-bit no-op.
    {
        AlignedVideoBackend aligned;
        prosper::video::set_backend(&aligned);
        AvpInitData aligned_data{};
        aligned_data.memory.obj = texture_expected_object;
        aligned_data.memory.allocate_texture = (void*)&on_avplayer_texture_allocate;
        aligned_data.memory.deallocate_texture = (void*)&on_avplayer_texture_deallocate;
        aligned_data.num_fb = 3;
        uint64_t ah = init((uint64_t)(uintptr_t)&aligned_data, 0, 0, 0, 0, 0);
        const char aligned_source[] = "aligned.mp4";
        CHECK(add(ah, (uint64_t)(uintptr_t)aligned_source, 0, 0, 0, 0) == 0,
              "aligned-width source opens through the registered backend");

        AvpStreamInfoEx ai{}; ai.size = sizeof(ai);
        uint32_t aw = 0, acl = 0, acr = 0, act = 0, acb = 0, ap = 0;
        CHECK(infoex(ah, 0, (uint64_t)(uintptr_t)&ai, 0, 0, 0) == 0 && ai.type == 1,
              "aligned-width stream metadata comes from the backend");
        memcpy(&aw,  ai.details + 0,  sizeof(aw));
        memcpy(&acl, ai.details + 20, sizeof(acl));
        memcpy(&acr, ai.details + 24, sizeof(acr));
        memcpy(&act, ai.details + 28, sizeof(act));
        memcpy(&acb, ai.details + 32, sizeof(acb));
        memcpy(&ap,  ai.details + 36, sizeof(ap));

        // THE ARM. crop_right == 0 is what a hardcoded 128 fails; width == pitch is what an
        // unconditional pad fails. Neither is expressible against the 1920 backend.
        CHECK(acr == 0, "aligned width crops NOTHING on the right (a hardcoded 128 fails here)");
        CHECK(aw == AlignedVideoBackend::kWidth && ap == AlignedVideoBackend::kWidth &&
                  aw == ap,
              "aligned width publishes width == pitch == visible (an unconditional pad fails here)");
        CHECK(ap - acl - acr == AlignedVideoBackend::kWidth,
              "the pitch-based crop spelling yields the full visible width when nothing is padded");

        // The height half, unpinned until now and asymmetric on purpose: prosper never stages padded
        // ROWS, and the code comment warns a future editor not to "finish the job" by making height
        // a macroblock-aligned 1088. This documents that intent where an edit would trip on it.
        CHECK(act == 0 && acb == 0,
              "height is not cropped: prosper stages no padded rows (720 stays 720, not 1088)");

        close(ah, 0, 0, 0, 0, 0);
        prosper::video::set_backend(nullptr);
    }

#ifndef _WIN32
    // ---- guest file-replacement reader (#1955) -------------------------------------------------
    // A title whose clip lives at an offset inside a container file (PPSA27624 plays a VideoClip out
    // of Unity's resources.resource) can only express that through sceAvPlayerInit's file-replacement
    // callbacks. Ignoring them made prosper demux the container's first bytes, which are a different
    // format entirely. The source must therefore come from the guest's reader, not the host path.
    {
        guest_container.assign(kGuestMediaOffset + kGuestMediaBytes, 0x11);
        for (size_t i = 0; i < kGuestMediaBytes; ++i)
            guest_container[kGuestMediaOffset + i] = static_cast<uint8_t>(0x40 + (i % 191));
        guest_file_open_calls = guest_file_close_calls = 0;
        guest_file_is_open = false;
        int file_object = 0x5678;
        guest_file_expected_object = &file_object;

        MemorySourceBackend memory_backend;
        prosper::video::set_backend(&memory_backend);
        AvpInitData file_data{};
        file_data.event.event_callback = (void*)&on_avplayer_event;
        file_data.file.obj = guest_file_expected_object;
        file_data.file.open = (void*)&on_avplayer_file_open;
        file_data.file.close = (void*)&on_avplayer_file_close;
        file_data.file.read_offset = (void*)&on_avplayer_file_read;
        file_data.file.size = (void*)&on_avplayer_file_size;
        uint64_t file_handle = init((uint64_t)(uintptr_t)&file_data, 0, 0, 0, 0, 0);
        const char container_source[] = "/app0/Media/resources.resource";
        CHECK(add(file_handle, (uint64_t)(uintptr_t)container_source, 0, 0, 0, 0) == 0,
              "a source with a guest file-replacement table opens");
        CHECK(memory_backend.memory_open_calls == 1 && memory_backend.host_open_calls == 0,
              "the source is demuxed from the guest reader's bytes, never from the host file");
        CHECK(memory_backend.received.size() == kGuestMediaBytes &&
                  memcmp(memory_backend.received.data(),
                         guest_container.data() + kGuestMediaOffset, kGuestMediaBytes) == 0,
              "the backend receives the media at the guest's offset, not the container's first bytes");
        CHECK(guest_file_open_calls == 1 && guest_file_close_calls == 1 && !guest_file_is_open,
              "the guest reader is opened once and closed once");
        close(file_handle, 0, 0, 0, 0, 0);

        // A guest table is not permission to lose the titles that work today: when the reader has
        // nothing to give, the host path must still be used.
        guest_file_expected_object = nullptr;   // every guest callback now fails
        FakeVideoBackend fallback;
        prosper::video::set_backend(&fallback);
        uint64_t fallback_handle = init((uint64_t)(uintptr_t)&file_data, 0, 0, 0, 0, 0);
        CHECK(add(fallback_handle, (uint64_t)(uintptr_t)native_source, 0, 0, 0, 0) == 0 &&
                  fallback.opened_path == "C:/prosper-test-app0/movie.mp4",
              "a failing guest reader falls back to the host path instead of failing the source");
        close(fallback_handle, 0, 0, 0, 0, 0);
    }
#endif
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
    const uint64_t synthetic_frame_address =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(frame.data));
    uint32_t width = 0, height = 0, crop_left = 0, crop_right = 0;
    uint32_t crop_top = 0, crop_bottom = 0, pitch = 0;
    double fps = 0.0;
    memcpy(&width, frame.details + 0, sizeof(width));
    memcpy(&height, frame.details + 4, sizeof(height));
    memcpy(&crop_left, frame.details + 20, sizeof(crop_left));
    memcpy(&crop_right, frame.details + 24, sizeof(crop_right));
    memcpy(&crop_top, frame.details + 28, sizeof(crop_top));
    memcpy(&crop_bottom, frame.details + 32, sizeof(crop_bottom));
    memcpy(&pitch, frame.details + 36, sizeof(pitch));
    memcpy(&fps, frame.details + 48, sizeof(fps));
    CHECK(width == 2048 && height == 1080 && pitch == 2048 && crop_left == 0 &&
              crop_right == 128 && crop_top == 0 && crop_bottom == 0 && fps == 30.0,
          "AvPlayerVideoEx uses the published 80-byte padded-pitch/crop layout");
    // #2011: the synthetic Ex path publishes the same contract as the native one — width is the
    // CODED extent the crop offsets trim, so both spellings of "visible width" agree at 1920.
    CHECK(width - crop_left - crop_right == 1920 && pitch - crop_left - crop_right == 1920,
          "synthetic Ex: the width- and pitch-based crop spellings agree on 1920 visible");
    CHECK(gpu::guest_linear_texture_row_pitch(synthetic_frame_address, pitch) == pitch,
          "fallback AvPlayer storage also publishes its exact physical layout");
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
    close(h2, 0, 0, 0, 0, 0);
    CHECK(gpu::guest_linear_texture_row_pitch(synthetic_frame_address, pitch) == 0,
          "Close removes fallback AvPlayer layout provenance");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
