// hle_service.cpp — HLE of PS5 system services (user, NP/online, mouse, app content,
// dialogs). Bring-up policy: openers return a valid positive handle; queries zero their
// output struct and report a sane "not signed in / no device" state and success, so the
// game gets consistent values instead of uninitialized memory.
// (Game-controller input — libScePad — moved to hle_pad.cpp with a real host backend.)
#include "dispatch.hpp"
#include "hle_json2.hpp"
#include "nid.hpp"
#include "callback_fs.hpp"
#include "platform_ui.hpp"
#include "video_backend.hpp"   // sceAvPlayer -> host hardware-decode backend (#705)
#include "../host/posix_shim.hpp"   // Darwin process_vm_readv shim + asm portability
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#ifdef _WIN32
#include <direct.h>     // _mkdir (SaveDataMemory persistence dir)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>   // mkdir
#include <sys/uio.h>    // process_vm_readv: fault-contained diagnostic snapshots
#include <unistd.h>
#endif

namespace prosper {

#ifdef _WIN32
extern "C" uint64_t prosper_call_guest_sysv4(uint64_t fn, uint64_t a0, uint64_t a1,
                                               uint64_t a2, uint64_t a3);
#endif

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define PW(x) ((void*)(uintptr_t)(x))

namespace { std::atomic<uint64_t> g_handle{1}; }

// Diagnostic logging for the Sony service families in this file, gated on PROSPER_SVCLOG=1 (same
// pattern as PROSPER_FILELOG/[file]). Dumps call args and a bounded hexdump of pointer-shaped args
// so PS5-only ABIs with no Kyty/shadPS4 reference can be pinned from live captures instead of
// guessed.
namespace {
bool svclog() { static int v = getenv("PROSPER_SVCLOG") ? 1 : 0; return v; }
bool svc_ptrish(uint64_t v) { return v >= 0x10000 && v < 0x7fffffffffffull; }
size_t svc_copy_words(uint64_t src, uint64_t* dst, size_t words) {
    const size_t bytes = words * sizeof(uint64_t);
#ifdef _WIN32
    SIZE_T copied = 0;
    ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)src, dst, bytes, &copied);
    return (size_t)copied / sizeof(uint64_t);
#else
    iovec local{dst, bytes};
    iovec remote{(void*)(uintptr_t)src, bytes};
    const ssize_t copied = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    return copied > 0 ? (size_t)copied / sizeof(uint64_t) : 0;
#endif
}
void svc_log(const char* fn, uint64_t a0, uint64_t a1, uint64_t a2,
             uint64_t a3, uint64_t a4, uint64_t a5, int dump_words = 8) {
    if (!svclog()) return;
    fprintf(stderr, "[svc] %s(%#" PRIx64 ", %#" PRIx64 ", %#" PRIx64
                    ", %#" PRIx64 ", %#" PRIx64 ", %#" PRIx64 ")\n",
            fn, a0, a1, a2, a3, a4, a5);
    const uint64_t args[6] = { a0, a1, a2, a3, a4, a5 };
    for (int i = 0; i < 6; i++) {
        if (!svc_ptrish(args[i])) continue;
        // Never read across the arg's 4 KiB page end: an out-param can be a tiny heap block whose
        // page neighbor is unmapped, and a diagnostic must not be able to fault the boot.
        uint64_t page_left = 0x1000 - (args[i] & 0xfff);
        int words = (int)(page_left / 8); if (words > dump_words) words = dump_words;
        if (words > 8) words = 8;
        uint64_t q[8]{};
        words = (int)svc_copy_words(args[i], q, (size_t)words);
        if (!words) continue;
        fprintf(stderr, "[svc]   a%d ->", i);
        for (int w = 0; w < words; w++) fprintf(stderr, " %016" PRIx64, q[w]);
        fprintf(stderr, "\n");
    }
}
}

// --- user service ---
HLE(s_user_initial)   { if (a0) *(int32_t*)PW(a0) = 1; return 0; }           // GetInitialUser -> userId 1
HLE(s_user_idlist)    { if (a0) { int32_t* p = (int32_t*)PW(a0); p[0] = 1; for (int i = 1; i < 4; i++) p[i] = -1; } return 0; }
// Bounded, non-padding write (strncpy would zero-pad the whole a2-byte buffer -> a
// stack-smash if a2 is large/garbage). snprintf writes only the string + NUL.
HLE(s_user_name)      { if (a1) snprintf((char*)PW(a1), a2 ? (size_t)a2 : 17, "%s", "Player"); return 0; }
HLE(s_user_int_out)   { if (a1) *(int32_t*)PW(a1) = 0; return 0; }           // accessibility getters -> 0
HLE(s_user_age)       { if (a1) *(int32_t*)PW(a1) = 18; return 0; }          // GetAgeLevel -> adult (no restriction)
// sceUserServiceGetEvent(SceUserServiceEvent* ev): the event stream. A real system delivers the initial
// user's LOGIN event once at startup, then reports "no more events" so the game's drain loop terminates.
// The previous (unimplemented) stub returned 0 = "got an event" but left the struct unfilled -> the game
// either drained a garbage event or never saw the login it waits on. SceUserServiceEvent = { int32
// eventType (0=LOGIN,1=LOGOUT); int32 userId }.
// NO_EVENT = 0x80960007 (Kyty Errno.h USER_SERVICE_ERROR_NO_EVENT). We first shipped 0x80960009 —
// a DIFFERENT UserService error — and the game's main-thread drain loop, not recognizing it as
// "no more events", retried GetEvent forever: the frame loop never built frame 2 (gdb-sampled spin,
// 4/6 PC samples inside this function). One wrong errno constant == a full render stall.
HLE(s_user_getevent)  {
    static std::atomic<int> delivered{0};
    if (a0 && delivered.exchange(1) == 0) { int32_t* ev = (int32_t*)PW(a0); ev[0] = 0; ev[1] = 1; return 0; }
    return 0x80960007ull;   // SCE_USER_SERVICE_ERROR_NO_EVENT
}
// sceSystemServiceReceiveEvent(SceSystemServiceEvent* ev): the system event stream (resume, launch-app,
// entitlement-update, share-menu, ...). Unregistered, it fell to the return-0 stub = "an event was
// received" while leaving the 8196-byte out-struct (4-byte eventType + 8192-byte union) uninitialized ->
// the guest dispatched on a garbage eventType (same harmful class as s_user_getevent above; imported by
// both Unity targets). The truthful idle answer is NO_EVENT with nothing written. shadPS4
// systemservice.cpp: NO_EVENT = 0x80A10004, PARAMETER (ev==NULL) = 0x80A10003.
HLE(s_sysservice_receiveevent) { if (!a0) return 0x80A10003ull; return 0x80A10004ull; }
HLE(s_ok)             { return 0; }
// ===== libSceAvPlayer (#324/#705): real playback lifecycle over a host video-decode backend =====
// The core owns the guest sceAvPlayer contract + per-player state + the guest event callback and
// pulls decoded frames from a registered VideoBackend (app-side hardware decode). With NO backend
// (headless/tests) it runs a synthetic black-frame lifecycle so the title's video state machine still
// completes and the boot advances. ABI structs mirror the published sceAvPlayer layout (guest is
// SysV LP64 == host on Linux, so guest pointers are read directly).
namespace {
// Guest event callback: void(void* object_ptr, uint32_t event, int32_t source_id, void* data).
using AvpEventCb = void (PROSPER_SYSV_ABI *)(void*, uint32_t, int32_t, void*);
enum : uint32_t { AVP_STOP = 0x01, AVP_READY = 0x02, AVP_PLAY = 0x03, AVP_PAUSE = 0x04, AVP_BUFFERING = 0x05 };

struct AvpMemAllocator  { void* obj; void* allocate; void* deallocate; void* allocate_texture; void* deallocate_texture; };
struct AvpFileReplace   { void* obj; void* open; void* close; void* read_offset; void* size; };
struct AvpEventReplace  { void* obj; void* event_callback; };
struct AvpInitData {                 // sceAvPlayerInit
    AvpMemAllocator memory; AvpFileReplace file; AvpEventReplace event;
    uint32_t debug_level; uint32_t base_priority; int32_t num_fb; uint8_t auto_start; uint8_t rsv[3]; const char* default_language;
};
struct AvpThreadInfo {
    uint32_t priority, stack_size; uint64_t affinity; uint8_t reserved[32];
};
struct AvpInitDataEx {               // sceAvPlayerInitEx (note: this_size first)
    uint64_t this_size; AvpMemAllocator memory; AvpFileReplace file; AvpEventReplace event;
    const char* default_language; uint32_t debug_level; uint8_t auto_start; uint8_t rsv[3];
    AvpThreadInfo audio_decoder, video_decoder, demuxer, event_thread, call_queue;
    AvpThreadInfo http_command_processor, http_segment_manager, http_streamlist, file_streaming;
    int32_t num_fb; uint8_t rsv2[4];
};
static_assert(sizeof(AvpThreadInfo) == 48 && sizeof(AvpInitDataEx) == 560 &&
              offsetof(AvpInitDataEx, auto_start) == 116 &&
              offsetof(AvpInitDataEx, num_fb) == 552, "AvPlayerInitDataEx ABI");
struct AvpUri           { const char* name; uint32_t length; };
struct AvpSourceDetails { AvpUri uri; uint8_t rsv1[64]; uint32_t source_type; uint8_t rsv2[44]; };
struct AvpFrameInfo {                // sceAvPlayerGetVideoData / GetAudioData out-param
    uint8_t* p_data; uint8_t reserved[4]; uint64_t timestamp;
    uint32_t d0, d1, d2, d3;         // AvPlayerStreamDetails union (16B): video={width,height,aspect,lang}
};
struct AvpVideoEx {                  // AvPlayerVideoEx (in the 80B details union)
    uint32_t width, height; float aspect; uint8_t lang[4]; uint8_t reserved0[4];
    uint32_t crop_l, crop_r, crop_t, crop_b, pitch;
    uint8_t luma_bd, chroma_bd, full_range, reserved1[5];
    double framerate; uint32_t colour_primaries, transfer_characteristics; uint8_t reserved2[16];
};
static_assert(sizeof(AvpVideoEx) == 80 && offsetof(AvpVideoEx, pitch) == 36 &&
              offsetof(AvpVideoEx, framerate) == 48, "AvPlayerVideoEx ABI");
struct AvpFrameInfoEx {              // sceAvPlayerGetVideoDataEx out-param (larger details union)
    void* p_data; uint8_t reserved[4]; uint64_t timestamp;
    union { AvpVideoEx video; uint8_t raw[80]; } details;
};
struct AvpVideo { uint32_t width, height; float aspect; uint8_t lang[4]; };
struct AvpAudio { uint16_t channels; uint8_t reserved[2]; uint32_t sample_rate, size; uint8_t lang[4]; };
struct AvpAudioEx {
    uint16_t channels; uint8_t reserved[2]; uint32_t sample_rate, size; uint8_t lang[4];
    uint8_t reserved1[64];
};
static_assert(sizeof(AvpAudio) == 16 && sizeof(AvpAudioEx) == 80, "AvPlayerAudio ABI");
struct AvpStreamInfo {
    uint32_t type; uint8_t reserved[4];
    union { AvpVideo video; AvpAudio audio; uint8_t raw[16]; } details; uint64_t duration;
};
struct AvpStreamInfoEx {
    uint64_t this_size; uint32_t type; uint8_t reserved[4];
    union { AvpVideoEx video; AvpAudioEx audio; uint8_t raw[80]; } details; uint64_t duration;
};
static_assert(sizeof(AvpStreamInfo) == 32 && offsetof(AvpStreamInfo, duration) == 24,
              "AvPlayerStreamInfo ABI");
static_assert(sizeof(AvpStreamInfoEx) == 104 && offsetof(AvpStreamInfoEx, duration) == 96,
              "AvPlayerStreamInfoEx ABI");

struct AvpPlayer {
    void* ev_obj = nullptr; AvpEventCb ev_cb = nullptr;
    bool auto_start = false, have_source = false, playing = false, paused = false, stop_fired = false;
    std::string guest_path;
    int backend_id = -1;             // >=0 when a host backend is decoding
    uint32_t width = 1920, height = 1080;
    uint64_t poll = 0;               // synthetic frames delivered through GetVideoData[Ex]
    uint64_t audio_poll = 0;
    uint64_t active_poll = 0;        // independent progress for IsActive-only playback loops
    std::vector<uint8_t> frame;      // synthetic black NV12 buffer (kept alive across GetVideoData)
    std::vector<int16_t> audio;      // synthetic silent stereo PCM
};
std::mutex g_avp_mx;
std::unordered_map<uint64_t, AvpPlayer> g_avp;

uint64_t avp_synth_frames() { static uint64_t n = []{ const char* e = getenv("PROSPER_AVP_SYNTH_FRAMES"); return e ? strtoull(e, nullptr, 0) : 120ull; }(); return n; }
uint64_t avp_synth_duration_ms() {
    static uint64_t n = [] {
        const char* e = getenv("PROSPER_AVP_DURATION_MS");
        return e ? strtoull(e, nullptr, 0) : avp_synth_frames() * 33;
    }();
    return n;
}
bool avp_log() { static const bool enabled = getenv("PROSPER_AVPLOG") != nullptr; return enabled; }

#if defined(__linux__)
// Linux import stubs swap the guest %fs to the host TCB before entering an HLE handler.  AvPlayer
// event callbacks go the other way and are guest code, so they must run with the caller's guest TCB.
// The entry shims below recover that TCB from the import-stub frame and publish it only for the
// dynamic extent of the HLE call.  Nested AvPlayer calls from the callback preserve the outer value.
thread_local uint64_t t_avp_callback_guest_fs = 0;
struct AvpEntryGuestFsScope {
    uint64_t previous;
    explicit AvpEntryGuestFsScope(uint64_t guest_fs)
        : previous(t_avp_callback_guest_fs) { t_avp_callback_guest_fs = guest_fs; }
    ~AvpEntryGuestFsScope() { t_avp_callback_guest_fs = previous; }
};
struct AvpCallbackGuestFsScope {
    uint64_t saved = 0;
    bool active = false;
    explicit AvpCallbackGuestFsScope(uint64_t guest_fs) {
        if (guest_fs) {
            __asm__ volatile("rdfsbase %0" : "=r"(saved));
            __asm__ volatile("wrfsbase %0" : : "r"(guest_fs));
            active = true;
        }
    }
    ~AvpCallbackGuestFsScope() {
        if (active) __asm__ volatile("wrfsbase %0" : : "r"(saved));
    }
};
#endif

// Fire the guest event callback. Caller MUST NOT hold g_avp_mx (the callback re-enters AvPlayer HLE).
void avp_fire(void* obj, AvpEventCb cb, uint32_t ev) {
    if (!cb) return;
    if (svclog() || avp_log()) fprintf(stderr, "[avp] -> event 0x%02x\n", ev);
#if defined(_WIN32)
    prosper_call_guest_sysv4((uint64_t)(uintptr_t)cb, (uint64_t)(uintptr_t)obj, ev, 0, 0);
#elif defined(__linux__)
    {
        AvpCallbackGuestFsScope guest_fs(t_avp_callback_guest_fs);
        cb(obj, ev, 0, nullptr);
    }
#else
    cb(obj, ev, 0, nullptr);
#endif
    if (avp_log()) fprintf(stderr, "[avp] <- event 0x%02x\n", ev);
}
} // namespace

HLE(s_avplayer_init) {   // AvPlayerHandle sceAvPlayerInit(AvPlayerInitData*)
    svc_log("sceAvPlayerInit", a0,a1,a2,a3,a4,a5);
    uint64_t h = g_handle.fetch_add(1);
    AvpPlayer p;
    if (auto* d = (const AvpInitData*)PW(a0)) { p.ev_obj = d->event.obj; p.ev_cb = (AvpEventCb)d->event.event_callback; p.auto_start = d->auto_start != 0; }
    { std::lock_guard<std::mutex> lk(g_avp_mx); g_avp[h] = std::move(p); }
    return h;   // non-NULL SceAvPlayerHandle
}
// sceAvPlayerInitEx returns an int32 error code (0 = success) and writes the handle to *out — a
// DIFFERENT ABI from sceAvPlayerInit (which returns the handle). Returning the handle here made the
// game read it as an error and abort the intro (live-captured, PPSA02664). CONFIDENCE: HIGH.
HLE(s_avplayer_initex) {   // s32 sceAvPlayerInitEx(const AvPlayerInitDataEx*, AvPlayerHandle* out)
    svc_log("sceAvPlayerInitEx", a0,a1,a2,a3,a4,a5);
    uint64_t h = g_handle.fetch_add(1);
    AvpPlayer p;
    if (auto* d = (const AvpInitDataEx*)PW(a0)) { p.ev_obj = d->event.obj; p.ev_cb = (AvpEventCb)d->event.event_callback; p.auto_start = d->auto_start != 0; }
    { std::lock_guard<std::mutex> lk(g_avp_mx); g_avp[h] = std::move(p); }
    if (a1) *(uint64_t*)PW(a1) = h;
    return 0;
}
HLE(s_avplayer_isactive) {   // bool sceAvPlayerIsActive(AvPlayerHandle)
    svc_log("sceAvPlayerIsActive", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false; uint64_t active = 0;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0;
        AvpPlayer& p = it->second;
        if (p.playing) {
            if (p.paused) return 1;
            p.active_poll++;
            bool still = (p.backend_id >= 0 && prosper::video::backend())
                             ? !prosper::video::backend()->eof(p.backend_id)
                             : (p.active_poll < avp_synth_frames());
            if (still) { active = 1; }
            else if (!p.stop_fired) { p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb; }
        }
    }
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);   // playback complete -> game advances past the video
    return active;
}
HLE(s_avp_postinit)   { svc_log("sceAvPlayerPostInit", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_avp_setlogcb)   { svc_log("sceAvPlayerSetLogCallback", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_avp_streamcount) {   // s32 sceAvPlayerStreamCount(handle)
    svc_log("sceAvPlayerStreamCount", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0);
    int count = it != g_avp.end() && it->second.have_source ? 2 : 0;
    if (avp_log()) fprintf(stderr, "[avp] stream-count handle=0x%llx -> %d\n",
                           (unsigned long long)a0, count);
    return count; // MP4-shaped synthetic source: video + audio
}
HLE(s_avp_getstreaminfo) { // s32 sceAvPlayerGetStreamInfo(handle, stream_id, out)
    svc_log("sceAvPlayerGetStreamInfo", a0,a1,a2,a3,a4,a5);
    auto* out = (AvpStreamInfo*)PW(a2); if (!out || a1 > 1) return 0x806a0001ull;
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0); if (it == g_avp.end() || !it->second.have_source) return 0x806a0001ull;
    const AvpPlayer& p = it->second;
    *out = {}; out->type = a1 == 0 ? 1u : 2u;
    if (a1 == 0) {
        out->details.video.width = p.width; out->details.video.height = p.height;
        out->details.video.aspect = (float)p.width / (float)p.height;
    } else {
        out->details.audio.channels = 2; out->details.audio.sample_rate = 48000;
        out->details.audio.size = 1024 * 2 * sizeof(int16_t);
    }
    out->duration = avp_synth_duration_ms();
    return 0;
}
HLE(s_avp_getstreaminfoex) { // s32 sceAvPlayerGetStreamInfoEx(handle, stream_id, out)
    svc_log("sceAvPlayerGetStreamInfoEx", a0,a1,a2,a3,a4,a5);
    auto* out = (AvpStreamInfoEx*)PW(a2); if (!out || a1 > 1) return 0x806a0001ull;
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0); if (it == g_avp.end() || !it->second.have_source) return 0x806a0001ull;
    const AvpPlayer& p = it->second; const uint64_t caller_size = out->this_size;
    *out = {}; out->this_size = caller_size; out->type = a1 == 0 ? 1u : 2u;
    if (a1 == 0) {
        out->details.video.width = p.width; out->details.video.height = p.height;
        out->details.video.aspect = (float)p.width / (float)p.height; out->details.video.pitch = p.width;
        out->details.video.luma_bd = 8; out->details.video.chroma_bd = 8;
        out->details.video.framerate = 30.0;
    } else {
        out->details.audio.channels = 2; out->details.audio.sample_rate = 48000;
        out->details.audio.size = 1024 * 2 * sizeof(int16_t);
    }
    out->duration = avp_synth_duration_ms();
    return 0;
}
HLE(s_avp_stream_ok) { svc_log("sceAvPlayerStreamControl", a0,a1,a2,a3,a4,a5); return 0; }
// Begin a source: resolve/open it, fire READY, and (auto_start) begin playback with PLAY.
static uint64_t avp_add_source(uint64_t handle, const char* guest_path) {
    void* obj = nullptr; AvpEventCb cb = nullptr; bool ready = false, play = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(handle); if (it == g_avp.end()) return 0x80000000ull;
        AvpPlayer& p = it->second;
        if (guest_path) p.guest_path = guest_path;
        p.have_source = true;
        p.poll = 0; p.audio_poll = 0; p.active_poll = 0; p.paused = false; p.stop_fired = false;
        if (auto* b = prosper::video::backend()) p.backend_id = b->open(p.guest_path);  // TODO(#705): guest->host path resolve for real backend
        obj = p.ev_obj; cb = p.ev_cb; ready = true;
        if (p.auto_start) { p.playing = true; play = true; }
    }
    if (ready) avp_fire(obj, cb, AVP_READY);
    if (play)  avp_fire(obj, cb, AVP_PLAY);
    if (avp_log()) fprintf(stderr, "[avp] add source handle=0x%llx path=%s auto_start=%d\n",
                           (unsigned long long)handle, guest_path ? guest_path : "", (int)play);
    return 0;
}
HLE(s_avp_addsource)   {   // s32 sceAvPlayerAddSource(handle, const char* filename)
    svc_log("sceAvPlayerAddSource", a0,a1,a2,a3,a4,a5);
    return avp_add_source(a0, (const char*)PW(a1));
}
HLE(s_avp_addsourceex) {   // s32 sceAvPlayerAddSourceEx(handle, AvPlayerUriType, AvPlayerSourceDetails*)
    svc_log("sceAvPlayerAddSourceEx", a0,a1,a2,a3,a4,a5);
    const char* path = nullptr;
    if (auto* d = (const AvpSourceDetails*)PW(a2)) path = d->uri.name;
    return avp_add_source(a0, path);
}
HLE(s_avp_start) {   // s32 sceAvPlayerStart(handle) — used when auto_start is false
    svc_log("sceAvPlayerStart", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool play = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0x80000000ull;
        AvpPlayer& p = it->second;
        if (p.have_source && !p.playing && !p.stop_fired) {
            p.playing = true; p.paused = false; play = true; obj = p.ev_obj; cb = p.ev_cb;
        }
    }
    if (play) avp_fire(obj, cb, AVP_PLAY);
    return 0;
}
HLE(s_avp_pause) {   // s32 sceAvPlayerPause(handle)
    svc_log("sceAvPlayerPause", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool notify = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0x806a0001ull;
        if (it->second.playing && !it->second.paused) {
            it->second.paused = true;
            obj = it->second.ev_obj; cb = it->second.ev_cb; notify = true;
        }
    }
    if (notify) avp_fire(obj, cb, AVP_PAUSE);
    return 0;
}
HLE(s_avp_resume) {   // s32 sceAvPlayerResume(handle)
    svc_log("sceAvPlayerResume", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool notify = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0x806a0001ull;
        if (it->second.playing && it->second.paused) {
            it->second.paused = false;
            obj = it->second.ev_obj; cb = it->second.ev_cb; notify = true;
        }
    }
    if (notify) avp_fire(obj, cb, AVP_PLAY);
    return 0;
}
// bool sceAvPlayerGetVideoData(handle, AvPlayerFrameInfo*) — deliver the next decoded frame (NV12).
HLE(s_avp_getvideodata) {
    svc_log("sceAvPlayerGetVideoData", a0,a1,a2,a3,a4,a5);
    auto* fi = (AvpFrameInfo*)PW(a1); if (!fi) return 0;
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false; uint64_t result = 0;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0);
        if (it == g_avp.end() || !it->second.playing || it->second.paused) return 0;
        AvpPlayer& p = it->second;
        if (auto* b = prosper::video::backend(); b && p.backend_id >= 0) {
            prosper::video::VideoFrame vf;
            if (b->next_video(p.backend_id, vf)) {
                fi->p_data = const_cast<uint8_t*>(vf.y); fi->timestamp = vf.pts_us / 1000;
                fi->d0 = vf.width; fi->d1 = vf.height; fi->d2 = 0; fi->d3 = 0;
                result = 1;
            } else if (b->eof(p.backend_id) && !p.stop_fired) {
                p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
            }
        } else if (p.poll < avp_synth_frames()) {
            // Synthetic (headless): a black NV12 frame at the default dimensions.  Advance on
            // frame delivery: Astro Bot never polls IsActive, so an IsActive-only counter made EOF
            // and the STOP event unreachable even though it continuously pulled video frames.
            size_t need = (size_t)p.width * p.height * 3 / 2;
            if (p.frame.size() != need) p.frame.assign(need, 0);
            fi->p_data = p.frame.data(); fi->timestamp = p.poll * 33ull; // ~30fps PTS (ms)
            fi->d0 = p.width; fi->d1 = p.height; fi->d2 = 0; fi->d3 = 0;
            p.poll++; result = 1;
        } else if (!p.stop_fired) {
            p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
        }
    }
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);
    if (avp_log()) fprintf(stderr, "[avp] video handle=0x%llx result=%llu stop=%d\n",
                           (unsigned long long)a0, (unsigned long long)result, (int)fire_stop);
    return result;
}
// bool sceAvPlayerGetVideoDataEx(handle, AvPlayerFrameInfoEx*) — the game uses this Ex variant.
HLE(s_avp_getvideodataex) {
    svc_log("sceAvPlayerGetVideoDataEx", a0,a1,a2,a3,a4,a5);
    auto* fi = (AvpFrameInfoEx*)PW(a1); if (!fi) return 0;
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false; uint64_t result = 0;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0);
        if (it == g_avp.end() || !it->second.playing || it->second.paused) return 0;
        AvpPlayer& p = it->second;
        if (auto* b = prosper::video::backend(); b && p.backend_id >= 0) {
            prosper::video::VideoFrame vf;
            if (b->next_video(p.backend_id, vf)) {
                fi->p_data = const_cast<uint8_t*>(vf.y); fi->timestamp = vf.pts_us / 1000;
                fi->details.video = {}; fi->details.video.width = vf.width; fi->details.video.height = vf.height;
                fi->details.video.aspect = vf.height ? (float)vf.width / (float)vf.height : 0.0f;
                fi->details.video.pitch = vf.y_stride ? vf.y_stride : vf.width;
                fi->details.video.luma_bd = 8; fi->details.video.chroma_bd = 8;
                fi->details.video.framerate = 30.0;
                result = 1;
            } else if (b->eof(p.backend_id) && !p.stop_fired) {
                p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
            }
        } else if (p.poll < avp_synth_frames()) {
            size_t need = (size_t)p.width * p.height * 3 / 2;        // synthetic black NV12
            if (p.frame.size() != need) p.frame.assign(need, 0);
            fi->p_data = p.frame.data(); fi->timestamp = p.poll * 33ull;
            fi->details.video = {}; fi->details.video.width = p.width; fi->details.video.height = p.height;
            fi->details.video.aspect = (float)p.width / (float)p.height; fi->details.video.pitch = p.width;
            fi->details.video.luma_bd = 8; fi->details.video.chroma_bd = 8;
            fi->details.video.framerate = 30.0;
            p.poll++; result = 1;
        } else if (!p.stop_fired) {
            p.playing = false; p.stop_fired = true; fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
        }
    }
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);
    if (avp_log()) fprintf(stderr, "[avp] video-ex handle=0x%llx result=%llu stop=%d\n",
                           (unsigned long long)a0, (unsigned long long)result, (int)fire_stop);
    return result;
}
HLE(s_avp_getaudiodata)   {   // bool sceAvPlayerGetAudioData(handle, AvPlayerFrameInfo*)
    svc_log("sceAvPlayerGetAudioData", a0,a1,a2,a3,a4,a5);
    auto* b = prosper::video::backend();
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0);
    if (it == g_avp.end() || !it->second.playing || it->second.paused) return 0;
    auto* fi = (AvpFrameInfo*)PW(a1); if (!fi) return 0;
    AvpPlayer& p = it->second;
    if (b && p.backend_id >= 0) {
        prosper::video::AudioFrame af;
        if (!b->next_audio(p.backend_id, af)) return 0;
        fi->p_data = (uint8_t*)const_cast<int16_t*>(af.pcm); fi->timestamp = af.pts_us / 1000;
        return 1;
    }
    if (p.audio_poll >= avp_synth_frames()) return 0;
    constexpr uint32_t samples = 1024, channels = 2;
    if (p.audio.size() != samples * channels) p.audio.assign(samples * channels, 0);
    fi->p_data = (uint8_t*)p.audio.data();
    fi->timestamp = p.audio_poll * 21ull; // 1024/48 kHz, milliseconds
    AvpAudio details{2, {}, 48000, samples * channels * sizeof(int16_t), {}};
    memcpy(&fi->d0, &details, sizeof(details));
    p.audio_poll++;
    if (avp_log()) fprintf(stderr, "[avp] audio handle=0x%llx result=1 poll=%llu\n",
                           (unsigned long long)a0, (unsigned long long)p.audio_poll);
    return 1;
}
HLE(s_avp_stop) {   // s32 sceAvPlayerStop(handle)
    svc_log("sceAvPlayerStop", a0,a1,a2,a3,a4,a5);
    void* obj = nullptr; AvpEventCb cb = nullptr; bool fire_stop = false;
    {
        std::lock_guard<std::mutex> lk(g_avp_mx);
        auto it = g_avp.find(a0); if (it == g_avp.end()) return 0;
        AvpPlayer& p = it->second;
        if (p.playing && !p.stop_fired) {
            p.playing = false; p.paused = false; p.stop_fired = true;
            fire_stop = true; obj = p.ev_obj; cb = p.ev_cb;
        }
    }
    if (fire_stop) avp_fire(obj, cb, AVP_STOP);
    return 0;
}
HLE(s_avp_close) {   // s32 sceAvPlayerClose(handle)
    svc_log("sceAvPlayerClose", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_avp_mx);
    auto it = g_avp.find(a0);
    if (it != g_avp.end()) {
        if (auto* b = prosper::video::backend(); b && it->second.backend_id >= 0) b->close(it->second.backend_id);
        g_avp.erase(it);
    }
    return 0;
}

#ifndef _WIN32
// Pass the HLE-entry stack pointer to callback-producing AvPlayer handlers.  On Linux it identifies
// the guest TCB saved by the import stub; on macOS callback_guest_fs_from_entry_stack() fails closed
// to zero because guest TLS uses trap emulation rather than a hardware %fs switch.
#define AVP_CALLBACK_ENTRY(entry, target, handler) \
    extern "C" uint64_t target(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t); \
    PROSPER_ASM_TRAMPOLINE(entry, target) \
    extern "C" void entry(); \
    extern "C" uint64_t target(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, \
                                uint64_t a4, uint64_t a5, uint64_t entry_rsp) { \
        const uint64_t guest_fs = callback_guest_fs_from_entry_stack(entry_rsp); \
        if (avp_log() && guest_fs) { \
            const uint64_t guest_ra = *(const uint64_t*)(uintptr_t)(entry_rsp + 0x30); \
            fprintf(stderr, "[avp] call %s guest_ra=0x%llx guest_fs=1\n", #handler, \
                    (unsigned long long)guest_ra); \
        } \
        (void)guest_fs; \
        /* Only Linux swaps hardware %fs at the import boundary. */ \
        PROSPER_AVP_ENTRY_SCOPE(guest_fs) \
        return handler(a0, a1, a2, a3, a4, a5); \
    }

#if defined(__linux__)
#define PROSPER_AVP_ENTRY_SCOPE(guest_fs) AvpEntryGuestFsScope avp_entry_guest_fs_scope(guest_fs);
#else
#define PROSPER_AVP_ENTRY_SCOPE(guest_fs)
#endif

AVP_CALLBACK_ENTRY(s_avp_addsource_entry, s_avp_addsource_entry_c, s_avp_addsource)
AVP_CALLBACK_ENTRY(s_avp_addsourceex_entry, s_avp_addsourceex_entry_c, s_avp_addsourceex)
AVP_CALLBACK_ENTRY(s_avp_start_entry, s_avp_start_entry_c, s_avp_start)
AVP_CALLBACK_ENTRY(s_avplayer_isactive_entry, s_avplayer_isactive_entry_c, s_avplayer_isactive)
AVP_CALLBACK_ENTRY(s_avp_getvideodata_entry, s_avp_getvideodata_entry_c, s_avp_getvideodata)
AVP_CALLBACK_ENTRY(s_avp_getvideodataex_entry, s_avp_getvideodataex_entry_c, s_avp_getvideodataex)
AVP_CALLBACK_ENTRY(s_avp_stop_entry, s_avp_stop_entry_c, s_avp_stop)
AVP_CALLBACK_ENTRY(s_avp_pause_entry, s_avp_pause_entry_c, s_avp_pause)
AVP_CALLBACK_ENTRY(s_avp_resume_entry, s_avp_resume_entry_c, s_avp_resume)

#undef PROSPER_AVP_ENTRY_SCOPE
#undef AVP_CALLBACK_ENTRY
#endif
// sceUserServiceGetGamePresets(userId, presets): MUST return success (0). The Unity engine's
// per-controller connection check (eboot 0x14707e0, reached from the pad "reset" path 0x1470ca0)
// calls this and treats ANY non-zero user-service return as "controller invalid" — it then clears the
// pad's connected flag (eboot+0x201d150) EVERY FRAME, so scePadGetLoginUserIdList-based enumeration
// reports 0 controllers and the game never calls scePadOpen. i.e. this one wrong errno silently killed
// ALL gamepad input in The Messenger (#234). The prior 0x80960006 was chosen to avoid the game reading
// an untouched (garbage) out-struct; instead return 0 AND zero the payload: the out-struct's first field
// is its byte size (the caller sets it, e.g. 0x30) — zero the bytes after it (bounded) for clean default
// presets. CONFIDENCE: HIGH — root-caused via HWBP/HWWATCH on the eboot connection flag; verified fix
// makes scePadOpen + scePadReadState fire and input register.
HLE(s_gamepresets) {
    if (a1) { uint32_t sz = *(uint32_t*)PW(a1); if (sz > 8 && sz <= 0x400) memset((char*)PW(a1) + 8, 0, sz - 8); }
    return 0;
}

// --- NP / online: an honest OFFLINE, SIGNED-OUT console (#306). --------------------------------
// The DOLL front-end boot flow stalls at UE4's InstallBundleManager PatchCheck because the Np
// sign-in queries returned success-with-garbage-out: "success" from sceNpGetOnlineId told the
// game a user IS signed in, pushing its patch/entitlement check onto online branches that then
// wait forever on fake Http/WebApi handles (docs/DOLL_LOADING_PROGRESSION.md §3). A real console
// with no PSN sign-in answers these with SCE_NP_ERROR_SIGNED_OUT so the flow resolves to its
// offline path (UE4 PatchCheck -> NoLoggedInUser).
// Error space verified against shadPS4 np_error.h (PS4-inherited; identical export names+NIDs in
// the PS5 3.20 libSceNpManager stub table). SIGNED_OUT = 0x80550006. CONFIDENCE: HIGH on the PS4
// semantic (shadPS4 returns exactly this from sceNpGetOnlineId/sceNpGetAccountIdA when no user is
// signed in), MED that the PS5 errno value is unchanged (same 0x8055 Np facility, same API).
static constexpr uint64_t NP_ERR_SIGNED_OUT = 0x80550006ull;   // SCE_NP_ERROR_SIGNED_OUT
HLE(s_np_state)       { if (a1) *(int32_t*)PW(a1) = 1; return 0; }           // SCE_NP_STATE_SIGNED_OUT
HLE(s_np_reach)       { if (a1) *(int32_t*)PW(a1) = 0; return 0; }
// sceNpGetAccountIdA(userId, u64* accountId): signed-out consoles zero the id AND return the
// signed-out error (shadPS4 np_manager.cpp:579 does exactly this). The previous success+0 was
// contradictory garbage ("you have a user; their account id is 0").
HLE(s_np_accountid)   { if (a1) *(uint64_t*)PW(a1) = 0; return NP_ERR_SIGNED_OUT; }
// sceNpGetAccountCountryA: signed-out, out-struct untouched (matching the sibling getters). Was returning
// SUCCESS with a zeroed country code -> the guest reads a blank as a valid region key and takes an
// age/region/store path (the #306 success-masks-offline wedge) instead of the clean offline branch.
HLE(s_np_country)     { return NP_ERR_SIGNED_OUT; }
// sceNpGetOnlineId(userId, SceNpOnlineId* out): the signed-out error, out untouched (shadPS4
// np_manager.cpp:618). The unimplemented success+garbage here is what faked the sign-in.
HLE(s_np_getonlineid) { svc_log("sceNpGetOnlineId", a0,a1,a2,a3,a4,a5); return NP_ERR_SIGNED_OUT; }
// sceNpGetNpId(userId, SceNpId* out): the signed-out error, out untouched (shadPS4 np_manager.cpp) — the
// most common identity getter. Was MISSING -> the return-0 stub told the guest a valid online identity
// existed (a garbage ~40-byte SceNpId), pushing it onto online/entitlement branches that dead-end. This
// was a direct hole in the #306 signed-out fix (the sibling getters below were done, this one wasn't).
HLE(s_np_getnpid)     { svc_log("sceNpGetNpId", a0,a1,a2,a3,a4,a5); return NP_ERR_SIGNED_OUT; }
// sceNpCheckNpAvailability / ...A / CheckNpReachability: an honest signed-out console. Were MISSING -> the
// return-0 stub answered "PSN is available", pushing the guest onto online branches that then wait forever
// (the #306 wedge class). NOTE: the async poll/request pairing (sceNpPollAsync / CreateAsyncRequest) is
// deliberately left for a follow-up -- its exact completion flow needs a live PROSPER_SVCLOG capture.
HLE(s_np_check_avail) { svc_log("sceNpCheckNpAvailability", a0,a1,a2,a3,a4,a5); return NP_ERR_SIGNED_OUT; }

// --- mouse (report a device that exists but has no input; pad -> hle_pad.cpp real backend) ---
HLE(s_open)           { return g_handle++; }                                 // sceMouseOpen -> handle
// sceMouseRead(handle, SceMouseData*, num) returns the number of mouse events read. SceMouseData
// (~0x18 bytes) is NOT ScePadData — sharing the pad stub returned one "valid" entry whose memset
// overran a single-entry mouse buffer, and the game consumed a phantom mouse event every call. No
// mouse attached: zero one entry defensively, report 0 events.
HLE(s_mouse_read)     { if (a1) memset(PW(a1), 0, 0x18); return 0; }

// --- libSceIme keyboard API (#186) ---
// PPSA02664 polls this every frame in its input loop. We have no physical PS5 keyboard, so we report a
// consistent "no keyboard connected" state: Open/Update succeed, GetResourceId returns an empty
// resource array (no keyboards), GetInfo reports a disconnected device — the game then uses controller
// / on-screen input instead of waiting on a keyboard. Struct layouts + field offsets verified against
// shadPS4 src/core/libraries/ime/ime_common.h; error codes from ime_error.h. Note sceImeKeyboardOpen
// returns an Error (0 = SCE_OK), NOT a handle — so success is 0. CONFIDENCE: HIGH on the layouts; MED
// on the device/status enum "disconnected" == 0.
//   OrbisImeKeyboardResourceIdArray: user_id@0(s32) resource_id[5]@4(u32)                     size 24
//   OrbisImeKeyboardInfo: user_id@0 device@4 type@8 repeat_delay@12 repeat_rate@16 status@20 rsv[12]@24  size 36
HLE(s_ime_kbd_open) { svc_log("sceImeKeyboardOpen", a0,a1,a2,a3,a4,a5); return 0; }  // register handler; no events to deliver
HLE(s_ime_update)   { svc_log("sceImeUpdate", a0,a1,a2,a3,a4,a5); return 0; }        // pump the queue: no events
// sceImeKeyboardGetResourceId(userId, OrbisImeKeyboardResourceIdArray* out): report the user's keyboard
// resource ids. Headless -> none (all-zero). A registered PlatformUi (a windowed frontend with a host
// keyboard) reports its own ids, so a title enables keyboard input (#347).
HLE(s_ime_kbd_resid) {
    svc_log("sceImeKeyboardGetResourceId", a0,a1,a2,a3,a4,a5);
    if (!a1) return 0x80BC0031ull;   // ORBIS_IME_ERROR_INVALID_ADDRESS
    uint8_t* p = (uint8_t*)PW(a1);
    *(int32_t*)(p + 0) = (int32_t)a0;                         // user_id echoes the caller
    uint32_t ids[5] = {0, 0, 0, 0, 0};
    int n = 0;
    if (auto* ui = platform_ui()) n = ui->keyboardResourceIds((int32_t)a0, ids, 5);
    if (n < 0) n = 0; if (n > 5) n = 5;
    for (int i = 0; i < 5; i++) *(uint32_t*)(p + 4 + i * 4) = (i < n) ? ids[i] : 0;
    return 0;
}
// sceImeKeyboardGetInfo(resourceId, OrbisImeKeyboardInfo* info): connected device info when a frontend
// reports a keyboard, else a disconnected (no-device) info.
HLE(s_ime_kbd_info) {
    svc_log("sceImeKeyboardGetInfo", a0,a1,a2,a3,a4,a5);
    if (!a1) return 0x80BC0031ull;   // ORBIS_IME_ERROR_INVALID_ADDRESS
    memset(PW(a1), 0, 36);           // default: device=0 type=0 repeat=0 status=0(disconnected)
    uint8_t* p = (uint8_t*)PW(a1);
    *(int32_t*)(p + 0) = 1;          // user_id = default user (matches sceUserServiceGetInitialUser)
    bool connected = false;
    if (auto* ui = platform_ui()) { uint32_t ids[5]; connected = ui->keyboardResourceIds(1, ids, 5) > 0; }
    if (connected) {
        *(uint32_t*)(p + 4)  = 1;    // device  = 1 (a keyboard is present)
        *(uint32_t*)(p + 20) = 1;    // status  = 1 (connected)
    }
    return 0;
}

// --- app content ---
// sceAppContentAppParamGetInt(paramId, int32_t* value): paramId 0 = SKU_FLAG, whose only valid values are
// TRIAL=1 / FULL=3 (there is NO 0). Writing 0 for it made the SKU check indeterminate -> a game testing
// `sku == FULL` drops into trial/demo mode (locked content, "buy full game" gating). Report FULL for a
// legally-owned title; other params (USER_DEFINED_PARAM_1..4) stay 0 absent a param.sfo read.
HLE(s_appcontent_int) { if (a1) *(int32_t*)PW(a1) = ((int32_t)a0 == 0) ? 3 : 0; return 0; }

// sceSystemServiceParamGetInt(SceSystemServiceParamId paramId, int32_t* value): a0=paramId, a1=value.
// The system-settings a blanket-zero stub gives are mostly harmless, EXCEPT the LANGUAGE (paramId 1):
// value 0 = SCE_SYSTEM_PARAM_LANG_JAPANESE, so games localise their UI/text to Japanese. Default to
// US English (SCE_SYSTEM_PARAM_LANG_ENGLISH_US = 1) instead. Configurable via PROSPER_SYS_LANG, which
// takes the Sony SCE_SYSTEM_PARAM_LANG_* enum (0=ja, 1=en-US, 2=fr, 4=de, 5=it, 9=ko, 18=en-GB, …).
// Date/time-format params (2/3) default to the US convention (0 = MM/DD/YYYY, 12-hour).
HLE(s_syss_param_int) {
    int32_t paramId = (int32_t)a0;           // a0 = paramId, a1 = int32_t* value out (matches s_appcontent_int)
    int32_t val = 0;
    if (paramId == 1) {                      // SCE_SYSTEM_SERVICE_PARAM_ID_LANG
        val = 1;                             // SCE_SYSTEM_PARAM_LANG_ENGLISH_US
        if (const char* e = getenv("PROSPER_SYS_LANG")) val = (int32_t)strtol(e, nullptr, 0);
    } else if (paramId == 1000) {            // SCE_SYSTEM_SERVICE_PARAM_ID_ENTER_BUTTON_ASSIGN
        // Cross = confirm (Western default). Previously fell through to val=0 = Circle, which inverts
        // ✕/○ confirm/cancel and on-screen button prompts relative to the en-US locale we present.
        val = 1;                             // Cross (shadPS4 default; Circle=0)
        if (const char* e = getenv("PROSPER_ENTER_BUTTON")) val = (int32_t)strtol(e, nullptr, 0);
    }
    if (a1) *(int32_t*)PW(a1) = val;
    return 0;
}

// sceSystemServiceGetStatus(SceSystemServiceStatus* status) — the out-struct is ARG 0 (single-arg
// call). This was aliased to s_appcontent_int, which returned success while writing 4 bytes through
// a1 — whatever stale value the caller left in RSI — and left the real status struct uninitialized.
// Layout cross-checked against Kyty LibSystemService.cpp:83: int32 eventNum @0; bool
// isSystemUiOverlaid @4, isInBackgroundExecution @5, isCpuMode7CpuNormal @6 (defaults TRUE),
// isGameLiveStreamingOnAir @7, isOutOfVrPlayArea @8; 12 bytes with tail padding.
HLE(s_syss_getstatus) {
    auto* st = (uint8_t*)PW(a0);
    if (!st) return 0x80A10003ull;   // SYSTEM_SERVICE_ERROR_PARAMETER (Kyty Errno.h:382)
    memset(st, 0, 12);
    st[6] = 1;                       // isCpuMode7CpuNormal = true
    return 0;
}

// sceSystemServiceGetDisplaySafeAreaInfo(SceSystemServiceDisplaySafeAreaInfo* info) — single-arg
// call, out-struct is ARG 0: { float ratio; uint8_t reserved[128]; } (shadPS4 systemservice.h).
// The default unimplemented stub returned 0 (SUCCESS) but left `ratio` uninitialized (typically 0.0
// from a fresh guest heap block). A safe-area ratio of 0 makes UE4's PS5 viewport code compute a
// DEGENERATE (zero-area) title-safe rect: the game scales its render/UI viewport by the ratio, and a
// zero ratio collapses the visible region — nothing to rasterize, so the RHI submits setup/compute
// but never geometry. Real hardware always reports ratio=1.0 for a display with overscan disabled
// (the modern default); shadPS4 hard-codes 1.0f. Fill ratio=1.0 and zero the reserved tail.
// CONFIDENCE: MED — struct + 1.0f contract confirmed against shadPS4; that a 0.0 ratio is what
// gates DOLL's scene draws is a hypothesis under test, but returning success with an unfilled
// out-struct is a bug regardless (same class as GetStatus / ParamGetString above).
HLE(s_syss_safearea) {
    auto* info = (uint8_t*)PW(a0);
    if (!info) return 0x80A10003ull;   // SYSTEM_SERVICE_ERROR_PARAMETER
    memset(info, 0, 0x84);             // sizeof {float + uint8_t[128]} = 132
    *(float*)info = 1.0f;              // ratio = full display, no overscan inset
    return 0;
}

// sceAppContentTemporaryDataMount2(option, SceAppContentMountPoint* mp) — mount the app's temp-data
// area and write its guest path into mp. SceAppContentMountPoint = char data[16] (shadPS4
// app_content.h; PS4/PS5 identical). shadPS4 writes exactly "/temp0\0" and returns 0. Our previous
// behavior (unimplemented stub -> return 0 = SUCCESS with mp untouched) made the game treat 16 bytes
// of uninitialized memory as its temp-data mount path and build file paths from it. We write EXACTLY
// 7 bytes ("/temp0" + NUL) — never the full 16, never more (cf. the f_fstat oversized-write lesson).
// hle_file.cpp translates the /temp0 prefix to a host-backed directory so subsequent I/O works.
HLE(s_appcontent_tmpmount2) {
    char* mp = (char*)PW(a1);
    if (!mp) return 0x80D90002ull;              // SCE_APP_CONTENT_ERROR_PARAMETER
    memcpy(mp, "/temp0", 7);
    return 0;
}
// sceAppContentTemporaryDataGetAvailableSpaceKb(mp, uint64_t* kb): temp0 is a 1 GiB scratch area on
// PS5; report it all free (shadPS4 does the same). The stub's return-0-only left *kb garbage — a
// value games use to size caches/allocations.
HLE(s_appcontent_tmpspace) { if (a1) *(uint64_t*)PW(a1) = 1048576ull; return 0; }

// sceAppContentGetAddcontInfoList(serviceLabel, Info* list, u32 listNum, u32* hitNum): the add-content
// (DLC) enumeration. The game first count-queries with list=NULL/num=0 and an out pointer in a3, then
// grows a hitNum-sized array. Was MISSING -> the return-0 stub reported success while leaving *hitNum
// (a3) uninitialized -> the engine sized an array from heap garbage = the 34 GB OOM / heap-overflow
// class (cf. #213). This dump has no DLC, so the truthful answer is SUCCESS with hitNum=0 (matches the
// sibling s_npent_addcont_list). CONFIDENCE: MED (no-DLC hitNum=0 is the retail answer; import inferred).
HLE(s_appcontent_addcont_list) {
    if (!svc_ptrish(a3)) return 0x80D90002ull;   // APP_CONTENT_ERROR_PARAMETER
    *(uint32_t*)PW(a3) = 0;                       // hitNum = 0: no add-content installed
    return 0;
}
// GetAddcontInfo / GetEntitlementKey / AddcontMount for an unknown label: no entitlement (offline, no DLC).
HLE(s_appcontent_no_entitlement) { return 0x80D90007ull; }   // APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT

// sceSystemServiceParamGetString(paramId, char* buf, size_t bufSize): fetch a system string parameter
// (e.g. the console/user nickname). The default unimplemented stub returned 0 (SUCCESS) but never wrote
// the buffer, so the game read whatever uninitialized bytes were there as a "valid" string and derefed
// into it (null-deref crash in managed code during scene load). Match this file's policy: write a valid
// NUL-terminated string and report success. CONFIDENCE: MED — signature (paramId, buf, size) is the
// documented Sony ABI; an empty string is a safe, defined default when we have no real system value.
HLE(s_param_string)   { if (a1 && a2) ((char*)PW(a1))[0] = '\0'; return 0; }

// Message-dialog LIFECYCLE (#144). Status enum: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3.
// The old handler returned FINISHED(3) UNCONDITIONALLY — including before any Open, where the real
// API reports NONE/INITIALIZED — so a guest polling GetStatus as a guard saw "dialog already done"
// at the wrong stage and skipped/duplicated its dialog logic. We track the real transitions:
// Initialize -> INITIALIZED, Open -> auto-dismiss to FINISHED (headless: no interactive UI, so the
// game's "wait until dismissed" loop still exits immediately), Close/Terminate -> back to NONE.
// GetResult -> zeroed struct = OK/no button pressed.
// A registered PlatformUi gets first refusal at Open; if it takes the dialog, status/result/close
// route there (real message box). Otherwise the core auto-dismisses headlessly. `g_msgdialog_backed`
// records which path Open chose. See platform_ui.hpp (#347).
namespace { std::atomic<int> g_msgdialog_status{0 /*NONE*/}; std::atomic<int> g_msgdialog_backed{0}; }
HLE(s_dialog_initialize) { g_msgdialog_backed.store(0); g_msgdialog_status.store(1 /*INITIALIZED*/); return 0; }
HLE(s_dialog_open) {
    if (auto* ui = platform_ui(); ui && ui->msgDialogOpen(a0)) { g_msgdialog_backed.store(1); return 0; }
    g_msgdialog_backed.store(0);
    g_msgdialog_status.store(3 /*FINISHED (auto-dismiss)*/);
    return 0;
}
HLE(s_dialog_close)      { if (g_msgdialog_backed.exchange(0)) { if (auto* ui = platform_ui()) ui->msgDialogClose(); } g_msgdialog_status.store(0 /*NONE*/); return 0; }
HLE(s_dialog_terminate)  { if (g_msgdialog_backed.exchange(0)) { if (auto* ui = platform_ui()) ui->msgDialogClose(); } g_msgdialog_status.store(0 /*NONE*/); return 0; }
HLE(s_dialog_status) {
    if (g_msgdialog_backed.load()) { if (auto* ui = platform_ui()) return (uint64_t)(unsigned)ui->msgDialogStatus(); }
    return (uint64_t)(unsigned)g_msgdialog_status.load();
}
// SceMsgDialogResult = { u32 mode; u32 result; u32 buttonId; char reserved[32] } = 0x2C bytes
// (shadPS4 msgdialog_ui.h DialogResult). Was memset(0x30) — 4 bytes PAST the caller's struct,
// exactly the f_fstat oversized-write class this file warns about.
HLE(s_dialog_result) {
    if (g_msgdialog_backed.load()) { if (auto* ui = platform_ui()) { (void)ui->msgDialogResult(a0); return 0; } }
    if (a0) memset(PW(a0), 0, 0x2C);
    return 0;
}

// --- libSceSaveDataDialog ---------------------------------------------------------------
// Uses the common-dialog status enum: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3. A generic
// success stub is not safe for UpdateStatus: Dead Cells polled it 1,135 times in 16 seconds because
// zero means NONE, leaving its level transition in a permanent wait (#768). The headless policy
// matches MsgDialog above: Open auto-dismisses to FINISHED so the guest can consume a neutral result.
//
// PS4/PS5 SDK layout used below (also used by shadPS4): param.mode is u32 at +0x34 and param.userData
// is a pointer at +0x70. Result begins with mode/result/buttonId/pad followed by three pointers. Write
// only that proven 0x28-byte prefix; the remaining 32 reserved bytes belong to the caller.
namespace {
    std::atomic<int> g_savedatadialog_status{0 /*NONE*/};
    std::atomic<uint32_t> g_savedatadialog_mode{0 /*INVALID*/};
    std::atomic<uint64_t> g_savedatadialog_user_data{0};
}
HLE(s_savedlg_initialize) {
    g_savedatadialog_mode.store(0);
    g_savedatadialog_user_data.store(0);
    g_savedatadialog_status.store(1 /*INITIALIZED*/);
    return 0;
}
HLE(s_savedlg_open) {
    if (a0) {
        const uint8_t* param = (const uint8_t*)PW(a0);
        g_savedatadialog_mode.store(*(const uint32_t*)(param + 0x34));
        g_savedatadialog_user_data.store(*(const uint64_t*)(param + 0x70));
    } else {
        g_savedatadialog_mode.store(0);
        g_savedatadialog_user_data.store(0);
    }
    g_savedatadialog_status.store(3 /*FINISHED (auto-dismiss)*/);
    return 0;
}
HLE(s_savedlg_status) { return (uint64_t)(unsigned)g_savedatadialog_status.load(); }
HLE(s_savedlg_result) {
    if (a0) {
        uint8_t* result = (uint8_t*)PW(a0);
        *(uint32_t*)(result + 0x00) = g_savedatadialog_mode.load();
        *(uint32_t*)(result + 0x04) = 0 /*CommonDialogResult::OK*/;
        *(uint32_t*)(result + 0x08) = 0 /*ButtonId::INVALID*/;
        *(uint32_t*)(result + 0x0c) = 0;
        *(uint64_t*)(result + 0x10) = 0 /*dirName*/;
        *(uint64_t*)(result + 0x18) = 0 /*save param*/;
        *(uint64_t*)(result + 0x20) = g_savedatadialog_user_data.load();
    }
    return 0;
}
HLE(s_savedlg_close) { g_savedatadialog_status.store(0 /*NONE*/); return 0; }
HLE(s_savedlg_terminate) {
    g_savedatadialog_status.store(0 /*NONE*/);
    g_savedatadialog_mode.store(0);
    g_savedatadialog_user_data.store(0);
    return 0;
}
HLE(s_savedlg_ready) { return 1; }
HLE(s_savedlg_progress) { return 0; }

// --- libSceImeDialog (on-screen text-entry dialog) (#191). We have no keyboard UI, so the dialog
// auto-completes: Init -> FINISHED immediately, so the game's "poll GetStatus until Finished" loop
// exits at once instead of hanging on a dialog that never appears; GetResult reports endStatus =
// OK/ENTER with the (unchanged/empty) input buffer; Term/Abort return to NONE.
// IMPORTANT: sceImeDialogGetStatus returns the IME-dialog's OWN enum (OrbisImeDialogStatus: NONE=0,
// RUNNING=1, FINISHED=2) — NOT the 4-value SceCommonDialogStatus that MsgDialog/ErrorDialog use
// (…RUNNING=2, FINISHED=3). Verified in shadPS4 ime_dialog.cpp. Returning 3 here made a game's poll
// loop never see Finished(2). We write only the 4-byte endStatus at GetResult offset 0 (the field the
// game branches on), never more, so a wrong tail-field guess can't corrupt the caller's struct.
// A registered PlatformUi (the app frontend) gets first refusal on the dialog: if it takes it
// (imeDialogOpen -> true), status/result/close route there so a real text field is shown; otherwise
// the core auto-completes headlessly (below). `g_imedialog_backed` records which path Init chose so a
// later poll/result/term goes to the same place. See platform_ui.hpp (#347).
namespace { std::atomic<int> g_imedialog_status{0 /*NONE*/}; std::atomic<int> g_imedialog_backed{0}; }
HLE(s_imedlg_init) {
    if (auto* ui = platform_ui(); ui && ui->imeDialogOpen(a0, a1)) { g_imedialog_backed.store(1); return 0; }
    g_imedialog_backed.store(0);
    g_imedialog_status.store(2 /*OrbisImeDialogStatus::Finished — auto-complete, no keyboard UI*/);
    return 0;
}
HLE(s_imedlg_status) {
    if (g_imedialog_backed.load()) { if (auto* ui = platform_ui()) return (uint64_t)(unsigned)ui->imeDialogStatus(); }
    return (uint64_t)(unsigned)g_imedialog_status.load();
}
HLE(s_imedlg_result) {
    if (g_imedialog_backed.load()) { if (auto* ui = platform_ui()) { (void)ui->imeDialogResult(a0); return 0; } }
    if (a0) *(int32_t*)PW(a0) = 0 /*SCE_IME_DIALOG_END_STATUS_OK*/;
    return 0;
}
HLE(s_imedlg_term)  { if (g_imedialog_backed.exchange(0)) { if (auto* ui = platform_ui()) ui->imeDialogClose(); } g_imedialog_status.store(0 /*NONE*/); return 0; }
HLE(s_imedlg_abort) { if (g_imedialog_backed.exchange(0)) { if (auto* ui = platform_ui()) ui->imeDialogClose(); } g_imedialog_status.store(0 /*NONE*/); return 0; }

// --- libSceNpTrophy2 (PS5 trophy system) — the DOLL 34.6 GB OOM (issue #213 diagnosis). ---------
// The guest's trophy bring-up (eboot+0xdbcb43..0xdbcc2e, gdb-captured live) calls
// sceNpTrophy2GetGameInfo(ctx, handle, out*, 0) — NID 4IzqhhUQ3nk named via nid_hash brute force —
// then grows TWO arrays from the out-struct's counts (u32 at out+0x4: 32-byte entries; a second
// 0x520-byte-entry array) and calls a sibling (y3zHpdZO6ME, unnamed) to fill them. The generic
// unimplemented stub returned 0 = SUCCESS with the out-struct UNWRITTEN, so the engine consumed
// heap garbage as a trophy count: gdb-captured count 0x408bd000 -> a 34,644,492,288-byte TArray
// grow ("Ran out of memory allocating 34644492288 bytes") — and when the garbage happened to be
// allocatable-huge instead, the minutes-long zero-fill starved the RenderThread until UE's
// "GameThread timed out waiting for RenderThread after 120.00 secs" watchdog killed the boot.
// Without a trophy backend the honest answer is FAILURE: a negative return takes the caller's
// clean invalid path (eboot+0xdbd239/0xdbd242: mark the trophy config unavailable, continue) —
// exactly the state a real console reports with no signed-in user. Only the SIGN of the return is
// consumed by this caller; the exact NpTrophy2 error space is unverified (no Kyty/shadPS4/stub
// reference), so the value is chosen inside the documented SCE_NP_TROPHY (0x8055xxxx) range.
// CONFIDENCE: HIGH that failure beats success+garbage; LOW on the specific error constant.
HLE(s_nptrophy2_unavailable) { return 0x80551500ull; }

// ===== Issue #232: the Sony services DOLL's level-load flow polls (PlayGo / SaveData / =========
// ===== NpTrophy2 lifecycle / Share). All NID<->name pairs verified against the PS5 3.20 ========
// ===== library stub tables (PS5-3.20_Libs/libSce{PlayGo,SaveData.native,NpTrophy2,Share}.c). ===
//
// DOLL's game-side workers (DollLevelPreloader / SaveLoadUpdate / DLCDataUpdate / ShareUpdate)
// gate the per-frame boot-flow state machine at eboot+0x5044740 on these services answering.
// Bare unimpl->0 stubs returned SUCCESS with every out-param unfilled (the recurring
// success+garbage-out bug class), so e.g. scePlayGoOpen "succeeded" without ever writing the
// handle the game then queries loci with.

// --- libScePlayGo: report ALL content installed and locus-local. --------------------------------
// PS4-inherited API (identical exported names on PS5 3.20); shapes cross-checked against shadPS4
// playgo.cpp + playgo_types.h and Kyty. A disc/fully-installed title is exactly this state on real
// hardware, so "everything present" is the truthful answer for our complete dump.
// Error space 0x80B2000x (shadPS4 playgo_types.h). CONFIDENCE: HIGH (two agreeing PS4 references,
// PS4-inherited surface).
static constexpr uint64_t PLAYGO_ERR_BAD_POINTER = 0x80B2000Aull;
static constexpr uint64_t PLAYGO_ERR_BAD_SIZE    = 0x80B2000Bull;
HLE(s_playgo_init)  { svc_log("scePlayGoInitialize", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_playgo_term)  { return 0; }
// scePlayGoOpen(u32* outHandle, const void* param): the handle the whole API is keyed on. The
// unimpl stub's success-with-unfilled-handle left the game querying loci with garbage.
HLE(s_playgo_open)  { svc_log("scePlayGoOpen", a0,a1,a2,a3,a4,a5);
                      if (!a0) return PLAYGO_ERR_BAD_POINTER;
                      *(uint32_t*)PW(a0) = 1; return 0; }
HLE(s_playgo_close) { return 0; }
// scePlayGoGetLocus(h, const u16* chunkIds, u32 n, s8* outLoci): every chunk is LOCAL_FAST (3).
HLE(s_playgo_getlocus) { svc_log("scePlayGoGetLocus", a0,a1,a2,a3,a4,a5, 2);
                         if (!a1 || !a3) return PLAYGO_ERR_BAD_POINTER;
                         if (!(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                         memset(PW(a3), 3 /*SCE_PLAYGO_LOCUS_LOCAL_FAST*/, (uint32_t)a2); return 0; }
// scePlayGoGetProgress(h, chunkIds, n, OrbisPlayGoProgress* out): one struct {u64 progressSize;
// u64 totalSize} summed over the queried chunks; fully-installed == progressSize==totalSize!=0.
HLE(s_playgo_getprogress) { svc_log("scePlayGoGetProgress", a0,a1,a2,a3,a4,a5, 2);
                            if (!a1 || !a3) return PLAYGO_ERR_BAD_POINTER;
                            if (!(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                            uint64_t* p = (uint64_t*)PW(a3);
                            p[0] = p[1] = (uint64_t)(uint32_t)a2 << 20;  // 1 MiB/chunk, done==total
                            return 0; }
// scePlayGoGetToDoList(h, OrbisPlayGoToDo* list, u32 n, u32* outEntries): nothing left to install.
HLE(s_playgo_gettodo) { if (!a3) return PLAYGO_ERR_BAD_POINTER; *(uint32_t*)PW(a3) = 0; return 0; }
HLE(s_playgo_settodo) { return 0; }
// scePlayGoGetChunkId(h, u16* list, u32 n, u32* outEntries): one chunk, id 0 (the minimal truthful
// shape for a fully-local title; we don't parse playgo-chunk.dat).
HLE(s_playgo_getchunkid) { if (!a3) return PLAYGO_ERR_BAD_POINTER;
                           if (a1 && !(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                           if (a1) { *(uint16_t*)PW(a1) = 0; *(uint32_t*)PW(a3) = 1; }
                           else *(uint32_t*)PW(a3) = 1;
                           return 0; }
// scePlayGoGetEta(h, chunkIds, n, s64* outEta): everything installed -> 0 seconds.
HLE(s_playgo_geteta) { if (!a1 || !a3) return PLAYGO_ERR_BAD_POINTER;
                       if (!(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                       *(int64_t*)PW(a3) = 0; return 0; }
// scePlayGoGetInstallSpeed(h, s32* out): FULL (2) — nothing throttled.
HLE(s_playgo_getspeed) { if (!a1) return PLAYGO_ERR_BAD_POINTER; *(int32_t*)PW(a1) = 2; return 0; }
// scePlayGoGetLanguageMask(h, u64* out): all languages present. CONFIDENCE: MED (mask semantics
// are per-language bits; all-ones = every language's chunks installed).
HLE(s_playgo_getlang) { if (!a1) return PLAYGO_ERR_BAD_POINTER;
                        *(uint64_t*)PW(a1) = ~0ull; return 0; }

// --- libSceSaveData (PS5 "native" surface): report a clean FRESH console — no existing save. ----
// DOLL calls Initialize3 -> CreateTransactionResource -> Mount3 -> Umount2 -> Prepare -> Commit
// (live-captured first-seen order). Initialize3 is PS4-inherited (Kyty returns OK). Mount3 /
// Prepare / Commit / CreateTransactionResource are PS5-only (present ONLY in the PS5 3.20
// libSceSaveData native stub; no Kyty/shadPS4 implementation exists), so their exact structs are
// unreferenced. Policy: a mount of a save that does not exist returns NOT_FOUND (0x809F0008 —
// shadPS4 savedata_error.h; same 0x809F error facility on PS5) and writes NOTHING — the truthful
// first-boot state on real hardware, which a shipped game must handle by proceeding to a fresh
// game. This is strictly better than the previous unimpl->0 "mount succeeded" with a garbage
// mount-result the game then reads paths from. Transaction bookkeeping calls succeed (they
// allocate/tear down local resources only). CONFIDENCE: HIGH on Initialize3/NOT_FOUND semantics;
// MED on Mount3's arg order (mount-desc in, result out — matches every PS4 Mount variant);
// LOW on Prepare/Commit internals (no-op success; PROSPER_SVCLOG captures their real args).
static constexpr uint64_t SAVE_DATA_ERR_PARAMETER = 0x809F0000ull;
static constexpr uint64_t SAVE_DATA_ERR_NOT_FOUND = 0x809F0008ull;
static constexpr uint64_t SAVE_DATA_ERR_NO_EVENT = 0x809F0018ull;
namespace { std::atomic<unsigned> g_savedata_umount_events{0}; }
HLE(s_savedata_init3)   { svc_log("sceSaveDataInitialize3", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_savedata_term)    { return 0; }
HLE(s_savedata_txres)   { svc_log("sceSaveDataCreateTransactionResource", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_savedata_txres_del) { return 0; }

// --- libSceSaveData "save-data memory" API (#191). A per-(user,slot) fixed-size memory block the
// managed SaveData layer reads/writes by offset and syncs to storage; on PS5 this is how a title
// keeps a small always-resident save (settings/progress). We back each slot with a host-memory
// block: Setup allocates it, Set copies guest->block, Get copies block->guest, Sync commits. Struct
// layouts + field offsets verified against shadPS4 save_data/savedata.cpp; error codes from
// savedata_error.h. CONFIDENCE: HIGH on the happy-path round-trip (the contract a title depends on).
//   OrbisSaveDataMemoryData:  buf@0(ptr) bufSize@8(u64) offset@16(s64)               size 64
//   OrbisSaveDataMemorySetup2: option@0 userId@4 memorySize@8 iconMemorySize@16
//                              initParam@24 initIcon@32 slotId@40                    size 64
//   OrbisSaveDataMemorySet2:  userId@0 [pad@4] data@8(ptr) param@16 icon@24 dataNum@32 slotId@36
//   OrbisSaveDataMemoryGet2:  userId@0 [pad@4] data@8(ptr) param@16 icon@24 slotId@32
namespace {
    std::mutex g_savemem_mx;
    std::unordered_map<uint64_t, std::vector<uint8_t>> g_savemem;   // key = (userId<<32)|slotId
    uint64_t savemem_key(int32_t userId, uint32_t slotId) {
        return ((uint64_t)(uint32_t)userId << 32) | slotId;
    }
    // Host file backing one SaveDataMemory slot, so a save survives a process restart (the API is the
    // ENTIRE save path for the Unity titles — no file Mount — so without this every relaunch looks like
    // a fresh console and the game restarts from scratch; likely root of #299). Dir: PROSPER_SAVEDATA_DIR
    // (default /tmp/prosper-savedata-mem), one file per (userId, slotId).
    std::string savemem_path(int32_t userId, uint32_t slotId) {
        static const std::string base = [] {
            const char* e = getenv("PROSPER_SAVEDATA_DIR");
            std::string b = e ? e : "/tmp/prosper-savedata-mem";
#ifdef _WIN32
            _mkdir(b.c_str());
#else
            mkdir(b.c_str(), 0777);
#endif
            return b;
        }();
        char name[64];
        snprintf(name, sizeof name, "/savemem_%d_%u.bin", (int)userId, (unsigned)slotId);
        return base + name;
    }
    std::vector<uint8_t> savemem_load(int32_t userId, uint32_t slotId) {
        std::vector<uint8_t> v;
        if (FILE* f = fopen(savemem_path(userId, slotId).c_str(), "rb")) {
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            if (n > 0) { v.resize((size_t)n); if (fread(v.data(), 1, (size_t)n, f) != (size_t)n) v.clear(); }
            fclose(f);
        }
        return v;
    }
    void savemem_store(int32_t userId, uint32_t slotId, const std::vector<uint8_t>& buf) {
        if (FILE* f = fopen(savemem_path(userId, slotId).c_str(), "wb")) {
            if (!buf.empty()) fwrite(buf.data(), 1, buf.size(), f);
            fclose(f);
        }
    }
    template <class T> T ld(uint64_t base, size_t off) {   // read a guest struct field at byte offset
        T v; memcpy(&v, (const uint8_t*)PW(base) + off, sizeof(T)); return v;
    }
    constexpr uint64_t SD_ERR_PARAMETER      = 0x809F0000ull;
    constexpr uint64_t SD_ERR_MEMORY_NOTREADY = 0x809F0012ull;   // Set/Get before a successful Setup
}
// sceSaveDataSetupSaveDataMemory2(setup*, result*): allocate/grow the slot's block to memorySize
// (zero-filling new bytes, preserving any bytes already there this session). result->existedMemorySize
// (@0) = the size the block had before — the game uses it to tell first-run from resume.
HLE(s_savemem_setup) {
    svc_log("sceSaveDataSetupSaveDataMemory2", a0,a1,a2,a3,a4,a5);
    if (!a0) return SD_ERR_PARAMETER;
    int32_t  userId  = ld<int32_t>(a0, 4);
    uint64_t memSize = ld<uint64_t>(a0, 8);
    uint32_t slotId  = ld<uint32_t>(a0, 40);
    uint64_t existed = 0;
    {
        std::lock_guard<std::mutex> lk(g_savemem_mx);
        auto& buf = g_savemem[savemem_key(userId, slotId)];
        if (buf.empty()) buf = savemem_load(userId, slotId);       // reload a prior session's save from disk
        existed = buf.size();                                       // -> existedMemorySize is truthful across restarts
        if (memSize > buf.size()) buf.resize(memSize, 0);          // grow only; never drop live data
    }
    if (a1) *(uint64_t*)PW(a1) = existed;                          // result->existedMemorySize
    return 0;
}
// sceSaveDataSetSaveDataMemory2(set*): copy each of dataNum {buf,bufSize,offset} descriptors from
// guest memory into the slot's block at its offset (bounds-clamped — never write past the block).
HLE(s_savemem_set) {
    svc_log("sceSaveDataSetSaveDataMemory2", a0,a1,a2,a3,a4,a5);
    if (!a0) return SD_ERR_PARAMETER;
    int32_t  userId  = ld<int32_t>(a0, 0);
    uint64_t dataPtr = ld<uint64_t>(a0, 8);
    uint32_t dataNum = ld<uint32_t>(a0, 32);
    uint32_t slotId  = ld<uint32_t>(a0, 36);
    std::lock_guard<std::mutex> lk(g_savemem_mx);
    auto it = g_savemem.find(savemem_key(userId, slotId));
    if (it == g_savemem.end()) return SD_ERR_MEMORY_NOTREADY;      // Setup not called for this slot
    auto& buf = it->second;
    for (uint32_t i = 0; i < dataNum && dataPtr; i++) {
        uint64_t d     = dataPtr + (uint64_t)i * 64;               // sizeof(OrbisSaveDataMemoryData)
        uint64_t gbuf  = ld<uint64_t>(d, 0);
        uint64_t gsize = ld<uint64_t>(d, 8);
        int64_t  off   = ld<int64_t>(d, 16);
        if (!gbuf || off < 0 || (uint64_t)off >= buf.size()) continue;
        uint64_t n = std::min<uint64_t>(gsize, buf.size() - (uint64_t)off);
        memcpy(buf.data() + off, PW(gbuf), n);
    }
    return 0;
}
// sceSaveDataGetSaveDataMemory2(get*): copy from the slot's block into the single {buf,bufSize,offset}
// descriptor (Get2 has no dataNum — one descriptor), bounds-clamped.
HLE(s_savemem_get) {
    svc_log("sceSaveDataGetSaveDataMemory2", a0,a1,a2,a3,a4,a5);
    if (!a0) return SD_ERR_PARAMETER;
    int32_t  userId  = ld<int32_t>(a0, 0);
    uint64_t dataPtr = ld<uint64_t>(a0, 8);
    uint32_t slotId  = ld<uint32_t>(a0, 32);
    std::lock_guard<std::mutex> lk(g_savemem_mx);
    auto it = g_savemem.find(savemem_key(userId, slotId));
    if (it == g_savemem.end()) return SD_ERR_MEMORY_NOTREADY;
    auto& buf = it->second;
    if (dataPtr) {
        uint64_t gbuf  = ld<uint64_t>(dataPtr, 0);
        uint64_t gsize = ld<uint64_t>(dataPtr, 8);
        int64_t  off   = ld<int64_t>(dataPtr, 16);
        if (gbuf && off >= 0 && (uint64_t)off < buf.size()) {
            uint64_t n = std::min<uint64_t>(gsize, buf.size() - (uint64_t)off);
            memcpy(PW(gbuf), buf.data() + off, n);
        }
    }
    return 0;
}
// sceSaveDataSyncSaveDataMemory(sync*): commit the slot's block. A session's Set is already visible to a
// later Get in-process; Sync additionally writes the block to a host file (savemem_store), so the save
// survives a process restart — Setup reloads it next launch, making existedMemorySize truthful.
HLE(s_savemem_sync) {
    svc_log("sceSaveDataSyncSaveDataMemory", a0,a1,a2,a3,a4,a5);
    if (!a0) return SD_ERR_PARAMETER;
    int32_t  userId = ld<int32_t>(a0, 0);
    uint32_t slotId = ld<uint32_t>(a0, 4);
    std::lock_guard<std::mutex> lk(g_savemem_mx);
    auto it = g_savemem.find(savemem_key(userId, slotId));
    if (it == g_savemem.end()) return SD_ERR_MEMORY_NOTREADY;
    savemem_store(userId, slotId, it->second);   // Sync commits the slot to disk (survives restart)
    return 0;
}
// sceSaveDataMount3(const Mount3* mount, MountResult* result). The mount desc layout is pinned
// from DOLL's OWN wrapper (eboot+0x2251610 disassembly, matching the live capture):
//   +0x00 u32 userId; +0x08 const char* dirName; +0x10 u64 blocks; +0x20 u32 mountMode;
//   +0x28 u32 transactionResourceId  (the id returned by sceSaveDataCreateTransactionResource)
// Live: dirName="Book", blocks 0x60 then 0x105, mode 1 (open-RO) then 5 (CREATE|RO).
// mountMode bits are PS4-inherited: 1=RDONLY 2=RDWR 4=CREATE 8=DESTRUCT_OFF 16=COPY_ICON.
// Behavior (real console semantics): open of a nonexistent save -> NOT_FOUND (fresh console;
// the game handles it and retries with CREATE); CREATE makes the save dir and mounts it.
// The 0x40-byte result is zeroed by the caller and fed to sceSaveDataPrepare(&{txId}, result);
// we fill it with the PS4 MountResult shape (the only referenced layout): mountPoint char[16]
// "/savedata0" @+0x00, requiredBlocks u64 @+0x10 = 0, mountStatus u32 @+0x1c (0=opened,
// 1=created). hle_file translates /savedata0 to the mounted host dir.
// CONFIDENCE: HIGH on the mount-desc layout + mode semantics (guest disasm + live capture +
// PS4 references agree); MED on the result layout (PS4 shape; the PS5 field placement is
// unproven — under live test which offsets the game actually reads).
HLE(s_savedata_mount3)  {
    svc_log("sceSaveDataMount3", a0,a1,a2,a3,a4,a5);
    if (!a0 || !a1) return SAVE_DATA_ERR_PARAMETER;
    const uint8_t* m = (const uint8_t*)PW(a0);
    const char* dirname = *(const char* const*)(m + 0x08);
    uint32_t mode = *(const uint32_t*)(m + 0x20);
    if (!dirname) return SAVE_DATA_ERR_PARAMETER;
    bool create = (mode & 0x24) != 0;   // CREATE(4) | CREATE2(0x20, create-if-missing; live: mode 0x20 remount)
    if (!savedata0_mount(dirname, create)) {
        if (svclog()) fprintf(stderr, "[svc]   Mount3 dir='%s' mode=%#x -> NOT_FOUND\n", dirname, mode);
        return SAVE_DATA_ERR_NOT_FOUND;
    }
    uint8_t* r = (uint8_t*)PW(a1);
    memset(r, 0, 0x40);
    memcpy(r, "/savedata0", 11);
    *(uint32_t*)(r + 0x1c) = create ? 1u : 0u;   // mountStatus: created vs opened
    if (svclog()) fprintf(stderr, "[svc]   Mount3 dir='%s' mode=%#x -> OK (create=%d)\n", dirname, mode, (int)create);
    return 0;
}
HLE(s_savedata_umount2) {
    svc_log("sceSaveDataUmount2", a0,a1,a2,a3,a4,a5);
    savedata0_umount();
    g_savedata_umount_events.fetch_add(1, std::memory_order_release);
    return 0;
}
// sceSaveDataGetMountInfo(mp, OrbisSaveDataMountInfo* info = { u64 blocks; u64 freeBlocks; u8 rsv[32] }, 48
// bytes): was MISSING -> success with garbage free-space, so a title sizing its save against freeBlocks
// could abort ("disk full") or corrupt its math. Report a generous, consistent size (256K blocks free).
HLE(s_savedata_mountinfo) {
    svc_log("sceSaveDataGetMountInfo", a0,a1,a2,a3,a4,a5);
    if (!a1) return 0x809F0000ull;   // SAVE_DATA_ERROR_PARAMETER
    uint8_t* i = (uint8_t*)PW(a1); memset(i, 0, 48);
    *(uint64_t*)(i + 0) = 0x40000; *(uint64_t*)(i + 8) = 0x40000;   // blocks / freeBlocks
    return 0;
}
HLE(s_savedata_prepare) { svc_log("sceSaveDataPrepare", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_savedata_commit)  { svc_log("sceSaveDataCommit", a0,a1,a2,a3,a4,a5); return 0; }
// sceSaveDataGetEventResult(eventParam, event): PS4 and PS5 share this NID and two-argument shape.
// Our file operations complete synchronously, but Umount2 still queues the completion event a title
// uses to finish its save job. Return one UMOUNT_BACKUP event per unmount, then NO_EVENT. Returning
// generic success with an untouched event made Dead Cells consume a fabricated type-0 completion.
// Event: { u32 type; s32 errorCode; s32 userId; u32 pad; titleId[16]; dirName[32]; reserved[40] },
// 104 bytes. CONFIDENCE: HIGH on signature/error/type/size (Dead Cells zeroes exactly 104 bytes and
// the identical PS4 NID/API defines that layout), MED on the PS5 tail field names.
HLE(s_savedata_get_event) {
    svc_log("sceSaveDataGetEventResult", a0,a1,a2,a3,a4,a5);
    if (!a1) return SAVE_DATA_ERR_PARAMETER;
    unsigned pending = g_savedata_umount_events.load(std::memory_order_acquire);
    while (pending) {
        if (g_savedata_umount_events.compare_exchange_weak(
                pending, pending - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            uint8_t* event = (uint8_t*)PW(a1);
            memset(event, 0, 104);
            *(uint32_t*)(event + 0) = 1;  // SCE_SAVE_DATA_EVENT_TYPE_UMOUNT_BACKUP
            *(int32_t*)(event + 8) = 1;  // initial user
            return 0;
        }
    }
    return SAVE_DATA_ERR_NO_EVENT;
}
// sceSaveDataDirNameSearch(const SearchCond* cond, SearchResult* result) — PS4-inherited contract,
// pinned by DOLL's own callsite (eboot+0x224e920): cond {u32 userId@0; titleId*@8=0; dirName*@0x10=0;
// key/order@0x18=0}, result {u32 hitNum@0; DirName* dirNames@8 (caller buffer, this+0x40);
// u32 dirNamesNum@0x10=0x400; u32 setNum@0x14(1.7+)} — the guest stores hitNum to this+0x8040 as its
// save count. Fresh console: 0 saves found, success (shadPS4 does exactly this when the save path
// doesn't exist). CONFIDENCE: HIGH (guest callsite disassembly + shadPS4 agree).
HLE(s_savedata_dirsearch) {
    svc_log("sceSaveDataDirNameSearch", a0,a1,a2,a3,a4,a5);
    if (!a1) return 0x809F0000ull;        // SAVE_DATA_ERROR_PARAMETER
    uint8_t* res = (uint8_t*)PW(a1);
    // Enumerate the save dirs on disk so a prior session's saves show up in the load/continue list (#299).
    // Was hard-coded to 0 hits, so persisted saves were invisible. Optional cond->dirName filter @cond+0x10.
    const char* filter = nullptr;
    if (a0) { uint64_t dnp = *(uint64_t*)((uint8_t*)PW(a0) + 0x10); if (dnp) filter = (const char*)PW(dnp); }
    std::vector<std::string> dirs = savedata0_list_dirs();
    uint32_t cap = *(uint32_t*)(res + 0x10);            // dirNamesNum (caller buffer capacity, entries)
    if (cap > 0x400) cap = 0x400;                        // clamp to the documented buffer size
    uint64_t buf = *(uint64_t*)(res + 0x08);            // DirName* dirNames (caller buffer)
    uint32_t total = 0;
    for (const std::string& name : dirs) {
        if (filter && name != filter) continue;
        if (buf && (!cap || total < cap)) {              // SceSaveDataDirName = char dirName[32]
            char* entry = (char*)PW(buf) + (size_t)total * 32;
            memset(entry, 0, 32);
            strncpy(entry, name.c_str(), 31);
        }
        total++;
    }
    uint32_t hit = (cap && total > cap) ? cap : total;
    *(uint32_t*)(res + 0x00) = hit;        // hitNum
    *(uint32_t*)(res + 0x14) = hit;        // setNum
    return 0;
}

// --- libSceNpTrophy2 lifecycle: succeed with valid ids (trophy CONTENT stays unavailable). ------
// PS4 NpTrophy ABI carried to Trophy2 (context/handle are small s32 ids written through arg0;
// Kyty LibNpTrophy + shadPS4 np_trophy agree on the PS4 shape). The game's trophy worker needs
// CreateContext/CreateHandle/RegisterContext to hand back usable ids so its bring-up completes;
// the info queries (GetGameInfo/GetTrophyInfoArray) keep returning "unavailable" (see
// s_nptrophy2_unavailable above) which the guest handles on a clean path. CONFIDENCE: MED.
HLE(s_nptrophy2_createctx)    { svc_log("sceNpTrophy2CreateContext", a0,a1,a2,a3,a4,a5);
                                if (a0) *(int32_t*)PW(a0) = 1; return 0; }
HLE(s_nptrophy2_createhandle) { svc_log("sceNpTrophy2CreateHandle", a0,a1,a2,a3,a4,a5);
                                if (a0) *(int32_t*)PW(a0) = 1; return 0; }
HLE(s_nptrophy2_regctx)       { svc_log("sceNpTrophy2RegisterContext", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_nptrophy2_ok)           { return 0; }

// --- libSceShare: succeed; sharing features are simply unavailable headless. --------------------
// Initialize + the SetContentParam/SetScreenshotOverlayImage setters carry no out-params the
// guest reads; plain success lets the ShareUpdate worker finish its bring-up. CONFIDENCE: MED.
HLE(s_share_ok) { return 0; }

// --- libSceNpUniversalDataSystem (PS5 telemetry/activities): hand out ids, stay inert. ----------
// PS5-only, no reference implementation; by symmetry with every Np Create* API the first arg of
// CreateContext/CreateHandle is the out-id pointer (pointer-range-guarded so a wrong guess can't
// fault). Offline console: everything else no-ops. CONFIDENCE: LOW (guarded).
HLE(s_npuds_create) { svc_log("sceNpUniversalDataSystemCreate*", a0,a1,a2,a3,a4,a5);
                      if (svc_ptrish(a0)) *(int32_t*)PW(a0) = 1; return 0; }
HLE(s_npuds_ok)     { return 0; }
// Dead Cells' wrappers pin CreateEvent(name, reserved, Event**, PropertyObject**) and pass both
// returned opaque objects to the setter/post/destroy family. They are never dereferenced by the
// guest. Allocate stable non-null ids and keep the offline telemetry sink inert.
HLE(s_npuds_create_event) {
    svc_log("sceNpUniversalDataSystemCreateEvent", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a2) || !svc_ptrish(a3)) return 0x80550003ull; // NP invalid argument
    *(uint64_t*)PW(a2) = g_handle.fetch_add(1);
    *(uint64_t*)PW(a3) = g_handle.fetch_add(1);
    return 0;
}
HLE(s_npuds_post_event) {
    svc_log("sceNpUniversalDataSystemPostEvent", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npuds_destroy_event) {
    svc_log("sceNpUniversalDataSystemDestroyEvent", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npuds_object_set_string) {
    svc_log("sceNpUniversalDataSystemEventPropertyObjectSetString", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_gameintent_init) {
    svc_log("sceNpGameIntentInitialize", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npent_addcont_info) {
    svc_log("sceNpEntitlementAccessGetAddcontEntitlementInfo", a0,a1,a2,a3,a4,a5);
    return NP_ERR_SIGNED_OUT;
}

// ===== Issue #306: honest OFFLINE console for the online/update/entitlement boot chain. =========
// DOLL's UE4 front-end runs a patch/entitlement check before the title screen; every subsystem in
// that chain answered success-with-garbage, so the check could neither succeed nor FAIL — the flow
// waited forever (docs/DOLL_LOADING_PROGRESSION.md). The blocks below give the chain the answers a
// real, network-disconnected, signed-out console gives.

// --- Guest-callback delivery discipline (shared by NetCtl + Np state callbacks). ----------------
// A registered callback is guest code: under PROSPER_GUEST_FS the HLE runs on the HOST %fs (the
// import swap-stub switched), so the guest callback must run with the GUEST %fs restored or its
// TLS accesses (UE MallocBinned caches!) read host TLS garbage. The swap-stub saves the guest fs
// base in its frame (push r11), so an asm entry shim (the f_apr_read_submit_entry pattern) hands
// the handler its entry %rsp. The guest swap path re-pushes args7/8/9, an alignment pad, then the
// saved r11: [rsp]=ret-to-stub, args at +8/+0x10/+0x18, pad at +0x20, guest fs at +0x28, and guest
// RA at +0x30. A [rsp] outside the stub region [0x6_0000_0000,0x7_0000_0000) means the
// host-context tail-jmp path (no swap happened) — call the callback on the current fs.
// Mechanism proven live by the PROSPER_NETCTL_CB experiment (run 7/9: delivered + consumed
// cleanly, no crash). CONFIDENCE: HIGH.
#ifndef _WIN32
namespace {
inline uint64_t cb_rd_fsbase() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
inline void     cb_wr_fsbase(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }
// RAII: run the enclosed guest callback on the guest %fs (no-op when guest_fs==0).
struct CbGuestFsScope {
    uint64_t saved = 0, active = 0;
    explicit CbGuestFsScope(uint64_t guest_fs) {
        if (guest_fs) { saved = cb_rd_fsbase(); cb_wr_fsbase(guest_fs); active = guest_fs; }
    }
    ~CbGuestFsScope() { if (active) cb_wr_fsbase(saved); }
};
}
#endif

// --- libSceNetCtl: a network-DISCONNECTED console (default ON since #306). ----------------------
// DOLL registers a NetCtl state callback once at boot (sceNetCtlRegisterCallback) and then pumps
// sceNetCtlCheckCallback EXACTLY once per frame forever (14,191 calls in a 240 s run). On real
// hardware CheckCallback invokes the registered callback on the calling thread with the current
// state — an offline console still delivers an immediate DISCONNECTED. Register records {func,arg}
// and writes the callback id (Kyty Network.cpp NetCtlRegisterCallback); CheckCallback invokes the
// callback ONCE with SCE_NET_CTL_EVENT_TYPE_DISCONNECTED (PS4-inherited constant = 1; identical
// export names+NIDs on PS5 3.20 — CONFIDENCE MED on the PS5 value). Was the gated experiment
// PROSPER_NETCTL_CB=1; proven correct+consumed live (DOLL run 7/9), now default ON.
// PROSPER_NETCTL_CB=0 restores the old unimplemented behavior.
#ifndef _WIN32
namespace {
std::atomic<uint64_t> g_netctl_cb_fn{0};
std::atomic<uint64_t> g_netctl_cb_arg{0};
std::atomic<int>      g_netctl_cb_delivered{0};
}
HLE(s_netctl_register_cb) {   // (func, arg, int* cid)
    svc_log("sceNetCtlRegisterCallback", a0,a1,a2,a3,a4,a5);
    g_netctl_cb_fn.store(a0);
    g_netctl_cb_arg.store(a1);
    if (svc_ptrish(a2)) *(int32_t*)PW(a2) = 1;   // callback id (Kyty Network.cpp NetCtlRegisterCallback)
    return 0;
}
// sceNetCtlGetState(int* state): 0 = DISCONNECTED (Kyty Network.cpp:1398 writes exactly this).
// Run-7 live capture: the game calls this for the FIRST time immediately after the DISCONNECTED
// callback delivery — the unimplemented success+garbage-out answer is what re-wedged the flow.
HLE(s_netctl_getstate) {
    svc_log("sceNetCtlGetState", a0,a1,a2,a3,a4,a5);
    if (svc_ptrish(a0)) *(int32_t*)PW(a0) = 0;   // SCE_NET_CTL_STATE_DISCONNECTED
    return 0;
}
extern "C" uint64_t s_netctl_check_cb_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                        uint64_t a3, uint64_t a4, uint64_t a5,
                                        uint64_t entry_rsp);
PROSPER_ASM_TRAMPOLINE(s_netctl_check_cb_entry, s_netctl_check_cb_c)
extern "C" void s_netctl_check_cb_entry();
extern "C" uint64_t s_netctl_check_cb_c(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t entry_rsp) {
    uint64_t fn = g_netctl_cb_fn.load();
    if (!fn || g_netctl_cb_delivered.exchange(1)) return 0;   // deliver the initial state exactly once
    uint64_t gfs = callback_guest_fs_from_entry_stack(entry_rsp);
    {
        CbGuestFsScope fs(gfs);
        ((void (*)(int, void*))(uintptr_t)fn)(1 /*SCE_NET_CTL_EVENT_TYPE_DISCONNECTED*/,
                                              (void*)(uintptr_t)g_netctl_cb_arg.load());
    }
    // NOTE: log only AFTER the scope restored the host %fs — host libc (fprintf) reads %fs-based
    // TLS and crashes on the guest %fs (learned the hard way: NULL+0x308 fault in libc).
    fprintf(stderr, "[svc] NetCtl state callback DELIVERED (eventType=DISCONNECTED, guest_fs=%d)\n",
            gfs ? 1 : 0);
    return 0;
}
#endif
// sceNetCtlGetInfo(int code, SceNetCtlInfo* info): a console with no network connection answers
// NOT_CONNECTED for the connection-dependent info codes and writes nothing (shadPS4 netctl.cpp:163
// returns ORBIS_NET_CTL_ERROR_NOT_CONNECTED = 0x80412108 for ALL codes when disconnected; Kyty
// only implements the connected path). PS4-inherited surface, identical export on PS5 3.20
// (obuxdTiwkF8). Works on all platforms (no callback machinery). CONFIDENCE: HIGH on semantics,
// MED on the PS5 errno value (same 0x8041 NetCtl facility).
HLE(s_netctl_getinfo) {
    svc_log("sceNetCtlGetInfo", a0,a1,a2,a3,a4,a5);
    return 0x80412108ull;   // SCE_NET_CTL_ERROR_NOT_CONNECTED
}

// --- libSceNpManager state callback: deliver SIGNED_OUT once (#306). ----------------------------
// DOLL registers its Np sign-in state callback via sceNpRegisterStateCallbackA and pumps
// sceNpCheckCallback. shadPS4 (offline mode) queues exactly one SIGNED_OUT event for the initial
// user and delivers it inside sceNpCheckCallback on the pumping thread; we mirror that. Callback-A
// prototype (shadPS4 np_manager.h): void cb(s32 userId, s32 state, void* userdata); state
// SIGNED_OUT = 1 (Unknown=0, SignedOut=1, SignedIn=2 — Kyty + shadPS4 agree). Register returns the
// positive callback id (shadPS4 RegisterStateCallbackA returns slot+1). CONFIDENCE: HIGH on the
// contract (two agreeing PS4 references, PS4-inherited surface; PS5 3.20 exports the same names).
#ifndef _WIN32
namespace {
struct NpStateCbSlot { std::atomic<uint64_t> fn{0}, arg{0}; std::atomic<int> delivered{0}; };
NpStateCbSlot     g_np_state_cbs[4];
std::atomic<int>  g_np_state_cb_n{0};
}
HLE(s_np_register_state_cbA) {   // (SceNpStateCallbackA func, void* userdata) -> callback id
    svc_log("sceNpRegisterStateCallbackA", a0,a1,a2,a3,a4,a5);
    if (!a0) return 0x80550003ull;   // SCE_NP_ERROR_INVALID_ARGUMENT
    int i = g_np_state_cb_n.fetch_add(1);
    if (i >= 4) { g_np_state_cb_n.store(4); return 0x8055001Dull; }  // SCE_NP_ERROR_CALLBACK_MAX
    g_np_state_cbs[i].arg.store(a1);
    g_np_state_cbs[i].fn.store(a0);
    return (uint64_t)(i + 1);
}
extern "C" uint64_t s_np_check_cb_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                    uint64_t a3, uint64_t a4, uint64_t a5,
                                    uint64_t entry_rsp);
PROSPER_ASM_TRAMPOLINE(s_np_check_cb_entry, s_np_check_cb_c)
extern "C" void s_np_check_cb_entry();
extern "C" uint64_t s_np_check_cb_c(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t entry_rsp) {
    int n = g_np_state_cb_n.load(); if (n > 4) n = 4;
    uint64_t gfs = callback_guest_fs_from_entry_stack(entry_rsp);
    for (int i = 0; i < n; i++) {
        uint64_t fn = g_np_state_cbs[i].fn.load();
        if (!fn || g_np_state_cbs[i].delivered.exchange(1)) continue;
        {
            CbGuestFsScope fs(gfs);
            ((void (*)(int32_t, int32_t, void*))(uintptr_t)fn)(
                1 /*initial userId (sceUserServiceGetInitialUser)*/, 1 /*SCE_NP_STATE_SIGNED_OUT*/,
                (void*)(uintptr_t)g_np_state_cbs[i].arg.load());
        }
        // Log only on the restored host %fs (fprintf on the guest %fs faults in libc TLS).
        fprintf(stderr, "[svc] Np state callback DELIVERED (userId=1, state=SIGNED_OUT, guest_fs=%d)\n",
                gfs ? 1 : 0);
    }
    return 0;
}
#endif

// --- libSceErrorDialog: the real Initialize/Open/Close lifecycle (auto-dismiss, headless). ------
// Status enum shared with CommonDialog: NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3 (shadPS4
// commondialog.h + error_dialog.cpp; PS4-inherited, identical export NIDs on PS5 3.20).
// GetStatus/UpdateStatus RETURN the status value directly (not an out-param). DOLL pumps
// sceErrorDialogUpdateStatus once per frame from boot (housekeeping); if the honest offline chain
// makes it Open the "could not download patch data" dialog, auto-dismiss to FINISHED so the flow's
// "wait until dismissed" loop exits — the MsgDialog precedent (#144). Open's param is
// { s32 size; s32 errorCode; s32 userId; s32 reserved } (shadPS4 error_dialog.cpp Param); the
// errorCode is printed unconditionally — a one-shot, high-value diagnostic of WHAT the game
// thinks failed. CONFIDENCE: HIGH (shadPS4 implements this exact lifecycle).
namespace { std::atomic<int> g_errdialog_status{0 /*NONE*/}; std::atomic<int> g_errdialog_backed{0}; }
HLE(s_errdialog_init)   { g_errdialog_backed.store(0); g_errdialog_status.store(1 /*INITIALIZED*/); return 0; }
HLE(s_errdialog_open)   {
    // Offer the error dialog to a registered PlatformUi first (a real message box); else auto-dismiss.
    if (auto* ui = platform_ui(); ui && ui->errorDialogOpen(a0)) { g_errdialog_backed.store(1); return 0; }
    g_errdialog_backed.store(0);
    uint32_t code = 0;
    if (svc_ptrish(a0)) code = *(const uint32_t*)((const char*)PW(a0) + 4);
    fprintf(stderr, "[svc] sceErrorDialogOpen(errorCode=%#x) -> auto-dismiss FINISHED\n", code);
    g_errdialog_status.store(3 /*FINISHED (auto-dismiss)*/);
    return 0;
}
HLE(s_errdialog_close)  { if (g_errdialog_backed.exchange(0)) { if (auto* ui = platform_ui()) ui->errorDialogClose(); } g_errdialog_status.store(3 /*FINISHED*/); return 0; }
HLE(s_errdialog_term)   { if (g_errdialog_backed.exchange(0)) { if (auto* ui = platform_ui()) ui->errorDialogClose(); } g_errdialog_status.store(0 /*NONE*/); return 0; }
HLE(s_errdialog_status) {
    if (g_errdialog_backed.load()) { if (auto* ui = platform_ui()) return (uint64_t)(unsigned)ui->errorDialogStatus(); }
    return (uint64_t)(unsigned)g_errdialog_status.load();
}

// --- libSceNpEntitlementAccess / libSceGameUpdate: observability first. -------------------------
// PS5-only surfaces with NO reference implementation (absent from Kyty and shadPS4); the PS5 3.20
// stub tables give names+NIDs only. DOLL calls sceNpEntitlementAccessInitialize (retried 3x) and
// sceGameUpdateInitialize once, then never the follow-ups (CreateRequest/Check) — so before
// inventing contracts, capture the REAL args live (PROSPER_SVCLOG=1) and keep the return the same
// success the unimplemented path produced (a real console's local library init succeeds offline
// too). The follow-up calls stay UNIMPLEMENTED deliberately: if the honest Np/NetCtl answers
// unblock the flow into them, they surface in the PROSPER_PROGRESS_UNIMPL dump and get pinned from
// their live args before being given behavior. CONFIDENCE: MED (init-succeeds is the real-console
// offline behavior; arg shapes intentionally not guessed).
HLE(s_npent_init)      { svc_log("sceNpEntitlementAccessInitialize", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_gameupdate_init) { svc_log("sceGameUpdateInitialize",          a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_gameupdate_term) { svc_log("sceGameUpdateTerminate",           a0,a1,a2,a3,a4,a5); return 0; }
// The entitlement follow-ups DOLL's main menu fires once the flow is unblocked (live-captured
// after the #306 gate fell). ABI pinned from the live capture (r8):
//   GetAddcontEntitlementInfoList(SceNpServiceLabel serviceLabel, Info* list, u32 listNum,
//                                 u32* hitNum)
// — the game first count-queries with list=NULL/num=0 and an out pointer in a3, then calls again
// with a 4-entry buffer (its four addcont slots). Matches the documented PS4 signature 1:1.
// This dump has NO additional content installed and no signed-in user, so the truthful console
// answer is SUCCESS with hitNum=0 (DLC entitlement lookups work offline from local state; there
// simply are none). CONFIDENCE: HIGH on the shape (live-pinned + PS4 doc agree), MED on hitNum=0
// being the exact retail no-DLC answer.
HLE(s_npent_addcont_list) {
    svc_log("sceNpEntitlementAccessGetAddcontEntitlementInfoList", a0,a1,a2,a3,a4,a5);
    if (svc_ptrish(a3)) *(uint32_t*)PW(a3) = 0;   // hitNum = 0: no addcont entitlements
    return 0;
}
// GetEntitlementKey(serviceLabel, const Label* label, Key* out) — live capture: retried 4x with
// identical args (the garbage-consuming retry signature). With no entitlement to derive a key
// from, honest FAILURE beats success+garbage-key; the exact NpEntitlementAccess errno space is
// unreferenced, so the generic Np signed-out error is used (only the sign is consumed by a clean
// caller). With hitNum=0 above the game should no longer ask at all. CONFIDENCE: LOW on the
// error constant, HIGH that failure is the truthful state.
HLE(s_npent_getkey) {
    svc_log("sceNpEntitlementAccessGetEntitlementKey", a0,a1,a2,a3,a4,a5);
    return NP_ERR_SIGNED_OUT;
}

// sceSaveDataTransferringMount (PS5-only, live-captured x4 in DOLL's save-slot menu): mount a
// PS4-era save for transfer/import. We have no PS4 save to transfer — the truthful fresh-console
// answer is NOT_FOUND with the result untouched. The previous success+garbage made the game read
// an EMPTY mount-point string out of the unwritten result and open '/GameSaveData245.dat' (no
// mount prefix) at filesystem root — observed live. CONFIDENCE: MED on semantics (same 0x809F
// facility + NOT_FOUND as Mount3 of a nonexistent save), LOW on the exact arg layout (nothing is
// written, so no layout is assumed; PROSPER_SVCLOG captures it).
HLE(s_savedata_transfermount) {
    svc_log("sceSaveDataTransferringMount", a0,a1,a2,a3,a4,a5);
    return SAVE_DATA_ERR_NOT_FOUND;
}

// sceSystemServiceGetNoticeScreenSkipFlag(bool* flag) — polled from DOLL's front-end menu.
// PS5-only (no reference). Live capture pinned the out-pointer to an ODD stack address
// (0x...ff307), so the flag is a single byte (bool), NOT an int32 — a 4-byte write would clobber
// 3 adjacent stack bytes. 0 = "no skip" is the inert default a retail console with no
// notice-screen state reports. CONFIDENCE: MED (byte-sized out pinned live; value semantics LOW).
HLE(s_syss_noticeskip) {
    svc_log("sceSystemServiceGetNoticeScreenSkipFlag", a0,a1,a2,a3,a4,a5);
    if (svc_ptrish(a0)) *(uint8_t*)PW(a0) = 0;
    return 0;
}

void register_service_hle() {
    register_json2_hle();
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
#ifndef _WIN32
    // NetCtl offline-console state delivery — default ON since #306 (see block comment above).
    // PROSPER_NETCTL_CB=0 restores the previous unimplemented behavior.
    {
        const char* e = getenv("PROSPER_NETCTL_CB");
        if (!e || strtol(e, nullptr, 0) != 0) {
            R("sceNetCtlRegisterCallback", s_netctl_register_cb);      // UJ+Z7Q+4ck0
            R("sceNetCtlCheckCallback",    s_netctl_check_cb_entry);   // iQw3iQPhvUQ
            R("sceNetCtlGetState",         s_netctl_getstate);         // uBPlr0lbuiI
        }
    }
#endif
    Hle::register_fn("obuxdTiwkF8", (HleFn)s_netctl_getinfo, "sceNetCtlGetInfo");  // NOT_CONNECTED
    // NpTrophy2: the config/info queries whose success-with-garbage-out crashed DOLL (see above).
    Hle::register_fn("4IzqhhUQ3nk", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetGameInfo");
    Hle::register_fn("y3zHpdZO6ME", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetTrophyInfoArray");
    // user service
    R("sceUserServiceGetInitialUser", s_user_initial);
    Hle::register_fn("eNb53LQJmIM", (HleFn)s_user_initial, "sceUserServiceGetForegroundUser");  // was MISSING -> garbage userId
    R("sceUserServiceGetEvent", s_user_getevent);   // deliver the initial-user LOGIN event once
    R("sceUserServiceGetLoginUserIdList", s_user_idlist);
    R("sceUserServiceGetUserName", s_user_name);
    R("sceUserServiceGetAccessibilityVibration", s_user_int_out);
    R("sceUserServiceGetAccessibilityPressAndHoldDelay", s_user_int_out);
    R("sceUserServiceGetAccessibilityZoomEnabled", s_user_int_out);
    // More (userId, int* out) getters the game queries at startup — same family: write a sane default
    // (accessibility off = 0; age level = adult) so the caller reads a deterministic value instead of
    // uninitialized stack. Registered by raw NID (guaranteed match; these names aren't in our NidDb).
    Hle::register_fn("woNpu+45RLk", (HleFn)s_user_age,     "sceUserServiceGetAgeLevel");
    Hle::register_fn("rnEhHqG-4xo", (HleFn)s_user_int_out, "sceUserServiceGetAccessibilityChatTranscription");
    Hle::register_fn("O6IW1-Dwm-w", (HleFn)s_user_int_out, "sceUserServiceGetAccessibilityZoomFollowFocus");
    Hle::register_fn("-3Y5GO+-i78", (HleFn)s_user_int_out, "sceUserServiceGetAccessibilityTriggerEffect");
    // sceUserServiceGetGamePresets (-sD02mFDBh4): returns 0 with a zeroed payload (see s_gamepresets).
    // History: it once returned 0x80960006 to avoid the game reading an untouched garbage struct — but
    // that non-zero errno made the Unity engine's per-controller check disconnect the pad every frame
    // and killed ALL gamepad input (#234). s_gamepresets now zeroes the payload after the caller-set
    // size field and returns success, satisfying both concerns.
    Hle::register_fn("-sD02mFDBh4", (HleFn)s_gamepresets, "sceUserServiceGetGamePresets");
    R("sceUserServiceInitialize", s_ok);
    R("sceUserServiceTerminate", s_ok);
    // NP — an honest signed-out console (#306). NIDs verified against the PS5 3.20
    // libSceNpManager stub table AND shadPS4's PS4 registrations (identical).
    R("sceNpGetState", s_np_state);
    R("sceNpGetNpReachabilityState", s_np_reach);
    R("sceNpGetAccountIdA", s_np_accountid);
    R("sceNpGetAccountCountryA", s_np_country);
    Hle::register_fn("XDncXQIJUSk", (HleFn)s_np_getonlineid, "sceNpGetOnlineId");
    Hle::register_fn("p-o74CnoNzY", (HleFn)s_np_getnpid,   "sceNpGetNpId");       // was MISSING -> faked identity
    Hle::register_fn("2rsFmlGWleQ", (HleFn)s_np_check_avail, "sceNpCheckNpAvailability");   // was MISSING -> faked "available"
    Hle::register_fn("8Z2Jc5GvGDI", (HleFn)s_np_check_avail, "sceNpCheckNpAvailabilityA");
    Hle::register_fn("KfGZg2y73oM", (HleFn)s_np_check_avail, "sceNpCheckNpReachability");
    Hle::register_fn("a8R9-75u4iM", (HleFn)s_np_accountid, "sceNpGetAccountId");  // non-A variant: zero id + SIGNED_OUT
    R("sceNpRegisterStateCallback", s_ok);
#ifndef _WIN32
    // sceNpCheckCallback pumps the registered A-callbacks (SIGNED_OUT delivered once, guest %fs).
    Hle::register_fn("3Zl8BePTh9Y", (HleFn)s_np_check_cb_entry,      "sceNpCheckCallback");
    Hle::register_fn("qQJfO8HAiaY", (HleFn)s_np_register_state_cbA,  "sceNpRegisterStateCallbackA");
    Hle::register_fn("M3wFXbYQtAA", (HleFn)s_ok,                     "sceNpUnregisterStateCallbackA");
#else
    R("sceNpCheckCallback", s_ok);
#endif
    // pad -> hle_pad.cpp (register_pad_hle). mouse:
    R("sceMouseInit", s_ok);
    R("sceMouseOpen", s_open);
    R("sceMouseRead", s_mouse_read);
    // app content / dialogs
    R("sceAppContentInitialize", s_ok);
    R("sceAppContentAppParamGetInt", s_appcontent_int);
    // temp-data mount: raw NIDs (names not in our NidDb). Unmount = OK (nothing to tear down).
    Hle::register_fn("buYbeLOGWmA", (HleFn)s_appcontent_tmpmount2, "sceAppContentTemporaryDataMount2");
    Hle::register_fn("SaKib2Ug0yI", (HleFn)s_appcontent_tmpspace, "sceAppContentTemporaryDataGetAvailableSpaceKb");
    Hle::register_fn("Gl6w5i0JokY", (HleFn)s_appcontent_tmpspace, "sceAppContentDownloadDataGetAvailableSpaceKb");  // was MISSING -> garbage KB
    Hle::register_fn("bcolXMmp6qQ", (HleFn)s_ok,                  "sceAppContentTemporaryDataUnmount");
    // add-content (DLC) enumeration — no-DLC truth (hitNum=0 / no-entitlement), NOT garbage-count OOM (#213 class)
    Hle::register_fn("xnd8BJzAxmk", (HleFn)s_appcontent_addcont_list,   "sceAppContentGetAddcontInfoList");
    Hle::register_fn("m47juOmH0VE", (HleFn)s_appcontent_no_entitlement, "sceAppContentGetAddcontInfo");
    Hle::register_fn("XTWR0UXvcgs", (HleFn)s_appcontent_no_entitlement, "sceAppContentGetEntitlementKey");
    Hle::register_fn("VANhIWcqYak", (HleFn)s_appcontent_no_entitlement, "sceAppContentAddcontMount");
    R("sceCommonDialogInitialize", s_ok);
    R("sceCommonDialogIsUsed", s_ok);   // 0 = not in use (our dialogs auto-dismiss) - intentional, not an unimpl log
    R("sceSystemServiceParamGetInt", s_syss_param_int);   // language-aware (US English default), not blanket 0
    // sceSystemServiceParamGetString (SsC-m-S9JTA): write a valid empty string (not an unfilled buffer).
    Hle::register_fn("SsC-m-S9JTA", (HleFn)s_param_string, "sceSystemServiceParamGetString");
    // message dialog: track the Initialize/Open/Close lifecycle (#144) — NONE/INITIALIZED before an
    // Open, then auto-dismiss to FINISHED so the startup dialog flow still completes headless.
    R("sceMsgDialogInitialize", s_dialog_initialize);  R("sceMsgDialogTerminate", s_dialog_terminate);
    R("sceMsgDialogOpen", s_dialog_open);              R("sceMsgDialogClose", s_dialog_close);
    R("sceMsgDialogUpdateStatus", s_dialog_status);
    R("sceMsgDialogGetStatus", s_dialog_status);
    R("sceMsgDialogGetResult", s_dialog_result);
    // libSceSaveDataDialog[.native] uses raw NIDs in current dumps. Register the complete lifecycle
    // so future modes do not fall back to success-without-state even though Dead Cells currently uses
    // only Initialize/Open/UpdateStatus (#768).
    Hle::register_fn("fH46Lag88XY", (HleFn)s_savedlg_close,      "sceSaveDataDialogClose");
    Hle::register_fn("yEiJ-qqr6Cg", (HleFn)s_savedlg_result,     "sceSaveDataDialogGetResult");
    Hle::register_fn("ERKzksauAJA", (HleFn)s_savedlg_status,     "sceSaveDataDialogGetStatus");
    Hle::register_fn("s9e3+YpRnzw", (HleFn)s_savedlg_initialize, "sceSaveDataDialogInitialize");
    Hle::register_fn("en7gNVnh878", (HleFn)s_savedlg_ready,      "sceSaveDataDialogIsReadyToDisplay");
    Hle::register_fn("4tPhsP6FpDI", (HleFn)s_savedlg_open,       "sceSaveDataDialogOpen");
    Hle::register_fn("V-uEeFKARJU", (HleFn)s_savedlg_progress,   "sceSaveDataDialogProgressBarInc");
    Hle::register_fn("hay1CfTmLyA", (HleFn)s_savedlg_progress,   "sceSaveDataDialogProgressBarSetValue");
    Hle::register_fn("YuH2FA7azqQ", (HleFn)s_savedlg_terminate,  "sceSaveDataDialogTerminate");
    Hle::register_fn("KK3Bdg1RWK0", (HleFn)s_savedlg_status,     "sceSaveDataDialogUpdateStatus");
    // libSceImeDialog (#191): auto-completing text-entry dialog (no keyboard UI). Raw NIDs.
    Hle::register_fn("NUeBrN7hzf0", (HleFn)s_imedlg_init,   "sceImeDialogInit");
    Hle::register_fn("IADmD4tScBY", (HleFn)s_imedlg_status, "sceImeDialogGetStatus");
    Hle::register_fn("x01jxu+vxlc", (HleFn)s_imedlg_result, "sceImeDialogGetResult");
    Hle::register_fn("gyTyVn+bXMw", (HleFn)s_imedlg_term,   "sceImeDialogTerm");
    Hle::register_fn("oBmw4xrmfKs", (HleFn)s_imedlg_abort,  "sceImeDialogAbort");
    R("sceSystemServiceHideSplashScreen", s_ok);
    // libSceIme keyboard API (#186): no physical keyboard -> consistent "none connected" state. Raw NIDs.
    Hle::register_fn("eaFXjfJv3xs", (HleFn)s_ime_kbd_open,  "sceImeKeyboardOpen");
    Hle::register_fn("-4GCfYdNF1s", (HleFn)s_ime_update,    "sceImeUpdate");
    Hle::register_fn("VkqLPArfFdc", (HleFn)s_ime_kbd_info,  "sceImeKeyboardGetInfo");
    Hle::register_fn("dKadqZFgKKQ", (HleFn)s_ime_kbd_resid, "sceImeKeyboardGetResourceId");
    // libSceAvPlayer (#324): let a post-credits / intro video complete so the game reaches its scene.
    // Init/InitEx must return a non-NULL handle; IsActive must report finished; the rest succeed as no-ops.
    R("sceAvPlayerInit",           s_avplayer_init);
    R("sceAvPlayerInitEx",         s_avplayer_initex);
    R("sceAvPlayerPostInit",       s_avp_postinit);
    R("sceAvPlayerSetLogCallback", s_avp_setlogcb);
#ifdef _WIN32
    R("sceAvPlayerAddSource",      s_avp_addsource);
    R("sceAvPlayerAddSourceEx",    s_avp_addsourceex);
    R("sceAvPlayerStart",          s_avp_start);
    R("sceAvPlayerIsActive",       s_avplayer_isactive);
    R("sceAvPlayerGetVideoData",   s_avp_getvideodata);
    R("sceAvPlayerGetVideoDataEx", s_avp_getvideodataex);
#else
    R("sceAvPlayerAddSource",      s_avp_addsource_entry);
    R("sceAvPlayerAddSourceEx",    s_avp_addsourceex_entry);
    R("sceAvPlayerStart",          s_avp_start_entry);
    R("sceAvPlayerIsActive",       s_avplayer_isactive_entry);
    R("sceAvPlayerGetVideoData",   s_avp_getvideodata_entry);
    R("sceAvPlayerGetVideoDataEx", s_avp_getvideodataex_entry);
#endif
    R("sceAvPlayerGetAudioData",   s_avp_getaudiodata);
#ifdef _WIN32
    R("sceAvPlayerStop",           s_avp_stop);
#else
    R("sceAvPlayerStop",           s_avp_stop_entry);
#endif
    R("sceAvPlayerClose",          s_avp_close);
    // PS5 raw NIDs verified against the 3.20 import stubs. Astro Bot enumerates streams before it
    // consumes GetVideoDataEx; returning the generic unresolved-import zero meant "no streams" and
    // left its logo movie state machine permanently resident even after synthetic EOF.
    Hle::register_fn("hdTyRzCXQeQ", (HleFn)s_avp_streamcount,   "sceAvPlayerStreamCount");
    Hle::register_fn("d8FcbzfAdQw", (HleFn)s_avp_getstreaminfo, "sceAvPlayerGetStreamInfo");
    Hle::register_fn("ctTAcF5DiKQ", (HleFn)s_avp_getstreaminfoex, "sceAvPlayerGetStreamInfoEx");
    Hle::register_fn("ODJK2sn9w4A", (HleFn)s_avp_stream_ok, "sceAvPlayerEnableStream");
    Hle::register_fn("BOVKAzRmuTQ", (HleFn)s_avp_stream_ok, "sceAvPlayerDisableStream");
    Hle::register_fn("buMCiJftcfw", (HleFn)s_avp_stream_ok, "sceAvPlayerChangeStream");
#ifdef _WIN32
    Hle::register_fn("9y5v+fGN4Wk", (HleFn)s_avp_pause, "sceAvPlayerPause");
    Hle::register_fn("w5moABNwnRY", (HleFn)s_avp_resume, "sceAvPlayerResume");
#else
    Hle::register_fn("9y5v+fGN4Wk", (HleFn)s_avp_pause_entry, "sceAvPlayerPause");
    Hle::register_fn("w5moABNwnRY", (HleFn)s_avp_resume_entry, "sceAvPlayerResume");
#endif
    Hle::register_fn("k-q+xOxdc3E", (HleFn)s_avp_stream_ok, "sceAvPlayerSetAvSyncMode");
    Hle::register_fn("OVths0xGfho", (HleFn)s_avp_stream_ok, "sceAvPlayerSetLooping");
    R("sceSystemServiceGetStatus", s_syss_getstatus);
    R("sceSystemServiceReceiveEvent", s_sysservice_receiveevent);   // NO_EVENT, don't leave the struct garbage
    // sceSystemServiceGetDisplaySafeAreaInfo (1n37q1Bvc5Y) — fill ratio=1.0 (see s_syss_safearea).
    Hle::register_fn("1n37q1Bvc5Y", (HleFn)s_syss_safearea, "sceSystemServiceGetDisplaySafeAreaInfo");

    // ---- Issue #232 services (raw NIDs; every pair verified against the PS5 3.20 stub tables) ----
    // libScePlayGo — everything installed & locus-local.
    Hle::register_fn("ts6GlZOKRrE", (HleFn)s_playgo_init,        "scePlayGoInitialize");
    Hle::register_fn("MPe0EeBGM-E", (HleFn)s_playgo_term,        "scePlayGoTerminate");
    Hle::register_fn("M1Gma1ocrGE", (HleFn)s_playgo_open,        "scePlayGoOpen");
    Hle::register_fn("Uco1I0dlDi8", (HleFn)s_playgo_close,       "scePlayGoClose");
    Hle::register_fn("uWIYLFkkwqk", (HleFn)s_playgo_getlocus,    "scePlayGoGetLocus");
    Hle::register_fn("-RJWNMK3fC8", (HleFn)s_playgo_getprogress, "scePlayGoGetProgress");
    Hle::register_fn("Nn7zKwnA5q0", (HleFn)s_playgo_gettodo,     "scePlayGoGetToDoList");
    Hle::register_fn("gUPGiOQ1tmQ", (HleFn)s_playgo_settodo,     "scePlayGoSetToDoList");
    Hle::register_fn("73fF1MFU8hA", (HleFn)s_playgo_getchunkid,  "scePlayGoGetChunkId");
    Hle::register_fn("v6EZ-YWRdMs", (HleFn)s_playgo_geteta,      "scePlayGoGetEta");
    Hle::register_fn("rvBSfTimejE", (HleFn)s_playgo_getspeed,    "scePlayGoGetInstallSpeed");
    Hle::register_fn("4AAcTU9R3XM", (HleFn)s_ok,                 "scePlayGoSetInstallSpeed");
    Hle::register_fn("3OMbYZBaa50", (HleFn)s_playgo_getlang,     "scePlayGoGetLanguageMask");
    Hle::register_fn("LosLlHOpNqQ", (HleFn)s_ok,                 "scePlayGoSetLanguageMask");
    Hle::register_fn("-Q1-u1a7p0g", (HleFn)s_ok,                 "scePlayGoPrefetch");
    // libSceSaveData (PS5 native surface) — fresh console: mount of a nonexistent save NOT_FOUND.
    Hle::register_fn("TywrFKCoLGY", (HleFn)s_savedata_init3,     "sceSaveDataInitialize3");
    // libSceSaveData "save-data memory" API (#191): a real per-(user,slot) memory block round-trip.
    Hle::register_fn("oQySEUfgXRA", (HleFn)s_savemem_setup, "sceSaveDataSetupSaveDataMemory2");
    Hle::register_fn("cduy9v4YmT4", (HleFn)s_savemem_set,   "sceSaveDataSetSaveDataMemory2");
    Hle::register_fn("QwOO7vegnV8", (HleFn)s_savemem_get,   "sceSaveDataGetSaveDataMemory2");
    Hle::register_fn("wiT9jeC7xPw", (HleFn)s_savemem_sync,  "sceSaveDataSyncSaveDataMemory");
    Hle::register_fn("yKDy8S5yLA0", (HleFn)s_savedata_term,      "sceSaveDataTerminate");
    Hle::register_fn("gjRZNnw0JPE", (HleFn)s_savedata_txres,     "sceSaveDataCreateTransactionResource");
    Hle::register_fn("lJUQuaKqoKY", (HleFn)s_savedata_txres_del, "sceSaveDataDeleteTransactionResource");
    Hle::register_fn("ZP4e7rlzOUk", (HleFn)s_savedata_mount3,    "sceSaveDataMount3");
    Hle::register_fn("uW4vfTwMQVo", (HleFn)s_savedata_umount2,   "sceSaveDataUmount2");
    Hle::register_fn("sDCBrmc61XU", (HleFn)s_savedata_prepare,   "sceSaveDataPrepare");
    Hle::register_fn("ie7qhZ4X0Cc", (HleFn)s_savedata_commit,    "sceSaveDataCommit");
    Hle::register_fn("dyIhnXq-0SM", (HleFn)s_savedata_dirsearch, "sceSaveDataDirNameSearch");
    Hle::register_fn("65VH0Qaaz6s", (HleFn)s_savedata_mountinfo, "sceSaveDataGetMountInfo");  // was MISSING -> garbage free-space
    Hle::register_fn("j8xKtiFj0SY", (HleFn)s_savedata_get_event, "sceSaveDataGetEventResult");
    // libSceNpTrophy2 lifecycle — valid ids; content queries stay "unavailable" (above).
    Hle::register_fn("Bagshr7OQ6Q", (HleFn)s_nptrophy2_createctx,    "sceNpTrophy2CreateContext");
    Hle::register_fn("Gz1rmUZpROM", (HleFn)s_nptrophy2_createhandle, "sceNpTrophy2CreateHandle");
    Hle::register_fn("bIDov3wBu5Q", (HleFn)s_nptrophy2_regctx,       "sceNpTrophy2RegisterContext");
    Hle::register_fn("sUXGfNMalIo", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2RegisterUnlockCallback");
    Hle::register_fn("sysY2FHYff4", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2DestroyContext");
    Hle::register_fn("d8P11CI40KE", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2DestroyHandle");
    Hle::register_fn("fYapWA9xVmA", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2AbortHandle");
    // libSceShare — succeed; sharing simply unavailable headless.
    Hle::register_fn("nBDD66kiFW8", (HleFn)s_share_ok, "sceShareInitialize");
    Hle::register_fn("0IL1keINExQ", (HleFn)s_share_ok, "sceShareTerminate");
    Hle::register_fn("ORspsWDXPps", (HleFn)s_share_ok, "sceShareSetContentParamForApplicationTitle");
    Hle::register_fn("T64o-315wbg", (HleFn)s_share_ok, "sceShareSetScreenshotOverlayImage");
    // libSceErrorDialog — real lifecycle, auto-dismiss (#306). NIDs from the PS5 3.20 stub table
    // (identical to shadPS4's PS4 registrations).
    Hle::register_fn("I88KChlynSs", (HleFn)s_errdialog_init,   "sceErrorDialogInitialize");
    Hle::register_fn("M2ZF-ClLhgY", (HleFn)s_errdialog_open,   "sceErrorDialogOpen");
    Hle::register_fn("jrpnVQfJYgQ", (HleFn)s_errdialog_open,   "sceErrorDialogOpenDetail");
    Hle::register_fn("wktCiyWoDTI", (HleFn)s_errdialog_open,   "sceErrorDialogOpenWithReport");
    Hle::register_fn("ekXHb1kDBl0", (HleFn)s_errdialog_close,  "sceErrorDialogClose");
    Hle::register_fn("9XAxK2PMwk8", (HleFn)s_errdialog_term,   "sceErrorDialogTerminate");
    Hle::register_fn("t2FvHRXzgqk", (HleFn)s_errdialog_status, "sceErrorDialogGetStatus");
    Hle::register_fn("WWiGuh9XfgQ", (HleFn)s_errdialog_status, "sceErrorDialogUpdateStatus");
    // libSceNpEntitlementAccess / libSceGameUpdate — observability (svc_log) with the real-console
    // "local init succeeds offline" return; follow-ups deliberately left unimplemented (see above).
    Hle::register_fn("jO8DM8oyego", (HleFn)s_npent_init,      "sceNpEntitlementAccessInitialize");
    Hle::register_fn("YJtKLttI9fM", (HleFn)s_gameupdate_init, "sceGameUpdateInitialize");
    Hle::register_fn("NSH-C-OmoNI", (HleFn)s_gameupdate_term, "sceGameUpdateTerminate");
    // Post-gate follow-ups (fire from DOLL's now-reachable main menu; NIDs from PS5 3.20 tables).
    Hle::register_fn("TFyU+KFBv54", (HleFn)s_npent_addcont_list,
                     "sceNpEntitlementAccessGetAddcontEntitlementInfoList");
    Hle::register_fn("5LiMEPuW0DQ", (HleFn)s_npent_getkey, "sceNpEntitlementAccessGetEntitlementKey");
    Hle::register_fn("WAzWTZm1H+I", (HleFn)s_savedata_transfermount, "sceSaveDataTransferringMount");
    Hle::register_fn("3RQ5aQfnstU", (HleFn)s_syss_noticeskip, "sceSystemServiceGetNoticeScreenSkipFlag");
    // libSceNpUniversalDataSystem — inert ids (guarded LOW-confidence out-writes).
    Hle::register_fn("sjaobBgqeB4", (HleFn)s_npuds_ok,     "sceNpUniversalDataSystemInitialize");
    Hle::register_fn("5zBnau1uIEo", (HleFn)s_npuds_create, "sceNpUniversalDataSystemCreateContext");
    Hle::register_fn("hT0IAEvN+M0", (HleFn)s_npuds_create, "sceNpUniversalDataSystemCreateHandle");
    Hle::register_fn("tpFJ8LIKvPw", (HleFn)s_npuds_ok,     "sceNpUniversalDataSystemRegisterContext");
    Hle::register_fn("p+GcLqwpL9M", (HleFn)s_npuds_create_event,
                     "sceNpUniversalDataSystemCreateEvent");
    Hle::register_fn("CzkKf7ahIyU", (HleFn)s_npuds_post_event,
                     "sceNpUniversalDataSystemPostEvent");
    Hle::register_fn("wG+84pnNIuo", (HleFn)s_npuds_destroy_event,
                     "sceNpUniversalDataSystemDestroyEvent");
    Hle::register_fn("MfDb+4Nln64", (HleFn)s_npuds_object_set_string,
                     "sceNpUniversalDataSystemEventPropertyObjectSetString");
    Hle::register_fn("m87BHxt-H60", (HleFn)s_gameintent_init,
                     "sceNpGameIntentInitialize");
    Hle::register_fn("xddD23+8TfQ", (HleFn)s_npent_addcont_info,
                     "sceNpEntitlementAccessGetAddcontEntitlementInfo");
    #undef R
}

} // namespace prosper
