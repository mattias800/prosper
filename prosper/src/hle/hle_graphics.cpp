// hle_graphics.cpp — HEADLESS graphics bring-up.
//
// The PS5 GPU stack — libSceAgc (low-level GPU command building) and libSceVideoOut (display /
// flip) — is not yet translated to the host GPU (the eventual AGC->Vulkan + RDNA2 shader
// recompiler work, milestones M4/M5). Until then, these HLE handlers provide *valid but empty*
// graphics objects so the game's own logic (init, main loop, simulation) executes headlessly.
// This lets us map the real graphics call sequence to translate, and prove the runtime runs.
//
// This is NOT rendering and NOT faking visible output: object getters return zeroed buffers the
// guest can read/write without faulting; action/submit calls are no-ops. Every entry here is a
// placeholder to be replaced by a real translation. Graphics functions live in sparsely-documented
// Sony libs, so they're registered by raw NID with a note on the observed role.
#include "dispatch.hpp"
#include "nid.hpp"
#include "../host/boot_program.hpp"   // #1659: shared guest-module labelling
#include "host/exec_image.hpp"
#include "gpu/videoout_present.hpp"
#include "gpu/gpu_execute.hpp"      // guest_readable (safe pointer probe for the diagnostic dumps)
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>
#include <unordered_set>
#include <utility>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define HLE7(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6)
#define HLE8(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6, \
                                       uint64_t a7)

extern "C" uint64_t prosper_guest_tsc_ns(); // shared sceKernelReadTsc/process-time clock

namespace {
// Optional fork-safe counter bumped when the guest calls into the graphics libs (libSceAgc /
// libSceVideoOut). Tests use it to assert the boot advanced all the way into GPU/display init — a
// regression guard for the entire runtime bring-up (loader → IL2CPP → C# startup → services →
// graphics). Forward-compatible: these calls keep happening as we implement more, so the guard
// stays valid even once deeper blockers (e.g. the locale-facet fault) are fixed.
volatile int* g_gfx_counter = nullptr;
inline void gfx_tick() { if (g_gfx_counter) *g_gfx_counter = *g_gfx_counter + 1; }  // avoid -Wvolatile
}

void set_gfx_call_counter(volatile int* counter) { g_gfx_counter = counter; }

// sceAgcGraphicsGetRegisterDefaults2 / ...Internal (NIDs 2JtWUUiYBXs / wRbq6ZjNop4). Per Kyty
// (Graphics.cpp:766, 1307), these return a RegisterDefaults table the game reads to seed GPU register
// state. Layout: tbl0/1/2/3 (ShaderRegister** arrays), unknown[2], index table, count@0x38. Returning
// an empty (count=0) table gave the game nothing to build its internal register-offset table from, so
// Unity's Gen5 init left that table null and a later register lookup null-derefed (eboot+0x3b5ea6).
// We now return the REAL default tables (127 cx+sh+uc registers), vendored from Kyty in
// agc_reg_defaults.cpp, so the game can build a populated register-offset table.
extern "C" void* prosper_agc_reg_defaults(unsigned int ver);           // public defaults (versioned keys)
extern "C" void* prosper_agc_reg_defaults_internal(unsigned int ver);  // internal defaults
HLE(g_agc_regdefs)     { gfx_tick(); return (uint64_t)(uintptr_t)prosper_agc_reg_defaults((unsigned)a0); }
HLE(g_agc_regdefs_int) { gfx_tick(); return (uint64_t)(uintptr_t)prosper_agc_reg_defaults_internal((unsigned)a0); }

// --- libSceVideoOut (display / frame presentation). ------------------------------------------
// Models a single connected 1080p60 display. Query functions return real, self-consistent values
// (not zeroed stubs) so the game's display setup — resolution query, output configuration, buffer
// registration — sees a normal main-bus HDMI output. Struct layouts are the Orbis/Gen5 VideoOut
// ABI (cross-checked vs Kyty Graphics/VideoOut.cpp + shadPS4 videoout/video_out.h). Frame
// presentation is still simulated (no swapchain yet); the swapchain lands behind these once a real
// window/surface exists. All output-struct writes are size-exact — over-writing smashes the guest's
// stack canary (cf. the historical f_fstat bug).
namespace {
    constexpr uint64_t kVoErrorInvalidHandle =
        (uint64_t)(int64_t)(int32_t)0x8029000b;  // SCE_VIDEO_OUT_ERROR_INVALID_HANDLE
    int32_t g_vo_handle = 0;
    std::mutex g_vo_handle_mx;
    std::unordered_set<int32_t> g_vo_handles;

    // Keep a live handle pinned until the HLE operation finishes. This makes validation and close
    // atomic with respect to each other: a concurrent Close cannot retire a handle after a caller
    // validates it but before that caller mutates flip, event, or buffer-registration state.
    class VideoOutHandleGuard {
    public:
        explicit VideoOutHandleGuard(uint64_t raw_handle)
            : lock_(g_vo_handle_mx),
              valid_(g_vo_handles.find((int32_t)raw_handle) != g_vo_handles.end()) {}
        bool valid() const { return valid_; }

    private:
        std::unique_lock<std::mutex> lock_;
        bool valid_;
    };
    // Flip bookkeeping (#82): completion status is written by TWO paths — SubmitFlip on
    // any guest thread and prosper_vo_flip_from_gpu on the submit thread — and read by GetFlipStatus
    // on Unity's pacer thread. They must move as ONE unit under a mutex: an unsynchronized reader
    // could observe the count advanced but a stale flipArg/buffer (torn triple), making the pacer
    // believe a flip completed with the wrong arg; concurrent flips could lose an increment.
    std::mutex g_flip_mx;
    uint64_t g_flip_count = 0;      // incremented per flip so GetFlipStatus shows progress
    int32_t  g_flip_rate = 0;       // 0=60 Hz, 1=30 Hz, 2=20 Hz (stored for future pacing)
    // The VideoOut ABI uses -1 for both fields until the first completed flip. Zero is a valid
    // buffer index/argument, so exposing zero early can fool a frame pacer into consuming a flip
    // that never happened (#394 F3).
    int32_t  g_current_buffer = -1;
    int64_t  g_last_flip_arg = -1;
    uint64_t g_last_flip_process_time = 0; // microseconds, like sceKernelGetProcessTime
    uint64_t g_last_flip_tsc = 0;          // 1 GHz guest counter, like sceKernelReadTsc
    uint64_t g_last_submit_tsc = 0;
    // sceVideoOutGetBufferLabelAddress exposes this contiguous per-port array directly to the
    // guest. The guest writes 1 when an EOP flip is ready; completing the next flip releases the
    // previously displayed slot back to 0. Keep the storage stable for the process lifetime.
    alignas(uint64_t) uint64_t g_buffer_labels[16] = {};
    int32_t g_previous_buffer = -1;
    void flip_advance(int32_t bufidx, int64_t flip_arg) {
        // The renderer completes flips synchronously, so submit and completion share one instant.
        // Sample before taking g_flip_mx: the deterministic guest clock reads the flip count.
        const uint64_t tsc = prosper_guest_tsc_ns();
        std::lock_guard<std::mutex> lk(g_flip_mx);
        if (g_previous_buffer >= 0 && g_previous_buffer < 16)
            g_buffer_labels[g_previous_buffer] = 0;
        g_previous_buffer = (bufidx >= 0 && bufidx < 16) ? bufidx : -1;
        g_flip_count++;
        g_current_buffer = bufidx;
        g_last_flip_arg = flip_arg;
        // Concurrent submitters can acquire the mutex in the opposite order to their clock samples.
        // Keep the published completion timeline monotonic in status/count order.
        const uint64_t completion_tsc = tsc < g_last_flip_tsc ? g_last_flip_tsc : tsc;
        g_last_flip_process_time = completion_tsc / 1000;
        g_last_flip_tsc = completion_tsc;
        g_last_submit_tsc = completion_tsc;
    }

    // The one display we advertise. 1920x1080 @ 59.94Hz (refresh-rate enum 3), 16:9, ~50".
    constexpr uint32_t kDispW = 1920, kDispH = 1080;
    constexpr uint64_t kRefresh5994 = 3;   // SCE_VIDEO_OUT_REFRESH_RATE_59_94HZ

    void vo_argtrace(const char* fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
        if (getenv("PROSPER_GFXLOG"))
            fprintf(stderr, "[vo] %s a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx\n", fn,
                    (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                    (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    }

    // Swapchain scaffolding: the set of display buffers the game registers via RegisterBuffers2.
    // These GPU addresses are the framebuffers the game flips between (triple-buffered here). The
    // back-half present path turns each into a swapchain image; SubmitFlip picks by buffer index.
    // Recorded now so that surface is ready even before real presentation exists.
    struct DisplayConfig {
        struct SetConfig {
            uint32_t width = 0, height = 0;
            uint64_t pixel_format = 0;
            uint32_t tiling_mode = 0;
            bool registered = false;
        } sets[4];
        uint32_t width = 0, height = 0;
        uint64_t pixel_format = 0;
        uint32_t tiling_mode = 0;
        int      buffer_num = 0;
        uint64_t buffer_addr[16] = {0};   // guest GPU-VA of each registered framebuffer
        uint8_t  buffer_set[16] = {0};    // set_index + 1; zero means the slot is unregistered
        uint64_t buffer_generation[16] = {0};
        uint64_t next_generation = 0;
        int      front_index = -1;
        uint64_t front_generation = 0;
        bool     configured = false;
    };
    DisplayConfig g_display;
    std::mutex g_display_mx;
    struct VideoOutBuffers { const void* data; const void* metadata; const void* reserved[2]; };
}

// Exposed for the back-half present path + tests: the registered display surface.
// prosper_vo_flip_count: total flips so far (either flip path) — read by the PROSPER_PROGRESS
// heartbeat in hle_agc.cpp as a cheap forward-progress signal for long diagnostic runs.
extern "C" uint64_t prosper_vo_flip_count() { std::lock_guard<std::mutex> lk(g_flip_mx); return g_flip_count; }
extern "C" int prosper_vo_flip_rate() { std::lock_guard<std::mutex> lk(g_flip_mx); return g_flip_rate; }
extern "C" int prosper_vo_buffer_count() {
    std::lock_guard<std::mutex> lk(g_display_mx); return g_display.buffer_num;
}
extern "C" uint32_t prosper_vo_display_width() {
    std::lock_guard<std::mutex> lk(g_display_mx); return g_display.width;
}
extern "C" uint32_t prosper_vo_display_height() {
    std::lock_guard<std::mutex> lk(g_display_mx); return g_display.height;
}
extern "C" uint64_t prosper_vo_display_format() {
    std::lock_guard<std::mutex> lk(g_display_mx); return g_display.pixel_format;
}
extern "C" uint64_t prosper_vo_buffer_addr(int i) {
    std::lock_guard<std::mutex> lk(g_display_mx);
    return (i >= 0 && i < g_display.buffer_num) ? g_display.buffer_addr[i] : 0;
}

namespace {
bool videoout_buffer_snapshot_locked(int buffer_index, VideoOutBufferSnapshot& out) {
    out = {};
    if (buffer_index < 0 || buffer_index >= 16) return false;
    const uint8_t tag = g_display.buffer_set[buffer_index];
    if (tag == 0) return false;
    const auto& config = g_display.sets[tag - 1];
    if (!config.registered) return false;
    out.address = g_display.buffer_addr[buffer_index];
    out.pixel_format = config.pixel_format;
    out.generation = g_display.buffer_generation[buffer_index];
    out.buffer_index = buffer_index;
    out.width = config.width;
    out.height = config.height;
    out.tiling_mode = config.tiling_mode;
    return true;
}

bool videoout_front_snapshot_locked(VideoOutBufferSnapshot& out) {
    if (!videoout_buffer_snapshot_locked(g_display.front_index, out) ||
        out.generation != g_display.front_generation) {
        out = {};
        return false;
    }
    return true;
}

bool videoout_buffer_bytes(const VideoOutBufferSnapshot& buffer, size_t& bytes) {
    if (!buffer.address || !buffer.width || !buffer.height) return false;
    const uint64_t pixels = (uint64_t)buffer.width * buffer.height;
    if (pixels > SIZE_MAX / 4) return false;
    bytes = (size_t)pixels * 4;
    return true;
}
} // namespace

bool videoout_select_buffer(int buffer_index, VideoOutBufferSnapshot& out) {
    std::lock_guard<std::mutex> lk(g_display_mx);
    if (!videoout_buffer_snapshot_locked(buffer_index, out)) return false;
    g_display.front_index = buffer_index;
    g_display.front_generation = out.generation;
    return true;
}

bool videoout_front_snapshot(VideoOutBufferSnapshot& out) {
    std::lock_guard<std::mutex> lk(g_display_mx);
    return videoout_front_snapshot_locked(out);
}

bool videoout_display_snapshot(VideoOutBufferSnapshot& out) {
    out = {};
    std::lock_guard<std::mutex> lk(g_display_mx);
    if (!g_display.configured) return false;
    out.pixel_format = g_display.pixel_format;
    out.width = g_display.width;
    out.height = g_display.height;
    out.tiling_mode = g_display.tiling_mode;
    return true;
}

int videoout_front_index() {
    std::lock_guard<std::mutex> lk(g_display_mx);
    VideoOutBufferSnapshot out;
    return videoout_front_snapshot_locked(out) ? out.buffer_index : -1;
}

void videoout_reset_front() {
    std::lock_guard<std::mutex> lk(g_display_mx);
    g_display.front_index = -1;
    g_display.front_generation = 0;
}

bool videoout_copy_buffer(const VideoOutBufferSnapshot& expected, std::vector<uint8_t>& out) {
    std::lock_guard<std::mutex> lk(g_display_mx);
    VideoOutBufferSnapshot current;
    if (!videoout_buffer_snapshot_locked(expected.buffer_index, current) ||
        current.generation != expected.generation) return false;
    size_t bytes = 0;
    if (!videoout_buffer_bytes(current, bytes)) return false;
    out.resize(bytes);
    std::memcpy(out.data(), reinterpret_cast<const void*>((uintptr_t)current.address), bytes);
    return true;
}

bool videoout_copy_front_buffer(std::vector<uint8_t>& out, VideoOutBufferSnapshot& metadata) {
    std::lock_guard<std::mutex> lk(g_display_mx);
    if (!videoout_front_snapshot_locked(metadata)) return false;
    size_t bytes = 0;
    if (!videoout_buffer_bytes(metadata, bytes)) return false;
    out.resize(bytes);
    std::memcpy(out.data(), reinterpret_cast<const void*>((uintptr_t)metadata.address), bytes);
    return true;
}

size_t videoout_copy_front_buffer(void* dst, size_t dst_cap, VideoOutBufferSnapshot* metadata) {
    if (!dst) return 0;
    std::lock_guard<std::mutex> lk(g_display_mx);
    VideoOutBufferSnapshot current;
    if (!videoout_front_snapshot_locked(current)) return 0;
    size_t bytes = 0;
    if (!videoout_buffer_bytes(current, bytes) || dst_cap < bytes) return 0;
    std::memcpy(dst, reinterpret_cast<const void*>((uintptr_t)current.address), bytes);
    if (metadata) *metadata = current;
    return bytes;
}
namespace { bool evlog() { static int v = getenv("PROSPER_EVLOG") ? 1 : 0; return v; } }
// Implemented in hle_kernel_time.cpp (the equeue backend). Register a flip/vblank event source so the
// ~60 Hz pump posts events into the given equeue; the game's WaitEqueue then returns them.
void prosper_eq_add_flip(uint64_t eq, int64_t ident, uint64_t udata);
void prosper_eq_add_vblank(uint64_t eq, int64_t ident, uint64_t udata);
void prosper_eq_add_eop(uint64_t eq, int64_t id, uint64_t udata);
void prosper_eq_trigger_flip(int64_t flip_arg);   // fire flip-completion events (data=flipArg)
// sceGnmAddEqEvent / GraphicsAddEqEvent (NID b0xyllnVY-I): register a GPU EOP/compute-ring completion
// event source on an equeue. Mirrors shadPS4 sceGnmAddEqEvent (id=GfxEop=0x40 for gfx). The submit path
// (hle_agc.cpp) fires prosper_eq_trigger_eop() on completion. NOTE: our target (The Messenger) never
// calls this — it uses the flip path + sync_on_address semaphores — so this is inert for it, but it is
// the correct, ref-validated behavior for the general case and lets test_equeue_events verify the path.
HLE(g_gnm_add_eq_event) {   // (eq, id, udata)
    if (evlog()) fprintf(stderr, "[ev] GnmAddEqEvent eq=0x%llx id=0x%llx udata=0x%llx\n",
        (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
    prosper_eq_add_eop(a0, (int64_t)a1, a2);
    return 0;
}
// sceAgcDriverAddEqEvent (NID w2rJhmD+dsE) — the libSceAgc DRIVER path's GPU-completion event
// registration (the Agc analogue of sceGnmAddEqEvent; this title uses the Agc path, not Gnm — the
// note above about GnmAddEqEvent being inert for The Messenger holds, but the game DOES register its
// completion source through THIS call, on its UnityFTMFlipQueue + "EOP QUEUE"). It was a trace-only
// stub, so no source was ever registered and those queues starved (the FTM dispatcher polled an empty
// queue forever). Wire it to the same EOP backend that agc_driver_submit_dcb fires on each submit.
// Diagnosis + fix from PR #29. CONFIDENCE: MED — a0=equeue confirmed by the equeue-name trace; the
// (id, udata) arg mapping mirrors the Gnm variant.
HLE(g_agc_add_eq_event) {   // (eq, id, udata, ...)
    if (evlog()) fprintf(stderr, "[ev] AgcDriverAddEqEvent eq=0x%llx id=0x%llx udata=0x%llx\n",
        (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
    prosper_eq_add_eop(a0, (int64_t)a1, a2);
    return 0;
}
HLE(g_vo_open) {
    gfx_tick();
    std::lock_guard<std::mutex> lk(g_vo_handle_mx);
    const int32_t handle = ++g_vo_handle + 0x1000;
    g_vo_handles.insert(handle);
    return (uint64_t)(int64_t)handle;
}
HLE(g_vo_close) {
    std::lock_guard<std::mutex> lk(g_vo_handle_mx);
    if (g_vo_handles.erase((int32_t)a0) != 1) return kVoErrorInvalidHandle;
    return 0;
}
// sceVideoOutAddFlipEvent(eq, handle, udata): register a flip-completion event source on an equeue.
HLE(g_vo_addflipevent) {
    VideoOutHandleGuard handle(a1);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    if (evlog()) fprintf(stderr, "[ev] AddFlipEvent eq=0x%llx handle=0x%llx udata=0x%llx\n",
        (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
    prosper_eq_add_flip(a0, (int64_t)(int32_t)a1, a2);
    return 0;
}
HLE(g_vo_addvblankevent) {
    VideoOutHandleGuard handle(a1);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    if (evlog()) fprintf(stderr, "[ev] AddVblankEvent eq=0x%llx handle=0x%llx udata=0x%llx\n",
        (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
    prosper_eq_add_vblank(a0, (int64_t)(int32_t)a1, a2);
    return 0;
}
// sceVideoOutGetBufferLabelAddress (OcQybQejHEY): return the stable array whose 16 u64 entries
// coordinate EOP flip ownership between the guest and VideoOut completion path.
HLE(g_vo_get_buffer_label_address) {
    if (!a1)
        return (uint64_t)(int64_t)(int32_t)0x80290002;  // SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    *(uintptr_t*)(uintptr_t)a1 = (uintptr_t)g_buffer_labels;
    return 16;
}
// sceVideoOutSetFlipRate (CBiu4mCE1DA): remember the port's requested divisor. Presentation is
// synchronous today, but callers must still observe the documented validation/state contract.
HLE(g_vo_set_flip_rate) {
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    const int32_t rate = (int32_t)a1;
    if (rate < 0 || rate > 2)
        return (uint64_t)(int64_t)(int32_t)0x80290001;  // SCE_VIDEO_OUT_ERROR_INVALID_VALUE
    std::lock_guard<std::mutex> lk(g_flip_mx);
    g_flip_rate = rate;
    return 0;
}
HLE(g_vo_submitflip)  {
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    if (evlog()) fprintf(stderr, "[ev] SubmitFlip handle=0x%llx bufidx=%lld flipmode=0x%llx fl013arg=0x%llx\n",
        (unsigned long long)a0, (long long)(int32_t)a1, (unsigned long long)a2, (unsigned long long)a3);
    const int32_t buffer_index = (int32_t)a1;
    if (buffer_index < -1 || buffer_index > 15)
        return (uint64_t)(int64_t)(int32_t)0x8029000a;  // SCE_VIDEO_OUT_ERROR_INVALID_INDEX
    flip_advance(buffer_index, (int64_t)a3);
    gpu::present_flip(buffer_index, (int64_t)a3);   // present the buffer (scanout front + count)
    prosper_eq_trigger_flip((int64_t)a3);   // flip completed (synchronous): fire the flip event
    return 0;
}
// The GPU-side flip: an in-stream SetFlip packet (agc_dcb_set_flip) processed by the CommandProcessor.
// This is how the game actually flips each frame — it queues the flip in the Dcb rather than calling
// sceVideoOutSubmitFlip. The GPU reaching the packet IS the flip moment (our fold is synchronous), and
// it must advance the SAME flip status (count/flipArg/currentBuffer) that GetFlipStatus reports:
// Unity's frame pacer polls for its submitted flipArg to complete before building the next frame, so
// dropping this packet stalls the game at one rendered frame forever.
extern "C" void prosper_vo_flip_from_gpu(uint32_t handle, int32_t bufidx, uint32_t flip_mode, int64_t flip_arg) {
    VideoOutHandleGuard live_handle(handle);
    if (!live_handle.valid()) return;
    if (evlog()) fprintf(stderr, "[ev] GpuFlip handle=0x%x bufidx=%d mode=0x%x fliparg=0x%llx\n",
                         handle, bufidx, flip_mode, (unsigned long long)flip_arg);
    flip_advance(bufidx, flip_arg);
    gpu::present_flip(bufidx, flip_arg);   // scanout bookkeeping, same as the API flip
    prosper_eq_trigger_flip(flip_arg);     // flip completed (synchronous): fire the flip event
}
HLE(g_vo_flippending) {
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    if (evlog()) fprintf(stderr, "[ev] IsFlipPending\n");
    return 0;  // never pending in the synchronous present model
}
HLE(g_vo_flipstatus)  { // (handle, SceVideoOutFlipStatus* status): report our simulated flip count.
    // This title uses the legacy 0x40-byte SceVideoOutFlipStatus ABI; the newer native PS5 0x80
    // object is a distinct generation. Public PS5 compatibility code plus the captured caller pin
    // these offsets. Writing more than 0x40 smashes the caller's stack canary. CONFIDENCE: HIGH.
    if (!a1)
        return (uint64_t)(int64_t)(int32_t)0x80290002;  // SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    uint8_t* s = (uint8_t*)(uintptr_t)a1; memset(s, 0, 0x40);
    uint64_t cnt, process_time, tsc, submit_tsc;
    int64_t arg;
    int32_t buf;
    { std::lock_guard<std::mutex> lk(g_flip_mx);   // one coherent completion snapshot
      cnt = g_flip_count; process_time = g_last_flip_process_time; tsc = g_last_flip_tsc;
      arg = g_last_flip_arg; submit_tsc = g_last_submit_tsc; buf = g_current_buffer; }
    *(uint64_t*)(s + 0x00) = cnt;          // count
    *(uint64_t*)(s + 0x08) = process_time; // processTime (microseconds)
    *(uint64_t*)(s + 0x10) = tsc;          // tsc (shared 1 GHz guest clock)
    *(int64_t*) (s + 0x18) = arg;          // flipArg
    *(uint64_t*)(s + 0x20) = submit_tsc;   // submitTsc (same instant: synchronous completion)
    // reserved0@0x28, gcQueueNum@0x30, and flipPendingNum@0x34 stay zero: no queued work.
    *(int32_t*) (s + 0x38) = buf;          // currentBuffer
    return 0;
}
// SceVideoOutResolutionStatus (0x30 bytes): report a real 1080p60 panel instead of the previous
// all-zero display, and initialize flags/reserved0/reserved1 rather than leaking caller garbage.
HLE(g_vo_resstatus)   {
    if (!a1)
        return (uint64_t)(int64_t)(int32_t)0x80290002;  // SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    uint8_t* s = (uint8_t*)(uintptr_t)a1; memset(s, 0, 0x30);
    *(uint32_t*)(s + 0x00) = kDispW;   *(uint32_t*)(s + 0x04) = kDispH;   // full w/h
    *(uint32_t*)(s + 0x08) = kDispW;   *(uint32_t*)(s + 0x0c) = kDispH;   // pane w/h
    *(uint64_t*)(s + 0x10) = kRefresh5994;
    *(float*)   (s + 0x18) = 50.0f;                                      // screen size (inch)
    return 0;
}

// The monotonic ~59.94 Hz timebase shared by sceVideoOutGetVblankStatus's `count` and
// sceVideoOutWaitVblank's wake-up boundary. They MUST share it in PHASE, not merely in period: a
// game that waits for a vblank and then reads the count has to observe the tick it just waited for,
// which is a statement about where the boundaries fall, not about how far apart they are. Anchored
// on first use so both are process-relative, exactly as the previous status-only anchor was.
//
// It is NOT the only vblank clock in the process, and this comment previously claimed it was.
// hle_kernel_time.cpp's `vblank_pump` (:731) posts the VideoOut vblank kevent from its own
// `nanosleep(16666667)` loop — 60.000 Hz, its own phase, started when the first equeue source is
// registered. So the kevent stream and this grid drift by roughly one tick per second and their
// boundaries do not coincide. Aligning them is a real follow-up; asserting they are aligned is
// wrong, and a title that cross-checks the two would see the disagreement.
// Internal linkage on purpose: the "epoch before now" ordering invariant below is only
// enforceable by reading this one file, so nothing outside it may reach these.
namespace {
constexpr uint64_t kVblankNs = 16683350;             // 59.94 Hz period
uint64_t vblank_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
uint64_t vblank_epoch_ns() { static const uint64_t t0 = vblank_now_ns(); return t0; }
}  // namespace

// sceVideoOutGetVblankStatus (SceVideoOutVblankStatus, 0x28 bytes — Kyty VideoOut.cpp:94:
// u64 count @0, u64 processTime @8, u64 tsc @0x10, u64 reserved @0x18, u8 flags @0x20).
// The count is derived from a monotonic ~59.94 Hz timebase — the previous ++g_vblank_count
// advanced once per CALL, so a poll loop saw a "vblank" on every poll (uncapped pacing) and the
// counter disagreed arbitrarily with the real 60 Hz equeue vblank pump.
HLE(g_vo_vblankstatus) {
    if (!a1)
        return (uint64_t)(int64_t)(int32_t)0x80290002;  // SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    uint8_t* s = (uint8_t*)(uintptr_t)a1; memset(s, 0, 0x28);
    // Anchor BEFORE sampling: on the very first call the epoch IS this instant, and sampling `ns`
    // first would make t0 the later of the two and underflow (ns - t0).
    const uint64_t t0 = vblank_epoch_ns();               // process-relative, first-use anchored
    const uint64_t ns = vblank_now_ns();
    // A live VideoOut handle has already observed at least one vblank. Returning zero on
    // the first query made Astro Bot compute a frame-rate sample as
    // `(frames * refresh) / (count - previous_count)` with both counts zero and trap on
    // IDIV. Keep the time-based behavior, but number the first process-relative vblank 1.
    *(uint64_t*)(s + 0x00) = 1 + (ns - t0) / kVblankNs;  // count: one tick per vblank period
    *(uint64_t*)(s + 0x08) = (ns - t0) / 1000;           // processTime (µs)
    *(uint64_t*)(s + 0x10) = ns;                         // tsc (monotonic ns; matches ReadTsc's unit)
    return 0;
}
// sceVideoOutWaitVblank (j6RaAUlaLv0): block the calling thread until the port's next vertical
// blank, then return 0. This is the CPU-side half of display pacing — the counterpart to the
// vblank kevent source — and titles use it as the whole body of a display/present thread's loop.
//
// It was unregistered, so it fell to the generic unimplemented stub and returned IMMEDIATELY. That
// is not a harmless no-op: the caller is a `for(;;) { WaitVblank(port); ...; }` thread, so instead
// of waking 59.94 times a second it spins one host core flat out and the work it paces (movie
// decode, present, frame timing) runs unbounded. Found on PPSA26414 (R-Type Delta: HD Boosted,
// #1746), whose display thread is exactly that loop.
//
// The wait boundary comes from vblank_epoch_ns()/kVblankNs — the SAME timebase
// sceVideoOutGetVblankStatus reports — so a thread that waits for a vblank and then reads the
// status observes the tick it waited for, not a disagreeing second clock. The handle is validated
// under g_vo_handle_mx and the guard is released BEFORE sleeping: holding it across a ~16 ms sleep
// would serialize every other VideoOut call in the process behind one waiter.
// CONFIDENCE: HIGH that blocking is the behaviour and that returning instantly is wrong, and HIGH
// on the shape (one argument, 0 on success). MED that the argument is an sceVideoOutOpen handle
// rather than a port index: the guest's call site passes exactly one register, but no title has yet
// been observed reaching this call with a handle we can compare against. That matters because the
// reject arm is NEW behaviour — the previous stub returned 0 for everything — so it is made
// fail-visible rather than silent: a title passing something we do not recognise would otherwise
// turn a paced loop into an error spin with nothing printed, which is worse than the original spin.
HLE(g_vo_wait_vblank) {
    { VideoOutHandleGuard handle(a0);
      if (!handle.valid()) {
          static std::atomic<bool> warned{false};
          if (!warned.exchange(true))
              fprintf(stderr, "[vo] sceVideoOutWaitVblank: unknown handle 0x%llx — returning "
                              "INVALID_HANDLE and NOT waiting. If this repeats, the argument is "
                              "probably not an sceVideoOutOpen handle (see the CONFIDENCE note).\n",
                      (unsigned long long)a0);
          return kVoErrorInvalidHandle;
      } }
    const uint64_t t0 = vblank_epoch_ns();
    const uint64_t now = vblank_now_ns();
    // Next strictly-future boundary: a caller that is already exactly on one still waits a full
    // period, which is what "wait for the NEXT vblank" means.
    const uint64_t next = t0 + ((now - t0) / kVblankNs + 1) * kVblankNs;
    // F5: duration_cast rather than a nanoseconds->time_point construction, which only compiles
    // because steady_clock::duration happens to be nanoseconds on every toolchain we build with.
    std::this_thread::sleep_until(std::chrono::steady_clock::time_point(
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::nanoseconds(next))));
    return 0;
}

// PROSPER_HDR (the frontends' --hdr flag): advertise an HDR-capable display. Default is SDR —
// the common case for prosper users, and the mode whose output path titles tonemap themselves.
// Deliberately uncached: capability queries are boot-time-rare, and tests exercise both modes in
// one process.
static bool prosper_hdr_output() { return getenv("PROSPER_HDR") != nullptr; }

// sceVideoOutGetDeviceCapabilityInfo (SceVideoOutDeviceCapabilityInfo: single u64 capability).
// Default: a plain SDR display (no HDR/BT2020) — capability 0. With PROSPER_HDR, set the
// BT2020/PQ capability bit (0x2 — the Orbis-era SCE_VIDEO_OUT_DEVICE_CAPABILITY_BT2020_PQ value;
// secondary-implementation agreement only, no Gen5 title has been observed reading a different
// bit yet). CONFIDENCE: MED on the exact bit; HIGH that 0 == SDR-only.
HLE(g_vo_devcap) {
    if (!a1)
        return (uint64_t)(int64_t)(int32_t)0x80290002;  // SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    *(uint64_t*)(uintptr_t)a1 = prosper_hdr_output() ? 0x2ull : 0ull;
    return 0;
}

// sceVideoOutIsOutputSupported (Nv8c-Kb+DUM): (handle, mode/feature) -> bool. A standard main-bus
// output supports the queried mode → 1.
HLE(g_vo_is_output_supported) {
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    vo_argtrace("IsOutputSupported", a0,a1,a2,a3,a4,a5);
    return 1;
}

// sceVideoOutSetBufferAttribute (i6-sR91Wt-4): initialize the legacy/Gen4 BufferAttribute from
// (attr, pixel_format, tiling_mode, aspect_ratio, width, height, pitch_in_pixel). The ABI object is
// 0x28 bytes: seven u32 fields followed by reserved u32/u64 storage. Size-exact initialization keeps
// reserved bytes deterministic without touching the caller's following object.
HLE7(g_vo_set_buffer_attribute) {
    if (!a0) return 0;
    uint8_t* p = (uint8_t*)(uintptr_t)a0;
    memset(p, 0, 0x28);
    *(uint32_t*)(p + 0x00) = (uint32_t)a1;   // pixel_format
    *(uint32_t*)(p + 0x04) = (uint32_t)a2;   // tiling_mode
    *(uint32_t*)(p + 0x08) = (uint32_t)a3;   // aspect_ratio
    *(uint32_t*)(p + 0x0c) = (uint32_t)a4;   // width
    *(uint32_t*)(p + 0x10) = (uint32_t)a5;   // height
    *(uint32_t*)(p + 0x14) = (uint32_t)a6;   // pitch_in_pixel
    return 0;
}

// sceVideoOutSetBufferAttribute2 (PjS5uASwcV8): fill the caller's VideoOutBufferAttribute2 (0x50
// bytes, Kyty layout) from (attr, pixel_format, tiling_mode, width, height, option, dcc_control,
// dcc_cb_clear). The final two arguments arrive in the guest's stack slots. Mirrors Kyty's setter.
// Size-exact write.
HLE8(g_vo_set_buffer_attribute2) {
    vo_argtrace("SetBufferAttribute2", a0,a1,a2,a3,a4,a5);
    if (!a0) return 0;
    uint8_t* p = (uint8_t*)(uintptr_t)a0; memset(p, 0, 0x50);
    *(uint32_t*)(p + 0x04) = (uint32_t)a2;   // tiling_mode
    *(uint32_t*)(p + 0x08) = 0;              // aspect_ratio (16:9)
    *(uint32_t*)(p + 0x0c) = (uint32_t)a3;   // width
    *(uint32_t*)(p + 0x10) = (uint32_t)a4;   // height
    *(uint64_t*)(p + 0x18) = a5;             // option
    *(uint64_t*)(p + 0x20) = a1;             // pixel_format
    *(uint64_t*)(p + 0x28) = a7;             // dcc_cb_register_clear_color
    *(uint32_t*)(p + 0x30) = (uint32_t)a6;   // dcc_control
    return 0;
}

// sceVideoOutRegisterBuffers (w3BY+tAEiQY): register a legacy flat framebuffer-address array and
// return the lowest free attribute-group index. The Gen4 attribute stores its 32-bit pixel format at
// offset 0; the remaining surface fields share the Gen5 offsets used by the present registry.
HLE(g_vo_register_buffers) {  // a0=handle a1=start a2=addresses a3=buffer_num a4=attribute
    vo_argtrace("RegisterBuffers", a0,a1,a2,a3,a4,a5);
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    const int start = (int)a1, num = (int)a3;
    if (!a2 || !a4)
        return (uint64_t)(int64_t)(int32_t)0x80290002;  // SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS
    if (start < 0 || start > 15 || num < 1 || num > 16 || start + num > 16)
        return (uint64_t)(int64_t)(int32_t)0x80290001;  // SCE_VIDEO_OUT_ERROR_INVALID_VALUE

    const uint8_t* attr = (const uint8_t*)(uintptr_t)a4;
    const DisplayConfig::SetConfig config = {
        *(const uint32_t*)(attr + 0x0c), *(const uint32_t*)(attr + 0x10),
        *(const uint32_t*)(attr + 0x00), *(const uint32_t*)(attr + 0x04), true
    };
    const auto* addresses = (const void* const*)(uintptr_t)a2;
    std::lock_guard<std::mutex> lk(g_display_mx);

    int set = 0;
    while (set < 4 && g_display.sets[set].registered) ++set;
    if (set == 4)
        return (uint64_t)(int64_t)(int32_t)0x8029000f;  // SCE_VIDEO_OUT_ERROR_NO_EMPTY_SLOT
    for (int i = 0; i < num; ++i) {
        if (g_display.buffer_set[start + i] != 0)
            return (uint64_t)(int64_t)(int32_t)0x80290010;  // SCE_VIDEO_OUT_ERROR_SLOT_OCCUPIED
    }

    g_display.width = config.width;
    g_display.height = config.height;
    g_display.pixel_format = config.pixel_format;
    g_display.tiling_mode = config.tiling_mode;
    g_display.sets[set] = config;
    {
        std::lock_guard<std::mutex> flip_lk(g_flip_mx);
        for (int i = 0; i < num; ++i) g_buffer_labels[start + i] = 0;
    }
    for (int i = 0; i < num; ++i) {
        g_display.buffer_addr[start + i] = (uint64_t)(uintptr_t)addresses[i];
        g_display.buffer_set[start + i] = (uint8_t)(set + 1);
        uint64_t generation = ++g_display.next_generation;
        if (generation == 0) generation = ++g_display.next_generation;
        g_display.buffer_generation[start + i] = generation;
    }
    if (g_display.buffer_num < start + num) g_display.buffer_num = start + num;
    g_display.configured = true;
    return (uint64_t)set;
}

// sceVideoOutRegisterBuffers2 (rKBUtgRrtbk): (handle, set_index, buffer_index_start, buffers,
// buffer_num, attribute, [category, option on stack]). Validate ranges like Kyty and accept. The
// buffers' GPU backing becomes swapchain images once the swapchain exists; for now, registration
// succeeds so the game proceeds to flips.
HLE(g_vo_register_buffers2) {  // a0=handle a1=set_index a2=buffer_index_start a3=buffers a4=buffer_num a5=attribute
    vo_argtrace("RegisterBuffers2", a0,a1,a2,a3,a4,a5);
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    int set = (int)a1, start = (int)a2, num = (int)a4;
    if (!a3 || !a5) return (uint64_t)(int64_t)-1;
    if (set < 0 || set > 3 || start < 0 || start > 15 || num < 1 || num > 16 || start + num > 16)
        return (uint64_t)(int64_t)-1;
    // Record the display surface (swapchain scaffolding). attribute is the 0x50-byte
    // VideoOutBufferAttribute2 we fill in SetBufferAttribute2: width@0x0c, height@0x10, format@0x20.
    const uint8_t* attr = (const uint8_t*)(uintptr_t)a5;
    const DisplayConfig::SetConfig config = {
        *(const uint32_t*)(attr + 0x0c), *(const uint32_t*)(attr + 0x10),
        *(const uint64_t*)(attr + 0x20), *(const uint32_t*)(attr + 0x04), true
    };
    const auto* bufs = (const VideoOutBuffers*)(uintptr_t)a3;
    std::lock_guard<std::mutex> lk(g_display_mx);
    if (g_display.sets[set].registered)
        return (uint64_t)(int64_t)(int32_t)0x8029000a;  // SCE_VIDEO_OUT_ERROR_INVALID_INDEX
    for (int i = 0; i < num; ++i) {
        if (g_display.buffer_set[start + i] != 0)
            return (uint64_t)(int64_t)(int32_t)0x80290010;  // SCE_VIDEO_OUT_ERROR_SLOT_OCCUPIED
    }
    g_display.width = config.width;
    g_display.height = config.height;
    g_display.pixel_format = config.pixel_format;
    g_display.tiling_mode = config.tiling_mode;
    g_display.sets[set] = config;
    {
        std::lock_guard<std::mutex> flip_lk(g_flip_mx);
        for (int i = 0; i < num; ++i) g_buffer_labels[start + i] = 0;
    }
    for (int i = 0; i < num; i++) {
        g_display.buffer_addr[start + i] = (uint64_t)(uintptr_t)bufs[i].data;
        g_display.buffer_set[start + i] = (uint8_t)(set + 1);
        uint64_t generation = ++g_display.next_generation;
        if (generation == 0) generation = ++g_display.next_generation;
        g_display.buffer_generation[start + i] = generation;
    }
    if (g_display.buffer_num < start + num) g_display.buffer_num = start + num;
    g_display.configured = true;
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[vo] display surface: %ux%u fmt=0x%llx %d buffers registered\n",
                g_display.width, g_display.height, (unsigned long long)g_display.pixel_format, num);
    return 0;
}

// sceVideoOutUnregisterBuffers (N5KDtkIjjJ4): release every slot owned by one attribute set.
// RegisterBuffers2 may place independent sets in disjoint ranges, so retain the other sets and
// recompute the highest visible slot instead of clearing the entire display unconditionally.
HLE(g_vo_unregister_buffers) {
    vo_argtrace("UnregisterBuffers", a0,a1,a2,a3,a4,a5);
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    const int set = (int)a1;
    if (set < 0 || set > 3)
        return (uint64_t)(int64_t)(int32_t)0x8029000a;  // SCE_VIDEO_OUT_ERROR_INVALID_INDEX

    std::lock_guard<std::mutex> lk(g_display_mx);
    const uint8_t tag = (uint8_t)(set + 1);
    bool found = false;
    {
        std::lock_guard<std::mutex> flip_lk(g_flip_mx);
        for (int i = 0; i < 16; ++i) {
            if (g_display.buffer_set[i] != tag) continue;
            found = true;
            if (g_display.front_index == i &&
                g_display.front_generation == g_display.buffer_generation[i]) {
                g_display.front_index = -1;
                g_display.front_generation = 0;
            }
            // A numeric slot can be registered again with a different backing surface. Retire the
            // old label identity now so a later flip cannot clear the new registration's label.
            g_buffer_labels[i] = 0;
            if (g_previous_buffer == i) g_previous_buffer = -1;
            g_display.buffer_set[i] = 0;
            g_display.buffer_addr[i] = 0;
            g_display.buffer_generation[i] = 0;
        }
    }
    if (!found)
        return (uint64_t)(int64_t)(int32_t)0x8029000a;  // SCE_VIDEO_OUT_ERROR_INVALID_INDEX
    g_display.sets[set] = {};

    int remaining = 16;
    while (remaining > 0 && g_display.buffer_set[remaining - 1] == 0) --remaining;
    g_display.buffer_num = remaining;
    g_display.configured = remaining != 0;
    if (!g_display.configured) {
        g_display.width = 0;
        g_display.height = 0;
        g_display.pixel_format = 0;
        g_display.tiling_mode = 0;
    } else {
        const int remaining_set = (int)g_display.buffer_set[remaining - 1] - 1;
        const auto& config = g_display.sets[remaining_set];
        g_display.width = config.width;
        g_display.height = config.height;
        g_display.pixel_format = config.pixel_format;
        g_display.tiling_mode = config.tiling_mode;
    }
    return 0;
}

// sceVideoOutConfigureOutput (w0hLuNarQxY): set the output mode (resolution/refresh/format). We only
// advertise one mode (1080p60), so accept the configuration.
HLE(g_vo_configure_output) {
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    vo_argtrace("ConfigureOutput", a0,a1,a2,a3,a4,a5);
    return 0;
}

// sceVideoOutGetOutputStatus (utPrVdxio-8): (handle, status*). The full out-struct layout has NO
// reference (Kyty predates it, shadPS4 only stubs the name), but the ONE consumer in the wild is
// now RE'd (issue #82; DOLL/PPSA17942 swapchain-init, eboot+0x2226fea..0x2227042, gdb-captured):
// it reads a single u32 at status+0x04 and compares it to 2 — the ==2 branch (together with two
// runtime capability checks) switches the display pipe to an HDR pixel format
// (0x8100070400000000) — i.e. status+0x04 is the output's dynamic-range/colorimetry mode with
// 2 = HDR. Previously we returned success with the struct UNWRITTEN, so the caller consumed
// uninitialized stack (usually != 2 -> SDR by luck — the classic success+garbage-out hazard).
// Write a defined 8-byte {u32 state=0, u32 mode} prefix: mode 0 by default (the SDR path,
// matching prosper's default SDR-only display) and mode 2 under PROSPER_HDR (the --hdr opt-in).
// 8 bytes cannot overrun any plausible real struct (the DOLL caller reserves 0x50 stack bytes
// for it). CONFIDENCE: MED on field semantics (single consumer, unambiguous read), HIGH that a
// defined write beats garbage.
HLE(g_vo_get_output_status) {
    vo_argtrace("GetOutputStatus", a0,a1,a2,a3,a4,a5);
    if (!a1)
        return (uint64_t)(int64_t)(int32_t)0x80290002;  // SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    if (a1 > 0xffffull) {
        *(uint32_t*)(uintptr_t)a1       = 0;   // +0x00: output state (0 = the boring/default state)
        // +0x04: dynamic-range mode (2 = HDR; 0 = SDR — the DOLL-RE'd field, issue #82).
        *(uint32_t*)(uintptr_t)(a1 + 4) = prosper_hdr_output() ? 2u : 0u;
    }
    return 0;
}

// sceVideoOutGetEventId (U2JJtSqNKZI — named via nid_hash brute-force over the libSceVideoOut stub
// corpus, issue #210): decode which VideoOut event a kevent delivered on a VideoOut equeue is.
// Contract (Kyty EventQueue.cpp:354 KernelGetEventId = plain ev->ident read; the VideoOut wrapper is
// the same field read on PS4 — shadPS4 videoout returns ev->ident after a filter check): the kevent's
// ident IS the event id (FLIP=0, VBLANK=1 — exactly what our flip/vblank posts set). The previous
// unimplemented-0 return decoded EVERY event (vblank included) as a FLIP: UE4's VideoOut pump thread
// dispatched vblank ticks into its flip-completion handler and never saw a real vblank tick.
// SceKernelEvent layout: ident @0 (i64), filter @8 (i16). CONFIDENCE: HIGH (both references agree and
// the field is ours to define — we post these events).
HLE(g_vo_get_event_id) {
    if (!a0) return (uint64_t)(int64_t)(int32_t)0x80290002;   // SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS
    const uint8_t* ev = (const uint8_t*)(uintptr_t)a0;
    if (*(const int16_t*)(ev + 8) != -13)                     // not a VideoOut-filter kevent
        return (uint64_t)(int64_t)(int32_t)0x80290011;        // SCE_VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE
    return (uint64_t)*(const int64_t*)ev;                     // ident: FLIP=0 / VBLANK=1 / ...
}

// sceVideoOutGetEventData (rWUTcKdkUzQ): return the data carried by a delivered VideoOut kevent.
// Prosper's producer owns this contract: flip completion posts the submitted flipArg verbatim at
// SceKernelEvent+0x10, while vblank posts its frame counter there. Do not apply the packed-event
// decoding used by backends whose producers encode time/count bits into this field.
HLE(g_vo_get_event_data) {
    if (!a0 || !a1)
        return (uint64_t)(int64_t)(int32_t)0x80290002;         // SCE_VIDEO_OUT_ERROR_INVALID_ADDRESS
    const uint8_t* ev = (const uint8_t*)(uintptr_t)a0;
    if (*(const int16_t*)(ev + 8) != -13)
        return (uint64_t)(int64_t)(int32_t)0x8029000d;         // SCE_VIDEO_OUT_ERROR_INVALID_EVENT
    *(int64_t*)(uintptr_t)a1 = *(const int64_t*)(ev + 0x10);
    return 0;
}

// sceVideoOutSetWindowModeMargins (MTxxrOCeSig — same brute-force naming): windowed-mode margin
// hint for the system compositor. No compositor here — accept and ignore. CONFIDENCE: HIGH (pure
// cosmetic setter; success unblocks the caller).
HLE(g_vo_set_window_mode_margins) {
    VideoOutHandleGuard handle(a0);
    if (!handle.valid()) return kVoErrorInvalidHandle;
    vo_argtrace("SetWindowModeMargins", a0,a1,a2,a3,a4,a5);
    return 0;
}

// Diagnostic tracer for the (undocumented) libSceAgc / libSceAgcDriver calls: logs the NID, the guest
// callsite, and all six args (gated on PROSPER_GFXLOG). Behaviour is identical to the unimplemented
// stub — returns 0, changes no control flow — it is purely observable. This is the RE bootstrap for
// the real AGC->Vulkan work (M4/M5), NOT faked output.
//
// Per-NID identification: each NID is bound to a distinct template thunk `glog_thunk<I>` that knows
// its own name (kAgcNids[I]). Plain POSIX stubs tail-jump into the thunk, while Linux guest-FS and
// Windows ABI bridge stubs CALL it. Capture the thunk's pre-prologue stack and unwrap either shape so
// the trace always reports the original guest return address, never the generated-stub return site.
// Result: `PROSPER_GFXLOG=1 boot_trace <dump>` emits a self-describing dataset (NID + guest callsite +
// args per call) — everything needed to RE the AGC object model.
namespace {
// The libSceAgc/AgcDriver NIDs the game calls (observed via boot_trace's unimplemented log). The
// first four are AgcDriver; the rest are libSceAgc functions used during GfxDevicePS5SharedData init
// whose null results currently leave GPU objects zeroed — the graphics blocker.
const char* const kAgcNids[] = {
    "MM4IZSEYytQ", "XlNp7jzGiPo", "Zw7uUVPulbw", "w2rJhmD+dsE",
    "23LRUSvYu1M", "BfBDZGbti7A", "TRO721eVt4g", "MWiElSNE8j8", "-KRzWekV120", "wr23dPKyWc0",
    "VmW0Tdpy420", "ZvwO9euwYzc", "vcmNN+AAXnY", "d-6uF9sZDIU", "+kSrjIVxKFE", "H7uZqCoNuWk",
    "f3dg2CSgRKY", "hvUfkUIQcOE", "6lNcCp+fxi4", "vRoArM9zaIk", "0fWWK5uG9rQ", "3KDcnM3lrcU",
    "57labkp+rSQ", "LtTouSCZjHM", "V++UgBtQhn0", "aJf+j5yntiU", "fPSCdQxgpSw", "i1jyy49AjXU",
    "tSBxhAPyytQ",   // fires once after the Sh patch pair during CreateWorkload; not in Kyty — RE via args
    "k3GhuSNmBLU",   // fires once just before "unity default resources" load; not in Kyty — RE via args
    "WmAc2MEj6Io",   // sceAgcDcbDmaData (shadPS4 aerolib) — OVERRIDDEN by agc_dcb_dma_data (real appender, #117)
    "IxYiarKlXxM",   // sceAgcDmaDataPatchSetDstAddressOrOffset — observe-only (packet now gets valid ptrs, #117)
    // Dead Cells resource-registration startup path (#539). These trace entries preserve the original
    // six arguments, which the generic unimplemented stub cannot do because it replaces a0 with the
    // import index. The two output queries are overridden by real handlers in hle_agc.
    "AOLcoIkQDgM",   // sceAgcDriverQueryResourceRegistrationUserMemoryRequirements
    "F0Y42t-3e18",   // sceAgcDriverInitResourceRegistration
    "U9ueyEhSkF4",   // sceAgcDriverRegisterDefaultOwner (shadPS4 symbol map)
    "X-Nm5KLREeg",   // sceAgcDriverRegisterOwner
    "W5z4eZrjEas",   // sceAgcDriverRegisterResource
    "F0ZXt5q0ZTA",   // sceAgcDriverGetDefaultOwner (shadPS4 symbol map)
    "uJziRsODk1c",   // sceAgcDriverGetResourceRegistrationMaxNameLength (overridden in hle_agc)
};
constexpr size_t kDefaultAgcNidCount = 32;
constexpr size_t kAgcNidCount = sizeof(kAgcNids) / sizeof(kAgcNids[0]);
static_assert(kAgcNidCount == kDefaultAgcNidCount + 7);

uint64_t glog_impl(const char* nid, void* ra,
                   uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    gfx_tick();
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[gfx] libSceAgc::%s  from %s+0x%llx  a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx\n",
                nid, prosper::guest_module_name((uint64_t)ra),
                (unsigned long long)prosper::guest_module_offset((uint64_t)ra),
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    // PROSPER_CTXDUMP: dump the register-context per-stage sub-objects' {source, user_data} pair at
    // each AGC call, treating a0 as the ctx wrapper (subs at a0+0x38+stage*0x70 — layout from the
    // eboot+0x3b0210 accessor + live fault frame). Diagnostic for the stale-[sub+8] shader-bind crash:
    // shows WHICH call first plants a shared ud pointer across subs. Reads are blind-but-plausible
    // (heap-interior a0), gated off by default.
    // PROSPER_SUBWATCH: on the H7uZqCoNuWk(ctx, 7, ...) call (the last AGC call before the pre-draw
    // pass that flips the stage subs' user_data to the shared scratch descriptor), arm a HW write-
    // watch on stage-1's ud slot [ctx+0x38+0x70+8] — catches the flip writer's RIP.
    if (getenv("PROSPER_SUBWATCH") && g_hwwatch_hook && !strcmp(nid, "H7uZqCoNuWk") && a1 == 7 &&
        a0 > 0x100000ull)
        g_hwwatch_hook(a0 + 0x38 + 0x70 + 8);
    if (getenv("PROSPER_CTXDUMP") && a0 > 0x100000ull) {
        fprintf(stderr, "  [ctx] %s a0=0x%llx subs:", nid, (unsigned long long)a0);
        for (int st = 0; st < 4; st++) {
            const uint64_t* sub = (const uint64_t*)(uintptr_t)(a0 + 0x38 + (uint64_t)st * 0x70);
            fprintf(stderr, " s%d={%llx,%llx}", st, (unsigned long long)sub[0], (unsigned long long)sub[1]);
        }
        fprintf(stderr, "\n");
    }
    // Shader-recompiler prep: dump the blob args of sceAgcCreateShader (f3dg2CSgRKY) to identify the
    // RDNA2 shader container format. Args are game-heap pointers (already mapped), so the reads are
    // safe; we only touch plausible pointers. Gated on PROSPER_AGCSHADER.
    if (getenv("PROSPER_AGCSHADER") && !strcmp(nid, "f3dg2CSgRKY")) {
        const uint64_t args[4] = {a0, a1, a2, a3};
        for (int ai = 0; ai < 4; ai++) {
            if (args[ai] < 0x100000ull) { fprintf(stderr, "  a%d=0x%llx (immediate)\n", ai, (unsigned long long)args[ai]); continue; }
            const uint8_t* p = (const uint8_t*)(uintptr_t)args[ai];
            fprintf(stderr, "  a%d=0x%llx: ", ai, (unsigned long long)args[ai]);
            for (int b = 0; b < 32; b++) fprintf(stderr, "%02x ", p[b]);
            fprintf(stderr, " | ");
            for (int b = 0; b < 32; b++) { char c = (char)p[b]; fprintf(stderr, "%c", (c >= 32 && c < 127) ? c : '.'); }
            fprintf(stderr, "\n");
        }
        // Dump the first shader's code blob (a2) to a file so it can be disassembled as RDNA2.
        static bool dumped = false;
        if (!dumped && a2 > 0x100000ull) {
            dumped = true;
            // Size: prefer an explicit a4 count; else derive it from the shader's own s_endpgm
            // terminator (RDNA2 SOPP 0xBF810000) instead of a fixed 0x200 fragment (#160). Scan is
            // bounded so a bad/short mapping can't run away.
            uint64_t sz;
            if (a4 && a4 <= 0x10000) {
                sz = a4;                                        // a4 is a plausible explicit byte count
            } else {
                const uint32_t* w = (const uint32_t*)(uintptr_t)a2;
                const uint64_t cap = 0x4000 / 4;                // scan up to 16 KiB of dwords
                uint64_t n = 0;
                for (; n < cap; n++) if (w[n] == 0xBF810000u) { n++; break; }   // include the s_endpgm dword
                sz = (n && n < cap) ? n * 4 : 0x2000;           // found -> exact end; else a generous window
            }
            // Path from PROSPER_AGCSHADER_OUT (default relative, so it works off this dev's machine);
            // report an fopen failure instead of silently writing nothing (#160).
            const char* out = getenv("PROSPER_AGCSHADER_OUT");
            if (!out || !*out) out = "shader0.bin";
            if (FILE* f = fopen(out, "wb")) {
                fwrite((const void*)(uintptr_t)a2, 1, sz, f); fclose(f);
                fprintf(stderr, "  [wrote %s: %llu bytes from a2]\n", out, (unsigned long long)sz);
            } else {
                fprintf(stderr, "  [PROSPER_AGCSHADER: fopen('%s') failed: %s]\n", out, strerror(errno));
            }
        }
    }
    // PROSPER_AGCRET: RE probe for the render thread's tight spin on Zw7uUVPulbw(ctx,0x81,1,ptr,ts,iter).
    // The stub returns 0 forever, so if the guest polls "is it ready? (0=no)" it spins. Return the env
    // value for that NID to test whether a non-zero ("ready"/handle) result breaks the spin and lets the
    // render thread proceed. Diagnostic only (gated) — the value that works tells us the real semantics.
    // Cached: glog_impl is the thunk for EVERY stubbed AGC NID, including the documented tight spin on
    // Zw7uUVPulbw — per-call getenv() environ scans here are paid on every iteration of that spin.
    static const char* const agcret = getenv("PROSPER_AGCRET");
    if (const char* rv = agcret) {
        if (!strcmp(nid, "Zw7uUVPulbw")) return (uint64_t)strtoull(rv, nullptr, 0);
    }
    // PROSPER_ZWDUMP: is Zw7uUVPulbw a GPU submit (does its context/args reference a PM4 command buffer
    // with draws we're not processing)? Dump the context (a0) and probe args as pointers, on the first
    // few calls, so we can tell "no scene" from "scene submitted via a path we don't hook."
    static const bool zwdump = getenv("PROSPER_ZWDUMP") != nullptr;
    if (zwdump && !strcmp(nid, "Zw7uUVPulbw")) {
        static int n = 0;
        if (n++ < 3) {
            // Guest memory is mapped at high addresses (0x70xx_........), so a valid pointer just needs to
            // be >0x100000 and not obviously garbage. probe reads 16 dwords.
            // Only guest-mapped ranges: modules [0x4_00000000,0x7_00000000) or heap [0x7000_00000000,
            // 0x8000_00000000). Excludes ASCII-as-value (e.g. "Thread"=0x646165726854) that would segfault.
            // A range check alone isn't enough — these windows are mostly UNMAPPED, so every deref is
            // additionally gated on guest_readable (a probe that can't SIGSEGV). A worker-thread fault
            // here would _exit(90) the emulator — the probe must never kill the process it diagnoses.
            auto looks_ptr = [](uint64_t p){ return (p >= 0x400000000ull && p < 0x700000000ull) ||
                                                    (p >= 0x700000000000ull && p < 0x800000000000ull); };
            auto probe = [&](const char* nm, uint64_t p){
                if (!looks_ptr(p) || !gpu::guest_readable(p, 16 * 4)) {
                    fprintf(stderr, "  [zw] %s=0x%llx (imm/not-ptr/unmapped)\n", nm, (unsigned long long)p); return; }
                const uint32_t* u = (const uint32_t*)(uintptr_t)p;
                fprintf(stderr, "  [zw] %s=0x%llx:", nm, (unsigned long long)p);
                for (int i = 0; i < 16; i++) fprintf(stderr, " %08x", u[i]);
                fprintf(stderr, "\n");
            };
            probe("a0", a0); probe("a3", a3); probe("a4", a4);
            if (looks_ptr(a0) && gpu::guest_readable(a0, 40 * 8)) {
                const uint64_t* ctx = (const uint64_t*)(uintptr_t)a0;
                for (int i = 0; i < 40; i++) {
                    uint64_t v = ctx[i];
                    if (looks_ptr(v) && gpu::guest_readable(v, 8 * 4)) {
                        const uint32_t* u = (const uint32_t*)(uintptr_t)v;
                        fprintf(stderr, "  [zw] a0[%d]=0x%llx ->", i, (unsigned long long)v);
                        for (int j = 0; j < 8; j++) fprintf(stderr, " %08x", u[j]);
                        fprintf(stderr, "\n");
                    }
                }
            }
        }
    }
    return 0;
}
template <size_t I>
__attribute__((noinline)) PROSPER_SYSV_ABI uint64_t glog_thunk(uint64_t a0, uint64_t a1, uint64_t a2,
                                              uint64_t a3, uint64_t a4, uint64_t a5) {
    const uint64_t entry_rsp = (uint64_t)(uintptr_t)__builtin_frame_address(0) + sizeof(uint64_t);
    const uint64_t guest_ra = hle_guest_return_address(entry_rsp);
    return glog_impl(kAgcNids[I], (void*)(uintptr_t)guest_ra, a0, a1, a2, a3, a4, a5);
}
template <size_t Offset, size_t... Is>
void register_agc_tracers(std::index_sequence<Is...>) {
    // Placeholders: hle_agc registers the REAL handlers for the implemented subset later and is meant
    // to override these tracing thunks — so mark them overridable and that override is not a shadow.
    (Hle::register_placeholder(kAgcNids[Offset + Is], (HleFn)&glog_thunk<Offset + Is>,
                               kAgcNids[Offset + Is]), ...);
}
}

void register_graphics_hle() {
    #define RN(nid, fn) Hle::register_fn(nid, (HleFn)(fn), nid)   // raw NID (graphics libs undocumented)
    #define R(str, fn)  Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    // libSceAgc getters whose results the guest dereferences → return stable zeroed objects.
    RN("2JtWUUiYBXs", g_agc_regdefs);       // GraphicsGetRegisterDefaults2       -> g_reg_defaults1
    RN("wRbq6ZjNop4", g_agc_regdefs_int);   // GraphicsGetRegisterDefaults2Internal -> g_reg_defaults2
    // All traced libSceAgc/AgcDriver NIDs → per-NID logging thunks (still return 0; observable only).
    register_agc_tracers<0>(std::make_index_sequence<kDefaultAgcNidCount>{});
    // Extra resource-registration tracing changes the generic unimplemented-stub scheduling enough to
    // suppress Dead Cells' intermittent startup failure (#539), so it must be explicitly diagnostic.
    if (getenv("PROSPER_AGC_REG_TRACE"))
        register_agc_tracers<kDefaultAgcNidCount>(
            std::make_index_sequence<kAgcNidCount - kDefaultAgcNidCount>{});
    // Override the w2rJhmD+dsE tracer with the real sceAgcDriverAddEqEvent (registers the GPU-completion
    // event source the game's FTM/EOP queues wait on). Must come AFTER the tracer registrations.
    RN("w2rJhmD+dsE", g_agc_add_eq_event);  // sceAgcDriverAddEqEvent (GPU EOP completion, Agc driver path)
    // libSceVideoOut display / flip
    R("sceVideoOutOpen", g_vo_open);            R("sceVideoOutClose", g_vo_close);
    R("sceVideoOutSubmitFlip", g_vo_submitflip);R("sceVideoOutIsFlipPending", g_vo_flippending);
    R("sceVideoOutSetFlipRate", g_vo_set_flip_rate); R("sceVideoOutAddFlipEvent", g_vo_addflipevent);
    R("sceVideoOutAddVblankEvent", g_vo_addvblankevent);
    R("sceVideoOutGetFlipStatus", g_vo_flipstatus);
    R("sceVideoOutGetResolutionStatus", g_vo_resstatus);
    R("sceVideoOutGetVblankStatus", g_vo_vblankstatus);
    R("sceVideoOutWaitVblank", g_vo_wait_vblank);   // j6RaAUlaLv0 (#1746)
    R("sceVideoOutGetDeviceCapabilityInfo", g_vo_devcap);
    R("sceVideoOutRegisterBuffers", g_vo_register_buffers);R("sceVideoOutSetBufferAttribute", g_vo_set_buffer_attribute);
    // PS5 "2"/query variants — the 5 previously-unimplemented VideoOut NIDs (resolved via shadPS4
    // aerolib + Kyty VideoOut.cpp). Registered by raw NID (PS5-specific, not in our name DB).
    RN("Nv8c-Kb+DUM", g_vo_is_output_supported);   // sceVideoOutIsOutputSupported
    RN("PjS5uASwcV8", g_vo_set_buffer_attribute2);  // sceVideoOutSetBufferAttribute2
    RN("rKBUtgRrtbk", g_vo_register_buffers2);       // sceVideoOutRegisterBuffers2
    RN("N5KDtkIjjJ4", g_vo_unregister_buffers);      // sceVideoOutUnregisterBuffers
    RN("OcQybQejHEY", g_vo_get_buffer_label_address); // sceVideoOutGetBufferLabelAddress (#394 F4)
    RN("utPrVdxio-8", g_vo_get_output_status);        // sceVideoOutGetOutputStatus
    RN("w0hLuNarQxY", g_vo_configure_output);          // sceVideoOutConfigureOutput
    RN("U2JJtSqNKZI", g_vo_get_event_id);              // sceVideoOutGetEventId (#210)
    RN("rWUTcKdkUzQ", g_vo_get_event_data);            // sceVideoOutGetEventData (#394 F5)
    RN("MTxxrOCeSig", g_vo_set_window_mode_margins);   // sceVideoOutSetWindowModeMargins (#210)
    RN("b0xyllnVY-I", g_gnm_add_eq_event);             // sceGnmAddEqEvent / GraphicsAddEqEvent (GPU EOP)
    #undef R
    #undef RN
}

} // namespace prosper
