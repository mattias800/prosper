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
#include <utility>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

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
extern "C" void* prosper_agc_reg_defaults(unsigned int ver);           // -> g_reg_defaults1 (127 regs)
extern "C" void* prosper_agc_reg_defaults_internal(unsigned int ver);  // -> g_reg_defaults2 (21 regs)
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
    int      g_vo_handle = 0;
    // Flip bookkeeping (#82): count/flipArg/currentBuffer are written by TWO paths — SubmitFlip on
    // any guest thread and prosper_vo_flip_from_gpu on the submit thread — and read by GetFlipStatus
    // on Unity's pacer thread. They must move as ONE unit under a mutex: an unsynchronized reader
    // could observe the count advanced but a stale flipArg/buffer (torn triple), making the pacer
    // believe a flip completed with the wrong arg; concurrent flips could lose an increment.
    std::mutex g_flip_mx;
    uint64_t g_flip_count = 0;      // incremented per flip so GetFlipStatus shows progress
    int32_t  g_current_buffer = 0;
    int64_t  g_last_flip_arg = 0;
    void flip_advance(int32_t bufidx, int64_t flip_arg) {
        std::lock_guard<std::mutex> lk(g_flip_mx);
        g_flip_count++;
        g_current_buffer = bufidx;
        g_last_flip_arg = flip_arg;
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
        uint32_t width = 0, height = 0;
        uint64_t pixel_format = 0;
        uint32_t tiling_mode = 0;
        int      buffer_num = 0;
        uint64_t buffer_addr[16] = {0};   // guest GPU-VA of each registered framebuffer
        bool     configured = false;
    };
    DisplayConfig g_display;
    struct VideoOutBuffers { const void* data; const void* metadata; const void* reserved[2]; };
}

// Exposed for the back-half present path + tests: the registered display surface.
// prosper_vo_flip_count: total flips so far (either flip path) — read by the PROSPER_PROGRESS
// heartbeat in hle_agc.cpp as a cheap forward-progress signal for long diagnostic runs.
extern "C" uint64_t prosper_vo_flip_count() { std::lock_guard<std::mutex> lk(g_flip_mx); return g_flip_count; }
extern "C" int      prosper_vo_buffer_count()   { return g_display.buffer_num; }
extern "C" uint32_t prosper_vo_display_width()  { return g_display.width; }
extern "C" uint32_t prosper_vo_display_height() { return g_display.height; }
extern "C" uint64_t prosper_vo_display_format() { return g_display.pixel_format; }
extern "C" uint64_t prosper_vo_buffer_addr(int i) {
    return (i >= 0 && i < g_display.buffer_num) ? g_display.buffer_addr[i] : 0;
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
HLE(g_vo_open)        { gfx_tick(); return (uint64_t)(int64_t)(++g_vo_handle + 0x1000); }  // positive handle
HLE(g_vo_close)       { return 0; }
// sceVideoOutAddFlipEvent(eq, handle, udata): register a flip-completion event source on an equeue.
HLE(g_vo_addflipevent) {
    if (evlog()) fprintf(stderr, "[ev] AddFlipEvent eq=0x%llx handle=0x%llx udata=0x%llx\n",
        (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
    prosper_eq_add_flip(a0, (int64_t)(int32_t)a1, a2);
    return 0;
}
HLE(g_vo_addvblankevent) {
    if (evlog()) fprintf(stderr, "[ev] AddVblankEvent eq=0x%llx handle=0x%llx udata=0x%llx\n",
        (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2);
    prosper_eq_add_vblank(a0, (int64_t)(int32_t)a1, a2);
    return 0;
}
HLE(g_vo_submitflip)  {
    if (evlog()) fprintf(stderr, "[ev] SubmitFlip handle=0x%llx bufidx=%lld flipmode=0x%llx fl013arg=0x%llx\n",
        (unsigned long long)a0, (long long)(int32_t)a1, (unsigned long long)a2, (unsigned long long)a3);
    flip_advance((int32_t)a1, (int64_t)a3);
    gpu::present_flip((int)(int32_t)a1, (int64_t)a3);   // present the buffer (scanout front + count)
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
    if (evlog()) fprintf(stderr, "[ev] GpuFlip handle=0x%x bufidx=%d mode=0x%x fliparg=0x%llx\n",
                         handle, bufidx, flip_mode, (unsigned long long)flip_arg);
    flip_advance(bufidx, flip_arg);
    gpu::present_flip(bufidx, flip_arg);   // scanout bookkeeping, same as the API flip
    prosper_eq_trigger_flip(flip_arg);     // flip completed (synchronous): fire the flip event
}
HLE(g_vo_flippending) { if (evlog()) fprintf(stderr, "[ev] IsFlipPending\n"); return 0; }        // never pending
HLE(g_vo_flipstatus)  { // (handle, SceVideoOutFlipStatus* status): report our simulated flip count.
    // SceVideoOutFlipStatus is exactly 0x40 bytes — writing more smashes the caller's stack canary!
    if (a1) { uint8_t* s = (uint8_t*)(uintptr_t)a1; memset(s, 0, 0x40);
              uint64_t cnt; int64_t arg; int32_t buf;
              { std::lock_guard<std::mutex> lk(g_flip_mx);   // consistent snapshot of the flip triple
                cnt = g_flip_count; arg = g_last_flip_arg; buf = g_current_buffer; }
              *(uint64_t*)(s + 0x00) = cnt;    // count
              *(int64_t*) (s + 0x18) = arg;    // flipArg
              *(int32_t*) (s + 0x38) = buf; }  // currentBuffer
    return 0;
}
// SceVideoOutResolutionStatus (0x20 bytes, Kyty VideoOutResolutionStatus): report a real 1080p60
// panel instead of the previous all-zero (which advertised a 0x0 display).
HLE(g_vo_resstatus)   {
    if (a1) { uint8_t* s = (uint8_t*)(uintptr_t)a1; memset(s, 0, 0x20);
              *(uint32_t*)(s + 0x00) = kDispW;   *(uint32_t*)(s + 0x04) = kDispH;   // full w/h
              *(uint32_t*)(s + 0x08) = kDispW;   *(uint32_t*)(s + 0x0c) = kDispH;   // pane w/h
              *(uint64_t*)(s + 0x10) = kRefresh5994;
              *(float*)   (s + 0x18) = 50.0f; }                                     // screen size (inch)
    return 0;
}

// sceVideoOutGetVblankStatus (SceVideoOutVblankStatus, 0x28 bytes — Kyty VideoOut.cpp:94:
// u64 count @0, u64 processTime @8, u64 tsc @0x10, u64 reserved @0x18, u8 flags @0x20).
// The count is derived from a monotonic ~59.94 Hz timebase — the previous ++g_vblank_count
// advanced once per CALL, so a poll loop saw a "vblank" on every poll (uncapped pacing) and the
// counter disagreed arbitrarily with the real 60 Hz equeue vblank pump.
HLE(g_vo_vblankstatus) {
    if (a1) { uint8_t* s = (uint8_t*)(uintptr_t)a1; memset(s, 0, 0x28);
              uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
              static const uint64_t t0 = ns;                       // process-relative, first-call anchored
              constexpr uint64_t kVblankNs = 16683350;             // 59.94 Hz period
              // A live VideoOut handle has already observed at least one vblank. Returning zero on
              // the first query made Astro Bot compute a frame-rate sample as
              // `(frames * refresh) / (count - previous_count)` with both counts zero and trap on
              // IDIV. Keep the time-based behavior, but number the first process-relative vblank 1.
              *(uint64_t*)(s + 0x00) = 1 + (ns - t0) / kVblankNs;  // count: one tick per vblank period
              *(uint64_t*)(s + 0x08) = (ns - t0) / 1000;           // processTime (µs)
              *(uint64_t*)(s + 0x10) = ns; }                       // tsc (monotonic ns; matches ReadTsc's unit)
    return 0;
}
// sceVideoOutGetDeviceCapabilityInfo (SceVideoOutDeviceCapabilityInfo: single u64 capability).
// Advertise a plain SDR display (no HDR/BT2020) — capability 0.
HLE(g_vo_devcap) { if (a1) *(uint64_t*)(uintptr_t)a1 = 0; return 0; }

// sceVideoOutIsOutputSupported (Nv8c-Kb+DUM): (port_type, mode/feature) -> bool. A standard main-bus
// output supports the queried mode → 1.
HLE(g_vo_is_output_supported) { vo_argtrace("IsOutputSupported", a0,a1,a2,a3,a4,a5); return 1; }

// sceVideoOutSetBufferAttribute2 (PjS5uASwcV8): fill the caller's VideoOutBufferAttribute2 (0x50
// bytes, Kyty layout) from (attr, pixel_format, tiling_mode, width, height, option, [dcc_control,
// dcc_cb_clear on stack]). Mirrors Kyty's setter. Size-exact write.
HLE(g_vo_set_buffer_attribute2) {  // a0=attr* a1=pixel_format a2=tiling a3=width a4=height a5=option
    vo_argtrace("SetBufferAttribute2", a0,a1,a2,a3,a4,a5);
    if (!a0) return 0;
    uint8_t* p = (uint8_t*)(uintptr_t)a0; memset(p, 0, 0x50);
    *(uint32_t*)(p + 0x04) = (uint32_t)a2;   // tiling_mode
    *(uint32_t*)(p + 0x08) = 0;              // aspect_ratio (16:9)
    *(uint32_t*)(p + 0x0c) = (uint32_t)a3;   // width
    *(uint32_t*)(p + 0x10) = (uint32_t)a4;   // height
    *(uint64_t*)(p + 0x18) = a5;             // option
    *(uint64_t*)(p + 0x20) = a1;             // pixel_format
    return 0;
}

// sceVideoOutRegisterBuffers2 (rKBUtgRrtbk): (handle, set_index, buffer_index_start, buffers,
// buffer_num, attribute, [category, option on stack]). Validate ranges like Kyty and accept. The
// buffers' GPU backing becomes swapchain images once the swapchain exists; for now, registration
// succeeds so the game proceeds to flips.
HLE(g_vo_register_buffers2) {  // a0=handle a1=set_index a2=buffer_index_start a3=buffers a4=buffer_num a5=attribute
    vo_argtrace("RegisterBuffers2", a0,a1,a2,a3,a4,a5);
    int start = (int)a2, num = (int)a4;
    if (!a3 || !a5) return (uint64_t)(int64_t)-1;
    if (start < 0 || start > 15 || num < 1 || num > 16 || start + num > 16) return (uint64_t)(int64_t)-1;
    // Record the display surface (swapchain scaffolding). attribute is the 0x50-byte
    // VideoOutBufferAttribute2 we fill in SetBufferAttribute2: width@0x0c, height@0x10, format@0x20.
    const uint8_t* attr = (const uint8_t*)(uintptr_t)a5;
    g_display.width        = *(const uint32_t*)(attr + 0x0c);
    g_display.height       = *(const uint32_t*)(attr + 0x10);
    g_display.pixel_format = *(const uint64_t*)(attr + 0x20);
    g_display.tiling_mode  = *(const uint32_t*)(attr + 0x04);
    const auto* bufs = (const VideoOutBuffers*)(uintptr_t)a3;
    for (int i = 0; i < num && start + i < 16; i++)
        g_display.buffer_addr[start + i] = (uint64_t)(uintptr_t)bufs[i].data;
    if (g_display.buffer_num < start + num) g_display.buffer_num = start + num;
    g_display.configured = true;
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[vo] display surface: %ux%u fmt=0x%llx %d buffers registered\n",
                g_display.width, g_display.height, (unsigned long long)g_display.pixel_format, num);
    return 0;
}

// sceVideoOutConfigureOutput (w0hLuNarQxY): set the output mode (resolution/refresh/format). We only
// advertise one mode (1080p60), so accept the configuration.
HLE(g_vo_configure_output) { vo_argtrace("ConfigureOutput", a0,a1,a2,a3,a4,a5); return 0; }

// sceVideoOutGetOutputStatus (utPrVdxio-8): (handle, status*). The full out-struct layout has NO
// reference (Kyty predates it, shadPS4 only stubs the name), but the ONE consumer in the wild is
// now RE'd (issue #82; DOLL/PPSA17942 swapchain-init, eboot+0x2226fea..0x2227042, gdb-captured):
// it reads a single u32 at status+0x04 and compares it to 2 — the ==2 branch (together with two
// runtime capability checks) switches the display pipe to an HDR pixel format
// (0x8100070400000000) — i.e. status+0x04 is the output's dynamic-range/colorimetry mode with
// 2 = HDR. Previously we returned success with the struct UNWRITTEN, so the caller consumed
// uninitialized stack (usually != 2 -> SDR by luck — the classic success+garbage-out hazard).
// Write a defined 8-byte {u32 state=0, u32 mode=0} prefix: mode 0 selects the SDR path
// deterministically, matching prosper's advertised SDR-only display. 8 bytes cannot overrun any
// plausible real struct (the DOLL caller reserves 0x50 stack bytes for it). CONFIDENCE: MED on
// field semantics (single consumer, unambiguous read), HIGH that a defined write beats garbage.
HLE(g_vo_get_output_status) {
    vo_argtrace("GetOutputStatus", a0,a1,a2,a3,a4,a5);
    if (a1 > 0xffffull) {
        *(uint32_t*)(uintptr_t)a1       = 0;   // +0x00: output state (0 = the boring/default state)
        *(uint32_t*)(uintptr_t)(a1 + 4) = 0;   // +0x04: dynamic-range mode (2 = HDR; 0 = SDR)
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

// sceVideoOutSetWindowModeMargins (MTxxrOCeSig — same brute-force naming): windowed-mode margin
// hint for the system compositor. No compositor here — accept and ignore. CONFIDENCE: HIGH (pure
// cosmetic setter; success unblocks the caller).
HLE(g_vo_set_window_mode_margins) { vo_argtrace("SetWindowModeMargins", a0,a1,a2,a3,a4,a5); return 0; }

// Diagnostic tracer for the (undocumented) libSceAgc / libSceAgcDriver calls: logs the NID, the guest
// callsite, and all six args (gated on PROSPER_GFXLOG). Behaviour is identical to the unimplemented
// stub — returns 0, changes no control flow — it is purely observable. This is the RE bootstrap for
// the real AGC->Vulkan work (M4/M5), NOT faked output.
//
// Per-NID identification: each NID is bound to a distinct template thunk `glog_thunk<I>` that knows
// its own name (kAgcNids[I]). The guest reaches the thunk via our stub's tail-jump, so at thunk entry
// [rsp] is the guest return address into eboot (base 0x400000000) — __builtin_return_address(0) gives
// the exact callsite. Result: `PROSPER_GFXLOG=1 boot_trace <dump>` emits a self-describing dataset
// (NID + callsite + args per call) — everything needed to RE the AGC object model.
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
        fprintf(stderr, "[gfx] libSceAgc::%s  from eboot+0x%llx  a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx\n",
                nid, (unsigned long long)((uint64_t)ra - 0x400000000ull),
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
    return glog_impl(kAgcNids[I], __builtin_return_address(0), a0, a1, a2, a3, a4, a5);
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
    R("sceVideoOutSetFlipRate", g_vo_close);    R("sceVideoOutAddFlipEvent", g_vo_addflipevent);
    R("sceVideoOutAddVblankEvent", g_vo_addvblankevent);
    R("sceVideoOutGetFlipStatus", g_vo_flipstatus);
    R("sceVideoOutGetResolutionStatus", g_vo_resstatus);
    R("sceVideoOutGetVblankStatus", g_vo_vblankstatus);
    R("sceVideoOutGetDeviceCapabilityInfo", g_vo_devcap);
    R("sceVideoOutRegisterBuffers", g_vo_close);R("sceVideoOutSetBufferAttribute", g_vo_close);
    // PS5 "2"/query variants — the 5 previously-unimplemented VideoOut NIDs (resolved via shadPS4
    // aerolib + Kyty VideoOut.cpp). Registered by raw NID (PS5-specific, not in our name DB).
    RN("Nv8c-Kb+DUM", g_vo_is_output_supported);   // sceVideoOutIsOutputSupported
    RN("PjS5uASwcV8", g_vo_set_buffer_attribute2);  // sceVideoOutSetBufferAttribute2
    RN("rKBUtgRrtbk", g_vo_register_buffers2);       // sceVideoOutRegisterBuffers2
    RN("utPrVdxio-8", g_vo_get_output_status);        // sceVideoOutGetOutputStatus
    RN("w0hLuNarQxY", g_vo_configure_output);          // sceVideoOutConfigureOutput
    RN("U2JJtSqNKZI", g_vo_get_event_id);              // sceVideoOutGetEventId (#210)
    RN("MTxxrOCeSig", g_vo_set_window_mode_margins);   // sceVideoOutSetWindowModeMargins (#210)
    RN("b0xyllnVY-I", g_gnm_add_eq_event);             // sceGnmAddEqEvent / GraphicsAddEqEvent (GPU EOP)
    #undef R
    #undef RN
}

} // namespace prosper
