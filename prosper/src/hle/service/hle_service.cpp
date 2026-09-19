// hle_service.cpp — HLE of PS5 system services (user, NP/online, mouse, app content,
// dialogs). Bring-up policy: openers return a valid positive handle; queries zero their
// output struct and report a sane "not signed in / no device" state and success, so the
// game gets consistent values instead of uninitialized memory.
// (Game-controller input — libScePad — moved to hle_pad.cpp with a real host backend.)
#include "hle/dispatch/dispatch.hpp"
#include "hle/service/hle_addcontent.hpp"
#include "hle/fs/save_paths.hpp"   // per-title save roots (#2734)
#include "hle/fs/save_param.hpp"   // the save's parameter block (#2786)
#include "hle/util/hle_json2.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"   // libkernel error encoding (libSceRandom reject arms)
#include "diagnostics/env_numeric.hpp"   // #3267: a typo must not unregister a default-ON NID family
#include "hle/dispatch/callback_fs.hpp"
#include "hle/input/ime_input.hpp"
#include "hle/service/platform_ui.hpp"
#include "hle/video/video_backend.hpp"   // sceAvPlayer -> host hardware-decode backend (#705)
#include "hle/video/h264_sps.hpp"        // SPS/VUI extraction for GetPictureInfo (#2898)
#include "gpu/texture/guest_texture_layout.hpp" // exact HLE-produced sampled-linear layouts
#include "host/platform/posix_shim.hpp"   // Darwin process_vm_readv shim + asm portability
#include "host/image/boot_program.hpp"  // guest_module_name: is a callback target guest code?
#include "host/platform/lifecycle.hpp"  // cooperative stop when the guest reports its own crash (#3119)
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#include <deque>
#include <filesystem>
#include <mutex>
#include <new>          // std::bad_alloc — the guest file-replacement buffer is guest-sized (#1955)
#include <set>
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
#include <sys/random.h> // getentropy: the host CSPRNG behind sceRandomGetRandomNumber
#include <unistd.h>
#endif
#include "hle/service/service_trace.hpp"
#include "hle/service/hle_handles.hpp"
#ifdef _WIN32
#include <bcrypt.h>     // BCryptGenRandom (prosper_core already links bcrypt on Windows)
#endif

namespace prosper {

#ifdef _WIN32
extern "C" uint64_t prosper_call_guest_sysv4(uint64_t fn, uint64_t a0, uint64_t a1,
                                               uint64_t a2, uint64_t a3);
#endif

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define PW(x) ((void*)(uintptr_t)(x))


// Diagnostic logging for the Sony service families in this file, gated on PROSPER_SVCLOG=1 (same
// pattern as PROSPER_FILELOG/[file]). Dumps call args and a bounded hexdump of pointer-shaped args
// so PS5-only ABIs with no Kyty/shadPS4 reference can be pinned from live captures instead of
// guessed.
namespace {
}

// PS5 Game Intent is how the shell starts a title at a selected activity instead of its default
// entry point. A host frontend can model that user action by setting an activity id before boot;
// without it the console truthfully has no pending intent. Keep the payload opaque to the guest:
// titles consume it through sceNpGameIntentGetPropertyValueString, as on the console.
// CONFIDENCE: HIGH — ABI/error contracts match Kyty; event id, type, key, and offsets are also
// independently exercised by Sonic's eboot and its launchActivity declaration in param.json.
namespace {
constexpr uint64_t NP_GAME_INTENT_ERROR_INVALID_ARGUMENT = 0x80553804ull;
constexpr uint64_t NP_GAME_INTENT_ERROR_INTENT_NOT_FOUND = 0x80553806ull;
constexpr uint64_t NP_GAME_INTENT_ERROR_VALUE_NOT_FOUND  = 0x80553807ull;
constexpr size_t NP_GAME_INTENT_TYPE_OFFSET     = 12;
constexpr size_t NP_GAME_INTENT_TYPE_SIZE       = 33;
constexpr size_t NP_GAME_INTENT_DATA_OFFSET     = 308;
constexpr size_t NP_GAME_INTENT_DATA_SIZE       = 16392;
constexpr uint32_t SYSTEM_SERVICE_EVENT_GAME_INTENT = 0x10000017u;

std::atomic<bool> g_gameintent_initialized{false};
std::atomic<bool> g_gameintent_event_delivered{false};
std::atomic<uint64_t> g_gameintent_data_ptr{0};

// Bound host-controlled text to the opaque intent payload capacity. The property accessor separately
// checks the caller's buffer, so a title with a smaller value limit gets an error, never truncation.
const char* gameintent_activity_id(size_t* length = nullptr) {
    const char* value = getenv("PROSPER_GAME_INTENT_ACTIVITY_ID");
    if (!value) return nullptr;
    size_t n = 0;
    while (n < NP_GAME_INTENT_DATA_SIZE && value[n]) ++n;
    if (n == 0 || n == NP_GAME_INTENT_DATA_SIZE) return nullptr;
    if (length) *length = n;
    return value;
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
// sceUserServiceGetUserNumber(userId, number): each local user has a stable controller/user number.
// Sonic imports the PS5 NID directly and consumes the out-param during its user bootstrap.  Returning
// success from the fallback stub without writing it leaves the caller's stack sentinel in place and
// prevents the frontend from reaching its pad/audio initialization.  This single-user model exposes
// default user 1 as number 1, consistent with the rest of the UserService handlers above.
HLE(s_user_number)    { if (a1) *(int32_t*)PW(a1) = 1; return 0; }
// --- libSceVideodec2 ---------------------------------------------------------------------------
// These ABI layouts/NIDs are the PS5 3.20 interfaces.  Decoding is intentionally a no-picture
// implementation for now: it consumes each access unit, reports a valid lifecycle, and lets the
// movie reach EOF instead of faking video pixels.  Critically, every successful query/open writes
// all output fields.  The former generic success stubs left a null compute queue and crashed libc
// at +0xfe66 on its first queue use.
//
// WHO ACTUALLY EXERCISES THIS, as recorded rather than as assumed.  A header line here used to read
// "Sonic's CRI Mana movie backend uses the native VideoDec2 compute queue and decoder lifecycle."
// It came from #1368 (8b37be95) — the same commit that added the unexplained compute_queue check
// below and an equally unsupported "Sonic Origins' CRI Mana backend passes it", which was already
// withdrawn.  No Videodec2 evidence for Sonic exists anywhere in docs/ or COMPATIBILITY.md, so it is
// withdrawn on the same basis rather than left to be inherited a third time.  What IS recorded:
//   - QueryComputeMemoryInfo only — Dragon Quest VII Reimagined and the UE4 bring-up.
//   - The full decoder lifecycle — Tales of Graces f Remastered (PPSA19991) via criMvPly, and only
//     after #1687; before that no title in this repo's history had ever reached CreateDecoder.
namespace {
// The current no-picture backend has no hardware decoder workspace: it only needs a stable,
// caller-owned address for the opaque compute-queue identity.  Keep one guest page so clients that
// require a non-null allocation can follow the real query/allocate lifecycle without reserving the
// 16 MiB placeholder used by software-decoding implementations.  That placeholder exhausted
// Sonic Origins' 80 MiB CRI CPU/GPU pool before its movie workers could allocate their own buffers.
}
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
    svc_log("sceUserServiceGetEvent", a0,a1,a2,a3,a4,a5);
    if (a0 && delivered.exchange(1) == 0) { int32_t* ev = (int32_t*)PW(a0); ev[0] = 0; ev[1] = 1; return 0; }
    return 0x80960007ull;   // SCE_USER_SERVICE_ERROR_NO_EVENT
}
// sceSystemServiceReceiveEvent(SceSystemServiceEvent* ev): the system event stream (resume, launch-app,
// entitlement-update, share-menu, ...). Unregistered, it fell to the return-0 stub = "an event was
// received" while leaving the 8196-byte out-struct (4-byte eventType + 8192-byte union) uninitialized ->
// the guest dispatched on a garbage eventType (same harmful class as s_user_getevent above; imported by
// both Unity targets). The truthful idle answer is NO_EVENT with nothing written. shadPS4
// systemservice.cpp: NO_EVENT = 0x80A10004, PARAMETER (ev==NULL) = 0x80A10003. When a host activity
// was selected, a real shell instead delivers SCE_SYSTEM_SERVICE_EVENT_GAME_INTENT once; the title
// then obtains the opaque launchActivity payload through libSceNpGameIntent.
HLE(s_sysservice_receiveevent) {
    svc_log("sceSystemServiceReceiveEvent", a0,a1,a2,a3,a4,a5);
    if (!a0) return 0x80A10003ull;
    if (g_gameintent_initialized.load(std::memory_order_acquire) && gameintent_activity_id()) {
        bool expected = false;
        if (g_gameintent_event_delivered.compare_exchange_strong(expected, true)) {
            const uint32_t event_type = SYSTEM_SERVICE_EVENT_GAME_INTENT;
            if (svc_write_bytes(a0, &event_type, sizeof(event_type))) return 0;
            g_gameintent_event_delivered.store(false);
            return 0x80A10003ull;
        }
    }
    return 0x80A10004ull;
}
HLE(s_ok)             { return 0; }


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
// The local network libraries allocate opaque contexts even on a disconnected console; connection
// state is reported separately through NetCtl/NP. Returning generic success (0) from these ID-returning
// constructors instead creates an invalid context and makes their owner's initialization fail. The
// signatures and positive-return contract agree with the PS5 3.20 symbol table and the independently
// implemented SDK surface; no online identity or connectivity is fabricated here.
namespace {
std::atomic<int32_t> g_net_pool_id{1};
std::atomic<int32_t> g_ssl_context_id{1};
std::atomic<int32_t> g_npweb_context_id{1};
std::atomic<int32_t> g_npweb_user_context_id{1001};
}
HLE(s_net_pool_create) {
    svc_log("sceNetPoolCreate", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a0) || (int32_t)a1 <= 0) return 0x80410118ull; // SCE_NET_ERROR_ENFILE
    return (uint64_t)(uint32_t)g_net_pool_id.fetch_add(1);
}
HLE(s_ssl_init) {
    svc_log("sceSslInit", a0,a1,a2,a3,a4,a5);
    if (!a0) return 0x8094000cull; // SCE_SSL_ERROR_OUT_OF_SIZE
    return (uint64_t)(uint32_t)g_ssl_context_id.fetch_add(1);
}
HLE(s_npweb_init) {
    svc_log("sceNpWebApi2Initialize", a0,a1,a2,a3,a4,a5);
    return (uint64_t)(uint32_t)g_npweb_context_id.fetch_add(1);
}
HLE(s_npweb_create_user_context) {
    svc_log("sceNpWebApi2CreateUserContext", a0,a1,a2,a3,a4,a5);
    return (uint64_t)(uint32_t)g_npweb_user_context_id.fetch_add(1);
}
HLE(s_netctl_getresult) {
    svc_log("sceNetCtlGetResult", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a1)) return 0x80412103ull; // SCE_NET_CTL_ERROR_INVALID_ADDR
    *(int32_t*)PW(a1) = 0;
    return 0;
}
// sceNpHasSignedUp(userId, bool* hasSignedUp): Sonic passes an ODD output address (..cdf), proving
// the result is one byte rather than int32. An offline initial user has no NP signup in this headless
// profile. Success + false lets the caller select its signed-out branch without inventing an online
// identity; the old success-with-untouched-stack answer made that branch random. CONFIDENCE: HIGH
// on the ABI and byte width (live PS5 title trace), MED on success+false for the offline profile.
HLE(s_np_has_signed_up) {
    svc_log("sceNpHasSignedUp", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a1)) return 0x80550003ull; // SCE_NP_ERROR_INVALID_ARGUMENT
    *(uint8_t*)PW(a1) = 0;
    return 0;
}

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
// --- IME keyboard event injection (issue #1093) -------------------------------------------------
// PPSA02664 (Alex Kidd) reads its "press any button" / menu input ONLY through the IME keyboard
// path: it opens a keyboard and calls sceImeUpdate(handler) every frame, and never touches
// libScePad or sceUserService. sceImeUpdate must invoke handler(arg, &event) for each queued
// keyboard event; the old stub returned without calling, delivered no input, and the title never
// advanced (injected pad input was inert — the game wasn't reading the pad).
//
// Event ABI derived directly from the guest handler at eboot+0xf2c540 (PPSA02664 disassembly):
//   handler(rdi=arg, rsi=SceImeEvent*)
//   SceImeEvent:  +0x00 u32 id   (0x101 = KEY_DOWN, 0x102 = KEY_UP; 0x103..0x106 = other kbd events)
//                 +0x08 u16 keycode  (USB HID usage id; indexes the guest's keycode->bit table)
//                 +0x0a u16 (modifier/char)   +0x0c u32 (status)
//   KEY_DOWN sets the guest's current + newly-pressed keyboard bitmasks; KEY_UP clears them.
// HID keycodes verified against the guest's table: Enter=0x28, Space=0x2c, Z=0x1d, A=0x04.
// CONFIDENCE: HIGH on the id/keycode/offset layout (read from the guest handler + its HID table).
namespace {
struct ImeKeyEvent { uint16_t hid; bool down; };
// OrbisImeKeyboardParam, shared by the PS4 compatibility API exposed on PS5. The callback context is
// supplied at open time, while sceImeUpdate supplies the callback function used for that pump. Keeping
// only the callback and dropping `arg` happened to work for handlers that ignored rdi, but faults any
// normal C++ thunk that expects its object pointer there.
struct ImeKeyboardParam {
    uint32_t option;
    uint32_t reserved1;
    uint64_t arg;
    uint64_t handler;
    uint64_t reserved2;
};
static_assert(sizeof(ImeKeyboardParam) == 0x20 && offsetof(ImeKeyboardParam, arg) == 0x08 &&
              offsetof(ImeKeyboardParam, handler) == 0x10);
struct ImeKeyboardContext {
    bool open = false;
    int32_t user_id = -1;
    uint64_t arg = 0;
    uint64_t handler = 0;
};
std::mutex g_ime_mx;
std::deque<ImeKeyEvent> g_ime_queue;
ImeKeyboardContext g_ime_keyboard;

void ime_deliver(uint64_t handler, uint64_t arg, const ImeKeyEvent& ev, uint64_t guest_fs) {
    if (!handler) return;
    alignas(16) uint8_t e[0x40] = {0};
    *(uint32_t*)(e + 0x00) = ev.down ? 0x101u : 0x102u;
    *(uint16_t*)(e + 0x08) = ev.hid;
    // The guest handler is guest code that reads guest TLS (its per-thread allocator etc.). But
    // sceImeUpdate is an HLE handler, so the import stub already swapped this thread to HOST %fs before
    // it ran (#1155). Calling the guest handler directly here therefore ran it on host %fs, so its
    // thread-local reads hit host glibc garbage and the first allocation it drives (the menu->gameplay
    // transition on a key press) faulted at a near-null address — Evergate's SIGSEGV at eboot+0xaf8431
    // (#1286). Restore the caller's guest %fs (recovered from the import-stub frame) for the duration of
    // the call, then swap back to host %fs. Linux-only: only Linux swaps hardware %fs at the import
    // boundary (guest_fs is 0 on Windows/macOS -> unchanged direct call).
#if defined(__linux__) && !defined(__APPLE__)
    uint64_t saved_fs = 0; bool swapped = false;
    if (guest_fs) {
        __asm__ volatile("rdfsbase %0" : "=r"(saved_fs));
        __asm__ volatile("wrfsbase %0" : : "r"(guest_fs));
        swapped = true;
    }
#else
    (void)guest_fs;
#endif
    // The handler is GUEST code, so it reads its arguments per the SysV ABI (rdi, rsi). On Windows the
    // host is MS x64 (rcx, rdx) and PROSPER_SYSV_ABI is empty on every platform (dispatch.hpp:27) --
    // the guest<->host conversion lives in the import-stub trampoline, which only covers the
    // guest->host direction. A raw call here therefore handed the guest whatever happened to be in
    // rdi/rsi. Observed on Blue Prince (PPSA25009): the guest read rsi == 0x28, the autokey's own
    // default Enter HID, as a pointer and faulted at eboot+0x137c063 (`mov eax,[rsi]`), after which
    // the title spun on `sce::Agc::suspendPoint` forever with a black screen. Use the same host->guest
    // trampoline the AvPlayer callbacks in this file already use.
    // Only GUEST code needs the trampoline. prosper's own tests install a HOST function as the
    // handler to observe delivery and the event shape; that address is already the host ABI and must
    // not be entered through the guest call path (doing so segfaults ime_input). Classify by module
    // aperture, the same way the rest of the tree names guest addresses.
    // CONFIDENCE: HIGH -- an IL2CPP title's handler is AOT code in the eboot or a PRX, and the
    // runtime-PRX pool is labelled too; there is no observed guest handler outside a module.
#ifdef _WIN32
    if (std::strcmp(prosper::guest_module_name((uint64_t)(uintptr_t)handler), "mapped/host") != 0) {
        prosper_call_guest_sysv4((uint64_t)(uintptr_t)handler, arg, (uint64_t)(uintptr_t)e, 0, 0);
    } else {
        // Unclassified target: prosper's own test double, or a guest handler somewhere this
        // classifier does not know about. The second case would take the host ABI and silently hand
        // the guest wrong arguments -- the exact defect this code fixes -- so say so rather than
        // reintroduce it quietly. Bounded; a host test double trips it at most a few times.
        static std::atomic<int> unclassified{0};
        if (unclassified.fetch_add(1) < 4)
            fprintf(stderr, "[ime] handler 0x%llx is outside every guest module aperture; calling it "
                            "directly. If this is guest code it will receive MS-x64 arguments and "
                            "misread them (#2136).\n", (unsigned long long)(uintptr_t)handler);
        ((void (PROSPER_SYSV_ABI *)(uint64_t, void*))(uintptr_t)handler)(arg, e);
    }
#else
    ((void (PROSPER_SYSV_ABI *)(uint64_t, void*))(uintptr_t)handler)(arg, e);
#endif
#if defined(__linux__) && !defined(__APPLE__)
    if (swapped) __asm__ volatile("wrfsbase %0" : : "r"(saved_fs));
#endif
}

// PROSPER_IME_AUTOKEY=1: deliver a repeating Enter down/up pulse so headless routes advance a
// "press any button" / menu that reads the IME keyboard. Diagnostic/verification lever; the real
// per-key input comes from a frontend via ime_push_key (below). PROSPER_IME_AUTOKEY_HID overrides
// the HID usage (default 0x28 Enter) and accepts a comma-separated LIST (e.g. "0x28,0x2c,0x1d") that
// is cycled one key per press-window — so a single headless run can probe which key a dialog accepts.
// Cadence: down, then up two ticks later, then the next key in the list two windows later.
void ime_autokey_tick(uint64_t handler, uint64_t arg, uint64_t guest_fs) {
    static const int on = [] { const char* e = getenv("PROSPER_IME_AUTOKEY");
                               return e && strtol(e, nullptr, 0) != 0 ? 1 : 0; }();
    if (!on) return;
    static const std::vector<uint16_t> hids = [] {
        std::vector<uint16_t> v;
        const char* e = getenv("PROSPER_IME_AUTOKEY_HID");
        if (e && *e) for (const char* p = e; *p; ) {
            char* end = nullptr; long val = strtol(p, &end, 0);
            if (end == p) break;
            v.push_back((uint16_t)val);
            p = (*end == ',') ? end + 1 : end;
        }
        if (v.empty()) v.push_back(0x28);   // default Enter
        return v;
    }();
    static std::atomic<uint64_t> tick{0};
    uint64_t t = tick.fetch_add(1);
    const uint16_t hid = hids[(t / 90) % hids.size()];
    if ((t % 90) == 0) {
        ime_deliver(handler, arg, {hid, true}, guest_fs);
        if (svclog()) fprintf(stderr, "[ime-autokey] deliver hid=0x%x down\n", hid);
    } else if ((t % 90) == 2) {
        ime_deliver(handler, arg, {hid, false}, guest_fs);
    }
}

// PROSPER_IME_SCRIPT: deterministic headless keyboard input, anchored to the first sceImeUpdate
// call carrying a non-null handler. Entries are `fN:HID` or `fA-B:HID`, separated by semicolons or
// newlines; HID accepts decimal or 0x-prefixed USB usage ids. A point entry is held for two update
// ticks. Prefix the value with '@' to load the same text from a route file. Unlike AUTOKEY this is a
// finite route: transitions are emitted only when a key enters/leaves an active window.
std::string ime_script_text(const char* source, std::string& error) {
    if (!source || !*source) return {};
    if (*source != '@') return source;
    FILE* f = fopen(source + 1, "rb");
    if (!f) { error = std::string("cannot open route file: ") + (source + 1); return {}; }
    std::string text;
    char buf[4096];
    while (size_t n = fread(buf, 1, sizeof buf, f)) text.append(buf, n);
    if (ferror(f)) error = std::string("cannot read route file: ") + (source + 1);
    fclose(f);
    return text;
}

bool ime_script_u64(const char* p, char** tail, uint64_t& value) {
    if (!p || !((*p >= '0' && *p <= '9'))) { if (tail) *tail = (char*)p; return false; }
    errno = 0;
    char* end = nullptr;
    const int base = p[0] == '0' && (p[1] == 'x' || p[1] == 'X') ? 16 : 10;
    const unsigned long long parsed = strtoull(p, &end, base);
    if (end == p || errno == ERANGE || parsed > UINT64_MAX) { if (tail) *tail = end; return false; }
    if (tail) *tail = end;
    value = (uint64_t)parsed;
    return true;
}

bool parse_ime_script_impl(const std::string& text, std::vector<ImeScriptWindow>& out,
                           std::string& error) {
    out.clear();
    size_t pos = 0;
    while (pos < text.size()) {
        size_t end = text.find_first_of(";\n\r", pos);
        if (end == std::string::npos) end = text.size();
        std::string item = text.substr(pos, end - pos);
        pos = end + 1;
        if (size_t hash = item.find('#'); hash != std::string::npos) item.resize(hash);
        size_t first_nonspace = item.find_first_not_of(" \t");
        if (first_nonspace == std::string::npos) continue;
        size_t last_nonspace = item.find_last_not_of(" \t");
        item = item.substr(first_nonspace, last_nonspace - first_nonspace + 1);
        if (item.empty()) continue;
        if (item[0] != 'f') { error = "entry must start with f: " + item; return false; }
        const char* p = item.c_str() + 1;
        char* tail = nullptr;
        uint64_t first = 0;
        if (!ime_script_u64(p, &tail, first)) { error = "invalid frame anchor: " + item; return false; }
        uint64_t last = first == UINT64_MAX ? first : first + 1;
        // Point entries hold down for two update ticks (or one at the representational maximum).
        if (*tail == '-') {
            p = tail + 1;
            if (!ime_script_u64(p, &tail, last) || last < first) {
                error = "invalid frame range: " + item; return false;
            }
        }
        if (*tail != ':') { error = "missing HID separator: " + item; return false; }
        p = tail + 1;
        uint64_t hid = 0;
        if (!ime_script_u64(p, &tail, hid)) { error = "invalid HID usage: " + item; return false; }
        while (*tail == ' ' || *tail == '\t') ++tail;
        if (*tail || hid == 0 || hid > UINT16_MAX) {
            error = "invalid HID usage: " + item; return false;
        }
        out.push_back({first, last, (uint16_t)hid});
    }
    error.clear();
    return true;
}

class ImeScriptRuntime {
public:
    ImeScriptRuntime() {
        const char* source = getenv("PROSPER_IME_SCRIPT");
        if (!source || !*source) return;
        std::string error;
        const std::string text = ime_script_text(source, error);
        if (error.empty()) parse_ime_script_impl(text, entries_, error);
        if (!error.empty()) fprintf(stderr, "[ime] PROSPER_IME_SCRIPT: %s\n", error.c_str());
    }

    void tick(uint64_t handler, uint64_t arg, uint64_t guest_fs) {
        if (!handler || entries_.empty()) return;
        bool become_dispatcher = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::vector<uint16_t> next;
            for (const auto& e : entries_)
                if (tick_ >= e.first && tick_ <= e.last &&
                    std::find(next.begin(), next.end(), e.hid) == next.end()) next.push_back(e.hid);
            for (uint16_t hid : active_)
                if (std::find(next.begin(), next.end(), hid) == next.end())
                    pending_.push_back({handler, arg, guest_fs, {hid, false}});
            for (uint16_t hid : next)
                if (std::find(active_.begin(), active_.end(), hid) == active_.end())
                    pending_.push_back({handler, arg, guest_fs, {hid, true}});
            active_ = std::move(next);
            ++tick_;
            if (!dispatching_ && !pending_.empty()) { dispatching_ = true; become_dispatcher = true; }
        }
        if (!become_dispatcher) return;
        for (;;) {
            Pending transition;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (pending_.empty()) { dispatching_ = false; return; }
                transition = pending_.front();
                pending_.pop_front();
            }
            // Never retain the runtime mutex across a call into guest code. A reentrant tick queues
            // behind transitions already published by this tick; this outer dispatcher drains it.
            ime_deliver(transition.handler, transition.arg, transition.ev, transition.guest_fs);
        }
    }

private:
    struct Pending { uint64_t handler = 0, arg = 0, guest_fs = 0; ImeKeyEvent ev{}; };
    std::vector<ImeScriptWindow> entries_;
    std::vector<uint16_t> active_;
    std::deque<Pending> pending_;
    uint64_t tick_ = 0;
    bool dispatching_ = false;
    std::mutex mutex_;
};

void ime_script_tick(uint64_t handler, uint64_t arg, uint64_t guest_fs) {
    static ImeScriptRuntime runtime;
    runtime.tick(handler, arg, guest_fs);
}

// Shared sceImeUpdate body: fire the autokey pulse (if enabled) then drain the queued keys, dispatching
// each on the caller's guest %fs (guest_fs == 0 => no swap, the Windows/macOS path). The queue mutex is
// released before every guest handler call (never held across guest code).
void ime_update_run(uint64_t handler, uint64_t guest_fs) {
    uint64_t arg = 0;
    {
        std::lock_guard<std::mutex> lk(g_ime_mx);
        if (g_ime_keyboard.open) arg = g_ime_keyboard.arg;
    }
    ime_autokey_tick(handler, arg, guest_fs);        // handler = the guest event handler
    ime_script_tick(handler, arg, guest_fs);
    for (;;) {
        ImeKeyEvent ev;
        { std::lock_guard<std::mutex> lk(g_ime_mx);
          if (g_ime_queue.empty()) break;
          ev = g_ime_queue.front(); g_ime_queue.pop_front(); }
        ime_deliver(handler, arg, ev, guest_fs);
    }
}
} // namespace

bool parse_ime_script_route(const std::string& text, std::vector<ImeScriptWindow>& out,
                            std::string* error) {
    std::string local_error;
    const bool ok = parse_ime_script_impl(text, out, local_error);
    if (error) *error = local_error;
    return ok;
}

// Frontend seam (declared in ime_input.hpp): push a keyboard key (USB HID usage id) into the IME
// queue; drained on the next sceImeUpdate. Thread-safe (called from the frontend's input thread).
// Mirrors pad_set_backend's role for libScePad.
void ime_push_key(uint16_t hid_usage, bool down) {
    std::lock_guard<std::mutex> lk(g_ime_mx);
    g_ime_queue.push_back({hid_usage, down});
}

HLE(s_ime_kbd_open) {
    svc_log("sceImeKeyboardOpen", a0,a1,a2,a3,a4,a5);
    ImeKeyboardParam param{};
    // Some older bring-up callers omitted the optional keyboard parameter entirely; retain their
    // successful no-input behavior. A supplied pointer, however, must be readable because its `arg`
    // is part of the callback ABI and remains live until KeyboardClose/re-open.
    if (a1 && !svc_copy_bytes(a1, &param, sizeof param)) return 0x80BC0031ull;
    std::lock_guard<std::mutex> lk(g_ime_mx);
    g_ime_keyboard = {true, (int32_t)a0, param.arg, param.handler};
    return 0;
}
HLE(s_ime_kbd_close) {
    svc_log("sceImeKeyboardClose", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_ime_mx);
    if (g_ime_keyboard.open && g_ime_keyboard.user_id == (int32_t)a0) g_ime_keyboard = {};
    return 0;
}
// sceImeUpdate invokes the guest event handler (guest code), so it must restore the caller's guest %fs
// around each dispatch (see ime_deliver / #1286). On Linux/macOS the import stub swapped this thread to
// host %fs, so recover the caller's guest %fs from the import-stub frame via the entry-stack trampoline
// (as the AvPlayer callbacks do) and hand it to ime_update_run. PROSPER_ASM_TRAMPOLINE is POSIX-only, so
// Windows/MinGW (which never swaps hardware %fs at the import boundary) registers a plain handler that
// dispatches directly with guest_fs == 0. callback_guest_fs_from_entry_stack also returns 0 for any
// non-guest frame, so the plain-tail-jump macOS path is likewise an unchanged direct call.
#ifndef _WIN32
extern "C" uint64_t s_ime_update_c(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                   uint64_t a4, uint64_t a5, uint64_t entry_rsp) {
    svc_log("sceImeUpdate", a0,a1,a2,a3,a4,a5);
    ime_update_run(a0, prosper::callback_guest_fs_from_entry_stack(entry_rsp));
    return 0;
}
PROSPER_ASM_TRAMPOLINE(s_ime_update_entry, s_ime_update_c)
extern "C" void s_ime_update_entry();
#else
HLE(s_ime_update) {   // Windows/MinGW: no import-boundary %fs swap -> dispatch directly (guest_fs = 0)
    svc_log("sceImeUpdate", a0,a1,a2,a3,a4,a5);
    ime_update_run(a0, 0);
    return 0;
}
#endif
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
namespace {
constexpr uint64_t APP_CONTENT_ERROR_PARAMETER = 0x80D90002ull;
constexpr uint64_t APP_CONTENT_ERROR_BUSY = 0x80D90003ull;
constexpr uint64_t APP_CONTENT_ERROR_NOT_FOUND = 0x80D90005ull;
constexpr uint64_t APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT = 0x80D90007ull;
constexpr uint64_t NP_ENTITLEMENT_ERROR_PARAMETER = 0x817D0002ull;
constexpr uint64_t NP_ENTITLEMENT_ERROR_NO_ENTITLEMENT = 0x817D0007ull;
// dlc_emu 0.3's NP list export compares listNum against 0x9c4 before touching either output.
constexpr uint32_t NP_ENTITLEMENT_ADDCONT_LIST_MAX = 2500;

// Direct PS5 guest evidence pins these boundaries. Sonic allocates exactly 0x1c bytes per
// NpEntitlementAccess list entry and passes a 20-byte label to AddcontMount; Crisis Core independently
// allocates the same stride and consumes exactly 16 key bytes. AppContent's inherited info is the
// label+download-status prefix. Sonic directly
// reads `download_status` at +0x18 and accepts INSTALLED=4. Package types PSAC=2 / PSAL=3 and the
// NP entitlement errno space are pinned to the dlc_emu producer at 43ae9aa (CONFIDENCE: HIGH).
struct AppContentAddcontInfo {
    char entitlement_label[20];
    uint32_t download_status;
};
struct NpAddcontEntitlementInfo {
    char entitlement_label[20];
    uint32_t package_type;
    uint32_t download_status;
};
static_assert(sizeof(AppContentAddcontInfo) == 24, "SceAppContentAddcontInfo ABI");
static_assert(offsetof(AppContentAddcontInfo, download_status) == 20,
              "SceAppContentAddcontInfo status offset");
static_assert(sizeof(NpAddcontEntitlementInfo) == 28,
              "SceNpEntitlementAccessAddcontEntitlementInfo ABI");
static_assert(offsetof(NpAddcontEntitlementInfo, package_type) == 20 &&
              offsetof(NpAddcontEntitlementInfo, download_status) == 24,
              "Np add-content info field offsets");

bool appcontent_valid_label_text(const char* text, size_t size) {
    if (!size || size > 16) return false;
    for (size_t i = 0; i < size; ++i) {
        const char ch = text[i];
        if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
              (ch >= '0' && ch <= '9')))
            return false;
    }
    return true;
}

bool appcontent_read_label(uint64_t address, std::string& label) {
    if (!svc_ptrish(address)) return false;
    char raw[20];
    if (!svc_copy_bytes(address, raw, sizeof(raw))) return false;
    size_t size = 0;
    while (size < 17 && raw[size]) ++size;
    if (size == 17 || raw[17] || raw[18] || raw[19] ||
        !appcontent_valid_label_text(raw, size)) return false;
    label.assign(raw, size);
    return true;
}

AppContentAddcontInfo appcontent_info(const InstalledAddcontent& entry) {
    AppContentAddcontInfo info{};
    std::memcpy(info.entitlement_label, entry.entitlement_label.c_str(),
                entry.entitlement_label.size() + 1);
    info.download_status = entry.download_status;
    return info;
}

NpAddcontEntitlementInfo npent_info(const InstalledAddcontent& entry) {
    NpAddcontEntitlementInfo info{};
    std::memcpy(info.entitlement_label, entry.entitlement_label.c_str(),
                entry.entitlement_label.size() + 1);
    info.package_type = entry.package_type;
    info.download_status = entry.download_status;
    return info;
}

const InstalledAddcontent* find_addcontent(const AddcontentInventorySnapshot& inventory,
                                           uint32_t service_label,
                                           const std::string& entitlement_label) {
    for (const InstalledAddcontent& entry : inventory.entries) {
        if ((entry.service_label == -1 ||
             static_cast<uint32_t>(entry.service_label) == service_label) &&
            entry.entitlement_label == entitlement_label) return &entry;
    }
    return nullptr;
}

template <typename GuestInfo, typename MakeInfo>
uint64_t addcontent_info_list(uint64_t list_address, uint64_t list_num,
                              uint64_t hit_num_address, uint32_t service_label,
                              uint64_t parameter_error, uint64_t inventory_error,
                              uint64_t max_list_num, MakeInfo make_info) {
    if (list_num > max_list_num) return parameter_error;
    const AddcontentInventorySnapshot inventory = addcontent_inventory_snapshot();
    if (inventory.state == AddcontentInventoryState::Invalid) return inventory_error;

    std::vector<const InstalledAddcontent*> matching;
    for (const InstalledAddcontent& entry : inventory.entries)
        if (entry.service_label == -1 ||
            static_cast<uint32_t>(entry.service_label) == service_label)
            matching.push_back(&entry);

    const uint32_t capacity = static_cast<uint32_t>(list_num);
    const uint32_t total = static_cast<uint32_t>(matching.size());
    // Both producer list APIs define NULL-list or zero-capacity as a count query, where hitNum is
    // mandatory. On a real list call hitNum is optional.
    if (list_address == 0 || capacity == 0) {
        if (!svc_ptrish(hit_num_address) ||
            !svc_write_bytes(hit_num_address, &total, sizeof(total))) return parameter_error;
        return 0;
    }
    const uint32_t written = static_cast<uint32_t>(std::min<size_t>(capacity, matching.size()));
    if (!svc_ptrish(list_address) ||
        written > (UINT64_MAX - list_address) / sizeof(GuestInfo)) return parameter_error;
    for (uint32_t i = 0; i < written; ++i) {
        const GuestInfo info = make_info(*matching[i]);
        if (!svc_write_bytes(list_address + uint64_t{i} * sizeof(GuestInfo),
                             &info, sizeof(info))) return parameter_error;
    }
    // The dlc_emu producer/reference writes min(listNum,total) entries but reports the full total in
    // hitNum for both AppContent and NpEntitlementAccess. Pin: drakmor/dlc_emu@43ae9aa,
    // dlc_content.cpp:1941-1965 and 2357-2385. This lets a short caller discover and retry the count.
    if (hit_num_address && !svc_write_bytes(hit_num_address, &total, sizeof(total)))
        return parameter_error;
    return 0;
}
} // namespace

// sceAppContentAppParamGetInt(SceAppContentAppParamId paramId, int32_t* value)
//   paramId 0 = SKU_FLAG, 1..4 = USER_DEFINED_PARAM_1..4.
//
// Every one of these is a property the installed application declares about ITSELF in its own
// sce_sys/param.json, so all of them are answered from that local declaration and nothing else. The
// previous stub hardcoded SKU_FLAG to FULL and answered every user-defined param with 0, and it is
// the second half that gated GTA V's (PPSA04263) story mode: the title reads USER_DEFINED_PARAM_1 —
// which its own param.json declares as 9 — stores it, and tests bit 3 to decide that its installed
// content needs no online entitlement lookup. Answering 0 cleared that bit and sent it down the
// online path instead, where sceNpEntitlementAccessGetSkuFlag left the guest's own pre-seeded TRIAL
// value standing and the title offered story mode as a purchase. That chain was only half the story:
// the param read was itself skipped, because sceSysmoduleIsLoaded reported every module loaded and
// GTA V therefore never ran its own sceAppContentInitialize. Fixed separately in #2002 (the query
// now answers from prosper's real load history); both halves are needed, and a three-arm A/B on
// #1873 shows neither alone opens the gate. The declared values matter well
// beyond that gate: Crisis Core (PPSA07809) uses USER_DEFINED_PARAM_2 as a bounded content-variant
// index, and Little Nightmares II/III use USER_DEFINED_PARAM_1 as an index into their own
// add-content list.
//
// param.json omits a userDefinedParamN the publishing tool left at its default, so an absent key is a
// declared zero. With no parseable param.json there is no declaration at all and the query fails
// rather than inventing one — a value invented here is indistinguishable from a real declaration to
// every caller. CONFIDENCE: HIGH.
HLE(s_appcontent_int) {
    // Logged because this is the head of a causal chain that is otherwise invisible: a title stores
    // a user-defined param in its own state and acts on it much later, so "what did prosper answer
    // here" is the question a stalled content gate needs answered first.
    svc_log("sceAppContentAppParamGetInt", a0,a1,a2,a3,a4,a5);
    const int32_t param_id = (int32_t)a0;
    if (param_id < 0 || param_id > 4 || !svc_ptrish(a1)) return APP_CONTENT_ERROR_PARAMETER;
    const AppParamDeclaration decl = app_param_declaration();
    if (!decl.declared) return APP_CONTENT_ERROR_NOT_FOUND;
    int32_t value = 0;
    if (param_id == 0) {
        if (!decl.sku_flag) return APP_CONTENT_ERROR_NOT_FOUND;
        value = (int32_t)*decl.sku_flag;
    } else {
        value = decl.user_param[param_id - 1];
    }
    return svc_write_bytes(a1, &value, sizeof(value)) ? 0 : APP_CONTENT_ERROR_PARAMETER;
}

// sceSystemServiceParamGetInt(SceSystemServiceParamId paramId, int32_t* value): a0=paramId, a1=value.
// The system-settings a blanket-zero stub gives are mostly harmless, EXCEPT the LANGUAGE (paramId 1):
// value 0 = SCE_SYSTEM_PARAM_LANG_JAPANESE, so games localise their UI/text to Japanese. Default to
// US English (SCE_SYSTEM_PARAM_LANG_ENGLISH_US = 1) instead. Configurable via PROSPER_SYS_LANG, which
// takes the Sony SCE_SYSTEM_PARAM_LANG_* enum (0=ja, 1=en-US, 2=fr, 4=de, 5=it, 9=ko, 18=en-GB, …).
// Date/time-format params (2/3) default to the US convention: date enum 2 = MM/DD/YYYY and
// time enum 0 = 12-hour. Date enum 0 is YYYYMMDD, despite the old comment claiming it was US.
HLE(s_syss_param_int) {
    int32_t paramId = (int32_t)a0;           // a0 = paramId, a1 = int32_t* value out (matches s_appcontent_int)
    int32_t val = 0;
    if (paramId == 1) {                      // SCE_SYSTEM_SERVICE_PARAM_ID_LANG
        val = 1;                             // SCE_SYSTEM_PARAM_LANG_ENGLISH_US
        if (const char* e = getenv("PROSPER_SYS_LANG")) val = (int32_t)strtol(e, nullptr, 0);
    } else if (paramId == 2) {               // SCE_SYSTEM_SERVICE_PARAM_ID_DATE_FORMAT
        val = 2;                             // SCE_SYSTEM_PARAM_DATE_FORMAT_MMDDYYYY
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
namespace {
// ErrorDialog is system-owned UI. Platform glue uses SystemService status' background -> foreground
// transition to resume requests after it closes, independently of the CommonDialog FINISHED value.
// A real backend may stay active across many status samples; the headless backend dismisses during
// Open and must still expose one observable enter/return pair instead of collapsing both edges into
// a permanent foreground value. Oregon Trail #1606 positive-controlled this exact contract twice:
// the account-completion writer ran only after both SystemService samples. No title address or
// title identity participates in this state machine.
enum ErrorDialogBackgroundPhase : unsigned {
    ErrorDialogIdle = 0,
    ErrorDialogHeadlessEnter,
    ErrorDialogHeadlessReturn,
    ErrorDialogBackedEnter,
    ErrorDialogBackedActive,
    ErrorDialogBackedReturn,
};
std::atomic<unsigned> g_error_dialog_background_phase{ErrorDialogIdle};

void begin_error_dialog_background(bool backed) {
    g_error_dialog_background_phase.store(
        backed ? ErrorDialogBackedEnter : ErrorDialogHeadlessEnter, std::memory_order_release);
}

void finish_error_dialog_background() {
    unsigned phase = g_error_dialog_background_phase.load(std::memory_order_acquire);
    for (;;) {
        unsigned next = phase;
        if (phase == ErrorDialogBackedEnter) next = ErrorDialogHeadlessEnter;
        else if (phase == ErrorDialogBackedActive) next = ErrorDialogBackedReturn;
        else return;
        if (g_error_dialog_background_phase.compare_exchange_weak(
                phase, next, std::memory_order_acq_rel, std::memory_order_acquire)) return;
    }
}

uint8_t sample_error_dialog_background() {
    unsigned phase = g_error_dialog_background_phase.load(std::memory_order_acquire);
    for (;;) {
        unsigned next = phase;
        uint8_t background = 0;
        const char* edge = nullptr;
        switch (phase) {
        case ErrorDialogHeadlessEnter:
            next = ErrorDialogHeadlessReturn; background = 1; edge = "enter"; break;
        case ErrorDialogHeadlessReturn:
            next = ErrorDialogIdle; background = 0; edge = "return"; break;
        case ErrorDialogBackedEnter:
            next = ErrorDialogBackedActive; background = 1; edge = "enter"; break;
        case ErrorDialogBackedActive:
            return 1;
        case ErrorDialogBackedReturn:
            next = ErrorDialogIdle; background = 0; edge = "return"; break;
        default:
            return 0;
        }
        if (!g_error_dialog_background_phase.compare_exchange_weak(
                phase, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue;
        }
        fprintf(stderr, "[svc] ErrorDialog system lifecycle background=%u phase=%s\n",
                (unsigned)background, edge);
        return background;
    }
}
}

HLE(s_syss_getstatus) {
    auto* st = (uint8_t*)PW(a0);
    if (!st) return 0x80A10003ull;   // SYSTEM_SERVICE_ERROR_PARAMETER (Kyty Errno.h:382)
    memset(st, 0, 12);
    // event_num is the first field. Titles use it to decide how many times to call ReceiveEvent;
    // advertising an activity only in ReceiveEvent leaves that function unreachable.
    if (g_gameintent_initialized.load(std::memory_order_acquire) &&
        !g_gameintent_event_delivered.load(std::memory_order_acquire) &&
        gameintent_activity_id()) {
        *(int32_t*)st = 1;
    }
    st[5] = sample_error_dialog_background();
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

// sceSystemServiceReportAbnormalTermination(cause) — the guest reporting a crash to the system.
//
// Registered so the line is READABLE. Unregistered, the dispatcher default fired as one opaque
// `unimplemented: libSceSystemService::3s8cHiCBKBE -> returning 0` among a dozen others, and
// resolving it needed a hand lookup against the PS5 3.20 stub table:
//   libSceSystemService.c:3976  sprx_dlsym(__handle, "3s8cHiCBKBE",
//                                          &__ptr_sceSystemServiceReportAbnormalTermination)
//
// IT DOES NOT MEAN THE TITLE IS FAILING NOW, and an earlier revision of this handler assumed it did.
// Independent review of #3120 opened five archived Tactics Ogre (PPSA03839) runs: the call is in ALL
// of them, always at line 8, during boot before the first frame -- including the run that then
// rendered 40,936 frames over 470 s and reached the tutorial battle, and the runs that stalled at
// ~frame 993. `prosper_on_unimpl` logs only on FIRST invocation, so that is the title's one call.
// It therefore has zero discriminating power: it is something this title does unconditionally at
// startup. One reading consistent with all five logs, offered as hypothesis and not finding: the API
// may report that the PREVIOUS session ended abnormally, and these runs are killed by `timeout`.
//
// So the default is to LOG AND CONTINUE. Stopping is opt-in via PROSPER_ABNORMAL_TERMINATION_STOP.
// Stopping by default would have been a live regression rather than a theoretical one: 23 of the 56
// local dumps import this NID, four of them rung-6 guarded (The Messenger, Blasphemous 2, Alex Kidd,
// Blue Prince), and `prosper_stop_requested()` is polled only by prosper-app -- so the snapshot
// matrix, which drives boot_trace and screenshot, cannot see the breakage at all. It would have
// failed only in a human's hands.
//
// SCOPE, stated because the opt-in is not uniform across frontends. prosper-app winds its run-loop
// down (main.cpp) and run_entry() never observes the stop, so the guest thread is not torn down. On
// screenshot and boot_trace nothing polls it, so the flag does not stop those runs -- and note it is
// not inert there either: prosper_wait_while_paused() returns false permanently once stop latches,
// which mutes the SDL3 audio sink in a frontend that keeps running.
//
// The return stays 0, identical to the dispatcher default, so no guest observes a different answer.
// The argument is logged raw; no layout for it is confirmed against any primary source.
// CONFIDENCE: HIGH on the NID identity. LOW on the argument, and on what a call implies about the
// title's state.
HLE(s_syss_report_abnormal_termination) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    // Read per call rather than caching in a static: this fires at most once in a run, so there is
    // nothing to optimise, and a cached sample would make the opt-in untestable in-process.
    const bool stop_run = getenv("PROSPER_ABNORMAL_TERMINATION_STOP") != nullptr;
    fprintf(stderr,
            "[prosper] GUEST REPORTED ABNORMAL TERMINATION: "
            "sceSystemServiceReportAbnormalTermination(0x%llx). The title has decided it is "
            "crashing -- this is the guest's own verdict, not prosper's. %s\n",
            (unsigned long long)a0,
            stop_run ? "PROSPER_ABNORMAL_TERMINATION_STOP is set; stopping the run."
                     : "Continuing: this title may call it at boot and run fine. Set "
                       "PROSPER_ABNORMAL_TERMINATION_STOP=1 to stop on it.");
    fflush(stderr);
    if (stop_run) prosper::prosper_request_stop();
    return 0;
}

// sceSystemServiceGetHdrToneMapLuminance(out). Kyty models the output as three floats in this order:
// max-full-frame, max, and min tone-map luminance. The imported function previously fell through to
// success-without-output, so display setup consumed poisoned values. Use Kyty's 80/1000/0 values as
// a deterministic fallback; this does not advertise or implement an HDR presentation path.
// CONFIDENCE: MED on the Kyty-derived 12-byte layout/order; LOW on the real-hardware values.
struct SysHdrToneMapLuminance {
    float max_full_frame;
    float max;
    float min;
};
static_assert(sizeof(SysHdrToneMapLuminance) == 12, "Kyty-modeled HDR tone-map luminance layout");
HLE(s_syss_hdr_luminance) {
    auto* luminance = (SysHdrToneMapLuminance*)PW(a0);
    if (!luminance) return 0x80A10003ull;   // SYSTEM_SERVICE_ERROR_PARAMETER
    luminance->max_full_frame = 80.0f;
    luminance->max = 1000.0f;
    luminance->min = 0.0f;
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

// sceAppContentGetAddcontInfoList(serviceLabel, Info* list, u32 listNum, u32* hitNum). A zero-capacity
// call is the count query; a shorter caller buffer receives only its capacity while hitNum reports the
// full matching count. With no manifest the retail no-DLC answer remains SUCCESS/hitNum=0. A malformed host
// inventory fails rather than masquerading as a valid empty install.
HLE(s_appcontent_addcont_list) {
    if (a0 > UINT32_MAX) return APP_CONTENT_ERROR_PARAMETER;
    return addcontent_info_list<AppContentAddcontInfo>(
        a1, a2, a3, static_cast<uint32_t>(a0), APP_CONTENT_ERROR_PARAMETER,
        APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT, UINT32_MAX, appcontent_info);
}

HLE(s_appcontent_addcont_info) {
    std::string label;
    if (a0 > UINT32_MAX || !appcontent_read_label(a1, label) || !svc_ptrish(a2))
        return APP_CONTENT_ERROR_PARAMETER;
    const AddcontentInventorySnapshot inventory = addcontent_inventory_snapshot();
    if (inventory.state == AddcontentInventoryState::Invalid)
        return APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT;
    const InstalledAddcontent* entry = find_addcontent(inventory, static_cast<uint32_t>(a0), label);
    if (!entry) return APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT;
    const AppContentAddcontInfo info = appcontent_info(*entry);
    return svc_write_bytes(a2, &info, sizeof(info)) ? 0 : APP_CONTENT_ERROR_PARAMETER;
}

HLE(s_appcontent_entitlement_key) {
    std::string label;
    if (a0 > UINT32_MAX || !appcontent_read_label(a1, label) || !svc_ptrish(a2))
        return APP_CONTENT_ERROR_PARAMETER;
    const AddcontentInventorySnapshot inventory = addcontent_inventory_snapshot();
    const InstalledAddcontent* entry = inventory.state == AddcontentInventoryState::Ready
        ? find_addcontent(inventory, static_cast<uint32_t>(a0), label) : nullptr;
    if (!entry) return APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT;
    return svc_write_bytes(a2, entry->entitlement_key.data(), entry->entitlement_key.size())
        ? 0 : APP_CONTENT_ERROR_PARAMETER;
}

HLE(s_appcontent_addcont_mount) {
    std::string label;
    if (a0 > UINT32_MAX || !appcontent_read_label(a1, label) || !svc_ptrish(a2))
        return APP_CONTENT_ERROR_PARAMETER;
    switch (addcontent_mount(static_cast<uint32_t>(a0), label, a2, svc_write_bytes)) {
    case AddcontentMountResult::NotFound: return APP_CONTENT_ERROR_NOT_FOUND;
    case AddcontentMountResult::Busy: return APP_CONTENT_ERROR_BUSY;
    case AddcontentMountResult::OutputError: return APP_CONTENT_ERROR_PARAMETER;
    case AddcontentMountResult::Mounted: return 0;
    }
    return APP_CONTENT_ERROR_NOT_FOUND;
}

// sceAppContentAddcontUnmount(const SceAppContentMountPoint* mountPoint) — release a claim taken by
// sceAppContentAddcontMount, identified by the 16-byte mount-point object that call handed back.
//
// Unregistered, this reached the dispatcher's `return 0`, which for this contract is SCE_OK: the
// guest was told the unmount succeeded while `entry.mounted` stayed set, so the NEXT mount of the
// same add-content returned BUSY and kept doing so forever. The title has no way to recover —
// from its point of view it released the entry — and locally-present content becomes unreachable.
// It presents as "the game refuses to load DLC it loaded a minute ago", arbitrarily far from here.
//
// This grants nothing. It clears a flag prosper itself set, and an unrecognised mount point frees
// nothing and returns NOT_FOUND. No ownership or entitlement answer is reachable from this path.
//
// The argument is the mount point rather than the entitlement label, mirroring the Mount pair
// (Mount takes serviceLabel + entitlementLabel and WRITES the mount point; Unmount takes the mount
// point back). CONFIDENCE: MED on the argument — it is the inherited PS4 AppContent shape and no
// local dump exercises a mount/unmount cycle, so it is not confirmed against a live guest. The
// residual risk is bounded by direction: if a title passes something else, the lookup fails and the
// guest gets NOT_FOUND, which is fail-visible and still strictly better than the silent success it
// gets today. What would settle it: a boot of a title whose dump declares `dlc_emu.ini`, with the
// argument registers traced at this NID.
HLE(s_appcontent_addcont_unmount) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    if (!svc_ptrish(a0)) return APP_CONTENT_ERROR_PARAMETER;
    // SceAppContentMountPoint is a 16-byte NUL-padded char array. Read it fault-contained: a bad
    // pointer must be a parameter error, not a fault, and not a success.
    char raw[16];
    if (!svc_copy_bytes(a0, raw, sizeof(raw))) return APP_CONTENT_ERROR_PARAMETER;
    const size_t len = strnlen(raw, sizeof(raw));
    switch (addcontent_unmount(std::string_view(raw, len))) {
    case AddcontentUnmountResult::Unmounted:  return 0;
    case AddcontentUnmountResult::NotMounted: return APP_CONTENT_ERROR_NOT_FOUND;
    }
    return APP_CONTENT_ERROR_NOT_FOUND;
}

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
// is a pointer at +0x70. Result begins with mode/result/buttonId/pad followed by three pointers. The
// dirName/param pointer slots are caller-owned output destinations, not fields for the service to
// replace. Write only mode/result/buttonId/userData and preserve every padding, pointer, and reserved
// byte. A PlatformUi that accepts Open remains the intended owner, but every callback obtains a
// registry lease first: unregistering the backend safely abandons the dialog instead of dereferencing
// a stale non-owning pointer or rerouting the dialog to a replacement backend.
namespace {
    std::atomic<int> g_savedatadialog_status{0 /*NONE*/};
    std::atomic<uint32_t> g_savedatadialog_mode{0 /*INVALID*/};
    std::atomic<uint64_t> g_savedatadialog_user_data{0};
    std::atomic<PlatformUi*> g_savedatadialog_ui{nullptr};

    void savedlg_close_owner() {
        PlatformUi* expected = g_savedatadialog_ui.exchange(nullptr);
        if (!expected) return;
        auto ui = platform_ui_lease(expected);
        if (ui) ui->saveDataDialogClose();
    }

    PlatformUiLease savedlg_owner_lease(PlatformUi*& expected) {
        expected = g_savedatadialog_ui.load(std::memory_order_acquire);
        if (!expected) return {};
        return platform_ui_lease(expected);
    }

    void savedlg_abandon_owner(PlatformUi* expected) {
        if (expected && g_savedatadialog_ui.compare_exchange_strong(expected, nullptr))
            g_savedatadialog_status.store(3 /*FINISHED: safe headless fallback*/);
    }
}
HLE(s_savedlg_initialize) {
    savedlg_close_owner();
    g_savedatadialog_mode.store(0);
    g_savedatadialog_user_data.store(0);
    g_savedatadialog_status.store(1 /*INITIALIZED*/);
    return 0;
}
HLE(s_savedlg_open) {
    savedlg_close_owner();
    if (a0 && a0 <= UINT64_MAX - 0x78) {
        uint32_t mode = 0;
        uint64_t user_data = 0;
        (void)svc_copy_bytes(a0 + 0x34, &mode, sizeof mode);
        (void)svc_copy_bytes(a0 + 0x70, &user_data, sizeof user_data);
        g_savedatadialog_mode.store(mode);
        g_savedatadialog_user_data.store(user_data);
    } else {
        g_savedatadialog_mode.store(0);
        g_savedatadialog_user_data.store(0);
    }
    auto ui = platform_ui_lease();
    if (ui && ui->saveDataDialogOpen(a0)) {
        g_savedatadialog_ui.store(ui.get(), std::memory_order_release);
        return 0;
    }
    g_savedatadialog_status.store(3 /*FINISHED (auto-dismiss)*/);
    return 0;
}
HLE(s_savedlg_status) {
    PlatformUi* expected = nullptr;
    auto ui = savedlg_owner_lease(expected);
    if (expected && ui)
        return (uint64_t)(unsigned)ui->saveDataDialogStatus();
    savedlg_abandon_owner(expected);
    return (uint64_t)(unsigned)g_savedatadialog_status.load();
}
HLE(s_savedlg_result) {
    PlatformUi* expected = nullptr;
    auto ui = savedlg_owner_lease(expected);
    if (expected && ui) {
        (void)ui->saveDataDialogResult(a0);
        return 0;
    }
    savedlg_abandon_owner(expected);
    if (a0 && a0 <= UINT64_MAX - 0x20) {
        const uint32_t mode = g_savedatadialog_mode.load();
        const uint32_t ok = 0 /*CommonDialogResult::OK*/;
        const uint32_t invalid = 0 /*ButtonId::INVALID*/;
        const uint64_t user_data = g_savedatadialog_user_data.load();
        (void)svc_write_bytes(a0 + 0x00, &mode, sizeof mode);
        (void)svc_write_bytes(a0 + 0x04, &ok, sizeof ok);
        (void)svc_write_bytes(a0 + 0x08, &invalid, sizeof invalid);
        (void)svc_write_bytes(a0 + 0x20, &user_data, sizeof user_data);
    }
    return 0;
}
HLE(s_savedlg_close) {
    savedlg_close_owner();
    g_savedatadialog_status.store(0 /*NONE*/);
    return 0;
}
HLE(s_savedlg_terminate) {
    savedlg_close_owner();
    g_savedatadialog_status.store(0 /*NONE*/);
    g_savedatadialog_mode.store(0);
    g_savedatadialog_user_data.store(0);
    return 0;
}
HLE(s_savedlg_ready) { return 1; }
HLE(s_savedlg_progress_inc) {
    PlatformUi* expected = nullptr;
    auto ui = savedlg_owner_lease(expected);
    if (expected && ui)
        ui->saveDataDialogProgressBarInc((uint32_t)a0, (uint32_t)a1);
    else
        savedlg_abandon_owner(expected);
    return 0;
}
HLE(s_savedlg_progress_set) {
    PlatformUi* expected = nullptr;
    auto ui = savedlg_owner_lease(expected);
    if (expected && ui)
        ui->saveDataDialogProgressBarSetValue((uint32_t)a0, (uint32_t)a1);
    else
        savedlg_abandon_owner(expected);
    return 0;
}

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
static constexpr uint64_t PLAYGO_ERR_BAD_CHUNK_ID = 0x80B2000Cull;

// Most PS5 dumps do not include sce_sys/playgo-chunk.dat, but UE IoStore preserves the same chunk
// ids in paired pakchunk<N>-*.utoc/.ucas files. Keep the PlayGo answers internally consistent with
// the content that is actually present: GetChunkId enumerates these ids and GetLocus/GetProgress
// reject everything else.
// Returning LOCAL_FAST for every possible u16 made DOLL probe through its 1000-id safety cap and left
// its optional-content state unresolved even though pakchunk1 had mounted successfully (#1373).
static std::vector<uint16_t> discover_playgo_chunks() {
    namespace fs = std::filesystem;
    std::vector<uint16_t> chunks;
    std::vector<fs::path> pak_dirs;
    bool saw_iostore_index = false;
    std::error_code ec;
    const fs::path app0(resolve_guest_path("/app0"));

    auto lower = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    };
    auto child_named = [&](const fs::path& parent, const char* wanted) -> fs::path {
        const std::string wanted_lower = lower(wanted);
        std::error_code iter_ec;
        for (fs::directory_iterator it(parent, iter_ec), end; !iter_ec && it != end;
             it.increment(iter_ec)) {
            if (it->is_directory(iter_ec) && lower(it->path().filename().string()) == wanted_lower)
                return it->path();
            iter_ec.clear();
        }
        return {};
    };
    auto add_project_paks = [&](const fs::path& project) {
        const fs::path content = child_named(project, "content");
        if (content.empty()) return;
        const fs::path paks = child_named(content, "paks");
        if (!paks.empty()) pak_dirs.push_back(paks);
    };

    add_project_paks(app0); // /app0/Content/Paks
    for (fs::directory_iterator it(app0, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_directory(ec)) add_project_paks(it->path()); // /app0/<Project>/Content/Paks
        ec.clear();
    }

    for (const fs::path& paks : pak_dirs) {
        std::vector<std::string> nonempty_files;
        std::error_code iter_ec;
        for (fs::directory_iterator it(paks, iter_ec), end; !iter_ec && it != end;
             it.increment(iter_ec)) {
            const bool regular = it->is_regular_file(iter_ec);
            if (iter_ec || !regular) {
                iter_ec.clear();
                continue;
            }
            const uintmax_t size = it->file_size(iter_ec);
            if (iter_ec || size == 0) {
                iter_ec.clear();
                continue;
            }
            nonempty_files.push_back(lower(it->path().filename().string()));
        }
        std::sort(nonempty_files.begin(), nonempty_files.end());
        nonempty_files.erase(std::unique(nonempty_files.begin(), nonempty_files.end()),
                             nonempty_files.end());
        for (const std::string& name : nonempty_files) {
            constexpr const char* prefix = "pakchunk";
            constexpr size_t prefix_len = 8;
            if (name.compare(0, prefix_len, prefix) != 0 ||
                name.size() < prefix_len + 2 + 5 ||
                name.compare(name.size() - 5, 5, ".utoc") != 0)
                continue;
            saw_iostore_index = true;
            const std::string data_name = name.substr(0, name.size() - 5) + ".ucas";
            if (!std::binary_search(nonempty_files.begin(), nonempty_files.end(), data_name))
                continue;
            size_t pos = prefix_len;
            uint32_t id = 0;
            while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
                id = id * 10 + static_cast<unsigned>(name[pos++] - '0');
                if (id > UINT16_MAX) break;
            }
            if (pos == prefix_len || pos >= name.size() || name[pos] != '-' || id > UINT16_MAX)
                continue;
            chunks.push_back(static_cast<uint16_t>(id));
        }
    }
    std::sort(chunks.begin(), chunks.end());
    chunks.erase(std::unique(chunks.begin(), chunks.end()), chunks.end());
    if (chunks.empty() && !saw_iostore_index)
        chunks.push_back(0); // compatibility fallback for non-IoStore titles
    return chunks;
}

static std::vector<uint16_t> g_playgo_chunks{0};
static bool playgo_has_chunk(uint16_t id) {
    return std::binary_search(g_playgo_chunks.begin(), g_playgo_chunks.end(), id);
}

HLE(s_playgo_init)  { svc_log("scePlayGoInitialize", a0,a1,a2,a3,a4,a5);
                      g_playgo_chunks = discover_playgo_chunks();
                      if (svclog()) {
                          std::fprintf(stderr, "[svc] PlayGo discovered %zu installed chunk(s):",
                                       g_playgo_chunks.size());
                          for (uint16_t id : g_playgo_chunks) std::fprintf(stderr, " %u", id);
                          std::fputc('\n', stderr);
                      }
                      return 0; }
HLE(s_playgo_term)  { return 0; }
// scePlayGoOpen(u32* outHandle, const void* param): the handle the whole API is keyed on. The
// unimpl stub's success-with-unfilled-handle left the game querying loci with garbage.
HLE(s_playgo_open)  { svc_log("scePlayGoOpen", a0,a1,a2,a3,a4,a5);
                      if (!a0) return PLAYGO_ERR_BAD_POINTER;
                      *(uint32_t*)PW(a0) = 1; return 0; }
HLE(s_playgo_close) { return 0; }
// scePlayGoGetLocus(h, const u16* chunkIds, u32 n, s8* outLoci): installed chunks are LOCAL_FAST.
HLE(s_playgo_getlocus) { svc_log("scePlayGoGetLocus", a0,a1,a2,a3,a4,a5, 2);
                         if (!a1 || !a3) return PLAYGO_ERR_BAD_POINTER;
                         if (!(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                         const auto* ids = (const uint16_t*)PW(a1);
                         auto* loci = (int8_t*)PW(a3);
                         for (uint32_t i = 0; i < (uint32_t)a2; ++i) {
                             if (!playgo_has_chunk(ids[i])) {
                                 loci[i] = 0; // SCE_PLAYGO_LOCUS_NOT_DOWNLOADED
                                 return PLAYGO_ERR_BAD_CHUNK_ID;
                             }
                             loci[i] = 3; // SCE_PLAYGO_LOCUS_LOCAL_FAST
                         }
                         return 0; }
// scePlayGoGetProgress(h, chunkIds, n, OrbisPlayGoProgress* out): one struct {u64 progressSize;
// u64 totalSize} summed over the queried chunks; fully-installed == progressSize==totalSize!=0.
HLE(s_playgo_getprogress) { svc_log("scePlayGoGetProgress", a0,a1,a2,a3,a4,a5, 2);
                            if (!a1 || !a3) return PLAYGO_ERR_BAD_POINTER;
                            if (!(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                            const auto* ids = (const uint16_t*)PW(a1);
                            for (uint32_t i = 0; i < (uint32_t)a2; ++i)
                                if (!playgo_has_chunk(ids[i])) return PLAYGO_ERR_BAD_CHUNK_ID;
                            uint64_t* p = (uint64_t*)PW(a3);
                            p[0] = p[1] = (uint64_t)(uint32_t)a2 << 20;  // 1 MiB/chunk, done==total
                            return 0; }
// scePlayGoGetToDoList(h, OrbisPlayGoToDo* list, u32 n, u32* outEntries): nothing left to install.
HLE(s_playgo_gettodo) { if (!a3) return PLAYGO_ERR_BAD_POINTER; *(uint32_t*)PW(a3) = 0; return 0; }
HLE(s_playgo_settodo) { return 0; }
// scePlayGoGetChunkId(h, u16* list, u32 n, u32* outEntries): enumerate the installed chunk ids.
HLE(s_playgo_getchunkid) { if (!a3) return PLAYGO_ERR_BAD_POINTER;
                           if (a1 && !(uint32_t)a2) return PLAYGO_ERR_BAD_SIZE;
                           const uint32_t count = static_cast<uint32_t>(g_playgo_chunks.size());
                           if (a1) {
                               const uint32_t copied = std::min((uint32_t)a2, count);
                               std::memcpy(PW(a1), g_playgo_chunks.data(), copied * sizeof(uint16_t));
                               *(uint32_t*)PW(a3) = copied;
                           } else *(uint32_t*)PW(a3) = count;
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
// mount-result the game then reads paths from. Transaction bookkeeping calls allocate/tear down
// local resources only, and Create returns the new resource's ID (see #1905 below).
// CONFIDENCE: HIGH on Initialize3/NOT_FOUND semantics;
// MED on Mount3's arg order (mount-desc in, result out — matches every PS4 Mount variant);
// LOW on Prepare/Commit internals (no-op success; PROSPER_SVCLOG captures their real args).
static constexpr uint64_t SAVE_DATA_ERR_PARAMETER = 0x809F0000ull;
static constexpr uint64_t SAVE_DATA_ERR_EXISTS = 0x809F0007ull;
static constexpr uint64_t SAVE_DATA_ERR_NOT_FOUND = 0x809F0008ull;
// From the same published table this file's other savedata codes come from. Spelled out rather than
// folded into a near-miss: "no save is mounted" and "the host write failed" are different facts, and
// answering either with PARAMETER would send a title looking at its own arguments.
// CONFIDENCE: MED on both numeric values; HIGH on the polarity, which is what a title branches on.
constexpr uint64_t SAVE_DATA_ERR_NOT_MOUNTED = 0x809F0004ull;
constexpr uint64_t SAVE_DATA_ERR_INTERNAL    = 0x809F000Bull;

// 0x809F0018 is the "the operation is STILL IN FLIGHT, keep waiting" code. That meaning is
// corroborated by four independent titles in the local dump set, each of which sleeps and re-polls
// on it -- better evidence than exists for most constants in this file:
//
//   PPSA15552 Dead Cells      +0x173c9b0  call GetEventResult; cmp eax,0x809f0018; jne <exit>;
//                                         mov edi,0x1f40; call sleep; jmp 0x173c9b0
//   PPSA15552 Dead Cells      +0x173cda0  same idiom, second site
//   PPSA28061 Earthion        +0x12f6d    cmp [rbp-0x7c],0x809f0018; jne; sleep(50 ms); jmp <repoll>
//   PPSA03831 Sonic Frontiers +0x18a2285  cmp eax,0x809f0018; jne; sleep(1); jmp <repoll>
//   PPSA05325 Sonic Origins   +0x940385   byte-identical (same SEGA save library)
//
// prosper has NOTHING in flight: every file operation here completes synchronously and the only
// event ever queued is Umount2's. So answering 0x809F0018 was not an unestablished value -- it was
// a well-established "still busy" returned in a state where nothing is busy, i.e. a permanent lie,
// and a guaranteed infinite hang for any title that reaches one of those loops.
//
// A DRAINED queue is therefore reported with NOT_FOUND. PPSA20447 (The First Berserker: Khazan)
// pins that value directly -- its game thread's drain loop leaves only on 0x809F0008:
//
//   eboot+0x796eb2c   jmp    0x796eb3d             ; loop ENTRY -- it polls first, sleeps after
//   eboot+0x796eb38   call   0x1565790             ; FPlatformProcess::Sleep(float)
//   eboot+0x796eb41.. vmovups/mov                  ; zero a 104-byte SceSaveDataEvent (96 + 8)
//   eboot+0x796eb67   xor    edi,edi               ; eventParam = NULL
//   eboot+0x796eb6f   call   0x8eaf3d0             ; sceSaveDataGetEventResult(NULL, &event)
//   eboot+0x796eb7b   mov    r12d,eax              ; r12d IS the return value
//   eboot+0x796eb88   cmp    r12d,0x809f0008       ; <-- the ONLY value that ends the wait
//   eboot+0x796eb8f   jne    0x796eb30             ; anything else: sleep and poll again
//   eboot+0x796eb91   movzx  eax,BYTE PTR [rip+..] ; a SECOND gate can still re-enter the loop
//   eboot+0x796eb9a   jne    0x796eb30
//
// Earthion const-compares BOTH, in consecutive instructions -- 0x809F0018 -> sleep and re-poll,
// then 0x809F0008 -> give up and return -- so the two codes are genuinely distinct in a shipping
// title's bytes and this file must not merge their meanings, only their current value.
//
// SPELLED AS A LITERAL, not aliased to SAVE_DATA_ERR_NOT_FOUND on purpose: the drained-queue answer
// HAPPENS to be NOT_FOUND today because prosper never has an operation in flight. An implementation
// that gives sceSaveDataMount3 a real asynchronous path (which Earthion's wait needs -- see below)
// must return SAVE_DATA_ERR_IN_FLIGHT while the operation runs, and should be able to do that
// without unpicking an alias.
// CONFIDENCE: HIGH on both values (five titles' own compare instructions).
static constexpr uint64_t SAVE_DATA_ERR_IN_FLIGHT = 0x809F0018ull;   // "still running, keep waiting"
static constexpr uint64_t SAVE_DATA_ERR_NO_EVENT  = 0x809F0008ull;   // drained: same value as NOT_FOUND
static_assert(SAVE_DATA_ERR_NO_EVENT == SAVE_DATA_ERR_NOT_FOUND,
              "a drained queue is reported with NOT_FOUND (PPSA20447 eboot+0x796eb88)");
static_assert(SAVE_DATA_ERR_IN_FLIGHT != SAVE_DATA_ERR_NO_EVENT,
              "the in-flight and drained codes are distinct in Earthion's bytes (eboot+0x12f6d/+0x12f82)");
namespace { std::atomic<unsigned> g_savedata_umount_events{0}; }
HLE(s_savedata_init3)   { svc_log("sceSaveDataInitialize3", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_savedata_term)    { return 0; }

// --- Transaction resources (#1905). sceSaveDataCreateTransactionResource returns the NEW RESOURCE'S
// ID, not 0-on-success. Returning 0 is the shape of the call that succeeds, so this looked correct
// and is not: Sonic Origins' (PPSA05325) save handler at eboot+0x93fdb0 is exactly
//     xor edi,edi; call sceSaveDataCreateTransactionResource; test eax,eax; jle <fail>
// — a zero return is read as failure, the handler records error 3, every later save operation
// returns that sticky error without running, and the title's boot coroutine polls the failed job
// forever. That single wrong return value is why PPSA05325 never leaves its first boot step.
// It really is a HANDLE and not merely "a positive number" — Sonic's full lifecycle proves it:
//   0x93fdc7  mov [r14+0xc0],eax        Create's result is retained in a member
//   0x9402cd  mov eax,[r14+0xc0]        ...loaded again to build the Mount3 descriptor,
//   0x9402d4  mov [rsp+0x68],eax        ...at descriptor base (rsp+0x40) + 0x28,
//   0x93f24e  mov edi,[rdi+0xc0]        ...and handed back as Delete's sole argument,
//   0x93f254  cmp edi,0xffffffff        guarded by the guest's own -1 "no resource" sentinel.
// That also independently reproduces the Mount3 +0x28 field this file documents from DQ7. The id is
// opaque to the guest, so any positive value satisfies it; we hand out a monotonic counter and keep
// the live set so Delete can reject one that was never created.
// CONFIDENCE: HIGH on the polarity, on "returns the resource id", and on Delete's argument being a
// scalar int32 in edi (the mov above precedes all four of Sonic's Delete sites and Oregon Trail's).
// Delete's error CODE for an unknown id is the residual unknown: PARAMETER is the facility's generic
// bad-argument code, and no title is observed passing an id Create did not return.
namespace {
std::mutex g_savedata_tx_mu;
int32_t g_savedata_tx_next = 1;
std::set<int32_t> g_savedata_tx_live;
}   // namespace

int32_t savedata_tx_resource_create() {
    std::lock_guard<std::mutex> lk(g_savedata_tx_mu);
    if (g_savedata_tx_next <= 0) return -1;          // counter exhausted (unreachable in practice)
    int32_t id = g_savedata_tx_next++;
    g_savedata_tx_live.insert(id);
    return id;
}

bool savedata_tx_resource_destroy(int32_t id) {
    std::lock_guard<std::mutex> lk(g_savedata_tx_mu);
    return g_savedata_tx_live.erase(id) != 0;
}

size_t savedata_tx_resource_live_count() {
    std::lock_guard<std::mutex> lk(g_savedata_tx_mu);
    return g_savedata_tx_live.size();
}

HLE(s_savedata_txres) {
    svc_log("sceSaveDataCreateTransactionResource", a0,a1,a2,a3,a4,a5);
    int32_t id = savedata_tx_resource_create();
    if (id <= 0) return SAVE_DATA_ERR_PARAMETER;
    return (uint64_t)(uint32_t)id;
}
HLE(s_savedata_txres_del) {
    svc_log("sceSaveDataDeleteTransactionResource", a0,a1,a2,a3,a4,a5);
    if (!savedata_tx_resource_destroy((int32_t)(uint32_t)a0)) return SAVE_DATA_ERR_PARAMETER;
    return 0;
}

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
    // The in-process view of the SaveDataMemory slots. Keyed by TITLE as well as (userId, slotId),
    // because (userId, slotId) is not unique across titles — every Unity title uses user 1, slot 0.
    //
    // No shipping frontend boots two titles in one process today: prosper-app's start_guest()
    // latches g_boot_attempted on the ATTEMPT and routes the second title through
    // relaunch_with_dump(), i.e. a new process. So this is not fixing a reachable collision; it is
    // keeping the cache and the on-disk layout partitioned the SAME way, so the cache cannot become
    // a second source of truth that disagrees with the files. What does exercise it is
    // test_savedata_title_namespace, which drives two application roots through one process — and a
    // cache that outlived the title switch would hand title B title A's block while the files were
    // correctly separated, which is a harder bug to see than the one being fixed.
    std::unordered_map<std::string, std::vector<uint8_t>> g_savemem;
    std::string savemem_key(int32_t userId, uint32_t slotId) {
        char suffix[48];
        snprintf(suffix, sizeof suffix, "/%d:%u", (int)userId, (unsigned)slotId);
        return save_title_namespace() + suffix;
    }
    // Host file backing one SaveDataMemory slot, so a save survives a process restart (the API is the
    // ENTIRE save path for the Unity titles — no file Mount — so without this every relaunch looks like
    // a fresh console and the game restarts from scratch; likely root of #299).
    //
    // PER TITLE: <PROSPER_SAVEDATA_DIR or the per-user default>/<TITLE_ID>/savemem_<user>_<slot>.bin.
    // (userId, slotId) is not unique across titles — every Unity title writes user 1 slot 0 — so a
    // flat directory made two titles share one save file (#2734). Not cached in a static: the title
    // component comes from set_app0_root()'s param.json parse, and caching would freeze whichever
    // title resolved it first. `create` is passed only by the writer, so reading a slot for a title
    // that has never saved does not manufacture a directory for it.
    std::string savemem_path(int32_t userId, uint32_t slotId, bool create = false) {
        const std::string base = create ? savedata_mem_ensure_dir() : savedata_mem_dir();
        if (base.empty()) return {};
        char name[64];
        snprintf(name, sizeof name, "/savemem_%d_%u.bin", (int)userId, (unsigned)slotId);
        return base + name;
    }
    std::vector<uint8_t> savemem_load(int32_t userId, uint32_t slotId) {
        std::vector<uint8_t> v;
        const std::string path = savemem_path(userId, slotId);
        if (path.empty()) return v;
        if (FILE* f = fopen(path.c_str(), "rb")) {
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            if (n > 0) { v.resize((size_t)n); if (fread(v.data(), 1, (size_t)n, f) != (size_t)n) v.clear(); }
            fclose(f);
        }
        return v;
    }
    void savemem_store(int32_t userId, uint32_t slotId, const std::vector<uint8_t>& buf) {
        const std::string path = savemem_path(userId, slotId, /*create=*/true);
        if (path.empty()) return;
        if (FILE* f = fopen(path.c_str(), "wb")) {
            if (!buf.empty()) fwrite(buf.data(), 1, buf.size(), f);
            fclose(f);
        }
    }
    template <class T> T ld(uint64_t base, size_t off) {   // read a guest struct field at byte offset
        T v; memcpy(&v, (const uint8_t*)PW(base) + off, sizeof(T)); return v;
    }
    uint64_t savemem_setup_block(int32_t userId, uint32_t slotId, uint64_t memSize) {
        uint64_t existed = 0;
        std::lock_guard<std::mutex> lk(g_savemem_mx);
        auto& buf = g_savemem[savemem_key(userId, slotId)];
        if (buf.empty()) buf = savemem_load(userId, slotId);
        existed = buf.size();
        if (memSize > buf.size()) buf.resize(memSize, 0);
        return existed;
    }
    void savemem_write_range(std::vector<uint8_t>& dst, uint64_t guestBuf, uint64_t guestSize,
                             int64_t offset) {
        if (!guestBuf || offset < 0 || (uint64_t)offset >= dst.size()) return;
        uint64_t n = std::min<uint64_t>(guestSize, dst.size() - (uint64_t)offset);
        memcpy(dst.data() + offset, PW(guestBuf), n);
    }
    void savemem_read_range(const std::vector<uint8_t>& src, uint64_t guestBuf, uint64_t guestSize,
                            int64_t offset) {
        if (!guestBuf || offset < 0 || (uint64_t)offset >= src.size()) return;
        uint64_t n = std::min<uint64_t>(guestSize, src.size() - (uint64_t)offset);
        memcpy(PW(guestBuf), src.data() + offset, n);
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
    uint64_t existed = savemem_setup_block(userId, slotId, memSize);
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
        savemem_write_range(buf, gbuf, gsize, off);
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
        savemem_read_range(buf, gbuf, gsize, off);
    }
    return 0;
}
// The original SaveDataMemory API passes scalar arguments and always addresses slot 0. These PS4-
// inherited exports are still present in the PS5 3.20 table; route them through the same backing
// store as the struct-based *2 API so callers do not receive fake success with no data transfer.
HLE(s_savemem_setup_v1) {
    svc_log("sceSaveDataSetupSaveDataMemory", a0,a1,a2,a3,a4,a5);
    savemem_setup_block((int32_t)a0, 0, a1);
    return 0;
}
HLE(s_savemem_set_v1) {
    svc_log("sceSaveDataSetSaveDataMemory", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_savemem_mx);
    auto it = g_savemem.find(savemem_key((int32_t)a0, 0));
    if (it == g_savemem.end()) return SD_ERR_MEMORY_NOTREADY;
    savemem_write_range(it->second, a1, a2, (int64_t)a3);
    return 0;
}
HLE(s_savemem_get_v1) {
    svc_log("sceSaveDataGetSaveDataMemory", a0,a1,a2,a3,a4,a5);
    std::lock_guard<std::mutex> lk(g_savemem_mx);
    auto it = g_savemem.find(savemem_key((int32_t)a0, 0));
    if (it == g_savemem.end()) return SD_ERR_MEMORY_NOTREADY;
    savemem_read_range(it->second, a1, a2, (int64_t)a3);
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
// All three pinned mount ABIs feed one backend/result writer. The PS4-inherited layouts come from
// the public libSceSaveData ABI and retain their NIDs in the PS5 3.20 table:
//   Mount:  dirName pointer @0x10, blocks @0x20, mode @0x28, size 0x50
//   Mount2: dirName pointer @0x08, blocks @0x10, mode @0x18, size 0x40
// Mount3's PS5-native layout is documented below. MountResult is the shared exact 0x40-byte shape.
static uint64_t savedata_mount_common(const char* api, const char* dirname,
                                      uint32_t mode, uint64_t result_va) {
    if (!dirname || !*dirname || !result_va) return SAVE_DATA_ERR_PARAMETER;
    const SaveDataMountPolicy policy = (mode & 0x04) ? SaveDataMountPolicy::Create
        : (mode & 0x20) ? SaveDataMountPolicy::OpenOrCreate
                        : SaveDataMountPolicy::Open;
    const SaveDataMountOutcome outcome = savedata0_mount(dirname, policy);
    if (outcome == SaveDataMountOutcome::Exists) {
        if (svclog()) fprintf(stderr, "[svc]   %s dir='%s' mode=%#x -> EXISTS\n",
                              api, dirname, mode);
        return SAVE_DATA_ERR_EXISTS;
    }
    if (outcome == SaveDataMountOutcome::NotFound) {
        if (svclog()) fprintf(stderr, "[svc]   %s dir='%s' mode=%#x -> NOT_FOUND\n",
                              api, dirname, mode);
        return SAVE_DATA_ERR_NOT_FOUND;
    }
    uint8_t* result = (uint8_t*)PW(result_va);
    memset(result, 0, 0x40);
    memcpy(result, "/savedata0", 11);
    const bool created = outcome == SaveDataMountOutcome::Created;
    *(uint32_t*)(result + 0x1c) = created ? 1u : 0u;
    if (svclog()) fprintf(stderr, "[svc]   %s dir='%s' mode=%#x -> OK (created=%d)\n",
                          api, dirname, mode, (int)created);
    return 0;
}

HLE(s_savedata_mount) {
    svc_log("sceSaveDataMount", a0,a1,a2,a3,a4,a5);
    if (!a0 || !a1) return SAVE_DATA_ERR_PARAMETER;
    const uint8_t* mount = (const uint8_t*)PW(a0);
    const char* dirname = *(const char* const*)(mount + 0x10);
    const uint32_t mode = *(const uint32_t*)(mount + 0x28);
    return savedata_mount_common("Mount", dirname, mode, a1);
}

HLE(s_savedata_mount2) {
    svc_log("sceSaveDataMount2", a0,a1,a2,a3,a4,a5);
    if (!a0 || !a1) return SAVE_DATA_ERR_PARAMETER;
    const uint8_t* mount = (const uint8_t*)PW(a0);
    const char* dirname = *(const char* const*)(mount + 0x08);
    const uint32_t mode = *(const uint32_t*)(mount + 0x18);
    return savedata_mount_common("Mount2", dirname, mode, a1);
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
    return savedata_mount_common("Mount3", dirname, mode, a1);
}
// The mount-point argument every one of these calls takes is SceSaveDataMountPoint { char data[16] }.
// Accept only the mount this build serves, so a title passing a different one is refused instead of
// having its request applied to whatever happens to be mounted.
//
// AN UNTERMINATED FIELD IS SAFE TO HAND IN -- but not because of the n=16, which is what this
// comment used to say. strncmp stops at the first NUL in EITHER string, and the literal's sits at
// index 10, so at most 11 bytes of the guest's field are ever examined and the 16 is a ceiling that
// is never reached. Measured rather than reasoned: with only 11 bytes mapped and the next page
// PROT_NONE, the comparison completes without faulting. That is also why a guest that fills all 16
// bytes gets a clean mismatch -- its byte 10 cannot be the literal's terminator -- instead of a read
// off the end of its struct.
//
// The consequence worth carrying: the bound is the LITERAL, so lengthening the accepted name is not
// free. A longer spelling pushes the read further into the field, and only at 16 characters does the
// n start doing any work. Widen the name and the ceiling becomes load-bearing for the first time.
//
// This is the only mount-point spelling check in the savedata surface -- GetMountInfo, SetParam and
// GetParam all go through it, so they cannot drift apart about which mount point is real.
static bool savedata_mount_point_ok(uint64_t mount_point_va) {
    if (!mount_point_va) return false;
    const char* mp = (const char*)PW(mount_point_va);
    return strncmp(mp, "/savedata0", 16) == 0;
}

// --- the two umount entry points (#3666) --------------------------------------------------------
//
// sceSaveDataUmount  (BMR4F-Uek3E, libSceSaveData)         PS4-namespace
// sceSaveDataUmount2 (uW4vfTwMQVo, libSceSaveData.native)  PS5-native, and the one titles use
//
// Umount2 previously read NONE of its arguments: it unmounted /savedata0, queued a completion
// event and returned SCE_OK whatever it was handed. A title passing a mount point this build does
// not serve therefore had a LIVE MOUNT DISCARDED on its behalf and was told it had succeeded --
// the #3653 shape, but mutating state rather than only misreporting it.
//
// THE ARGUMENT ORDER IS NOT THE OBVIOUS ONE, and this is the finding to carry: Umount2's mount
// point is the SECOND argument. The first is a small flags word. Reading it as (mountPoint, ...)
// by analogy with Umount would have validated the flags word as a pointer and refused every
// correct call.
//
//   int sceSaveDataUmount2(uint32_t flags, const SceSaveDataMountPoint* mountPoint);
//
// CONFIDENCE: HIGH on that shape. Established top-of-hierarchy first -- a LIVE CAPTURE of Dead
// Cells (PPSA15552) on its own snapshot route, PROSPER_SVCLOG=1, three calls, every one of them:
//
//   [svc] sceSaveDataUmount2(0x1, 0x7f8953c20c60, ...)
//   [svc]   a1 -> 746164657661732f 0000000000003061 ...      <- "/savedat" "a0\0"
//
// and corroborated statically in four independent titles, where the pointer handed to Umount2 is
// provably the MountResult a mount call just filled in (mountPoint char[16] sits at its +0x00, so
// &result and &result.mountPoint are the same address):
//
//   PPSA03831 Sonic Frontiers  eboot+0x165f177 mounts into &rsp[0], then eboot+0x165f180-87 does
//                              `mov rsi,rsp / xor edi,edi / call Umount2` -- the same buffer.
//   PPSA15552 Dead Cells       r15 is the result out-param of Mount3 at eboot+0x173c90c and the
//                              rsi of Umount2 at eboot+0x173c960.
//   PPSA20447 Khazan           eboot+0x1531bc6 `mov rsi,rbx / xor edi,edi` after Commit.
//   PPSA17942 DOLL             eboot+0x2251792-97 `mov rsi,r14 / xor edi,edi`, the wrapper's own
//                              second parameter passed straight through.
//
// In every site rdi and rsi are the only registers set after the preceding call, so the call takes
// TWO arguments; rdx/rcx are clobbered garbage (they differ per call in the live capture).
//
// THE FLAGS WORD IS DELIBERATELY NOT VALIDATED. Observed values are 0, 1 and 0x10001 -- Sonic
// Frontiers builds its at eboot+0x18a2401-0d as `v` or `v + 0x10000` under a runtime condition, so
// bit 16 is an independent flag rather than part of a small enum. What those bits MEAN is not
// established, and a validator aimed at a guessed flag set would refuse correct calls to buy
// nothing. CONFIDENCE: MED that the word is flags at all (the conditional bit-16 set is the
// evidence); no confidence is claimed about individual bits, so none are rejected. If it later
// turns out to carry something prosper must honour -- a "discard without commit" bit, say -- that
// is a behaviour to add, not a check.
//
// THE CONTRACT, now the same one for both entry points and the same one GetMountInfo/SetParam/
// GetParam use:
//
//   input                                    answer
//   mount point null, or not this build's    SAVE_DATA_ERR_PARAMETER    (0x809F0000)
//   valid /savedata0, nothing mounted        SAVE_DATA_ERR_NOT_MOUNTED  (0x809F0004)
//   valid /savedata0, live mount             0, and the mount is gone
//
// Umount previously answered NOT_FOUND to BOTH error rows, collapsing "your argument is wrong" and
// "nothing is mounted" into one code that means neither. CONFIDENCE: LOW that PARAMETER/NOT_MOUNTED
// is what the firmware returns, and the local corpus cannot raise it: of the 61 local dumps with an
// eboot, ZERO import BMR4F-Uek3E -- Umount, UmountSys and UmountWithBackup are absent from every
// one of them, while 34 import Umount2 (exactly the 34 that import Mount3). So this row is
// consistency with the four sibling handlers, not evidence, and nothing local can falsify it. What
// WOULD: a title that const-compares an umount return against 0x809F0008.
// CONFIDENCE: HIGH on the polarity, which is the part titles branch on -- discarding a mount that
// was never asked for, and reporting success for an unmount that did not happen, are both wrong.
//
// THE COMPLETION EVENT IS NOW QUEUED ONLY BY AN UNMOUNT THAT HAPPENED. g_savedata_umount_events
// used to be bumped unconditionally, so a refused call still fed sceSaveDataGetEventResult a
// fabricated UMOUNT_BACKUP completion for an operation that never ran -- the same false-success
// class one layer down. A synchronous failure starts no operation and so completes none.
//
// No snapshot of the mount state is taken here, unlike GetMountInfo: savedata0_umount() tests and
// clears under one hold of g_save0_mx, so "was it mounted" and "unmount it" cannot be separated by
// a concurrent umount on another guest thread. Two threads racing to unmount therefore produce
// exactly one success and one NOT_MOUNTED rather than two successes and two events.
HLE(s_savedata_umount) {
    svc_log("sceSaveDataUmount", a0,a1,a2,a3,a4,a5);
    if (!savedata_mount_point_ok(a0)) return SAVE_DATA_ERR_PARAMETER;
    if (!savedata0_umount()) return SAVE_DATA_ERR_NOT_MOUNTED;
    return 0;
}
HLE(s_savedata_umount2) {
    svc_log("sceSaveDataUmount2", a0,a1,a2,a3,a4,a5);   // a0 = flags, a1 = const SceSaveDataMountPoint*
    if (!savedata_mount_point_ok(a1)) return SAVE_DATA_ERR_PARAMETER;
    if (!savedata0_umount()) return SAVE_DATA_ERR_NOT_MOUNTED;
    g_savedata_umount_events.fetch_add(1, std::memory_order_release);
    return 0;
}
// sceSaveDataGetMountInfo(const SceSaveDataMountPoint* mp, SceSaveDataMountInfo* info):
//   SceSaveDataMountInfo { u64 blocks; u64 freeBlocks; u8 reserved[32] } -- 48 bytes.
//
// This call was MISSING first (success with garbage free-space, so a title sizing its save against
// freeBlocks could abort "disk full" or corrupt its math), and the fix for that then IGNORED `mp`
// entirely: any non-null `info` got a cleared block, two fixed capacity numbers and SCE_OK. So a
// null mount point, "/savedata1", and "/savedata0" AFTER an unmount all received a successful
// capacity report for a mount that was not there (#3653). The one call whose job is to describe a
// live mount could not say that a mount was absent.
//
// Identity and lifecycle now go through exactly the contract sceSaveDataSetParam/GetParam below
// already use -- the bounded savedata_mount_point_ok() above and savedata0_mounted_dir() -- so this
// file holds ONE notion of what is mounted rather than two that can disagree.
//
// ERROR PRECEDENCE, and what is actually established about it. Arguments are checked before
// lifecycle: a caller that passed a mount point this build does not serve is looking at its own
// argument, not at a mount that went away. No published ABI text pins that order, and the local
// corpus cannot pin it either -- of the five call sites in the three dumps that import this NID
// (PPSA15319 x2, PPSA20447 x2, PPSA28061 x1), tools/re/nid_gate_scan.py buckets four as `nonzero`,
// its one bucket meaning the result is gated on zero/non-zero and then dead on every reachable
// path, and the fifth as `forward`, which leaves the scan's reach without ever const-comparing. No
// local title can tell PARAMETER from NOT_MOUNTED. So: CONFIDENCE: LOW on the ordering between the
// two error codes (it follows the sibling handlers in this file, which is consistency rather than
// evidence); CONFIDENCE: HIGH on the polarity every observed call site does read -- an invalid or
// inactive mount must not return SCE_OK.
//
// On an error return the guest's buffer is left EXACTLY as it was, which is this library's existing
// policy rather than a new one: Mount3's NOT_FOUND path "writes NOTHING" and GetParam refuses before
// touching `out`. Validation therefore happens before the first store, not after it.
//
// The capacity figures are deliberately NOT touched by this change: real used/free accounting is
// #3654. They remain a fixed, consistent 256K blocks, derived from nothing.
constexpr size_t   SAVE_DATA_MOUNT_INFO_SIZE           = 48;
constexpr size_t   SAVE_DATA_MOUNT_INFO_OFF_BLOCKS     = 0;
constexpr size_t   SAVE_DATA_MOUNT_INFO_OFF_FREE       = 8;
constexpr size_t   SAVE_DATA_MOUNT_INFO_OFF_RESERVED   = 16;
constexpr uint64_t SAVE_DATA_MOUNT_INFO_FIXED_BLOCKS   = 0x40000;   // #3654 owns real accounting
static_assert(SAVE_DATA_MOUNT_INFO_OFF_RESERVED + 32 == SAVE_DATA_MOUNT_INFO_SIZE,
              "SceSaveDataMountInfo is blocks + freeBlocks + reserved[32]");
HLE(s_savedata_mountinfo) {
    svc_log("sceSaveDataGetMountInfo", a0,a1,a2,a3,a4,a5);
    if (!savedata_mount_point_ok(a0) || !a1) return SAVE_DATA_ERR_PARAMETER;
    // A COPY of the mounted path, not a mount LEASE: savedata0_mounted_dir() releases the mount
    // mutex before returning, so a concurrent sceSaveDataUmount2 on another guest thread can land
    // the instant after this test and the answer below is then one sample old. That is sound here
    // only because nothing after this line depends on the mount still being live -- this handler
    // touches no file. Whoever implements #3654 must not read the emptiness test as a lease: a stat
    // of the save directory has to treat a path that vanished under it as NOT_MOUNTED, not as an
    // internal error, and must not assume the directory it measured is still /savedata0.
    if (savedata0_mounted_dir().empty()) return SAVE_DATA_ERR_NOT_MOUNTED;
    uint8_t* info = (uint8_t*)PW(a1);
    memset(info, 0, SAVE_DATA_MOUNT_INFO_SIZE);   // reserved[32] reads back zeroed, as before
    *(uint64_t*)(info + SAVE_DATA_MOUNT_INFO_OFF_BLOCKS) = SAVE_DATA_MOUNT_INFO_FIXED_BLOCKS;
    *(uint64_t*)(info + SAVE_DATA_MOUNT_INFO_OFF_FREE)   = SAVE_DATA_MOUNT_INFO_FIXED_BLOCKS;
    return 0;
}
// --- sceSaveDataSetParam / sceSaveDataGetParam (#2786) -----------------------------------------
// Both were unregistered, so both reached the dispatcher's `return 0` -- and 0 is SCE_OK here. A
// title was told its save metadata had been written when nothing was, and read back a zeroed block
// it could not tell apart from what it had set. Observed on Sonic Frontiers (PPSA03831) as the only
// unimplemented NID on a route that reaches GameModeStage; called once per boot.
//
// ABI (PS4-inherited; the NIDs are the ones libSceSaveData / libSceSaveData_native export on 3.20):
//   sceSaveDataSetParam(const SceSaveDataMountPoint* mp, SceSaveDataParamType type,
//                       const void* buf, size_t bufSize)
//   sceSaveDataGetParam(const SceSaveDataMountPoint* mp, SceSaveDataParamType type,
//                       void* buf, size_t bufSize, size_t* gotSize)
// SceSaveDataParam, the TYPE_ALL block, is 1328 bytes: title[128], subTitle[128], detail[1024],
// userParam u32, pad u32, mtime (time_t), reserved[32].
// CONFIDENCE: HIGH on the NIDs (PS5 3.20 export table) and on the two signatures' argument order;
// MED on the exact SceSaveDataParam field offsets, which come from the published PS4 ABI rather
// than from a live capture of a PS5 title reading one. Every offset is a named constant asserted in
// tests/hle/test_savedata_param.cpp, so a future capture that contradicts one changes a constant
// and reddens a named arm rather than silently disagreeing with the guest.
enum : uint32_t {
    SAVE_DATA_PARAM_TYPE_ALL = 0,
    SAVE_DATA_PARAM_TYPE_TITLE = 1,
    SAVE_DATA_PARAM_TYPE_SUB_TITLE = 2,
    SAVE_DATA_PARAM_TYPE_DETAIL = 3,
    SAVE_DATA_PARAM_TYPE_USER_PARAM = 4,
    SAVE_DATA_PARAM_TYPE_MTIME = 5,
};
constexpr size_t SAVE_DATA_TITLE_MAXSIZE = 128;
constexpr size_t SAVE_DATA_SUBTITLE_MAXSIZE = 128;
constexpr size_t SAVE_DATA_DETAIL_MAXSIZE = 1024;
constexpr size_t SAVE_DATA_PARAM_ALL_SIZE = 1328;
constexpr size_t SAVE_DATA_PARAM_OFF_TITLE = 0;
constexpr size_t SAVE_DATA_PARAM_OFF_SUBTITLE = 128;
constexpr size_t SAVE_DATA_PARAM_OFF_DETAIL = 256;
constexpr size_t SAVE_DATA_PARAM_OFF_USER_PARAM = 1280;
constexpr size_t SAVE_DATA_PARAM_OFF_MTIME = 1288;
// The byte count TYPE_<x> transfers. 0 means "not a type this can answer", which is refused rather
// than guessed: answering an unknown type with silent success is exactly what #2786 was.
static size_t savedata_param_type_size(uint32_t type) {
    switch (type) {
        case SAVE_DATA_PARAM_TYPE_ALL:        return SAVE_DATA_PARAM_ALL_SIZE;
        case SAVE_DATA_PARAM_TYPE_TITLE:      return SAVE_DATA_TITLE_MAXSIZE;
        case SAVE_DATA_PARAM_TYPE_SUB_TITLE:  return SAVE_DATA_SUBTITLE_MAXSIZE;
        case SAVE_DATA_PARAM_TYPE_DETAIL:     return SAVE_DATA_DETAIL_MAXSIZE;
        case SAVE_DATA_PARAM_TYPE_USER_PARAM: return sizeof(uint32_t);
        case SAVE_DATA_PARAM_TYPE_MTIME:      return sizeof(int64_t);
        default:                              return 0;
    }
}
// A guest string field is a fixed-size char array that may be exactly full, i.e. unterminated.
static std::string savedata_param_text(const uint8_t* field, size_t capacity) {
    size_t n = 0;
    while (n < capacity && field[n]) n++;
    return std::string((const char*)field, n);
}
static void savedata_param_write_text(uint8_t* field, size_t capacity, const std::string& value) {
    memset(field, 0, capacity);
    const size_t n = value.size() < capacity - 1 ? value.size() : capacity - 1;
    memcpy(field, value.data(), n);
}
HLE(s_savedata_setparam) {
    svc_log("sceSaveDataSetParam", a0,a1,a2,a3,a4,a5);
    const uint32_t type = (uint32_t)a1;
    const size_t need = savedata_param_type_size(type);
    if (!savedata_mount_point_ok(a0) || !a2 || !need || a3 < need) return SAVE_DATA_ERR_PARAMETER;
    const std::string dir = savedata0_mounted_dir();
    if (dir.empty()) return SAVE_DATA_ERR_NOT_MOUNTED;

    // Read-modify-write: a title sets TITLE and SUBTITLE in separate calls, so setting one field
    // must not erase the others. An absent block starts empty rather than from a guess.
    SaveDataParam param;
    save_param_load(dir, param);
    const uint8_t* buf = (const uint8_t*)PW(a2);
    bool mtime_from_guest = false;
    switch (type) {
        case SAVE_DATA_PARAM_TYPE_ALL: {
            param.title = savedata_param_text(buf + SAVE_DATA_PARAM_OFF_TITLE,
                                              SAVE_DATA_TITLE_MAXSIZE);
            param.sub_title = savedata_param_text(buf + SAVE_DATA_PARAM_OFF_SUBTITLE,
                                                  SAVE_DATA_SUBTITLE_MAXSIZE);
            param.detail = savedata_param_text(buf + SAVE_DATA_PARAM_OFF_DETAIL,
                                               SAVE_DATA_DETAIL_MAXSIZE);
            memcpy(&param.user_param, buf + SAVE_DATA_PARAM_OFF_USER_PARAM, sizeof(uint32_t));
            int64_t mtime = 0;
            memcpy(&mtime, buf + SAVE_DATA_PARAM_OFF_MTIME, sizeof(int64_t));
            // A zeroed mtime in an otherwise filled-in block means "you time-stamp it", which is
            // what the console does on write.
            if (mtime > 0) { param.mtime = mtime; mtime_from_guest = true; }
            break;
        }
        case SAVE_DATA_PARAM_TYPE_TITLE:
            param.title = savedata_param_text(buf, SAVE_DATA_TITLE_MAXSIZE); break;
        case SAVE_DATA_PARAM_TYPE_SUB_TITLE:
            param.sub_title = savedata_param_text(buf, SAVE_DATA_SUBTITLE_MAXSIZE); break;
        case SAVE_DATA_PARAM_TYPE_DETAIL:
            param.detail = savedata_param_text(buf, SAVE_DATA_DETAIL_MAXSIZE); break;
        case SAVE_DATA_PARAM_TYPE_USER_PARAM:
            memcpy(&param.user_param, buf, sizeof(uint32_t)); break;
        case SAVE_DATA_PARAM_TYPE_MTIME:
            memcpy(&param.mtime, buf, sizeof(int64_t)); mtime_from_guest = true; break;
        default: return SAVE_DATA_ERR_PARAMETER;
    }
    if (!mtime_from_guest) param.mtime = (int64_t)time(nullptr);
    if (!save_param_store(dir, param, app_param_declaration().title_id,
                          std::filesystem::path(dir).filename().string())) {
        // Loud, and an error rather than SCE_OK: the entire point of #2786 is that this call must
        // never report a write it did not perform.
        fprintf(stderr, "[svc] sceSaveDataSetParam: could not write the save parameter block to "
                        "\"%s\"\n", save_param_path(dir).c_str());
        return SAVE_DATA_ERR_INTERNAL;
    }
    if (svclog())
        fprintf(stderr, "[svc]   SetParam type=%u title=\"%s\" sub=\"%s\" userParam=%u -> stored\n",
                type, param.title.c_str(), param.sub_title.c_str(), param.user_param);
    return 0;
}

HLE(s_savedata_getparam) {
    svc_log("sceSaveDataGetParam", a0,a1,a2,a3,a4,a5);
    const uint32_t type = (uint32_t)a1;
    const size_t need = savedata_param_type_size(type);
    if (!savedata_mount_point_ok(a0) || !a2 || !need || a3 < need) return SAVE_DATA_ERR_PARAMETER;
    const std::string dir = savedata0_mounted_dir();
    if (dir.empty()) return SAVE_DATA_ERR_NOT_MOUNTED;

    // A save whose title has never set a parameter block genuinely has none, and an empty block
    // with a real byte count is the honest report of that. It is still not the pre-fix behaviour,
    // which wrote NOTHING into the caller's buffer and reported SCE_OK, so the title read back
    // whatever its own stack happened to hold.
    SaveDataParam param;
    save_param_load(dir, param);
    uint8_t* out = (uint8_t*)PW(a2);
    switch (type) {
        case SAVE_DATA_PARAM_TYPE_ALL:
            memset(out, 0, SAVE_DATA_PARAM_ALL_SIZE);
            savedata_param_write_text(out + SAVE_DATA_PARAM_OFF_TITLE, SAVE_DATA_TITLE_MAXSIZE,
                                      param.title);
            savedata_param_write_text(out + SAVE_DATA_PARAM_OFF_SUBTITLE,
                                      SAVE_DATA_SUBTITLE_MAXSIZE, param.sub_title);
            savedata_param_write_text(out + SAVE_DATA_PARAM_OFF_DETAIL, SAVE_DATA_DETAIL_MAXSIZE,
                                      param.detail);
            memcpy(out + SAVE_DATA_PARAM_OFF_USER_PARAM, &param.user_param, sizeof(uint32_t));
            memcpy(out + SAVE_DATA_PARAM_OFF_MTIME, &param.mtime, sizeof(int64_t));
            break;
        case SAVE_DATA_PARAM_TYPE_TITLE:
            savedata_param_write_text(out, SAVE_DATA_TITLE_MAXSIZE, param.title); break;
        case SAVE_DATA_PARAM_TYPE_SUB_TITLE:
            savedata_param_write_text(out, SAVE_DATA_SUBTITLE_MAXSIZE, param.sub_title); break;
        case SAVE_DATA_PARAM_TYPE_DETAIL:
            savedata_param_write_text(out, SAVE_DATA_DETAIL_MAXSIZE, param.detail); break;
        case SAVE_DATA_PARAM_TYPE_USER_PARAM:
            memcpy(out, &param.user_param, sizeof(uint32_t)); break;
        case SAVE_DATA_PARAM_TYPE_MTIME:
            memcpy(out, &param.mtime, sizeof(int64_t)); break;
        default: return SAVE_DATA_ERR_PARAMETER;
    }
    // gotSize is optional on this contract; write it only when the caller asked for one.
    if (a4) { const uint64_t got = need; memcpy(PW(a4), &got, sizeof(uint64_t)); }
    if (svclog())
        fprintf(stderr, "[svc]   GetParam type=%u -> %zu bytes (title=\"%s\" userParam=%u)\n",
                type, need, param.title.c_str(), param.user_param);
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
// the identical PS4 NID/API defines that layout). The dirName SLOT is now HIGH too, not MED:
// PPSA28061 Earthion zeroes 0x68 bytes and then does memcmp(event + 0x20, <dirName>, 0x20) at
// eboot+0x12f99, which pins dirName[32] at +0x20 in a 104-byte struct from a second title's bytes.
//
// KNOWN GAP -- prosper posts NO PER-OPERATION COMPLETION EVENT, and one title is already waiting for
// one. The only event this ever queues comes from Umount2, and it is zero-filled: no titleId, no
// dirName. Earthion's wait loop (eboot+0x12f30) polls until it gets an event whose dirName matches
// the directory it is waiting on, and takes its match branch at +0x12fb4 only when that memcmp
// succeeds -- which is STRUCTURALLY UNREACHABLE here. It survives only because it also has a
// give-up branch on the drained code. Implementing async Mount3/Prepare/Commit means (a) returning
// SAVE_DATA_ERR_IN_FLIGHT while the operation runs and (b) queueing a completion event that carries
// the operation's dirName -- not just bumping a counter. Tracked on #2909 / #1880.
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
            // dirName (+0x20) is deliberately left zeroed: this event is not attributed to any one
            // save directory, and inventing a name here would make Earthion's memcmp match the
            // wrong operation. See the KNOWN GAP above.
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

// --- libSceShare / libSceGameLiveStreaming: local lifecycle, features unavailable headless. -----
// The PS5 3.20 import table supplies the exact NIDs. Kyty's matching SDK surface gives the call
// shapes: ShareSetContentParam takes one required C string; GameLiveStreamingInitialize takes only
// a heap size. Neither successful call has an out-param. CONFIDENCE: HIGH for the live Sonic call
// shapes (PROSPER_SVCLOG), MED for the reference-derived invalid-param value.
HLE(s_share_ok) { return 0; }
HLE(s_share_content_param) {
    svc_log("sceShareSetContentParam", a0,a1,a2,a3,a4,a5);
    return a0 ? 0 : 0x81960002ull; // SCE_SHARE_ERROR_INVALID_PARAM (Kyty libShare.cpp)
}
HLE(s_live_streaming_init) {
    svc_log("sceGameLiveStreamingInitialize", a0,a1,a2,a3,a4,a5);
    return 0;
}

// --- libSceNpUniversalDataSystem (PS5 telemetry/activities): hand out ids, stay inert. ----------
// PS5-only, no reference implementation; by symmetry with every Np Create* API the first arg of
// CreateContext/CreateHandle is the out-id pointer (pointer-range-guarded so a wrong guess can't
// fault). Offline console: everything else no-ops. CONFIDENCE: LOW (guarded).
HLE(s_npuds_create) { svc_log("sceNpUniversalDataSystemCreate*", a0,a1,a2,a3,a4,a5);
                      if (svc_ptrish(a0)) *(int32_t*)PW(a0) = 1; return 0; }
HLE(s_npuds_ok)     { return 0; }
// Two observed CreateEvent call shapes (both write opaque ids the guest never dereferences):
//  * Dead Cells: CreateEvent(name?, reserved, Event** a2, PropertyObject** a3) — a2 AND a3 are
//    out-pointers.
//  * Alex Kidd DX (PPSA02664, svc dump): CreateEvent(name a0="activityTerminate", ctx a1,
//    Event** a2, 0, PropertyObject** a4, 0) — outs are a2 and a4, a3 is literal 0, a5 literal 0.
//    The old a3-must-be-a-pointer guard returned NP-invalid-argument here, and the title retried
//    every frame forever — holding its cutscene->gameplay transition on a black screen (#320).
// Accept both: a2 is always the event out; the property out is a3 when pointer-like. The a4 form is
// admitted ONLY for Alex Kidd's exact trailing shape (a3==0 AND a5==0) — svc_ptrish is a wide range
// check that cannot by itself tell a real out-pointer from leftover-register garbage, so requiring
// both trailing reserved words to be zero pins the write to the observed 6-arg layout. A different
// title that legitimately passes a3==0 with a non-zero/garbage a5 hits neither property write and
// still returns success (0), which is what actually stops the retry loop. CONFIDENCE: LOW.
HLE(s_npuds_create_event) {
    svc_log("sceNpUniversalDataSystemCreateEvent", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a2)) return 0x80550003ull; // NP invalid argument
    *(uint64_t*)PW(a2) = g_handle.fetch_add(1);
    if (svc_ptrish(a3))                                *(uint64_t*)PW(a3) = g_handle.fetch_add(1);
    else if (a3 == 0 && a5 == 0 && svc_ptrish(a4))     *(uint64_t*)PW(a4) = g_handle.fetch_add(1);
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
    if (!g_gameintent_initialized.exchange(true, std::memory_order_acq_rel))
        g_gameintent_event_delivered.store(false, std::memory_order_release);
    g_gameintent_data_ptr.store(0, std::memory_order_release);
    return 0;
}
HLE(s_gameintent_term) {
    svc_log("sceNpGameIntentTerminate", a0,a1,a2,a3,a4,a5);
    g_gameintent_data_ptr.store(0, std::memory_order_release);
    g_gameintent_event_delivered.store(false, std::memory_order_release);
    g_gameintent_initialized.store(false, std::memory_order_release);
    return 0;
}
HLE(s_gameintent_receive) {
    svc_log("sceNpGameIntentReceiveIntent", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a0)) return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;

    const int32_t invalid_user = -1;
    const char empty_type[NP_GAME_INTENT_TYPE_SIZE]{};
    static const uint8_t empty_data[NP_GAME_INTENT_DATA_SIZE]{};
    if (!svc_write_bytes(a0 + 8, &invalid_user, sizeof(invalid_user)) ||
        !svc_write_bytes(a0 + NP_GAME_INTENT_TYPE_OFFSET, empty_type, sizeof(empty_type)) ||
        !svc_write_bytes(a0 + NP_GAME_INTENT_DATA_OFFSET, empty_data, sizeof(empty_data))) {
        return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;
    }
    g_gameintent_data_ptr.store(0, std::memory_order_release);

    if (!g_gameintent_initialized.load(std::memory_order_acquire))
        return NP_GAME_INTENT_ERROR_INTENT_NOT_FOUND;
    if (!gameintent_activity_id()) return NP_GAME_INTENT_ERROR_INTENT_NOT_FOUND;

    const int32_t initial_user = 1;
    constexpr char launch_activity[] = "launchActivity";
    if (!svc_write_bytes(a0 + 8, &initial_user, sizeof(initial_user)) ||
        !svc_write_bytes(a0 + NP_GAME_INTENT_TYPE_OFFSET, launch_activity,
                         sizeof(launch_activity))) {
        return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;
    }
    g_gameintent_data_ptr.store(a0 + NP_GAME_INTENT_DATA_OFFSET, std::memory_order_release);
    return 0;
}
HLE(s_gameintent_get_property_string) {
    svc_log("sceNpGameIntentGetPropertyValueString", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a0) || !svc_ptrish(a1) || !svc_ptrish(a2) || a3 == 0)
        return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;

    char key[sizeof("activityId")]{};
    if (!svc_copy_bytes(a1, key, sizeof(key))) return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;

    size_t activity_length = 0;
    const char* activity = gameintent_activity_id(&activity_length);
    if (a0 != g_gameintent_data_ptr.load(std::memory_order_acquire) || !activity ||
        memcmp(key, "activityId", sizeof(key)) != 0) {
        const char empty = '\0';
        if (!svc_write_bytes(a2, &empty, sizeof(empty)))
            return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;
        return NP_GAME_INTENT_ERROR_VALUE_NOT_FOUND;
    }
    if (a3 < activity_length + 1) return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;
    if (!svc_write_bytes(a2, activity, activity_length + 1))
        return NP_GAME_INTENT_ERROR_INVALID_ARGUMENT;
    return 0;
}
HLE(s_npent_addcont_info) {
    svc_log("sceNpEntitlementAccessGetAddcontEntitlementInfo", a0,a1,a2,a3,a4,a5);
    std::string label;
    if (a0 > UINT32_MAX || !appcontent_read_label(a1, label) || !svc_ptrish(a2))
        return NP_ENTITLEMENT_ERROR_PARAMETER;
    const AddcontentInventorySnapshot inventory = addcontent_inventory_snapshot();
    const InstalledAddcontent* entry = inventory.state == AddcontentInventoryState::Ready
        ? find_addcontent(inventory, static_cast<uint32_t>(a0), label) : nullptr;
    if (!entry) return NP_ENTITLEMENT_ERROR_NO_ENTITLEMENT;
    const NpAddcontEntitlementInfo info = npent_info(*entry);
    return svc_write_bytes(a2, &info, sizeof(info)) ? 0 : NP_ENTITLEMENT_ERROR_PARAMETER;
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
namespace {
#ifndef _WIN32
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
#endif
}

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
namespace {
#ifndef _WIN32
std::atomic<uint64_t> g_netctl_cb_fn{0};
std::atomic<uint64_t> g_netctl_cb_arg{0};
std::atomic<int>      g_netctl_cb_delivered{0};
#endif
}
#ifndef _WIN32
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
namespace {
#ifndef _WIN32
struct NpStateCbSlot { std::atomic<uint64_t> fn{0}, arg{0}; std::atomic<int> delivered{0}; };
NpStateCbSlot     g_np_state_cbs[4];
std::atomic<int>  g_np_state_cb_n{0};
#endif
}
#ifndef _WIN32
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
// thinks failed. A PlatformUi that accepts Open is pinned as the non-owning expected backend; every
// later callback obtains a registry lease for that exact pointer. Unregister/replacement abandons
// safely to headless FINISHED and can never reroute to a backend that did not accept the request.
// CONFIDENCE: HIGH (shadPS4 implements this exact lifecycle).
namespace {
std::atomic<int> g_errdialog_status{0 /*NONE*/};
std::atomic<PlatformUi*> g_errdialog_ui{nullptr};

void errdialog_close_owner() {
    PlatformUi* expected = g_errdialog_ui.exchange(nullptr);
    if (!expected) return;
    auto ui = platform_ui_lease(expected);
    if (ui) ui->errorDialogClose();
}

PlatformUiLease errdialog_owner_lease(PlatformUi*& expected) {
    expected = g_errdialog_ui.load(std::memory_order_acquire);
    if (!expected) return {};
    return platform_ui_lease(expected);
}

void errdialog_abandon_owner(PlatformUi* expected) {
    if (!expected || !g_errdialog_ui.compare_exchange_strong(expected, nullptr)) return;
    // Publish the bounded foreground-return phase before the fallback FINISHED status. A thread
    // that acquires FINISHED can therefore never observe the pre-dialog idle phase.
    finish_error_dialog_background();
    g_errdialog_status.store(3 /*FINISHED: safe headless fallback*/, std::memory_order_release);
}
}
HLE(s_errdialog_init)   {
    errdialog_close_owner();
    g_error_dialog_background_phase.store(ErrorDialogIdle);
    g_errdialog_status.store(1 /*INITIALIZED*/);
    return 0;
}
HLE(s_errdialog_open)   {
    errdialog_close_owner();
    g_error_dialog_background_phase.store(ErrorDialogIdle);
    // Offer the error dialog to a registered PlatformUi first (a real message box); else auto-dismiss.
    {
        auto ui = platform_ui_lease();
        if (ui) {
            begin_error_dialog_background(true);
            if (ui->errorDialogOpen(a0)) {
                g_errdialog_ui.store(ui.get(), std::memory_order_release);
                return 0;
            }
        }
    }
    uint32_t code = 0;
    if (svc_ptrish(a0)) code = *(const uint32_t*)((const char*)PW(a0) + 4);
    fprintf(stderr, "[svc] sceErrorDialogOpen(errorCode=%#x) -> auto-dismiss FINISHED\n", code);
    // Arm first: publishing FINISHED before this release lets another guest thread act on dialog
    // completion while SystemService still exposes the pre-dialog idle state.
    begin_error_dialog_background(false);
    g_errdialog_status.store(3 /*FINISHED (auto-dismiss)*/, std::memory_order_release);
    return 0;
}
HLE(s_errdialog_close)  {
    errdialog_close_owner();
    finish_error_dialog_background();
    g_errdialog_status.store(3 /*FINISHED*/);
    return 0;
}
HLE(s_errdialog_term)   {
    errdialog_close_owner();
    finish_error_dialog_background();
    g_errdialog_status.store(0 /*NONE*/);
    return 0;
}
HLE(s_errdialog_status) {
    PlatformUi* expected = nullptr;
    auto ui = errdialog_owner_lease(expected);
    if (expected && ui) {
        const int status = ui->errorDialogStatus();
        if (status == 0 /*NONE*/ || status == 3 /*FINISHED*/) finish_error_dialog_background();
        return (uint64_t)(unsigned)status;
    }
    if (expected) errdialog_abandon_owner(expected);
    return (uint64_t)(unsigned)g_errdialog_status.load(std::memory_order_acquire);
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
// Entries come only from the validated local installation inventory. A title with no dlc_emu.ini
// retains the retail no-DLC answer SUCCESS/hitNum=0; a present but invalid manifest fails visibly.
// CONFIDENCE: HIGH on the shape (two independent PS5 guests allocate 0x1c bytes per entry and Sonic
// directly consumes status at +0x18). Package enum values and errno behavior are producer-pinned.
HLE(s_npent_addcont_list) {
    svc_log("sceNpEntitlementAccessGetAddcontEntitlementInfoList", a0,a1,a2,a3,a4,a5);
    if (a0 > UINT32_MAX) return NP_ENTITLEMENT_ERROR_PARAMETER;
    return addcontent_info_list<NpAddcontEntitlementInfo>(
        a1, a2, a3, static_cast<uint32_t>(a0), NP_ENTITLEMENT_ERROR_PARAMETER,
        NP_ENTITLEMENT_ERROR_NO_ENTITLEMENT, NP_ENTITLEMENT_ADDCONT_LIST_MAX, npent_info);
}
// GetEntitlementKey(serviceLabel, const Label* label, Key* out) — live capture: retried 4x with
// identical args (the garbage-consuming retry signature). Crisis Core independently consumes
// exactly 16 bytes from the successful output. Validated installed records use either their explicit
// key or the producer's deterministic default; unknown labels retain honest failure.
HLE(s_npent_getkey) {
    svc_log("sceNpEntitlementAccessGetEntitlementKey", a0,a1,a2,a3,a4,a5);
    std::string label;
    if (a0 > UINT32_MAX || !appcontent_read_label(a1, label) || !svc_ptrish(a2))
        return NP_ENTITLEMENT_ERROR_PARAMETER;
    const AddcontentInventorySnapshot inventory = addcontent_inventory_snapshot();
    const InstalledAddcontent* entry = inventory.state == AddcontentInventoryState::Ready
        ? find_addcontent(inventory, static_cast<uint32_t>(a0), label) : nullptr;
    if (!entry) return NP_ENTITLEMENT_ERROR_NO_ENTITLEMENT;
    return svc_write_bytes(a2, entry->entitlement_key.data(), entry->entitlement_key.size())
        ? 0 : NP_ENTITLEMENT_ERROR_PARAMETER;
}

// sceNpEntitlementAccessGetSkuFlag(int32_t* skuFlag) — the SAME question
// sceAppContentAppParamGetInt answers for paramId 0, asked through libSceNpEntitlementAccess. It is a
// platform query about the SKU of the LOCALLY INSTALLED application, not a query about anything a
// user purchased from a network service, so it is answered from the same one local derivation and the
// two libraries cannot disagree.
//
// The NID was registered nowhere, so it fell to the dispatcher's unimplemented stub, which reports
// SUCCESS while leaving the out pointer untouched. That is worse than a wrong value: of the four
// titles in the project's local dumps that call this, three (PPSA07809, PPSA02154, PPSA05143) hand it
// an uninitialized stack slot and then act on whatever residue was there, and the fourth (PPSA04263)
// pre-seeds TRIAL and keeps it. Failing when the SKU is unknown is a path all four already handle —
// each falls back to its own conservative default — whereas reporting a SKU prosper cannot derive is
// a value the guest cannot tell from a real one.
//
// THE OUT POINTER LEADS (arg 0), and that needed proving rather than assuming: both neighbouring exports
// in this library (GetAddcontEntitlementInfoList, GetEntitlementKey) lead with a SceNpServiceLabel, so
// if this one did too, writing through a0 would be writing through a label and the whole fix would be
// inert with every test still green. A live GTA V (PPSA04263) boot under PROSPER_SVCLOG settles it — the
// second line is the load-bearing one, because it is an observation of the slot rather than a reading of
// guest code:
//   [svc] sceNpEntitlementAccessGetSkuFlag(0x7f4fc82d6e2c, 0, 0xb, 0, 0x1d, 0xb)
//   [svc]   a0 -> 0001a9bf00000001 0000003ab0ba4900 …
// a0 is a stack address, and the low dword at it is 0x00000001 — the guest's own pre-seeded TRIAL
// default, which is what the unimplemented stub used to leave standing. Only the SHAPE of the pointer
// carries the argument: 0x7f4f… is run-local under ASLR, so a later run will print different digits and
// that is not a disagreement.
// What this does NOT establish: whether a trailing SceNpServiceLabel parameter exists. An
// (out, label = 0) two-argument form would log byte-identically, and the sibling capture in the same run
// shows label 0 is exactly what these calls pass — GetAddcontEntitlementInfoList(0, 0, 0, 0x7f4f…, …).
// Unresolved and immaterial here: this handler ignores a1 either way.
//
// CONFIDENCE: HIGH on the contract and the enum (guest-pinned in three titles).
//
// LOW on the exact errno for the unknown-SKU case, and deliberately left admitted rather than dressed
// up. NO_ENTITLEMENT is a semantic stretch for "prosper cannot derive this application's SKU": it is a
// placeholder picked inside the producer-pinned 0x817D NpEntitlementAccess facility, NOT a known
// contract, and a more precise-looking value invented here would be worse than the stretch, because
// the next reader would treat a specific code as evidence that the contract is known.
// What makes the placeholder safe: no caller's behaviour depends on the value. All four titles in the
// project's local dumps test only for non-zero and fall back to their own conservative default, so the
// observable behaviour is identical for any error in this facility.
// What would settle it, and let this be replaced without re-deriving anything above: a title observed
// branching on a specific error code from this call, or the value observed from real firmware.
HLE(s_npent_skuflag) {
    svc_log("sceNpEntitlementAccessGetSkuFlag", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a0)) return NP_ENTITLEMENT_ERROR_PARAMETER;
    const AppParamDeclaration decl = app_param_declaration();
    if (!decl.declared || !decl.sku_flag) return NP_ENTITLEMENT_ERROR_NO_ENTITLEMENT;
    const int32_t value = (int32_t)*decl.sku_flag;
    return svc_write_bytes(a0, &value, sizeof(value)) ? 0 : NP_ENTITLEMENT_ERROR_PARAMETER;
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

// sceSaveDataTransferringMountPs4 — the SIBLING of the call above, and the one Sonic Frontiers
// (PPSA03831) uses. It mounts the **PS4 edition's** save data so a PS5 title can import it on first
// run. A PS5 installation that was never upgraded from a PS4 copy has no such save area, and that
// is every installation prosper can present: prosper has no PS4 save-data store at all, and no
// local dump carries one, so "no PS4 save data exists here" is derived from local inventory rather
// than assumed — the same NOT_FOUND a Mount3 of a nonexistent save returns, in the same 0x809F
// facility.
//
// Unregistered, this reached `prosper_on_unimpl`'s `return 0` — the FALSE SUCCESS class (#2081) —
// and it produced the *identical* downstream signature its sibling's comment records for DOLL.
// Frontiers zeroes its 32-byte mount-point result, calls this, is told the mount succeeded, and
// then formats "<mountPoint>/gamedata" out of the still-empty result and opens **`/gamedata`** at
// filesystem root. That open fails ENOENT, the title retries it once per frame forever, and the
// boot state machine never leaves GameModeInitialize: measured live, ~1,450 failed `/gamedata`
// opens and 1,319 dispatcher hits on this NID in one 60 s CPU-only arm. All five guest call sites
// gate on the result (`test eax,eax`, `tools/re/nid_gate_scan.py --nid RjMlsR8EXrw`), and the
// error arm returns straight out of the transfer check — so an honest error is a path the title
// already has, not one this invents.
//
// Nothing is written to the result: a caller that reads a mount point after an error is reading
// its own buffer, which is exactly what must not be papered over. CONFIDENCE: HIGH that success
// is wrong here (the guest's own use of the unwritten result is observed); MED on NOT_FOUND being
// the precise firmware errno, LOW on the argument layout (nothing is assumed — nothing is read).
HLE(s_savedata_transfermount_ps4) {
    svc_log("sceSaveDataTransferringMountPs4", a0,a1,a2,a3,a4,a5);
    return SAVE_DATA_ERR_NOT_FOUND;
}

// sceSaveDataDirNameSearchPs4 (X4MYzukPc3g) -- the PS4 sibling of sceSaveDataDirNameSearch
// (dyIhnXq-0SM, registered below), and with the entry point above the ONLY two PS4-namespace exports
// libSceSaveData has across all 275 PS5 3.20 libraries. Fixing it closes the pair (#2210).
//
// Unregistered, this reached prosper_on_unimpl's `return 0` -- SCE_OK for this contract, the FALSE
// SUCCESS class (#2081). The caller is told a search over PS4 save data SUCCEEDED and the result
// struct it passed is never written. That reads as "zero hits" only because callers happen to zero
// the struct first: right by accident, not by construction. A caller reusing a result struct, or one
// whose hit count lands on non-zero stack residue, is handed a count over memory nothing wrote --
// the #213 shape, where a garbage count sized a 34 GB array.
//
// NOT_FOUND is derived from local inventory, not assumed: prosper has no PS4 save-data store at all
// (its two save areas are both PS5-side -- PROSPER_SAVEDATA_DIR -> save-data-memory, PROSPER_SAVE0 ->
// the mounted /savedata0) and no local dump carries a PS4 save area, so a search over PS4 save data
// honestly finds nothing. Same 0x809F facility, same answer a search of a nonexistent save returns.
//
// NOTHING is written to the result, deliberately. Returning SCE_OK with an explicitly-zeroed hit
// count would also be defensible -- but only if the argument layout were established from live
// evidence, and it is not. Inventing a written result is the exact mirror of the defect being fixed
// (#2208's failure was a guest reading an unwritten result). A caller that reads a result after an
// error is reading its own buffer, which is what must not be papered over.
//
// CONFIDENCE: HIGH that success is wrong here (it is SCE_OK over an unwritten out-struct by
// construction); MED on NOT_FOUND being the precise firmware errno; LOW on the argument layout --
// nothing is assumed about it because nothing is read or written.
HLE(s_savedata_dirname_search_ps4) {
    svc_log("sceSaveDataDirNameSearchPs4", a0,a1,a2,a3,a4,a5);
    // #3124: answer this exactly as the PS5 sibling above does, with an EXPLICIT zero-hit result.
    //
    // #2302 replaced the dispatcher's `return 0` with NOT_FOUND, and its diagnosis was right: SCE_OK
    // over an out-struct nothing wrote is the false-success class, correct only by accident when the
    // caller happens to pre-zero. But it changed the answer for a call it had not seen, and said so:
    // "Not observed being CALLED at any boot depth reached so far". Tactics Ogre: Reborn (PPSA03839)
    // calls it, once, right after sceSaveDataInitialize3 -- and on NOT_FOUND the title submits two
    // DCBs, draws once and then stops submitting while staying alive, rendering one black frame for
    // the rest of the run. Bisected over 700 commits.
    //
    // The commit named the better option and declined it only for want of evidence: "Returning
    // SCE_OK with an explicitly-zeroed hit count would also be defensible -- but only if the
    // argument layout were established from live evidence, and it is not."
    //
    // IT IS NOW, twice over. Captured with PROSPER_SVCLOG=1 on this title, the PS4 and PS5 calls
    // carry a byte-identical result struct -- [0x00]=0, [0x08]=caller buffer pointer, [0x10]=0x400
    // capacity -- and s_savedata_dirsearch above already writes that same layout (hitNum @0x00,
    // dirNames @0x08, dirNamesNum @0x10, setNum @0x14), derived from live evidence in #299.
    //
    // So this is not an invented result. Zero hits is the honest answer: prosper has no PS4 save
    // area at all -- both its save roots are PS5-side (PROSPER_SAVEDATA_DIR, PROSPER_SAVE0) -- so a
    // search over PS4 save data genuinely finds nothing. Writing the count makes that true BY
    // CONSTRUCTION rather than by the caller's habit of zeroing first, which is precisely what
    // #2302 objected to. On the observed caller the write is a no-op (the field is already 0); on
    // one that does not pre-zero it replaces stack residue, the #213 shape where a garbage count
    // sized a 34 GB array.
    //
    // The charter's rule also binds here: the same question must be answered the same way through
    // every library that exposes it. The PS5 spelling enumerates and reports its count; the PS4
    // spelling reporting a hard error for the same question was a divergence on top of the
    // regression.
    //
    // CONFIDENCE: HIGH that zero hits is the correct answer and that SCE_OK unblocks the title
    // (measured both ways). HIGH on the layout -- two independent captures plus the sibling's
    // established use of the same offsets.
    if (!a1) return 0x809F0000ull;             // SAVE_DATA_ERROR_PARAMETER, as the sibling does
    uint8_t* res = (uint8_t*)PW(a1);
    if (!res) return 0x809F0000ull;
    *(uint32_t*)(res + 0x00) = 0;              // hitNum
    *(uint32_t*)(res + 0x14) = 0;              // setNum
    return 0;
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

// ===== libSceRandom ============================================================================
//
// `sceRandomGetRandomNumber(void* buf, size_t size)` is the library's ONLY export (PS5 3.20
// `libSceRandom.c` lists exactly one `sprx_dlsym` line), and 27 of the 44 local dumps import it.
//
// Unregistered, it reached the dispatcher's `return 0` — and for this contract 0 is SCE_OK. The
// guest was told its buffer had been filled while nothing wrote to it, so it read back whatever
// its own stack or heap already held and used that as entropy. The worst property of that failure
// is that it is STABLE: a fresh stack allocation at the same call site tends to hold the same
// residue on every call, so a "random" session id or nonce can be identical all run. Nothing
// crashes and no diagnostic fires; only code that inspected the buffer could notice. (#2065, swept
// under #2081.)
//
// The fix is a real implementation rather than a fail-visible stub, because the honest answer here
// is cheap: the host has a CSPRNG. Returning an error instead would be strictly worse — this is a
// *value-producing* contract, and guests do gate on it.
//
// How the guests actually consume it, measured with `tools/re/nid_gate_scan.py --nid PI7jIZj4pcE`
// over the local dumps (call sites classified by what happens to eax):
//
//     PPSA08804  nonzero=2 ignored=1        PPSA04263  nonzero=2
//     PPSA24651  nonzero=1                  PPSA13579  nonzero=1
//     PPSA17942  ignored=1                  PPSA19244  forward=1 ignored=1
//
// So most call sites branch on "did this fail", and — the load-bearing part — NOT ONE compares the
// result against a specific constant (no `const` / `other-cmp` site anywhere). That is what makes
// the reject arms safe to add: the guest reads only the SIGN of the answer, never its value.
// CONFIDENCE: HIGH that success must fill the buffer and that failure must be non-zero;
// LOW on the exact error constant — the SCE_RANDOM error space is not in the 3.20 dump (which
// carries names and NIDs only) and no call site discriminates it, so the libkernel encoding is
// used per the project's default rather than an invented SCE_RANDOM_* value.
namespace {

// The published single-request cap. A guest asking for more than this gets an error on hardware, so
// prosper must not quietly serve it: a partial fill reported as success would recreate the exact
// bug this handler exists to remove, and an over-long fill would paper over a guest bug that real
// hardware rejects.
//
// CONFIDENCE: MED — the cap is from the API documentation, not from a live capture. This is also
// the ONLY arm of this handler that can newly FAIL a call the previous stub "succeeded", so it is
// the one place a wrong constant costs something rather than merely being imprecise.
//
// What is NOT established: the request sizes local titles actually pass. `nid_gate_scan` classifies
// what the guest does with `eax` and cannot recover an argument, so no instrument here has measured
// the `size` operand at any call site — deliberately stated rather than left as an implied "we
// checked". Settling it needs the argument registers read at the call, e.g. PROSPER_SVCLOG-style
// logging on this NID or a hardware breakpoint at the import. If a title is ever found requesting
// more, this constant is where to look first.
constexpr uint64_t kRandomMaxBytes = 64;

// Fill `bytes` from the host CSPRNG. Returns false if the host cannot supply entropy — which the
// caller MUST surface as an error, never as a zero-filled success. Deterministic zeros presented as
// random are the same lie in a different costume.
bool svc_host_entropy(void* dst, size_t bytes) {
#ifdef _WIN32
    return BCryptGenRandom(nullptr, (PUCHAR)dst, (ULONG)bytes,
                           BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
    auto* p = static_cast<unsigned char*>(dst);
    while (bytes) {
        const size_t chunk = bytes < 256 ? bytes : 256;   // getentropy's published maximum
        if (getentropy(p, chunk) != 0) return false;
        p += chunk;
        bytes -= chunk;
    }
    return true;
#endif
}

} // namespace

HLE(s_random_get_random_number) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    const uint64_t buf = a0, size = a1;
    // A zero-length request asks for nothing and trivially gets it; there is no buffer to leave
    // stale, so this is a real success rather than the empty kind.
    if (size == 0) return 0;
    if (size > kRandomMaxBytes) return prosper::hle::kSceKernelErrorEINVAL;
    if (!svc_ptrish(buf)) return prosper::hle::kSceKernelErrorEFAULT;

    unsigned char tmp[kRandomMaxBytes];
    if (!svc_host_entropy(tmp, (size_t)size))
        return prosper::hle::sce_kernel_error(prosper::hle::FreeBsdErrno::EIo);
    // Fault-contained: an unmapped or unwritable guest buffer must return an error, not take down
    // the emulator, and must not report success for a write that did not land.
    if (!svc_write_bytes(buf, tmp, (size_t)size)) return prosper::hle::kSceKernelErrorEFAULT;
    return 0;
}

void register_service_hle() {
    register_json2_hle();
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
#ifndef _WIN32
    // NetCtl offline-console state delivery — default ON since #306 (see block comment above).
    // PROSPER_NETCTL_CB=0 restores the previous unimplemented behavior.
    {
        // DEFAULT ON since #306. `strtol` answered 0 for `=yes`/`=true`/`=on`, and the consequence
        // is not a lost diagnostic: the three callback NIDs guarded here (RegisterCallback,
        // CheckCallback, GetState) go UNREGISTERED, so the guest gets the pre-#306 unimplemented
        // behaviour from what the operator read as "enable it". GetInfo and GetResult below are
        // registered unconditionally and are unaffected (#3267).
        const char* e = getenv("PROSPER_NETCTL_CB");
        if (prosper::diag::env_u64_or_default_auto("PROSPER_NETCTL_CB", e, 1ull) != 0) {
            R("sceNetCtlRegisterCallback", s_netctl_register_cb);      // UJ+Z7Q+4ck0
            R("sceNetCtlCheckCallback",    s_netctl_check_cb_entry);   // iQw3iQPhvUQ
            R("sceNetCtlGetState",         s_netctl_getstate);         // uBPlr0lbuiI
        }
    }
#endif
    Hle::register_fn("obuxdTiwkF8", (HleFn)s_netctl_getinfo, "sceNetCtlGetInfo");  // NOT_CONNECTED
    Hle::register_fn("0cBgduPRR+M", (HleFn)s_netctl_getresult, "sceNetCtlGetResult");
    Hle::register_fn("dgJBaeJnGpo", (HleFn)s_net_pool_create, "sceNetPoolCreate");
    Hle::register_fn("hdpVEUDFW3s", (HleFn)s_ssl_init, "sceSslInit");
    // libSceHttp2 lives in src/hle/net/hle_http2.cpp now, sceHttp2Init with it (#2894): the
    // context it returns is a slot that library's own create path validates against.
    Hle::register_fn("+o9816YQhqQ", (HleFn)s_npweb_init, "sceNpWebApi2Initialize");
    Hle::register_fn("sk54bi6FtYM", (HleFn)s_npweb_create_user_context,
                     "sceNpWebApi2CreateUserContext");
    // NpTrophy2: the config/info queries whose success-with-garbage-out crashed DOLL (see above).
    // ALL FIVE of the library's info queries must answer here, not just the two that a title
    // happened to crash on. Each writes its result through a caller-supplied out-struct, so any one
    // of them left unregistered returns the dispatcher's 0 — SCE_OK — over memory nothing wrote,
    // which is the failure #213 diagnosed (a heap-garbage trophy count sized a 34 GB array). The
    // singular/plural pairs are the trap: registering `…TrophyInfoArray` and not `…TrophyInfo`
    // leaves the identical shape live behind a name that looks covered. #1956, swept under #2081.
    Hle::register_fn("4IzqhhUQ3nk", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetGameInfo");
    Hle::register_fn("y3zHpdZO6ME", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetTrophyInfoArray");
    Hle::register_fn("EwNylPdWUTM", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetTrophyInfo");
    Hle::register_fn("DoZWauG8mu0", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetGroupInfo");
    Hle::register_fn("+PDSI6WgPRc", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetGroupInfoArray");
    // libSceRandom's only export — see the block comment above s_random_get_random_number.
    Hle::register_fn("PI7jIZj4pcE", (HleFn)s_random_get_random_number, "sceRandomGetRandomNumber");
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
    Hle::register_fn("qbwy0Ub8b3M", (HleFn)s_user_number, "sceUserServiceGetUserNumber");
    // Sonic imports LoginDialog only to initialize the service at startup.  There is no UI to show
    // until Open is requested, so initialization is a truthful successful no-op in the headless HLE.
    Hle::register_fn("qP-EvQRl2Hc", (HleFn)s_ok, "sceLoginDialogInitialize");
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
    Hle::register_fn("Oad3rvY-NJQ", (HleFn)s_np_has_signed_up, "sceNpHasSignedUp");
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
    // Add-content (DLC) enumeration/mount comes only from the validated installed inventory.
    // No manifest retains no-DLC truth (hitNum=0 / no entitlement), not garbage-count OOM (#213).
    Hle::register_fn("xnd8BJzAxmk", (HleFn)s_appcontent_addcont_list,   "sceAppContentGetAddcontInfoList");
    Hle::register_fn("m47juOmH0VE", (HleFn)s_appcontent_addcont_info,   "sceAppContentGetAddcontInfo");
    Hle::register_fn("XTWR0UXvcgs", (HleFn)s_appcontent_entitlement_key, "sceAppContentGetEntitlementKey");
    Hle::register_fn("VANhIWcqYak", (HleFn)s_appcontent_addcont_mount, "sceAppContentAddcontMount");
    // The inverse. Unregistered it answered SCE_OK without releasing the claim, so a
    // mount/unmount/re-mount cycle was permanently BUSY (#2004, swept under #2081).
    Hle::register_fn("3rHWaV-1KC4", (HleFn)s_appcontent_addcont_unmount, "sceAppContentAddcontUnmount");
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
    Hle::register_fn("V-uEeFKARJU", (HleFn)s_savedlg_progress_inc, "sceSaveDataDialogProgressBarInc");
    Hle::register_fn("hay1CfTmLyA", (HleFn)s_savedlg_progress_set, "sceSaveDataDialogProgressBarSetValue");
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
    Hle::register_fn("PMVehSlfZ94", (HleFn)s_ime_kbd_close, "sceImeKeyboardClose");
#ifdef _WIN32
    Hle::register_fn("-4GCfYdNF1s", (HleFn)s_ime_update, "sceImeUpdate");         // plain handler (guest_fs=0)
#else
    Hle::register_fn("-4GCfYdNF1s", (HleFn)s_ime_update_entry, "sceImeUpdate");   // entry-rsp trampoline (#1286)
#endif
    Hle::register_fn("VkqLPArfFdc", (HleFn)s_ime_kbd_info,  "sceImeKeyboardGetInfo");
    Hle::register_fn("dKadqZFgKKQ", (HleFn)s_ime_kbd_resid, "sceImeKeyboardGetResourceId");
    R("sceSystemServiceGetStatus", s_syss_getstatus);
    R("sceSystemServiceReceiveEvent", s_sysservice_receiveevent);   // NO_EVENT, don't leave the struct garbage
    // sceSystemServiceGetDisplaySafeAreaInfo (1n37q1Bvc5Y) — fill ratio=1.0 (see s_syss_safearea).
    Hle::register_fn("1n37q1Bvc5Y", (HleFn)s_syss_safearea, "sceSystemServiceGetDisplaySafeAreaInfo");
    Hle::register_fn("mPpPxv5CZt4", (HleFn)s_syss_hdr_luminance,
                     "sceSystemServiceGetHdrToneMapLuminance");
    // #3119: the guest reporting its OWN crash must not look like a hang. See the handler.
    Hle::register_fn("3s8cHiCBKBE", (HleFn)s_syss_report_abnormal_termination,
                     "sceSystemServiceReportAbnormalTermination");

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
    Hle::register_fn("v7AAAMo0Lz4", (HleFn)s_savemem_setup_v1, "sceSaveDataSetupSaveDataMemory");
    Hle::register_fn("h3YURzXGSVQ", (HleFn)s_savemem_set_v1,   "sceSaveDataSetSaveDataMemory");
    Hle::register_fn("7Bt5pBC-Aco", (HleFn)s_savemem_get_v1,   "sceSaveDataGetSaveDataMemory");
    Hle::register_fn("wiT9jeC7xPw", (HleFn)s_savemem_sync,  "sceSaveDataSyncSaveDataMemory");
    Hle::register_fn("yKDy8S5yLA0", (HleFn)s_savedata_term,      "sceSaveDataTerminate");
    Hle::register_fn("gjRZNnw0JPE", (HleFn)s_savedata_txres,     "sceSaveDataCreateTransactionResource");
    Hle::register_fn("lJUQuaKqoKY", (HleFn)s_savedata_txres_del, "sceSaveDataDeleteTransactionResource");
    Hle::register_fn("32HQAQdwM2o", (HleFn)s_savedata_mount,     "sceSaveDataMount");
    Hle::register_fn("0z45PIH+SNI", (HleFn)s_savedata_mount2,    "sceSaveDataMount2");
    Hle::register_fn("ZP4e7rlzOUk", (HleFn)s_savedata_mount3,    "sceSaveDataMount3");
    Hle::register_fn("BMR4F-Uek3E", (HleFn)s_savedata_umount,    "sceSaveDataUmount");
    Hle::register_fn("uW4vfTwMQVo", (HleFn)s_savedata_umount2,   "sceSaveDataUmount2");
    Hle::register_fn("sDCBrmc61XU", (HleFn)s_savedata_prepare,   "sceSaveDataPrepare");
    Hle::register_fn("ie7qhZ4X0Cc", (HleFn)s_savedata_commit,    "sceSaveDataCommit");
    // #2786: both were unregistered, so both answered SCE_OK while storing and reading nothing.
    Hle::register_fn("85zul--eGXs", (HleFn)s_savedata_setparam,  "sceSaveDataSetParam");
    Hle::register_fn("XgvSuIdnMlw", (HleFn)s_savedata_getparam,  "sceSaveDataGetParam");
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
    Hle::register_fn("7QZtURYnXG4", (HleFn)s_share_content_param, "sceShareSetContentParam");
    Hle::register_fn("ORspsWDXPps", (HleFn)s_share_ok, "sceShareSetContentParamForApplicationTitle");
    Hle::register_fn("T64o-315wbg", (HleFn)s_share_ok, "sceShareSetScreenshotOverlayImage");
    Hle::register_fn("kvYEw2lBndk", (HleFn)s_live_streaming_init, "sceGameLiveStreamingInitialize");
    Hle::register_fn("9yK6Fk8mKOQ", (HleFn)s_share_ok, "sceGameLiveStreamingTerminate");
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
    Hle::register_fn("lPDO62PpJIA", (HleFn)s_npent_skuflag, "sceNpEntitlementAccessGetSkuFlag");
    Hle::register_fn("WAzWTZm1H+I", (HleFn)s_savedata_transfermount, "sceSaveDataTransferringMount");
    Hle::register_fn("RjMlsR8EXrw", (HleFn)s_savedata_transfermount_ps4, "sceSaveDataTransferringMountPs4");
    Hle::register_fn("X4MYzukPc3g", (HleFn)s_savedata_dirname_search_ps4, "sceSaveDataDirNameSearchPs4");
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
    Hle::register_fn("0HBYxYAjmf0", (HleFn)s_gameintent_term,
                     "sceNpGameIntentTerminate");
    Hle::register_fn("jEIXUAr9XE8", (HleFn)s_gameintent_receive,
                     "sceNpGameIntentReceiveIntent");
    Hle::register_fn("rPl0INNc-M8", (HleFn)s_gameintent_get_property_string,
                     "sceNpGameIntentGetPropertyValueString");
    Hle::register_fn("xddD23+8TfQ", (HleFn)s_npent_addcont_info,
                     "sceNpEntitlementAccessGetAddcontEntitlementInfo");
    #undef R
}

} // namespace prosper
