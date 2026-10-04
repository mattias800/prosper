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

// PS5 Game Intent is how the shell starts a title at a selected activity instead of its default
// entry point. A host frontend can model that user action by setting an activity id before boot;
// without it the console truthfully has no pending intent. Keep the payload opaque to the guest:
// titles consume it through sceNpGameIntentGetPropertyValueString, as on the console.
// CONFIDENCE: HIGH — ABI/error contracts match Kyty; event id, type, key, and offsets are also
// independently exercised by Sonic's eboot and its launchActivity declaration in param.json.
namespace {
}

// --- NP / online: an honest OFFLINE, SIGNED-OUT console (#306). --------------------------------
// The DOLL front-end boot flow stalls at UE4's InstallBundleManager PatchCheck because the Np
// sign-in queries returned success-with-garbage-out: "success" from sceNpGetOnlineId told the
// game a user IS signed in, pushing its patch/entitlement check onto online branches that then
// wait forever on fake Http/WebApi handles (docs/games/DOLL_LOADING_PROGRESSION.md §3). A real console
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
std::atomic<int32_t> g_npweb_context_id{1};
std::atomic<int32_t> g_npweb_user_context_id{1001};
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

// --- app content ---
namespace {
} // namespace

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
}

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

// --- libSceNpTrophy (v1) — the PS4-era trophy API on the same honest console. ------------------
// A title still on the v1 API asks the question NpTrophy2 already answers: no signed-in user, so
// no trophy content. The lifecycle constructors hand back valid local ids so the bring-up
// completes; every content/unlock/list query fails so a caller cannot consume unwritten out-
// structs as trophy data (the #213 class). The v1 facility constant is ORBIS_NP_TROPHY_ERROR_
// USER_NOT_LOGGED_IN (0x8055161D), verified against shadPS4 np_error.h — a PS4-inherited
// facility whose export names and NIDs are identical on the PS5 3.20 stub table. CONFIDENCE: HIGH
// that failure beats success+garbage-out; MED on the exact PS5 value (PS4 value, same 0x8055 Np
// facility and API).
static constexpr uint64_t NP_TROPHY_V1_USER_NOT_LOGGED_IN = 0x8055161Dull;
HLE(s_nptrophy_v1_createctx) {
    svc_log("sceNpTrophyCreateContext", a0, a1, a2, a3, a4, a5);
    if (a0) *(int32_t*)PW(a0) = 1;
    return 0;
}
HLE(s_nptrophy_v1_createhandle) {
    svc_log("sceNpTrophyCreateHandle", a0, a1, a2, a3, a4, a5);
    if (a0) *(int32_t*)PW(a0) = 1;
    return 0;
}
HLE(s_nptrophy_v1_unavailable) {
    return NP_TROPHY_V1_USER_NOT_LOGGED_IN;
}

// --- libSceNpTus / libSceNpScore — PSN data transport on a signed-out console. ----------------
// TUS moves save data between devices; Score moves score-board data. Neither reaches the PSN
// data path with no signed-in user, so every data operation answers the NP-core SIGNED_OUT that
// NpManager gives for the same question — a title must not hear "signed out" from one library
// and "data served" from another. The context/request constructors stay valid (small local ids,
// no identity fabricated, the NpWebApi2/NpTrophy2 contract). The TUS/Score per-facility error
// spaces are not in the 3.20 dump, so the verified NP-core value is used rather than an
// invented facility value; out-params of the refused calls are left untouched. CONFIDENCE: HIGH
// on the sign and the untouched out-params; MED on the exact constant.
HLE(s_np_ctx) {
    svc_log("np ctx/request create", a0, a1, a2, a3, a4, a5);
    if (a0) *(int32_t*)PW(a0) = 1;
    return 0;
}
HLE(s_np_offline) {
    return NP_ERR_SIGNED_OUT;
}

// --- libSceNpCommerce — the PS Store dialog, headless lifecycle like the sign-in prompt (#3784).
// Initialize -> INITIALIZED; Open auto-dismisses to FINISHED (no interactive UI headless, so the
// game's "wait until dismissed" loop still exits); Close -> FINISHED; Terminate -> NONE;
// GetStatus/UpdateStatus report the state. GetResult fails SIGNED_OUT: no signed-in user can
// complete a store transaction, and success with an unwritten out-struct would be the #306
// wedge. Status values follow the shared dialog convention (NONE=0, INITIALIZED=1, RUNNING=2,
// FINISHED=3, see the sign-in dialog above). CONFIDENCE: MED.
namespace {
std::atomic<int32_t> g_commerce_dialog_status{0 /*NONE*/};
}
HLE(s_commerce_init) {
    svc_log("sceNpCommerceDialogInitialize", a0, a1, a2, a3, a4, a5);
    g_commerce_dialog_status.store(1 /*INITIALIZED*/);
    return 0;
}
HLE(s_commerce_open) {
    svc_log("sceNpCommerceDialogOpen", a0, a1, a2, a3, a4, a5);
    g_commerce_dialog_status.store(3 /*FINISHED*/);
    return 0;
}
HLE(s_commerce_status) {
    return (uint64_t)(unsigned)g_commerce_dialog_status.load();
}
HLE(s_commerce_close) {
    g_commerce_dialog_status.store(3 /*FINISHED*/);
    return 0;
}
HLE(s_commerce_term) {
    g_commerce_dialog_status.store(0 /*NONE*/);
    return 0;
}

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


// --- libSceNpUniversalDataSystem event property objects and arrays (#4275). -------------------------
// Silksong builds activity telemetry from standalone property objects and arrays. Live shapes
// (PROSPER_SVCLOG-equivalent probe, PPSA12544):
//   CreateEventPropertyObject(Object** out)     CreateEventPropertyArray(Array** out)
//   EventPropertyArraySetString(Array*, const char* value)
//   EventPropertyObjectSetArray(Object*, const char* key, Array*)
//   DestroyEventPropertyObject(Object*)         DestroyEventPropertyArray(Array*)
// Unregistered, the Create calls returned SCE_OK without writing *out, and the guest went on to
// set, attach and destroy whatever stale value that slot already held (Destroy received exactly the
// out-slot's previous contents). Each handle is a real zeroed host block -- in case a title reads
// through it -- recorded with its kind, so set/attach/destroy can be validated against what was
// actually created. Nothing is recorded or transmitted: an offline console keeps no telemetry.
// CONFIDENCE: HIGH for the argument shapes (live capture), MED for the invalid-argument code
// (shared with the rest of this library's handlers).
namespace {
// EventObject: a property object handed out by sceNpUniversalDataSystemCreateEvent. It is an object
// to the setters, but the event owns it, so DestroyEventPropertyObject refuses it and it is never
// freed here. Those handles come from a counter, not an allocation; the record of them is bounded
// (oldest dropped first) because a title may create events for the whole session.
enum class UdsKind : uint8_t { Object, Array, EventObject };
std::mutex g_uds_mx;
std::unordered_map<uint64_t, UdsKind>& uds_live() {
    static std::unordered_map<uint64_t, UdsKind> live;
    return live;
}
constexpr uint64_t kUdsInvalidArgument = 0x80550003ull;   // SCE_NP_ERROR_INVALID_ARGUMENT
constexpr size_t kUdsHandleBytes = 256;
uint64_t uds_create(uint64_t out, UdsKind kind) {
    if (!svc_ptrish(out)) return kUdsInvalidArgument;
    void* block = std::calloc(1, kUdsHandleBytes);
    if (!block) return 0x80550001ull;   // SCE_NP_ERROR_OUT_OF_MEMORY
    const uint64_t handle = (uint64_t)(uintptr_t)block;
    {
        std::lock_guard lock(g_uds_mx);
        uds_live()[handle] = kind;
    }
    *(uint64_t*)PW(out) = handle;
    return 0;
}
bool uds_is(uint64_t handle, UdsKind kind) {
    std::lock_guard lock(g_uds_mx);
    const auto it = uds_live().find(handle);
    if (it == uds_live().end()) return false;
    return it->second == kind || (kind == UdsKind::Object && it->second == UdsKind::EventObject);
}
constexpr size_t kUdsEventObjectRecord = 1024;
void uds_note_event_object(uint64_t handle) {
    static std::deque<uint64_t> order;
    std::lock_guard lock(g_uds_mx);
    uds_live()[handle] = UdsKind::EventObject;
    order.push_back(handle);
    if (order.size() > kUdsEventObjectRecord) {
        const auto it = uds_live().find(order.front());
        if (it != uds_live().end() && it->second == UdsKind::EventObject) uds_live().erase(it);
        order.pop_front();
    }
}
uint64_t uds_destroy(uint64_t handle, UdsKind kind) {
    {
        std::lock_guard lock(g_uds_mx);
        const auto it = uds_live().find(handle);
        if (it == uds_live().end() || it->second != kind) return kUdsInvalidArgument;
        uds_live().erase(it);
    }
    std::free(PW(handle));
    return 0;
}
}  // namespace
HLE(s_npuds_create_property_object) {
    svc_log("sceNpUniversalDataSystemCreateEventPropertyObject", a0,a1,a2,a3,a4,a5);
    return uds_create(a0, UdsKind::Object);
}
HLE(s_npuds_create_property_array) {
    svc_log("sceNpUniversalDataSystemCreateEventPropertyArray", a0,a1,a2,a3,a4,a5);
    return uds_create(a0, UdsKind::Array);
}
HLE(s_npuds_array_set_string) {
    svc_log("sceNpUniversalDataSystemEventPropertyArraySetString", a0,a1,a2,a3,a4,a5);
    if (!uds_is(a0, UdsKind::Array) || !svc_ptrish(a1)) return kUdsInvalidArgument;
    return 0;
}
HLE(s_npuds_object_set_array) {
    svc_log("sceNpUniversalDataSystemEventPropertyObjectSetArray", a0,a1,a2,a3,a4,a5);
    if (!uds_is(a0, UdsKind::Object) || !svc_ptrish(a1) || !uds_is(a2, UdsKind::Array))
        return kUdsInvalidArgument;
    return 0;
}
HLE(s_npuds_destroy_property_object) {
    svc_log("sceNpUniversalDataSystemDestroyEventPropertyObject", a0,a1,a2,a3,a4,a5);
    return uds_destroy(a0, UdsKind::Object);
}
HLE(s_npuds_destroy_property_array) {
    svc_log("sceNpUniversalDataSystemDestroyEventPropertyArray", a0,a1,a2,a3,a4,a5);
    return uds_destroy(a0, UdsKind::Array);
}

// sceNpSessionSignalingInitialize(const InitParam*): Silksong passes an input-only parameter block
// (a version word, a size and thread attributes) and reads no out-parameter. Signaling is the
// peer-connection layer of online sessions; initialising it needs no network, so an offline console
// succeeds here and fails later, at the first call that would actually reach a peer. A null block is
// refused. CONFIDENCE: MED (live call shape; no reference for the error value beyond the library's
// family code).
HLE(s_np_session_signaling_initialize) {
    svc_log("sceNpSessionSignalingInitialize", a0,a1,a2,a3,a4,a5);
    return svc_ptrish(a0) ? 0 : kUdsInvalidArgument;
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
    // The property object is recorded so EventPropertyObjectSetArray accepts it (#4275 review):
    // Metaphor attaches arrays only to objects that came from here.
    uint64_t properties = 0;
    if (svc_ptrish(a3))                                *(uint64_t*)PW(a3) = properties = g_handle.fetch_add(1);
    else if (a3 == 0 && a5 == 0 && svc_ptrish(a4))     *(uint64_t*)PW(a4) = properties = g_handle.fetch_add(1);
    if (properties) uds_note_event_object(properties);
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

// ===== Issue #306: honest OFFLINE console for the online/update/entitlement boot chain. =========
// DOLL's UE4 front-end runs a patch/entitlement check before the title screen; every subsystem in
// that chain answered success-with-garbage, so the check could neither succeed nor FAIL — the flow
// waited forever (docs/games/DOLL_LOADING_PROGRESSION.md). The blocks below give the chain the answers a
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
}


// Registration for this library, called by register_builtin_hle(). One entry point per
// Sony library keeps every handler `static` to the translation unit implementing it (#3735).
// This file's own trivial success stub, so the NP registrations below do not reach for
// hle_service.cpp's. `HLE()` makes it `static`; a per-translation-unit copy is the shape every
// other split library uses (#3735).
HLE(s_np_ok) { return 0; }

// --- libSceSigninDialog: the system "sign in to PlayStation Network" dialog (#3784). -------------
// A signed-out console (above) is what makes a title raise this dialog, and on real hardware the
// user can decline it; the title then continues offline. The honest headless answer is therefore
// the dialog's normal lifecycle ending with the user having closed it: Initialize -> INITIALIZED,
// Open -> (the system UI runs and is dismissed) -> FINISHED, Terminate -> NONE. The status enum is
// the shared SceCommonDialogStatus (NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3), and
// UpdateStatus/GetStatus RETURN it rather than writing an out-parameter.
//
// Evidence, all from guest code (the 3.20 stub table gives names and NIDs only):
//   * Metaphor: ReFantazio polls `sceSigninDialogUpdateStatus` at eboot+0x863e89, compares the
//     RETURNED value with 3 (`cmp ecx,3`) and only then calls sceSigninDialogTerminate and its own
//     completion callback. Unregistered, the dispatcher's `return 0` read as NONE forever, so the
//     title sat on a black screen after its SYSTEM settings page with the guest still running.
//     Its Open argument is a 16-byte block built on the stack: { u32 size = 0x10; s32 userId;
//     u64 reserved = 0 } (eboot+0x862cae..0x862cbd).
//   * A census of every local dump with `tools/re/nid_gate_scan.py --nid Bw31liTFT3A` finds ten
//     titles importing UpdateStatus, and every classified call site compares the return with a
//     constant: `cmp eax,3` in Grand Theft Auto V and in PPSA21564 (twice), `cmp r14d,2` (RUNNING)
//     in Sonic Frontiers. No site treats the value as an error code.
// Metaphor imports only Initialize/Open/UpdateStatus/Terminate. GetResult stays UNREGISTERED on
// purpose: its result struct is not pinned by any local call site, and writing a guessed layout
// would be worse than the fail-visible unimplemented line. Close and GetStatus share the lifecycle
// above and need no layout.
// CONFIDENCE: HIGH on the lifecycle and the returned-status convention (guest compare sites in
// three independent titles); MED that no error returns are needed for out-of-order calls, which no
// local title exercises.
namespace {
std::atomic<int>& signin_dialog_status() {
    static std::atomic<int> status{0 /*NONE*/};
    return status;
}
}
HLE(s_signin_dialog_init) {
    signin_dialog_status().store(1 /*INITIALIZED*/);
    return 0;
}
HLE(s_signin_dialog_open) {
    int32_t user = 0;
    if (svc_ptrish(a0)) user = *(const int32_t*)((const char*)PW(a0) + 4);
    fprintf(stderr, "[svc] sceSigninDialogOpen(userId=%d) -> headless: dismissed, FINISHED\n", user);
    signin_dialog_status().store(3 /*FINISHED*/);
    return 0;
}
HLE(s_signin_dialog_status) {
    return (uint64_t)(unsigned)signin_dialog_status().load();
}
HLE(s_signin_dialog_close) {
    signin_dialog_status().store(3 /*FINISHED*/);
    return 0;
}
HLE(s_signin_dialog_term) {
    signin_dialog_status().store(0 /*NONE*/);
    return 0;
}

void register_np_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    // libSceSigninDialog (#3784); NIDs from the PS5 3.20 libSceSigninDialog stub table.
    Hle::register_fn("mlYGfmqE3fQ", (HleFn)s_signin_dialog_init,   "sceSigninDialogInitialize");
    Hle::register_fn("JlpJVoRWv7U", (HleFn)s_signin_dialog_open,   "sceSigninDialogOpen");
    Hle::register_fn("Bw31liTFT3A", (HleFn)s_signin_dialog_status, "sceSigninDialogUpdateStatus");
    Hle::register_fn("2m077aeC+PA", (HleFn)s_signin_dialog_status, "sceSigninDialogGetStatus");
    Hle::register_fn("M3OkENHcyiU", (HleFn)s_signin_dialog_close,  "sceSigninDialogClose");
    Hle::register_fn("LXlmS6PvJdU", (HleFn)s_signin_dialog_term,   "sceSigninDialogTerminate");
    Hle::register_fn("obuxdTiwkF8", (HleFn)s_netctl_getinfo, "sceNetCtlGetInfo");  // NOT_CONNECTED
    Hle::register_fn("0cBgduPRR+M", (HleFn)s_netctl_getresult, "sceNetCtlGetResult");
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
    R("sceNpRegisterStateCallback", s_np_ok);
    // libSceNpTrophy2 lifecycle — valid ids; content queries stay "unavailable" (above).
    Hle::register_fn("Bagshr7OQ6Q", (HleFn)s_nptrophy2_createctx,    "sceNpTrophy2CreateContext");
    Hle::register_fn("Gz1rmUZpROM", (HleFn)s_nptrophy2_createhandle, "sceNpTrophy2CreateHandle");
    Hle::register_fn("bIDov3wBu5Q", (HleFn)s_nptrophy2_regctx,       "sceNpTrophy2RegisterContext");
    Hle::register_fn("sUXGfNMalIo", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2RegisterUnlockCallback");
    Hle::register_fn("sysY2FHYff4", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2DestroyContext");
    Hle::register_fn("d8P11CI40KE", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2DestroyHandle");
    Hle::register_fn("fYapWA9xVmA", (HleFn)s_nptrophy2_ok,           "sceNpTrophy2AbortHandle");
    // libSceNpTrophy (v1) — PS4-era trophy API: valid lifecycle ids, no content (see block above).
    // NIDs from the PS5 3.20 libSceNpTrophy stub table.
    Hle::register_fn("HbkjbobZlCY", (HleFn)s_nptrophy_v1_createctx, "sceNpTrophyCreateContext");
    Hle::register_fn("K7U6tEAQf7c", (HleFn)s_nptrophy_v1_createhandle, "sceNpTrophyCreateHandle");
    Hle::register_fn("OdPIOFpEAvU", (HleFn)s_nptrophy_v1_createhandle,
                     "sceNpTrophyIntCreateHandle");
    Hle::register_fn("DJCAxto9SEU", (HleFn)s_nptrophy2_regctx, "sceNpTrophyRegisterContext");
    Hle::register_fn("KTnHs7W-9Uk", (HleFn)s_nptrophy2_ok, "sceNpTrophyAbortHandle");
    Hle::register_fn("O9plkqa2e0k", (HleFn)s_nptrophy2_ok, "sceNpTrophyIntAbortHandle");
    Hle::register_fn("NV18n8OcheI", (HleFn)s_nptrophy2_ok, "sceNpTrophySystemRemoveAll");
    Hle::register_fn("FiPpytPUPMA", (HleFn)s_nptrophy2_ok,
                     "sceNpTrophySystemUnregisterTitleSyncedCallback");
    Hle::register_fn("BBkBINKo6gw", (HleFn)s_nptrophy2_ok,
                     "sceNpTrophySystemUnregisterTitleUpdateCallback");
    Hle::register_fn("Hit5yM0Teo0", (HleFn)s_nptrophy2_ok, "sceNpTrophySystemWrapDebugLockTrophy");
    Hle::register_fn("I5R5Ogfbk68", (HleFn)s_nptrophy2_ok,
                     "sceNpTrophySystemWrapDebugUnlockTrophy");
    Hle::register_fn("MNbxxwNdlHY", (HleFn)s_nptrophy2_ok, "sceNpTrophySystemWrapRemoveUserData");
    Hle::register_fn("IYP3f2W09og", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophyGetGameInfo");
    Hle::register_fn("ATUwGfspKic", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophyGetGroupInfo");
    Hle::register_fn("KqUVGDgQBm0", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophyGetTrophyInfo");
    Hle::register_fn("A4uMPmErD4I", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophyGetGroupIcon");
    Hle::register_fn("OBL+l6HG9xk", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophyGetTrophyIcon");
    Hle::register_fn("BvdThnVvwdY", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophyNumInfoGetTotal");
    Hle::register_fn("G8xmRUFao68", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophyUnlockTrophy");
    Hle::register_fn("N9jpdPz5f-8", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophyShowTrophyList");
    Hle::register_fn("N3CQzag7-zs", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophyIntGetProgress");
    Hle::register_fn("EF9zjnlAzIA", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophyIntNetSyncTitle");
    Hle::register_fn("EXiyfabxFNQ", (HleFn)s_nptrophy_v1_unavailable,
                     "sceNpTrophyIntNetSyncTitles");
    Hle::register_fn("MaLlLHKP+No", (HleFn)s_nptrophy_v1_unavailable, "sceNpTrophySystemDbgCtl");
    Hle::register_fn("GrV7Y4IhWkc", (HleFn)s_nptrophy_v1_unavailable,
                     "sceNpTrophySystemWrapGetGroupDetails");
    Hle::register_fn("AJD0VSnMfW0", (HleFn)s_nptrophy_v1_unavailable,
                     "sceNpTrophySystemWrapGetPlayedTrophyTitles");
    Hle::register_fn("CSh2GsVQTzs", (HleFn)s_nptrophy_v1_unavailable,
                     "sceNpTrophySystemWrapGetTitleDetails");
    Hle::register_fn("PtaF5aJl7k0", (HleFn)s_nptrophy_v1_unavailable,
                     "sceNpTrophySystemWrapGetTrophyDetailsArray");
    Hle::register_fn("BMmIU5R9IHY", (HleFn)s_nptrophy_v1_unavailable,
                     "sceNpTrophySystemWrapGetTrophyTitleIdsByNpTitleId");
    // libSceNpTus / libSceNpScore — valid local ids, no PSN data (SIGNED_OUT; see block above).
    Hle::register_fn("Bhy8+oecGac", (HleFn)s_np_ctx, "sceNpTusCreateTitleCtx");
    Hle::register_fn("Fn-dGukBgnY", (HleFn)s_np_ctx, "sceNpTusCreateNpTitleCtxA");
    Hle::register_fn("MRVb2Cf0GHg", (HleFn)s_np_ctx, "sceNpTssCreateNpTitleCtx");
    Hle::register_fn("FBtrk+7lk14", (HleFn)s_np_ctx, "sceNpTssCreateNpTitleCtxA");
    Hle::register_fn("Hbh2aBvvmvM", (HleFn)s_np_ctx, "sceNpTusCreateRequest");
    Hle::register_fn("KW9M0bQ-Zx0", (HleFn)s_np_ctx, "sceNpScoreCreateTitleCtx");
    Hle::register_fn("AW8qyjYrUbk", (HleFn)s_np_ctx, "sceNpScoreCreateRequest");
    Hle::register_fn("Geq1bMwgZYo", (HleFn)s_np_ok, "sceNpTusAbortRequest");
    Hle::register_fn("KGKDdRCFx8c", (HleFn)s_np_ok, "sceNpTusSetThreadParam");
    Hle::register_fn("Fi7kmKbX6hk", (HleFn)s_np_ok, "sceNpScoreAbortRequest");
    Hle::register_fn("NK8-SgYf6r4", (HleFn)s_np_ok, "sceNpScoreDeleteRequest");
    Hle::register_fn("CxK68584JAU", (HleFn)s_np_ok, "sceNpScoreSetThreadParam");
    Hle::register_fn("C3xZj35v8Z8", (HleFn)s_np_ok, "sceNpScoreSetTimeout");
    // The data path: TUS multi-slot/friends/multi-user variables and cross-save, TUS+TSS
    // get/set/try/delete, Score record/get/ranking/censor, and the async completion calls that
    // would deliver PSN data.
    Hle::register_fn("MrVmNrJDbG8", (HleFn)s_np_offline, "sceNpTusAddAndGetVariable");
    Hle::register_fn("APFah4-5Xec", (HleFn)s_np_offline, "sceNpTusAddAndGetVariableA");
    Hle::register_fn("GdB427dT3Iw", (HleFn)s_np_offline, "sceNpTusAddAndGetVariableAAsync");
    Hle::register_fn("A2UmHdK04c8", (HleFn)s_np_offline, "sceNpTusAddAndGetVariableAsync");
    Hle::register_fn("Okr6FBSrkJw", (HleFn)s_np_offline, "sceNpTusAddAndGetVariableVUser");
    Hle::register_fn("EDT5bP6YzBo", (HleFn)s_np_offline, "sceNpTusDeleteMultiSlotData");
    Hle::register_fn("CXzUOM9sXU0", (HleFn)s_np_offline, "sceNpTusDeleteMultiSlotDataA");
    Hle::register_fn("K-+Yqc-NppQ", (HleFn)s_np_offline, "sceNpTusDeleteMultiSlotDataAAsync");
    Hle::register_fn("ButwCvsydkk", (HleFn)s_np_offline, "sceNpTusDeleteMultiSlotDataVUser");
    Hle::register_fn("GYhbiRtkE1Y", (HleFn)s_np_offline, "sceNpTusDeleteMultiSlotVariable");
    Hle::register_fn("JwnE9Oa1uF8", (HleFn)s_np_offline, "sceNpTusDeleteMultiSlotVariableA");
    Hle::register_fn("HOzszO4ONWU", (HleFn)s_np_offline, "sceNpTusGetData");
    Hle::register_fn("CWEHUFkY1qI", (HleFn)s_np_offline, "sceNpTusGetDataA");
    Hle::register_fn("BzG8mG9YlKY", (HleFn)s_np_offline, "sceNpTusGetDataAAsync");
    Hle::register_fn("CaH+Sxlw32k", (HleFn)s_np_offline, "sceNpTusGetDataAVUser");
    Hle::register_fn("OoFvgzwawAY", (HleFn)s_np_offline, "sceNpTusGetDataAVUserAsync");
    Hle::register_fn("OHtKS5V1T5k", (HleFn)s_np_offline, "sceNpTusGetDataAsync");
    Hle::register_fn("FTE3OvH61qo", (HleFn)s_np_offline, "sceNpTusGetDataForCrossSave");
    Hle::register_fn("PLxFGYCJwww", (HleFn)s_np_offline, "sceNpTusGetDataForCrossSaveVUser");
    Hle::register_fn("JR6kI-8f+Hk", (HleFn)s_np_offline, "sceNpTusGetDataVUserAsync");
    Hle::register_fn("Cixh7HDKWfk", (HleFn)s_np_offline, "sceNpTusGetFriendsDataStatusA");
    Hle::register_fn("My+pAALkHp8", (HleFn)s_np_offline, "sceNpTusGetFriendsVariable");
    Hle::register_fn("ArImtTqUSGM", (HleFn)s_np_offline, "sceNpTusGetFriendsVariableAAsync");
    Hle::register_fn("IFYWOwYI6DY", (HleFn)s_np_offline, "sceNpTusGetFriendsVariableAsync");
    Hle::register_fn("JgcNwFHoOL4", (HleFn)s_np_offline, "sceNpTusGetMultiSlotDataStatus");
    Hle::register_fn("M33Y2TnyonE", (HleFn)s_np_offline, "sceNpTusGetMultiSlotDataStatusA");
    Hle::register_fn("LcPB2rnhQqo", (HleFn)s_np_offline, "sceNpTusGetMultiSlotVariableAsync");
    Hle::register_fn("OFxVYJEkcmc", (HleFn)s_np_offline, "sceNpTusGetMultiSlotVariableVUser");
    Hle::register_fn("FxNDPDnWfMc", (HleFn)s_np_offline, "sceNpTusGetMultiUserDataStatusA");
    Hle::register_fn("KG9+4eIb+cY", (HleFn)s_np_offline, "sceNpTusGetMultiUserVariable");
    Hle::register_fn("IRje5yEXS0U", (HleFn)s_np_offline, "sceNpTusGetMultiUserVariableAsync");
    Hle::register_fn("DB0vaHTzA6g", (HleFn)s_np_offline, "sceNpTusGetMultiUserVariableVUser");
    Hle::register_fn("N7b6dmpQNiI", (HleFn)s_np_offline, "sceNpTusPollAsync");
    Hle::register_fn("INrufkNCkiE", (HleFn)s_np_offline, "sceNpTusSetData");
    Hle::register_fn("FzxN3tOouj8", (HleFn)s_np_offline, "sceNpTusSetDataA");
    Hle::register_fn("Iu58d6g6uwU", (HleFn)s_np_offline, "sceNpTusSetDataAAsync");
    Hle::register_fn("EbWqOt3QjKU", (HleFn)s_np_offline, "sceNpTusSetDataAVUser");
    Hle::register_fn("ORhzSuuXwxo", (HleFn)s_np_offline, "sceNpTusSetDataVUser");
    Hle::register_fn("M6aYoa47YgI", (HleFn)s_np_offline, "sceNpTusSetMultiSlotVariable");
    Hle::register_fn("Mf-WMA0jYCc", (HleFn)s_np_offline, "sceNpTusSetMultiSlotVariableA");
    Hle::register_fn("JJ9GGMludxY", (HleFn)s_np_offline, "sceNpTusSetMultiSlotVariableAsync");
    Hle::register_fn("FCz0hTJFyh4", (HleFn)s_np_offline, "sceNpTusSetMultiSlotVariableVUser");
    Hle::register_fn("OkC55HsotJ4", (HleFn)s_np_offline, "sceNpTusTryAndSetVariable");
    Hle::register_fn("Eup4MP1wNtc", (HleFn)s_np_offline, "sceNpTusTryAndSetVariableA");
    Hle::register_fn("LGTjTkHPHTE", (HleFn)s_np_offline, "sceNpTusTryAndSetVariableAAsync");
    Hle::register_fn("IGIcxlUabSA", (HleFn)s_np_offline, "sceNpTusTryAndSetVariableAVUser");
    Hle::register_fn("BQfR51i4kck", (HleFn)s_np_offline, "sceNpTusTryAndSetVariableAsync");
    Hle::register_fn("JbitD262GhY", (HleFn)s_np_offline, "sceNpTusTryAndSetVariableVUser");
    Hle::register_fn("BYPJFWzFPjA", (HleFn)s_np_offline, "sceNpTusWaitAsync");
    Hle::register_fn("PSUR+UoLS6c", (HleFn)s_np_offline, "sceNpTssGetData");
    Hle::register_fn("FL+Z3zCKNTs", (HleFn)s_np_offline, "sceNpTssGetSmallStorage");
    Hle::register_fn("P2Pe4LGS2II", (HleFn)s_np_offline, "sceNpTssGetSmallStorageAsync");
    Hle::register_fn("E5NZIzggbuk", (HleFn)s_np_offline, "sceNpTssGetStorageAsync");
    Hle::register_fn("Gb3TI0mDYiI", (HleFn)s_np_offline, "sceNpScoreCensorComment");
    Hle::register_fn("IeOvDyN-aZc", (HleFn)s_np_offline, "sceNpScoreCensorCommentAsync");
    Hle::register_fn("A0Avi9kebsY", (HleFn)s_np_offline, "sceNpScoreGetBoardInfoAsync");
    Hle::register_fn("MkuIzUw6utQ", (HleFn)s_np_offline, "sceNpScoreGetFriendsRanking");
    Hle::register_fn("AMbOn+-6eXA", (HleFn)s_np_offline, "sceNpScoreGetFriendsRankingA");
    Hle::register_fn("DkoVok6FFEI", (HleFn)s_np_offline, "sceNpScoreGetGameData");
    Hle::register_fn("NmZEgoiEq6Y", (HleFn)s_np_offline, "sceNpScoreGetRankingByNpId");
    Hle::register_fn("G1DfNRstkSQ", (HleFn)s_np_offline, "sceNpScorePollAsync");
    Hle::register_fn("LcoVwcBjQ9E", (HleFn)s_np_offline, "sceNpScoreRecordGameData");
    Hle::register_fn("FgL5PwYzrrw", (HleFn)s_np_offline, "sceNpScoreRecordGameDataAsync");
    Hle::register_fn("DT0XBtgtOSI", (HleFn)s_np_offline, "sceNpScoreRecordScore");
    Hle::register_fn("L4oAo9in0TA", (HleFn)s_np_offline, "sceNpScoreSanitizeComment");
    Hle::register_fn("Pqk8SC63p1U", (HleFn)s_np_offline, "sceNpScoreWaitAsync");
    // libSceNpCommerce — headless PS Store dialog + local icon controls (see block above).
    Hle::register_fn("0aR2aWmQal4", (HleFn)s_commerce_init, "sceNpCommerceDialogInitialize");
    Hle::register_fn("DfSCDRA3EjY", (HleFn)s_commerce_open, "sceNpCommerceDialogOpen");
    Hle::register_fn("IXmfUaze9So", (HleFn)s_commerce_open, "sceNpCommerceDialogOpen2");
    Hle::register_fn("CCbC+lqqvF0", (HleFn)s_commerce_status, "sceNpCommerceDialogGetStatus");
    Hle::register_fn("LR5cwFMMCVE", (HleFn)s_commerce_status, "sceNpCommerceDialogUpdateStatus");
    Hle::register_fn("r42bWcQbtZY", (HleFn)s_np_offline, "sceNpCommerceDialogGetResult");
    Hle::register_fn("NU3ckGHMFXo", (HleFn)s_commerce_close, "sceNpCommerceDialogClose");
    Hle::register_fn("m-I92Ab50W8", (HleFn)s_commerce_term, "sceNpCommerceDialogTerminate");
    Hle::register_fn("DHmwsa6S8Tc", (HleFn)s_np_ok, "sceNpCommerceShowPsStoreIcon");
    Hle::register_fn("dsqCVsNM0Zg", (HleFn)s_np_ok, "sceNpCommerceHidePsStoreIcon");
    Hle::register_fn("uKTDW8hk-ts", (HleFn)s_np_ok, "sceNpCommerceSetPsStoreIconLayout");
    // libSceNpAuth — OAuth request lifecycle: valid local request ids, no authorization codes
    // (SIGNED_OUT; the PS4 facility in shadPS4 np_auth.cpp answers the getters exactly this way
    // when no user is signed in). NIDs from the PS5 3.20 libSceNpAuth stub table.
    Hle::register_fn("6bwFkosYRQg", (HleFn)s_np_ctx, "sceNpAuthCreateRequest");
    Hle::register_fn("N+mr7GjTvr8", (HleFn)s_np_ctx, "sceNpAuthCreateAsyncRequest");
    Hle::register_fn("cE7wIsqXdZ8", (HleFn)s_np_ok, "sceNpAuthAbortRequest");
    Hle::register_fn("H8wG9Bk-nPc", (HleFn)s_np_ok, "sceNpAuthDeleteRequest");
    Hle::register_fn("PM3IZCw-7m0", (HleFn)s_np_ok, "sceNpAuthSetTimeout");
    Hle::register_fn("gjSyfzSsDcE", (HleFn)s_np_offline, "sceNpAuthPollAsync");
    Hle::register_fn("SK-S7daqJSE", (HleFn)s_np_offline, "sceNpAuthWaitAsync");
    Hle::register_fn("KAUXQ9GdWp8", (HleFn)s_np_offline, "sceNpAuthGetAuthorizationCodeA");
    Hle::register_fn("KI4dHLlTNl0", (HleFn)s_np_offline, "sceNpAuthGetAuthorizationCodeV3");
    Hle::register_fn("IDX0S5EsEh4", (HleFn)s_np_offline, "sceNpAuthGetAuthorizedAppCode");
    Hle::register_fn("OaB-LoJqHis", (HleFn)s_np_offline, "sceNpAuthGetIdToken");
    Hle::register_fn("RdsFVsgSpZY", (HleFn)s_np_offline, "sceNpAuthGetIdTokenV3");
    // libSceNpUtility — init succeeds; the PSN-backed lookup/bandwidth/word-filter paths are
    // SIGNED_OUT; the lookup request/title-ctx constructors keep their local ids.
    Hle::register_fn("G6iWw8aUQtA", (HleFn)s_np_ok, "sceNpUtilityInit");
    Hle::register_fn("F6Dl+2zlua0", (HleFn)s_np_ok, "sceNpAppInfoIntInitialize");
    Hle::register_fn("M9+zoKE8cBA", (HleFn)s_np_ok, "sceNpAppInfoIntFinalize");
    Hle::register_fn("pLr1fEQS1z8", (HleFn)s_np_ok, "sceNpBandwidthTestShutdown");
    Hle::register_fn("kvdMF48mB3Y", (HleFn)s_np_ok, "sceNpBandwidthTestAbort");
    Hle::register_fn("hqzi1IHdQQQ", (HleFn)s_np_offline, "sceNpBandwidthTestInitStartDownload");
    Hle::register_fn("mA0zsbqm+kA", (HleFn)s_np_offline, "sceNpBandwidthTestInitStartUpload");
    Hle::register_fn("BYIZGKm6bO4", (HleFn)s_np_offline, "sceNpBandwidthTestGetStatus");
    Hle::register_fn("DVZE+fAhgFY", (HleFn)s_np_offline, "sceNpLookupNetInit");
    Hle::register_fn("JXlTj9RRCFo", (HleFn)s_np_ok,
                     "sceNpLookupNetIsInit");   // 0 = not initialized (honest: it is not)
    Hle::register_fn("CQr9UxPHUFs", (HleFn)s_np_ctx, "sceNpLookupCreateRequest");
    Hle::register_fn("M533Q+LU7EQ", (HleFn)s_np_ctx, "sceNpLookupCreateTitleCtx");
    Hle::register_fn("ALaxchvEEnk", (HleFn)s_np_ok, "sceNpLookupDeleteRequest");
    Hle::register_fn("GtqDK9zkoIE", (HleFn)s_np_ok, "sceNpLookupDeleteTitleCtx");
    Hle::register_fn("OYz4v5Uek9U", (HleFn)s_np_ok, "sceNpLookupAbortRequest");
    Hle::register_fn("EMV72WO7V34", (HleFn)s_np_ok, "sceNpLookupSetTimeout");
    Hle::register_fn("GnEVmFiV6OI", (HleFn)s_np_offline, "sceNpLookupNetNpId");
    Hle::register_fn("D6tnM1Uti4g", (HleFn)s_np_offline, "sceNpLookupNpId");
    Hle::register_fn("F4EVrruHuy8", (HleFn)s_np_offline, "sceNpLookupPollAsync");
    Hle::register_fn("IX9dAus6baE", (HleFn)s_np_offline, "sceNpLookupWaitAsync");
    Hle::register_fn("Or5SShyG0dk", (HleFn)s_np_offline, "sceNpWordFilterPollAsync");
    Hle::register_fn("M7ivWj5yKzg", (HleFn)s_np_offline, "sceNpWordFilterWaitAsync");
    // libSceNpMatching2 — room/matching state is PSN-backed, so every state query and
    // connection establishment is SIGNED_OUT; the pure local parameter setters succeed.
    // CONFIDENCE: HIGH on the sign; the context out-width is unverified, so the constructor is
    // refused rather than written (a wrong-width write would hand the guest a corrupt id).
    Hle::register_fn("HHZpTF30wto", (HleFn)s_np_ok, "sceNpMatching2SetExtraInitParam");
    Hle::register_fn("AupHEf8WOhM", (HleFn)s_np_ok, "sceNpMatching2SignalingSetPort");
    Hle::register_fn("ODxEHb9f7B8", (HleFn)s_np_ok, "sceNpMatching2SignalingAbortConnection");
    Hle::register_fn("Kxlf9+pa0GY", (HleFn)s_np_offline, "sceNpMatching2CreateContextInternal");
    Hle::register_fn("Hddl5xnQQEY", (HleFn)s_np_offline,
                     "sceNpMatching2GetRoomJoinedSlotMaskLocal");
    Hle::register_fn("FagjVl+bHFI", (HleFn)s_np_offline,
                     "sceNpMatching2GetWorldIdArrayForAllServers");
    Hle::register_fn("DMxxNNLh6ms", (HleFn)s_np_offline, "sceNpMatching2SetRoomDataInternalExt");
    Hle::register_fn("EcYuZkNhHI8", (HleFn)s_np_offline,
                     "sceNpMatching2SignalingEstablishConnection");
    Hle::register_fn("GkvclTMjNdI", (HleFn)s_np_offline, "sceNpMatching2SignalingGetPort");
    // libSceNpSns — third-party link (Facebook/Twitch/YouTube) status and tokens are PSN-backed:
    // SIGNED_OUT; the request lifecycle keeps its local ids.
    Hle::register_fn("D+F4GKuY3oE", (HleFn)s_np_ctx, "sceNpSnsIntCreateRequest");
    Hle::register_fn("E2nPI+P0g8o", (HleFn)s_np_ctx, "sceNpSnsTwitchCreateRequest");
    Hle::register_fn("Ho3XDEydBjM", (HleFn)s_np_ctx, "sceNpSnsYouTubeCreateRequest");
    Hle::register_fn("KLqzbBxATrU", (HleFn)s_np_ok, "sceNpSnsIntDeleteRequest");
    Hle::register_fn("OPj-CXHNEFE", (HleFn)s_np_ok, "sceNpSnsIntAbortRequest");
    Hle::register_fn("PLJbF9b-Who", (HleFn)s_np_ok, "sceNpSnsFacebookAbortRequest");
    Hle::register_fn("JTvAKV1iQkE", (HleFn)s_np_ok, "sceNpSnsFacebookDeleteRequest");
    Hle::register_fn("E-sppToVlnc", (HleFn)s_np_ok, "sceNpSnsTwitchAbortRequest");
    Hle::register_fn("DXUeCRu7DLE", (HleFn)s_np_ok, "sceNpSnsYouTubeAbortRequest");
    Hle::register_fn("GGfyLTvE+LI", (HleFn)s_np_ok, "sceNpSnsYouTubeDeleteRequest");
    Hle::register_fn("FtyS8XLBqNE", (HleFn)s_np_offline, "sceNpSnsFacebookGetAccessToken");
    Hle::register_fn("JzCW8dx4mKk", (HleFn)s_np_offline, "sceNpSnsIntFbGetGameAccessToken");
    Hle::register_fn("OINq9QxFYqU", (HleFn)s_np_offline, "sceNpSnsIntFbGetGameAccessTokenAllowed");
    Hle::register_fn("O6K3LE8qXsc", (HleFn)s_np_offline, "sceNpSnsIntFbGetSystemAccessToken");
    Hle::register_fn("Gp+C91igTkE", (HleFn)s_np_offline, "sceNpSnsIntTwGetSystemAccessToken");
    Hle::register_fn("ExrhvJ8QANU", (HleFn)s_np_offline, "sceNpSnsIntYtGetAccessToken");
    Hle::register_fn("A+mSQ2U6wWY", (HleFn)s_np_offline, "sceNpSnsIntYtRefreshMasterToken");
    Hle::register_fn("AW7NonbfeFk", (HleFn)s_np_offline, "sceNpSnsIntLinkedStatus");
    Hle::register_fn("Pz-00XG-VNU", (HleFn)s_np_offline, "sceNpSnsIntUnlink");
    Hle::register_fn("GHLVam6hZZ4", (HleFn)s_np_offline, "sceNpSnsTwitchGetAccessToken");
    Hle::register_fn("FElwHkpLvmw", (HleFn)s_np_offline, "sceNpSnsYouTubeGetAccessToken");
    // libSceShare — succeed; sharing simply unavailable headless.
    Hle::register_fn("nBDD66kiFW8", (HleFn)s_share_ok, "sceShareInitialize");
    Hle::register_fn("0IL1keINExQ", (HleFn)s_share_ok, "sceShareTerminate");
    Hle::register_fn("7QZtURYnXG4", (HleFn)s_share_content_param, "sceShareSetContentParam");
    Hle::register_fn("ORspsWDXPps", (HleFn)s_share_ok, "sceShareSetContentParamForApplicationTitle");
    Hle::register_fn("T64o-315wbg", (HleFn)s_share_ok, "sceShareSetScreenshotOverlayImage");
    Hle::register_fn("9yK6Fk8mKOQ", (HleFn)s_share_ok, "sceGameLiveStreamingTerminate");
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
    Hle::register_fn("ysmw6J-P8Ak", (HleFn)s_np_session_signaling_initialize, "sceNpSessionSignalingInitialize");
    Hle::register_fn("4llLk7YJRTE", (HleFn)s_npuds_array_set_string, "sceNpUniversalDataSystemEventPropertyArraySetString");
    Hle::register_fn("Wxbg5x3pTXA", (HleFn)s_npuds_object_set_array, "sceNpUniversalDataSystemEventPropertyObjectSetArray");
    Hle::register_fn("kKUH0Viib3c", (HleFn)s_npuds_destroy_property_object, "sceNpUniversalDataSystemDestroyEventPropertyObject");
    Hle::register_fn("W-0xwY0ZMjw", (HleFn)s_npuds_destroy_property_array, "sceNpUniversalDataSystemDestroyEventPropertyArray");
    Hle::register_fn("Hm7qubT3b70", (HleFn)s_npuds_create_property_array, "sceNpUniversalDataSystemCreateEventPropertyArray");
    Hle::register_fn("s6W4Zl4Slgk", (HleFn)s_npuds_create_property_object, "sceNpUniversalDataSystemCreateEventPropertyObject");
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
#ifndef _WIN32
    // sceNpCheckCallback pumps the registered A-callbacks (SIGNED_OUT delivered once, guest %fs).
    Hle::register_fn("3Zl8BePTh9Y", (HleFn)s_np_check_cb_entry,      "sceNpCheckCallback");
    Hle::register_fn("qQJfO8HAiaY", (HleFn)s_np_register_state_cbA,  "sceNpRegisterStateCallbackA");
    Hle::register_fn("M3wFXbYQtAA", (HleFn)s_np_ok,                     "sceNpUnregisterStateCallbackA");
#else
    R("sceNpCheckCallback", s_np_ok);
#endif
    #undef R
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

} // namespace

} // namespace prosper
