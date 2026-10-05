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
// --- libSceNpWebApi2 request + push-event surface (22 of the library's 39 exports) -------------
// The PSN web API has no service behind it on this signed-out console, so handles are created
// locally but every call that would reach PSN answers SCE_NP_ERROR_SIGNED_OUT -- the same answer
// NpManager and the NP sub-libraries give (#4417), so no library tells the guest a different
// story. Arity and out-param placement are confirmed from guest call sites (The Messenger,
// Alex Kidd: CreateRequest writes its id through a5; Worms PPSA20052: the push-event
// constructors RETURN their ids and a negative return is the failure). CONFIDENCE: HIGH on that
// lifecycle shape. The INVALID_ARGUMENT value 0x80553402 is not in the 3.20 stub dump (it holds
// no error constants); it comes from a secondary implementation, CONFIDENCE: MED.
//   Create*/PushEventCreate*/RegisterCallback*  -> local id (positive, never 0)
//   Abort/Delete/Terminate/Unregister           -> 0 (nothing to tear down offline)
//   Send/ReadData/HeaderValue/HeaderLength      -> SIGNED_OUT, out-params untouched
//   CheckTimeout (void)                         -> 0 (no timeouts pending offline)
//   CreatePushContext / StartPushContextCallback -> SIGNED_OUT (no push service to host or
//                                                   start), out-params untouched
namespace {
constexpr uint64_t kWebApiErrInvalidArgument = 0x80553402ull;
std::atomic<int64_t> g_npweb_request_id{1};
std::atomic<int32_t> g_npweb_push_id{1};
// Positive ids only: a negative return is the guest's failure test, so a wrapped counter would
// read as an error. Same guard as s_np_ctx.
uint64_t next_npweb_push_id() {
    int32_t id = g_npweb_push_id.fetch_add(1);
    if (id <= 0) {
        g_npweb_push_id.store(2);
        id = 1;
    }
    return (uint64_t)(uint32_t)id;
}
}  // namespace
HLE(s_npweb_create_request) {
    svc_log("sceNpWebApi2CreateRequest", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a5)) return kWebApiErrInvalidArgument;
    *(int64_t*)PW(a5) = g_npweb_request_id.fetch_add(1);
    return 0;
}
HLE(s_npweb_abort_request) {
    svc_log("sceNpWebApi2AbortRequest", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npweb_add_header) {
    svc_log("sceNpWebApi2AddHttpRequestHeader", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npweb_delete_request) {
    svc_log("sceNpWebApi2DeleteRequest", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npweb_delete_user_context) {
    svc_log("sceNpWebApi2DeleteUserContext", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npweb_send_request) {
    svc_log("sceNpWebApi2SendRequest", a0,a1,a2,a3,a4,a5);
    return NP_ERR_SIGNED_OUT;
}
HLE(s_npweb_read_data) {
    svc_log("sceNpWebApi2ReadData", a0,a1,a2,a3,a4,a5);
    return NP_ERR_SIGNED_OUT;
}
HLE(s_npweb_get_header_value) {
    svc_log("sceNpWebApi2GetHttpResponseHeaderValue", a0,a1,a2,a3,a4,a5);
    return NP_ERR_SIGNED_OUT;
}
HLE(s_npweb_get_header_length) {
    svc_log("sceNpWebApi2GetHttpResponseHeaderValueLength", a0,a1,a2,a3,a4,a5);
    return NP_ERR_SIGNED_OUT;
}
HLE(s_npweb_check_timeout) {
    svc_log("sceNpWebApi2CheckTimeout", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npweb_terminate) {
    svc_log("sceNpWebApi2Terminate", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npweb_push_create_filter) {
    svc_log("sceNpWebApi2PushEventCreateFilter", a0,a1,a2,a3,a4,a5);
    return next_npweb_push_id();
}
HLE(s_npweb_push_create_handle) {
    svc_log("sceNpWebApi2PushEventCreateHandle", a0,a1,a2,a3,a4,a5);
    return next_npweb_push_id();
}
HLE(s_npweb_push_create_pushctx) {
    svc_log("sceNpWebApi2PushEventCreatePushContext", a0,a1,a2,a3,a4,a5);
    return NP_ERR_SIGNED_OUT;
}
HLE(s_npweb_push_delete_filter) {
    svc_log("sceNpWebApi2PushEventDeleteFilter", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npweb_push_delete_handle) {
    svc_log("sceNpWebApi2PushEventDeleteHandle", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npweb_push_delete_pushctx) {
    svc_log("sceNpWebApi2PushEventDeletePushContext", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npweb_push_register_callback) {
    svc_log("sceNpWebApi2PushEventRegisterCallback", a0,a1,a2,a3,a4,a5);
    // Stored handle, never called: no push service exists offline. A null callback is refused
    // (secondary implementation; CONFIDENCE: MED on the code).
    if (!svc_ptrish(a2)) return kWebApiErrInvalidArgument;
    return next_npweb_push_id();
}
HLE(s_npweb_push_register_pushctx_callback) {
    svc_log("sceNpWebApi2PushEventRegisterPushContextCallback", a0,a1,a2,a3,a4,a5);
    // (userCtx, filterId, cb, arg) at the Worms call site; null cb refused as above.
    if (!svc_ptrish(a2)) return kWebApiErrInvalidArgument;
    return next_npweb_push_id();
}
HLE(s_npweb_push_start_pushctx_callback) {
    svc_log("sceNpWebApi2PushEventStartPushContextCallback", a0,a1,a2,a3,a4,a5);
    // Start is the call that would begin service delivery; there is no context to start.
    return NP_ERR_SIGNED_OUT;
}
HLE(s_npweb_push_unregister_callback) {
    svc_log("sceNpWebApi2PushEventUnregisterCallback", a0,a1,a2,a3,a4,a5);
    return 0;
}
HLE(s_npweb_push_unregister_pushctx_callback) {
    svc_log("sceNpWebApi2PushEventUnregisterPushContextCallback", a0,a1,a2,a3,a4,a5);
    return 0;
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
// A title still on the v1 API asks the question NpTrophy2 already answers, so it gets the same
// answer: the lifecycle constructors hand back valid local ids so the bring-up completes, and
// every content/unlock/list query fails with the value s_nptrophy2_unavailable returns, so a
// caller cannot consume unwritten out-structs as trophy data (the #213 class). Trophies are a
// local feature and UserService reports user 1 logged in, so a "user not logged in" answer here
// would contradict that library. The PS4 v1 headers in other emulators (shadPS4 np_error.h) list
// USER_NOT_LOGGED_IN = 0x8055161D; that is a hypothesis about the PS4 facility, not a PS5
// observation, and is deliberately not used. CONFIDENCE: HIGH that failure beats
// success+garbage-out; LOW on the specific constant (same as Trophy2).
//
// CreateContext/CreateHandle keep the PS4 ABI the Trophy2 handlers use (small s32 id written
// through arg0), but only through a plausible guest pointer.
HLE(s_nptrophy_v1_createctx) {
    svc_log("sceNpTrophyCreateContext", a0, a1, a2, a3, a4, a5);
    if (svc_ptrish(a0)) *(int32_t*)PW(a0) = 1;
    return 0;
}
HLE(s_nptrophy_v1_createhandle) {
    svc_log("sceNpTrophyCreateHandle", a0, a1, a2, a3, a4, a5);
    if (svc_ptrish(a0)) *(int32_t*)PW(a0) = 1;
    return 0;
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
//
// The constructors RETURN the new id in eax; arg0 is not an out-pointer. Guest call sites pin
// this: Uncharted calls the argument-less sceNpAuthCreateRequest with rdi still pointing at a
// live object and keeps eax (test/js); PPSA25258 and GTA V keep CreateAsyncRequest's return as
// the request id and pass it on. TUS/Score/Lookup take values (serviceLabel, titleCtxId) in arg0,
// so arg0 is never dereferenced here. CONFIDENCE: HIGH.
namespace {
std::atomic<int32_t> g_np_next_ctx_id{1};
}
HLE(s_np_ctx) {
    svc_log("np ctx/request create", a0, a1, a2, a3, a4, a5);
    int32_t id = g_np_next_ctx_id.fetch_add(1);
    if (id <= 0) {   // wrapped: restart the sequence rather than hand out a non-positive id
        g_np_next_ctx_id.store(2);
        id = 1;
    }
    return (uint64_t)(uint32_t)id;
}
HLE(s_np_offline) {
    return NP_ERR_SIGNED_OUT;
}
// The lifecycle calls of the batch-2 sub-libraries (Tus/Score/Auth/Utility/Matching2/Sns) that
// succeed locally: abort/delete/timeout/thread-param setters and the like. A handler of its own
// rather than a reuse of s_np_ok, so the NpManager registrations s_np_ok serves stay countable
// (tools/re/test_hle_handler_map.py pins them).
HLE(s_np_sublib_ok) {
    return 0;
}

// --- libSceNpCommerce — the PS Store dialog, headless lifecycle like the sign-in prompt (#3784).
// Initialize -> INITIALIZED; Open auto-dismisses to FINISHED (no interactive UI headless, so the
// game's "wait until dismissed" loop still exits — Dead Cells polls UpdateStatus for 3);
// Close -> FINISHED; Terminate -> NONE; GetStatus/UpdateStatus report the state. Status values
// follow the shared dialog convention (NONE=0, INITIALIZED=1, RUNNING=2, FINISHED=3, see the
// sign-in dialog above). CONFIDENCE: MED.
//
// GetResult(result*): titles read the result without checking the return (PPSA20052 reads
// byte [result+4], the `authorized` flag, after ignoring the call's return), so the struct is
// written rather than left as caller stack. No store transaction or authorization happened, which
// is derivable locally: result = USER_CANCELED (1, the dialog closed without completing anything)
// and authorized = 0. This answers no ownership question. CONFIDENCE: MED on the layout (offset 4
// pinned by that call site), LOW on result = 1 versus 0.
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
HLE(s_commerce_result) {
    if (!svc_ptrish(a0)) return 0x80550003ull;   // SCE_NP_ERROR_INVALID_ARGUMENT
    *(int32_t*)PW(a0) = 1;               // result: USER_CANCELED — nothing was completed
    *((uint8_t*)PW(a0) + 4) = 0;          // authorized: no
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
    // libSceNpWebApi2 request + push-event surface: handles are created, every network
    // call that would reach PSN answers SIGNED_OUT instead of blocking (see the handlers above).
    Hle::register_fn("3EI-OSJ65Xc", (HleFn)s_npweb_create_request, "sceNpWebApi2CreateRequest");
    Hle::register_fn("zpiPsH7dbFQ", (HleFn)s_npweb_abort_request, "sceNpWebApi2AbortRequest");
    Hle::register_fn("egOOvrnF6mI", (HleFn)s_npweb_add_header, "sceNpWebApi2AddHttpRequestHeader");
    Hle::register_fn("vvzWO-DvG1s", (HleFn)s_npweb_delete_request, "sceNpWebApi2DeleteRequest");
    Hle::register_fn("9X9+cneTGUU", (HleFn)s_npweb_delete_user_context,
                     "sceNpWebApi2DeleteUserContext");
    Hle::register_fn("lQOCF84lvzw", (HleFn)s_npweb_send_request, "sceNpWebApi2SendRequest");
    Hle::register_fn("OOY9+ObfKec", (HleFn)s_npweb_read_data, "sceNpWebApi2ReadData");
    Hle::register_fn("hksbskNToEA", (HleFn)s_npweb_get_header_value,
                     "sceNpWebApi2GetHttpResponseHeaderValue");
    Hle::register_fn("HwP3aM+c85c", (HleFn)s_npweb_get_header_length,
                     "sceNpWebApi2GetHttpResponseHeaderValueLength");
    Hle::register_fn("3Tt9zL3tkoc", (HleFn)s_npweb_check_timeout, "sceNpWebApi2CheckTimeout");
    Hle::register_fn("bEvXpcEk200", (HleFn)s_npweb_terminate, "sceNpWebApi2Terminate");
    Hle::register_fn("MsaFhR+lPE4", (HleFn)s_npweb_push_create_filter,
                     "sceNpWebApi2PushEventCreateFilter");
    Hle::register_fn("WV1GwM32NgY", (HleFn)s_npweb_push_create_handle,
                     "sceNpWebApi2PushEventCreateHandle");
    Hle::register_fn("NNVf18SlbT8", (HleFn)s_npweb_push_create_pushctx,
                     "sceNpWebApi2PushEventCreatePushContext");
    Hle::register_fn("KJdPcOGmK58", (HleFn)s_npweb_push_delete_filter,
                     "sceNpWebApi2PushEventDeleteFilter");
    Hle::register_fn("fIATVMo4Y1w", (HleFn)s_npweb_push_delete_handle,
                     "sceNpWebApi2PushEventDeleteHandle");
    Hle::register_fn("QafxeZM3WK4", (HleFn)s_npweb_push_delete_pushctx,
                     "sceNpWebApi2PushEventDeletePushContext");
    Hle::register_fn("fY3QqeNkF8k", (HleFn)s_npweb_push_register_callback,
                     "sceNpWebApi2PushEventRegisterCallback");
    Hle::register_fn("lxtHJMwBsaU", (HleFn)s_npweb_push_register_pushctx_callback,
                     "sceNpWebApi2PushEventRegisterPushContextCallback");
    Hle::register_fn("AAj9X+4aGYA", (HleFn)s_npweb_push_start_pushctx_callback,
                     "sceNpWebApi2PushEventStartPushContextCallback");
    Hle::register_fn("hOnIlcGrO6g", (HleFn)s_npweb_push_unregister_callback,
                     "sceNpWebApi2PushEventUnregisterCallback");
    Hle::register_fn("PmyrbbJSFz0", (HleFn)s_npweb_push_unregister_pushctx_callback,
                     "sceNpWebApi2PushEventUnregisterPushContextCallback");
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
    // Every NID is computed from its name by R(), so it cannot drift from the export it names.
    R("sceNpTrophyCreateContext", s_nptrophy_v1_createctx);
    R("sceNpTrophyCreateHandle", s_nptrophy_v1_createhandle);
    R("sceNpTrophyIntCreateHandle", s_nptrophy_v1_createhandle);
    R("sceNpTrophyRegisterContext", s_nptrophy2_regctx);
    R("sceNpTrophyDestroyContext", s_nptrophy2_ok);
    R("sceNpTrophyDestroyHandle", s_nptrophy2_ok);
    R("sceNpTrophyAbortHandle", s_nptrophy2_ok);
    R("sceNpTrophyIntAbortHandle", s_nptrophy2_ok);
    R("sceNpTrophySystemRemoveAll", s_nptrophy2_ok);
    R("sceNpTrophySystemUnregisterTitleSyncedCallback", s_nptrophy2_ok);
    R("sceNpTrophySystemUnregisterTitleUpdateCallback", s_nptrophy2_ok);
    R("sceNpTrophySystemWrapDebugLockTrophy", s_nptrophy2_ok);
    R("sceNpTrophySystemWrapDebugUnlockTrophy", s_nptrophy2_ok);
    R("sceNpTrophySystemWrapRemoveUserData", s_nptrophy2_ok);
    R("sceNpTrophyGetGameInfo", s_nptrophy2_unavailable);
    R("sceNpTrophyGetTrophyUnlockState", s_nptrophy2_unavailable);
    R("sceNpTrophyGetGroupInfo", s_nptrophy2_unavailable);
    R("sceNpTrophyGetTrophyInfo", s_nptrophy2_unavailable);
    R("sceNpTrophyGetGroupIcon", s_nptrophy2_unavailable);
    R("sceNpTrophyGetTrophyIcon", s_nptrophy2_unavailable);
    R("sceNpTrophyNumInfoGetTotal", s_nptrophy2_unavailable);
    R("sceNpTrophyUnlockTrophy", s_nptrophy2_unavailable);
    R("sceNpTrophyShowTrophyList", s_nptrophy2_unavailable);
    R("sceNpTrophyIntGetProgress", s_nptrophy2_unavailable);
    R("sceNpTrophyIntNetSyncTitle", s_nptrophy2_unavailable);
    R("sceNpTrophyIntNetSyncTitles", s_nptrophy2_unavailable);
    R("sceNpTrophySystemDbgCtl", s_nptrophy2_unavailable);
    R("sceNpTrophySystemWrapGetGroupDetails", s_nptrophy2_unavailable);
    R("sceNpTrophySystemWrapGetPlayedTrophyTitles", s_nptrophy2_unavailable);
    R("sceNpTrophySystemWrapGetTitleDetails", s_nptrophy2_unavailable);
    R("sceNpTrophySystemWrapGetTrophyDetailsArray", s_nptrophy2_unavailable);
    R("sceNpTrophySystemWrapGetTrophyTitleIdsByNpTitleId", s_nptrophy2_unavailable);
    // libSceNpTus / libSceNpScore — valid local ids, no PSN data (SIGNED_OUT; see block above).
    R("sceNpTusCreateTitleCtx", s_np_ctx);
    R("sceNpTusCreateNpTitleCtxA", s_np_ctx);
    R("sceNpTssCreateNpTitleCtx", s_np_ctx);
    R("sceNpTssCreateNpTitleCtxA", s_np_ctx);
    R("sceNpTusCreateRequest", s_np_ctx);
    R("sceNpScoreCreateTitleCtx", s_np_ctx);
    R("sceNpScoreCreateRequest", s_np_ctx);
    R("sceNpTusAbortRequest", s_np_sublib_ok);
    R("sceNpTusSetThreadParam", s_np_sublib_ok);
    R("sceNpScoreAbortRequest", s_np_sublib_ok);
    R("sceNpScoreDeleteRequest", s_np_sublib_ok);
    R("sceNpScoreSetThreadParam", s_np_sublib_ok);
    R("sceNpScoreSetTimeout", s_np_sublib_ok);
    // The data path: TUS multi-slot/friends/multi-user variables and cross-save, TUS+TSS
    // get/set/try/delete, Score record/get/ranking/censor, and the async completion calls that
    // would deliver PSN data.
    R("sceNpTusAddAndGetVariable", s_np_offline);
    R("sceNpTusAddAndGetVariableA", s_np_offline);
    R("sceNpTusAddAndGetVariableAAsync", s_np_offline);
    R("sceNpTusAddAndGetVariableAsync", s_np_offline);
    R("sceNpTusAddAndGetVariableVUser", s_np_offline);
    R("sceNpTusDeleteMultiSlotData", s_np_offline);
    R("sceNpTusDeleteMultiSlotDataA", s_np_offline);
    R("sceNpTusDeleteMultiSlotDataAAsync", s_np_offline);
    R("sceNpTusDeleteMultiSlotDataVUser", s_np_offline);
    R("sceNpTusDeleteMultiSlotVariable", s_np_offline);
    R("sceNpTusDeleteMultiSlotVariableA", s_np_offline);
    R("sceNpTusGetData", s_np_offline);
    R("sceNpTusGetDataA", s_np_offline);
    R("sceNpTusGetDataAAsync", s_np_offline);
    R("sceNpTusGetDataAVUser", s_np_offline);
    R("sceNpTusGetDataAVUserAsync", s_np_offline);
    R("sceNpTusGetDataAsync", s_np_offline);
    R("sceNpTusGetDataForCrossSave", s_np_offline);
    R("sceNpTusGetDataForCrossSaveVUser", s_np_offline);
    R("sceNpTusGetDataVUserAsync", s_np_offline);
    R("sceNpTusGetFriendsDataStatusA", s_np_offline);
    R("sceNpTusGetFriendsVariable", s_np_offline);
    R("sceNpTusGetFriendsVariableAAsync", s_np_offline);
    R("sceNpTusGetFriendsVariableAsync", s_np_offline);
    R("sceNpTusGetMultiSlotDataStatus", s_np_offline);
    R("sceNpTusGetMultiSlotDataStatusA", s_np_offline);
    R("sceNpTusGetMultiSlotVariableAsync", s_np_offline);
    R("sceNpTusGetMultiSlotVariableVUser", s_np_offline);
    R("sceNpTusGetMultiUserDataStatusA", s_np_offline);
    R("sceNpTusGetMultiUserVariable", s_np_offline);
    R("sceNpTusGetMultiUserVariableAsync", s_np_offline);
    R("sceNpTusGetMultiUserVariableVUser", s_np_offline);
    R("sceNpTusPollAsync", s_np_offline);
    R("sceNpTusSetData", s_np_offline);
    R("sceNpTusSetDataA", s_np_offline);
    R("sceNpTusSetDataAAsync", s_np_offline);
    R("sceNpTusSetDataAVUser", s_np_offline);
    R("sceNpTusSetDataVUser", s_np_offline);
    R("sceNpTusSetMultiSlotVariable", s_np_offline);
    R("sceNpTusSetMultiSlotVariableA", s_np_offline);
    R("sceNpTusSetMultiSlotVariableAsync", s_np_offline);
    R("sceNpTusSetMultiSlotVariableVUser", s_np_offline);
    R("sceNpTusTryAndSetVariable", s_np_offline);
    R("sceNpTusTryAndSetVariableA", s_np_offline);
    R("sceNpTusTryAndSetVariableAAsync", s_np_offline);
    R("sceNpTusTryAndSetVariableAVUser", s_np_offline);
    R("sceNpTusTryAndSetVariableAsync", s_np_offline);
    R("sceNpTusTryAndSetVariableVUser", s_np_offline);
    R("sceNpTusWaitAsync", s_np_offline);
    R("sceNpTssGetData", s_np_offline);
    R("sceNpTssGetSmallStorage", s_np_offline);
    R("sceNpTssGetSmallStorageAsync", s_np_offline);
    R("sceNpTssGetStorageAsync", s_np_offline);
    R("sceNpScoreCensorComment", s_np_offline);
    R("sceNpScoreCensorCommentAsync", s_np_offline);
    R("sceNpScoreGetBoardInfoAsync", s_np_offline);
    R("sceNpScoreGetFriendsRanking", s_np_offline);
    R("sceNpScoreGetFriendsRankingA", s_np_offline);
    R("sceNpScoreGetGameData", s_np_offline);
    R("sceNpScoreGetRankingByNpId", s_np_offline);
    R("sceNpScorePollAsync", s_np_offline);
    R("sceNpScoreRecordGameData", s_np_offline);
    R("sceNpScoreRecordGameDataAsync", s_np_offline);
    R("sceNpScoreRecordScore", s_np_offline);
    R("sceNpScoreSanitizeComment", s_np_offline);
    R("sceNpScoreWaitAsync", s_np_offline);
    // libSceNpCommerce — headless PS Store dialog + local icon controls (see block above).
    R("sceNpCommerceDialogInitialize", s_commerce_init);
    R("sceNpCommerceDialogOpen", s_commerce_open);
    R("sceNpCommerceDialogOpen2", s_commerce_open);   // not in the 3.20 table; imported by 5 local dumps
    R("sceNpCommerceDialogGetStatus", s_commerce_status);
    R("sceNpCommerceDialogUpdateStatus", s_commerce_status);
    R("sceNpCommerceDialogGetResult", s_commerce_result);
    R("sceNpCommerceDialogClose", s_commerce_close);
    R("sceNpCommerceDialogTerminate", s_commerce_term);
    R("sceNpCommerceShowPsStoreIcon", s_np_sublib_ok);
    R("sceNpCommerceHidePsStoreIcon", s_np_sublib_ok);
    R("sceNpCommerceSetPsStoreIconLayout", s_np_sublib_ok);
    // libSceNpAuth — OAuth request lifecycle: valid local request ids, no authorization codes
    // (SIGNED_OUT, the answer NpManager gives; shadPS4's np_auth.cpp answers the getters the same
    // way when no user is signed in).
    R("sceNpAuthCreateRequest", s_np_ctx);
    R("sceNpAuthCreateAsyncRequest", s_np_ctx);
    R("sceNpAuthAbortRequest", s_np_sublib_ok);
    R("sceNpAuthDeleteRequest", s_np_sublib_ok);
    R("sceNpAuthSetTimeout", s_np_sublib_ok);
    R("sceNpAuthPollAsync", s_np_offline);
    R("sceNpAuthWaitAsync", s_np_offline);
    R("sceNpAuthGetAuthorizationCodeA", s_np_offline);
    R("sceNpAuthGetAuthorizationCodeV3", s_np_offline);
    R("sceNpAuthGetIdToken", s_np_offline);
    R("sceNpAuthGetIdTokenV3", s_np_offline);
    // libSceNpUtility — init succeeds; the PSN-backed lookup/bandwidth/word-filter paths are
    // SIGNED_OUT; the lookup request/title-ctx constructors keep their local ids.
    R("sceNpUtilityInit", s_np_sublib_ok);
    R("sceNpAppInfoIntInitialize", s_np_sublib_ok);
    R("sceNpAppInfoIntFinalize", s_np_sublib_ok);
    R("sceNpBandwidthTestShutdown", s_np_sublib_ok);
    R("sceNpBandwidthTestAbort", s_np_sublib_ok);
    R("sceNpBandwidthTestInitStartDownload", s_np_offline);
    R("sceNpBandwidthTestInitStartUpload", s_np_offline);
    R("sceNpBandwidthTestGetStatus", s_np_offline);
    R("sceNpLookupNetInit", s_np_offline);
    R("sceNpLookupNetIsInit", s_np_sublib_ok);   // 0 = not initialized (honest: it is not)
    R("sceNpLookupCreateRequest", s_np_ctx);
    R("sceNpLookupCreateTitleCtx", s_np_ctx);
    R("sceNpLookupDeleteRequest", s_np_sublib_ok);
    R("sceNpLookupDeleteTitleCtx", s_np_sublib_ok);
    R("sceNpLookupAbortRequest", s_np_sublib_ok);
    R("sceNpLookupSetTimeout", s_np_sublib_ok);
    R("sceNpLookupNetNpId", s_np_offline);
    R("sceNpLookupNpId", s_np_offline);
    R("sceNpLookupPollAsync", s_np_offline);
    R("sceNpLookupWaitAsync", s_np_offline);
    R("sceNpWordFilterPollAsync", s_np_offline);
    R("sceNpWordFilterWaitAsync", s_np_offline);
    // libSceNpMatching2 — room/matching state is PSN-backed, so every state query and
    // connection establishment is SIGNED_OUT; the pure local parameter setters succeed.
    // CONFIDENCE: HIGH on the sign; the context out-width is unverified, so the constructor is
    // refused rather than written (a wrong-width write would hand the guest a corrupt id).
    R("sceNpMatching2SetExtraInitParam", s_np_sublib_ok);
    R("sceNpMatching2SignalingSetPort", s_np_sublib_ok);
    R("sceNpMatching2SignalingAbortConnection", s_np_sublib_ok);
    R("sceNpMatching2CreateContextInternal", s_np_offline);
    R("sceNpMatching2GetRoomJoinedSlotMaskLocal", s_np_offline);
    R("sceNpMatching2GetWorldIdArrayForAllServers", s_np_offline);
    R("sceNpMatching2SetRoomDataInternalExt", s_np_offline);
    R("sceNpMatching2SignalingEstablishConnection", s_np_offline);
    R("sceNpMatching2SignalingGetPort", s_np_offline);
    // libSceNpSns — third-party link (Facebook/Twitch/YouTube) status and tokens are PSN-backed:
    // SIGNED_OUT; the request lifecycle keeps its local ids.
    R("sceNpSnsIntCreateRequest", s_np_ctx);
    R("sceNpSnsTwitchCreateRequest", s_np_ctx);
    R("sceNpSnsYouTubeCreateRequest", s_np_ctx);
    R("sceNpSnsIntDeleteRequest", s_np_sublib_ok);
    R("sceNpSnsIntAbortRequest", s_np_sublib_ok);
    R("sceNpSnsFacebookAbortRequest", s_np_sublib_ok);
    R("sceNpSnsFacebookDeleteRequest", s_np_sublib_ok);
    R("sceNpSnsTwitchAbortRequest", s_np_sublib_ok);
    R("sceNpSnsYouTubeAbortRequest", s_np_sublib_ok);
    R("sceNpSnsYouTubeDeleteRequest", s_np_sublib_ok);
    R("sceNpSnsFacebookGetAccessToken", s_np_offline);
    R("sceNpSnsIntFbGetGameAccessToken", s_np_offline);
    R("sceNpSnsIntFbGetGameAccessTokenAllowed", s_np_offline);
    R("sceNpSnsIntFbGetSystemAccessToken", s_np_offline);
    R("sceNpSnsIntTwGetSystemAccessToken", s_np_offline);
    R("sceNpSnsIntYtGetAccessToken", s_np_offline);
    R("sceNpSnsIntYtRefreshMasterToken", s_np_offline);
    R("sceNpSnsIntLinkedStatus", s_np_offline);
    R("sceNpSnsIntUnlink", s_np_offline);
    R("sceNpSnsTwitchGetAccessToken", s_np_offline);
    R("sceNpSnsYouTubeGetAccessToken", s_np_offline);
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
