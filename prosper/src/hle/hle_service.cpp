// hle_service.cpp — HLE of PS5 system services (user, NP/online, mouse, app content,
// dialogs). Bring-up policy: openers return a valid positive handle; queries zero their
// output struct and report a sane "not signed in / no device" state and success, so the
// game gets consistent values instead of uninitialized memory.
// (Game-controller input — libScePad — moved to hle_pad.cpp with a real host backend.)
#include "dispatch.hpp"
#include "nid.hpp"
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#ifdef __linux__
#include <sys/mman.h>   // msync page-mapped probe in svc_log (diagnostic-only)
#endif

namespace prosper {

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define PW(x) ((void*)(uintptr_t)(x))

namespace { std::atomic<uint64_t> g_handle{1}; }

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

// --- NP / online (single-player: report signed-out / unreachable, success) ---
HLE(s_np_state)       { if (a1) *(int32_t*)PW(a1) = 1; return 0; }           // SCE_NP_STATE_SIGNED_OUT
HLE(s_np_reach)       { if (a1) *(int32_t*)PW(a1) = 0; return 0; }
HLE(s_np_accountid)   { if (a1) *(uint64_t*)PW(a1) = 0; return 0; }
HLE(s_np_country)     { if (a1) memset(PW(a1), 0, 4); return 0; }

// --- mouse (report a device that exists but has no input; pad -> hle_pad.cpp real backend) ---
HLE(s_open)           { return g_handle++; }                                 // sceMouseOpen -> handle
// sceMouseRead(handle, SceMouseData*, num) returns the number of mouse events read. SceMouseData
// (~0x18 bytes) is NOT ScePadData — sharing the pad stub returned one "valid" entry whose memset
// overran a single-entry mouse buffer, and the game consumed a phantom mouse event every call. No
// mouse attached: zero one entry defensively, report 0 events.
HLE(s_mouse_read)     { if (a1) memset(PW(a1), 0, 0x18); return 0; }

// --- app content ---
HLE(s_appcontent_int) { if (a1) *(int32_t*)PW(a1) = 0; return 0; }

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
namespace { std::atomic<int> g_msgdialog_status{0 /*NONE*/}; }
HLE(s_dialog_initialize) { g_msgdialog_status.store(1 /*INITIALIZED*/); return 0; }
HLE(s_dialog_open)       { g_msgdialog_status.store(3 /*FINISHED (auto-dismiss)*/); return 0; }
HLE(s_dialog_close)      { g_msgdialog_status.store(0 /*NONE*/); return 0; }
HLE(s_dialog_terminate)  { g_msgdialog_status.store(0 /*NONE*/); return 0; }
HLE(s_dialog_status)     { return (uint64_t)(unsigned)g_msgdialog_status.load(); }
// SceMsgDialogResult = { u32 mode; u32 result; u32 buttonId; char reserved[32] } = 0x2C bytes
// (shadPS4 msgdialog_ui.h DialogResult). Was memset(0x30) — 4 bytes PAST the caller's struct,
// exactly the f_fstat oversized-write class this file warns about.
HLE(s_dialog_result)   { if (a0) memset(PW(a0), 0, 0x2C); return 0; }

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

// Diagnostic logging for this service family, gated on PROSPER_SVCLOG=1 (same pattern as
// PROSPER_FILELOG/[file]). Dumps call args and a bounded hexdump of pointer-shaped args so the
// PS5-only ABIs (sceSaveDataMount3/Prepare/Commit — no Kyty/shadPS4 reference exists) can be
// pinned from live captures instead of guessed.
namespace {
bool svclog() { static int v = getenv("PROSPER_SVCLOG") ? 1 : 0; return v; }
bool svc_ptrish(uint64_t v) { return v >= 0x10000 && v < 0x7fffffffffffull; }
void svc_log(const char* fn, uint64_t a0, uint64_t a1, uint64_t a2,
             uint64_t a3, uint64_t a4, uint64_t a5, int dump_words = 8) {
    if (!svclog()) return;
    fprintf(stderr, "[svc] %s(%#lx, %#lx, %#lx, %#lx, %#lx, %#lx)\n",
            fn, (unsigned long)a0, (unsigned long)a1, (unsigned long)a2,
            (unsigned long)a3, (unsigned long)a4, (unsigned long)a5);
    const uint64_t args[6] = { a0, a1, a2, a3, a4, a5 };
    for (int i = 0; i < 6; i++) {
        if (!svc_ptrish(args[i])) continue;
        // Never read across the arg's 4 KiB page end: an out-param can be a tiny heap block whose
        // page neighbor is unmapped, and a diagnostic must not be able to fault the boot.
        uint64_t page_left = 0x1000 - (args[i] & 0xfff);
        int words = (int)(page_left / 8); if (words > dump_words) words = dump_words;
#ifdef __linux__
        // Integer-valued args can masquerade as pointers; probe the page is actually mapped
        // (msync on an unmapped range fails ENOMEM) before dereferencing — a diagnostic must
        // never be able to fault the boot.
        if (msync((void*)(uintptr_t)(args[i] & ~0xfffull), 1, MS_ASYNC) != 0) continue;
#endif
        fprintf(stderr, "[svc]   a%d ->", i);
        const uint64_t* q = (const uint64_t*)PW(args[i]);
        for (int w = 0; w < words; w++) fprintf(stderr, " %016lx", (unsigned long)q[w]);
        fprintf(stderr, "\n");
    }
}
}

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
HLE(s_savedata_init3)   { svc_log("sceSaveDataInitialize3", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_savedata_term)    { return 0; }
HLE(s_savedata_txres)   { svc_log("sceSaveDataCreateTransactionResource", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_savedata_txres_del) { return 0; }
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
HLE(s_savedata_umount2) { svc_log("sceSaveDataUmount2", a0,a1,a2,a3,a4,a5); savedata0_umount(); return 0; }
HLE(s_savedata_prepare) { svc_log("sceSaveDataPrepare", a0,a1,a2,a3,a4,a5); return 0; }
HLE(s_savedata_commit)  { svc_log("sceSaveDataCommit", a0,a1,a2,a3,a4,a5); return 0; }
// sceSaveDataDirNameSearch(const SearchCond* cond, SearchResult* result) — PS4-inherited contract,
// pinned by DOLL's own callsite (eboot+0x224e920): cond {u32 userId@0; titleId*@8=0; dirName*@0x10=0;
// key/order@0x18=0}, result {u32 hitNum@0; DirName* dirNames@8 (caller buffer, this+0x40);
// u32 dirNamesNum@0x10=0x400; u32 setNum@0x14(1.7+)} — the guest stores hitNum to this+0x8040 as its
// save count. Fresh console: 0 saves found, success (shadPS4 does exactly this when the save path
// doesn't exist). CONFIDENCE: HIGH (guest callsite disassembly + shadPS4 agree).
HLE(s_savedata_dirsearch) {
    svc_log("sceSaveDataDirNameSearch", a0,a1,a2,a3,a4,a5);
    if (!a1) return 0x809F0000ull;        // SAVE_DATA_ERROR_PARAMETER
    uint32_t* res = (uint32_t*)PW(a1);
    res[0] = 0;                            // hitNum = 0 (no existing saves)
    res[5] = 0;                            // setNum (offset 0x14) = 0
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

void register_service_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    // NpTrophy2: the config/info queries whose success-with-garbage-out crashed DOLL (see above).
    Hle::register_fn("4IzqhhUQ3nk", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetGameInfo");
    Hle::register_fn("y3zHpdZO6ME", (HleFn)s_nptrophy2_unavailable, "sceNpTrophy2GetTrophyInfoArray");
    // user service
    R("sceUserServiceGetInitialUser", s_user_initial);
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
    // NP
    R("sceNpGetState", s_np_state);
    R("sceNpGetNpReachabilityState", s_np_reach);
    R("sceNpGetAccountIdA", s_np_accountid);
    R("sceNpGetAccountCountryA", s_np_country);
    R("sceNpCheckCallback", s_ok);
    R("sceNpRegisterStateCallback", s_ok);
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
    Hle::register_fn("bcolXMmp6qQ", (HleFn)s_ok,                  "sceAppContentTemporaryDataUnmount");
    R("sceCommonDialogInitialize", s_ok);
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
    R("sceSystemServiceHideSplashScreen", s_ok);
    R("sceSystemServiceGetStatus", s_syss_getstatus);
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
    Hle::register_fn("yKDy8S5yLA0", (HleFn)s_savedata_term,      "sceSaveDataTerminate");
    Hle::register_fn("gjRZNnw0JPE", (HleFn)s_savedata_txres,     "sceSaveDataCreateTransactionResource");
    Hle::register_fn("lJUQuaKqoKY", (HleFn)s_savedata_txres_del, "sceSaveDataDeleteTransactionResource");
    Hle::register_fn("ZP4e7rlzOUk", (HleFn)s_savedata_mount3,    "sceSaveDataMount3");
    Hle::register_fn("uW4vfTwMQVo", (HleFn)s_savedata_umount2,   "sceSaveDataUmount2");
    Hle::register_fn("sDCBrmc61XU", (HleFn)s_savedata_prepare,   "sceSaveDataPrepare");
    Hle::register_fn("ie7qhZ4X0Cc", (HleFn)s_savedata_commit,    "sceSaveDataCommit");
    Hle::register_fn("dyIhnXq-0SM", (HleFn)s_savedata_dirsearch, "sceSaveDataDirNameSearch");
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
    // libSceNpUniversalDataSystem — inert ids (guarded LOW-confidence out-writes).
    Hle::register_fn("sjaobBgqeB4", (HleFn)s_npuds_ok,     "sceNpUniversalDataSystemInitialize");
    Hle::register_fn("5zBnau1uIEo", (HleFn)s_npuds_create, "sceNpUniversalDataSystemCreateContext");
    Hle::register_fn("hT0IAEvN+M0", (HleFn)s_npuds_create, "sceNpUniversalDataSystemCreateHandle");
    Hle::register_fn("tpFJ8LIKvPw", (HleFn)s_npuds_ok,     "sceNpUniversalDataSystemRegisterContext");
    #undef R
}

} // namespace prosper
