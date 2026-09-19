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
// The local network libraries allocate opaque contexts even on a disconnected console; connection
// state is reported separately through NetCtl/NP. Returning generic success (0) from these ID-returning
// constructors instead creates an invalid context and makes their owner's initialization fail. The
// signatures and positive-return contract agree with the PS5 3.20 symbol table and the independently
// implemented SDK surface; no online identity or connectivity is fabricated here.
namespace {
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
#endif
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
#endif
}

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


// Registration for this library, called by register_builtin_hle(). One entry point per
// Sony library keeps every handler `static` to the translation unit implementing it (#3735).
void register_savedata_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
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
    Hle::register_fn("WAzWTZm1H+I", (HleFn)s_savedata_transfermount, "sceSaveDataTransferringMount");
    Hle::register_fn("RjMlsR8EXrw", (HleFn)s_savedata_transfermount_ps4, "sceSaveDataTransferringMountPs4");
    Hle::register_fn("X4MYzukPc3g", (HleFn)s_savedata_dirname_search_ps4, "sceSaveDataDirNameSearchPs4");
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
