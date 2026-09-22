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
#include "hle/net/sce_net_errors.hpp"  // libSceNet error encoding (#3545)
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
// The local network libraries allocate opaque contexts even on a disconnected console; connection
// state is reported separately through NetCtl/NP. Returning generic success (0) from these ID-returning
// constructors instead creates an invalid context and makes their owner's initialization fail. The
// signatures and positive-return contract agree with the PS5 3.20 symbol table and the independently
// implemented SDK surface; no online identity or connectivity is fabricated here.
namespace {
std::atomic<int32_t> g_net_pool_id{1};
std::atomic<int32_t> g_ssl_context_id{1};
}
HLE(s_net_pool_create) {
    svc_log("sceNetPoolCreate", a0,a1,a2,a3,a4,a5);
    if (!svc_ptrish(a0) || (int32_t)a1 <= 0) return prosper::net::kNetErrorMFile; // errno 24 (EMFILE); see sce_net_errors.hpp
    return (uint64_t)(uint32_t)g_net_pool_id.fetch_add(1);
}
HLE(s_ssl_init) {
    svc_log("sceSslInit", a0,a1,a2,a3,a4,a5);
    if (!a0) return 0x8094000cull; // SCE_SSL_ERROR_OUT_OF_SIZE
    return (uint64_t)(uint32_t)g_ssl_context_id.fetch_add(1);
}

// --- mouse (report a device that exists but has no input; pad -> hle_pad.cpp real backend) ---
HLE(s_open)           { return g_handle++; }                                 // sceMouseOpen -> handle
// sceMouseRead(handle, SceMouseData*, num) returns the number of mouse events read. SceMouseData
// (~0x18 bytes) is NOT ScePadData — sharing the pad stub returned one "valid" entry whose memset
// overran a single-entry mouse buffer, and the game consumed a phantom mouse event every call. No
// mouse attached: zero one entry defensively, report 0 events.
HLE(s_mouse_read)     { if (a1) memset(PW(a1), 0, 0x18); return 0; }

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
HLE(s_live_streaming_init) {
    svc_log("sceGameLiveStreamingInitialize", a0,a1,a2,a3,a4,a5);
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
    Hle::register_fn("dgJBaeJnGpo", (HleFn)s_net_pool_create, "sceNetPoolCreate");
    Hle::register_fn("hdpVEUDFW3s", (HleFn)s_ssl_init, "sceSslInit");
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
    R("sceSystemServiceHideSplashScreen", s_ok);
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
    Hle::register_fn("3RQ5aQfnstU", (HleFn)s_syss_noticeskip, "sceSystemServiceGetNoticeScreenSkipFlag");
    Hle::register_fn("kvYEw2lBndk", (HleFn)s_live_streaming_init, "sceGameLiveStreamingInitialize");
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
