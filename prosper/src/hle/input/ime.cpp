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


// Registration for the on-screen keyboard and text-entry dialog, called by
// register_builtin_hle(). One entry point per library keeps every handler `static` to the
// translation unit implementing it (#3735).
void register_ime_hle() {
    // libSceImeDialog (#191): auto-completing text-entry dialog (no keyboard UI). Raw NIDs.
    Hle::register_fn("NUeBrN7hzf0", (HleFn)s_imedlg_init,   "sceImeDialogInit");
    Hle::register_fn("IADmD4tScBY", (HleFn)s_imedlg_status, "sceImeDialogGetStatus");
    Hle::register_fn("x01jxu+vxlc", (HleFn)s_imedlg_result, "sceImeDialogGetResult");
    Hle::register_fn("gyTyVn+bXMw", (HleFn)s_imedlg_term,   "sceImeDialogTerm");
    Hle::register_fn("oBmw4xrmfKs", (HleFn)s_imedlg_abort,  "sceImeDialogAbort");
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
}

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
}   // namespace

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
