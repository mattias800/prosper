// hle_pad.cpp — libScePad HLE: real game-controller input.
//
// Replaces the earlier zero-filling pad stubs (hle_service.cpp) with a correct ScePadData/
// ScePadControllerInformation contract. Input comes from a pluggable PadBackend (input/pad.hpp):
// prosper_core's default is neutral/disconnected, and a host frontend (SDL3 gamepad or Linux evdev,
// under frontends/) installs itself from the harness — so this file, and all of prosper_core, is
// free of host-device code and fully unit-testable.
//
// This also fixes a latent stub bug: the old scePadReadState wrote only 48 of the 120-byte struct,
// leaving connected/timestamp/count uninitialized. The full struct is now filled.
//
// Diagnostics: PROSPER_PADLOG traces calls; PROSPER_PAD_PRESS injects a connected pad with CROSS
// held (hardware-free way to drive a "Press [button]" screen / verify the full path). CONFIDENCE:
// HIGH (contract), MED (which titles gate on `connected`).
#include "dispatch.hpp"
#include "nid.hpp"
#include "../input/pad.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace prosper {

using namespace prosper::input;

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define PW(x) ((void*)(uintptr_t)(x))

namespace {

std::atomic<int> g_pad_handle{1};

// PROSPER_PADLOG=1: trace pad calls (rate-limited) so a boot run shows the game polling input.
bool padlog() { static const bool on = getenv("PROSPER_PADLOG") != nullptr; return on; }
bool pad_script_log() { static const bool on = getenv("PROSPER_PAD_SCRIPT_LOG") != nullptr; return on; }
void padlog_once(const char* what, const HostPadState* s) {
    if (!padlog()) return;
    static std::atomic<int> n{0};
    int i = n.fetch_add(1);
    if (i < 8 || (i % 512) == 0) {
        if (s) fprintf(stderr, "[pad] %s call#%d connected=%d buttons=0x%x lx=%02x ly=%02x\n",
                       what, i, (int)s->connected, s->buttons, s->left_x, s->left_y);
        else   fprintf(stderr, "[pad] %s call#%d\n", what, i);
    }
}

// Monotonic microsecond timestamp for the report (Sony fills a process-time stamp).
uint64_t now_us() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

// --- PROSPER_PAD_SCRIPT: hardware-free timed button sequence -------------------------------------
// Drives a scripted controller so a headless run can navigate menus (e.g. the title→menu→save→name
// flow in issue #163) with no host device. Format: a ';'-separated list of "<seconds>:<button>[+..]"
// entries, e.g. "3:start;9:start;16:cross;24:cross;31:up+cross". Prefix with '@' to load a route
// file; files may use newlines, comments, and explicit ranges such as "f300-340:cross". Time points
// use PROSPER_PAD_HOLD ms (default 300); flip points use PROSPER_PAD_FRAME_HOLD (default 8). When a
// script is set the pad reports
// CONNECTED for the whole run (a menu that gates on a controller sees one).
//
// Timing is anchored to the FIRST input poll, not process start: the game only reads the pad once it
// reaches the interactive menu, so t=0 == "menu appeared" — robust to how long asset loading takes.
// CONFIDENCE: HIGH (mechanism); MED (that the game's submit maps to OPTIONS="Start" / CROSS).
// The parse + time-eval live in pad.cpp (pure, unit-tested); this file supplies getenv + the clock.
const std::vector<PadScriptEntry>& pad_script() {
    static const std::vector<PadScriptEntry> script = [] {
        const char* env = getenv("PROSPER_PAD_SCRIPT");
        if (!env) return std::vector<PadScriptEntry>{};
        std::string error;
        auto loaded = load_pad_script(env, &error);
        if (!error.empty()) fprintf(stderr, "[pad] PROSPER_PAD_SCRIPT: %s\n", error.c_str());
        return loaded;
    }();
    return script;
}

double pad_hold_secs() {
    static const double h = [] {
        const char* e = getenv("PROSPER_PAD_HOLD");
        return e ? atof(e) / 1000.0 : 0.30;   // ms -> s
    }();
    return h;
}

// t0 (us) of the first poll; 0 until set. steady_clock so it matches now_us().
std::atomic<uint64_t> g_pad_t0_us{0};

// Frame anchor: the flip count at the first poll, so `f<N>:` entries measure flips-since-first-poll.
// prosper_vo_flip_count() is the guest's presented-frame counter (hle_graphics.cpp), boot-speed-
// invariant — unlike wall-clock it lands the same input on the same game state across builds (#302).
extern "C" uint64_t prosper_vo_flip_count();
std::atomic<uint64_t> g_pad_flip0{std::numeric_limits<uint64_t>::max()};

int64_t pad_frame_hold() {   // how many flips to hold a frame-anchored press (default 8)
    static const int64_t h = [] { const char* e = getenv("PROSPER_PAD_FRAME_HOLD"); return e ? (int64_t)atoll(e) : 8; }();
    return h;
}

int64_t pad_frame_now() {
    const uint64_t flips = prosper_vo_flip_count();
    uint64_t base = g_pad_flip0.load(std::memory_order_relaxed);
    if (base == std::numeric_limits<uint64_t>::max()) {
        g_pad_flip0.compare_exchange_strong(base, flips, std::memory_order_relaxed);
        base = g_pad_flip0.load(std::memory_order_relaxed);
    }
    return flips >= base ? (int64_t)(flips - base) : 0;
}

// Overlay any active scripted press onto `s`. Returns true if a script is driving the pad.
bool apply_pad_script(HostPadState& s, int64_t frame) {
    const auto& script = pad_script();
    if (script.empty()) return false;
    s.connected = true;                                     // a controller is present for the whole run
    uint64_t t0 = g_pad_t0_us.load(std::memory_order_relaxed);
    if (t0 == 0) {                                          // anchor to this first poll
        uint64_t expect = 0, now = now_us();
        if (g_pad_t0_us.compare_exchange_strong(expect, now)) t0 = now; else t0 = expect;
    }
    double elapsed = (now_us() - t0) / 1e6;
    const uint32_t scripted =
        pad_script_buttons_at(script, elapsed, pad_hold_secs(), frame, pad_frame_hold());
    if (pad_script_log()) {
        static std::atomic<uint32_t> previous{std::numeric_limits<uint32_t>::max()};
        const uint32_t observed = previous.exchange(scripted, std::memory_order_relaxed);
        if (observed != scripted) {
            const std::string names = pad_button_names(scripted);
            fprintf(stderr, "[pad-script] elapsed=%.3f frame=%lld buttons=%s\n", elapsed,
                    (long long)frame, names.empty() ? "neutral" : names.c_str());
        }
    }
    s.buttons |= scripted;
    return true;
}

// Record the final button stream as explicit flip ranges. Completed intervals are flushed immediately;
// only a button still held when a process is force-killed can be absent from the route tail.
void pad_record(int64_t frame, uint32_t buttons) {
    static const char* output = getenv("PROSPER_PAD_RECORD");
    if (!output || !*output) return;
    struct Recorder {
        std::mutex mutex;
        bool initialized = false;
        FILE* file = nullptr;
        uint32_t previous = 0;
        int64_t start = 0;
    };
    static Recorder recorder;
    std::lock_guard<std::mutex> lock(recorder.mutex);
    if (!recorder.initialized) {
        recorder.initialized = true;
        std::filesystem::path path(output);
        std::error_code ec;
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            fprintf(stderr, "[pad] PROSPER_PAD_RECORD: cannot create '%s': %s\n",
                    path.parent_path().string().c_str(), ec.message().c_str());
        } else if ((recorder.file = fopen(output, "w"))) {
            fprintf(recorder.file,
                    "# recorded PROSPER_PAD_SCRIPT route; fN is display flips since first pad poll\n");
            fflush(recorder.file);
            fprintf(stderr, "[pad] recording input route -> %s\n", output);
        } else {
            fprintf(stderr, "[pad] PROSPER_PAD_RECORD: cannot open '%s': %s\n",
                    output, strerror(errno));
        }
    }
    if (!recorder.file || buttons == recorder.previous) return;
    if (recorder.previous) {
        const int64_t end = std::max(frame, recorder.start + 1);
        const std::string names = pad_button_names(recorder.previous);
        if (!names.empty()) {
            fprintf(recorder.file, "f%lld-%lld:%s\n", (long long)recorder.start,
                    (long long)end, names.c_str());
            fflush(recorder.file);
        }
    }
    recorder.previous = buttons;
    recorder.start = frame;
}

// Snapshot the controller for a given pad handle via the installed backend. With no host frontend
// installed, prosper_core's default backend reports a neutral, disconnected pad (fully-formed
// struct). PROSPER_PAD_PRESS overrides with a synthetic connected pad (CROSS held); PROSPER_PAD_SCRIPT
// overrides with a timed button sequence — both are device-independent end-to-end test drivers.
HostPadState snapshot(int /*handle*/, const char* what) {
    HostPadState s;   // neutral, disconnected by default
    pad_backend()->poll(0, s);
    if (getenv("PROSPER_PAD_PRESS")) {
        s.connected = true;
        s.buttons  |= SCE_PAD_BUTTON_CROSS;
    }
    const int64_t frame = pad_frame_now();
    apply_pad_script(s, frame);
    pad_record(frame, s.buttons);
    padlog_once(what, &s);
    return s;
}

} // namespace

// scePadInit / scePadClose / effector no-ops -> OK.
HLE(pad_ok) { if (padlog()) fprintf(stderr, "[pad] init/ok call\n"); return 0; }

// scePadOpen(userId, type, index, param) -> positive handle.
HLE(pad_open) { int h = g_pad_handle.fetch_add(1);
    if (padlog()) fprintf(stderr, "[pad] OPEN userId=%llu type=%llu index=%llu -> handle=%d\n",
                          (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2, h);
    return (uint64_t)h; }

// scePadGetHandle(userId, type, index) -> handle (games that opened once may re-query it).
HLE(pad_get_handle) { return 1; }

// scePadIsValidHandle(handle) -> nonzero if the handle is an open pad. Was MISSING -> returned 0, which a
// caller reads as "invalid". Our handles start at 1, so report valid for any positive handle.
HLE(pad_is_valid_handle) { return a0 >= 1 ? 1 : 0; }

// scePadReadState(handle, ScePadData* out) -> 0 on success. One current snapshot.
HLE(pad_read_state) {
    if (!a1) return 0;
    HostPadState s = snapshot((int)a0, __func__);
    pad_fill_data((ScePadData*)PW(a1), s, now_us(), s.connected ? 1 : 0);
    return 0;
}

// scePadRead(handle, ScePadData* out, int num) -> number of states written (>=1). We report the
// single current state (no historical buffering yet), which is a valid, common driver behavior.
HLE(pad_read) {
    int num = (int)(int64_t)a2;
    if (!a1 || num < 1) return 0;
    HostPadState s = snapshot((int)a0, __func__);
    pad_fill_data((ScePadData*)PW(a1), s, now_us(), s.connected ? 1 : 0);
    return 1;   // one state written
}

// scePadGetControllerInformation(handle, ScePadControllerInformation* out) -> 0.
HLE(pad_get_info) {
    if (!a1) return 0;
    HostPadState s = snapshot((int)a0, __func__);
    pad_fill_controller_info((ScePadControllerInformation*)PW(a1), s.connected, s.connected ? 1 : 0);
    return 0;
}

// scePadDeviceClassGetExtendedInformation(handle, out*) -> 0. Report a zeroed extended-info block
// (standard device class, no extra capabilities). OrbisPadDeviceClassExtendedInformation is exactly
// 20 bytes (0x14): deviceClass(4) + reserved[4](4) + a 12-byte classData union (shadPS4 pad.h). Writing
// 0x20 overran a guest-stack-allocated struct by 12 bytes → stack-canary smash (the #283 class); write
// only the real 0x14 (under-writing is safe, over-writing is not — same lesson as pad_parse below).
HLE(pad_ext_info) { if (a1) memset(PW(a1), 0, 0x14); return 0; }

// scePadDeviceClassParseData(handle, const OrbisPadData* in, OrbisPadDeviceClassData* out) -> 0.
// Parses a raw device-class HID report (steering wheels, guitars, etc.) into a structured form. A
// standard DualSense carries NO device-class payload, so the correct answer is "no valid data": clear
// the out's leading fields — deviceClass (u32 @0) = STANDARD/0 and bDataValid (bool @4) = false — so a
// game's `if (out.bDataValid)` sees false rather than consuming uninitialized memory. Bounded 8-byte
// write (never the full union) per the oversized-write lesson; cross-checked vs shadPS4 pad.cpp
// (returns OK). CONFIDENCE: MED — normal input still flows through scePadRead.
HLE(pad_class_parse) { if (a2) memset(PW(a2), 0, 8); return 0; }

// scePadGetTriggerEffectState(handle, ScePadTriggerEffectState* out) -> 0. DualSense adaptive-trigger
// feedback state. No reference layout (PS5-only; absent from shadPS4/Kyty) and The Messenger is a 2D
// platformer that does not drive adaptive triggers, so report a neutral/zeroed state and succeed.
// Bounded 8-byte clear at the out pointer avoids leaving garbage without risking an overrun of an
// unknown-size struct. CONFIDENCE: LOW — arg/layout unverified; never called on this title.
HLE(pad_trigger_state) { if (a1) memset(PW(a1), 0, 8); return 0; }

void register_pad_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("scePadInit", pad_ok);
    R("scePadOpen", pad_open);
    R("scePadOpenExt", pad_open);              // was MISSING -> returned 0 (read as a valid handle 0 / error)
    R("scePadIsValidHandle", pad_is_valid_handle);   // was MISSING -> 0 = "invalid"
    R("scePadClose", pad_ok);
    R("scePadGetHandle", pad_get_handle);
    R("scePadGetControllerInformation", pad_get_info);
    R("scePadDeviceClassGetExtendedInformation", pad_ext_info);
    R("scePadReadState", pad_read_state);
    R("scePadRead", pad_read);
    // Effectors we accept but don't yet drive on the host (vibration/lightbar/motion): return OK so
    // the game's setup path proceeds. Force feedback via evdev EV_FF is a follow-up.
    R("scePadSetVibration", pad_ok);
    R("scePadSetMotionSensorState", pad_ok);
    R("scePadSetTiltCorrectionState", pad_ok);
    R("scePadSetAngularVelocityDeadbandState", pad_ok);
    R("scePadResetLightBar", pad_ok);
    R("scePadSetLightBar", pad_ok);
    R("scePadResetOrientation", pad_ok);
    R("scePadSetVibrationMode", pad_ok);       // vibration-mode config (output) — accept, no-op
    R("scePadSetTriggerEffect", pad_ok);       // DualSense adaptive-trigger effect (output) — accept, no-op
    R("scePadDeviceClassParseData", pad_class_parse);   // raw device-class report -> "no valid data"
    R("scePadGetTriggerEffectState", pad_trigger_state);// adaptive-trigger state -> zeroed/neutral
    #undef R
}

} // namespace prosper
