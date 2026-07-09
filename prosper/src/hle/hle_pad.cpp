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
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

namespace prosper {

using namespace prosper::input;

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define PW(x) ((void*)(uintptr_t)(x))

namespace {

std::atomic<int> g_pad_handle{1};

// PROSPER_PADLOG=1: trace pad calls (rate-limited) so a boot run shows the game polling input.
bool padlog() { static const bool on = getenv("PROSPER_PADLOG") != nullptr; return on; }
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
// entries, e.g. "3:start;9:start;16:cross;24:cross;31:up+cross". Each entry presses its button(s)
// for PROSPER_PAD_HOLD ms (default 300) starting at <seconds>. When a script is set the pad reports
// CONNECTED for the whole run (a menu that gates on a controller sees one).
//
// Timing is anchored to the FIRST input poll, not process start: the game only reads the pad once it
// reaches the interactive menu, so t=0 == "menu appeared" — robust to how long asset loading takes.
// CONFIDENCE: HIGH (mechanism); MED (that the game's submit maps to OPTIONS="Start" / CROSS).
// The parse + time-eval live in pad.cpp (pure, unit-tested); this file supplies getenv + the clock.
const std::vector<PadScriptEntry>& pad_script() {
    static const std::vector<PadScriptEntry> script = [] {
        const char* env = getenv("PROSPER_PAD_SCRIPT");
        return env ? parse_pad_script(env) : std::vector<PadScriptEntry>{};
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

// Overlay any active scripted press onto `s`. Returns true if a script is driving the pad.
bool apply_pad_script(HostPadState& s) {
    const auto& script = pad_script();
    if (script.empty()) return false;
    s.connected = true;                                     // a controller is present for the whole run
    uint64_t t0 = g_pad_t0_us.load(std::memory_order_relaxed);
    if (t0 == 0) {                                          // anchor to this first poll
        uint64_t expect = 0, now = now_us();
        if (g_pad_t0_us.compare_exchange_strong(expect, now)) t0 = now; else t0 = expect;
    }
    double elapsed = (now_us() - t0) / 1e6;
    s.buttons |= pad_script_buttons_at(script, elapsed, pad_hold_secs());
    return true;
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
    apply_pad_script(s);
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
// (standard device class, no extra capabilities). Conservative 0x20 write to avoid overrun.
HLE(pad_ext_info) { if (a1) memset(PW(a1), 0, 0x20); return 0; }

void register_pad_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("scePadInit", pad_ok);
    R("scePadOpen", pad_open);
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
    #undef R
}

} // namespace prosper
