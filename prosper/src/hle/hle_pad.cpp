// hle_pad.cpp — libScePad HLE: real game-controller input.
//
// Replaces the earlier zero-filling pad stubs (hle_service.cpp) with a correct ScePadData/
// ScePadControllerInformation contract backed by a host controller (Linux evdev; see input/pad.cpp).
//
// Gating: live host input is behind PROSPER_PAD=1 so the default boot stays deterministic (no
// nondeterministic device state in ctest / headless runs) — exactly the project's gated-switch
// policy (cf. PROSPER_RENDER, PROSPER_GUEST_FS). With the gate off, the pad reports a valid but
// disconnected/neutral state (identical observable behavior to the old stub, but now filling the
// FULL 120-byte struct instead of only 48 bytes — the old stub left connected/timestamp/count
// uninitialized). CONFIDENCE: HIGH (contract), MED (which titles gate on `connected`).
#include "dispatch.hpp"
#include "nid.hpp"
#include "../input/pad.hpp"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>

namespace prosper {

using namespace prosper::input;

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define PW(x) ((void*)(uintptr_t)(x))

namespace {

std::atomic<int> g_pad_handle{1};

bool live_input() {
    static const bool on = (getenv("PROSPER_PAD") != nullptr) && pad_backend_available();
    return on;
}

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

// Snapshot the host controller for a given pad handle. Off the live path, returns a neutral,
// disconnected pad (centered sticks, no buttons) — matching the pre-existing "device exists but no
// input" bring-up behavior, now with a fully-formed struct.
HostPadState snapshot(int /*handle*/, const char* what) {
    HostPadState s;   // neutral, disconnected by default
    if (live_input()) pad_backend_poll(0, s);
    padlog_once(what, &s);
    return s;
}

} // namespace

// scePadInit / scePadClose / effector no-ops -> OK.
HLE(pad_ok) { return 0; }

// scePadOpen(userId, type, index, param) -> positive handle.
HLE(pad_open) { return (uint64_t)g_pad_handle.fetch_add(1); }

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
