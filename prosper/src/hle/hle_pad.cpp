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

// Snapshot the controller for a given pad handle via the installed backend. With no host frontend
// installed, prosper_core's default backend reports a neutral, disconnected pad (fully-formed
// struct). PROSPER_PAD_PRESS overrides with a synthetic connected pad (CROSS held) for hardware-free
// end-to-end testing — a device-independent successor to the old stub's inline press injection.
HostPadState snapshot(int /*handle*/, const char* what) {
    HostPadState s;   // neutral, disconnected by default
    pad_backend()->poll(0, s);
    if (getenv("PROSPER_PAD_PRESS")) {
        s.connected = true;
        s.buttons  |= SCE_PAD_BUTTON_CROSS;
    }
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
