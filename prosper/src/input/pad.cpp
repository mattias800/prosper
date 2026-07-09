// pad.cpp — pure mapping core for the libScePad HLE (see pad.hpp).
// No hardware, no globals, no platform code — just HostPadState -> Sony ABI structs. The byte
// layout is asserted here so a struct-layout regression fails at compile time, not silently in the
// game. The backend lives in pad_evdev.cpp.
#include "pad.hpp"
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <atomic>

namespace prosper::input {

// --- layout guards: these MUST match the Sony/Kyty ScePadData or the game reads wrong bytes ------
static_assert(sizeof(ScePadData) == 120, "ScePadData must be 120 bytes");
static_assert(offsetof(ScePadData, buttons)            == 0x00, "buttons");
static_assert(offsetof(ScePadData, left_stick_x)       == 0x04, "left_stick_x");
static_assert(offsetof(ScePadData, analog_buttons_l2)  == 0x08, "analog_buttons_l2");
static_assert(offsetof(ScePadData, orientation_x)      == 0x0c, "orientation_x");
static_assert(offsetof(ScePadData, acceleration_x)     == 0x1c, "acceleration_x");
static_assert(offsetof(ScePadData, angular_velocity_x) == 0x28, "angular_velocity_x");
static_assert(offsetof(ScePadData, touch_num)          == 0x34, "touch_num");
static_assert(offsetof(ScePadData, connected)          == 0x4c, "connected");
static_assert(offsetof(ScePadData, timestamp)          == 0x50, "timestamp");
static_assert(offsetof(ScePadData, ext_unit_id)        == 0x58, "ext_unit_id");
static_assert(offsetof(ScePadData, connected_count)    == 0x68, "connected_count");
static_assert(offsetof(ScePadData, device_unique_data) == 0x6c, "device_unique_data");

static_assert(sizeof(ScePadControllerInformation) == 32, "ScePadControllerInformation must be 32 bytes");
static_assert(offsetof(ScePadControllerInformation, device_class) == 0x10, "device_class");

uint8_t pad_axis_u8(int raw, int min, int max) {
    if (max <= min) return 0x80;
    long v = (255L * (long)(raw - min)) / (long)(max - min);
    if (v < 0)   v = 0;
    if (v > 255) v = 255;
    return (uint8_t)v;
}

uint32_t pad_trigger_buttons(uint8_t l2, uint8_t r2) {
    uint32_t b = 0;
    if (l2 > kPadTriggerButtonThreshold) b |= SCE_PAD_BUTTON_L2;
    if (r2 > kPadTriggerButtonThreshold) b |= SCE_PAD_BUTTON_R2;
    return b;
}

void pad_fill_data(ScePadData* out, const HostPadState& s, uint64_t timestamp, uint8_t connected_count) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));

    out->buttons           = s.buttons;
    out->left_stick_x      = s.left_x;
    out->left_stick_y      = s.left_y;
    out->right_stick_x     = s.right_x;
    out->right_stick_y     = s.right_y;
    out->analog_buttons_l2 = s.l2;
    out->analog_buttons_r2 = s.r2;

    // No motion fusion yet: report an identity orientation (w=1) and zero accel/gyro. A game that
    // reads orientation gets a valid, level quaternion rather than a zero (non-unit) one.
    out->orientation_w = 1.0f;

    // Touchpad: report "no active touches" but give the two touch slots distinct ids (matches the
    // Sony/Kyty convention; some titles assume id!=0 on an empty slot).
    out->touch_num = 0;
    out->touch0_id = 1;
    out->touch1_id = 2;

    out->connected            = s.connected ? 1 : 0;
    out->timestamp            = timestamp;
    out->connected_count      = connected_count;
    out->device_unique_data_len = 0;
}

// --- pluggable backend: default neutral (headless), plus the injector -----------------------------
namespace {
// The built-in backend: no host device. Reports a neutral, disconnected pad so the HLE contract is
// fully defined with zero dependencies. A frontend replaces this via pad_set_backend().
struct NeutralPadBackend : PadBackend {
    bool poll(int /*index*/, HostPadState& out) override { out = HostPadState{}; return false; }
};
NeutralPadBackend        g_neutral;
std::atomic<PadBackend*> g_backend{&g_neutral};   // a frontend may install from another thread
} // namespace

void pad_set_backend(PadBackend* backend) { g_backend.store(backend ? backend : &g_neutral); }
PadBackend* pad_backend() { return g_backend.load(); }

void pad_fill_controller_info(ScePadControllerInformation* out, bool connected, uint8_t connected_count) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));

    // DualSense-class touchpad geometry (Kyty reference values).
    out->touch_pixel_density = 44.86f;
    out->touch_resolution_x  = 1920;
    out->touch_resolution_y  = 1080;
    // ~8000/32768 of full range, minus the 0x80 center, as a symmetric 8-bit dead zone.
    out->stick_dead_zone_left  = (uint8_t)(pad_axis_u8(8000, -32768, 32767) - 0x80);
    out->stick_dead_zone_right = out->stick_dead_zone_left;
    out->connection_type = 0;       // 0 = local (wired/BT), not remote-play
    out->connected_count = connected_count;
    out->connected       = connected ? 1 : 0;
    out->device_class    = 0;       // SCE_PAD_DEVICE_CLASS_STANDARD
}

// --- Scripted input (PROSPER_PAD_SCRIPT) pure helpers — see pad.hpp -----------------------------
uint32_t pad_button_by_name(const std::string& n) {
    if (n == "start" || n == "options") return SCE_PAD_BUTTON_OPTIONS;
    if (n == "cross" || n == "x")       return SCE_PAD_BUTTON_CROSS;
    if (n == "circle" || n == "o")      return SCE_PAD_BUTTON_CIRCLE;
    if (n == "square")                  return SCE_PAD_BUTTON_SQUARE;
    if (n == "triangle")                return SCE_PAD_BUTTON_TRIANGLE;
    if (n == "up")                      return SCE_PAD_BUTTON_UP;
    if (n == "down")                    return SCE_PAD_BUTTON_DOWN;
    if (n == "left")                    return SCE_PAD_BUTTON_LEFT;
    if (n == "right")                   return SCE_PAD_BUTTON_RIGHT;
    if (n == "l1") return SCE_PAD_BUTTON_L1; if (n == "r1") return SCE_PAD_BUTTON_R1;
    if (n == "l2") return SCE_PAD_BUTTON_L2; if (n == "r2") return SCE_PAD_BUTTON_R2;
    if (n == "l3") return SCE_PAD_BUTTON_L3; if (n == "r3") return SCE_PAD_BUTTON_R3;
    return 0;
}

std::vector<PadScriptEntry> parse_pad_script(const std::string& spec) {
    std::vector<PadScriptEntry> v;
    size_t i = 0;
    while (i < spec.size()) {
        size_t semi = spec.find(';', i);
        std::string tok = spec.substr(i, semi == std::string::npos ? std::string::npos : semi - i);
        i = (semi == std::string::npos) ? spec.size() : semi + 1;
        size_t colon = tok.find(':');
        if (colon == std::string::npos) continue;
        double t = atof(tok.substr(0, colon).c_str());
        std::string btns = tok.substr(colon + 1);
        uint32_t mask = 0;
        size_t j = 0;
        while (j < btns.size()) {                       // '+'-separated buttons pressed together
            size_t plus = btns.find('+', j);
            std::string one = btns.substr(j, plus == std::string::npos ? std::string::npos : plus - j);
            j = (plus == std::string::npos) ? btns.size() : plus + 1;
            mask |= pad_button_by_name(one);
        }
        if (mask) v.push_back({t, mask});
    }
    return v;
}

uint32_t pad_script_buttons_at(const std::vector<PadScriptEntry>& script, double elapsed_secs, double hold_secs) {
    uint32_t mask = 0;
    for (const auto& e : script)
        if (elapsed_secs >= e.t_secs && elapsed_secs < e.t_secs + hold_secs) mask |= e.button_mask;
    return mask;
}

} // namespace prosper::input
