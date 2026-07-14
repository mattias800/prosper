// pad.cpp — pure mapping core for the libScePad HLE (see pad.hpp).
// No hardware, no globals, no platform code — just HostPadState -> Sony ABI structs. The byte
// layout is asserted here so a struct-layout regression fails at compile time, not silently in the
// game. The backend lives in pad_evdev.cpp.
#include "pad.hpp"
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <cmath>
#include <fstream>
#include <sstream>

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

static_assert(sizeof(ScePadControllerInformation) == 28, "ScePadControllerInformation must be 28 bytes (Sony reserve[8], #283)");
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
// The built-in backend: no host device, but reports ONE connected controller with neutral input.
// A PS5 title legitimately gates progression on a pad being present (e.g. PPSA02664 loops a
// "Please, connect a controller to continue." message dialog forever while none is connected), so the
// headless/dev-testing default must present one — otherwise the game never reaches its title/gameplay.
// Input stays neutral (no buttons); PROSPER_PAD_PRESS / PROSPER_PAD_SCRIPT drive actual presses on top.
// A real frontend (SDL3 / evdev) replaces this via pad_set_backend(); the harness app will manage the
// connected-controller count there. CONFIDENCE: HIGH (controller-presence gate is live-observed).
struct NeutralPadBackend : PadBackend {
    bool poll(int /*index*/, HostPadState& out) override { out = HostPadState{}; out.connected = true; return true; }
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

std::string pad_button_names(uint32_t mask) {
    static constexpr struct { uint32_t bit; const char* name; } names[] = {
        {SCE_PAD_BUTTON_UP, "up"}, {SCE_PAD_BUTTON_DOWN, "down"},
        {SCE_PAD_BUTTON_LEFT, "left"}, {SCE_PAD_BUTTON_RIGHT, "right"},
        {SCE_PAD_BUTTON_CROSS, "cross"}, {SCE_PAD_BUTTON_CIRCLE, "circle"},
        {SCE_PAD_BUTTON_SQUARE, "square"}, {SCE_PAD_BUTTON_TRIANGLE, "triangle"},
        {SCE_PAD_BUTTON_L1, "l1"}, {SCE_PAD_BUTTON_R1, "r1"},
        {SCE_PAD_BUTTON_L2, "l2"}, {SCE_PAD_BUTTON_R2, "r2"},
        {SCE_PAD_BUTTON_L3, "l3"}, {SCE_PAD_BUTTON_R3, "r3"},
        {SCE_PAD_BUTTON_OPTIONS, "options"},
    };
    std::string result;
    for (const auto& name : names) {
        if (!(mask & name.bit)) continue;
        if (!result.empty()) result += '+';
        result += name.name;
    }
    return result;
}

std::vector<PadScriptEntry> parse_pad_script(const std::string& spec) {
    std::vector<PadScriptEntry> v;
    size_t i = 0;
    while (i < spec.size()) {
        size_t sep = spec.find_first_of(";\n", i);
        std::string tok = spec.substr(i, sep == std::string::npos ? std::string::npos : sep - i);
        i = (sep == std::string::npos) ? spec.size() : sep + 1;
        if (size_t comment = tok.find('#'); comment != std::string::npos) tok.erase(comment);
        auto trim = [](std::string s) {
            const size_t first = s.find_first_not_of(" \t\r\f\v");
            if (first == std::string::npos) return std::string{};
            return s.substr(first, s.find_last_not_of(" \t\r\f\v") - first + 1);
        };
        tok = trim(tok);
        if (tok.empty()) continue;
        size_t colon = tok.find(':');
        if (colon == std::string::npos) continue;
        std::string head = trim(tok.substr(0, colon));
        // A leading 'f' marks a FRAME-anchored entry ("f300:cross" -> flip 300); else it's seconds.
        bool frame_anchored = (!head.empty() && (head[0] == 'f' || head[0] == 'F'));
        std::string window = frame_anchored ? head.substr(1) : head;
        size_t dash = window.find('-', 1);
        auto parse_number = [](const std::string& text, double& out) {
            char* end = nullptr;
            const double value = strtod(text.c_str(), &end);
            if (end == text.c_str() || *end != '\0' || value < 0.0 || !std::isfinite(value)) return false;
            out = value;
            return true;
        };
        double start = 0.0, end = 0.0;
        if (!parse_number(trim(window.substr(0, dash)), start)) continue;
        if (dash != std::string::npos) {
            if (!parse_number(trim(window.substr(dash + 1)), end) || end <= start) continue;
        }
        std::string btns = tok.substr(colon + 1);
        uint32_t mask = 0;
        size_t j = 0;
        while (j < btns.size()) {                       // '+'-separated buttons pressed together
            size_t plus = btns.find('+', j);
            std::string one = trim(btns.substr(j, plus == std::string::npos ? std::string::npos : plus - j));
            j = (plus == std::string::npos) ? btns.size() : plus + 1;
            mask |= pad_button_by_name(one);
        }
        if (mask) v.push_back({start, mask, frame_anchored, end});
    }
    return v;
}

std::vector<PadScriptEntry> load_pad_script(const std::string& source, std::string* error) {
    if (error) error->clear();
    if (source.empty() || source[0] != '@') return parse_pad_script(source);
    const std::string path = source.substr(1);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (error) *error = path.empty() ? "route path is empty" : "cannot open route file: " + path;
        return {};
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    if (file.bad()) {
        if (error) *error = "cannot read route file: " + path;
        return {};
    }
    return parse_pad_script(contents.str());
}

uint32_t pad_script_buttons_at(const std::vector<PadScriptEntry>& script, double elapsed_secs, double hold_secs,
                               int64_t frame_count, int64_t frame_hold) {
    uint32_t mask = 0;
    for (const auto& e : script) {
        if (e.frame_anchored) {
            int64_t f = (int64_t)e.t_secs;   // frame number lives in t_secs for frame-anchored entries
            int64_t end = e.end > e.t_secs ? (int64_t)e.end : f + frame_hold;
            if (frame_count >= 0 && frame_count >= f && frame_count < end) mask |= e.button_mask;
        } else {
            double end = e.end > e.t_secs ? e.end : e.t_secs + hold_secs;
            if (elapsed_secs >= e.t_secs && elapsed_secs < end) mask |= e.button_mask;
        }
    }
    return mask;
}

} // namespace prosper::input
