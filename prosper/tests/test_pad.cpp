// test_pad — libScePad input contract (pure, no hardware).
// Verifies: (1) the ScePadData / ScePadControllerInformation byte layout, (2) the HostPadState ->
// ScePadData mapping (buttons, sticks, triggers, connected, timestamp, identity orientation),
// (3) axis normalization + dead-zone math, and (4) the HLE functions are registered and fill the
// FULL 120-byte struct (the old stub only wrote 48 bytes, leaving connected/timestamp garbage).
#include "../src/input/pad.hpp"
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

using namespace prosper;
using namespace prosper::input;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int main() {
    printf("== test_pad ==\n");

    // (1) Layout — the exact Sony/Kyty ScePadData.
    CHECK(sizeof(ScePadData) == 120, "sizeof(ScePadData) == 120");
    CHECK(offsetof(ScePadData, left_stick_x) == 0x04, "left_stick_x @ 0x04");
    CHECK(offsetof(ScePadData, analog_buttons_l2) == 0x08, "analog_buttons_l2 @ 0x08");
    CHECK(offsetof(ScePadData, orientation_x) == 0x0c, "orientation_x @ 0x0c");
    CHECK(offsetof(ScePadData, connected) == 0x4c, "connected @ 0x4c");
    CHECK(offsetof(ScePadData, timestamp) == 0x50, "timestamp @ 0x50");
    CHECK(offsetof(ScePadData, connected_count) == 0x68, "connected_count @ 0x68");
    // Sony's real ScePadControllerInformation is 0x1c = 28 bytes (deviceClass@0x10 + reserve[8]),
    // matched by shadPS4/Kyty AND by the live guest, whose compiler placed its stack canary exactly
    // 0x1c after the struct. A 32-byte struct overran that 28-byte buffer -> canary crash (#283).
    CHECK(sizeof(ScePadControllerInformation) == 28, "sizeof(ScePadControllerInformation) == 28");
    CHECK(offsetof(ScePadControllerInformation, device_class) == 0x10, "device_class @ 0x10");
    CHECK(sizeof(ScePadExtendedControllerInformation) == 44,
          "sizeof(ScePadExtendedControllerInformation) == 44");
    CHECK(offsetof(ScePadExtendedControllerInformation, pad_type1) == 0x1c,
          "extended pad_type1 @ 0x1c");
    CHECK(offsetof(ScePadExtendedControllerInformation, capability) == 0x20,
          "extended capability @ 0x20");
    CHECK(offsetof(ScePadExtendedControllerInformation, class_data) == 0x24,
          "extended class_data @ 0x24");

    // (2) Neutral state -> centered sticks, zero buttons, released triggers, identity orientation.
    {
        HostPadState s;   // defaults
        ScePadData d;
        memset(&d, 0xAA, sizeof(d));   // poison to prove every field is written
        pad_fill_data(&d, s, 0, 0);
        CHECK(d.buttons == 0, "neutral: buttons == 0");
        CHECK(d.left_stick_x == 0x80 && d.left_stick_y == 0x80, "neutral: left stick centered");
        CHECK(d.right_stick_x == 0x80 && d.right_stick_y == 0x80, "neutral: right stick centered");
        CHECK(d.analog_buttons_l2 == 0 && d.analog_buttons_r2 == 0, "neutral: triggers released");
        CHECK(d.orientation_w == 1.0f, "neutral: identity orientation w=1");
        CHECK(d.orientation_x == 0.0f && d.acceleration_x == 0.0f, "neutral: accel/orient zeroed");
        CHECK(d.connected == 0, "neutral: not connected");
        CHECK(d.touch0_id == 1 && d.touch1_id == 2, "touch slot ids 1/2");
        CHECK(d.device_unique_data_len == 0, "device_unique_data_len 0 (poison overwritten)");
    }

    // (3) A pressed state -> exact button bits + stick/trigger bytes + connected + timestamp.
    {
        HostPadState s;
        s.buttons = SCE_PAD_BUTTON_CROSS | SCE_PAD_BUTTON_L2 | SCE_PAD_BUTTON_UP;
        s.left_x = 0x00; s.left_y = 0xFF;      // left stick: full left (x=0), full down (y=0xff)
        s.right_x = 0x40; s.right_y = 0xC0;
        s.l2 = 0xFF; s.r2 = 0x10;
        s.connected = true;
        ScePadData d;
        pad_fill_data(&d, s, 0x123456789ABCull, 1);
        CHECK(d.buttons == (SCE_PAD_BUTTON_CROSS | SCE_PAD_BUTTON_L2 | SCE_PAD_BUTTON_UP),
              "pressed: button bitmap exact");
        CHECK(d.left_stick_x == 0x00 && d.left_stick_y == 0xFF, "pressed: left stick bytes");
        CHECK(d.right_stick_x == 0x40 && d.right_stick_y == 0xC0, "pressed: right stick bytes");
        CHECK(d.analog_buttons_l2 == 0xFF && d.analog_buttons_r2 == 0x10, "pressed: trigger bytes");
        CHECK(d.connected == 1, "pressed: connected");
        CHECK(d.timestamp == 0x123456789ABCull, "pressed: timestamp copied");
        CHECK(d.connected_count == 1, "pressed: connected_count");
    }

    // (4) Axis normalization + dead zone.
    CHECK(pad_axis_u8(-32768, -32768, 32767) == 0,   "axis min -> 0");
    CHECK(pad_axis_u8(32767, -32768, 32767) == 255,  "axis max -> 255");
    CHECK(pad_axis_u8(0, -32768, 32767) == 127,      "axis center -> ~127");
    CHECK(pad_axis_u8(100, 0, 0) == 0x80,            "degenerate range -> center 0x80");
    CHECK(pad_axis_u8(-99999, -32768, 32767) == 0,   "axis below min clamps to 0");

    // (4b) Analog trigger -> digital L2/R2 bit (thresholded so rest/noise doesn't trip it).
    CHECK(pad_trigger_buttons(0, 0) == 0, "triggers at rest -> no digital bit");
    CHECK(pad_trigger_buttons(kPadTriggerButtonThreshold, 0) == 0, "trigger at threshold -> no bit");
    CHECK(pad_trigger_buttons(0xFF, 0) == SCE_PAD_BUTTON_L2, "L2 full -> L2 bit only");
    CHECK(pad_trigger_buttons(0, 0xFF) == SCE_PAD_BUTTON_R2, "R2 full -> R2 bit only");
    CHECK(pad_trigger_buttons(0xFF, 0xFF) == (SCE_PAD_BUTTON_L2 | SCE_PAD_BUTTON_R2), "both -> both bits");

    // (5) Controller information.
    {
        ScePadControllerInformation ci;
        memset(&ci, 0xAA, sizeof(ci));
        pad_fill_controller_info(&ci, true, 1);
        CHECK(ci.touch_resolution_x == 1920 && ci.touch_resolution_y == 1080, "info: touch resolution");
        CHECK(ci.connected == 1, "info: connected");
        CHECK(ci.connected_count == 1, "info: connected_count");
        CHECK(ci.device_class == 0, "info: standard device class");
        CHECK(ci.stick_dead_zone_left == ci.stick_dead_zone_right, "info: symmetric dead zone");

        ScePadExtendedControllerInformation ext;
        memset(&ext, 0xAA, sizeof(ext));
        pad_fill_extended_controller_info(&ext, true, 1);
        CHECK(ext.base.connected == 1 && ext.base.connected_count == 1,
              "extended info: populated controller-info base");
        bool extension_zero = true;
        const auto* extension_bytes = reinterpret_cast<const uint8_t*>(&ext.pad_type1);
        for (size_t i = 0; i < sizeof(ext) -
                                   offsetof(ScePadExtendedControllerInformation, pad_type1);
             ++i)
            extension_zero &= extension_bytes[i] == 0;
        CHECK(extension_zero, "extended info: standard-pad extension zeroed");
    }

    // (6) HLE registration + full-struct fill. The built-in NeutralPadBackend presents one connected
    // controller for development/testing; the short route also verifies input-read counter ownership.
    {
        // Explicit one-read windows make counter movement visible. Metadata queries below must not
        // consume either window; the first two successful input reads must see p0 and p1 in order.
        set_test_env("PROSPER_PAD_SCRIPT", "0-0.05:triangle;p0-1:cross;p1-2:options");
        register_builtin_hle();
        CHECK(nid_hash("scePadGetExtControllerInformation") == "hGbf2QTBmqc",
              "scePadGetExtControllerInformation hashes to its public NID");
        HleFn read_state = Hle::lookup(nid_hash("scePadReadState"));
        HleFn read       = Hle::lookup(nid_hash("scePadRead"));
        HleFn get_info   = Hle::lookup(nid_hash("scePadGetControllerInformation"));
        HleFn get_ext_info = Hle::lookup(nid_hash("scePadGetExtControllerInformation"));
        HleFn open       = Hle::lookup(nid_hash("scePadOpen"));
        HleFn open_ext   = Hle::lookup(nid_hash("scePadOpenExt"));
        HleFn close      = Hle::lookup(nid_hash("scePadClose"));
        HleFn is_valid   = Hle::lookup(nid_hash("scePadIsValidHandle"));
        CHECK(read_state && read && get_info && get_ext_info && open && open_ext && close && is_valid,
              "pad HLE functions registered");

        const uint64_t handle = open ? open(1, 0, 0, 0, 0, 0) : 0;
        const uint64_t ext_handle = open_ext ? open_ext(1, 0, 1, 0, 0, 0) : 0;
        CHECK(handle >= 1, "scePadOpen -> positive handle");
        CHECK(ext_handle >= 1 && ext_handle != handle,
              "scePadOpenExt -> distinct positive handle");
        if (is_valid) {
            CHECK(is_valid(handle, 0, 0, 0, 0, 0) == 1,
                  "scePadIsValidHandle accepts the opened handle");
            CHECK(is_valid(ext_handle, 0, 0, 0, 0, 0) == 1,
                  "scePadIsValidHandle accepts the OpenExt handle");
            CHECK(is_valid(handle + 0x1000, 0, 0, 0, 0, 0) == 0,
                  "scePadIsValidHandle rejects an unknown positive handle");
        }

        constexpr uint64_t invalid_handle = 0xffffffff80920003ull;
        const uint64_t unknown = handle + 0x1000;
        if (close)
            CHECK(close(unknown, 0, 0, 0, 0, 0) == invalid_handle,
                  "scePadClose rejects an unknown handle");

        // Invalid-handle failures happen before polling or touching output. The later valid reads
        // therefore still consume the scripted p0 and p1 windows in order.
        ScePadData invalid_data;
        std::memset(&invalid_data, 0x5a, sizeof invalid_data);
        ScePadControllerInformation invalid_info;
        std::memset(&invalid_info, 0x6b, sizeof invalid_info);
        ScePadExtendedControllerInformation invalid_ext_info;
        std::memset(&invalid_ext_info, 0x4d, sizeof invalid_ext_info);
        if (read_state)
            CHECK(read_state(unknown, (uint64_t)(uintptr_t)&invalid_data, 0, 0, 0, 0) ==
                      invalid_handle,
                  "scePadReadState rejects an unknown handle");
        if (read)
            CHECK(read(unknown, (uint64_t)(uintptr_t)&invalid_data, 1, 0, 0, 0) ==
                      invalid_handle,
                  "scePadRead rejects an unknown handle");
        if (get_info)
            CHECK(get_info(unknown, (uint64_t)(uintptr_t)&invalid_info, 0, 0, 0, 0) ==
                      invalid_handle,
                  "scePadGetControllerInformation rejects an unknown handle");
        if (get_ext_info)
            CHECK(get_ext_info(unknown, (uint64_t)(uintptr_t)&invalid_ext_info, 0, 0, 0, 0) ==
                      invalid_handle,
                  "scePadGetExtControllerInformation rejects an unknown handle");
        bool invalid_data_untouched = true;
        const auto* invalid_data_bytes = reinterpret_cast<const uint8_t*>(&invalid_data);
        for (size_t i = 0; i < sizeof invalid_data; ++i)
            invalid_data_untouched &= invalid_data_bytes[i] == 0x5a;
        bool invalid_info_untouched = true;
        const auto* invalid_info_bytes = reinterpret_cast<const uint8_t*>(&invalid_info);
        for (size_t i = 0; i < sizeof invalid_info; ++i)
            invalid_info_untouched &= invalid_info_bytes[i] == 0x6b;
        bool invalid_ext_info_untouched = true;
        const auto* invalid_ext_info_bytes = reinterpret_cast<const uint8_t*>(&invalid_ext_info);
        for (size_t i = 0; i < sizeof invalid_ext_info; ++i)
            invalid_ext_info_untouched &= invalid_ext_info_bytes[i] == 0x4d;
        CHECK(invalid_data_untouched && invalid_info_untouched && invalid_ext_info_untouched,
              "invalid-handle Pad calls leave outputs untouched");

        if (get_info) {
            ScePadControllerInformation ci{};
            CHECK(get_info(handle, (uint64_t)(uintptr_t)&ci, 0, 0, 0, 0) == 0,
                  "scePadGetControllerInformation -> 0 (OK)");
            CHECK(ci.connected == 1, "info query sees script-driven controller connectivity");
        }
        if (get_ext_info) {
            struct {
                ScePadExtendedControllerInformation info;
                uint8_t canary[12];
            } guarded;
            std::memset(&guarded, 0xa7, sizeof guarded);
            CHECK(get_ext_info(handle, (uint64_t)(uintptr_t)&guarded.info, 0, 0, 0, 0) == 0,
                  "scePadGetExtControllerInformation -> 0 (OK)");
            CHECK(guarded.info.base.connected == 1 && guarded.info.base.device_class == 0,
                  "extended info reports a connected standard controller");
            bool extension_zero = true;
            const auto* extension_bytes =
                reinterpret_cast<const uint8_t*>(&guarded.info.pad_type1);
            for (size_t i = 0; i < sizeof guarded.info -
                                       offsetof(ScePadExtendedControllerInformation, pad_type1);
                 ++i)
                extension_zero &= extension_bytes[i] == 0;
            bool canary_untouched = true;
            for (uint8_t byte : guarded.canary) canary_untouched &= byte == 0xa7;
            CHECK(extension_zero && canary_untouched,
                  "extended info writes exactly 0x2c bytes and zeroes its extension");
        }

        // The legacy seconds origin is the first pad poll, including an information query. If the
        // origin moved to the first state read, the short Triangle window would incorrectly fire below.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (read) {
            ScePadData unused{};
            CHECK(read(handle, 0, 1, 0, 0, 0) == 0 &&
                  read(handle, (uint64_t)(uintptr_t)&unused, 0, 0, 0, 0) == 0,
                  "failed scePadRead calls do not consume input windows");
        }

        if (read_state) {
            ScePadData d;
            memset(&d, 0x5A, sizeof(d));
            uint64_t r = read_state(handle, (uint64_t)(uintptr_t)&d, 0, 0, 0, 0);
            CHECK(r == 0, "scePadReadState -> 0 (OK)");
            // The struct's TAIL (bytes the old 48-byte stub never touched) must now be written:
            CHECK(d.connected == 1, "readstate: one controller connected by default (dev/testing default)");
            CHECK(d.left_stick_x == 0x80, "readstate: sticks centered");
            CHECK(d.orientation_w == 1.0f, "readstate: identity orientation");
            CHECK(d.buttons == SCE_PAD_BUTTON_CROSS,
                  "readstate: metadata preserves seconds origin without consuming p0 input window");
        }

        if (read) {
            if (get_info) {
                ScePadControllerInformation ci{};
                get_info(handle, (uint64_t)(uintptr_t)&ci, 0, 0, 0, 0);
            }
            ScePadData d;
            uint64_t n = read(handle, (uint64_t)(uintptr_t)&d, 4 /*num*/, 0, 0, 0);
            CHECK(n == 1, "scePadRead -> 1 state");
            CHECK(d.buttons == SCE_PAD_BUTTON_OPTIONS,
                  "scePadRead: intervening metadata query did not consume p1 input window");
        }
        if (close && is_valid) {
            CHECK(close(handle, 0, 0, 0, 0, 0) == 0,
                  "scePadClose retires the opened handle");
            CHECK(close(ext_handle, 0, 0, 0, 0, 0) == 0,
                  "scePadClose retires the OpenExt handle");
            CHECK(is_valid(handle, 0, 0, 0, 0, 0) == 0,
                  "scePadIsValidHandle rejects a closed handle");
            CHECK(is_valid(ext_handle, 0, 0, 0, 0, 0) == 0,
                  "scePadIsValidHandle rejects a closed OpenExt handle");
            std::memset(&invalid_data, 0x7c, sizeof invalid_data);
            std::memset(&invalid_ext_info, 0x3e, sizeof invalid_ext_info);
            CHECK(read_state(handle, (uint64_t)(uintptr_t)&invalid_data, 0, 0, 0, 0) ==
                      invalid_handle && close(handle, 0, 0, 0, 0, 0) == invalid_handle,
                  "closed handles stay invalid for input and repeated close");
            CHECK(get_ext_info(handle, (uint64_t)(uintptr_t)&invalid_ext_info, 0, 0, 0, 0) ==
                      invalid_handle,
                  "scePadGetExtControllerInformation rejects a closed handle");
            bool closed_output_untouched = true;
            for (size_t i = 0; i < sizeof invalid_data; ++i)
                closed_output_untouched &= invalid_data_bytes[i] == 0x7c;
            bool closed_ext_output_untouched = true;
            for (size_t i = 0; i < sizeof invalid_ext_info; ++i)
                closed_ext_output_untouched &= invalid_ext_info_bytes[i] == 0x3e;
            CHECK(closed_output_untouched && closed_ext_output_untouched,
                  "closed-handle Pad calls leave output untouched");
        }
        set_test_env("PROSPER_PAD_SCRIPT", "");
    }

    // (7) PROSPER_PAD_SCRIPT pure helpers — parse + time-eval (drives menus headless, issue #163).
    {
        CHECK(pad_button_by_name("start")   == SCE_PAD_BUTTON_OPTIONS, "name: start -> OPTIONS");
        CHECK(pad_button_by_name("options") == SCE_PAD_BUTTON_OPTIONS, "name: options -> OPTIONS");
        CHECK(pad_button_by_name("cross")   == SCE_PAD_BUTTON_CROSS,   "name: cross -> CROSS");
        CHECK(pad_button_by_name("x")       == SCE_PAD_BUTTON_CROSS,   "name: x -> CROSS");
        CHECK(pad_button_by_name("r2")      == SCE_PAD_BUTTON_R2,      "name: r2 -> R2");
        CHECK(pad_button_by_name("up")      == SCE_PAD_BUTTON_UP,      "name: up -> UP");
        CHECK(pad_button_by_name("nonsense")== 0,                      "name: unknown -> 0");
        CHECK(pad_button_names(SCE_PAD_BUTTON_OPTIONS) == "options", "names: options is canonical");
        CHECK(pad_button_names(SCE_PAD_BUTTON_UP | SCE_PAD_BUTTON_CROSS) == "up+cross",
              "names: stable combined-button order");
        CHECK(pad_button_names(0).empty(), "names: neutral mask is empty");
        auto round_trip = parse_pad_script("f0:" +
            pad_button_names(SCE_PAD_BUTTON_LEFT | SCE_PAD_BUTTON_SQUARE | SCE_PAD_BUTTON_R1));
        CHECK(round_trip.size() == 1 && round_trip[0].button_mask ==
              (SCE_PAD_BUTTON_LEFT | SCE_PAD_BUTTON_SQUARE | SCE_PAD_BUTTON_R1),
              "names: recorded text round-trips through parser");

        PadRecordState flip_record;
        CHECK(flip_record.observe(9, 0).empty(), "record: initial neutral flip emits nothing");
        CHECK(flip_record.observe(10, SCE_PAD_BUTTON_CROSS).empty(),
              "record: flip press starts an interval");
        CHECK(flip_record.observe(12, SCE_PAD_BUTTON_CROSS).empty(),
              "record: unchanged flip state emits nothing");
        CHECK(flip_record.observe(15, SCE_PAD_BUTTON_OPTIONS) == "f10-15:cross\n",
              "record: flip transition closes the previous interval");
        CHECK(flip_record.observe(15, 0) == "f15-16:options\n",
              "record: same-flip release preserves a non-empty interval");

        PadRecordState read_record(PadRecordAxis::pad_read);
        CHECK(read_record.observe(3, SCE_PAD_BUTTON_UP | SCE_PAD_BUTTON_CROSS).empty(),
              "record: pad-read press starts an interval");
        const std::string read_interval = read_record.observe(5, 0);
        CHECK(read_interval == "p3-5:up+cross\n",
              "record: pad-read release emits a canonical p-range");
        auto recorded_read = parse_pad_script(read_interval);
        CHECK(recorded_read.size() == 1 && recorded_read[0].read_anchored &&
              recorded_read[0].t_secs == 3.0 && recorded_read[0].end == 5.0 &&
              recorded_read[0].button_mask == (SCE_PAD_BUTTON_UP | SCE_PAD_BUTTON_CROSS),
              "record: pad-read interval round-trips through the route parser");

        auto s = parse_pad_script("3:start;9.5:cross;16:up+cross");
        CHECK(s.size() == 3, "parse: 3 entries");
        CHECK(s[0].t_secs == 3.0 && s[0].button_mask == SCE_PAD_BUTTON_OPTIONS, "parse: entry0 t/mask");
        CHECK(s[1].t_secs == 9.5 && s[1].button_mask == SCE_PAD_BUTTON_CROSS,   "parse: fractional time");
        CHECK(s[2].button_mask == (SCE_PAD_BUTTON_UP | SCE_PAD_BUTTON_CROSS),   "parse: '+' combines buttons");
        const double hold = 0.30;

        auto analog = parse_pad_script(
            "1-3:circle+left-stick-left;2-4:left-stick-right+right-stick-up");
        CHECK(analog.size() == 2, "parse: stick directions are recognized actions");
        CHECK(analog[0].button_mask == SCE_PAD_BUTTON_CIRCLE &&
              analog[0].axis_mask == PAD_SCRIPT_LEFT_X && analog[0].left_x == -1,
              "parse: button and left-stick direction share an entry");
        CHECK(analog[1].axis_mask == (PAD_SCRIPT_LEFT_X | PAD_SCRIPT_RIGHT_Y) &&
              analog[1].left_x == 1 && analog[1].right_y == -1,
              "parse: independent stick axes combine");
        auto analog_state = pad_script_state_at(analog, 1.5, hold);
        CHECK(analog_state.button_mask == SCE_PAD_BUTTON_CIRCLE && analog_state.left_x == -1 &&
              analog_state.axis_mask == PAD_SCRIPT_LEFT_X,
              "eval: active stick direction and button are returned together");
        analog_state = pad_script_state_at(analog, 2.5, hold);
        CHECK(analog_state.button_mask == SCE_PAD_BUTTON_CIRCLE && analog_state.left_x == 0 &&
              analog_state.right_y == -1 &&
              analog_state.axis_mask == (PAD_SCRIPT_LEFT_X | PAD_SCRIPT_RIGHT_Y),
              "eval: overlapping opposing directions explicitly center their axis");
        HostPadState overlaid;
        overlaid.buttons = SCE_PAD_BUTTON_L1;
        overlaid.right_x = 0x22;
        pad_apply_script_state(overlaid, analog_state);
        CHECK(overlaid.buttons == (SCE_PAD_BUTTON_L1 | SCE_PAD_BUTTON_CIRCLE),
              "overlay: scripted buttons combine with backend buttons");
        CHECK(overlaid.left_x == 0x80 && overlaid.left_y == 0x80 &&
              overlaid.right_x == 0x22 && overlaid.right_y == 0x00,
              "overlay: active axes map to Sony bytes and unspecified axes remain unchanged");

        auto trigger_move = parse_pad_script("1-3:r2+left-stick-left");
        CHECK(trigger_move.size() == 1 && trigger_move[0].button_mask == SCE_PAD_BUTTON_R2 &&
              trigger_move[0].axis_mask == PAD_SCRIPT_LEFT_X && trigger_move[0].left_x == -1,
              "parse: R2 and left-stick movement share a route entry");
        const auto trigger_move_state = pad_script_state_at(trigger_move, 2.0, hold);
        CHECK(trigger_move_state.button_mask == SCE_PAD_BUTTON_R2 &&
              trigger_move_state.axis_mask == PAD_SCRIPT_LEFT_X &&
              trigger_move_state.left_x == -1,
              "eval: combined R2 and movement stay active inside their explicit range");

        // Malformed pieces are skipped, not fatal; an all-unknown entry drops out.
        auto s2 = parse_pad_script("nocolon;5:garbagebtn;7:down");
        CHECK(s2.size() == 1 && s2[0].button_mask == SCE_PAD_BUTTON_DOWN, "parse: skips malformed/empty entries");

        // Time-eval: a button is held only within [t, t+hold).
        CHECK(pad_script_buttons_at(s, 2.9, hold) == 0,                       "eval: before window -> none");
        CHECK(pad_script_buttons_at(s, 3.0, hold) == SCE_PAD_BUTTON_OPTIONS,  "eval: at window start -> pressed");
        CHECK(pad_script_buttons_at(s, 3.29, hold) == SCE_PAD_BUTTON_OPTIONS, "eval: within window -> pressed");
        CHECK(pad_script_buttons_at(s, 3.30, hold) == 0,                      "eval: at window end (exclusive) -> released");
        CHECK(pad_script_buttons_at(s, 16.1, hold) == (SCE_PAD_BUTTON_UP | SCE_PAD_BUTTON_CROSS), "eval: combined press");
        CHECK(parse_pad_script("").empty(), "parse: empty spec -> no entries");

        // Frame-anchored entries (#302): "f<N>:" fires at flip N, not wall-clock second N.
        auto fs = parse_pad_script("f300:cross;5:start");
        CHECK(fs.size() == 2, "frame: 2 entries");
        CHECK(fs[0].frame_anchored && fs[0].t_secs == 300.0 && fs[0].button_mask == SCE_PAD_BUTTON_CROSS,
              "frame: 'f300:cross' parses frame-anchored at 300");
        CHECK(!fs[1].frame_anchored && fs[1].t_secs == 5.0, "frame: '5:start' stays seconds-anchored");
        // Frame entry ignores wall-clock, fires in [f, f+frame_hold); seconds entry ignores frame_count.
        const int64_t fhold = 8;
        CHECK(pad_script_buttons_at(fs, /*secs*/999.0, hold, /*frame*/299, fhold) == 0,
              "frame: before flip window (even far in wall-time) -> none");
        CHECK(pad_script_buttons_at(fs, 999.0, hold, 300, fhold) == SCE_PAD_BUTTON_CROSS,
              "frame: at flip 300 -> pressed");
        CHECK(pad_script_buttons_at(fs, 999.0, hold, 307, fhold) == SCE_PAD_BUTTON_CROSS,
              "frame: within [300,308) -> pressed");
        CHECK(pad_script_buttons_at(fs, 999.0, hold, 308, fhold) == 0,
              "frame: at window end (exclusive) -> released");
        CHECK(pad_script_buttons_at(fs, 5.1, hold, /*no frame info*/-1, fhold) == SCE_PAD_BUTTON_OPTIONS,
              "frame: seconds entry still fires with frame_count=-1; frame entry does not");

        // Pad-read anchors (#302): "p<N>:" fires on successful state read N, independent of time/flips.
        auto reads = parse_pad_script("p100:cross;P120-125:options;7:circle;f40:square");
        CHECK(reads.size() == 4, "read: seconds/frame/read entries parse together");
        CHECK(reads[0].read_anchored && !reads[0].frame_anchored &&
              reads[0].t_secs == 100.0 && reads[0].button_mask == SCE_PAD_BUTTON_CROSS,
              "read: 'p100:cross' parses pad-read-anchored at 100");
        CHECK(reads[1].read_anchored && reads[1].t_secs == 120.0 && reads[1].end == 125.0,
              "read: explicit range preserves its exclusive end");
        const int64_t rhold = 4;
        CHECK(pad_script_buttons_at(reads, 999.0, hold, 999, fhold, 99, rhold) == 0,
              "read: before point window -> none despite unrelated time/frame values");
        CHECK(pad_script_buttons_at(reads, 999.0, hold, 999, fhold, 100, rhold) ==
                  SCE_PAD_BUTTON_CROSS,
              "read: point starts at the requested pad read");
        CHECK(pad_script_buttons_at(reads, 999.0, hold, 999, fhold, 103, rhold) ==
                  SCE_PAD_BUTTON_CROSS,
              "read: point uses the configured default read hold");
        CHECK(pad_script_buttons_at(reads, 999.0, hold, 999, fhold, 104, rhold) == 0,
              "read: point ends exclusively");
        CHECK(pad_script_buttons_at(reads, 0.0, hold, 0, fhold, 124, rhold) ==
                  SCE_PAD_BUTTON_OPTIONS,
              "read: explicit range is active before its end");
        CHECK(pad_script_buttons_at(reads, 0.0, hold, 0, fhold, 125, rhold) == 0,
              "read: explicit range ends exclusively");
        CHECK(pad_script_buttons_at(reads, 7.1, hold, -1, fhold, -1, rhold) ==
                  SCE_PAD_BUTTON_CIRCLE,
              "read: unavailable count suppresses read entries without affecting seconds entries");
        auto invalid_counts = parse_pad_script(
            "p1.5:cross;p9007199254740992:cross;f2.5:cross;"
            "p9007199254740990.5:cross;p1-9007199254740990.5:cross;p5:circle");
        CHECK(invalid_counts.size() == 1 && invalid_counts[0].read_anchored &&
              invalid_counts[0].t_secs == 5.0,
              "read: fractional or inexact count anchors are rejected before conversion");

        // File syntax: newlines/comments, explicit time/frame/read ranges, and @path loading.
        auto ranges = parse_pad_script("# checkpoint\n f10-20:cross # hold\n3-4.5:up+cross\n");
        CHECK(ranges.size() == 2, "route: comments/newlines parse two entries");
        CHECK(ranges[0].frame_anchored && ranges[0].t_secs == 10.0 && ranges[0].end == 20.0,
              "route: frame range preserves exclusive end");
        CHECK(!ranges[1].frame_anchored && ranges[1].t_secs == 3.0 && ranges[1].end == 4.5,
              "route: time range preserves exclusive end");
        CHECK(pad_script_buttons_at(ranges, 0.0, hold, 19, fhold) == SCE_PAD_BUTTON_CROSS,
              "route: frame range active before explicit end");
        CHECK(pad_script_buttons_at(ranges, 0.0, hold, 20, fhold) == 0,
              "route: frame range ends exclusively");
        CHECK(pad_script_buttons_at(ranges, 4.4, hold, -1, fhold) ==
              (SCE_PAD_BUTTON_UP | SCE_PAD_BUTTON_CROSS), "route: time range active");

        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto route_path = std::filesystem::temp_directory_path() /
                                ("prosper_test_pad_route_" + std::to_string(unique) + ".pad");
        { std::ofstream route(route_path); route << "# loaded route\nf7-12:options\np20-24:cross\n"; }
        std::string route_error;
        auto loaded = load_pad_script("@" + route_path.string(), &route_error);
        CHECK(route_error.empty() && loaded.size() == 2 && loaded[0].frame_anchored &&
              loaded[0].end == 12.0 && loaded[1].read_anchored && loaded[1].end == 24.0,
              "route: @path loads and parses flip/read entries");
        std::filesystem::remove(route_path);
        auto missing = load_pad_script("@/this/prosper/route/does/not/exist.pad", &route_error);
        CHECK(missing.empty() && !route_error.empty(), "route: missing @path reports an error");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
