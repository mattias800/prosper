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

using namespace prosper;
using namespace prosper::input;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

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
    CHECK(sizeof(ScePadControllerInformation) == 32, "sizeof(ScePadControllerInformation) == 32");
    CHECK(offsetof(ScePadControllerInformation, device_class) == 0x10, "device_class @ 0x10");

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
    }

    // (6) HLE registration + full-struct fill (no PROSPER_PAD -> neutral, but complete + connected=0).
    {
        register_builtin_hle();
        HleFn read_state = Hle::lookup(nid_hash("scePadReadState"));
        HleFn read       = Hle::lookup(nid_hash("scePadRead"));
        HleFn get_info   = Hle::lookup(nid_hash("scePadGetControllerInformation"));
        HleFn open       = Hle::lookup(nid_hash("scePadOpen"));
        CHECK(read_state && read && get_info && open, "pad HLE functions registered");

        if (open) CHECK(open(1, 0, 0, 0, 0, 0) >= 1, "scePadOpen -> positive handle");

        if (read_state) {
            ScePadData d;
            memset(&d, 0x5A, sizeof(d));
            uint64_t r = read_state(1, (uint64_t)(uintptr_t)&d, 0, 0, 0, 0);
            CHECK(r == 0, "scePadReadState -> 0 (OK)");
            // The struct's TAIL (bytes the old 48-byte stub never touched) must now be written:
            CHECK(d.connected == 0, "readstate: connected written (was uninit in old stub)");
            CHECK(d.left_stick_x == 0x80, "readstate: sticks centered");
            CHECK(d.orientation_w == 1.0f, "readstate: identity orientation");
        }

        if (read) {
            ScePadData d;
            uint64_t n = read(1, (uint64_t)(uintptr_t)&d, 4 /*num*/, 0, 0, 0);
            CHECK(n == 1, "scePadRead -> 1 state");
        }
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
