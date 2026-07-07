// test_pad_sdl3 — headless self-test of the SDL3 gamepad frontend.
// Proves the full SDL path links, initializes, and honors the PadBackend contract with no device
// attached: install the backend, poll -> disconnected/neutral, and confirm it feeds the pure
// ScePadData mapping. Runs in CI with no physical controller (SDL yields zero gamepads).
#include "pad_sdl3.hpp"
#include "../../src/input/pad.hpp"
#include <cstdio>

using namespace prosper;
using namespace prosper::input;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_pad_sdl3 ==\n");

    bool installed = install_sdl3_pad_backend();
    CHECK(installed, "install_sdl3_pad_backend() (SDL gamepad init)");
    if (!installed) {
        // If SDL gamepad can't init in this environment, the core default must remain active and safe.
        HostPadState s; bool present = pad_backend()->poll(0, s);
        CHECK(!present && !s.connected, "fallback: neutral default backend still safe");
        printf(fails ? "== FAIL ==\n" : "== PASS (SDL unavailable, safe fallback) ==\n");
        return fails ? 1 : 0;
    }

    // Backend installed. With no controller attached, poll reports disconnected + neutral.
    HostPadState s;
    bool present = pad_backend()->poll(0, s);
    CHECK(!present, "poll with no device -> false");
    CHECK(!s.connected, "no device -> disconnected");
    CHECK(s.left_x == 0x80 && s.left_y == 0x80, "no device -> sticks centered");
    CHECK(s.l2 == 0 && s.r2 == 0, "no device -> triggers released");

    // Drive the pure mapping through the backend result: struct is fully formed.
    ScePadData d;
    pad_fill_data(&d, s, 0, 0);
    CHECK(d.connected == 0 && d.orientation_w == 1.0f, "mapping: disconnected + identity orientation");

    shutdown_sdl3_pad_backend();
    // After shutdown the neutral core default is restored.
    HostPadState s2; bool p2 = pad_backend()->poll(0, s2);
    CHECK(!p2 && !s2.connected, "after shutdown: neutral default restored");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
