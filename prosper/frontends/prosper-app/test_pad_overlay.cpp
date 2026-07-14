#include "pad_overlay.hpp"

#include <cstdio>

using namespace prosper::input;

namespace {

int failures = 0;
#define CHECK(expr) do { if (!(expr)) { std::printf("FAIL line %d: %s\n", __LINE__, #expr); ++failures; } } while (0)

struct PhysicalPads final : PadBackend {
    bool poll(int index, HostPadState& out) override {
        if (index > 1) return false;
        out = {};
        out.connected = true;
        out.buttons = index == 0 ? SCE_PAD_BUTTON_SQUARE : SCE_PAD_BUTTON_CIRCLE;
        out.left_x = 0x90;
        out.right_x = 0x22;
        out.l2 = 80;
        return true;
    }
};

} // namespace

int main() {
    prosper::frontend::KeyboardPadOverlay overlay;

    HostPadState state;
    CHECK(overlay.poll(0, state));
    CHECK(state.connected && state.buttons == 0 && state.left_x == 0x80);
    CHECK(!overlay.poll(1, state));

    PhysicalPads physical;
    overlay.set_fallback(&physical);
    HostPadState keyboard;
    keyboard.buttons = SCE_PAD_BUTTON_CROSS;
    keyboard.left_x = 0;
    keyboard.r2 = 120;
    overlay.set_keyboard_state(keyboard);

    CHECK(overlay.poll(0, state));
    CHECK(state.connected);
    CHECK(state.buttons == (SCE_PAD_BUTTON_SQUARE | SCE_PAD_BUTTON_CROSS));
    CHECK(state.left_x == 0 && state.right_x == 0x22);
    CHECK(state.l2 == 80 && state.r2 == 120);

    CHECK(overlay.poll(1, state));
    CHECK(state.buttons == SCE_PAD_BUTTON_CIRCLE && state.left_x == 0x90);
    CHECK(!overlay.poll(2, state));

    if (failures) std::printf("FAIL: %d\n", failures);
    else std::printf("PASS\n");
    return failures ? 1 : 0;
}
