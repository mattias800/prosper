// Keyboard -> virtual DualSense mapping (#2234).
//
// The mapping is a contract with the player's fingers, so these arms name the KEY and the CONTROL,
// not the internals. Two things here are worth more than the rest:
//
//   * the right stick is reachable at all -- it had no keyboard binding before #2234, so camera
//     control was physical-pad-only and a first-person title could be walked but not looked around;
//   * WASD no longer moves the left stick. That is the one non-additive change in the layout, so it
//     is asserted in both directions rather than left to be discovered in a title that stops moving.

#include "keyboard_pad_map.hpp"

#include <cstdio>
#include <initializer_list>

using namespace prosper::input;
using prosper::frontend::PadKey;
using prosper::frontend::map_keyboard_to_pad;

namespace {

int failures = 0;
#define CHECK(expr) do { if (!(expr)) { std::printf("FAIL line %d: %s\n", __LINE__, #expr); ++failures; } } while (0)

// The state produced by holding exactly `held` and nothing else.
HostPadState press(std::initializer_list<PadKey> held, bool enter_maps_to_options = true) {
    return map_keyboard_to_pad(
        [&](PadKey k) {
            for (PadKey h : held) if (h == k) return true;
            return false;
        },
        enter_maps_to_options);
}

bool centred(const HostPadState& s) {
    return s.left_x == 0x80 && s.left_y == 0x80 && s.right_x == 0x80 && s.right_y == 0x80;
}

} // namespace

int main() {
    // --- nothing held ---------------------------------------------------------------------------
    {
        const HostPadState idle = press({});
        CHECK(idle.buttons == 0);
        CHECK(centred(idle));
        CHECK(idle.l2 == 0 && idle.r2 == 0);
        // The keyboard pad is always "present": the overlay composes it over a physical pad, and a
        // title that polls for a connected controller must find one whether or not a key is down.
        CHECK(idle.connected);
    }

    // --- the right stick, which is the gap this closes -------------------------------------------
    CHECK(press({PadKey::I}).right_y == 0x00);   // up
    CHECK(press({PadKey::K}).right_y == 0xff);   // down
    CHECK(press({PadKey::J}).right_x == 0x00);   // left
    CHECK(press({PadKey::L}).right_x == 0xff);   // right
    // A diagonal deflects both axes; a camera that can only be moved on one axis at a time is not
    // a camera.
    {
        const HostPadState diag = press({PadKey::I, PadKey::L});
        CHECK(diag.right_y == 0x00 && diag.right_x == 0xff);
    }
    // Opposite keys cancel, the way a physical stick reports when it is not pushed.
    CHECK(press({PadKey::I, PadKey::K}).right_y == 0x80);
    CHECK(press({PadKey::J, PadKey::L}).right_x == 0x80);
    // And the right stick must not disturb the left, or a title reading both gets a phantom walk.
    {
        const HostPadState right_only = press({PadKey::I, PadKey::J});
        CHECK(right_only.left_x == 0x80 && right_only.left_y == 0x80);
        CHECK(right_only.buttons == 0);
    }

    // --- the left stick moved to TFGH ------------------------------------------------------------
    CHECK(press({PadKey::T}).left_y == 0x00);
    CHECK(press({PadKey::G}).left_y == 0xff);
    CHECK(press({PadKey::F}).left_x == 0x00);
    CHECK(press({PadKey::H}).left_x == 0xff);
    CHECK(press({PadKey::T, PadKey::G}).left_y == 0x80);
    CHECK(press({PadKey::F, PadKey::H}).left_x == 0x80);
    // H used to be R2. If it still were, walking right would fire a trigger.
    CHECK(press({PadKey::H}).buttons == 0);
    CHECK(press({PadKey::H}).r2 == 0);

    // --- the D-pad, and the split that is not additive -------------------------------------------
    CHECK(press({PadKey::W}).buttons == SCE_PAD_BUTTON_UP);
    CHECK(press({PadKey::S}).buttons == SCE_PAD_BUTTON_DOWN);
    CHECK(press({PadKey::A}).buttons == SCE_PAD_BUTTON_LEFT);
    CHECK(press({PadKey::D}).buttons == SCE_PAD_BUTTON_RIGHT);
    // Arrows are the same D-pad, so a player who reaches for them is not left without one.
    CHECK(press({PadKey::ArrowUp}).buttons    == SCE_PAD_BUTTON_UP);
    CHECK(press({PadKey::ArrowDown}).buttons  == SCE_PAD_BUTTON_DOWN);
    CHECK(press({PadKey::ArrowLeft}).buttons  == SCE_PAD_BUTTON_LEFT);
    CHECK(press({PadKey::ArrowRight}).buttons == SCE_PAD_BUTTON_RIGHT);
    // THE behaviour change: WASD and the arrows drive the D-pad ONLY. Before #2234 they drove the
    // left stick too, so a stick-only title responded to WASD and now will not.
    CHECK(centred(press({PadKey::W})));
    CHECK(centred(press({PadKey::A, PadKey::S, PadKey::D})));
    CHECK(centred(press({PadKey::ArrowUp, PadKey::ArrowLeft})));
    // ...and the converse, so the split cannot rot back together from the other side.
    CHECK(press({PadKey::T, PadKey::F, PadKey::G, PadKey::H}).buttons == 0);

    // --- face buttons: the diamond read left -> bottom -> right -> top ---------------------------
    CHECK(press({PadKey::N}).buttons      == SCE_PAD_BUTTON_SQUARE);
    CHECK(press({PadKey::M}).buttons      == SCE_PAD_BUTTON_CROSS);
    CHECK(press({PadKey::Comma}).buttons  == SCE_PAD_BUTTON_CIRCLE);
    CHECK(press({PadKey::Period}).buttons == SCE_PAD_BUTTON_TRIANGLE);
    CHECK(press({PadKey::Space}).buttons  == SCE_PAD_BUTTON_CROSS);   // kept: it is under the thumb
    CHECK(press({PadKey::M, PadKey::Space}).buttons == SCE_PAD_BUTTON_CROSS);  // not doubled

    // --- shoulders, and the trigger analog that must agree with the button bit --------------------
    CHECK(press({PadKey::Z}).buttons == SCE_PAD_BUTTON_L1);
    CHECK(press({PadKey::C}).buttons == SCE_PAD_BUTTON_R1);
    CHECK(press({PadKey::Z}).l2 == 0 && press({PadKey::C}).r2 == 0);
    {
        // A title may read the L2 button bit or the analog travel. Reporting a pressed trigger with
        // zero travel -- or travel with no bit -- is a disagreement the guest cannot see around.
        const HostPadState l2 = press({PadKey::X});
        CHECK(l2.buttons == SCE_PAD_BUTTON_L2 && l2.l2 == 255 && l2.r2 == 0);
        const HostPadState r2 = press({PadKey::V});
        CHECK(r2.buttons == SCE_PAD_BUTTON_R2 && r2.r2 == 255 && r2.l2 == 0);
    }

    // --- stick clicks, unreachable from the keyboard for the same reason the right stick was ------
    CHECK(press({PadKey::B}).buttons     == SCE_PAD_BUTTON_L3);
    CHECK(press({PadKey::Slash}).buttons == SCE_PAD_BUTTON_R3);

    // --- Enter is Options, except while the host owns Enter ---------------------------------------
    CHECK(press({PadKey::Enter}).buttons == SCE_PAD_BUTTON_OPTIONS);
    CHECK(press({PadKey::Enter}, /*enter_maps_to_options=*/false).buttons == 0);

    // --- no key in the enum is dead ---------------------------------------------------------------
    // A key added to PadKey but never wired reaches the frontend's scancode table and the README and
    // still does nothing. Holding any one key alone must change SOMETHING about the pad.
    for (int i = 0; i < static_cast<int>(PadKey::Count); ++i) {
        const PadKey k = static_cast<PadKey>(i);
        const HostPadState s = press({k});
        const bool moved = s.buttons != 0 || !centred(s) || s.l2 != 0 || s.r2 != 0;
        if (!moved) { std::printf("FAIL: PadKey %d is mapped to nothing\n", i); ++failures; }
    }

    if (failures) std::printf("FAIL: %d\n", failures);
    else std::printf("PASS\n");
    return failures ? 1 : 0;
}
