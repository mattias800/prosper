#pragma once

// Keyboard -> virtual DualSense mapping (#2234).
//
// This lives apart from main.cpp, and takes a key-query callback instead of reading SDL directly,
// so the mapping can be asserted key-by-key by a test that never links SDL. The frontend keeps only
// the Key -> SDL_Scancode table, which is flat enough to audit by eye; every decision with judgment
// in it -- which key deflects which stick, which face button sits under which finger, what a
// diagonal does -- is here, and is covered.
//
// Layout. Two clusters per hand, and a hand uses one at a time; that is the point of the layout
// rather than a side effect of it:
//
//     W A S D    D-pad up / left / down / right      (arrow keys do the same)
//     T F G H    LEFT stick  up / left / down / right
//     I J K L    RIGHT stick up / left / down / right
//     N M , .    Square / Cross / Circle / Triangle
//     Z X C V    L1 / L2 / R1 / R2
//     B    /     L3 / R3
//     Space      Cross      (under the thumb, and it costs nothing to keep)
//     Enter      Options    (what Start is called on PlayStation)
//
// The right stick had NO keyboard binding at all before this. `HostPadState` carried right_x/right_y
// and they reached `ScePadData` correctly, but nothing on the keyboard path ever wrote them, so they
// stayed centred for the whole run -- camera control was reachable only from a physical pad, which
// is most titles past rung 3. Blue Prince is first person: you could walk but you could not look.
//
// One change here is NOT additive. WASD used to drive the D-pad *and* the left stick together, so a
// stick-only title responded to WASD. They are separate now: WASD is the D-pad, TFGH is the stick.
// That is the intent of having a separate stick cluster, and it is also the one thing in this file
// that can make a title stop responding to keys that worked yesterday.

#include "input/pad.hpp"

#include <cstdint>

namespace prosper::frontend {

// Named by the PHYSICAL key, not by what it does, so the mapping below is the thing under test
// rather than a rename of its own answer.
enum class PadKey {
    W, A, S, D,
    T, F, G, H,
    I, J, K, L,
    N, M, Comma, Period,
    Z, X, C, V,
    B, Slash,
    Space, Enter,
    ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
    Count,
};

namespace detail {

// Sony's 8-bit stick convention: 0x80 centred, 0x00 up/left, 0xff down/right. A key is digital, so
// it gives full deflection -- the same thing the left stick already did before this file existed.
// Both directions held cancel, which is what a physical stick reports when it is not pushed.
inline uint8_t stick_axis(bool negative, bool positive) {
    if (negative == positive) return 0x80;
    return negative ? 0x00 : 0xff;
}

}  // namespace detail

// `down(k)` answers whether key `k` is currently held.
//
// `enter_maps_to_options` is false while the host owns Enter (Alt+Enter is the window's, and the
// launcher UI uses it before a guest boots), matching the behaviour this replaces.
template <typename KeyDown>
input::HostPadState map_keyboard_to_pad(KeyDown down, bool enter_maps_to_options) {
    using namespace input;
    const auto k = [&](PadKey key) { return static_cast<bool>(down(key)); };

    const bool dpad_up    = k(PadKey::W) || k(PadKey::ArrowUp);
    const bool dpad_down  = k(PadKey::S) || k(PadKey::ArrowDown);
    const bool dpad_left  = k(PadKey::A) || k(PadKey::ArrowLeft);
    const bool dpad_right = k(PadKey::D) || k(PadKey::ArrowRight);

    HostPadState st;
    uint32_t b = 0;
    if (dpad_up)    b |= SCE_PAD_BUTTON_UP;
    if (dpad_down)  b |= SCE_PAD_BUTTON_DOWN;
    if (dpad_left)  b |= SCE_PAD_BUTTON_LEFT;
    if (dpad_right) b |= SCE_PAD_BUTTON_RIGHT;

    // The face buttons are a diamond and N M , . is a row, so left-to-right is a convention either
    // way. Read the diamond left -> bottom -> right -> top, which puts Cross -- the usual confirm --
    // under the strongest finger.
    if (k(PadKey::N))      b |= SCE_PAD_BUTTON_SQUARE;
    if (k(PadKey::M) || k(PadKey::Space)) b |= SCE_PAD_BUTTON_CROSS;
    if (k(PadKey::Comma))  b |= SCE_PAD_BUTTON_CIRCLE;
    if (k(PadKey::Period)) b |= SCE_PAD_BUTTON_TRIANGLE;

    // Shoulders sit on the left hand because the right hand owns IJKL and N/M/,/. at the same time.
    // Not ideal, and not avoidable given that.
    if (k(PadKey::Z)) b |= SCE_PAD_BUTTON_L1;
    if (k(PadKey::X)) b |= SCE_PAD_BUTTON_L2;
    if (k(PadKey::C)) b |= SCE_PAD_BUTTON_R1;
    if (k(PadKey::V)) b |= SCE_PAD_BUTTON_R2;

    // L3/R3 are commonly sprint and crouch in 3D titles, and were unreachable from the keyboard for
    // the same reason the right stick was. B and / are free and sit at the outside of each cluster.
    if (k(PadKey::B))     b |= SCE_PAD_BUTTON_L3;
    if (k(PadKey::Slash)) b |= SCE_PAD_BUTTON_R3;

    if (k(PadKey::Enter) && enter_maps_to_options) b |= SCE_PAD_BUTTON_OPTIONS;

    st.buttons = b;
    st.left_x  = detail::stick_axis(k(PadKey::F), k(PadKey::H));
    st.left_y  = detail::stick_axis(k(PadKey::T), k(PadKey::G));
    st.right_x = detail::stick_axis(k(PadKey::J), k(PadKey::L));
    st.right_y = detail::stick_axis(k(PadKey::I), k(PadKey::K));

    // The triggers are also analog. A key gives the full pull, and it must agree with the L2/R2
    // button bits above -- a title may read either, and disagreeing between them is a bug the guest
    // cannot see around.
    st.l2 = k(PadKey::X) ? 255 : 0;
    st.r2 = k(PadKey::V) ? 255 : 0;

    st.connected = true;
    return st;
}

}  // namespace prosper::frontend
