#pragma once

#include "input/pad.hpp"

#include <mutex>

namespace prosper::frontend {

// Adds a keyboard-backed virtual pad 0 without hiding an installed physical-pad backend.
class KeyboardPadOverlay final : public input::PadBackend {
public:
    void set_fallback(input::PadBackend* backend) { fallback_ = backend; }

    void set_keyboard_state(const input::HostPadState& state) {
        std::lock_guard<std::mutex> lock(mutex_);
        keyboard_ = state;
    }

    bool poll(int index, input::HostPadState& out) override {
        const bool physical = fallback_ && fallback_ != this && fallback_->poll(index, out);
        if (index != 0) return physical;

        input::HostPadState keyboard;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            keyboard = keyboard_;
        }
        if (!physical) out = {};
        out.buttons |= keyboard.buttons;
        // A centred axis is "the keyboard is not asking", so it must not overwrite a physical
        // stick that IS deflected -- that is what makes this an overlay rather than a replacement.
        // The right stick had no keyboard source at all until #2234, and consequently no merge
        // here: adding it to the map alone would have left it centred, since this is where the
        // composed state is decided.
        if (keyboard.left_x  != 0x80) out.left_x  = keyboard.left_x;
        if (keyboard.left_y  != 0x80) out.left_y  = keyboard.left_y;
        if (keyboard.right_x != 0x80) out.right_x = keyboard.right_x;
        if (keyboard.right_y != 0x80) out.right_y = keyboard.right_y;
        if (keyboard.l2 > out.l2) out.l2 = keyboard.l2;
        if (keyboard.r2 > out.r2) out.r2 = keyboard.r2;
        out.connected = true;
        return true;
    }

private:
    std::mutex mutex_;
    input::HostPadState keyboard_;
    input::PadBackend* fallback_ = nullptr;
};

} // namespace prosper::frontend
