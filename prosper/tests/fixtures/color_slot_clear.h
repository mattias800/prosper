// color_slot_clear.h -- what a colour slot above 1 starts from when nothing is loaded into it.
//
// Slots 0 and 1 are told their starting colour by the caller of render_draws_rgba (its two clear
// arguments). Slots 2..7 had no such argument: they started from the first clear colour a draw of
// the pass programmed for the slot, or opaque black. A target the guest had fast-cleared through
// its DCC metadata therefore started a pass from the wrong colour whenever it was bound above
// slot 1 (#4624) -- which is where an engine's G-buffer attachments live.
//
// In order: the retained uniform colour the caller passed for the slot; the pass's own programmed
// clear for it; opaque black. Split out of render_runner.h so the order can be tested without a
// device.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>

namespace prosper::test {

// `uniform` is the caller's colour for the slot (four floats) or null. `Draws` is any range of
// draws whose `ps` points at a state with `color_targets[slot].has_clear` / `.clear`.
template <typename Draws>
std::array<float, 4> color_slot_clear_value(const float* uniform, const Draws& draws,
                                            uint32_t slot) {
    std::array<float, 4> value{0.0f, 0.0f, 0.0f, 1.0f};
    if (uniform) {
        std::copy_n(uniform, value.size(), value.begin());
        return value;
    }
    for (const auto& draw : draws) {
        if (!draw.ps || !draw.ps->color_targets[slot].has_clear) continue;
        std::copy(std::begin(draw.ps->color_targets[slot].clear),
                  std::end(draw.ps->color_targets[slot].clear), value.begin());
        break;
    }
    return value;
}

}   // namespace prosper::test
