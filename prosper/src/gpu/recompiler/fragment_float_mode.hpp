#pragma once
#include <cstdint>

namespace prosper::gpu {

// The producing draw's SPI_SHADER_PGM_RSRC1_PS.FLOAT_MODE, not a host float-control
// capability or a replay request. Missing launch state is unknown, never a guessed mode.
struct FragmentFloatMode {
    bool available = false;
    uint8_t value = 0;

    constexpr bool canonical() const { return available || value == 0; }
    constexpr bool preserves_f32_inputs() const { return (value & 0x10u) != 0; }
    bool operator==(const FragmentFloatMode&) const = default;
};

} // namespace prosper::gpu
