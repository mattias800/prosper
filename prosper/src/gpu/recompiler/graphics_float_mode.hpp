// Producing pre-raster hardware stage state; unavailable never means an observed mode of zero.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct GraphicsFloatMode {
    bool available = false;
    uint8_t value = 0;
    bool ieee_mode = false;

    constexpr bool canonical() const { return available || (value == 0 && !ieee_mode); }
    bool operator==(const GraphicsFloatMode&) const = default;
};

// RDNA2 RSRC1_VS/GS/HS: FLOAT_MODE[19:12], IEEE_MODE[23]. ES and LS are front
// halves of the merged GS and HS hardware programs, not independent RSRC1 banks.
constexpr GraphicsFloatMode graphics_float_mode_from_rsrc1(uint32_t word) {
    return {true, static_cast<uint8_t>((word >> 12) & 0xff), (word & (1u << 23)) != 0};
}
} // namespace prosper::gpu
