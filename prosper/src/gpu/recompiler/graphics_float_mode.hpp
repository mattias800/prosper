// Producing pre-raster hardware stage state; unavailable never means an observed mode of zero.
#pragma once
#include "gpu/recompiler/fp16_overflow_mode.hpp"
#include <cstdint>
namespace prosper::gpu {
struct GraphicsFloatMode {
    bool available = false;
    uint8_t value = 0;
    bool ieee_mode = false;

    Fp16OverflowMode fp16_overflow{}; // independent of the older MODE/IEEE observation

    constexpr bool canonical() const {
        return (available || (value == 0 && !ieee_mode)) && fp16_overflow.canonical();
    }
    bool operator==(const GraphicsFloatMode&) const = default;
};

// RDNA2 RSRC1_VS/GS/HS: FLOAT_MODE[19:12], IEEE_MODE[23]. ES and LS are front
// halves of the merged GS and HS hardware programs, not independent RSRC1 banks.
constexpr GraphicsFloatMode graphics_float_mode_from_rsrc1(uint32_t word) {
    // No stage argument: HS and VS/GS use different overflow bits. This old decoder grants
    // only its documented MODE/IEEE authority; actual producers publish overflow separately.
    return {true, static_cast<uint8_t>((word >> 12) & 0xff), (word & (1u << 23)) != 0};
}
} // namespace prosper::gpu
