#pragma once
#include <cstdint>

namespace prosper::gpu {

// Observed SPI_SHADER_PGM_RSRC1_PS launch flags, independent of FLOAT_MODE and host
// float-control capabilities. An absent register is not a register observed with zero bits.
struct FragmentFloatFlags {
    bool available = false;
    bool ieee_mode = false;
    bool dx10_clamp = false;

    constexpr bool canonical() const {
        return available || (!ieee_mode && !dx10_clamp);
    }
    bool operator==(const FragmentFloatFlags&) const = default;
};

// Exact observed launch-register evidence, not an extra arithmetic policy. Retaining the other
// RSRC1_PS bits must never manufacture missing FLOAT_MODE/IEEE/DX10 inputs or FP16_OVFL authority.
struct FragmentLaunchRsrc1 {
    bool available = false;
    uint32_t value = 0;

    constexpr bool canonical() const { return available || value == 0; }
    bool operator==(const FragmentLaunchRsrc1&) const = default;
};

} // namespace prosper::gpu
