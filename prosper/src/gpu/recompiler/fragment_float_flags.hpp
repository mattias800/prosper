#pragma once

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

} // namespace prosper::gpu
