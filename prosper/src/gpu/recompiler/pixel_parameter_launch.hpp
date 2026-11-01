#pragma once

#include <cstdint>
#include <optional>

namespace prosper::gpu {
// Producing SPI_SHADER_PGM_RSRC2_PS, supplied by the caller rather than inferred from code.
// This locates the automatic primitive-mask/LDS word; it supplies none of that word's bits.
struct PixelParameterLaunch {
    bool available = false;
    uint32_t value = 0;

    constexpr bool canonical() const { return available || value == 0; }
    bool operator==(const PixelParameterLaunch&) const = default;
};

std::optional<uint32_t> pixel_parameter_sgpr(const PixelParameterLaunch&);
} // namespace prosper::gpu
