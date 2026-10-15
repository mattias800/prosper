// Numeric reference for the PackHalfLo IR primitive. Scale the exact binary32 value into
// binary16 significand units, then round ties to even; integer addition carries the exponent.
#include "gpu/recompiler/ir/eval.hpp"
#include <optional>
#include "gpu/recompiler/ir/eval_half.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
namespace prosper::gpu::ir {
Known pack_half_rte(uint32_t bits) {
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const double magnitude = std::fabs(double(std::bit_cast<float>(bits)));
    if (std::isnan(magnitude)) return std::nullopt;
    if (magnitude >= 65520.0) return sign | 0x7c00u;
    if (magnitude == 0) return sign;
    int exponent = 0;
    std::frexp(magnitude, &exponent);
    --exponent;
    const bool subnormal = magnitude < 0x1p-14;
    const double scaled = std::ldexp(magnitude, subnormal ? 24 : 10 - exponent);
    uint32_t rounded = static_cast<uint32_t>(std::floor(scaled));
    const double remainder = scaled - rounded;
    if (remainder > 0.5 || (remainder == 0.5 && (rounded & 1u))) ++rounded;
    const uint32_t prefix = subnormal ? 0u : static_cast<uint32_t>(exponent + 14) * 1024u;
    return sign | (prefix + rounded);
}
} // namespace prosper::gpu::ir
