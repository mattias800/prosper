// Fixed normalized fields use nominal binary32 RTE scaling and an even integer tie break.
// The unsigned mini-float reference knows finite results only: exceptional conversion/payload
// authority is not inferred from the existing emitted helper. No shader folding domain is added.
#include "gpu/recompiler/ir/eval_buffer_store_fields.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
namespace prosper::gpu::ir {
Known evaluate_pack_normalized_field(uint32_t bits, uint32_t mode) {
    const uint32_t width = mode & 31u;
    const bool sign = (mode & 0x100u) != 0;
    if ((mode & ~0x11fu) || width < 2 || width > 16) return std::nullopt;
    const float value = std::bit_cast<float>(bits);
    if (std::isnan(value)) return std::nullopt;
    const uint32_t norm = (1u << (width - sign)) - 1;
    const float scaled = std::clamp(value, sign ? -1.0f : 0.0f, 1.0f) * static_cast<float>(norm);
    const float whole = std::floor(scaled), fraction = scaled - whole;
    int32_t rounded = static_cast<int32_t>(whole);
    if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) ++rounded;
    return static_cast<uint32_t>(rounded) & ((1u << width) - 1);
}
Known evaluate_pack_unsigned_float(uint32_t bits, uint32_t mantissa) {
    if (mantissa != 5 && mantissa != 6) return std::nullopt;
    const uint32_t exponent = (bits >> 23) & 255, fraction = bits & 0x7fffff;
    if (exponent == 255) return std::nullopt;   // no claim about infinity or NaN conversion/payload
    if ((bits >> 31) || !exponent) return 0;
    if (exponent > 142) return std::nullopt;   // the narrowed result would not be finite
    const uint32_t shift = exponent < 113 ? 136 - mantissa - exponent : 23 - mantissa;
    const uint64_t integer = 0x800000u | fraction;
    if (shift > 32) return 0;
    uint32_t rounded = static_cast<uint32_t>(integer >> shift);
    const uint64_t tail = integer & ((uint64_t{1} << shift) - 1), half = uint64_t{1} << (shift - 1);
    if (tail > half || (tail == half && (rounded & 1))) ++rounded;
    if (exponent < 113) return std::min(rounded, 1u << mantissa);
    uint32_t target_exponent = exponent - 112;
    if (rounded == (1u << (mantissa + 1))) {
        ++target_exponent;
        rounded = 0;
    }
    if (target_exponent >= 31) return std::nullopt;
    return (target_exponent << mantissa) | (rounded & ((1u << mantissa) - 1));
}
}   // namespace prosper::gpu::ir
