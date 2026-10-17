// Independent format truth: field positions from AMD 70648's memory-format tables.
// Normalized fractions are rounded by integer quotient/remainder, not the IR evaluator or
// host floating division. Binary16 and unsigned F11/F10 widen by constructing IEEE fields.
#pragma once

#include "gpu/recompiler/ir/ops/buffer_format.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace prosper::test::ir::buffer_format {
using prosper::gpu::ir::BufferFieldKind;
using prosper::gpu::ir::BufferFieldShape;

inline uint32_t sign_extend(uint32_t raw, uint32_t width) {
    const uint32_t sign = 1u << (width - 1);
    return (raw ^ sign) - sign;
}

inline uint32_t normalized_fraction(uint32_t magnitude, uint32_t denominator, bool negative) {
    const uint32_t sign = negative ? 0x80000000u : 0;
    if (!magnitude) return 0;
    // SNORM's extra negative endpoint clamps; every other component is already in [-1,1].
    if (magnitude >= denominator) return sign | 0x3f800000u;
    int exponent = 0;
    while ((uint64_t{magnitude} << -exponent) < denominator) --exponent;
    const uint64_t scaled = uint64_t{magnitude} << (23 - exponent);
    uint32_t significand = static_cast<uint32_t>(scaled / denominator);
    const uint64_t remainder = scaled % denominator;
    if (remainder * 2 > denominator || (remainder * 2 == denominator && (significand & 1)))
        ++significand;
    if (significand == 0x1000000u) {
        significand >>= 1;
        ++exponent;
    }
    return sign | (static_cast<uint32_t>(exponent + 127) << 23) | (significand & 0x7fffffu);
}

inline std::optional<uint32_t> half_bits(uint32_t half) {
    const uint32_t sign = (half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 31u, fraction = half & 1023u;
    if (exponent == 31) return fraction ? std::nullopt : std::optional{sign | 0x7f800000u};
    if (!exponent) {
        if (!fraction) return sign;
        int unbiased = -14;
        while (!(fraction & 1024u)) {
            fraction <<= 1;
            --unbiased;
        }
        return sign | (static_cast<uint32_t>(unbiased + 127) << 23) | ((fraction & 1023u) << 13);
    }
    return sign | ((exponent + 112u) << 23) | (fraction << 13);
}

inline std::optional<uint32_t> field_truth(uint32_t word, const BufferFieldShape& shape) {
    if (shape.kind == BufferFieldKind::Half)
        return half_bits((word >> (shape.offset ? 16 : 0)) & 0xffffu);
    const uint32_t raw = (word >> shape.offset) & ((1u << shape.width) - 1u);
    switch (shape.kind) {
        case BufferFieldKind::Unsigned: return raw;
        case BufferFieldKind::Signed: return sign_extend(raw, shape.width);
        case BufferFieldKind::Ufloat: return half_bits(raw << (shape.width == 11 ? 4 : 5));
        case BufferFieldKind::Unorm:
            return normalized_fraction(raw, static_cast<uint32_t>(shape.norm), false);
        case BufferFieldKind::Snorm: {
            const bool negative = (raw & (1u << (shape.width - 1))) != 0;
            const uint32_t magnitude = negative ? ((1u << shape.width) - raw) : raw;
            return normalized_fraction(magnitude, static_cast<uint32_t>(shape.norm), negative);
        }
        case BufferFieldKind::Half: break;
    }
    return std::nullopt;
}

inline std::vector<BufferFieldShape> field_shapes() {
    std::vector<BufferFieldShape> shapes;
    for (const auto kind : {BufferFieldKind::Unsigned, BufferFieldKind::Signed,
                            BufferFieldKind::Unorm, BufferFieldKind::Snorm}) {
        for (uint32_t width : {2u, 8u, 10u, 16u}) {
            const uint32_t limit = width == 2 ? 1 : width == 10 ? 3 : 32 / width;
            const float norm =
                static_cast<float>((1u << (width - (kind == BufferFieldKind::Snorm))) - 1);
            for (uint32_t c = 0; c < limit; ++c)
                shapes.push_back({kind, width == 2 ? 30u : c * width, width, norm});
        }
    }
    shapes.push_back({BufferFieldKind::Half, 0, 16});
    shapes.push_back({BufferFieldKind::Half, 16, 16});
    for (uint32_t c = 0; c < 3; ++c)
        shapes.push_back({BufferFieldKind::Ufloat, c * 11, c == 2 ? 10u : 11u});
    return shapes;
}
} // namespace prosper::test::ir::buffer_format
