// Independent integer normalization and hand bits for the deliberately constant-only admission.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <vector>

namespace prosper::test::ir::exact_integer_float {
struct Hand {
    uint32_t raw;
    std::optional<uint32_t> signed_bits, unsigned_bits;
};
inline constexpr std::array kHand{
    Hand{0x00000000, {}, {}},
    Hand{0x00000001, 0x3f800000, 0x3f800000},
    Hand{0x00000040, 0x42800000, 0x42800000},
    Hand{0x00ffffff, 0x4b7fffff, 0x4b7fffff},
    Hand{0x01000000, 0x4b800000, 0x4b800000},
    Hand{0x01000001, {}, {}},
    Hand{0x01000002, 0x4b800001, 0x4b800001},
    Hand{0x02000002, {}, {}},
    Hand{0x02000004, 0x4c000001, 0x4c000001},
    Hand{0x7fffff80, 0x4effffff, 0x4effffff},
    Hand{0x7fffffff, {}, {}},
    Hand{0x80000000, 0xcf000000, 0x4f000000},
    Hand{0x80000001, {}, {}},
    Hand{0x80000080, 0xceffffff, {}},
    Hand{0xffffff00, 0xc3800000, 0x4f7fffff},
    Hand{0xfffffff0, 0xc1800000, {}},
    Hand{0xffffffff, 0xbf800000, {}},
};
// Test-side 64-bit magnitude allows ordinary signed negation even for guest INT_MIN.
// Reconstruct the represented integer to reject any lossy significand, independently of eval.
inline std::optional<uint32_t> reference_bits(uint32_t raw, bool is_signed) {
    const int64_t value = is_signed ? int64_t{std::bit_cast<int32_t>(raw)} : int64_t{raw};
    if (!value)
        return std::nullopt; // Zero is deliberately outside the portable raw-bit contract.
    const uint64_t magnitude = static_cast<uint64_t>(value < 0 ? -value : value);
    const uint32_t exponent = static_cast<uint32_t>(std::bit_width(magnitude) - 1u);
    const uint64_t significand =
        exponent > 23 ? magnitude >> (exponent - 23) : magnitude << (23 - exponent);
    const uint64_t restored =
        exponent > 23 ? significand << (exponent - 23) : significand >> (23 - exponent);
    if (restored != magnitude) return std::nullopt;
    return (value < 0 ? 0x80000000u : 0u) | ((exponent + 127u) << 23) |
           static_cast<uint32_t>(significand & 0x7fffffu);
}
inline std::vector<uint32_t> words(uint32_t opcode, uint32_t selector, uint32_t literal = 0,
                                   bool old = false) {
    std::vector<uint32_t> out;
    if (old)
        out.push_back(vop1(opcode - 0x180, kVdst, selector));
    else
        out.insert(out.end(), {0xd4000001u | (opcode << 16), selector});
    if (selector == 255) out.push_back(literal);
    return out;
}
inline uint32_t inline_word(uint32_t selector) {
    return selector <= 192 ? selector - 128 : 192u - selector;
}
inline std::vector<uint32_t> literals() {
    std::vector<uint32_t> out;
    for (const auto& h : kHand) out.push_back(h.raw);
    // Each significant-bit boundary: exact high mantissa plus the nearest discarded-bit error.
    for (uint32_t shift = 1; shift <= 8; ++shift) {
        const uint32_t magnitude = 0x800001u << shift;
        for (uint32_t raw : {magnitude, 0u - magnitude, magnitude + 1u, 0u - magnitude - 1u})
            out.push_back(raw);
    }
    return out;
}
} // namespace prosper::test::ir::exact_integer_float
