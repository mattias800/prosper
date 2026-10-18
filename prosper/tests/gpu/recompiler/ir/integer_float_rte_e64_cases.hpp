// Test-owned encodings, hand-worked IEEE754 results, and independent integer rounding oracle.
#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace prosper::test::ir::integer_float_rte {
struct Hand {
    uint32_t raw, signed_bits, unsigned_bits;
};
inline constexpr Hand hand[] = {
    {0, 0, 0},
    {1, 0x3f800000, 0x3f800000},
    {0xffffffff, 0xbf800000, 0x4f800000},
    {0x80000000, 0xcf000000, 0x4f000000},
    {0x7fffffff, 0x4f000000, 0x4f000000},
    {0x01000001, 0x4b800000, 0x4b800000},
    {0x01000003, 0x4b800002, 0x4b800002},
    {0xfefffffd, 0xcb800002, 0x4f7f0000},
    {0x01ffffff, 0x4c000000, 0x4c000000},
    {0x80000040, 0xcf000000, 0x4f000000},
    {0x80000041, 0xceffffff, 0x4f000000},
    {0x80000080, 0xceffffff, 0x4f000000},
    {0x80000081, 0xceffffff, 0x4f000001},
};
inline uint32_t reference(uint32_t raw, bool signed_input) {
    const uint32_t sign = signed_input ? raw & 0x80000000u : 0;
    const uint32_t magnitude = sign ? 0u - raw : raw;
    if (!magnitude) return 0;
    uint32_t exponent = std::bit_width(magnitude) - 1u;
    uint32_t mantissa;
    if (exponent <= 23)
        mantissa = magnitude << (23u - exponent);
    else {
        const uint32_t discarded = exponent - 23u;
        mantissa = magnitude >> discarded;
        const uint32_t remainder = magnitude - (mantissa << discarded);
        const uint32_t midpoint = 1u << (discarded - 1u);
        if (remainder > midpoint || (remainder == midpoint && (mantissa & 1u))) ++mantissa;
        if (mantissa == 0x1000000u) {
            mantissa >>= 1;
            ++exponent;
        }
    }
    return sign | ((exponent + 127u) << 23) | (mantissa & 0x7fffffu);
}
struct Case {
    uint32_t opcode = 0x185, wave = 32, source = 258;
    bool legacy = false;
    uint32_t mode = prosper::gpu::kDefaultComputePgmRsrc1;
    uint32_t modifier = 0; // 1 ABS, 2 NEG, 3 CLAMP, 4 OMOD
    bool accepted = true;
};
inline std::vector<uint32_t> words(const Case& c) {
    const uint32_t prefix = c.legacy ? 0xd0000001u : 0xd4000001u;
    return {prefix | (c.opcode << 16) | (c.modifier == 1 ? 1u << 8 : 0) |
                (c.modifier == 3 ? 1u << 15 : 0),
            c.source | (c.modifier == 2 ? 1u << 28 : 0) | (c.modifier == 4 ? 1u << 27 : 0)};
}
inline LaneProgram program(const Case& c) {
    auto code = words(c);
    if (c.source == 50)
        code.insert(code.begin(), {0xbeb203ff, 0x01000003}); // s_mov s50, 16777219
    auto p = lane_program(code, 64, c.wave);
    p.config.compute_pgm_rsrc1 = c.mode;
    return p;
}
inline LaneInputs inputs(uint32_t seed, uint32_t mask = 0) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        if (lane < std::size(hand)) in.s0[lane] = hand[lane].raw;
        in.old_d[lane] = 0xd0000000u + static_cast<uint32_t>(lane) * 1009u;
        if (mask == 1) in.active[lane] = true;
        if (mask == 2) in.active[lane] = false;
    }
    return in;
}
inline uint32_t source(const Case& c, const LaneInputs& in, size_t lane) {
    return c.source == 50 ? 0x01000003 : c.source == 257 ? in.old_d[lane] : in.s0[lane];
}
inline std::vector<uint32_t> buffer(const LaneInputs& in) {
    std::vector<uint32_t> out;
    for (const auto* values : {&in.s0, &in.s1, &in.old_d})
        out.insert(out.end(), values->begin(), values->end());
    for (bool active : in.active) out.push_back(active ? 1u : 0u);
    return out;
}
inline std::vector<uint32_t> expected(const Case& c, const LaneInputs& in) {
    auto out = buffer(in);
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        out[lane] =
            in.active[lane] ? reference(source(c, in, lane), c.opcode == 0x185) : in.old_d[lane];
    return out;
}
inline std::vector<Case> cases() {
    std::vector<Case> out;
    for (uint32_t op : {0x185u, 0x186u})
        for (uint32_t wave : {32u, 64u})
            for (bool legacy : {false, true}) {
                for (uint32_t source : {258u, 257u, 50u}) out.push_back({op, wave, source, legacy});
                for (uint32_t mode : {UINT32_MAX, 0x31000u, 0x32000u, 0x33000u}) {
                    Case c{op, wave, 258, legacy};
                    c.mode = mode;
                    c.accepted = false;
                    out.push_back(c);
                }
                for (uint32_t missing_source : {51u, 511u}) {
                    Case c{op, wave, missing_source, legacy};
                    c.accepted = false;
                    out.push_back(c);
                }
                for (uint32_t modifier : {1u, 2u, 3u, 4u}) {
                    Case c{op, wave, 258, legacy};
                    c.modifier = modifier;
                    c.accepted = false;
                    out.push_back(c);
                }
            }
    return out;
}
} // namespace prosper::test::ir::integer_float_rte
