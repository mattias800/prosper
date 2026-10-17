// Test-owned half-step normalization and integer tie answers; no host floating arithmetic/eval.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
namespace prosper::test::ir::cvt_rpi_exact {
struct Hand {
    uint32_t bits, answer;
};
inline constexpr std::array kHand{
    Hand{0, 0},
    Hand{0x80000000, 0},
    Hand{0x3f000000, 1},
    Hand{0xbf000000, 0},
    Hand{0x3f800000, 1},
    Hand{0xbf800000, UINT32_MAX},
    Hand{0x3fc00000, 2},
    Hand{0xbfc00000, UINT32_MAX},
    Hand{0x40200000, 3},
    Hand{0xc0200000, 0xfffffffe},
    Hand{0x4affffff, 8388608},
    Hand{0xcaffffff, 0xff800001},
    Hand{0x4afffffe, 8388607},
    Hand{0xcafffffe, 0xff800001},
    Hand{0x40800000, 4},
    Hand{0xc0800000, 0xfffffffc},
};
// Reconstruct 2*x in 64-bit integer arithmetic, then round its signed half toward +infinity.
// This division/reconstruction reference is independent of the production low-bit-mask predicate.
inline std::optional<int64_t> half_units(uint32_t bits) {
    const auto magnitude = bits & 0x7fffffffu;
    if (!magnitude) return 0;
    const int exponent = static_cast<int>(magnitude >> 23) - 127;
    if (exponent < -1 || exponent >= 23) return std::nullopt;
    const uint64_t numerator = 0x800000u | (magnitude & 0x7fffffu);
    const uint64_t denominator = uint64_t{1} << (22 - exponent);
    const uint64_t units = numerator / denominator;
    if (units * denominator != numerator) return std::nullopt;
    return bits >> 31 ? -static_cast<int64_t>(units) : static_cast<int64_t>(units);
}
inline std::optional<uint32_t> reference(uint32_t bits) {
    const auto units = half_units(bits);
    if (!units) return std::nullopt;
    const auto m = units.value_or(0);
    return static_cast<uint32_t>(m >= 0 ? (m + 1) / 2 : -((-m) / 2));
}
inline uint32_t encode_units(int32_t units) {
    if (!units) return 0;
    const uint32_t magnitude = static_cast<uint32_t>(units < 0 ? -units : units);
    const uint32_t top = std::bit_width(magnitude) - 1;
    return (units < 0 ? 0x80000000u : 0u) | ((top + 126u) << 23) |
           ((magnitude << (23 - top)) & 0x7fffffu);
}
inline std::vector<uint32_t> words(uint32_t selector, uint32_t literal = 0, uint32_t encoding = 0) {
    std::vector<uint32_t> result;
    if (encoding == 2)
        result.push_back(vop1(0x0c, kVdst, selector));
    else
        result = {encoding ? 0xd18c0001u : 0xd58c0001u, selector};
    if (selector == 255) result.push_back(literal);
    return result;
}
inline uint32_t inline_bits(uint32_t selector) {
    // Bit conversion comes from the same architectural constant table as decode, not an op list.
    return std::bit_cast<uint32_t>(prosper::gpu::inline_float_value(selector));
}
inline LaneInputs inputs(uint32_t prefix = 0) {
    auto in = lane_inputs(5360);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.old_d[lane] = 0x53600000u + static_cast<uint32_t>(lane) * 1009u;
        if (prefix == 1) in.active[lane] = true;
        if (prefix == 2) in.active[lane] = false;
    }
    return in;
}
inline LaneProgram program(uint32_t selector, uint32_t literal = 0, uint32_t encoding = 0,
                           uint32_t wave = 64, uint32_t prefix = 0, uint32_t launch = UINT32_MAX) {
    auto code = words(selector, literal, encoding);
    if (prefix == 3) code.push_back(vop1(0x37, kVdst, 256 + kVdst));
    auto p = lane_program(code, 64, wave);
    if (prefix == 1) {
        bool found = false;
        for (auto it = p.code.begin(); it != p.code.end(); ++it)
            if (*it == 0x7daa0880u) {
                p.code.erase(it);
                found = true;
                break;
            }
        if (!found) p.code.clear();   // Fail compilation if the fixture's expected prefix moves.
    }
    p.config.compute_pgm_rsrc1 = launch;
    return p;
}
}   // namespace prosper::test::ir::cvt_rpi_exact
