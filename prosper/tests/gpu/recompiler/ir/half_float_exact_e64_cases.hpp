// Test-owned bit-field widening and architectural half-width inline encodings.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
namespace prosper::test::ir::half_float_exact {
struct Hand {
    uint32_t half, answer;
};
inline constexpr std::array kHand{
    Hand{0x0400, 0x38800000}, Hand{0x0401, 0x38802000}, Hand{0x07ff, 0x38ffe000},
    Hand{0x0800, 0x39000000}, Hand{0x3118, 0x3e230000}, Hand{0x3800, 0x3f000000},
    Hand{0x3bff, 0x3f7fe000}, Hand{0x3c00, 0x3f800000}, Hand{0x3c01, 0x3f802000},
    Hand{0x4000, 0x40000000}, Hand{0x4400, 0x40800000}, Hand{0x7bff, 0x477fe000},
    Hand{0x8400, 0xb8800000}, Hand{0xb118, 0xbe230000}, Hand{0xbc01, 0xbf802000},
    Hand{0xfbff, 0xc77fe000},
};
inline constexpr std::array<uint32_t, 9> kInlineHalf{0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000,
                                                     0xc000, 0x4400, 0xc400, 0x3118};
// Normal binary16 sign/exponent/fraction fields embed exactly in binary32. No host FP or eval.
inline std::optional<uint32_t> reference(uint32_t raw) {
    const auto magnitude = raw & 0x7fffu;
    if ((raw & 0xffff0000u) || magnitude < 0x0400 || magnitude > 0x7bff) return std::nullopt;
    return ((raw & 0x8000u) << 16) | ((((raw >> 10) & 31u) + 112u) << 23) | ((raw & 0x3ffu) << 13);
}
inline std::vector<uint32_t> words(uint32_t selector, uint32_t literal = 0, uint32_t encoding = 0) {
    std::vector<uint32_t> result;
    if (encoding == 2)
        result.push_back(vop1(0x0b, kVdst, selector));
    else
        result = {encoding ? 0xd18b0001u : 0xd58b0001u, selector};
    if (selector == 255) result.push_back(literal);
    return result;
}
inline LaneInputs inputs(uint32_t shape = 0) {
    auto in = lane_inputs(5362);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.old_d[lane] = 0x53620000u + static_cast<uint32_t>(lane) * 1009u;
        if (shape == 1) in.active[lane] = true;
        if (shape == 2) in.active[lane] = false;
    }
    return in;
}
inline LaneProgram program(uint32_t selector, uint32_t literal = 0, uint32_t encoding = 0,
                           uint32_t wave = 64, uint32_t shape = 0, uint32_t launch = UINT32_MAX) {
    auto code = words(selector, literal, encoding);
    if (shape == 3) code.push_back(vop1(0x37, kVdst, 256 + kVdst));
    auto p = lane_program(code, 64, wave);
    if (shape == 1) {
        bool found = false;
        for (auto it = p.code.begin(); it != p.code.end(); ++it)
            if (*it == 0x7daa0880u) {
                p.code.erase(it);
                found = true;
                break;
            }
        if (!found) p.code.clear();
    }
    p.config.compute_pgm_rsrc1 = launch;
    return p;
}
} // namespace prosper::test::ir::half_float_exact
