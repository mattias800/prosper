// Test-owned encodings and input words, shared by execution and exported corpus witnesses.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>
namespace prosper::test::ir::float_integer_e64 {
struct Form {
    uint32_t source = 258, literal = 0, scalar = 0;
    bool initialize = false;
};
inline Form form(uint32_t index) {
    Form f;
    if (index == 1) f.source = 242; // Inline F32 1.0.
    if (index == 2) f.source = 129; // Integer word 1, interpreted as an F32 subnormal.
    if (index == 3 || index == 4) {
        f.source = 255;
        f.literal = index == 3 ? 0x3fe00000 : 0xff800001; // 1.75 or a negative NaN.
    }
    if (index == 5) f.source = 241; // Inline F32 -0.5.
    if (index == 6) f.source = 193; // Integer -1 is an F32 NaN encoding.
    if (index == 7) {
        f.source = 50;
        f.scalar = 0xc0780000; // -3.875.
        f.initialize = true;
    }
    return f;
}
inline uint32_t source(Form f, const LaneInputs& in, size_t lane) {
    if (f.source == 258) return in.s0[lane];
    if (f.source == 257) return in.old_d[lane];
    if (f.source == 242) return 0x3f800000;
    if (f.source == 241) return 0xbf000000;
    if (f.source == 129) return 1;
    if (f.source == 193) return 0xffffffff;
    if (f.source == 255) return f.literal;
    return f.scalar;
}
inline uint32_t modified(uint32_t word, uint32_t modifiers) {
    if (modifiers & 1u) word &= 0x7fffffffu;
    if (modifiers & 2u) word ^= 0x80000000u;
    return word;
}
inline std::vector<uint32_t> words(uint32_t opcode, Form f = {}, uint32_t modifiers = 0,
                                   uint32_t omod = 0, bool wide = true) {
    std::vector<uint32_t> out;
    if (f.initialize) out.insert(out.end(), {0xbeb203ff, f.scalar});
    if (wide)
        out.insert(out.end(), {0xd4000001u | (opcode << 16) | ((modifiers & 1u) << 8),
                               f.source | ((modifiers & 2u) << 28) | (omod << 27)});
    else
        out.push_back(vop1(opcode - 0x180, kVdst, f.source));
    if (f.source == 255) out.push_back(f.literal);
    return out;
}
inline std::vector<uint32_t> sdwa(uint32_t opcode, uint32_t modifiers) {
    return {vop1(opcode - 0x180, kVdst, 249),
            kVsrc0 | (6u << 8) | (6u << 16) | ((modifiers & 2u) << 19) | ((modifiers & 1u) << 21)};
}
inline LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t edges[] = {
        0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x3f7fffff, 0x3f800000, 0x3fe00000, 0xbfe00000, 0xbf7fffff,
        0x4effffff, 0x4f000000, 0xcf000000, 0xcf000001, 0x4f7fffff, 0x4f800000,
        0x7f800000, 0xff800000, 0x7fc00001, 0xff800001, 0x7f7fffff, 0xff7fffff,
    };
    std::copy(std::begin(edges), std::end(edges), in.s0.begin());
    // Every hand boundary is active in at least one of the paired seeds; the other lanes
    // retain the harness's differing random words and masked old destinations.
    for (size_t i = 0; i < std::size(edges); ++i) in.active[i] = ((i + seed) % 3u) != 0;
    return in;
}
} // namespace prosper::test::ir::float_integer_e64
