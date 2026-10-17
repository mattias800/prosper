// Native dynamic WRITELANE fixtures. The scalar source is uniform; old VDST differs per lane.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>
namespace prosper::test::ir::native_writelane {
enum class Source { Scalar, InlineMinusOne, Literal };
inline uint32_t source_word(Source source) {
    return source == Source::InlineMinusOne ? 0xffffffffu : 0x5370abcdu;
}
inline std::vector<uint32_t> selectors(uint32_t wave) {
    return {0u, 1u, 5u, wave - 1u, wave, wave + 5u, 0xffffffffu, 0x80000021u};
}
inline LaneInputs inputs(uint32_t wave, uint32_t mask = 0) {
    auto in = lane_inputs(0x5370u + wave, wave);
    for (uint32_t lane = 0; lane < wave; ++lane) {
        in.old_d[lane] = 0x56000000u | (lane * 1009u + 13u);
        if (mask == 1) in.active[lane] = true;
        if (mask == 2) in.active[lane] = false;
    }
    return in;
}
inline LaneProgram program(uint32_t wave, uint32_t selector, Source source, uint32_t native,
                           uint32_t mask = 0, bool following = false) {
    // s20 is the source; s21 is a real SGPR selector (even when its value is constant).
    std::vector<uint32_t> words{0xbe9403ffu, source_word(source), 0xbe9503ffu, selector};
    const uint32_t src0 = source == Source::Scalar           ? 20u
                          : source == Source::InlineMinusOne ? 193u
                                                             : 255u;
    words.insert(words.end(), {0xd7610001u, src0 | (21u << 9)});
    if (source == Source::Literal) words.push_back(source_word(source));
    if (following) words.push_back(vop1(0x37, kVdst, 256u + kVdst));
    auto p = lane_program(words, wave, wave);
    p.config.native_subgroup_size = native;
    if (mask == 1) {
        const auto cmp = std::find(p.code.begin(), p.code.end(), 0x7daa0880u);
        if (cmp == p.code.end())
            p.code.clear();
        else
            p.code.erase(cmp);
    }
    return p;
}
inline uint32_t answer(const LaneInputs& in, uint32_t lane, uint32_t selector, Source source,
                       bool following = false) {
    uint32_t word = lane == selector % in.lanes() ? source_word(source) : in.old_d[lane];
    if (following && in.active[lane]) word = ~word;
    return word;
}
} // namespace prosper::test::ir::native_writelane
