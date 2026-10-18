// AMD70648 DPP16 ROW_XMASK/ROW_ROR:8, FI=0, BC=1. Independent lane/result oracle.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>
namespace prosper::test::ir::dpp_row_xor {
enum class Combiner { Mov, Min, Max, Or, AddFloat, AddUnsigned };
struct Case {
    uint32_t wave, control;
    Combiner combiner;
    bool overlap = false;
    uint32_t bad_flags = 0;
};
inline std::vector<Case> cases() {
    std::vector<Case> out;
    for (uint32_t wave : {32u, 64u})
        for (uint32_t control = 0x160; control <= 0x170; ++control)
            for (auto op : {Combiner::Mov, Combiner::Min, Combiner::Max, Combiner::Or,
                            Combiner::AddFloat, Combiner::AddUnsigned})
                for (bool overlap : {false, true})
                    out.push_back({wave, control == 0x170 ? 0x128 : control, op, overlap});
    return out;
}
inline std::vector<Case> refused_cases() {
    std::vector<Case> out;
    for (uint32_t wave : {32u, 64u})
        for (uint32_t bits : {0x80000u, 0x40000u, 0x1000000u, 0x10000000u})
            out.push_back({wave, 0x168, Combiner::Mov, false, bits});
    return out;
}
inline uint32_t opcode(Combiner c) {
    switch (c) {
        case Combiner::Mov: return 1;
        case Combiner::Min: return 0xf;
        case Combiner::Max: return 0x10;
        case Combiner::Or: return 0x1c;
        case Combiner::AddFloat: return 3;
        case Combiner::AddUnsigned: return 0x25;
    }
    return 0;
}
inline LaneProgram program(const Case& c) {
    const uint32_t dst = c.overlap ? kVsrc0 : kVdst;
    std::vector<uint32_t> words{c.combiner == Combiner::Mov ? vop1(opcode(c.combiner), dst, 250)
                                                            : vop2(opcode(c.combiner), dst, 250),
                                (0xff080002u | (c.control << 8)) ^ c.bad_flags};
    if (c.overlap) words.push_back(vop1(1, kVdst, 256 + kVsrc0));
    auto p = lane_program(words, c.wave, c.wave);
    p.config.native_subgroup_size = c.wave;
    return p;
}
inline LaneInputs inputs(const Case& c, uint32_t mask) {
    const uint32_t wave = c.wave;
    LaneInputs in;
    for (uint32_t lane = 0; lane < wave; ++lane) {
        const float a = static_cast<float>(static_cast<int32_t>(lane) - 31) * 0.25f;
        const float b = c.combiner == Combiner::Min   ? -(8.25f + static_cast<float>(lane) * 0.125f)
                        : c.combiner == Combiner::Max ? 8.25f + static_cast<float>(lane) * 0.125f
                                                      : 256.0f + static_cast<float>(lane) * 0.5f;
        in.s0.push_back(std::bit_cast<uint32_t>(a));
        in.s1.push_back(std::bit_cast<uint32_t>(b));
        in.old_d.push_back(0xd0060000u + lane);
        in.active.push_back(mask == 0 || (mask == 1 && lane % 4 != 0));
    }
    return in;
}
inline std::vector<uint32_t> buffer(const LaneInputs& in) {
    std::vector<uint32_t> out;
    for (auto* words : {&in.s0, &in.s1, &in.old_d})
        out.insert(out.end(), words->begin(), words->end());
    for (bool active : in.active) out.push_back(active ? 1u : 0u);
    return out;
}
inline uint32_t bounded_source(const Case& c, const LaneInputs& in, uint32_t lane) {
    const uint32_t stride = c.control == 0x128 ? 8 : c.control - 0x160;
    const uint32_t peer = lane / 16 * 16 + ((lane % 16) ^ stride);
    return in.active[peer] ? in.s0[peer] : 0;
}
inline std::vector<uint32_t> expected(const Case& c, const LaneInputs& in) {
    std::vector<uint32_t> out;
    for (uint32_t lane = 0; lane < c.wave; ++lane) {
        const auto a = bounded_source(c, in, lane), b = in.s1[lane];
        uint32_t d = a;
        switch (c.combiner) {
            case Combiner::Mov: break;
            case Combiner::Min:
                d = std::bit_cast<uint32_t>(
                    std::min(std::bit_cast<float>(a), std::bit_cast<float>(b)));
                break;
            case Combiner::Max:
                d = std::bit_cast<uint32_t>(
                    std::max(std::bit_cast<float>(a), std::bit_cast<float>(b)));
                break;
            case Combiner::Or: d = a | b; break;
            case Combiner::AddFloat:
                d = std::bit_cast<uint32_t>(std::bit_cast<float>(a) + std::bit_cast<float>(b));
                break;
            case Combiner::AddUnsigned: d = a + b; break;
        }
        out.push_back(in.active[lane] ? d : in.old_d[lane]);
    }
    return out;
}
}   // namespace prosper::test::ir::dpp_row_xor
