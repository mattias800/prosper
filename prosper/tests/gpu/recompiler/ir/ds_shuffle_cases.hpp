// Synthetic native DS programs, genuine all-lane inputs, and an ISA-only lane oracle.
#pragma once

#include "gpu/recompiler/ir/vector_lane_harness.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace prosper::test::ir::ds_shuffle {
enum class Kind { Bpermute, Quad, Mask };
enum class Exec { Partial, Full, Empty, Unmasked };
struct Case {
    uint32_t wave;
    Kind kind;
    uint32_t control, pattern;
    Exec exec;
    uint32_t dst;
};
inline std::vector<Case> cases() {
    std::vector<Case> out;
    for (uint32_t wave : {32u, 64u}) {
        for (uint32_t offset : {0u, 1u, 3u, 4u, 127u, 65535u})
            for (uint32_t pattern : {0u, 1u})
                for (Exec exec : {Exec::Partial, Exec::Full, Exec::Empty, Exec::Unmasked})
                    for (uint32_t dst : {1u, 2u, 3u})
                        out.push_back({wave, Kind::Bpermute, offset, pattern, exec, dst});
        for (uint32_t control : {0x8000u, 0x80e4u, 0x80b1u, 0x804eu, 0x801bu, 0x001fu, 0x041fu,
                                 0x7c1fu, 0x03e0u, 0x020fu, 0x000fu})
            for (Exec exec : {Exec::Partial, Exec::Full, Exec::Empty, Exec::Unmasked})
                for (uint32_t dst : {1u, 2u})
                    out.push_back(
                        {wave, control & 0x8000u ? Kind::Quad : Kind::Mask, control, 0, exec, dst});
    }
    return out;
}
inline LaneInputs inputs(const Case& c) {
    constexpr std::array<uint32_t, 8> addresses = {0u,   3u,          7u,          127u,
                                                   128u, 0xfffffffcu, 0xffffffffu, 0xffff0004u};
    LaneInputs in;
    for (uint32_t lane = 0; lane < c.wave; ++lane) {
        in.s0.push_back(c.kind == Kind::Bpermute
                            ? (c.pattern == 0 ? lane * 4 : addresses[lane % addresses.size()])
                            : 0x62100000u + lane * 0x1021u);
        in.s1.push_back(0x34200000u + lane * 0x2011u);
        in.old_d.push_back(0xd0000000u + lane);
        in.active.push_back(c.exec == Exec::Full || c.exec == Exec::Unmasked ||
                            (c.exec == Exec::Partial && lane % 4 != 0));
    }
    return in;
}
inline LaneProgram program(const Case& c) {
    std::vector<uint32_t> words;
    if (c.kind == Kind::Bpermute)
        words = {0xdacc0000u | c.control, (c.dst << 24) | (kVsrc1 << 8) | kVsrc0};
    else
        words = {0xd8d40000u | c.control, (c.dst << 24) | kVsrc0};
    if (c.dst != kVdst) words.push_back(vop1(1, kVdst, 256 + c.dst));
    auto p = lane_program(words, c.wave, c.wave);
    p.config.native_subgroup_size = c.wave;
    if (c.exec == Exec::Unmasked) p.code.erase(p.code.begin() + 9); // omit harness CMPX
    return p;
}
inline std::vector<uint32_t> buffer(const LaneInputs& in) {
    std::vector<uint32_t> out;
    for (const auto* words : {&in.s0, &in.s1, &in.old_d})
        out.insert(out.end(), words->begin(), words->end());
    for (bool active : in.active) out.push_back(active ? 1u : 0u);
    return out;
}
inline uint32_t peer(const Case& c, const LaneInputs& in, uint32_t lane) {
    // AMD70648 10.4.4: byte OFFSET is added before alignment; each Wave64 half is independent.
    if (c.kind == Kind::Bpermute) {
        const uint32_t byte_address = in.s0[lane] + c.control;
        return (lane / 32) * 32 + (byte_address / 4) % 32;
    }
    // AMD70648 12.13.1: four two-bit selectors, or group32 AND/OR/XOR masks.
    if (c.kind == Kind::Quad) return (lane / 4) * 4 + (c.control >> ((lane % 4) * 2)) % 4;
    const uint32_t selected =
        (((lane % 32) & (c.control % 32)) | ((c.control / 32) % 32)) ^ ((c.control / 1024) % 32);
    return (lane / 32) * 32 + selected;
}
inline std::vector<uint32_t> expected(const Case& c, const LaneInputs& in) {
    std::vector<uint32_t> out;
    for (uint32_t lane = 0; lane < c.wave; ++lane) {
        const uint32_t selected = peer(c, in, lane);
        const uint32_t value = !in.active[selected]       ? 0u
                               : c.kind == Kind::Bpermute ? in.s1[selected]
                                                          : in.s0[selected];
        out.push_back(in.active[lane] ? value : in.old_d[lane]);
    }
    return out;
}
} // namespace prosper::test::ir::ds_shuffle
