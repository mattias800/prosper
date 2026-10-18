// AMD70648 Tables90/91: FI0 zeroes inactive SRC0; BC gates only an out-of-row write.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace prosper::test::ir::dpp16_source {
enum class Op { Mov, Cvt, Add, Sub, SubRev, Mul, Min, Max, Mac, Fmac };
struct Case {
    uint32_t wave, control;
    Op op;
    bool bound;
};
inline constexpr std::array kFloatOps{Op::Add, Op::Sub, Op::SubRev, Op::Mul,
                                      Op::Min, Op::Max, Op::Mac,    Op::Fmac};
inline std::vector<Case> cases() {
    std::vector<Case> out;
    for (uint32_t wave : {32u, 64u}) {
        for (uint32_t control = 0x111; control <= 0x11f; ++control)
            for (bool bound : {false, true}) {
                out.push_back({wave, control, Op::Mov, bound});
                for (const auto op : kFloatOps) out.push_back({wave, control, op, bound});
            }
        // All 256 tables, including broadcasts and duplicates, at both wave widths.
        for (uint32_t control = 0; control < 256; ++control)
            out.push_back({wave, control, Op::Mov, true});
        for (uint32_t control : {0u, 0x1bu, 0x4eu, 0xa0u, 0xb1u, 0xe4u}) {
            out.push_back({wave, control, Op::Mov, false});
            for (bool bound : {false, true}) {
                out.push_back({wave, control, Op::Cvt, bound});
                for (const auto op : kFloatOps) out.push_back({wave, control, op, bound});
            }
        }
        // Additional conversion forms observed in the changed recorded modules.
        for (uint32_t control : {0x44u, 0xeeu, 0xf5u})
            out.push_back({wave, control, Op::Cvt, true});
    }
    return out;
}
inline uint32_t opcode(Op op) {
    switch (op) {
        case Op::Mov: return 1;
        case Op::Cvt: return 7;
        case Op::Add: return 3;
        case Op::Sub: return 4;
        case Op::SubRev: return 5;
        case Op::Mul: return 8;
        case Op::Min: return 0xf;
        case Op::Max: return 0x10;
        case Op::Mac: return 0x1f;
        case Op::Fmac: return 0x2b;
    }
    return 0;
}
inline bool unary(Op op) {
    return op == Op::Mov || op == Op::Cvt;
}
inline LaneProgram program(const Case& c, bool fetch_inactive = false) {
    const uint32_t first =
        unary(c.op) ? vop1(opcode(c.op), kVdst, 250) : vop2(opcode(c.op), kVdst, 250);
    const uint32_t flags = 0xff000002u | (c.control << 8) | (uint32_t(c.bound) << 19) |
                           (uint32_t(fetch_inactive) << 18);
    auto p = lane_program({first, flags}, c.wave, c.wave);
    p.config.native_subgroup_size = c.wave;
    return p;
}
inline LaneInputs inputs(const Case& c, uint32_t mask) {
    LaneInputs in;
    for (uint32_t lane = 0; lane < c.wave; ++lane) {
        in.s0.push_back(std::bit_cast<uint32_t>(1.0f + static_cast<float>(lane)));
        in.s1.push_back(std::bit_cast<uint32_t>(20.0f + static_cast<float>(lane) * 0.25f));
        in.old_d.push_back(std::bit_cast<uint32_t>(128.25f + static_cast<float>(lane) * 0.25f));
        in.active.push_back(mask == 0 || (mask == 1 && lane % 4 != 0));
    }
    return in;
}
inline std::vector<uint32_t> buffer(const LaneInputs& in) {
    std::vector<uint32_t> out;
    for (const auto* a : {&in.s0, &in.s1, &in.old_d}) out.insert(out.end(), a->begin(), a->end());
    for (const bool active : in.active) out.push_back(active ? 1u : 0u);
    return out;
}
inline std::vector<uint32_t> expected(const Case& c, const LaneInputs& in) {
    std::vector<uint32_t> out;
    for (uint32_t lane = 0; lane < c.wave; ++lane) {
        const bool row = c.control >= 0x111;
        const bool in_bounds = !row || lane % 16 >= c.control - 0x110;
        if (!in.active[lane] || (!c.bound && !in_bounds)) {
            out.push_back(in.old_d[lane]);
            continue;
        }
        const uint32_t peer = row ? (in_bounds ? lane - (c.control - 0x110) : lane)
                                  : lane / 4 * 4 + ((c.control >> (2 * (lane % 4))) & 3);
        const uint32_t word = in_bounds && in.active[peer] ? in.s0[peer] : 0;
        if (c.op == Op::Mov) {
            out.push_back(word);
            continue;
        }
        const float a = std::bit_cast<float>(word), b = std::bit_cast<float>(in.s1[lane]);
        if (c.op == Op::Cvt) {
            out.push_back(static_cast<uint32_t>(a));
            continue;
        }
        // Every value is an exact quarter here, including both multiply-adds. This
        // tests the source transform independently of fused-rounding portability.
        float d = 0;
        switch (c.op) {
            case Op::Add: d = a + b; break;
            case Op::Sub: d = a - b; break;
            case Op::SubRev: d = b - a; break;
            case Op::Mul: d = a * b; break;
            case Op::Min: d = a < b ? a : b; break;
            case Op::Max: d = a > b ? a : b; break;
            case Op::Mac:
            case Op::Fmac: d = a * b + std::bit_cast<float>(in.old_d[lane]); break;
            default: break;
        }
        out.push_back(std::bit_cast<uint32_t>(d));
    }
    return out;
}
} // namespace prosper::test::ir::dpp16_source
