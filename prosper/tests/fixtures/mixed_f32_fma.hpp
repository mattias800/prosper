// AMD 70648 12.10: MIX chooses full F32 or either F16 half independently for each source.
// Finite RTE reference only. GLSL Fma's weaker hosted precision is tested separately on Vulkan.
#pragma once

#include "gpu/recompiler/ir/vector_lane_harness.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::test::mixed_f32 {
using namespace prosper::test::ir;

enum class OperandForm : uint8_t { Vgpr, Sgpr, InlineFloat, InlineInt, Literal, Destination };
struct Shape {
    uint32_t half = 0, high = 0, abs = 0, neg = 0;
    bool clamp = false;
    OperandForm form = OperandForm::Vgpr;
    uint32_t slot = 0;
};
inline constexpr uint32_t kConstant = 0x3c013800u;
inline constexpr uint32_t kRsrc1 = 0xf0000u;   // RTE, preserve F32/F16 inputs and outputs

inline uint32_t operand(Shape shape, uint32_t slot) {
    if (slot == shape.slot) {
        switch (shape.form) {
            case OperandForm::Sgpr: return 0;
            case OperandForm::InlineFloat:
                return 242;   // +1.0 at the selected width
            // Nonzero inline integer bits are subnormal at either source width. Their guest
            // MODE/hosted unpack contract is separate; exercise the integer spelling of zero.
            case OperandForm::InlineInt: return 128;
            case OperandForm::Literal: return 255;
            case OperandForm::Destination: return 256 + kVdst;
            case OperandForm::Vgpr: break;
        }
    }
    constexpr uint32_t regs[] = {kVsrc0, kVsrc1, kVsrc2};
    return 256 + regs[slot];
}

// llvm-mc gfx1030: plain = cc200001 04160702; half/high SRC1 = cc201001 14160702.
inline std::vector<uint32_t> words(Shape shape = {}, uint32_t opcode = 0x20) {
    std::vector<uint32_t> out = {
        0xcc000000u | (opcode << 16) | kVdst | (shape.abs << 8) | (shape.high << 11) |
            ((shape.half & 4u) << 12) | (shape.clamp ? 1u << 15 : 0u),
        operand(shape, 0) | (operand(shape, 1) << 9) | (operand(shape, 2) << 18) |
            ((shape.half & 3u) << 27) | (shape.neg << 29)};
    if (shape.form == OperandForm::Literal) out.push_back(kConstant);
    return out;
}

inline LaneProgram program(Shape shape, uint32_t wave, uint32_t opcode = 0x20) {
    auto out = lane_program(words(shape, opcode), 64, wave, true);
    out.config.compute_pgm_rsrc1 = kRsrc1;
    if (shape.form == OperandForm::Sgpr)
        out.code.insert(out.code.begin(), {0xbe8003ffu, kConstant});
    return out;
}

inline LaneInputs inputs(Shape shape = {}) {
    auto out = lane_inputs(522520);
    const uint32_t full[3][4] = {{0x3f800001, 0x3fc00000, 0xbf800000, 0x40000000},
                                 {0x3f7ffffe, 0x40000000, 0x40000000, 0x3f000000},
                                 {0xbf800000, 0x3f000000, 0x40400000, 0xbf000000}};
    const uint16_t half[3][4] = {{0x3c01, 0xbe00, 0x4300, 0x35c0},
                                 {0x3bfe, 0x3e00, 0xb800, 0x4200},
                                 {0xbc00, 0x3800, 0x3a00, 0xc100}};
    out.s2.resize(out.lanes());
    for (std::size_t lane = 0; lane < out.lanes(); ++lane) {
        // Both packed halves and the full F32 are finite normal values. Aliasing VDST must
        // not turn this F32 correction into a test of an unpromised half-denormal profile.
        out.old_d[lane] = 0x3e003c00u + (uint32_t(lane & 31u) << 16) + uint32_t(lane & 31u);
        for (uint32_t slot = 0; slot < 3; ++slot) {
            uint32_t value = full[slot][lane % 4];
            uint32_t selected = half[slot][lane % 4];
            if (slot == 2 && shape.half == 2 && lane % 4 == 0) value = 0xbf7fc000;
            if (slot == 2 && shape.clamp) {
                constexpr uint32_t ff[] = {0x40000000, 0x40400000, 0x40800000, 0x40a00000};
                constexpr uint16_t hh[] = {0x4000, 0x4200, 0x4400, 0x4500};
                value = ff[lane % 4];
                selected = hh[lane % 4];
            }
            if (shape.half & (1u << slot)) {
                const uint32_t decoy = selected ^ 0x8000u;
                value =
                    shape.high & (1u << slot) ? (selected << 16) | decoy : selected | (decoy << 16);
            }
            (slot == 0 ? out.s0 : slot == 1 ? out.s1 : out.s2)[lane] = value;
        }
    }
    out.active[4] = true;
    return out;
}

// Exact binary16-to-binary32 expansion for the finite test operands, independently of IR eval.
inline float expand_half(uint32_t bits) {
    const uint32_t exponent = (bits >> 10) & 31u;
    const float magnitude = std::ldexp(float((bits & 1023u) | (exponent ? 1024u : 0u)),
                                       exponent ? int(exponent) - 25 : -24);
    return bits & 0x8000u ? -magnitude : magnitude;
}

inline std::array<uint32_t, 3> sources(Shape shape, const LaneInputs& in, std::size_t lane) {
    std::array<uint32_t, 3> out{in.s0[lane], in.s1[lane], in.s2[lane]};
    for (uint32_t slot = 0; slot < 3; ++slot) {
        const bool half = (shape.half & (1u << slot)) != 0;
        if (slot == shape.slot) {
            switch (shape.form) {
                case OperandForm::Sgpr:
                case OperandForm::Literal: out[slot] = kConstant; break;
                case OperandForm::InlineFloat: out[slot] = half ? 0x3c00 : 0x3f800000; break;
                case OperandForm::InlineInt: out[slot] = 0; break;
                case OperandForm::Destination: out[slot] = in.old_d[lane]; break;
                case OperandForm::Vgpr: break;
            }
        }
        if (half) {
            const uint32_t selected = (out[slot] >> ((shape.high & (1u << slot)) ? 16 : 0)) & 65535;
            out[slot] = std::bit_cast<uint32_t>(expand_half(selected));
        }
        if (shape.abs & (1u << slot)) out[slot] &= 0x7fffffffu;
        if (shape.neg & (1u << slot)) out[slot] ^= 0x80000000u;
    }
    return out;
}

inline uint32_t expected(Shape shape, const LaneInputs& in, std::size_t lane) {
    if (!in.active[lane]) return in.old_d[lane];
    const auto src = sources(shape, in, lane);
    float result = std::fma(std::bit_cast<float>(src[0]), std::bit_cast<float>(src[1]),
                            std::bit_cast<float>(src[2]));
    if (shape.clamp) result = std::fmin(std::fmax(result, 0.0f), 1.0f);
    return std::bit_cast<uint32_t>(result);
}

inline std::vector<Shape> shapes() {
    std::vector<Shape> out;
    for (uint32_t half = 0; half < 8; ++half)
        for (uint32_t high = 0; high < 8; ++high)
            if (!(high & ~half)) out.push_back({half, high});
    for (uint32_t slot = 0; slot < 3; ++slot) {
        out.push_back({5, 1, 1u << slot});
        out.push_back({5, 1, 0, 1u << slot});
        out.push_back({5, 1, 1u << slot, 1u << slot});
        for (auto form : {OperandForm::Sgpr, OperandForm::InlineFloat, OperandForm::InlineInt,
                          OperandForm::Literal, OperandForm::Destination})
            for (uint32_t half : {0u, 1u << slot}) {
                out.push_back({half, 0, 0, 0, false, form, slot});
                if (half) out.push_back({half, half, 0, 0, false, form, slot});
            }
    }
    out.push_back({0, 0, 0, 0, true});
    out.push_back({7, 5, 0, 0, true});
    // The changed recorded program's PC1221: full S0, high-half S1, and negated inline zero S2.
    out.push_back({2, 2, 0, 4, false, OperandForm::InlineInt, 2});
    return out;
}
}   // namespace prosper::test::mixed_f32
