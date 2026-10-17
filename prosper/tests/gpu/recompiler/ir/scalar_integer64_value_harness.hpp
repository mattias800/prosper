// Uniform guest SALU operands and independent mathematical references for pair-value tests.
#pragma once
#include "gpu/recompiler/ir/ops/scalar_integer64_values.hpp"
#include "gpu/recompiler/ir/scalar_encoding.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <array>
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <vector>
namespace prosper::test::ir {
struct Scalar64Case {
    uint32_t lo, hi, other_lo, other_hi;
    bool scc;
};
inline std::array<gpu::ir::Known64, 5> scalar64_inputs(const Scalar64Case& x) {
    return {x.lo, x.hi, x.other_lo, x.other_hi, uint64_t(x.scc)};
}
// Reference uses bit enumeration, rather than the definition's shifts/count/find-msb graph.
inline uint64_t scalar64_reference(const gpu::ir::ScalarInteger64Def& def, const Scalar64Case& x) {
    using F = gpu::Rdna2Format;
    const uint64_t a = (uint64_t(x.hi) << 32) | x.lo;
    if (def.fmt == F::SOPC) {
        const bool eq = x.lo == x.other_lo && x.hi == x.other_hi;
        return def.opcode == 0x12 ? eq : !eq;
    }
    if (def.fmt == F::SOP2 && def.opcode == 0x0b)
        return x.scc ? a : (uint64_t(x.other_hi) << 32) | x.other_lo;
    uint64_t out = 0;
    if (def.fmt == F::SOP2) {
        const uint32_t offset = x.other_lo & 63, count = (x.other_lo >> 16) & 127;
        for (uint32_t bit = 0; bit < count && bit + offset < 64; ++bit)
            if ((a >> (bit + offset)) & 1) out |= uint64_t(1) << bit;
    } else if (def.opcode == gpu::kSop1OpcodeBcnt1I32B64) {
        for (uint32_t bit = 0; bit < 64; ++bit) out += (a >> bit) & 1;
    } else if (def.opcode == 0x16) {
        out = 0xffffffffu;
        for (uint32_t bit = 0; bit < 64; ++bit)
            if ((a >> bit) & 1) out = 63 - bit;
    } else {
        for (uint32_t bit = 0; bit < 32; ++bit)
            if ((x.lo >> bit) & 1) out |= uint64_t(3) << (2 * bit);
    }
    return out;
}
inline void scalar64_mov(std::vector<uint32_t>& out, uint32_t dst, uint32_t word) {
    out.insert(out.end(), {sop1(3, dst, kLiteral), word});
}
// Form0 uses s[50:51]/s[54:55]; form1 aliases SDST with S0; form2 aliases S1.
// Sources are uniform S_MOV literals in each guest wave. No per-lane scalar load is fabricated.
inline Scalar64Case scalar64_form_case(Scalar64Case x, uint32_t form) {
    if (form == 3) {
        x.lo = 1;
        x.hi = 0;
    }
    if (form == 4) {
        x.lo = 0xffffffffu;
        x.hi = 0xffffffffu;
    }
    if (form == 5) {
        x.lo = 0x89abcdefu;
        x.hi = 0;
    }
    if (form == 6) {
        x.other_lo = 0;
        x.other_hi = 0;
    }
    if (form == 7) {
        x.other_lo = 0xffffffffu;
        x.other_hi = 0xffffffffu;
    }
    if (form == 8) {
        x.other_lo = 0x00100004u;
        x.other_hi = 0;
    }
    return x;
}
inline std::vector<uint32_t> scalar64_forms(const gpu::ir::ScalarInteger64Def& def) {
    std::vector<uint32_t> out{0, 1, 2, 3, 4, 5};
    if (def.fmt != gpu::Rdna2Format::SOP1) out.insert(out.end(), {6, 7, 8});
    return out;
}
inline std::vector<uint32_t> scalar64_program(const gpu::ir::ScalarInteger64Def& def,
                                              const Scalar64Case& x, uint32_t output,
                                              uint32_t form = 0, bool restore_exec = true) {
    using F = gpu::Rdna2Format;
    std::vector<uint32_t> out;
    scalar64_mov(out, 50, x.lo);
    scalar64_mov(out, 51, x.hi);
    scalar64_mov(out, 54, x.other_lo);
    scalar64_mov(out, 55, x.other_hi);
    out.push_back(sopc(x.scc ? 6 : 7, 50, 50));
    const uint32_t dst = form == 1 ? 50 : form == 2 ? 54 : 60;
    const uint32_t s0 = form == 3 ? 0x81 : form == 4 ? 0xc1 : form == 5 ? kLiteral : 50;
    const uint32_t s1 = form == 6 ? 0x80 : form == 7 ? 0xc1 : form == 8 ? kLiteral : 54;
    if (def.fmt == F::SOP1) out.push_back(sop1(def.opcode, dst, s0));
    if (def.fmt == F::SOP2) out.push_back(sop2(def.opcode, dst, s0, s1));
    if (def.fmt == F::SOPC) out.push_back(sopc(def.opcode, s0, s1));
    if (form == 5) out.push_back(0x89abcdefu);
    if (form == 8) out.push_back(0x00100004u);
    // Scalar instructions still execute with partial EXEC. Read the scalar results after restore.
    if (restore_exec) out.push_back(0xbefe04c1u);
    if (output == 2) {
        out.push_back(sop2(0x0a, 62, kInline0 + 1, kInline0));
        out.push_back(vop1(1, kVdst, 62));
    } else
        out.push_back(vop1(1, kVdst, dst + output));
    return out;
}
inline std::vector<Scalar64Case> scalar64_cases() {
    std::vector<Scalar64Case> out = {{0, 0, 0, 0, false},
                                     {0xffffffffu, 0xffffffffu, 0x400000u, 0, true},
                                     {0x89abcdefu, 0x01234567u, 0x00100004u, 0, false},
                                     {0x89abcdefu, 0x01234567u, 0x00200020u, 0, true},
                                     {0x89abcdefu, 0x01234567u, 0x0024001cu, 0, false},
                                     {0x76543210u, 0xfedcba98u, 0x007f003fu, 0, true},
                                     {1, 0, 0, 0, false},
                                     {0x80000000u, 0, 1, 1, true},
                                     {0, 1, 0, 1, false},
                                     {0, 0x80000000u, 0, 0x7fffffffu, true},
                                     {0x80000000u, 5, 0x7fffffffu, 5, false},
                                     {7, 5, 7, 5, true},
                                     {0x00010001u, 0, 0, 1, false},
                                     {0xffffffffu, 0xffffffffu, 0x007f0000u, 0, true}};
    uint32_t seed = 0x5350;
    const auto next = [&] { return seed = seed * 1664525u + 1013904223u; };
    for (uint32_t i = 0; i < 16; ++i) {
        Scalar64Case x{next(), next(), next(), next(), (i & 1) != 0};
        if ((i % 3) == 0) {
            x.other_lo = x.lo;
            x.other_hi = x.hi;
        }
        out.push_back(x);
    }
    return out;
}
}   // namespace prosper::test::ir
