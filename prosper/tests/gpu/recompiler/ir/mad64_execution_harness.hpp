// Four raw inputs and independent bit-serial65-bit arithmetic for MAD64 execution tests.
#pragma once
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/ir/vector_mask_harness.hpp"
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace prosper::test::ir::mad64 {
struct Raw {
    uint32_t a, b, lo, hi;
};
struct Answer {
    uint32_t lo = 0, hi = 0, carry = 0;
};
inline Answer reference(Raw x, bool is_signed) {
    const int64_t signed_product =
        int64_t(std::bit_cast<int32_t>(x.a)) * int64_t(std::bit_cast<int32_t>(x.b));
    const uint64_t product =
        is_signed ? static_cast<uint64_t>(signed_product) : uint64_t(x.a) * x.b;
    const uint64_t addend = uint64_t(x.lo) | (uint64_t(x.hi) << 32);
    Answer out;
    uint32_t carry = 0;
    for (uint32_t bit = 0; bit < 65; ++bit) {
        const uint32_t a =
            bit < 64 ? uint32_t((product >> bit) & 1) : uint32_t(is_signed && signed_product < 0);
        const uint32_t b =
            bit < 64 ? uint32_t((addend >> bit) & 1) : uint32_t(is_signed && (x.hi >> 31));
        const uint32_t sum = a + b + carry, value = sum & 1;
        carry = sum >> 1;
        if (bit < 32)
            out.lo |= value << bit;
        else if (bit < 64)
            out.hi |= value << (bit - 32);
        else
            out.carry = value;
    }
    return out;
}
enum class Form {
    Plain,
    Sgpr0,
    Sgpr1,
    SgprBoth,
    Inline0,
    Inline1,
    Literal0,
    Literal1,
    PairSgpr,
    AddOne,
    AddMinusOne,
    Null,
    AddLiteral,
    AliasSources,
    AliasAddend,
    AliasHigh,
    UncachedFullExec
};
inline std::vector<Form> forms(uint32_t opcode) {
    std::vector<Form> out;
    for (uint32_t i = 0; i <= static_cast<uint32_t>(Form::UncachedFullExec); ++i)
        if (opcode == 0x176 || i != static_cast<uint32_t>(Form::AddLiteral))
            out.push_back(static_cast<Form>(i));
    return out;
}
inline uint32_t destination(Form f) {
    return f == Form::UncachedFullExec ? 10u
           : f == Form::AliasSources   ? 2u
           : f == Form::AliasAddend    ? 5u
           : f == Form::AliasHigh      ? 4u
                                       : 1u;
}
inline Raw raw(const LaneInputs& in, size_t lane, Form f) {
    Raw x{in.s0[lane], in.s1[lane], in.s2[lane], in.s3[lane]};
    if (f == Form::Sgpr0 || f == Form::SgprBoth) x.a = 0x80000001;
    if (f == Form::Sgpr1 || f == Form::SgprBoth) x.b = 0xffffffff;
    if (f == Form::Inline0) x.a = UINT32_MAX;
    if (f == Form::Inline1) x.b = UINT32_MAX;
    if (f == Form::Literal0) x.a = 0x89abcdef;
    if (f == Form::Literal1) x.b = 0x89abcdef;
    if (f == Form::PairSgpr) {
        x.lo = 0xffffffff;
        x.hi = 0x80000000;
    }
    if (f == Form::AddOne) {
        x.lo = 1;
        x.hi = 0;
    }
    if (f == Form::AddMinusOne) {
        x.lo = UINT32_MAX;
        x.hi = UINT32_MAX;
    }
    if (f == Form::Null) {
        x.lo = 0;
        x.hi = 0;
    }
    if (f == Form::AddLiteral) {
        x.lo = 0x89abcdef;
        x.hi = 0;
    }
    return x;
}
inline uint32_t old_word(const LaneInputs& in, size_t lane, uint32_t reg) {
    switch (reg) {
        case 1: return in.old_d[lane];
        case 2: return in.s0[lane];
        case 3: return in.s1[lane];
        case 4: return in.active[lane] ? 1u : 0u;
        case 5: return in.s2[lane];
        case 6: return in.s3[lane];
        default: return 0;
    }
}
inline std::vector<uint32_t> words(uint32_t opcode, Form f, uint32_t sdst) {
    std::vector<uint32_t> out;
    const auto scalar = [&](uint32_t reg, uint32_t bits) {
        out.push_back(0xbe8003ffu | (reg << 16));
        out.push_back(bits);
    };
    uint32_t a = 258, b = 259, c = 261;
    if (f == Form::Sgpr0 || f == Form::SgprBoth) {
        scalar(50, 0x80000001);
        a = 50;
    }
    if (f == Form::Sgpr1 || f == Form::SgprBoth) {
        scalar(52, 0xffffffff);
        b = 52;
    }
    if (f == Form::Inline0) a = 193;
    if (f == Form::Inline1) b = 193;
    if (f == Form::Literal0) a = 255;
    if (f == Form::Literal1) b = 255;
    if (f == Form::PairSgpr) {
        scalar(54, 0xffffffff);
        scalar(55, 0x80000000);
        c = 54;
    }
    if (f == Form::AddOne) c = 129;
    if (f == Form::AddMinusOne) c = 193;
    if (f == Form::Null) c = 125;
    if (f == Form::AddLiteral) c = 255;
    out.push_back(0xd4000000u | (opcode << 16) | (sdst << 8) | destination(f));
    out.push_back(a | (b << 9) | (c << 18));
    if (f == Form::Literal0 || f == Form::Literal1 || f == Form::AddLiteral)
        out.push_back(0x89abcdef);
    return out;
}
inline LaneProgram program(uint32_t opcode, Form f, uint32_t sdst, uint32_t wave, uint32_t output) {
    auto body = words(opcode, f, sdst);
    if (output == 2) {
        const auto reader = read_mask(sdst);
        body.insert(body.end(), reader.begin(), reader.end());
    }
    auto out = lane_program(body, 64, wave, true, true);
    if (f == Form::UncachedFullExec) out.code.erase(out.code.begin() + 13);
    if (output != 2) out.code[out.code.size() - 2] = 0x80020000u | ((destination(f) + output) << 8);
    return out;
}
inline LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    uint32_t state = seed;
    for (size_t i = 0; i < in.lanes(); ++i) {
        in.s2.push_back(state = state * 1664525u + 1013904223u);
        in.s3.push_back(state = state * 1664525u + 1013904223u);
    }
    return in;
}
inline LaneRun execute(const std::vector<uint32_t>& module, const LaneInputs& in) {
    std::vector<uint32_t> buffer;
    for (const auto* a : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), a->begin(), a->end());
    for (bool a : in.active) buffer.push_back(a ? 1u : 0u);
    buffer.insert(buffer.end(), in.s2.begin(), in.s2.end());
    buffer.insert(buffer.end(), in.s3.begin(), in.s3.end());
    bpermute_oracle::Interpreter vm(module);
    auto out = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    out.resize(in.lanes());
    return {out, {}};
}
}   // namespace prosper::test::ir::mad64
