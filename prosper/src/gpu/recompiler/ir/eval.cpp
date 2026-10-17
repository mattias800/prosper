// eval.cpp — the evaluation half of each IR opcode's definition (see ir.hpp).
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/eval_conversion.hpp"
#include "gpu/recompiler/ir/eval_half.hpp"
#include "gpu/recompiler/ir/eval_normalized_pack.hpp"
#include "gpu/recompiler/ir/eval_normalized_field.hpp"
#include "gpu/recompiler/ir/eval_float_unary.hpp"
#include "gpu/recompiler/ir/eval_float_compare.hpp"
#include "gpu/recompiler/ir/eval_integer64.hpp"
#include "gpu/recompiler/ir/eval_float64.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cmath>

namespace prosper::gpu::ir {

namespace {

int32_t s(uint32_t v) { return std::bit_cast<int32_t>(v); }

uint32_t bit_reverse(uint32_t v) {
    uint32_t r = 0;
    for (int i = 0; i < 32; ++i) r |= ((v >> i) & 1u) << (31 - i);
    return r;
}

// SpirvCompute::bfe_u: OpBitFieldUExtract(base, min(off, 32), min(cnt, 32 - min(off, 32))).
uint32_t bfe_u(uint32_t base, uint32_t offset, uint32_t count) {
    const uint32_t off = std::min(offset, 32u);
    const uint32_t cnt = std::min(count, 32u - off);
    if (cnt == 0) return 0;
    const uint32_t field = base >> off;   // off < 32 here: cnt > 0 implies off < 32
    return cnt == 32 ? field : field & ((1u << cnt) - 1u);
}

} // namespace

Known evaluate_op(Opcode op, uint32_t imm, uint32_t a, uint32_t b, uint32_t c) {
    switch (op) {
        case Opcode::Input: return std::nullopt;
        case Opcode::ConstU32: return imm;
        case Opcode::ConstTrue: return 1u;
        case Opcode::ConstFalse: return 0u;
        case Opcode::BoolAnd: return uint32_t(bool(a) && bool(b));
        case Opcode::BoolOr: return uint32_t(bool(a) || bool(b));
        case Opcode::BoolNot: return uint32_t(!a);
        case Opcode::F32Eq: return uint32_t(std::bit_cast<float>(a) == std::bit_cast<float>(b));
        case Opcode::F32Abs:
            // A NaN payload is unspecified by GLSL FAbs, so no payload is guessed.
            if ((a & 0x7fffffffu) > 0x7f800000u) return std::nullopt;
            return a & 0x7fffffffu;
        case Opcode::F32Lt: return uint32_t(std::bit_cast<float>(a) < std::bit_cast<float>(b));
        case Opcode::F32UnordNe: {
            const float x = std::bit_cast<float>(a), y = std::bit_cast<float>(b);
            return uint32_t(std::isnan(x) || std::isnan(y) || x != y);
        }
        case Opcode::F32Le:
        case Opcode::F32Gt:
        case Opcode::F32Lg:
        case Opcode::F32UnordLt:
        case Opcode::F32UnordEq:
        case Opcode::F32UnordLe:
        case Opcode::F32UnordGt:
        case Opcode::F32UnordGe: return evaluate_float_compare(op, a, b);
        case Opcode::F32Clamp01:
            return std::bit_cast<uint32_t>(
                std::fmin(std::fmax(std::bit_cast<float>(a), 0.0f), 1.0f));
        case Opcode::F32Ldexp:
            // The integer-backed emitter preserves specials verbatim. +/-512 already takes every
            // finite binary32 value beyond its representable scaling range; bounding prevents
            // host integer-exponent arithmetic overflow without changing the rounded result.
            if ((a & 0x7f800000u) == 0x7f800000u) return a;
            return std::bit_cast<uint32_t>(
                std::scalbn(std::bit_cast<float>(a), std::clamp(s(b), -512, 512)));
        case Opcode::PackNorm16: return evaluate_pack_norm16(a, imm);
        case Opcode::NormalizeField: return evaluate_normalized_field(imm, a);
        case Opcode::PackHalfLo: return pack_half_rte(a);
        case Opcode::F32Fma: {
            // Selected finite, normal-or-zero binary32 RTE reference. GLSL.std.450 Fma's
            // Vulkan accuracy is inherited from Mul+Add, so this is not a portable device
            // guarantee. Like the other float ALU evaluator, it is not a guest constant fold.
            const float x = std::bit_cast<float>(a), y = std::bit_cast<float>(b),
                        z = std::bit_cast<float>(c);
            const auto supported = [](float value) {
                return std::isfinite(value) && std::fpclassify(value) != FP_SUBNORMAL;
            };
            if (!supported(x) || !supported(y) || !supported(z)) return std::nullopt;
            const float d = std::fma(x, y, z);
            if (!supported(d)) return std::nullopt;
            return std::bit_cast<uint32_t>(d);
        }
        case Opcode::F32AddNoContraction: return evaluate_op(Opcode::F32Add, imm, a, b, c);
        case Opcode::F32MulNoContraction: return evaluate_op(Opcode::F32Mul, imm, a, b, c);
        case Opcode::F32Add:
        case Opcode::F32Sub:
        case Opcode::F32Mul:
        case Opcode::F32NMin:
        case Opcode::F32NMax: {
            const float x = std::bit_cast<float>(a), y = std::bit_cast<float>(b);
            // No payload guess for an unspecified two-NaN result, nor a zero-sign guess
            // for GLSL NMin/NMax. This evaluator is not added to the guest constant folder.
            const bool minmax = op == Opcode::F32NMin || op == Opcode::F32NMax;
            if (minmax && ((std::isnan(x) && std::isnan(y)) || (x == 0 && y == 0)))
                return std::nullopt;
            const float d = op == Opcode::F32Add    ? x + y
                            : op == Opcode::F32Sub  ? x - y
                            : op == Opcode::F32Mul  ? x * y
                            : op == Opcode::F32NMin ? std::fmin(x, y)
                                                    : std::fmax(x, y);
            if (std::isnan(d)) return std::nullopt;
            return std::bit_cast<uint32_t>(d);
        }
        case Opcode::FfbhU32: return a ? uint32_t(std::countl_zero(a)) : 0xffffffffu;
        case Opcode::F32Fract:
        case Opcode::F32Trunc:
        case Opcode::F32Ceil:
        case Opcode::F32RoundEven:
        case Opcode::F32Floor:
        case Opcode::F32Exp2:
        case Opcode::F32Log2:
        case Opcode::F32Rcp:
        case Opcode::F32InverseSqrt:
        case Opcode::F32Sqrt:
        case Opcode::F32Sin:
        case Opcode::F32Cos: return evaluate_float_unary(op, a);
        case Opcode::I32ToF32:
        case Opcode::U32ToF32:
        case Opcode::F32ToU32Sat:
        case Opcode::F32ToI32Sat:
        case Opcode::BfeS:
        case Opcode::UnpackHalf:
        case Opcode::F32Ge: return evaluate_conversion(op, imm, a, b, c);
        case Opcode::IAdd: return a + b;
        case Opcode::ISub: return a - b;
        case Opcode::IMul: return a * b;
        case Opcode::UMulHi: return static_cast<uint32_t>((uint64_t{a} * b) >> 32);
        case Opcode::SMulHi:
            return static_cast<uint32_t>(static_cast<uint64_t>(int64_t{s(a)} * s(b)) >> 32);
        case Opcode::And: return a & b;
        case Opcode::Or: return a | b;
        case Opcode::Xor: return a ^ b;
        case Opcode::Not: return ~a;
        case Opcode::BitReverse: return bit_reverse(a);
        case Opcode::FindUMsbU32:
            return a ? Known(31u - static_cast<uint32_t>(std::countl_zero(a))) : Known(UINT32_MAX);
        case Opcode::BitCount: return static_cast<uint32_t>(std::popcount(a));
        case Opcode::Shl: return b < 32 ? Known(a << b) : std::nullopt;
        case Opcode::LShr: return b < 32 ? Known(a >> b) : std::nullopt;
        case Opcode::AShr: return b < 32 ? Known(static_cast<uint32_t>(s(a) >> b)) : std::nullopt;
        case Opcode::SMin: return static_cast<uint32_t>(std::min(s(a), s(b)));
        case Opcode::SMax: return static_cast<uint32_t>(std::max(s(a), s(b)));
        case Opcode::UMin: return std::min(a, b);
        case Opcode::UMax: return std::max(a, b);
        case Opcode::BfeU: return bfe_u(a, b, c);
        case Opcode::Select:
        case Opcode::BoolSelect: return a ? b : c;
        case Opcode::IEq: return static_cast<uint32_t>(a == b);
        case Opcode::INe: return static_cast<uint32_t>(a != b);
        case Opcode::ULt: return static_cast<uint32_t>(a < b);
        case Opcode::ULe: return static_cast<uint32_t>(a <= b);
        case Opcode::UGt: return static_cast<uint32_t>(a > b);
        case Opcode::UGe: return static_cast<uint32_t>(a >= b);
        case Opcode::SLt: return static_cast<uint32_t>(s(a) < s(b));
        case Opcode::SLe: return static_cast<uint32_t>(s(a) <= s(b));
        case Opcode::SGt: return static_cast<uint32_t>(s(a) > s(b));
        case Opcode::SGe: return static_cast<uint32_t>(s(a) >= s(b));
        // A 64-bit word does not fit these 32-bit arguments: evaluate_wide routes them to
        // evaluate_integer64 (eval_integer64.cpp) and evaluate_float64 (eval_float64.cpp).
        case Opcode::PackU64:
        case Opcode::U64Lo:
        case Opcode::U64Hi:
        case Opcode::U64Eq:
        case Opcode::U64Ne:
        case Opcode::U64Lt:
        case Opcode::U64Le:
        case Opcode::U64Gt:
        case Opcode::U64Ge:
        case Opcode::U64Shl:
        case Opcode::U64LShr:
        case Opcode::U64AShr:
        case Opcode::U64Bfe:
        case Opcode::F64Lt:
        case Opcode::F64Eq:
        case Opcode::F64Le:
        case Opcode::F64Gt:
        case Opcode::F64Lg:
        case Opcode::F64Ge:
        case Opcode::F64UnordLt:
        case Opcode::F64UnordEq:
        case Opcode::F64UnordLe:
        case Opcode::F64UnordGt:
        case Opcode::F64UnordNe:
        case Opcode::F64UnordGe: return std::nullopt;
        // Cross-lane: one invocation cannot know another's value (eval_lanes.hpp).
        case Opcode::LaneId:
        case Opcode::Shuffle: return std::nullopt;
    }
    return std::nullopt;
}

std::vector<Known64> evaluate_wide(const Block& block, std::span<const Known64> inputs) {
    std::vector<Known64> values(block.size());
    for (size_t i = 0; i < block.size(); ++i) {
        const Inst& inst = block.insts()[i];
        if (inst.op == Opcode::Input) {
            if (inst.imm < inputs.size()) values[i] = inputs[inst.imm];
            if (inst.type != Type::U64 && values[i] && *values[i] > UINT32_MAX)
                values[i] = std::nullopt;
            continue;
        }
        const uint8_t arity = opcode_info(inst.op).arity;
        uint64_t args[3] = {0, 0, 0};
        bool known = true;
        for (uint8_t slot = 0; slot < arity; ++slot) {
            const Known64& arg = values[inst.args[slot].index];
            known = known && arg.has_value();
            if (arg) args[slot] = *arg;
        }
        if (!known) continue;
        if (is_integer64_opcode(inst.op)) {
            values[i] = evaluate_integer64(inst.op, args[0], args[1], args[2]);
            continue;
        }
        if (is_float64_opcode(inst.op)) {
            values[i] = evaluate_float64(inst.op, args[0], args[1]);
            continue;
        }
        // Every other opcode's arguments are U32 or Bool, which fit in 32 bits.
        const Known v = evaluate_op(inst.op, inst.imm, static_cast<uint32_t>(args[0]),
                                    static_cast<uint32_t>(args[1]), static_cast<uint32_t>(args[2]));
        if (v) values[i] = *v;
    }
    return values;
}

std::vector<Known> evaluate(const Block& block, std::span<const Known> inputs) {
    const std::vector<Known64> wide_inputs(inputs.begin(), inputs.end());
    const std::vector<Known64> wide = evaluate_wide(block, wide_inputs);
    std::vector<Known> values(block.size());
    for (size_t i = 0; i < block.size(); ++i)
        if (wide[i] && block.insts()[i].type != Type::U64)
            values[i] = static_cast<uint32_t>(*wide[i]);
    return values;
}

} // namespace prosper::gpu::ir
