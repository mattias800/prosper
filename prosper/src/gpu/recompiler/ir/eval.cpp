// eval.cpp — the evaluation half of each IR opcode's definition (see ir.hpp).
#include "gpu/recompiler/ir/eval.hpp"

#include <algorithm>
#include <bit>
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
    }
    return std::nullopt;
}

std::vector<Known> evaluate(const Block& block, std::span<const Known> inputs) {
    std::vector<Known> values(block.size());
    for (size_t i = 0; i < block.size(); ++i) {
        const Inst& inst = block.insts()[i];
        if (inst.op == Opcode::Input) {
            if (inst.imm < inputs.size()) values[i] = inputs[inst.imm];
            continue;
        }
        const uint8_t arity = opcode_info(inst.op).arity;
        uint32_t args[3] = {0, 0, 0};
        bool known = true;
        for (uint8_t slot = 0; slot < arity; ++slot) {
            const Known& arg = values[inst.args[slot].index];
            known = known && arg.has_value();
            if (arg) args[slot] = *arg;
        }
        if (known) values[i] = evaluate_op(inst.op, inst.imm, args[0], args[1], args[2]);
    }
    return values;
}

} // namespace prosper::gpu::ir
