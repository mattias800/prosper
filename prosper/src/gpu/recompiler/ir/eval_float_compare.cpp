// eval_float_compare.cpp — the evaluation half of the binary32 relations (see ir.hpp).
//
// SPIR-V's ordered comparisons (OpFOrd*) are false when either operand is a NaN, and the
// unordered ones (OpFUnord*) are true. A relation is IEEE-754's on the values, so +0 equals -0 and
// each infinity compares as the extreme of its sign. Denormal words compare as their values.
#include "gpu/recompiler/ir/eval_float_compare.hpp"

#include <bit>
#include <cmath>

namespace prosper::gpu::ir {

Known evaluate_float_compare(Opcode op, uint32_t a, uint32_t b) {
    const float x = std::bit_cast<float>(a), y = std::bit_cast<float>(b);
    const bool unordered = std::isnan(x) || std::isnan(y);
    switch (op) {
        case Opcode::F32Le: return uint32_t(!unordered && x <= y);
        case Opcode::F32Gt: return uint32_t(!unordered && x > y);
        case Opcode::F32Lg: return uint32_t(!unordered && x != y);
        case Opcode::F32UnordLt: return uint32_t(unordered || x < y);
        case Opcode::F32UnordEq: return uint32_t(unordered || x == y);
        case Opcode::F32UnordLe: return uint32_t(unordered || x <= y);
        case Opcode::F32UnordGt: return uint32_t(unordered || x > y);
        case Opcode::F32UnordGe: return uint32_t(unordered || x >= y);
        default: return std::nullopt;
    }
}

} // namespace prosper::gpu::ir
