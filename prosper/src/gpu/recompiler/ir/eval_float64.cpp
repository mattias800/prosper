// eval_float64.cpp — the evaluation half of the binary64 relations (see ir.hpp).
#include "gpu/recompiler/ir/eval_float64.hpp"

#include <bit>
#include <cmath>

namespace prosper::gpu::ir {

bool is_float64_opcode(Opcode op) {
    return op >= Opcode::F64Lt && op <= Opcode::F64UnordGe;
}

Known64 evaluate_float64(Opcode op, uint64_t a, uint64_t b) {
    const double x = std::bit_cast<double>(a), y = std::bit_cast<double>(b);
    const bool unordered = std::isnan(x) || std::isnan(y);
    switch (op) {
        case Opcode::F64Lt: return uint64_t(!unordered && x < y);
        case Opcode::F64Eq: return uint64_t(!unordered && x == y);
        case Opcode::F64Le: return uint64_t(!unordered && x <= y);
        case Opcode::F64Gt: return uint64_t(!unordered && x > y);
        case Opcode::F64Lg: return uint64_t(!unordered && x != y);
        case Opcode::F64Ge: return uint64_t(!unordered && x >= y);
        case Opcode::F64UnordLt: return uint64_t(unordered || x < y);
        case Opcode::F64UnordEq: return uint64_t(unordered || x == y);
        case Opcode::F64UnordLe: return uint64_t(unordered || x <= y);
        case Opcode::F64UnordGt: return uint64_t(unordered || x > y);
        case Opcode::F64UnordNe: return uint64_t(unordered || x != y);
        case Opcode::F64UnordGe: return uint64_t(unordered || x >= y);
        default: return std::nullopt;
    }
}

} // namespace prosper::gpu::ir
