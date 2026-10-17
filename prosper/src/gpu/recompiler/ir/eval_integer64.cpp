// eval_integer64.cpp — the evaluation half of the 64-bit word opcodes (see ir.hpp).
#include "gpu/recompiler/ir/eval_integer64.hpp"

namespace prosper::gpu::ir {

bool is_integer64_opcode(Opcode op) {
    return op >= Opcode::PackU64 && op <= Opcode::U64Ge;
}

Known64 evaluate_integer64(Opcode op, uint64_t a, uint64_t b) {
    switch (op) {
        case Opcode::PackU64: return (b << 32) | (a & 0xffffffffu);
        case Opcode::U64Lo: return a & 0xffffffffu;
        case Opcode::U64Hi: return a >> 32;
        case Opcode::U64Eq: return uint64_t(a == b);
        case Opcode::U64Ne: return uint64_t(a != b);
        case Opcode::U64Lt: return uint64_t(a < b);
        case Opcode::U64Le: return uint64_t(a <= b);
        case Opcode::U64Gt: return uint64_t(a > b);
        case Opcode::U64Ge: return uint64_t(a >= b);
        default: return std::nullopt;
    }
}

} // namespace prosper::gpu::ir
