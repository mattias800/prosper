// eval_integer64.cpp — the evaluation half of the 64-bit word opcodes (see ir.hpp).
#include "gpu/recompiler/ir/eval_integer64.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
#include <optional>

namespace prosper::gpu::ir {

bool is_integer64_opcode(Opcode op) {
    return op >= Opcode::PackU64 && op <= Opcode::U64Bfe;
}

Known64 evaluate_integer64(Opcode op, uint64_t a, uint64_t b, uint64_t c) {
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
        case Opcode::U64Shl: return b < 64 ? Known64(a << b) : std::nullopt;
        case Opcode::U64LShr: return b < 64 ? Known64(a >> b) : std::nullopt;
        case Opcode::U64AShr:
            if (b >= 64) return std::nullopt;
            if (!b || !(a >> 63)) return a >> b;
            return (a >> b) | (~uint64_t{0} << (64 - b));
        case Opcode::U64Bfe: {
            if (b >= 64 || c > UINT32_MAX) return std::nullopt;
            const uint64_t count = c < 64 - b ? c : 64 - b;
            if (!count) return uint64_t(0);
            const uint64_t mask = count == 64 ? UINT64_MAX : (uint64_t(1) << count) - 1;
            return (a >> b) & mask;
        }
        default: return std::nullopt;
    }
}

} // namespace prosper::gpu::ir
