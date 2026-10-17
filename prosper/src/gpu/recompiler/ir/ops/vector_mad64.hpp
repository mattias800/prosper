// The MAD64 lane's two raw result words and unmasked carry, independent of register ownership.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
struct Mad64In {
    Value s0, s1, add_lo, add_hi;
};
struct Mad64Out {
    Value lo, hi, carry;
};
struct VectorMad64Def {
    Rdna2Format fmt;
    uint32_t opcode;
    Mad64Out (*build)(Builder&, const Mad64In&);
};
std::span<const VectorMad64Def> vector_mad64_defs();
const VectorMad64Def* find_vector_mad64(Rdna2Format fmt, uint32_t opcode);
struct KnownMad64 {
    Known lo, hi, carry;
};
KnownMad64 eval_vector_mad64(const VectorMad64Def& def, Known s0, Known s1, Known add_lo,
                             Known add_hi);
struct Mad64Ids {
    uint32_t lo, hi, carry;
};
Mad64Ids emit_vector_mad64(SpirvCompute& b, const VectorMad64Def& def, uint32_t s0, uint32_t s1,
                           uint32_t add_lo, uint32_t add_hi);
}   // namespace prosper::gpu::ir
