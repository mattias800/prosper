// vector_cndmask.hpp — the ONE definition of v_cndmask_b32's per-lane value (ADR 0012).
//
// D = mask bit ? S1 : S0. The definition sees only the lane's mask bit, a Bool. Which mask that bit
// comes from (VCC for VOP2, SRC2's VCC or SGPR pair for VOP3) and the source modifiers or SDWA
// selects applied to S0 and S1 are the lowering's (ir/lower/vop2.cpp, ir/lower/vop3_cndmask.cpp).
#pragma once

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <span>

namespace prosper::gpu {
struct SpirvCompute;
}

namespace prosper::gpu::ir {

struct VectorCndmaskDef {
    Rdna2Format fmt;
    uint32_t opcode;
    Value (*build)(Builder& ir, Value mask, Value when_set, Value when_clear);
};

std::span<const VectorCndmaskDef> vector_cndmask_defs();
const VectorCndmaskDef* find_vector_cndmask(Rdna2Format fmt, uint32_t opcode);

// One lane: `mask` is 0 or 1.
Known eval_vector_cndmask(const VectorCndmaskDef& def, Known mask, Known s0, Known s1);
// The arguments are in SpirvCompute::sel's order (condition, value when set, value when clear), so a
// call written where the old `b.sel(m, s1, s0)` was evaluates its argument expressions the same way.
uint32_t emit_vector_cndmask(SpirvCompute& b, const VectorCndmaskDef& def, uint32_t mask,
                             uint32_t s1, uint32_t s0);

}   // namespace prosper::gpu::ir
