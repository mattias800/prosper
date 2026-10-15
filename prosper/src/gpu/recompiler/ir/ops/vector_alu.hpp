// vector_alu.hpp — the ONE definition of each per-lane vector-ALU data operation (ADR 0012).
//
// The vector counterpart of scalar_alu.hpp: a VectorOpDef's `build` appends D = f(S0, S1) for one
// lane to an IR Block. The lowering emits it (emit_vector_op) and tests evaluate it
// (eval_vector_op) through the same function. Only the operation lives here; DPP and SDWA operand
// transforms, OMOD/CLAMP and the EXEC-predicated VDST write stay in ir/lower/, because they are the
// same for every opcode of a format.
//
// Today: the VOP2-encoded integer and bitwise group. The VOP3 (e64) encodings of the same operations
// are not admitted by emit_alu; admitting them through these definitions is a behaviour change for a
// later, corpus-gated PR.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <span>

namespace prosper::gpu {
struct SpirvCompute;
}

namespace prosper::gpu::ir {

struct VectorOpIn {
    Value s0, s1;
};

struct VectorOpDef {
    Rdna2Format fmt;
    uint32_t opcode;
    Value (*build)(Builder& ir, const VectorOpIn& in);
};

// nullptr when (fmt, opcode) has no vector definition.
const VectorOpDef* find_vector_op(Rdna2Format fmt, uint32_t opcode);
std::span<const VectorOpDef> vector_alu_defs();

// One lane, on known sources.
Known eval_vector_op(const VectorOpDef& def, Known s0, Known s1);
// The SPIR-V id of D, given the ids of the (already transformed) sources.
uint32_t emit_vector_op(SpirvCompute& b, const VectorOpDef& def, uint32_t s0, uint32_t s1);

} // namespace prosper::gpu::ir
