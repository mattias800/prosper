// vector_alu.hpp — the ONE definition of each per-lane vector-ALU data operation (ADR 0012).
//
// The vector counterpart of scalar_alu.hpp: a VectorOpDef's `build` appends D = f(S0, S1) for one
// lane to an IR Block. The lowering emits it (emit_vector_op) and tests evaluate it
// (eval_vector_op) through the same function. Only the operation lives here; DPP and SDWA operand
// transforms, OMOD/CLAMP and the EXEC-predicated VDST write stay in ir/lower/, because they are the
// same for every opcode of a format.
//
// VOP2 data operations and their canonical e64 aliases share these builders. Encoding
// admission, unsupported modifiers and EXEC predication remain in the lowerings.
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

// One arithmetic definition shared by the e32 and e64 data encodings.
Value build_vector_mul_u32_u24(Builder& ir, const VectorOpIn& in);
Value build_vector_min_i32(Builder& ir, const VectorOpIn& in);
Value build_vector_max_i32(Builder& ir, const VectorOpIn& in);
Value build_vector_min_u32(Builder& ir, const VectorOpIn& in);
Value build_vector_max_u32(Builder& ir, const VectorOpIn& in);
Value build_vector_lshrrev_b32(Builder& ir, const VectorOpIn& in);
Value build_vector_ashrrev_i32(Builder& ir, const VectorOpIn& in);
Value build_vector_lshlrev_b32(Builder& ir, const VectorOpIn& in);
Value build_vector_and_b32(Builder& ir, const VectorOpIn& in);
Value build_vector_or_b32(Builder& ir, const VectorOpIn& in);
Value build_vector_xor_b32(Builder& ir, const VectorOpIn& in);
Value build_vector_xnor_b32(Builder& ir, const VectorOpIn& in);
Value build_vector_add_nc_u32(Builder& ir, const VectorOpIn& in);
Value build_vector_sub_nc_u32(Builder& ir, const VectorOpIn& in);
Value build_vector_subrev_nc_u32(Builder& ir, const VectorOpIn& in);

// nullptr when (fmt, opcode) has no vector definition.
const VectorOpDef* find_vector_op(Rdna2Format fmt, uint32_t opcode);
std::span<const VectorOpDef> vector_alu_defs();

// One lane, on known sources.
Known eval_vector_op(const VectorOpDef& def, Known s0, Known s1);
// The SPIR-V id of D, given the ids of the (already transformed) sources.
uint32_t emit_vector_op(SpirvCompute& b, const VectorOpDef& def, uint32_t s0, uint32_t s1);

} // namespace prosper::gpu::ir
