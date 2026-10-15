// scalar_alu.hpp — the ONE definition of each RDNA2 scalar-ALU data operation (ADR 0012, #2514).
//
// A ScalarOpDef's `build` appends the operation to an IR Block. That single function is both what
// the constant folder evaluates (eval_scalar_op, through ir/eval) and what the emitter writes as
// SPIR-V (emit_scalar_op, through ir/emit), so the two cannot disagree about an operation's
// semantics: before this, SOP2 was implemented once in emit_alu and again in the folder's switch.
//
// Only the plain data semantics live here: D and SCC as functions of S0, S1 and the incoming SCC.
// Which register receives D, wave-mask domains and descriptor bookkeeping stay in ir/lower/.
// Which ops write SCC and which read it is the instruction-effects table's answer
// (rdna2_instruction_effects.hpp); test_ir_scalar_alu_ops checks every def against it.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <span>

namespace prosper::gpu {
struct SpirvCompute;
}

namespace prosper::gpu::ir {

// Where an instruction's S0 and S1 come from.
enum class ScalarOperands : uint8_t {
    Src0,          // SOP1: S0 = SSRC0
    Src0Src1,      // SOP2, SOPC: S0 = SSRC0, S1 = SSRC1
    DstSimmSext,   // SOPK: S0 = the SDST register's current value, S1 = sign-extended SIMM16
    DstSimmZext,   // SOPK unsigned compares: S1 = zero-extended SIMM16
    SimmSext,      // s_movk_i32: S1 = sign-extended SIMM16, no S0
};

struct ScalarOpIn {
    Value s0, s1, scc;
};
// `d` is invalid for a compare; `scc` is invalid when the operation leaves SCC unchanged.
struct ScalarOpOut {
    Value d, scc;
};

struct ScalarOpDef {
    Rdna2Format fmt;
    uint32_t opcode;
    ScalarOperands operands;
    ScalarOpOut (*build)(Builder& ir, const ScalarOpIn& in);
};

// nullptr when (fmt, opcode) is not a plain scalar data operation.
const ScalarOpDef* find_scalar_op(Rdna2Format fmt, uint32_t opcode);
std::span<const ScalarOpDef> scalar_alu_defs();       // SOP1, SOP2, SOPK
std::span<const ScalarOpDef> scalar_compare_defs();   // SOPC, SOPK compares

// The SIMM16 operand as the definition reads it (sign- or zero-extended per `operands`).
uint32_t scalar_op_simm16(const ScalarOpDef& def, const Rdna2Inst& in);

struct ScalarOpValue {
    Known d, scc;
};
// Constant evaluation; `scc` is the incoming SCC (nullopt when unknown).
ScalarOpValue eval_scalar_op(const ScalarOpDef& def, Known s0, Known s1, Known scc);

struct ScalarOpIds {
    uint32_t d = 0, scc = 0;
};
// Emission: the SPIR-V ids of S0, S1 and the incoming SCC (0 when absent).
ScalarOpIds emit_scalar_op(SpirvCompute& b, const ScalarOpDef& def, uint32_t s0, uint32_t s1,
                           uint32_t scc);

} // namespace prosper::gpu::ir
