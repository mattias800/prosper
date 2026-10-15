// scalar_alu_lowering.hpp — emit_alu's entry points for the scalar ALU formats (ADR 0012).
//
// Each lowers one decoded SOP1/SOP2/SOPC/SOPK instruction into `b`, updating `rs`, with emit_alu's
// contract: the return value is what emit_alu returns, and `ok` is cleared when the instruction is
// an ALU op this stage cannot express.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::ir {

bool lower_sop1(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave);
bool lower_sop2(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave);
bool lower_sopc(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave);
bool lower_sopk(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave);

} // namespace prosper::gpu::ir
