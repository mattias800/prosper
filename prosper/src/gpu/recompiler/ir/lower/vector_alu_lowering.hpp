// vector_alu_lowering.hpp — emit_alu's entry points for the vector ALU formats moved onto the IR
// (ADR 0012). Same contract as scalar_alu_lowering.hpp: the return value is what emit_alu returns,
// and `ok` is cleared when the instruction is an ALU op this stage cannot express.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::ir {

bool lower_vop2(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave);

} // namespace prosper::gpu::ir
