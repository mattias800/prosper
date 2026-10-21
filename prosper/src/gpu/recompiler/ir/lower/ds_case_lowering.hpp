// DS case routes mechanically relocated from emit_alu; direct SPIR-V emission is unchanged.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::gpu::ir {
bool lower_ds_case(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave);
} // namespace prosper::gpu::ir
