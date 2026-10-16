// Numeric stage only: existing source proofs and EXEC-predicated publication stay with the caller.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
} // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop3_float_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok);
} // namespace prosper::gpu::ir
