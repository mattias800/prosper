// Plain explicit-normal half constants only; root retains the final EXEC-predicated write.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
} // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop3_half_float_exact(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok);
} // namespace prosper::gpu::ir
