// Numeric data lowering only. The caller owns final CLAMP refusal and predication.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop3_binary_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                               bool allow_wave);
} // namespace prosper::gpu::ir
