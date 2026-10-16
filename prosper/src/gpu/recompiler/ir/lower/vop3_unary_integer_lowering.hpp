// Numeric result stage only; root retains source proofs and predicated destination publication.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop3_unary_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                              bool allow_wave);
}
