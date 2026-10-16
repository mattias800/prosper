// Value helper only: true means this subgroup owns D, even when an operand cleared ok.
// This helper owns ordinary source-modifier refusal; the caller retains CLAMP and predication.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop3_ternary_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                bool allow_wave);
}
