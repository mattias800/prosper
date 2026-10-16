// Caller owns common destination and EXEC-predicated publication stages.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop3_permute(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                        bool allow_wave);
}
