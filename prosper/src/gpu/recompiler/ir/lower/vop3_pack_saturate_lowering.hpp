// Caller retains destination capture and EXEC-predicated publication.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop3_pack_saturate(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                              bool allow_wave);
}
