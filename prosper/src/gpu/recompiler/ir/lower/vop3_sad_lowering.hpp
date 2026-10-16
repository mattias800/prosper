// Value-stage ownership; the caller keeps its common destination and predication stages.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop3_sad(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave);
}
