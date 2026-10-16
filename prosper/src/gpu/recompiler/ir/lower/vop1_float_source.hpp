// Compatibility source preparation retained at the original VOP1 pipeline position.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
void lower_vop1_float_source(SpirvCompute&, const Rdna2Inst&, uint32_t&);
}
