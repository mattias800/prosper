// Owns only the existing VOP1 F32 unary value cases. False leaves the legacy switch in charge.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop1_unary_float(SpirvCompute& b, const Rdna2Inst& in, uint32_t a, uint32_t& d);
}
