// Integer VOP1 tail only. emit_alu retains operand lookup, every special route and VDST writes.
// Return true when this subset owns the instruction (even if ok was cleared); false to fall back.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop1_integer(SpirvCompute& b, const Rdna2Inst& in, uint32_t a, uint32_t& d, bool& ok);
}
