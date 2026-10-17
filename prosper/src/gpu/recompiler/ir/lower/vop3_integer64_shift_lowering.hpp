// Value and placement of the existing VOP3 integer64 shifts; source rules remain unchanged.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop3_integer64_shift(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok);
}
