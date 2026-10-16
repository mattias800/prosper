// Existing integer-dot entry; the caller retains all shared admission guards.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
namespace ir {
bool lower_vop3p_integer_dot(SpirvCompute&, RegState&, const Rdna2Inst&, bool&, bool);
}
}   // namespace prosper::gpu
