#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
namespace ir {
bool lower_vop3_integer_float_rte(SpirvCompute&, RegState&, const Rdna2Inst&, bool&);
}
} // namespace prosper::gpu
