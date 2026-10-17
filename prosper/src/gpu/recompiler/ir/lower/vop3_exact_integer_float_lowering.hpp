// A plain, explicit, nonzero integer constant whose F32 result is exact needs no MODE guess.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
namespace ir {
bool lower_vop3_exact_integer_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok);
} // namespace ir
} // namespace prosper::gpu
