#pragma once
namespace prosper::gpu {
struct Rdna2Inst;
struct RegState;
struct SpirvCompute;
namespace ir {
bool lower_vop3_subword_to_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok);
} // namespace ir
} // namespace prosper::gpu
