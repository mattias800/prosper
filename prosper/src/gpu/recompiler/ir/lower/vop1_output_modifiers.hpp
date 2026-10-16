#pragma once
#include <cstdint>
namespace prosper::gpu {
struct Rdna2Inst;
struct SpirvCompute;
namespace ir {
void lower_vop1_output_modifiers(SpirvCompute& b, const Rdna2Inst& in, uint32_t& d, bool& ok);
} // namespace ir
} // namespace prosper::gpu
