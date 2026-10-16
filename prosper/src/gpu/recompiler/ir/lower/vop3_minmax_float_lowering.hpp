// Root-owned float modifiers and publication remain at their original stage boundaries.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using Vop3MinmaxFloatSource = std::function<uint32_t(int)>;
using Vop3MinmaxFloatResult = std::function<uint32_t(uint32_t)>;
bool lower_vop3_minmax_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                             bool allow_wave, const Vop3MinmaxFloatSource& fv,
                             const Vop3MinmaxFloatResult& fresult);
}   // namespace prosper::gpu::ir
