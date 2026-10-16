// Root-owned source/result modifiers retain their original positions and clamp-routing flag.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using Vop3FloatSource = std::function<uint32_t(int)>;
using Vop3FloatResult = std::function<uint32_t(uint32_t)>;
bool lower_vop3_mad_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, const Vop3FloatSource& fv,
                          const Vop3FloatResult& fresult);
}   // namespace prosper::gpu::ir
