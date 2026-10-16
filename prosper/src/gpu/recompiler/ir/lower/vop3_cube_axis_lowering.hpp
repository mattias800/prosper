// The caller retains operand/result modifiers and EXEC-predicated destination publication.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using CubeAxisSource = std::function<uint32_t(int)>;
using CubeAxisResult = std::function<uint32_t(uint32_t)>;
bool lower_vop3_cube_axis(SpirvCompute&, RegState&, const Rdna2Inst&, bool&, bool,
                          const CubeAxisSource&, const CubeAxisResult&);
}   // namespace prosper::gpu::ir
