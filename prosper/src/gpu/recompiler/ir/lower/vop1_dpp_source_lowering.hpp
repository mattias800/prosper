// The relocated VOP1 DPP stage; admission and non-compute paths retain their original owner.
#pragma once
#include <cstdint>
#include <optional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
std::optional<bool> lower_vop1_dpp_source(SpirvCompute&, RegState&, const Rdna2Inst&, bool&,
                                          uint32_t&, uint32_t&);
}
