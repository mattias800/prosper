// Native DS shuffle-only ALU routes. A disengaged result leaves the remaining DS case to its owner.
#pragma once

#include <optional>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu

namespace prosper::gpu::ir {
std::optional<bool> lower_ds_cross_lane(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                        bool& ok);
}   // namespace prosper::gpu::ir
