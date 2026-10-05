// PROSPER_NATIVE_BC_CHAIN_AUDIT: does a packed native BC mip chain agree with itself?
#pragma once
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::frontend::submit_renderer {

// A real mip level is close to the 2x2 box of the level above it; a foreign, misplaced or unwritten
// region is not. `level_mad[i]` is the mean absolute RGB difference between level i + 1 and the box
// of level i, over every level the packed bytes cover. `shifted_control` is level 1 against a
// half-width-shifted box of level 0: on a flat texture both it and level_mad[0] are near zero,
// which is "nothing to tell apart", not "placed correctly".
struct NativeBcChainAudit {
    std::vector<double> level_mad;
    double shifted_control = 0.0;
};

// `chain` is the layout the native upload stages: every level's 4x4 blocks, detiled, level 0
// first, with no padding between levels. Levels the bytes do not cover are simply absent from the
// result. Returns an empty result for a non-BC format.
NativeBcChainAudit audit_native_bc_chain(const uint8_t* chain, size_t chain_bytes, uint32_t width,
                                         uint32_t height, uint32_t levels, uint32_t block_bytes,
                                         prosper::gpu::DataFormat format);

// One `[native-bc-audit]` line for a texture identity: the level-1 pair the line has always
// carried, then `levels_mad=` for levels 1..N-1 with a `t` suffix on tail-packed levels.
void log_native_bc_chain_audit(const std::vector<uint8_t>& chain,
                               const prosper::gpu::MipChainPlan& plan, uint32_t levels,
                               uint32_t block_bytes, const prosper::gpu::ShaderResource& resource);

}   // namespace prosper::frontend::submit_renderer
