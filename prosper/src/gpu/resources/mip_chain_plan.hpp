// mip_chain_plan.hpp — where every level of a guest texture's declared mip chain lives, and whether
// the compute backend materializes that chain.
//
// The T# declares a chain length (`declared_mip_levels`); `image_base_level_view` resolves the
// SELECTED level and shifts `gpu_addr`/`width`/`height` onto it. Nothing downstream could place the
// OTHER levels, so the compute backend created every image with `mipLevels = 1` and the recompiler
// declined IMAGE_LOAD_MIP whenever the descriptor declared more than one level — an operation the
// guest genuinely issues (#3048, Sonic Frontiers' three scene-width stage kernels).
//
// This header answers both halves from ONE derivation so they cannot disagree: the backend asks how
// many levels to create and where each level's guest bytes are, and the recompiler asks whether an
// explicit LOD may be emitted at all. Reject-by-default — an unmodelled shape yields `level_count 1`
// and the historical single-level behaviour, never a guessed offset.
#pragma once
#include "gpu/resources/shader_resources.hpp"

#include <cstdint>
#include <vector>

namespace prosper::gpu {

// One level's source placement inside the guest allocation. `byte_offset` is measured from the
// ALLOCATION base (which is below `ShaderResource::gpu_addr` whenever the selected level is not the
// allocation's first stored byte). A packed-tail level shares the allocation's first macroblock, so
// it reports `byte_offset 0` plus the element coordinates the tail-aware detile helpers consume.
struct MipChainLevel {
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t byte_offset = 0;
    uint64_t byte_size = 0;
    uint32_t tail_x = 0;
    uint32_t tail_y = 0;
    uint32_t tail_block_bytes = 0;
    bool in_tail = false;
};

struct MipChainPlan {
    bool valid = false;
    uint32_t level_count = 1;
    // levels[0] is the resource's own selected level, so `allocation_base` is
    // `gpu_addr - levels[0].byte_offset` and `allocation_bytes` bounds the readable span.
    uint64_t allocation_bytes = 0;
    std::vector<MipChainLevel> levels;
};

// Geometry only: where each declared level lives. Valid only when EVERY level of the chain resolves
// through the modelled GFX10 thin-2D placement.
MipChainPlan shader_resource_mip_chain_plan(const ShaderResource& resource);

// The number of mip levels the COMPUTE backend materializes for this resource. 1 means the
// historical single-level image. Called by `live_compute`'s image creation and by the recompiler's
// MIMG lowering; a divergence between the two would emit an explicit LOD against a level that does
// not exist, so both must read this one function (the #2265 lesson).
uint32_t shader_resource_compute_mip_chain_levels(const ShaderResource& resource);

// The complete chain implied by a level-zero extent (floor(log2(max(w,h))) + 1).
uint32_t full_mip_chain_levels(uint32_t width, uint32_t height);

} // namespace prosper::gpu
