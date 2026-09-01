#include "gpu/resources/mip_chain_plan.hpp"

#include "gpu/texture/tile.hpp"

#include <algorithm>

namespace prosper::gpu {

uint32_t full_mip_chain_levels(uint32_t width, uint32_t height) {
    if (!width || !height) return 0;
    uint32_t levels = 1;
    for (uint32_t extent = std::max(width, height); extent > 1u; extent >>= 1) ++levels;
    return levels;
}

namespace {

// The sampled formats whose compute upload is a straight per-texel copy. Anything that widens,
// broadcasts or repacks on the way to the staging buffer (R8 coverage broadcast, the RGBA8
// narrowing of FP16, packed R11G11B10, block-compressed decode, a depth view) is deliberately not
// here: a per-level upload would have to reproduce that conversion at every extent, and this first
// landing keeps the level-0 bytes byte-identical to what the single-level path already produces.
//
// IF YOU WIDEN THIS LIST, read `live_compute.cpp`'s conversion cascade first. This whitelist picks
// the level COUNT; a separate branch condition there does the copying, and only that branch can
// write levels 1..N-1. Adding a format here that the branch does not take would otherwise leave
// those levels as whatever the staging allocation held. That case is caught -- the cascade ends in
// a drift detector that declines instead of uploading it -- so the failure is a declined dispatch,
// not corrupt texels; but the fix is to teach the cascade, not to widen only this list. The two are
// deliberately NOT collapsed: this function has callers (the emitter, the compile key) that have no
// BoundImage, no reflected descriptor and no live target, so the count must stay derivable from the
// descriptor alone.
bool native_straight_copy_sampled_format(DataFormat format, uint32_t components) {
    switch (format) {
        case DataFormat::Float32:
            return components == 1 || components == 2 || components == 4;
        case DataFormat::Uint32:
        case DataFormat::Uint16:
        case DataFormat::Unorm16:
        case DataFormat::Uint8:
            return components == 1 || components == 2 || components == 4;
        case DataFormat::Unorm8:
            return components == 2 || components == 4;
        default:
            return false;
    }
}

} // namespace

MipChainPlan shader_resource_mip_chain_plan(const ShaderResource& resource) {
    MipChainPlan plan;
    if (resource.cls != ResourceClass::Texture) return plan;
    if (resource.declared_mip_levels < 2u || resource.sample_count != 1u) return plan;
    // A selected level that is itself packed in the shared tail leaves `gpu_addr` at the allocation
    // base while every sibling level shares that same block; the offsets below assume the ordinary
    // "selected level owns a disjoint byte range" form and must not be applied to it.
    if (resource.in_mip_tail) return plan;
    // A layered or volume view selects one slice of a per-slice chain, and a compressed base is not
    // ordinary tiled texels at any level. Both remain fail-closed.
    if (resource.depth != 1u || resource.layer_stride_bytes || resource.layer_mip_offset_bytes)
        return plan;
    if (resource.compression_enabled || resource.metadata_addr) return plan;
    if (resource.img_dim != 1u && resource.img_dim != 5u) return plan;

    const uint32_t element_width = resource.mip_chain_element_width;
    const uint32_t element_height = resource.mip_chain_element_height;
    const uint32_t bytes_per_block = resource.mip_chain_bytes_per_block;
    const uint32_t max_mip = resource.mip_chain_max_level;
    if (!element_width || !element_height || !bytes_per_block || max_mip >= 16u) return plan;
    // Only the view that starts at the allocation's own level zero is modelled. A shifted BASE_LEVEL
    // would need every offset rebased onto the selected level, and no measured title issues one
    // alongside a dynamic mip operand.
    if (resource.mip_chain_base_level != 0u) return plan;
    // Uncompressed only: for a block format the element grid is smaller than the texel extent, and
    // this equality is what proves the two agree.
    if (element_width != resource.width || element_height != resource.height) return plan;
    if (resource.declared_mip_levels > max_mip + 1u) return plan;
    // Linear chains place each level on a 256-byte-aligned pitch, which the per-level copy below
    // does not yet reproduce; every measured multi-level guest texture here is tiled.
    if (!tile_mode_is_tiled(resource.tile_mode)) return plan;

    const uint32_t full = full_mip_chain_levels(resource.width, resource.height);
    if (!full) return plan;
    const uint32_t level_count = std::min(resource.declared_mip_levels, full);
    if (level_count < 2u) return plan;

    const size_t allocation_bytes = tiled_mip_chain_bytes(
        element_width, element_height, bytes_per_block, resource.tile_mode, max_mip);
    if (!allocation_bytes) return plan;

    plan.levels.resize(level_count);
    for (uint32_t level = 0; level < level_count; ++level) {
        const TiledMipLevelLayout layout = tiled_mip_level_layout(
            element_width, element_height, bytes_per_block, resource.tile_mode, max_mip, level);
        if (!layout.supported) return {};
        MipChainLevel& out = plan.levels[level];
        out.width = std::max(resource.width >> level, 1u);
        out.height = std::max(resource.height >> level, 1u);
        out.in_tail = layout.in_tail;
        if (layout.in_tail) {
            // Tail levels are addressed by element coordinate inside the allocation's first block.
            if (!layout.tail_block_bytes) return {};
            out.byte_offset = 0;
            out.byte_size = layout.tail_block_bytes;
            out.tail_x = layout.tail_x;
            out.tail_y = layout.tail_y;
            out.tail_block_bytes = layout.tail_block_bytes;
        } else {
            out.byte_offset = layout.byte_offset;
            out.byte_size = tiled_surface_bytes(
                out.width, out.height, resource.tile_mode, 0, bytes_per_block);
            if (!out.byte_size) return {};
        }
        if (out.byte_offset > allocation_bytes ||
            out.byte_size > allocation_bytes - out.byte_offset)
            return {};
    }
    // The selected level must be the one the resource's own fields describe, or `gpu_addr` and
    // `levels[0]` name different bytes.
    if (plan.levels[0].in_tail) return {};
    plan.valid = true;
    plan.level_count = level_count;
    plan.allocation_bytes = allocation_bytes;
    return plan;
}

uint32_t shader_resource_compute_mip_chain_levels(const ShaderResource& resource) {
    // Cheapest possible rejection first. This is called per resource on the compile-key path, which
    // runs for every dispatch, and almost every texture declares a single level -- so the common
    // answer must not reach the plan's allocation.
    if (resource.declared_mip_levels < 2u ||
        resource.cls != ResourceClass::Texture) return 1u;
    const uint32_t components = resource.num_components ? resource.num_components : 1u;
    if (!native_straight_copy_sampled_format(resource.format, components)) return 1u;
    if (resource.depth_compare) return 1u;
    // The allocation base sits BELOW gpu_addr, so the source must be ordinary guest memory. A
    // host_data-backed resource (capture replay, synthetic fixtures) exposes only the selected
    // level's span and cannot answer for the rest of the chain.
    if (resource.host_data) return 1u;
    const uint32_t component_bytes = data_format_bytes(resource.format);
    if (!component_bytes ||
        component_bytes * components != resource.mip_chain_bytes_per_block)
        return 1u;
    const MipChainPlan plan = shader_resource_mip_chain_plan(resource);
    return plan.valid ? plan.level_count : 1u;
}

} // namespace prosper::gpu
