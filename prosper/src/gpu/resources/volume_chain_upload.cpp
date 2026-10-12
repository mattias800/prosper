#include "gpu/resources/volume_chain_upload.hpp"

#include "gpu/texture/tile.hpp"

#include <cstring>

namespace prosper::gpu {

bool mip_chain_plan_is_volume_tail(const MipChainPlan& plan) {
    if (!plan.valid || plan.levels.empty()) return false;
    for (const MipChainLevel& level : plan.levels) {
        if (!level.in_tail || !level.depth) return false;
    }
    return true;
}

namespace {

// w*h*d texels at backend width, saturating instead of wrapping: a wrapped size would
// alias one level's staging onto another's.
bool level_backend_bytes(const MipChainLevel& level, uint64_t backend_bytes_per_texel,
                         uint64_t& out) {
    out = 0;
    if (!level.width || !level.height || !level.depth || !backend_bytes_per_texel) return false;
    // SIZE_MAX, not UINT64_MAX: every caller materializes these extents, so a size no
    // allocation can hold must refuse here rather than truncate a vector size below.
    uint64_t texels = level.width;
    if (level.height && texels <= SIZE_MAX / level.height)
        texels *= level.height;
    else
        return false;
    if (level.depth && texels <= SIZE_MAX / level.depth)
        texels *= level.depth;
    else
        return false;
    if (backend_bytes_per_texel && texels <= SIZE_MAX / backend_bytes_per_texel)
        texels *= backend_bytes_per_texel;
    else
        return false;
    out = texels;
    return true;
}

} // namespace

VolumeChainStaging volume_chain_staging_layout(const MipChainPlan& plan,
                                               uint64_t backend_bytes_per_texel,
                                               uint64_t alignment) {
    VolumeChainStaging out;
    if (!mip_chain_plan_is_volume_tail(plan) || !backend_bytes_per_texel || !alignment) return out;
    out.offsets.assign(plan.level_count, 0);
    uint64_t offset = 0;
    for (uint32_t level = 0; level < plan.level_count; ++level) {
        uint64_t level_bytes = 0;
        if (!level_backend_bytes(plan.levels[level], backend_bytes_per_texel, level_bytes))
            return {};
        // Level zero opens the upload; every later level starts aligned.
        if (level) {
            if (offset > UINT64_MAX - alignment) return {};
            offset = (offset + alignment - 1u) / alignment * alignment;
            if (offset > SIZE_MAX - level_bytes) return {};
            out.offsets[level] = offset;
        }
        if (level_bytes > SIZE_MAX - offset) return {};
        offset += level_bytes;
    }
    out.total_bytes = offset;
    return out;
}

bool upload_volume_tail_chain_levels(const MipChainPlan& plan, const uint8_t* allocation_base,
                                     size_t allocation_bytes, uint32_t tile_mode,
                                     uint32_t guest_bytes_per_texel,
                                     uint64_t backend_bytes_per_texel, bool broadcast_r8,
                                     uint8_t* upload, const uint64_t* staging_offsets,
                                     size_t staging_count) {
    if (!mip_chain_plan_is_volume_tail(plan) || !allocation_base || !allocation_bytes || !upload ||
        !staging_offsets || staging_count != plan.level_count)
        return false;
    // The only conversions this file models: identity, and the single-level R8 broadcast.
    const bool native = !broadcast_r8 && guest_bytes_per_texel == backend_bytes_per_texel;
    const bool r8_broadcast =
        broadcast_r8 && guest_bytes_per_texel == 1u && backend_bytes_per_texel == 4u;
    if (!native && !r8_broadcast) return false;
    for (uint32_t level = 1; level < plan.level_count; ++level) {
        const MipChainLevel& source = plan.levels[level];
        uint64_t level_bytes = 0;
        if (!level_backend_bytes(source, backend_bytes_per_texel, level_bytes) ||
            staging_offsets[level] > SIZE_MAX - level_bytes)
            return false;
        uint64_t linear_bytes = 0;
        if (!level_backend_bytes(source, guest_bytes_per_texel, linear_bytes)) return false;
        // Detile one level through its in-block origin into a scratch extent; the scratch
        // is sized by the GUEST width so the conversion below reads exactly what detile
        // wrote, never staging-allocation residue.
        std::vector<uint8_t> linear(static_cast<size_t>(linear_bytes), 0);
        if (!detile_volume_tail_level(linear.data(), allocation_base, allocation_bytes,
                                      source.width, source.height, source.depth, tile_mode,
                                      guest_bytes_per_texel, source.tail_x, source.tail_y))
            return false;
        uint8_t* destination = upload + staging_offsets[level];
        if (native) {
            std::memcpy(destination, linear.data(), static_cast<size_t>(linear_bytes));
        } else {
            // Texel count from the guest byte count (guest width is 1 here by the
            // r8_broadcast contract above), so this cannot overflow past what detile wrote.
            const size_t texels = static_cast<size_t>(linear_bytes);
            for (size_t t = 0; t < texels; ++t) {
                const uint8_t v = linear[t];
                destination[t * 4u + 0] = v;
                destination[t * 4u + 1] = v;
                destination[t * 4u + 2] = v;
                destination[t * 4u + 3] = v;
            }
        }
    }
    return true;
}

} // namespace prosper::gpu
