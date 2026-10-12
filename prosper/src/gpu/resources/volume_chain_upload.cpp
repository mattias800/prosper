#include "gpu/resources/volume_chain_upload.hpp"

#include "gpu/texture/tile.hpp"

namespace prosper::gpu {

bool mip_chain_plan_is_volume_tail(const MipChainPlan& plan) {
    if (!plan.valid || plan.levels.empty()) return false;
    if (plan.levels.size() != plan.level_count) return false;
    for (const MipChainLevel& level : plan.levels) {
        if (!level.in_tail || !level.depth) return false;
    }
    return true;
}

bool volume_chain_backend_width_agrees(const MipChainPlan& plan, uint32_t guest_bytes_per_texel,
                                       uint64_t backend_bytes_per_texel, bool broadcast_r8) {
    if (!mip_chain_plan_is_volume_tail(plan) || !guest_bytes_per_texel || !backend_bytes_per_texel)
        return false;
    if (broadcast_r8) return guest_bytes_per_texel == 1u && backend_bytes_per_texel == 4u;
    return guest_bytes_per_texel == backend_bytes_per_texel;
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
                                     uint8_t* upload, size_t upload_bytes,
                                     const uint64_t* staging_offsets, size_t staging_count) {
    if (!mip_chain_plan_is_volume_tail(plan) || !allocation_base || !upload || !staging_offsets ||
        staging_count != plan.level_count)
        return false;
    // The detiler zero-fills short reads and reports success, so a truncated source must
    // refuse here, before any level is read: zeros baked in as texels are silent wrong
    // content. Production passes the plan's own span, readability-probed before upload.
    if (allocation_bytes < plan.allocation_bytes) return false;
    // The only conversions this file models: identity, and the single-level R8 broadcast.
    const bool native = !broadcast_r8 && guest_bytes_per_texel == backend_bytes_per_texel;
    const bool r8_broadcast =
        broadcast_r8 && guest_bytes_per_texel == 1u && backend_bytes_per_texel == 4u;
    if (!native && !r8_broadcast) return false;
    // Pass 1: validate every level before writing any. Offsets must fit the destination
    // and run monotonically past level zero (whose bytes open the upload) without
    // overlapping; scratch is capped at one tail block, which every placeable level fits
    // (origins address inside the shared 64 KiB block by construction).
    uint64_t prior_end = 0;
    if (!level_backend_bytes(plan.levels[0], backend_bytes_per_texel, prior_end)) return false;
    for (uint32_t level = 1; level < plan.level_count; ++level) {
        const MipChainLevel& source = plan.levels[level];
        uint64_t level_bytes = 0, linear_bytes = 0;
        if (!level_backend_bytes(source, backend_bytes_per_texel, level_bytes) ||
            !level_backend_bytes(source, guest_bytes_per_texel, linear_bytes))
            return false;
        if (linear_bytes > 65536u) return false;
        if (staging_offsets[level] > upload_bytes ||
            level_bytes > upload_bytes - staging_offsets[level])
            return false;
        if (staging_offsets[level] < prior_end) return false;
        if (level_bytes > UINT64_MAX - staging_offsets[level]) return false;
        prior_end = staging_offsets[level] + level_bytes;
    }
    // Pass 2: write. Native levels detile straight into their staging region (mirroring
    // the 2D uploader); broadcast levels detile into a scratch extent sized by the GUEST
    // width, so the conversion reads exactly what detile wrote. A detile failure here
    // returns false: unreachable while the plan and the detiler agree on fit (same
    // derivation both directions; the overrun arm pins detile's side), and fail-visible
    // regardless.
    for (uint32_t level = 1; level < plan.level_count; ++level) {
        const MipChainLevel& source = plan.levels[level];
        uint8_t* destination = upload + staging_offsets[level];
        if (native) {
            if (!detile_volume_tail_level(destination, allocation_base, allocation_bytes,
                                          source.width, source.height, source.depth, tile_mode,
                                          guest_bytes_per_texel, source.tail_x, source.tail_y))
                return false;
        } else {
            uint64_t linear_bytes = 0;
            if (!level_backend_bytes(source, guest_bytes_per_texel, linear_bytes)) return false;
            std::vector<uint8_t> linear(static_cast<size_t>(linear_bytes), 0);
            if (!detile_volume_tail_level(linear.data(), allocation_base, allocation_bytes,
                                          source.width, source.height, source.depth, tile_mode,
                                          guest_bytes_per_texel, source.tail_x, source.tail_y))
                return false;
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
