#pragma once
// unorm10_mirror.hpp -- which compute results take the packed R10G10B10A2 exact-result mirror.
//
// Graphics holds a guest 2_10_10_10 UNORM colour target as an RGBA8 image (backend_color_format folds
// A2B10G10R10 to R8G8B8A8), while compute stores the same result natively as A2B10G10R10. Before the
// mirror the only way from one to the other was the CPU: read the result back, unpack every texel to RGBA8
// (publish_unorm10_as_rgba8) and have the renderer upload it again. The mirror converts on the GPU instead
// (PackedRttConversion::record_packed10_to_rgba8) into a second half of the staging buffer, then bit-copies
// that into the borrowed renderer image.
//
// The conversion is integer arithmetic with the CPU path's rounding, not a blit: a blit converts UNORM
// through float, and the driver may round a tie either way. Measured on RADV, 48 of the 1,024 10-bit values
// convert one step low that way, so a blit would change guest-visible pixels by driver (#4709 review).
#include <mutex>

#include <vulkan/vulkan.h>

#include "gpu/resources/shader_resources.hpp"

namespace prosper::frontend {

// A four-component packed R10G10B10A2 UNORM storage result.
inline bool is_unorm10_rgba_storage(const prosper::gpu::ShaderResource& r) {
    return r.format == prosper::gpu::DataFormat::Unorm2_10_10_10 && r.num_components == 4;
}

// Whether a writable storage binding should reserve the RGBA8 second half when its staging buffer is
// created: a native packed-10 2D result whose staging is exactly one 32-bit word per texel.
template <class Bound>
bool unorm10_mirror_wants_scratch(const Bound& bi, const prosper::gpu::ShaderResource& r,
                                  VkDeviceSize staging_bytes) {
    return bi.storage_writeback && bi.native_float_storage && is_unorm10_rgba_storage(r) &&
           r.depth == 1 && staging_bytes == VkDeviceSize{r.width} * r.height * 4u;
}

// The result can take the mirror only with its scratch half reserved and the integer conversion pass
// ready. Prepares the pass and binds its descriptor set to the whole doubled staging buffer. `soft_ok`
// is the caller's non-fatal Vulkan result check.
template <class Context, class Bound, class SoftOk>
bool unorm10_mirror_ready(Context& ctx, Bound& bi, VkBuffer staging, VkDeviceSize result_bytes,
                          SoftOk&& soft_ok) {
    if (!bi.unorm10_mirror_scratch || !staging) return false;
    std::lock_guard cache_lock(ctx.pipeline_cache_mutex);
    auto& conversion = ctx.unorm10_mirror_conversion;
    if (!soft_ok(conversion.initialize(ctx.physical, ctx.device, ctx.pipeline_cache),
                 "unorm10-mirror-pipeline") ||
        !conversion.fits(result_bytes / 4u, 2u) ||
        (!bi.unorm10_set && !soft_ok(conversion.allocate_binding(bi.unorm10_pool, bi.unorm10_set),
                                     "unorm10-mirror-descriptors")))
        return false;
    conversion.bind_buffer(bi.unorm10_set, staging, result_bytes * 2u);
    return true;
}

}   // namespace prosper::frontend
