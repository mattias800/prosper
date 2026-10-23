#include "shared/live/submit_renderer/image_resources.hpp"
#include "shared/live/submit_renderer/draw_resources.hpp"
#include "shared/live/submit_renderer/callback_types.hpp"
#include <stddef.h>
#include <stdint.h>
#include <algorithm>

namespace prosper::frontend::submit_renderer {
// clang-format off: preserve the byte-identical moved definitions.
size_t consume_image_renderer_mip_chain(DrawResourceContext& ctx, const RendererMipChainLayout& chain, uint32_t width, uint32_t height, VkFormat format) {
    auto& pinned_renderer_mip_targets = ctx.pinned_renderer_mip_targets;
        size_t matched = 0;
        for (uint32_t level = 0; level < chain.level_count; ++level) {
            const uint64_t id = chain.level_ids[level];
            const uint32_t level_width = std::max(width >> level, 1u);
            const uint32_t level_height = std::max(height >> level, 1u);
            for (PinnedRendererMipTarget& target : pinned_renderer_mip_targets) {
                if (target.id != id || target.width != level_width ||
                    target.height != level_height || target.format != format)
                    continue;
                if (!target.consumed) ++matched;
                target.consumed = true;
                break;
            }
        }
        return matched;
}

size_t acquire_image_texstore_slot(DrawResourceContext& ctx) {
    auto& texstore = ctx.texstore;
    auto& texstore_pinned = ctx.texstore_pinned;
    auto& texstore_used = ctx.texstore_used;
        while (texstore_used < texstore.size() && texstore_pinned[texstore_used])
            ++texstore_used;
        if (texstore_used == texstore.size()) {
            texstore.emplace_back();
            texstore_pinned.push_back(false);
        }
        const size_t slot = texstore_used;
        ++texstore_used;
        return slot;
}

void clear_image_depth_array_snapshots(DrawResourceContext& ctx) {
    auto& depth_array_snapshots = ctx.depth_array_snapshots;
    auto& depth_array_snapshot_bytes = ctx.depth_array_snapshot_bytes;
        depth_array_snapshots.clear();
        depth_array_snapshot_bytes = 0;
}
// clang-format on

} // namespace prosper::frontend::submit_renderer
