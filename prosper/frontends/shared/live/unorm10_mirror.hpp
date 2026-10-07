#pragma once
// unorm10_mirror.hpp -- keep a packed R10G10B10A2 compute result on the GPU.
//
// Graphics holds a guest 2_10_10_10 UNORM colour target as an RGBA8 image (backend_color_format folds
// A2B10G10R10 to R8G8B8A8), while compute stores the same result natively as A2B10G10R10. Before this
// path the only way from one to the other was the CPU: read the result back, unpack every texel to RGBA8
// (publish_unorm10_as_rgba8) and have the renderer upload it again. A 1:1 blit does that conversion on the
// GPU, straight into the borrowed renderer image, so the exact-result mirror can accept these results like
// any other format. A bit copy cannot: the two formats share a texel size but not a meaning.
//
// Rounding: a blit converts UNORM through float and back. The CPU path rounds to nearest with integer
// arithmetic; an implementation may differ from it by at most one 8-bit step, which is below what the
// RGBA8 target can represent faithfully in the first place.
#include <cstdint>

#include <vulkan/vulkan.h>

#include "gpu/resources/shader_resources.hpp"

namespace prosper::frontend {

// A four-component packed R10G10B10A2 UNORM storage result.
inline bool is_unorm10_rgba_storage(const prosper::gpu::ShaderResource& r) {
    return r.format == prosper::gpu::DataFormat::Unorm2_10_10_10 && r.num_components == 4;
}

// The blit region for a 1:1 copy of mip 0 / layer 0 of a width x height image.
inline VkImageBlit unorm10_mirror_blit_region(uint32_t width, uint32_t height) {
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};
    blit.dstOffsets[1] = blit.srcOffsets[1];
    return blit;
}

// Records the converting copy from the compute result `source` (A2B10G10R10, written by the dispatch) into
// `destination` (RGBA8, already in TRANSFER_DST_OPTIMAL). `source_in_transfer_src` says whether the caller
// has already moved `source` to TRANSFER_SRC_OPTIMAL; otherwise it is still GENERAL straight after the
// dispatch and this orders the shader write before the transfer read. Both formats' blit features are
// mandatory in Vulkan (A2B10G10R10_UNORM_PACK32: BLIT_SRC; R8G8B8A8_UNORM: BLIT_DST). CONFIDENCE: HIGH.
inline void record_unorm10_mirror_blit(VkCommandBuffer command, VkImage source,
                                       bool source_in_transfer_src, VkImage destination,
                                       uint32_t width, uint32_t height) {
    if (!source_in_transfer_src) {
        VkImageMemoryBarrier written{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        written.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        written.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        written.oldLayout = written.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        written.srcQueueFamilyIndex = written.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        written.image = source;
        written.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &written);
    }
    const VkImageBlit blit = unorm10_mirror_blit_region(width, height);
    vkCmdBlitImage(command, source,
                   source_in_transfer_src ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                          : VK_IMAGE_LAYOUT_GENERAL,
                   destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
}

}  // namespace prosper::frontend
